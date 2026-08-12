//! Worker pool — spawns `n` OS threads, each running a tight check loop.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::framework::GpuFramework;
use crate::puzzles::PuzzleSet;
use crate::progress::Progress;

/// Runtime limits for the worker pool.
pub struct RuntimeLimits {
    pub duration_secs: Option<f64>,
    pub heartbeat_secs: f64,
}

/// What the lottery workers scan against.
pub enum ScanTarget {
    /// Embedded 77 puzzles, range-constrained key generation in [2^70, 2^160).
    PuzzleSet(&'static PuzzleSet),
}

/// Match surfaced by the pool.
#[derive(Debug, Clone)]
pub struct MatchEvent {
    pub private_key: [u8; 32],
    pub compressed: Vec<u8>,
    pub uncompressed: Vec<u8>,
    pub worker_id: u32,
    pub chunk_id: Option<u32>,    // puzzle-mode: which worklist chunk this came from (None = lottery)
    pub key_index: u64,
    pub elapsed: f64,
    /// Which embedded puzzle matched, when the match is tied to a specific
    /// puzzle number.  None for GPU-mode matches before CPU-side verification.
    pub puzzle_number: Option<u32>,
}

pub fn run(
    n_workers: usize,
    target: ScanTarget,
    limits: RuntimeLimits,
    gpu: bool,
    framework: GpuFramework,
) -> (Arc<Progress>, Vec<MatchEvent>) {
    // GPU worker counts as an extra "worker" for the heartbeat's alive counter
    // when enabled.
    let n_total = n_workers + gpu as usize;
    let progress = Arc::new(Progress::new(n_total as u64));
    let matches = Arc::new(std::sync::Mutex::new(Vec::<MatchEvent>::new()));
    let deadline = limits
        .duration_secs
        .map(|s| Instant::now() + Duration::from_secs_f64(s));
    let start = Instant::now();

    // 命中即停：`hit_flag` 标记首个命中者（赢家负责打印 [HIT]），`stop_flag`
    // 让所有 worker 与 heartbeat 立即停止。没有命中时两者保持 false，行为不变。
    let stop_flag = Arc::new(AtomicBool::new(false));
    let hit_flag = Arc::new(AtomicBool::new(false));

    let handles: Vec<_> = (0..n_workers)
        .map(|wid| {
            let progress = progress.clone();
            let matches = matches.clone();
            let stop_flag = stop_flag.clone();
            let hit_flag = hit_flag.clone();

            // Clone the scan target for each worker.
            let ScanTarget::PuzzleSet(ps) = &target;
            let worker_target = WorkerMode::Puzzle(*ps);

            thread::spawn(move || {
                worker_loop(
                    wid as u32,
                    &worker_target,
                    &progress,
                    &matches,
                    &stop_flag,
                    &hit_flag,
                    deadline,
                    start,
                );
            })
        })
        .collect();

    // ── GPU lottery worker (optional) ────────────────────────────────────────
    // One additional worker that scans with the GPU (100k independent random
    // walkers, re-seeded every RESEED_INTERVAL_KEYS).  Shares the same progress
    // and matches as the CPU workers.  Falls back to CPU-only if no GPU device.
    let gpu_handle = if gpu {
        let ScanTarget::PuzzleSet(ps) = &target;
        let gpu_progress = progress.clone();
        let gpu_matches = matches.clone();
        let gpu_stop = stop_flag.clone();
        let gpu_hit = hit_flag.clone();
        let gpu_deadline = deadline;
        let gpu_start = start;
        let ps = *ps; // &'static — copy the reference into the closure.
        match framework {
            GpuFramework::WebGpu => {
                Some(thread::spawn(move || {
                    crate::gpu::lottery::worker(
                        ps,
                        gpu_progress,
                        gpu_matches,
                        gpu_stop,
                        gpu_hit,
                        gpu_deadline,
                        gpu_start,
                    );
                }))
            }
            GpuFramework::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    Some(thread::spawn(move || {
                        crate::cuda::lottery::worker(
                            ps,
                            gpu_progress,
                            gpu_matches,
                            gpu_stop,
                            gpu_hit,
                            gpu_deadline,
                            gpu_start,
                        );
                    }))
                }
                #[cfg(not(feature = "cuda"))]
                {
                    eprintln!("  [GPU] CUDA feature not compiled -- falling back to CPU-only.");
                    None
                }
            }
            GpuFramework::Auto => {
                // Resolved to a concrete backend in main() before we get here.
                unreachable!("framework Auto must be resolved before workers::run")
            }
        }
    } else {
        None
    };

    // ── heartbeat ticker (just print line, no channel) ───────────────
    let hb_progress = progress.clone();
    let hb_deadline = deadline;
    let hb_stop = stop_flag.clone();
    let hb_interval = limits.heartbeat_secs;
    let hb_handle = thread::spawn(move || {
        let mut prev_total = 0u64;
        let mut prev_instant = Instant::now();
        loop {
            thread::sleep(Duration::from_secs_f64(hb_interval));
            if hb_deadline.is_some_and(|dl| Instant::now() >= dl) {
                break;
            }
            // 命中即停：命中的 worker 置 stop_flag 后 ticker 也尽快退出。
            if hb_stop.load(Ordering::Relaxed) {
                break;
            }
            let total = hb_progress.checked.load(std::sync::atomic::Ordering::Relaxed);
            let alive = hb_progress.workers_alive.load(std::sync::atomic::Ordering::Relaxed);
            let now = Instant::now();
            let dt = now.duration_since(prev_instant).as_secs_f64();
            let rate = if dt > 0.1 { (total - prev_total) as f64 / dt } else { 0.0 };
            prev_total = total;
            prev_instant = now;
            println!(
                "  [HEARTBEAT] Keys: {:>14} | Speed: {:>10} H/s | Workers: {}",
                fmt_comma(total),
                fmt_comma(rate as u64),
                alive,
            );
        }
    });

    for h in handles {
        let _ = h.join();
    }
    // Join the GPU worker (if spawned).  It observes the same deadline as the
    // CPU workers, so it should already be finishing; this just ensures we
    // wait for its final match flush before reporting.
    if let Some(h) = gpu_handle {
        let _ = h.join();
    }
    // Signal the heartbeat ticker to stop and wait for it.  Without the join,
    // the ticker could print a stray heartbeat after workers have finished.
    drop(hb_handle);

    let matches = match Arc::try_unwrap(matches) {
        Ok(m) => m.into_inner().unwrap_or_else(|e| e.into_inner()),
        Err(a) => a.lock().unwrap_or_else(|e| e.into_inner()).clone(),
    };
    (progress, matches)
}

/// Per-worker scan mode, resolved once before the thread spawns.
enum WorkerMode {
    Puzzle(&'static PuzzleSet),
}

fn worker_loop(
    wid: u32,
    mode: &WorkerMode,
    progress: &Progress,
    matches: &std::sync::Mutex<Vec<MatchEvent>>,
    stop_flag: &AtomicBool,
    hit_flag: &AtomicBool,
    deadline: Option<Instant>,
    start: Instant,
) {
    let secp = secp256k1::Secp256k1::new();

    // Scalar constant "1" for +1 tweak.
    let tweak = secp256k1::Scalar::from_be_bytes({
        let mut one = [0u8; 32];
        one[31] = 1;
        one
    })
    .expect("Scalar 1 is always valid");

    let point_g = crate::btc::generator_public_key();

    // Initialize based on mode.
    let WorkerMode::Puzzle(ps) = mode;
    let mut sk = new_key_puzzle(ps);
    let mut pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);

    let mut local_count = 0u64;

    loop {
        // Performance guard: check deadline every ~2048 iterations.
        if local_count.is_multiple_of(2048) {
            if deadline.is_some_and(|dl| Instant::now() >= dl) {
                break;
            }

            // 命中即停：别的 worker 命中了，尽快退出。
            if stop_flag.load(Ordering::Relaxed) {
                break;
            }

            // End-of-range check: if the key has left its current range (reached
            // the end, walked into a gap, or exceeded 2^160), re-seed with a fresh
            // puzzle + key.
            let sk_bytes = sk.secret_bytes();
            let still_in_range = find_active_range(ps, &sk_bytes)
                .map(|active| crate::puzzles::be_lt(&sk_bytes, &active.end))
                .unwrap_or(false);
            if !still_in_range {
                sk = new_key_puzzle(ps);
                pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);
                local_count = 0;
            }
        }

        let pk_c = pk.serialize();
        let pk_u = pk.serialize_uncompressed();

        let h_c = crate::btc::hash160(&pk_c);
        let h_u = crate::btc::hash160(&pk_u);

        let hit = ps.contains(&h_c) || ps.contains(&h_u);

        if hit {
            let puzzle_number = ps.puzzle_number_for_hash160(&h_c)
                .or_else(|| ps.puzzle_number_for_hash160(&h_u));
            let ev = MatchEvent {
                private_key: sk.secret_bytes(),
                compressed: pk_c.to_vec(),
                uncompressed: pk_u.to_vec(),
                worker_id: wid,
                chunk_id: None,
                key_index: local_count,
                elapsed: start.elapsed().as_secs_f64(),
                puzzle_number,
            };
            // 锁中毒也照样保存命中（命中是最珍贵的事件，绝不丢弃）。
            let mut g = matches.lock().unwrap_or_else(|e| e.into_inner());
            g.push(ev.clone());
            // 命中即停：首个命中的 worker 立即打印（含私钥）并通知所有 worker 停止。
            if hit_flag
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                println!(
                    "[HIT] 🎯 puzzle=#{} worker=#{} idx={} sk_hex={}",
                    puzzle_number.map_or_else(String::new, |n| n.to_string()),
                    wid,
                    fmt_comma(local_count),
                    hex::encode(ev.private_key),
                );
            }
            stop_flag.store(true, Ordering::SeqCst);
            break;
        }

        // Advance: scalar +1 (private key, for reporting), point add +G (public key).
        // add_tweak only fails if sk == n-1 (curve order); our keys are always < 2^160 << n.
        sk = sk
            .add_tweak(&tweak)
            .expect("sk < n-1 guaranteed by range-constrained generation");
        pk = pk
            .combine(&point_g)
            .expect("pk + G cannot be infinity for sk < n-1");
        local_count += 1;

        if local_count.is_multiple_of(1_000) {
            progress.increment(1_000);
        }
    }
}

/// Find which puzzle range (if any) a key belongs to by checking the highest set bit.
/// Returns the PuzzleRange if found.
fn find_active_range<'a>(ps: &'a PuzzleSet, key: &[u8; 32]) -> Option<&'a crate::puzzles::PuzzleRange> {
    // Find highest set bit in the 32-byte key.
    let mut bit_pos = None;
    for (i, &b) in key.iter().enumerate() {
        if b != 0 {
            let bit_in_byte = 7 - b.leading_zeros() as usize;
            bit_pos = Some((31 - i) * 8 + bit_in_byte);
            break;
        }
    }
    let bit_pos = bit_pos?;
    // Look up the puzzle at this bit position.
    ps.range_for_bit(bit_pos)
}

/// Generate a random key within a randomly-chosen puzzle range.
fn new_key_puzzle(ps: &PuzzleSet) -> secp256k1::SecretKey {
    let idx = ps.pick_random_puzzle();
    let range = &ps.ranges()[idx];
    let mut buf = [0u8; 32];
    ps.generate_key_in_range(range, &mut buf);
    secp256k1::SecretKey::from_byte_array(buf)
        .expect("key_in_range always produces a valid key in [2^70, 2^160)")
}

/// Format an integer with comma separators.
pub fn fmt_comma(n: u64) -> String {
    n.to_string()
        .as_bytes()
        .rchunks(3)
        .rev()
        .map(std::str::from_utf8)
        .collect::<Result<Vec<&str>, _>>()
        .unwrap_or_default()
        .join(",")
}
