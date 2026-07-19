//! Puzzle-mode scanner.
//!
//! Reads a btcpuzzle-style worklist JSON file that subdivides a puzzle
//! key-range into `chunks`.  Each chunk defines:
//!
//! - `start_hex`:  leftmost private key in the sub-range.  May be < 64 hex
//!   chars; it is always zero-padded on the *left* (high bytes) to a full
//!   32-byte secp256k1 key.
//! - `range_bits`: width of the sub-range, so the chunk spans `2^range_bits`
//!   keys `[start, start + 2^range_bits)`.
//! - `status`:     `"pending"`, `"running"`, or `"finished"`.
//! - `current_hex` (optional): scan progress written back so that on resume
//!   the next run starts here.
//!
//! Per worker: pick a random *pending* chunk, claim it (status = "running"),
//! then walk every key `current ..= end` by scalar +1 (the same tight `+= 1`
//! loop used in lottery mode for speed).  When the whole sub-range is done,
//! the chunk is marked `finished`.  On SIGINT (Ctrl+C) every worker flushes
//! its current scanning position into the worklist JSON, reverts the chunk
//! to `"pending"`, and exits — so a later invocation resumes cleanly.
//!
//! Current bytes saved per chunk:
//!
//! ```json
//! {
//!   "chunk_index": 42,
//!   "start_hex": "8200000000000000000",
//!   "range_bits": 65,
//!   "status": "pending",
//!   "current_hex": "821a3f ... e2"   // last key processed on abort; cleared on finish
//! }
//! ```

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rand::TryRng;
use serde::{Deserialize, Serialize};

use crate::addrs;
use crate::btc;
use crate::progress::Progress;
use crate::workers::{fmt_comma, MatchEvent};

/// Wall-clock interval (seconds) between worklist auto-saves on the ticker.
/// The ticker holds the PuzzleCtx lock briefly, serialising disk writes
/// away from the worker hot path.  Workers update in-memory `current_hex`
/// every 2048 iterations (see the scan loop); the ticker persists whatever
/// it sees every SAVE_INTERVAL_SECS.  10 minutes is cheap and bounds the
/// worst-case resume loss to ~5 ms of scanning (2048 keys at 500 kkeys/s).
const SAVE_INTERVAL_SECS: u64 = 600;

// ── worklist JSON shape ──────────────────────────────────────────────────────

/// On-disk puzzle worklist.  Mirrors the btcpuzzle range-split format plus the
/// optional `current_hex` field used for checkpointing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PuzzleFile {
    pub puzzle_number: u32,
    pub total_bits: u32,
    #[serde(default)]
    pub chunk_bits_used: u32,
    pub total_chunks: u32,
    pub completed_chunks: u32,
    pub target: String,
    /// Optional expected RIPEMD-160 hash (40 hex chars) of `target`.  When
    /// present it is checked against the hash decoded from `target` — a mismatch
    /// means the JSON is inconsistent and we abort.  Absent ⇒ skipped.
    pub hash160: Option<String>,
    pub chunks: Vec<Chunk>,
}

/// A single sub-range in the puzzle worklist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub chunk_index: u32,
    pub start_hex: String,
    pub range_bits: u32,
    pub status: String,

    /// Snapshot of the most recent key processed; `None` until first scan
    /// starts.  When present, the next chunk iteration resumes from here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current_hex: Option<String>,
}

// ── runtime state ────────────────────────────────────────────────────────────

/// Shared state across workers: the parsed worklist and the path to persist to.
#[derive(Debug)]
struct PuzzleCtx {
    file: PuzzleFile,
    path: PathBuf,
}

// ── public entry point ──────────────────────────────────────────────────────

/// Run the puzzle loop.  Returns total keys checked and any match events.
///
/// `rotate_keys` enables *random-rotation* mode: each claim scans at most this
/// many keys before the chunk is parked (status ← "pending", `current_hex`
/// saved) and the worker moves on to a fresh random pending chunk.  `None`
/// preserves classic behaviour — a chunk is scanned to completion per claim.
/// When set, `n_workers` slots each churn through random chunks, giving a
/// "jump around the worklist" effect instead of sweeping it in order.
pub fn run(
    path: &Path,
    n_workers: usize,
    heartbeat_secs: f64,
    rotate_keys: Option<u64>,
) -> (Arc<Progress>, Vec<MatchEvent>) {
    // ── 1. load the worklist ────────────────────────────────────────────────
    let mut file: PuzzleFile = serde_json::from_reader(std::io::BufReader::new(
        std::fs::File::open(path).unwrap_or_else(|e| {
            eprintln!("[puzzle] cannot open {}: {}", path.display(), e);
            std::process::exit(2);
        }),
    ))
    .unwrap_or_else(|e| {
        eprintln!("[puzzle] invalid JSON in {}: {}", path.display(), e);
        std::process::exit(2);
    });

    // Convert the target BTC address to its 20-byte hash160 for fast compare.
    // This decode happens once at startup; both CPU workers and the GPU worker
    // reuse the resulting 20-byte value — no Base58 work on the hot path.
    let target_h160 = addrs::p2pkh_addr_to_hash160(&file.target).unwrap_or_else(|| {
        eprintln!("[puzzle] target {} is not a valid P2PKH address", file.target);
        std::process::exit(2);
    });

    // If the JSON ships an expected hash160, sanity-check it against the value
    // we just decoded from `target`.  A mismatch means the worklist is
    // internally inconsistent — we abort rather than scan the wrong set.
    // Absent ⇒ no check (backward-compatible with files that omit the field).
    if let Some(ref h160_hex) = file.hash160 {
        match hash160_from_hex(h160_hex) {
            Some(expected) if expected == target_h160 => {
                eprintln!("[puzzle] hash160 OK ({h160_hex})");
            }
            Some(_) => {
                eprintln!(
                    "[puzzle] hash160 MISMATCH: JSON says {h160_hex}, \
                     target {} decodes to {}",
                    file.target,
                    hex::encode(target_h160)
                );
                std::process::exit(2);
            }
            None => {
                eprintln!("[puzzle] hash160 in JSON is not valid hex: {h160_hex}");
                std::process::exit(2);
            }
        }
    }

    // Crash-recovery: any chunk left "running" from a previous killed run was
    // in-flight — revert it to pending so it can be reclaimed.
    for c in file.chunks.iter_mut() {
        if c.status == "running" {
            c.status = "pending".to_string();
        }
    }

    let summary = chunk_summary(&file);
    println!(
        "[puzzle #{}] target={}  chunks={}  (pending={}, running={}, finished={})",
        file.puzzle_number, file.target, file.total_chunks, summary.0, summary.1, summary.2,
    );

    if summary.0 + summary.1 == 0 {
        println!("[puzzle] all chunks already finished — nothing to do.");
        return (Arc::new(Progress::new(0)), Vec::new());
    }

    // ── 2. shared state + SIGINT flag ───────────────────────────────────────
    let ctx = Arc::new(Mutex::new(PuzzleCtx {
        file,
        path: path.to_path_buf(),
    }));
    let progress = Arc::new(Progress::new(n_workers as u64));
    let matches = Arc::new(Mutex::new(Vec::<MatchEvent>::new()));
    let stop_flag = Arc::new(AtomicBool::new(false));

    let stop_handler = stop_flag.clone();
    ctrlc::set_handler(move || {
        // First Ctrl+C → start graceful shutdown.
        if stop_handler
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            eprintln!("\n[puzzle] Ctrl+C — saving progress and stopping workers …");
        } else {
            // Second Ctrl+C hard-aborts (default behaviour).
            eprintln!("\n[puzzle] second Ctrl+C — aborting immediately");
            std::process::exit(130);
        }
    })
    .expect("[puzzle] failed to install SIGINT handler");

    // ── 3. launch worker threads ─────────────────────────────────────────────
    let start = Instant::now();
    let mut handles = Vec::with_capacity(n_workers);
    for wid in 0..n_workers {
        handles.push(std::thread::spawn({
            let ctx = ctx.clone();
            let progress = progress.clone();
            let matches = matches.clone();
            let stop_flag = stop_flag.clone();
            move || {
                puzzle_worker(
                    wid as u32,
                    target_h160,
                    ctx,
                    &progress,
                    &matches,
                    &stop_flag,
                    start,
                    rotate_keys,
                )
            }
        }));
    }

    // ── 3b. GPU worker thread ────────────────────────────────────────────────
    // One additional worker that claims whatever pending chunks the CPU workers
    // haven't taken and scans them with the GPU (100k strided walkers, dense
    // zero-overlap tiling, per-dispatch checkpoint).  Falls back to CPU-only if
    // no Metal device is present.  Rotation is intentionally NOT passed through:
    // with GPU saturating throughput there's no need to park-and-rotate chunks.
    let gpu_ctx = ctx.clone();
    let gpu_progress = progress.clone();
    let gpu_matches = matches.clone();
    let gpu_stop = stop_flag.clone();
    let gpu_handle = std::thread::spawn(move || {
        gpu_puzzle_worker(
            target_h160,
            gpu_ctx,
            &gpu_progress,
            &gpu_matches,
            &gpu_stop,
            start,
        )
    });

    // ── 4. heartbeat ticker: periodic save + status line ─────────────────────
    let hb_ctx = ctx.clone();
    let hb_stop = stop_flag.clone();
    let hb_progress = progress.clone();
    let hb_handle = std::thread::spawn(move || {
        let mut prev_total = 0u64;
        let mut prev_time = Instant::now();
        let mut last_save = Instant::now();
        loop {
            std::thread::sleep(Duration::from_secs_f64(heartbeat_secs));
            if hb_stop.load(Ordering::Relaxed) {
                break;
            }

            let now = Instant::now();
            let total = hb_progress.checked.load(Ordering::Relaxed);
            let dt = now.duration_since(prev_time).as_secs_f64();
            let rate = if dt > 0.1 {
                (total - prev_total) as f64 / dt
            } else {
                0.0
            };
            prev_total = total;
            prev_time = now;

            let s = {
                let ctx = hb_ctx.lock().unwrap_or_else(|e| e.into_inner());

                let summary = chunk_summary(&ctx.file);
                let done_pct = if ctx.file.total_chunks > 0 {
                    summary.2 as f64 / ctx.file.total_chunks as f64 * 100.0
                } else {
                    100.0
                };
                // Chunk indices currently held by a worker (status="running").
                // Cheap to collect here: ticker already holds ctx for the
                // save-throttle check, and the list is tiny (≤ n_workers).
                let running_idxs: Vec<u32> = ctx
                    .file
                    .chunks
                    .iter()
                    .filter(|c| c.status == "running")
                    .map(|c| c.chunk_index)
                    .collect();
                let idxs_label = if running_idxs.len() <= 8 {
                    format!("{:?}", running_idxs)
                } else {
                    format!(
                        "{} chunks: {:?} … {:?}",
                        running_idxs.len(),
                        &running_idxs[..4],
                        &running_idxs[running_idxs.len() - 4..]
                    )
                };
                format!(
                    "[puzzle] pend={} running={} done={} ({:.1}%) \
                     idxs={}  keys={}  rate={}/s",
                    summary.0,
                    summary.1,
                    summary.2,
                    done_pct,
                    idxs_label,
                    fmt_comma(total),
                    fmt_comma(rate as u64),
                )
            };
            eprintln!("  {s}");

            // Throttled disk save on the heartbeat cadence.
            if now.duration_since(last_save) >= Duration::from_secs(SAVE_INTERVAL_SECS) {
                let ctx = hb_ctx.lock().unwrap_or_else(|e| e.into_inner());
                if ctx.save().is_ok() {
                    last_save = now;
                }
            }
        }
    });

    // ── 5. join workers, then do one final save ─────────────────────────────
    for h in handles {
        drop(h.join());
    }
    drop(gpu_handle.join()); // GPU worker (no-op if it fell back to CPU-only)
    drop(hb_handle); // ticker sees stop flag on its next tick and exits

    let (final_file, final_matches) = {
        let ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = ctx.save() {
            eprintln!("[puzzle] final save failed: {e}");
        }
        let fm = match Arc::try_unwrap(matches) {
            Ok(m) => m.into_inner().unwrap_or_else(|e| e.into_inner()),
            Err(a) => a.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        };
        (ctx.file.clone(), fm)
    };

    // ── 6. final report ─────────────────────────────────────────────────────
    eprintln!();
    eprintln!("──────────────────────────────────────────────────");
    eprintln!("  PUZZLE SCAN COMPLETE  (#{})", final_file.puzzle_number);
    eprintln!("──────────────────────────────────────────────────");
    let summary = chunk_summary(&final_file);
    eprintln!("  Target     : {}", final_file.target);
    eprintln!(
        "  Chunks     : {} total — pending {}, running {}, finished {}",
        final_file.total_chunks, summary.0, summary.1, summary.2,
    );
    eprintln!(
        "  Keys       : {}",
        fmt_comma(progress.checked.load(Ordering::Relaxed)),
    );
    eprintln!(
        "  Duration   : {:.2}s ({:.2}m)",
        start.elapsed().as_secs_f64(),
        start.elapsed().as_secs_f64() / 60.0,
    );
    if final_matches.is_empty() {
        eprintln!("  Match      : none");
    } else {
        eprintln!(
            "  Match      : {} event(s) — see got.txt",
            final_matches.len()
        );
    }
    eprintln!();

    (progress, final_matches)
}

// ── one worker loop ─────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn puzzle_worker(
    wid: u32,
    target_h160: [u8; 20],
    ctx: Arc<Mutex<PuzzleCtx>>,
    progress: &Progress,
    matches: &Mutex<Vec<MatchEvent>>,
    stop_flag: &AtomicBool,
    start: Instant,
    rotate_keys: Option<u64>,
) {
    let secp = secp256k1::Secp256k1::new();
    // Scalar(1) so `sk += 1` per iteration.  Also kept to preserve the SK ↔ PK
    // pairing: sk.add_tweak advances the private key, pk.combine(&G) advances
    // the public key, and they stay in sync because (sk+1)*G == sk*G + G.
    let one = secp256k1::Scalar::from_be_bytes({
        let mut b = [0u8; 32];
        b[31] = 1;
        b
    })
    .expect("scalar 1 is always valid");

    // Generator point G as a compressed PublicKey.  Parsed once per worker —
    // cheap (single affine decode) vs. pay-per-key if re-parsed in the loop.
    let point_g = crate::btc::generator_public_key();

    loop {
        // ── pick + claim a pending chunk ────────────────────────────────────
        let claim = {
            let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
            // Only *pending* chunks are claimable.  "running" chunks belong to
            // another worker — trying to re-claim them would double-scan.
            // Crash recovery (a chunk left "running" by a killed run) is
            // handled at startup, where we reset those back to "pending".
            let pendings: Vec<usize> = ctx
                .file
                .chunks
                .iter()
                .enumerate()
                .filter(|(_, c)| c.status == "pending")
                .map(|(i, _)| i)
                .collect();
            if pendings.is_empty() {
                return; // all done
            }
            pick_random(&pendings).map(|&idx| {
                // Narrow inner scopes so the mutable borrow of ctx ends
                // before ctx.save() below.
                // `end_bytes` is the chunk's *true* exclusive bound, derived
                // from its nominal start (`start_hex`).  It must NOT be
                // derived from the resume position — resuming mid-chunk must
                // not extend the scan past the chunk's real boundary (which
                // would double-scan into the next chunk, or past the puzzle
                // range).  Carried out of this scope so the scan loop can use
                // it directly.
                let (chunk_id, start_hex, end_bytes) = {
                    let chunk = &mut ctx.file.chunks[idx];
                    let base = parse_hex_key(&chunk.start_hex);
                    let end  = chunk_end(&base, chunk.range_bits);
                    // Resume from prior current_hex IFF it lies strictly
                    // within [base..end).  Otherwise fall back to base —
                    // this self-heals a corrupt / out-of-range checkpoint.
                    let resumed = chunk.current_hex.as_deref().and_then(|h| {
                        let ck = parse_hex_key(h);
                        (ck >= base && ck < end).then_some(h.to_owned())
                    });
                    chunk.status = "running".to_string();
                    // Drop stale checkpoint — worker re-establishes its
                    // own as it scans.
                    chunk.current_hex = None;
                    (
                        chunk.chunk_index,
                        resumed.unwrap_or(chunk.start_hex.clone()),
                        end,
                    )
                };
                // Persist `status = running` to disk immediately so an
                // observer reading the JSON right after this claim sees it.
                // Without this the status only surfaces on the next
                // heartbeat auto-save (up to SAVE_INTERVAL_SECS away).
                if let Err(e) = ctx.save() {
                    eprintln!("[puzzle] post-claim flush failed: {e}");
                }
                let start_bytes = parse_hex_key(&start_hex);
                (idx, chunk_id, start_bytes, end_bytes)
            })
        };

        let (idx, chunk_id, start_bytes, end_bytes) = match claim {
            Some(c) => c,
            None => return,
        };

        let mut sk = match secp256k1::SecretKey::from_byte_array(start_bytes) {
            Ok(k) => k,
            Err(_) => {
                // The nominal `start_hex` is outside [1, n-1]; skip this chunk.
                let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
                ctx.file.chunks[idx].status = "pending".to_string();
                continue;
            }
        };
        let mut local_count = 0u64;

        // 初始一次完整标量乘 `pk = sk * G`。之后循环不再做标量乘，只用点加
        // `pk = pk + G` 推进 —— 比每步 from_secret_key 便宜 10-20×。
        let mut pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);

        let done = 'scan: loop {
            // ── hot path (exactly mirrors lottery worker_loop) ─────────────
            // Pubkey derive + dhash160 compare.  No branches, no IO, no lock
            // on the per-key critical path — this is what keeps the rate at
            // ~500 kkeys/s.  We only fall through to the boundary check on
            // every 2048th iteration.
            //
            // Note: `pk` is already derived — no per-key scalar mult here.
            let pk_c = pk.serialize();
            let pk_u = pk.serialize_uncompressed();

            if h160_eq(&pk_c, target_h160) || h160_eq(&pk_u, target_h160) {
                let ev = MatchEvent {
                    private_key: sk.secret_bytes(),
                    compressed: pk_c.to_vec(),
                    uncompressed: pk_u.to_vec(),
                    worker_id: wid,
                    chunk_id: Some(chunk_id),
                    key_index: local_count,
                    elapsed: start.elapsed().as_secs_f64(),
                };
                if let Ok(mut g) = matches.lock() {
                    g.push(ev);
                }
            }

            // ── mutate state ───────────────────────────────────────────────
            // 配对推进：sk +1（标量，mod n，用于断点续传和报告），
            // pk + G（点加，底层 gej_add_ge，零次 doubling）。
            // 两者保持 pk == sk * G 不变式。
            sk = match sk.add_tweak(&one) {
                Ok(next) => next,
                Err(_) => break 'scan false, // scalar overflow → stop
            };
            pk = match pk.combine(&point_g) {
                Ok(next) => next,
                // pk == -G (无穷远点) ⟺ sk == n-1 ⟺ 等价于 scalar overflow.
                Err(_) => break 'scan false,
            };
            local_count += 1;
            if local_count.is_multiple_of(1000) {
                progress.increment(1000);
            }

            // ── boundary check (amortised ~2000× like lottery) ─────────────
            // Every 2048 iterations we do three cheap things:
            //   1. SIGINT flag — break out if the user pressed Ctrl+C.
            //   2. End-of-range — break out if we've scanned the whole chunk.
            //   3. Refresh in-memory current_hex — this is the snapshot the
            //      ticker persists to disk every SAVE_INTERVAL_SECS.  Writing
            //      it every 2048 iterations (~5 ms at 500 kkeys/s) means the
            //      ticker almost always has a fresh resume point, bounding
            //      the worst-case loss to a few microseconds of scanning.
            // The release backend compiles the `is_multiple_of` modulo to a
            // single `test` instruction; the predictor lock-stamps the
            // not-taken path so the hot loop stays tight.
            if local_count.is_multiple_of(2048) {
                if stop_flag.load(Ordering::Relaxed) {
                    break 'scan false;
                }
                if sk.secret_bytes() >= end_bytes {
                    break 'scan true;
                }
                // ── rotation (random-subrange mode) ───────────────────────────
                // Park the chunk after `rotate_keys` scanned *this claim* and
                // let the worker move on to a fresh random pending chunk.
                // `local_count` resets every claim, so this is "per-claim
                // budget", not cumulative across re-claims of the same chunk.
                // A chunk narrower than the budget still finishes via the
                // end-of-range check above, so rotation only ever kicks in for
                // chunks wider than the budget.
                if let Some(rot) = rotate_keys {
                    if local_count >= rot {
                        break 'scan false; // finalize saves current_hex + pending
                    }
                }
                let cur = sk.secret_bytes();
                let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
                ctx.file.chunks[idx].current_hex = Some(hex_encode_key(&cur));
                // Disk persist happens on the ticker every SAVE_INTERVAL_SECS;
                // here we only refresh the in-memory snapshot.
            }
        };

        // ── finalize the chunk ──────────────────────────────────────────────
        // On disk right away — the chunk's new status + current_hex must be
        // visible to a future run, not just to the next ticker auto-save.
        {
            let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
            let chunk = &mut ctx.file.chunks[idx];
            if done {
                chunk.status = "finished".to_string();
                chunk.current_hex = None;
                ctx.file.completed_chunks += 1;
            } else {
                // either SIGINT or scalar overflow — preserve progress & re-enable
                chunk.status = "pending".to_string();
                chunk.current_hex = Some(hex_encode_key(&sk.secret_bytes()));
            }
            sync_flush_chunk(&mut ctx);
        }

        if stop_flag.load(Ordering::Relaxed) {
            return;
        }
    }
}

// ── GPU worker ──────────────────────────────────────────────────────────────

/// A deterministic sub-range claimed by the GPU worker.  Mirrors the per-chunk
/// view the CPU worker scans (start key, exclusive end, resume position).
struct GpuChunk {
    idx: usize,
    chunk_id: u32,
    start: [u8; 32], // inclusive first key to scan (resume position if any)
    end: [u8; 32],   // exclusive upper bound
}

/// Pick + claim a pending chunk for the GPU worker (same policy as CPU workers:
/// random pending chunk, "running" marker, persist immediately).  Returns `None`
/// when no pending chunks remain.
fn gpu_claim_chunk(ctx: &mut PuzzleCtx) -> Option<GpuChunk> {
    let pendings: Vec<usize> = ctx
        .file
        .chunks
        .iter()
        .enumerate()
        .filter(|(_, c)| c.status == "pending")
        .map(|(i, _)| i)
        .collect();
    if pendings.is_empty() {
        return None;
    }
    let &idx = pick_random(&pendings)?;
    let chunk = &mut ctx.file.chunks[idx];
    let base = parse_hex_key(&chunk.start_hex);
    let end = chunk_end(&base, chunk.range_bits);
    // Resume from prior current_hex IFF it lies within [base..end); else base.
    let resumed = chunk.current_hex.as_deref().and_then(|h| {
        let ck = parse_hex_key(h);
        (&ck >= &base && crate::gpu::convert::be_lt(&ck, &end)).then_some(ck)
    });
    let start = resumed.unwrap_or(base);
    chunk.status = "running".to_string();
    chunk.current_hex = None;
    let out = Some(GpuChunk {
        idx,
        chunk_id: chunk.chunk_index,
        start,
        end,
    });
    if let Err(e) = ctx.save() {
        eprintln!("[puzzle] GPU post-claim flush failed: {e}");
    }
    out
}

/// Finalize a GPU-scanned chunk: either finished (whole range done) or parked
/// (SIGINT) with the current scan position preserved for resume.
fn gpu_finalize_chunk(ctx: &mut PuzzleCtx, idx: usize, done: bool, current: [u8; 32]) {
    let chunk = &mut ctx.file.chunks[idx];
    if done {
        chunk.status = "finished".to_string();
        chunk.current_hex = None;
        ctx.file.completed_chunks += 1;
    } else {
        chunk.status = "pending".to_string();
        chunk.current_hex = Some(hex_encode_key(&current));
    }
    sync_flush_chunk(ctx);
}

/// Convert a GPU match (`scalar` = winning private key as LE limbs) into the
/// shared `MatchEvent`.  Re-derives the compressed pubkey on the CPU so the
/// output mirrors CPU-worker matches.
fn gpu_match_to_event(
    m: &crate::gpu::GpuMatchOutput,
    chunk: &GpuChunk,
) -> MatchEvent {
    let priv_be = crate::gpu::convert::limbs_to_be_bytes(&m.scalar);
    let secp = secp256k1::Secp256k1::new();
    let pk = secp256k1::SecretKey::from_byte_array(priv_be)
        .and_then(|sk| Ok(secp256k1::PublicKey::from_secret_key(&secp, &sk)))
        .unwrap_or(secp256k1::PublicKey::from_slice(
            &crate::btc::GENERATOR_COMPRESSED,
        )
        .unwrap());
    let compressed = pk.serialize().to_vec();
    let uncompressed = pk.serialize_uncompressed().to_vec();
    // Informational only: offset of the key within the chunk.  Compute it
    // defensively — the GPU scalar is reconstructed from LE limbs and may be
    // off by a hair, so a naive subtraction can underflow (panic).  Saturate
    // to 0 rather than crash on what is purely a report field.
    let key_index = if crate::gpu::convert::be_lt(&priv_be, &chunk.start) {
        0
    } else {
        let diff = crate::gpu::convert::scalar_sub_be(&priv_be, &chunk.start);
        if diff > i64::MAX as u64 {
            0
        } else {
            diff
        }
    };
    MatchEvent {
        private_key: priv_be,
        compressed,
        uncompressed,
        worker_id: 0, // GPU worker id — report distinguishes via key origin; kept 0
        chunk_id: Some(chunk.chunk_id),
        key_index,
        elapsed: 0.0, // filled in by the caller after CPU verification
    }
}

/// One GPU worker thread.  Claims pending chunks from the shared worklist and
/// dense-tiles each `[start, end)` with 100k strided walkers.  Per dispatch it
/// covers `N × steps_per_call` keys with no overlap, advancing the checkpoint.
/// Runs alongside the CPU workers — all of them pull from the same pending
/// queue, so CPU and GPU share the load without double-scanning.
#[allow(clippy::too_many_arguments)]
fn gpu_puzzle_worker(
    target_h160: [u8; 20],
    ctx: Arc<Mutex<PuzzleCtx>>,
    progress: &Progress,
    matches: &Mutex<Vec<MatchEvent>>,
    stop_flag: &AtomicBool,
    start: Instant,
) {
    // Set up GPU.  If no Metal device is available (CI, headless) we log and
    // fall back to CPU-only — never block the whole run on a missing GPU.
    let gpu_ctx = match crate::gpu::GpuContext::new_blocking(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[puzzle] GPU unavailable ({e}) — running CPU-only.");
            return;
        }
    };
    eprintln!(
        "[puzzle] GPU worker up on {}",
        gpu_ctx.device_name()
    );
    let candidates = crate::gpu::convert::hash160_to_candidates(&target_h160);
    let mut scanner = match crate::gpu::GpuScanner::new(gpu_ctx, &candidates) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[puzzle] GpuScanner::new failed ({e}) — running CPU-only.");
            return;
        }
    };
    // Dense-tiling config: stride = N threads, single target candidate.
    scanner.stride = crate::gpu::NUM_GPU_THREADS;
    scanner.num_candidates = 1;

    // Keep `rotate_keys` disabled (the user-supplied chunk queue already
    // subdivides the range; rotation only ever made sense for CPU-only sweeps).

    // Per-dispatch coverage (keys).  Constant once calibrated.
    let dispatch_keys = crate::gpu::NUM_GPU_THREADS as u64
        * scanner.steps_per_call as u64;

    loop {
        // ── claim a pending chunk ────────────────────────────────────────────
        let claim = {
            let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
            gpu_claim_chunk(&mut ctx)
        };
        let chunk = match claim {
            Some(c) => c,
            None => return, // all chunks claimed/finished
        };

        // Seed walkers at `start + i` so they tile [start, start+N) with stride N.
        if scanner.seed_range(chunk.start).is_err() {
            eprintln!("[puzzle] GPU seed_range failed — parking chunk");
            let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
            gpu_finalize_chunk(&mut ctx, chunk.idx, false, chunk.start);
            continue;
        }

        let mut current = chunk.start; // next key NOT yet covered
        let mut parked = false;

        // ── scan the chunk in N·steps_per_call-key dispatches ───────────────
        loop {
            // Decide this dispatch's step count.  A full dispatch covers
            // `dispatch_keys` keys; the final (partial) dispatch is trimmed so the
            // walkers land on or just past `end`.  The catch: the chunk width can
            // exceed 2^64 (puzzle #76 spans 2^65), so we MUST NOT compute
            // `end - current` directly (it overflows u64).  Instead we compare
            // `end` against `current + dispatch_keys` (adding the small u64 never
            // overflows) and only subtract once we know the remainder fits.
            let steps = if crate::gpu::convert::be_lt(&current, &chunk.end) {
                let reach = crate::gpu::convert::scalar_add_be(&current, dispatch_keys);
                // `reach >= end`  ⟺  `end - current <= dispatch_keys` (no overflow).
                if !crate::gpu::convert::be_lt(&reach, &chunk.end) {
                    let remaining = crate::gpu::convert::scalar_sub_be(&chunk.end, &current);
                    let n = crate::gpu::NUM_GPU_THREADS as u64;
                    std::cmp::max(1, (remaining + n - 1) / n) as u32
                } else {
                    scanner.steps_per_call
                }
            } else {
                0
            };
            if steps == 0 {
                break; // reached the exclusive end
            }
            // Temporarily set steps_per_call for this (possibly final, partial)
            // dispatch, restoring the default afterwards.
            let saved_steps = scanner.steps_per_call;
            scanner.steps_per_call = steps;
            let batch = crate::gpu::NUM_GPU_THREADS as u64 * steps as u64;
            match scanner.step() {
                Ok(batch_matches) => {
                    if !batch_matches.is_empty() {
                        let mut g = matches.lock().unwrap_or_else(|e| e.into_inner());
                        for m in &batch_matches {
                            let mut ev = gpu_match_to_event(m, &chunk);
                            // CPU verification — a real puzzle solver never trusts the
                            // GPU candidate flag alone: re-derive the pubkey, hash160
                            // it, and confirm it equals the target.  Spurious GPU
                            // matches (impossible with a 160-bit hash, but defense in
                            // depth) are dropped here silently.
                            let h = btc::hash160(&ev.compressed);
                            if h == target_h160 {
                                ev.elapsed = start.elapsed().as_secs_f64();
                                g.push(ev);
                            }
                        }
                    }
                    progress.increment(batch);
                }
                Err(e) => {
                    eprintln!("[puzzle] GPU step failed ({e}) — parking chunk");
                    parked = true;
                    break;
                }
            }
            scanner.steps_per_call = saved_steps;

            // Advance checkpoint by exactly the keys this dispatch covered.
            current = crate::gpu::convert::scalar_add_be(&current, batch);

            // Refresh the in-memory resume position (disk flush is on the ticker).
            {
                let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
                ctx.file.chunks[chunk.idx].current_hex = Some(hex_encode_key(&current));
            }

            if stop_flag.load(Ordering::Relaxed) {
                parked = true;
                break;
            }
        }

        // ── finalize ────────────────────────────────────────────────────────
        {
            let mut ctx = ctx.lock().unwrap_or_else(|e| e.into_inner());
            let done = !parked
                && !crate::gpu::convert::be_lt(&current, &chunk.end);
            gpu_finalize_chunk(&mut ctx, chunk.idx, done, current);
        }

        if stop_flag.load(Ordering::Relaxed) {
            return;
        }
    }
}

// ── hex helpers ─────────────────────────────────────────────────────────────

/// Parse a hex string as a 32-byte big-endian key.  Odd-length inputs get a
/// leading "0" to make them even; the result is left-padded (high bytes) to a
/// full 64 hex-char / 32-byte key.
pub(crate) fn parse_hex_key(hex_str: &str) -> [u8; 32] {
    let s = hex_str.strip_prefix("0x").unwrap_or(hex_str);
    let s = if s.len() % 2 == 1 {
        format!("0{s}")
    } else {
        s.to_string()
    };
    assert!(
        s.len() <= 64,
        "hex key {} exceeds 64 chars ({} chars)",
        s,
        s.len()
    );
    let padded = format!("{:0>64}", s); // left-pad with zeros to 64 chars
                                    // hex::decode requires an even-length string — guaranteed by:
                                    //   (a) odd inputs get a leading zero above,
                                    //   (b) the resulting length (max 64) is even.
    let raw = hex::decode(padded).unwrap_or_else(|e| panic!("invalid hex key {hex_str}: {e}"));
    let mut out = [0u8; 32];
    out.copy_from_slice(&raw);
    out
}

/// Hex-encode a 32-byte key (always 64 hex chars, no padding ambiguity).
pub(crate) fn hex_encode_key(bytes: &[u8; 32]) -> String {
    hex::encode(bytes)
}

/// Parse a 40-character hex string as a 20-byte hash160.  Returns `None` if the
/// string isn't exactly 40 hex chars (the canonical RIPEMD-160 length).
fn hash160_from_hex(s: &str) -> Option<[u8; 20]> {
    if s.len() != 40 {
        return None;
    }
    let raw = hex::decode(s).ok()?;
    if raw.len() != 20 {
        return None;
    }
    let mut out = [0u8; 20];
    out.copy_from_slice(&raw);
    Some(out)
}

/// Compute the exclusive upper bound (`start + 2^range_bits`) for a chunk as a
/// 32-byte big-endian key.  `range_bits` must be ≤ 255.
pub(crate) fn chunk_end(start: &[u8; 32], range_bits: u32) -> [u8; 32] {
    assert!(range_bits <= 255, "range_bits {range_bits} out of range");
    let mut result = *start;
    // 2^range_bits in a big-endian [u8; 32]:
    //   byte index (from MSB) = 31 - (range_bits / 8)
    //   bit within that byte (counting from the byte's LSB) = `range_bits % 8`
    let byte_idx = (31 - (range_bits / 8)) as usize;
    let bit_in_byte = (range_bits % 8) as u8;
    let mut carry = (1u16) << bit_in_byte;
    let v = result[byte_idx] as u16 + carry;
    result[byte_idx] = (v & 0xFF) as u8;
    carry = v >> 8;
    let mut i = byte_idx;
    while carry > 0 && i > 0 {
        i -= 1;
        let v = result[i] as u16 + carry;
        result[i] = (v & 0xFF) as u8;
        carry = v >> 8;
    }
    result
}

/// Compare a serialised compressed/uncompressed pubkey against the target's
/// 20-byte hash160.  Inlinable, branchless-friendly wrapper around what used
/// to be two separate `hash160` calls on the hot path.
#[inline(always)]
fn h160_eq(pubkey: &[u8], target: [u8; 20]) -> bool {
    btc::hash160(pubkey) == target
}

/// Persist the worklist chunk metadata (status + current_hex) to disk.
/// Used after both claim and finalize so a Ctrl+C or hard-kill between
/// heartbeat ticks cannot silently drop a ~10-minute window of scanning.
/// Failures are logged but never panic — losing a pending write is preferable
/// to tearing down a worker mid-scan.
#[inline]
fn sync_flush_chunk(ctx: &mut PuzzleCtx) {
    if let Err(e) = ctx.save() {
        eprintln!("[puzzle] chunk flush failed: {e}");
    }
}

/// Pick a random element of a non-empty slice using the OS RNG.  Falls back to
/// the first element of the slice if entropy is unavailable.
fn pick_random<T>(slice: &[T]) -> Option<&T> {
    if slice.is_empty() {
        return None;
    }
    let mut buf = [0u8; 4];
    if rand::rngs::SysRng.try_fill_bytes(&mut buf).is_err() {
        return Some(&slice[0]);
    }
    let r = u32::from_le_bytes(buf) as usize % slice.len();
    Some(&slice[r])
}

/// Return the (pending, running, finished) counts of a worklist.
fn chunk_summary(file: &PuzzleFile) -> (usize, usize, usize) {
    let mut pending = 0usize;
    let mut running = 0usize;
    let mut finished = 0usize;
    for c in &file.chunks {
        match c.status.as_str() {
            "pending" => pending += 1,
            "running" => running += 1,
            "finished" => finished += 1,
            _ => {}
        }
    }
    (pending, running, finished)
}

impl PuzzleCtx {
    /// Persist the worklist back to disk atomically (write-temp + rename).
    fn save(&self) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.file).map_err(|e| e.to_string())?;
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, &json).map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &self.path).map_err(|e| e.to_string())?;
        Ok(())
    }
}
