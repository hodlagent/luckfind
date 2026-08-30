//! CUDA lottery worker — random-walk scanner against the full 77-puzzle set.
//!
//! Mirrors `crate::gpu::lottery::worker` but uses the CUDA backend instead of
//! WebGPU.  The scanning algorithm is identical: 100k independent walkers,
//! each walking P += G (stride 1), checking every key against all 77 candidate
//! hash160s, with periodic re-seeding for uniform key-space coverage.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::btc;
use crate::config::BtcCheck;
use crate::cuda::scanner::CudaScanner;
use crate::gpu::{convert, GpuMatchOutput};
use crate::progress::Progress;
use crate::workers::MatchEvent;

/// Re-seed the GPU walkers after this many keys have been scanned.
pub const RESEED_INTERVAL_KEYS: u64 = 1 << 26;

/// One CUDA GPU lottery worker thread for a single device ordinal.
pub fn worker(
    puzzle_set: &'static crate::puzzles::PuzzleSet,
    device_index: u32,
    progress: Arc<Progress>,
    matches: Arc<Mutex<Vec<MatchEvent>>>,
    stop_flag: Arc<AtomicBool>,
    hit_flag: Arc<AtomicBool>,
    deadline: Option<Instant>,
    start: Instant,
    check: BtcCheck,
) {
    // Set up CUDA scanner on this device.  If that device is unavailable (e.g.
    // a too-old GPU that cannot JIT the PTX), log and fall back — other devices
    // have their own workers and are unaffected.
    let candidates = convert::puzzle_set_to_candidates(puzzle_set);
    let candidate_count = puzzle_set.ranges().len() as u32;

    let mut scanner = match CudaScanner::new_on_device(&candidates, device_index) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "  [CUDA] device #{device_index} scanner init failed ({e}) — lottery running CPU-only on this card."
            );
            return;
        }
    };
    eprintln!(
        "  [CUDA] lottery worker #{} up on {}",
        scanner.device_index(),
        scanner.device_name()
    );

    // Lottery config: stride 1 (step = G), all candidates.
    scanner.stride = 1;
    scanner.num_candidates = candidate_count;
    scanner.check_compressed_pk = check.compressed as u32;
    scanner.check_uncompressed_pk = check.uncompressed as u32;

    // Seed walkers at random positions inside the puzzle key space.
    if let Err(e) = scanner.init_random(puzzle_set) {
        eprintln!("  [CUDA] init_random failed ({e}) — lottery running CPU-only.");
        return;
    }

    let batch: u64 = crate::cuda::NUM_GPU_THREADS as u64 * scanner.steps_per_call as u64;
    let mut keys_since_reseed = 0u64;

    loop {
        // Check deadline (if any) before dispatching.
        if deadline.is_some_and(|dl| Instant::now() >= dl) {
            break;
        }
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        match scanner.step() {
            Ok(batch_matches) => {
                if !batch_matches.is_empty() {
                    let verified: Vec<MatchEvent> = {
                        let mut g = matches.lock().unwrap_or_else(|e| e.into_inner());
                        let mut out = Vec::new();
                        for m in &batch_matches {
                            let mut ev = gpu_match_to_event(m, puzzle_set, device_index, start);
                            // CPU verification — re-derive the pubkey, hash160 it, and
                            // confirm it matches the puzzle set.  Gated by the same
                            // `[btc]` switches as the kernel, so a serialisation that
                            // is disabled for checking is never accepted here either.
                            // The `matched` lookup keeps the two candidate pushes in
                            // if/else exclusivity so `ev` is moved exactly once.
                            let matched = if check.compressed {
                                puzzle_set
                                    .puzzle_number_for_hash160(&btc::hash160(&ev.compressed))
                            } else {
                                None
                            };
                            if let Some(pn) = matched {
                                ev.puzzle_number = Some(pn);
                                g.push(ev.clone());
                                out.push(ev);
                            } else if check.uncompressed {
                                if let Some(pn) = puzzle_set
                                    .puzzle_number_for_hash160(&btc::hash160(&ev.uncompressed))
                                {
                                    ev.puzzle_number = Some(pn);
                                    g.push(ev.clone());
                                    out.push(ev);
                                }
                                // else: spurious match — drop silently.
                            }
                        }
                        out
                    };
                    if let Some(first) = verified.first() {
                        if hit_flag
                            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                            .is_ok()
                        {
                            println!(
                                "[HIT] 🎯 puzzle=#{} worker=CUDA[{}] sk_hex={}",
                                first.puzzle_number
                                    .map_or_else(String::new, |n| n.to_string()),
                                device_index,
                                hex::encode(first.private_key),
                            );
                        }
                        stop_flag.store(true, Ordering::SeqCst);
                    }
                }
                progress.increment(batch);
                keys_since_reseed += batch;
            }
            Err(e) => {
                eprintln!("  [CUDA] step failed ({e}) — stopping CUDA worker (CPU workers continue).");
                break;
            }
        }

        // Re-seed walkers periodically.
        if keys_since_reseed >= RESEED_INTERVAL_KEYS {
            if let Err(e) = scanner.init_random(puzzle_set) {
                eprintln!("  [CUDA] re-seed failed ({e}) — keeping current walkers.");
            }
            keys_since_reseed = 0;
        }
    }

    eprintln!(
        "  [CUDA] lottery worker #{} done. total_keys={}",
        scanner.device_index(),
        scanner.total_ops
    );
}

/// Convert a CUDA match output into a MatchEvent.
fn gpu_match_to_event(
    m: &GpuMatchOutput,
    _puzzle_set: &crate::puzzles::PuzzleSet,
    device_index: u32,
    start: Instant,
) -> MatchEvent {
    let priv_be = convert::limbs_to_be_bytes(&m.scalar);
    let secp = secp256k1::Secp256k1::new();
    let pk = secp256k1::SecretKey::from_byte_array(priv_be)
        .and_then(|sk| Ok(secp256k1::PublicKey::from_secret_key(&secp, &sk)))
        .unwrap_or(crate::btc::generator_public_key());
    MatchEvent {
        private_key: priv_be,
        compressed: pk.serialize().to_vec(),
        uncompressed: pk.serialize_uncompressed().to_vec(),
        // CUDA device ordinal — distinguishes which physical card found it.
        worker_id: device_index,
        chunk_id: None,
        key_index: 0,
        elapsed: start.elapsed().as_secs_f64(),
        puzzle_number: None,
    }
}
