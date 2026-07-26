//! GPU lottery worker — random-walk scanner against the full 78-puzzle set.
//!
//! One GPU worker thread runs alongside the CPU workers in `workers::run`.  It
//! seeds 100k independent walkers at random positions inside the puzzle key space
//! and lets each walk `P += G` (stride 1), checking every key against all 78
//! candidate hash160s.  Unlike puzzle mode there is no worklist and no range to
//! tile: the walkers run forever, and we periodically **re-seed** them at fresh
//! random positions to keep sampling the key space uniformly.
//!
//! Re-seeding is the key difference from puzzle mode's deterministic tiling:
//! a walker that starts at `s` and walks +1 covers only the arithmetic
//! progression `s, s+1, s+2, ...`.  Without re-seeding, 100k walkers would
//! each own one infinite ray and never revisit the space between rays.  By
//! re-seeding every `RESEED_INTERVAL_KEYS` keys we collapse all rays and
//! redistribute — giving uniform random coverage of [2^70, 2^160) over time.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::btc;
use crate::gpu::{convert, GpuContext, GpuMatchOutput, GpuScanner, NUM_GPU_THREADS};
use crate::progress::Progress;
use crate::workers::MatchEvent;

/// Re-seed the GPU walkers after this many keys have been scanned.
///
/// Trade-off: re-seeding costs one CPU-side batch of 100k scalar mults
/// (~10-15 ms on ~8 cores, parallelized with rayon) during which the GPU is
/// idle.  At ~100 Mkeys/s that is a re-seed every ~0.67 s and an overhead of
/// ~2%.  Large enough that the stall is negligible; small enough that walkers
/// are redistributed frequently and cover the key space uniformly.
pub const RESEED_INTERVAL_KEYS: u64 = 1 << 26;

/// One GPU lottery worker thread.
///
/// `puzzle_set` is the embedded 78-puzzle table (used for range-constrained
/// seeding AND for CPU-side match verification).  `progress` and `matches` are
/// shared with the CPU workers.  `deadline` is the global runtime limit
/// (None = run forever, until SIGINT).  `start` is the run start instant for
/// match timestamps.
pub fn worker(
    puzzle_set: &'static crate::puzzles::PuzzleSet,
    progress: Arc<Progress>,
    matches: Arc<Mutex<Vec<MatchEvent>>>,
    deadline: Option<Instant>,
    start: Instant,
) {
    // Set up GPU.  If no device is available (CI, headless, no Metal/Vulkan)
    // we log and fall back to CPU-only — never block the whole run on a GPU.
    let ctx = match GpuContext::new_blocking(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  [GPU] unavailable ({e}) — lottery running CPU-only.");
            return;
        }
    };
    eprintln!("  [GPU] lottery worker up on {}", ctx.device_name());

    // Candidate buffer: all 78 puzzle hash160s.  The shader's candidate array
    // is fixed at 78 slots; `num_candidates = 78` checks them all.
    let candidates = convert::puzzle_set_to_candidates(puzzle_set);
    let candidate_count = puzzle_set.ranges().len() as u32;

    let mut scanner = match GpuScanner::new(ctx, &candidates) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  [GPU] GpuScanner::new failed ({e}) — lottery running CPU-only.");
            return;
        }
    };
    // Lottery config: stride 1 (step = G), all 78 candidates.  These are already
    // the `new()` defaults, but set them explicitly for clarity.
    scanner.stride = 1;
    scanner.num_candidates = candidate_count;

    // Seed walkers at random positions inside the puzzle key space.
    if let Err(e) = scanner.init_random(puzzle_set) {
        eprintln!("  [GPU] init_random failed ({e}) — lottery running CPU-only.");
        return;
    }

    let batch: u64 = NUM_GPU_THREADS as u64 * scanner.steps_per_call as u64;
    let mut keys_since_reseed = 0u64;

    loop {
        // Check deadline (if any) before dispatching.
        if deadline.is_some_and(|dl| Instant::now() >= dl) {
            break;
        }

        // Throttle: on non-macOS, sleep_ms leaves GPU headroom for display.
        // (Handled inside scanner.step().)

        match scanner.step() {
            Ok(batch_matches) => {
                if !batch_matches.is_empty() {
                    let mut g = matches.lock().unwrap_or_else(|e| e.into_inner());
                    for m in &batch_matches {
                        let mut ev = gpu_match_to_event(m, puzzle_set, start);
                        // CPU verification — re-derive the pubkey, hash160 it, and
                        // confirm it matches the puzzle set.  Defense in depth
                        // against spurious GPU matches (impossible with a 160-bit
                        // hash, but cheap insurance).
                        let h = btc::hash160(&ev.compressed);
                        if let Some(pn) = puzzle_set.puzzle_number_for_hash160(&h) {
                            ev.puzzle_number = Some(pn);
                            g.push(ev);
                        }
                        // Uncompressed variant: the GPU kernel hashes both the
                        // compressed and uncompressed pubkeys.  We verified the
                        // compressed one above; check the uncompressed too.
                        else {
                            let h_u = btc::hash160(&ev.uncompressed);
                            if let Some(pn) = puzzle_set.puzzle_number_for_hash160(&h_u) {
                                ev.puzzle_number = Some(pn);
                                g.push(ev);
                            }
                            // else: spurious match — drop silently.
                        }
                    }
                }
                progress.increment(batch);
                keys_since_reseed += batch;
            }
            Err(e) => {
                eprintln!("  [GPU] step failed ({e}) — stopping GPU worker (CPU workers continue).");
                break;
            }
        }

        // Re-seed: collapse all walker rays and redistribute at fresh random
        // positions.  This is what keeps the scan a uniform random sample of
        // the key space instead of 100k fixed arithmetic progressions.
        if keys_since_reseed >= RESEED_INTERVAL_KEYS {
            if let Err(e) = scanner.init_random(puzzle_set) {
                eprintln!("  [GPU] re-seed failed ({e}) — keeping current walkers.");
            }
            keys_since_reseed = 0;
        }
    }

    eprintln!(
        "  [GPU] lottery worker done. total_keys={}",
        scanner.total_ops
    );
}

/// Convert a GPU match (`scalar` = winning private key as LE limbs) into the
/// shared `MatchEvent`.  Re-derives the compressed pubkey on the CPU so the
/// output mirrors CPU-worker matches.  `puzzle_number` is filled in by the
/// caller after CPU verification.
fn gpu_match_to_event(
    m: &GpuMatchOutput,
    _puzzle_set: &crate::puzzles::PuzzleSet,
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
        worker_id: 0, // GPU worker — report distinguishes via origin; kept 0
        chunk_id: None,
        key_index: 0, // meaningless for lottery (no chunk, no sequential index)
        elapsed: start.elapsed().as_secs_f64(),
        puzzle_number: None, // filled in by the caller after CPU verification
    }
}

/// A small helper so the deadline check can be tested without a GPU.  Not used
/// in the hot path — kept here to document the re-seed cadence.
#[allow(dead_code)]
pub fn reseed_interval() -> Duration {
    // Approximate: at ~100 Mkeys/s, RESEED_INTERVAL_KEYS keys take ~0.67 s.
    // The real cadence depends on GPU throughput; this is just for logging/tests.
    Duration::from_nanos(RESEED_INTERVAL_KEYS * 1_000_000_000 / 100_000_000)
}
