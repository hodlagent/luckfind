//! Remote (LAN-hub) worker mode.
//!
//! `luckfind --remote http://{hub_ip}:42069` turns this machine into a worker
//! for a puzzle worklist owned by a lan-hub (`/Users/jerin/Dev/lan-hub`).  The
//! hub holds the SQLite `.db` and is the single writer; workers never open a
//! local database.  Each worker thread:
//!
//!   1. claims one chunk over HTTP (`POST /api/chunks/claim`),
//!   2. scans it with the shared CPU core (`crate::puzzle::scan_chunk`),
//!   3. keeps the lease alive with throttled heartbeats (~30s ≪ the hub's 120s
//!      reclaim timeout) that also carry the scan position,
//!   4. reports back via `done` (finished / match) or `release` (parked, with
//!      the resume position — forward `current`, reverse `end`).
//!
//! Crash recovery is the hub's job: a worker that dies (or loses the network)
//! stops heartbeating, and the hub reclaims the lease after 120s, reverting the
//! chunk to `pending` at the last reported position.  A 404/409 from the hub
//! means our lease is already gone — we abandon the chunk and re-claim, never
//! crashing.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::json;

use crate::btc;
use crate::progress::Progress;
use crate::puzzle::{
    self, abbr_hex, hash160_from_hex, hex_encode_key, parse_hex_key, scan_chunk,
    term_line, term_status, ResumePosition, ScanChunkOptions, ScanDir,
};
use crate::workers::{fmt_comma, MatchEvent};

/// Per-claim rotation budget: park the chunk and re-claim after this many keys,
/// exactly matching local puzzle mode (`puzzle::ROTATION_BUDGET`).
const ROTATION_BUDGET: u64 = puzzle::ROTATION_BUDGET;

/// Lease-refresh cadence.  The hub reclaims leases after 120s, so 30s gives a
/// comfortable margin while keeping LAN traffic negligible.
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);

/// Sleep between claim attempts when the hub has nothing to hand out (every
/// pending chunk is already running on another worker).
const CLAIM_IDLE: Duration = Duration::from_secs(2);

// ── hub API response shapes (see lan-hub backend/app/routes.py) ──────────────

#[derive(Debug, Deserialize)]
struct HubStatus {
    meta: HubMeta,
    summary: HubSummary,
    #[serde(default)]
    workers: Vec<HubWorker>,
}

#[derive(Debug, Deserialize)]
struct HubMeta {
    puzzle_number: u32,
    target: String,
    hash160: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HubSummary {
    pending: u64,
    running: u64,
    #[serde(default)]
    finished: u64,
}

#[derive(Debug, Deserialize)]
struct HubWorker {
    #[serde(default)]
    worker_id: String,
    #[serde(default)]
    chunk_count: usize,
    #[serde(default)]
    chunks: Vec<HubWorkerChunk>,
}

#[derive(Debug, Deserialize)]
struct HubWorkerChunk {
    #[serde(default)]
    progress_pct: f64,
}

#[derive(Debug, Deserialize)]
struct ClaimResponse {
    granted: usize,
    #[serde(default)]
    chunks: Vec<ClaimedChunk>,
}

#[derive(Debug, Deserialize)]
struct ClaimedChunk {
    id: u32,
    current_hex: String,
    end_hex: String,
}

/// True when the hub says the lease is gone: 404 (no lease on that chunk) or
/// 409 (the chunk is now owned by someone else).  The worker then abandons the
/// chunk and re-claims instead of fighting over it.
fn is_lease_lost(e: &ureq::Error) -> bool {
    matches!(e, ureq::Error::StatusCode(404 | 409))
}

// ── blocking HTTP wrapper ────────────────────────────────────────────────────

/// Thin wrapper over `ureq::Agent` for the lan-hub API.  Timeouts are set so a
/// dead hub never hangs a worker thread forever (the hub reclaims leases after
/// 120s regardless of whether the worker noticed).
struct HubClient {
    agent: ureq::Agent,
    base: String,
}

impl HubClient {
    fn new(base: &str) -> Self {
        let base = base.trim_end_matches('/').to_string();
        // Split the chain so the concrete `ConfigBuilder<AgentScope>` type is
        // pinned before `new_agent` (the timeout builders are generic over the
        // scope and would otherwise leave it ambiguous).
        let builder = ureq::config::Config::builder()
            .timeout_connect(Some(Duration::from_secs(5)))
            .timeout_per_call(Some(Duration::from_secs(15)));
        let agent = builder.build().new_agent();
        Self { agent, base }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    fn status(&self) -> Result<HubStatus, ureq::Error> {
        let resp = self.agent.get(&self.url("/api/status")).call()?;
        resp.into_body().read_json()
    }

    fn claim(&self, worker_id: &str, count: usize) -> Result<ClaimResponse, ureq::Error> {
        let resp = self
            .agent
            .post(&self.url("/api/chunks/claim"))
            .send_json(json!({ "worker_id": worker_id, "count": count }))?;
        resp.into_body().read_json()
    }

    fn heartbeat(
        &self,
        chunk_id: u32,
        worker_id: &str,
        current_hex: Option<String>,
        end_hex: Option<String>,
        keys: Option<u64>,
        rate: Option<f64>,
    ) -> Result<(), ureq::Error> {
        let body = ChunkUpdateBody {
            worker_id: worker_id.to_string(),
            current_hex,
            end_hex,
            keys,
            rate,
        };
        let _ = self
            .agent
            .post(&self.url(&format!("/api/chunks/{chunk_id}/heartbeat")))
            .send_json(body)?;
        Ok(())
    }

    fn done(&self, chunk_id: u32, worker_id: &str) -> Result<(), ureq::Error> {
        let _ = self
            .agent
            .post(&self.url(&format!("/api/chunks/{chunk_id}/done")))
            .send_json(json!({ "worker_id": worker_id }))?;
        Ok(())
    }

    fn release(
        &self,
        chunk_id: u32,
        worker_id: &str,
        current_hex: Option<String>,
        end_hex: Option<String>,
    ) -> Result<(), ureq::Error> {
        let body = ChunkUpdateBody {
            worker_id: worker_id.to_string(),
            current_hex,
            end_hex,
            keys: None,
            rate: None,
        };
        let _ = self
            .agent
            .post(&self.url(&format!("/api/chunks/{chunk_id}/release")))
            .send_json(body)?;
        Ok(())
    }
}

/// `{worker_id, current_hex?, end_hex?, keys?, rate?}` — only present fields are
/// sent (the hub's Pydantic model accepts either, `end_hex` being the new
/// reverse-park field; `keys`/`rate` are transient metrics the hub caches in
/// memory, and older hubs simply ignore the unknown keys).
#[derive(serde::Serialize)]
struct ChunkUpdateBody {
    worker_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    current_hex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    end_hex: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keys: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rate: Option<f64>,
}

// ── entry point ──────────────────────────────────────────────────────────────

pub fn run(
    remote_url: &str,
    worker_id: String,
    n_workers: usize,
    heartbeat_secs: f64,
    output_dir: Option<&Path>,
) -> (Arc<Progress>, Vec<MatchEvent>) {
    let client = Arc::new(HubClient::new(remote_url));

    // ── 1. connect to the hub and read the puzzle meta ─────────────────────
    let (target_h160, puzzle_number, summary) = connect(&client);
    if summary.pending + summary.running == 0 {
        println!("[remote] hub reports no pending or running chunks — nothing to do.");
        return (Arc::new(Progress::new(0)), Vec::new());
    }

    // ── 2. shared state + SIGINT handler (mirrors puzzle.rs) ───────────────
    let progress = Arc::new(Progress::new(n_workers as u64));
    let matches = Arc::new(Mutex::new(Vec::<MatchEvent>::new()));
    let stop_flag = Arc::new(AtomicBool::new(false));
    // 命中即停：hit_flag 记录首个命中者（赢家负责打印 [HIT]）；stop_flag 由
    // 命中或首次 Ctrl+C 置位，只负责让 worker/ticker 停下。
    let hit_flag = Arc::new(AtomicBool::new(false));
    // sigint_flag 区分「第一次 Ctrl+C」（优雅退出）与第二次（立即 abort）。
    let sigint_flag = Arc::new(AtomicBool::new(false));

    let sigint_handler = sigint_flag.clone();
    let stop_handler = stop_flag.clone();
    ctrlc::set_handler(move || {
        if sigint_handler
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            stop_handler.store(true, Ordering::SeqCst);
            term_line("[remote] Ctrl+C — releasing chunks and stopping workers …");
        } else {
            term_line("[remote] second Ctrl+C — aborting immediately");
            std::process::exit(130);
        }
    })
    .expect("[remote] failed to install SIGINT handler");

    // ── 3. worker threads + status ticker ──────────────────────────────────
    let start = Instant::now();
    let mut handles = Vec::with_capacity(n_workers);
    for wid in 0..n_workers {
        let client = client.clone();
        let worker_id = worker_id.clone();
        let progress = progress.clone();
        let matches = matches.clone();
        let stop_flag = stop_flag.clone();
        let hit_flag = hit_flag.clone();
        handles.push(std::thread::spawn(move || {
            remote_worker(
                wid as u32,
                &client,
                &worker_id,
                target_h160,
                puzzle_number,
                &progress,
                &matches,
                &stop_flag,
                &hit_flag,
                start,
            )
        }));
    }

    let ticker_client = client.clone();
    let ticker_id = worker_id.clone();
    let ticker_stop = stop_flag.clone();
    let ticker_progress = progress.clone();
    let ticker_handle = std::thread::spawn(move || {
        ticker(
            &ticker_client,
            &ticker_id,
            &ticker_progress,
            &ticker_stop,
            heartbeat_secs,
        )
    });

    for h in handles {
        drop(h.join());
    }
    // Every worker has returned.  Stop the ticker and wait for it to exit so
    // a stale status line can't race the final report — the "all done" return
    // path never sets stop_flag, so set it explicitly now that nothing else is
    // running.
    stop_flag.store(true, Ordering::SeqCst);
    let _ = ticker_handle.join();

    let final_matches = {
        match Arc::try_unwrap(matches) {
            Ok(m) => m.into_inner().unwrap_or_else(|e| e.into_inner()),
            Err(a) => a.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        }
    };

    // 命中即停收尾顺序：终端已打印 [HIT] → 先落盘 aman_<TS>.txt（worker PC 本地）。
    crate::report::flush_match_files(&final_matches, output_dir);

    // ── 4. final report ────────────────────────────────────────────────────
    term_line("");
    eprintln!("──────────────────────────────────────────────────");
    eprintln!("  REMOTE SCAN COMPLETE  (hub={})", client.base);
    eprintln!("──────────────────────────────────────────────────");
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
            "  Match      : {} event(s) — see aman_<TS>.txt",
            final_matches.len()
        );
        for m in &final_matches {
            let chunk_label = match m.chunk_id {
                Some(id) => format!(" chunk={id}"),
                None => String::new(),
            };
            eprintln!(
                "    worker={}{} idx={} sk_hex={}",
                m.worker_id,
                chunk_label,
                fmt_comma(m.key_index),
                hex::encode(m.private_key),
            );
        }
    }
    eprintln!();

    (progress, final_matches)
}

/// Fetch `/api/status` once at startup (retrying until the hub is reachable,
/// up to ~90s), then decode + verify the target hash160 — the same startup
/// check local puzzle mode performs against its worklist.
fn connect(client: &HubClient) -> ([u8; 20], Option<u32>, HubSummary) {
    let deadline = Instant::now() + Duration::from_secs(90);
    let status = loop {
        match client.status() {
            Ok(s) => break s,
            Err(e) => {
                if Instant::now() >= deadline {
                    eprintln!("[remote] hub unreachable after 90s ({e}) — giving up.");
                    std::process::exit(2);
                }
                term_line(&format!("[remote] hub unreachable ({e}) — retrying …"));
                std::thread::sleep(Duration::from_secs(2));
            }
        }
    };

    let meta = &status.meta;
    let target_h160 = btc::legacy_address_hash160(&meta.target).unwrap_or_else(|| {
        eprintln!("[remote] hub target {} is not a valid P2PKH address", meta.target);
        std::process::exit(2);
    });
    if let Some(ref h160) = meta.hash160 {
        match hash160_from_hex(h160) {
            Some(expected) if expected == target_h160 => {
                term_line(&format!("[remote] hash160 OK ({h160})"));
            }
            Some(_) => {
                eprintln!(
                    "[remote] hash160 MISMATCH: hub says {h160}, target {} decodes to {}",
                    meta.target,
                    hex::encode(target_h160)
                );
                std::process::exit(2);
            }
            None => {
                eprintln!("[remote] hub hash160 is not valid hex: {h160}");
                std::process::exit(2);
            }
        }
    }

    term_line(&format!(
        "[remote] hub={} puzzle=#{} target={}  pend={} running={} done={}",
        client.base,
        meta.puzzle_number,
        meta.target,
        status.summary.pending,
        status.summary.running,
        status.summary.finished,
    ));
    (target_h160, Some(meta.puzzle_number), status.summary)
}

/// Background status line: every `heartbeat_secs`, query the hub and rewrite
/// the in-place line with the global view plus this worker's chunk progress.
fn ticker(
    client: &HubClient,
    worker_id: &str,
    progress: &Progress,
    stop_flag: &AtomicBool,
    heartbeat_secs: f64,
) {
    let mut prev_total = 0u64;
    let mut prev_time = Instant::now();
    loop {
        // Sleep in short slices so a stop (Ctrl+C / hit / all-done) is noticed
        // within ~0.2s instead of on the next full heartbeat_secs tick — this
        // keeps run()'s final `join` prompt and prevents a stale status line
        // after the workers have all returned.
        let slice = Duration::from_millis(200);
        let mut slept = Duration::ZERO;
        while slept < Duration::from_secs_f64(heartbeat_secs) {
            if stop_flag.load(Ordering::Relaxed) {
                return;
            }
            std::thread::sleep(slice);
            slept += slice;
        }
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        let now = Instant::now();
        let total = progress.checked.load(Ordering::Relaxed);
        let dt = now.duration_since(prev_time).as_secs_f64();
        let rate = if dt > 0.1 {
            (total - prev_total) as f64 / dt
        } else {
            0.0
        };
        prev_total = total;
        prev_time = now;

        let line = match client.status() {
            Ok(s) => {
                let sum = &s.summary;
                let total_chunks = sum.pending + sum.running + sum.finished;
                let done_pct = if total_chunks > 0 {
                    sum.finished as f64 / total_chunks as f64 * 100.0
                } else {
                    100.0
                };
                let mine = s
                    .workers
                    .iter()
                    .find(|w| w.worker_id == worker_id)
                    .map(|w| {
                        if w.chunks.is_empty() {
                            format!("{} chunks", w.chunk_count)
                        } else {
                            let avg: f64 =
                                w.chunks.iter().map(|c| c.progress_pct).sum::<f64>()
                                    / w.chunks.len() as f64;
                            format!("{} chunks avg {avg:.1}%", w.chunk_count)
                        }
                    })
                    .unwrap_or_else(|| "no lease".to_string());
                format!(
                    "[remote] pend={} running={} done={} ({:.1}%) {}  keys={} rate={}/s",
                    sum.pending,
                    sum.running,
                    sum.finished,
                    done_pct,
                    mine,
                    fmt_comma(total),
                    fmt_comma(rate.round() as u64),
                )
            }
            Err(e) => format!("[remote] hub status unavailable ({e})"),
        };
        term_status(&format!("  {line}"));
    }
}

// ── one worker thread ────────────────────────────────────────────────────────

/// Claim → scan → report loop.  Each thread holds up to one chunk at a time
/// (claim count = 1), so `n_workers` threads keep `n_workers` chunks in flight
/// on the hub.  All threads share the same `worker_id` — the hub leases chunks
/// by chunk id, not by worker, so this is fine.
#[allow(clippy::too_many_arguments)]
fn remote_worker(
    wid: u32,
    client: &HubClient,
    worker_id: &str,
    target_h160: [u8; 20],
    puzzle_number: Option<u32>,
    progress: &Progress,
    matches: &Mutex<Vec<MatchEvent>>,
    stop_flag: &AtomicBool,
    hit_flag: &AtomicBool,
    start: Instant,
) {
    // Throttle for claim-failure logging so a hub outage prints once / 30s,
    // not once / 2s.
    let mut last_fail_log = Instant::now();

    loop {
        if stop_flag.load(Ordering::Relaxed) {
            return;
        }

        // ── claim one chunk ───────────────────────────────────────────────
        let claimed = match client.claim(worker_id, 1) {
            Ok(c) => c,
            Err(e) => {
                // Transport error (hub down / slow).  Back off and retry; if
                // we held a lease, the hub reclaims it after 120s on its own.
                if last_fail_log.elapsed() >= Duration::from_secs(30) {
                    term_line(&format!("[remote] claim failed ({e}) — retrying …"));
                    last_fail_log = Instant::now();
                }
                std::thread::sleep(CLAIM_IDLE);
                continue;
            }
        };

        if claimed.granted == 0 || claimed.chunks.is_empty() {
            // Nothing pending: every chunk is running elsewhere, or the puzzle
            // is complete.  Check the hub before spinning.
            if stop_flag.load(Ordering::Relaxed) {
                return;
            }
            match client.status() {
                Ok(s) if s.summary.pending + s.summary.running == 0 => return, // all done
                _ => {}
            }
            std::thread::sleep(CLAIM_IDLE);
            continue;
        }

        let chunk = &claimed.chunks[0];
        let chunk_id = chunk.id;
        let start_bytes = parse_hex_key(&chunk.current_hex);
        let end_bytes = parse_hex_key(&chunk.end_hex);

        // 随机扫描方向（每次 claim 掷一次硬币），与本地模式一致：两个方向覆盖
        // 相同的 key 集合 [start, end)，只改变遍历顺序。
        let reverse = puzzle::pick_random(&[true, false]).copied().unwrap_or(false);
        let dir = if reverse {
            ScanDir::Reverse
        } else {
            ScanDir::Forward
        };
        term_line(&format!(
            "[claim] w={wid} chunk={chunk_id} range={}..{} dir={}",
            abbr_hex(&start_bytes),
            abbr_hex(&end_bytes),
            if reverse { "REV" } else { "FWD" },
        ));

        // ── scan with the shared core ─────────────────────────────────────
        // The on_position closure sends a heartbeat every HEARTBEAT_INTERVAL
        // (30s ≪ the hub's 120s reclaim).  Forward sends `current`, reverse
        // sends `end` — the two fields carry the full resume position.  If the
        // hub ever reports the lease lost (404/409), we stop reporting, scan
        // to the end of the claim, and re-claim; the hub already reverted the
        // chunk to pending at the last successful heartbeat.
        let mut lease_lost = false;
        let mut last_hb = Instant::now();
        // Worker-wide cumulative keys at the last heartbeat — the delta over
        // the heartbeat window is the rate broadcast to the hub (same source
        // as the status line's `rate=/s`).
        let mut last_keys = progress.checked.load(Ordering::Relaxed);
        // Set by the heartbeat closure when the hub reports the lease lost
        // (404/409); scan_chunk checks it on its next 2048-cadence stop and
        // abandons the rest of this claim instead of scanning to the rotation
        // budget — the hub already reverted the chunk to pending.
        let abort_flag = AtomicBool::new(false);
        let outcome = scan_chunk(ScanChunkOptions {
            target_h160,
            puzzle_number,
            worker_id: wid,
            chunk_id: Some(chunk_id),
            start: start_bytes,
            end: end_bytes,
            dir,
            rotate_keys: Some(ROTATION_BUDGET),
            progress,
            matches,
            stop_flag,
            hit_flag,
            abort_flag: Some(&abort_flag),
            start_elapsed: start,
            on_position: &mut |pos: &ResumePosition| {
                if lease_lost || last_hb.elapsed() < HEARTBEAT_INTERVAL {
                    return;
                }
                let (cur, end) = match dir {
                    ScanDir::Forward => (Some(hex_encode_key(&pos.current)), None),
                    ScanDir::Reverse => (None, Some(hex_encode_key(&pos.end))),
                };
                // Broadcast the worker-wide cumulative keys and the rate over
                // this heartbeat window; the hub keeps them in memory only.
                let keys = progress.checked.load(Ordering::Relaxed);
                let dt = last_hb.elapsed().as_secs_f64();
                let rate = if dt > 0.1 {
                    (keys - last_keys) as f64 / dt
                } else {
                    0.0
                };
                match client.heartbeat(chunk_id, worker_id, cur, end, Some(keys), Some(rate)) {
                    Ok(()) => {
                        last_hb = Instant::now();
                        last_keys = keys;
                    }
                    Err(e) => {
                        if is_lease_lost(&e) {
                            lease_lost = true;
                            abort_flag.store(true, Ordering::Relaxed);
                        } else {
                            // Transport error: re-arm the throttle so we retry
                            // in HEARTBEAT_INTERVAL instead of every 2048 keys.
                            last_hb = Instant::now();
                        }
                    }
                }
            },
        });

        // ── report the outcome ───────────────────────────────────────────
        if lease_lost {
            continue; // hub already parked our chunk; nothing to finalize
        }

        if outcome.matched {
            // Win: mark the chunk finished on the hub, then stop (scan_chunk
            // already set stop_flag + hit_flag for every worker).
            if let Err(e) = client.done(chunk_id, worker_id) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[remote] done failed ({e}) — hub will reclaim the chunk"
                    ));
                }
            }
            return;
        }

        if outcome.invalid_start {
            // Start key outside [1, n-1]: park it back at its original start so
            // a future claim can re-pick it (mirrors the local finalize).
            if let Err(e) = client.release(
                chunk_id,
                worker_id,
                Some(hex_encode_key(&start_bytes)),
                None,
            ) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[remote] release failed ({e}) — hub will reclaim the chunk"
                    ));
                }
            }
        } else if outcome.done {
            // Whole range scanned — finished.
            if let Err(e) = client.done(chunk_id, worker_id) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[remote] done failed ({e}) — hub will reclaim the chunk"
                    ));
                }
            }
        } else {
            // Parked (rotation budget hit or SIGINT): send the resume position.
            // Forward → current = next key to scan.  Reverse → end = sk + 1
            // (current stays at start).  Identical semantics to local mode.
            let (cur, end) = match dir {
                ScanDir::Forward => (Some(hex_encode_key(&outcome.sk)), None),
                ScanDir::Reverse => (
                    None,
                    Some(hex_encode_key(&crate::gpu::convert::scalar_add_be(
                        &outcome.sk,
                        1,
                    ))),
                ),
            };
            if let Err(e) = client.release(chunk_id, worker_id, cur, end) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[remote] release failed ({e}) — hub will reclaim the chunk"
                    ));
                }
            }
        }

        if stop_flag.load(Ordering::Relaxed) {
            return;
        }
    }
}
