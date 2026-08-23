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
//!   4. reports back via `win` (match) / `done` (finished) or `release`
//!      (parked, with the resume position — forward `current`, reverse `end`).
//!
//! Solved signaling: when a worker finds the key it posts `/api/win`; the hub
//! persists the win record and marks the puzzle solved.  That state is
//! broadcast on claim / heartbeat / status responses, and every worker stops
//! promptly when it sees it (mid-claim workers abort on the next heartbeat;
//! idle workers exit on the next claim).
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
    term_line, term_status, PuzzleScannerBackend, ResumePosition, ScanChunkOptions, ScanDir,
};
use crate::workers::{fmt_comma, MatchEvent};

/// Per-claim rotation budget: park the chunk and re-claim after this many keys,
/// exactly matching local puzzle mode (`puzzle::ROTATION_BUDGET`).
const ROTATION_BUDGET: u64 = puzzle::ROTATION_BUDGET;

/// Per-claim rotation budget for the GPU remote worker.  Mirrors the local
/// puzzle mode's GPU cadence (`gpu_rotate_keys = 2^31` in `main.rs`).  At
/// ~100 Mkeys/s a claim lasts ~20s, so the release-at-rotation also keeps the
/// hub's resume position fresher than the 30s heartbeat alone.
const GPU_ROTATION_BUDGET: u64 = 1u64 << 31;

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
    /// puzzle 已被某个 worker 命中（hub 落盘 win 记录）。旧 hub 不返回 → false。
    #[serde(default)]
    solved: bool,
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
    /// hub 已 solved：别的 worker 已命中 → 本 worker 应停止。旧 hub 不返回 → false。
    #[serde(default)]
    solved: bool,
}

/// `POST /api/chunks/{id}/heartbeat` 的响应体：`{ok, solved?}`。solved 用于让正在
/// 扫 chunk 的 worker 尽快停止；旧 hub 不返回 solved → 默认 false。
#[derive(Debug, Default, Deserialize)]
struct HeartbeatResp {
    #[serde(default)]
    solved: bool,
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

    /// Heartbeat。Ok(true) 表示 hub 已 solved（别的 worker 命中）——调用方应尽快
    /// 停止本 claim。响应体解析失败按未 solved 处理（旧 hub 只回 `{ok:true}`）。
    fn heartbeat(
        &self,
        chunk_id: u32,
        worker_id: &str,
        current_hex: Option<String>,
        end_hex: Option<String>,
        keys: Option<u64>,
        rate: Option<f64>,
    ) -> Result<bool, ureq::Error> {
        let body = ChunkUpdateBody {
            worker_id: worker_id.to_string(),
            current_hex,
            end_hex,
            keys,
            rate,
        };
        let resp = self
            .agent
            .post(&self.url(&format!("/api/chunks/{chunk_id}/heartbeat")))
            .send_json(body)?;
        let parsed: HeartbeatResp = resp.into_body().read_json().unwrap_or_default();
        Ok(parsed.solved)
    }

    fn done(&self, chunk_id: u32, worker_id: &str) -> Result<(), ureq::Error> {
        let _ = self
            .agent
            .post(&self.url(&format!("/api/chunks/{chunk_id}/done")))
            .send_json(json!({ "worker_id": worker_id }))?;
        Ok(())
    }

    /// 命中上报（取代 done）：hub 落 win 记录 + 置 puzzle solved。409（已被别的
    /// worker 先标记 solved）按 lease 丢失处理即可——本 worker 放弃本 chunk。
    fn win(&self, chunk_id: u32, worker_id: &str) -> Result<(), ureq::Error> {
        let _ = self
            .agent
            .post(&self.url("/api/win"))
            .send_json(json!({ "worker_id": worker_id, "chunk_id": chunk_id }))?;
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
    framework: crate::framework::GpuFramework,
    gpu_available: bool,
) -> (Arc<Progress>, Vec<MatchEvent>) {
    let client = Arc::new(HubClient::new(remote_url));

    // ── 1. connect to the hub and read the puzzle meta ─────────────────────
    let (target_h160, puzzle_number, summary) = connect(&client);
    if summary.pending + summary.running == 0 {
        println!("[remote] hub reports no pending or running chunks — nothing to do.");
        return (Arc::new(Progress::new(0)), Vec::new());
    }

    // ── 2. shared state + SIGINT handler (mirrors puzzle.rs) ───────────────
    // GPU worker count for the heartbeat's alive counter: WebGPU = 1 worker,
    // CUDA = one thread per physical device (mainboard + discrete cards).
    let gpu_worker_count = if gpu_available {
        match framework {
            crate::framework::GpuFramework::WebGpu => 1,
            crate::framework::GpuFramework::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::CudaScanner::device_count() as usize
                }
                #[cfg(not(feature = "cuda"))]
                {
                    0
                }
            }
            crate::framework::GpuFramework::Auto => 0, // resolved before remote::run
        }
    } else {
        0
    };
    let progress = Arc::new(Progress::new((n_workers + gpu_worker_count) as u64));
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

    // ── 3b. GPU worker thread(s) ─────────────────────────────────────────────
    // One worker per physical CUDA device (or a single WebGPU worker) that
    // claims pending chunks from the hub and scans them with the GPU (100k
    // strided walkers, dense zero-overlap tiling, heartbeats + release-at-
    // rotation for resume).  Runs alongside the CPU workers, all pulling from
    // the same hub claim pool.  Skipped when main() resolved no usable GPU
    // device (framework stays as resolved, never Auto).
    let mut gpu_handles: Vec<_> = Vec::new();
    if gpu_available {
        match framework {
            crate::framework::GpuFramework::WebGpu => {
                let client = client.clone();
                let worker_id = worker_id.clone();
                let progress = progress.clone();
                let matches = matches.clone();
                let stop_flag = stop_flag.clone();
                let hit_flag = hit_flag.clone();
                gpu_handles.push(std::thread::spawn(move || {
                    remote_gpu_worker_entry(
                        &client,
                        &worker_id,
                        target_h160,
                        puzzle_number,
                        &progress,
                        &matches,
                        &stop_flag,
                        &hit_flag,
                        start,
                        framework,
                        0, // WebGPU: single device
                    )
                }));
            }
            crate::framework::GpuFramework::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    let n_devices = crate::cuda::CudaScanner::device_count() as u32;
                    for dev_idx in 0..n_devices {
                        // One worker thread per device; each clones its own Arc.
                        let client = client.clone();
                        let worker_id = worker_id.clone();
                        let progress = progress.clone();
                        let matches = matches.clone();
                        let stop_flag = stop_flag.clone();
                        let hit_flag = hit_flag.clone();
                        gpu_handles.push(std::thread::spawn(move || {
                            remote_gpu_worker_entry(
                                &client,
                                &worker_id,
                                target_h160,
                                puzzle_number,
                                &progress,
                                &matches,
                                &stop_flag,
                                &hit_flag,
                                start,
                                framework,
                                dev_idx,
                            )
                        }));
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    term_line("[remote] CUDA feature not compiled — running CPU-only.");
                }
            }
            crate::framework::GpuFramework::Auto => {
                // Resolved to a concrete backend in main() before we get here.
                unreachable!("framework Auto must be resolved before remote::run")
            }
        }
    }

    for h in handles {
        drop(h.join());
    }
    // The GPU workers finish their current claim on their own (or stop promptly
    // once the hub runs out of chunks / a hit fires), so join them before the
    // stop_flag below — otherwise the flag would cut them off mid-chunk and leak
    // the lease.
    for h in gpu_handles {
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
    if meta.solved {
        term_line(&format!(
            "[remote] hub 已 solved（有 worker 命中 puzzle #{}）— 无需工作。",
            meta.puzzle_number,
        ));
        std::process::exit(0);
    }
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

        if claimed.solved {
            // 别的 worker 已命中：hub 不再发任务，本 worker 停止。
            term_line("[remote] hub 已 solved（其他 worker 找到私钥）— 退出。");
            return;
        }

        if claimed.granted == 0 || claimed.chunks.is_empty() {
            // Nothing pending: every chunk is running elsewhere, or the puzzle
            // is complete.  Check the hub before spinning.
            if stop_flag.load(Ordering::Relaxed) {
                return;
            }
            match client.status() {
                Ok(s) if s.meta.solved || s.summary.pending + s.summary.running == 0 => return, // solved / all done
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
        // Set by the heartbeat closure when the hub reports solved (someone
        // else found the key); we then abort the scan and skip the release.
        let mut solved_flag = false;
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
                    Ok(solved) => {
                        last_hb = Instant::now();
                        last_keys = keys;
                        if solved {
                            // 别的 worker 已命中：尽快放弃本 claim（下轮 claim 退出）。
                            solved_flag = true;
                            abort_flag.store(true, Ordering::Relaxed);
                        }
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
        if solved_flag {
            // 别的 worker 已命中：hub 已 solved，无需 release/park——下轮 claim
            // 读到 solved 即退出。
            continue;
        }

        if outcome.matched {
            // Win: report the hit to the hub (`/api/win` — hub 落 win 记录 + 置
            // solved), then stop (scan_chunk already set stop_flag + hit_flag).
            if let Err(e) = client.win(chunk_id, worker_id) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[remote] win failed ({e}) — hub will reclaim the chunk"
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

// ── GPU remote worker ────────────────────────────────────────────────────────

/// Entry point for the remote GPU worker: resolve the backend, set up the
/// scanner, and hand off to the shared `remote_gpu_worker` loop.  Returns
/// (CPU-only) when the selected backend has no usable device — the run simply
/// continues with the CPU workers, exactly like local puzzle mode's GPU worker.
#[allow(clippy::too_many_arguments)]
fn remote_gpu_worker_entry(
    client: &HubClient,
    worker_id: &str,
    target_h160: [u8; 20],
    puzzle_number: Option<u32>,
    progress: &Progress,
    matches: &Mutex<Vec<MatchEvent>>,
    stop_flag: &AtomicBool,
    hit_flag: &AtomicBool,
    start: Instant,
    framework: crate::framework::GpuFramework,
    device_index: u32,
) {
    // Only the CUDA arm consumes `device_index` (WebGPU always uses device 0);
    // silence the unused-param warning in a non-CUDA build.
    let _ = device_index;
    match framework {
        crate::framework::GpuFramework::WebGpu => {
            let gpu_ctx = match crate::gpu::GpuContext::new_blocking(0) {
                Ok(c) => c,
                Err(e) => {
                    term_line(&format!(
                        "[remote] GPU unavailable ({e}) — running CPU-only."
                    ));
                    return;
                }
            };
            term_line(&format!(
                "[remote] GPU worker up on {}",
                gpu_ctx.device_name()
            ));
            let candidates = crate::gpu::convert::hash160_to_candidates(&target_h160);
            let mut scanner = match crate::gpu::GpuScanner::new(gpu_ctx, &candidates) {
                Ok(s) => s,
                Err(e) => {
                    term_line(&format!(
                        "[remote] GpuScanner::new failed ({e}) — running CPU-only."
                    ));
                    return;
                }
            };
            // Dense-tiling config: stride = N threads, single target candidate.
            scanner.stride = crate::gpu::NUM_GPU_THREADS;
            scanner.num_candidates = 1;
            remote_gpu_worker(
                client,
                worker_id,
                target_h160,
                puzzle_number,
                scanner,
                progress,
                matches,
                stop_flag,
                hit_flag,
                start,
                "GPU",
            );
        }
        crate::framework::GpuFramework::Cuda => {
            #[cfg(feature = "cuda")]
            {
                let candidates = crate::gpu::convert::hash160_to_candidates(&target_h160);
                let mut scanner =
                    match crate::cuda::CudaScanner::new_on_device(&candidates, device_index) {
                        Ok(s) => s,
                        Err(e) => {
                            term_line(&format!(
                                "[remote] CUDA device #{device_index} init failed ({e}) — running CPU-only on this card."
                            ));
                            return;
                        }
                    };
                term_line(&format!(
                    "[remote] CUDA worker #{} up on {}",
                    scanner.device_index(),
                    scanner.device_name()
                ));
                scanner.stride = crate::gpu::NUM_GPU_THREADS;
                scanner.num_candidates = 1;
                // Per-card label so [claim] / [HIT] lines identify the GPU.
                // Compute before `scanner` is moved into the worker.
                let backend_label = format!("CUDA[{}]", scanner.device_index());
                remote_gpu_worker(
                    client,
                    worker_id,
                    target_h160,
                    puzzle_number,
                    scanner,
                    progress,
                    matches,
                    stop_flag,
                    hit_flag,
                    start,
                    &backend_label,
                );
            }
            #[cfg(not(feature = "cuda"))]
            {
                term_line("[remote] CUDA feature not compiled — running CPU-only.");
            }
        }
        crate::framework::GpuFramework::Auto => {
            // Resolved to a concrete backend in main() before we get here.
            unreachable!("framework Auto must be resolved before remote::run")
        }
    }
}

/// Claim → GPU dense-tile scan → report loop for one remote GPU worker thread.
/// Mirrors the local puzzle GPU loop (`puzzle_gpu_scan_loop`) but pulls chunks
/// from the hub over HTTP and persists progress via throttled heartbeats +
/// release-at-rotation instead of a local SQLite worklist.
///
/// The GPU scans *forward only* (dense tiling seeds walker `i` at `start + i`
/// with stride N) — unlike the CPU workers' per-claim direction coin-flip.  That
/// is fine: the hub hands out exclusive chunks, so any traversal order covers
/// the same key set with no overlap; only the in-order walk differs.
///
/// Lease handling mirrors the CPU worker: a 404/409 heartbeat/release means the
/// hub already reverted the chunk to pending at the last reported position, so
/// we abandon the claim and re-claim rather than fight over it.
#[allow(clippy::too_many_arguments)]
fn remote_gpu_worker<S: PuzzleScannerBackend>(
    client: &HubClient,
    worker_id: &str,
    target_h160: [u8; 20],
    puzzle_number: Option<u32>,
    mut scanner: S,
    progress: &Progress,
    matches: &Mutex<Vec<MatchEvent>>,
    stop_flag: &AtomicBool,
    hit_flag: &AtomicBool,
    start: Instant,
    backend_label: &str,
) {
    // Per-dispatch coverage (keys) once the scanner is configured.  Constant
    // once calibrated: `steps_per_call` defaults to 1 at construction.
    let dispatch_keys = crate::gpu::NUM_GPU_THREADS as u64
        * *scanner.steps_per_call() as u64;

    // Throttle for claim-failure logging so a hub outage prints once / 30s.
    let mut last_fail_log = Instant::now();

    loop {
        if stop_flag.load(Ordering::Relaxed) {
            return;
        }

        // ── claim one chunk ───────────────────────────────────────────────
        let claimed = match client.claim(worker_id, 1) {
            Ok(c) => c,
            Err(e) => {
                if last_fail_log.elapsed() >= Duration::from_secs(30) {
                    term_line(&format!("[remote] claim failed ({e}) — retrying …"));
                    last_fail_log = Instant::now();
                }
                std::thread::sleep(CLAIM_IDLE);
                continue;
            }
        };
        if claimed.solved {
            // 别的 worker 已命中：hub 不再发任务，本 worker 停止。
            term_line("[remote] hub 已 solved（其他 worker 找到私钥）— 退出。");
            return;
        }
        if claimed.granted == 0 || claimed.chunks.is_empty() {
            // Nothing pending: every chunk is running elsewhere, or the puzzle
            // is complete.  Check the hub before spinning.
            if stop_flag.load(Ordering::Relaxed) {
                return;
            }
            match client.status() {
                Ok(s) if s.meta.solved || s.summary.pending + s.summary.running == 0 => return, // solved / all done
                _ => {}
            }
            std::thread::sleep(CLAIM_IDLE);
            continue;
        }

        let chunk = &claimed.chunks[0];
        let chunk_id = chunk.id;
        let start_bytes = parse_hex_key(&chunk.current_hex);
        let end_bytes = parse_hex_key(&chunk.end_hex);

        term_line(&format!(
            "[claim] w={backend_label} chunk={chunk_id} range={}..{}",
            abbr_hex(&start_bytes),
            abbr_hex(&end_bytes),
        ));

        // ── seed the 100k strided walkers at start + i ─────────────────────
        if scanner.seed_range(start_bytes).is_err() {
            term_line(&format!(
                "[remote] {backend_label} seed_range failed — releasing chunk"
            ));
            // Park back at the original start so a future claim re-picks it.
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
            continue;
        }

        let mut current = start_bytes; // next key NOT yet covered
        let mut scanned_keys: u64 = 0; // keys covered this claim (rotation)
        let mut hit = false; // CPU-verified match → this chunk is the winner
        let mut lease_lost = false;
        // 心跳报告 hub 已 solved（别的 worker 命中）→ 提前退出，不 park。
        let mut solved = false;
        let mut last_hb = Instant::now();
        // Worker-wide cumulative keys at the last heartbeat — the delta over
        // the heartbeat window is the rate broadcast to the hub.
        let mut last_keys = progress.checked.load(Ordering::Relaxed);

        // ── scan the chunk in N·steps_per_call-key dispatches ──────────────
        loop {
            // Decide this dispatch's step count.  A full dispatch covers
            // `dispatch_keys` keys; the final (partial) dispatch is trimmed so
            // the walkers land on or just past `end`.  The chunk width can
            // exceed 2^64, so never compute `end - current` directly — compare
            // `end` against `current + dispatch_keys` (the small add never
            // overflows) and only subtract once the remainder is known to fit.
            let steps = if crate::gpu::convert::be_lt(&current, &end_bytes) {
                let reach = crate::gpu::convert::scalar_add_be(&current, dispatch_keys);
                // `reach >= end`  ⟺  `end - current <= dispatch_keys` (no overflow).
                if !crate::gpu::convert::be_lt(&reach, &end_bytes) {
                    let remaining = crate::gpu::convert::scalar_sub_be(&end_bytes, &current);
                    let n = crate::gpu::NUM_GPU_THREADS as u64;
                    std::cmp::max(1, (remaining + n - 1) / n) as u32
                } else {
                    *scanner.steps_per_call()
                }
            } else {
                0
            };
            if steps == 0 {
                break; // reached the exclusive end
            }
            // 命中即停：别的 worker 命中了，在 dispatch 间隙尽快退出（不再多跑一趟）。
            if stop_flag.load(Ordering::Relaxed) {
                break;
            }

            // ── heartbeat (throttled to HEARTBEAT_INTERVAL) ────────────────
            // Broadcast the scan position so the hub can resume the chunk if
            // this worker dies, plus the worker-wide keys + rate.  Forward
            // scanning always reports `current` (the next key to scan).
            if last_hb.elapsed() >= HEARTBEAT_INTERVAL {
                let keys = progress.checked.load(Ordering::Relaxed);
                let dt = last_hb.elapsed().as_secs_f64();
                let rate = if dt > 0.1 {
                    (keys - last_keys) as f64 / dt
                } else {
                    0.0
                };
                match client.heartbeat(
                    chunk_id,
                    worker_id,
                    Some(hex_encode_key(&current)),
                    None,
                    Some(keys),
                    Some(rate),
                ) {
                    Ok(hub_solved) => {
                        last_hb = Instant::now();
                        last_keys = keys;
                        if hub_solved {
                            // 别的 worker 已命中：尽快退出（下轮 claim 停止）。
                            solved = true;
                            break;
                        }
                    }
                    Err(e) => {
                        if is_lease_lost(&e) {
                            // Hub reverted the chunk — abandon this claim.
                            lease_lost = true;
                            break;
                        }
                        // Transport error: re-arm the throttle so we retry in
                        // HEARTBEAT_INTERVAL instead of every dispatch.
                        last_hb = Instant::now();
                    }
                }
            }

            // ── one dispatch ──────────────────────────────────────────────
            // Temporarily set steps_per_call for this (possibly final, partial)
            // dispatch, restoring the default afterwards.
            let saved_steps = *scanner.steps_per_call();
            *scanner.steps_per_call() = steps;
            let batch = crate::gpu::NUM_GPU_THREADS as u64 * steps as u64;
            match scanner.step() {
                Ok(batch_matches) => {
                    if !batch_matches.is_empty() {
                        // Collect CPU-verified matches inside the lock, then
                        // drop the lock before printing ([HIT] is never emitted
                        // under it).
                        let verified: Vec<MatchEvent> = {
                            let mut g = matches.lock().unwrap_or_else(|e| e.into_inner());
                            let mut out = Vec::new();
                            for m in &batch_matches {
                                let mut ev = crate::puzzle::gpu_match_to_event(
                                    m,
                                    chunk_id,
                                    start_bytes,
                                    puzzle_number,
                                );
                                // CPU verification — never trust the GPU
                                // candidate flag alone; spurious matches are
                                // dropped silently.  The shader checks both
                                // serialisations (c-or-u, like the CPU worker),
                                // so re-verify both here too.
                                let h = btc::hash160(&ev.compressed);
                                let h_u = btc::hash160(&ev.uncompressed);
                                if h == target_h160 || h_u == target_h160 {
                                    ev.elapsed = start.elapsed().as_secs_f64();
                                    g.push(ev.clone());
                                    out.push(ev);
                                }
                            }
                            out
                        };
                        if let Some(first) = verified.first() {
                            hit = true;
                            // 命中即停：首个命中的 worker 立即打印（含私钥）并通知
                            // 所有 worker 停止。
                            if hit_flag
                                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                                .is_ok()
                            {
                                term_line(&format!(
                                    "[HIT] 🎯 puzzle=#{} worker={backend_label} chunk={} sk_hex={}",
                                    puzzle_number.map_or_else(String::new, |n| n.to_string()),
                                    chunk_id,
                                    hex::encode(first.private_key),
                                ));
                            }
                            stop_flag.store(true, Ordering::SeqCst);
                        }
                    }
                    progress.increment(batch);
                }
                Err(e) => {
                    term_line(&format!(
                        "[remote] {backend_label} step failed ({e}) — releasing chunk"
                    ));
                    break;
                }
            }
            *scanner.steps_per_call() = saved_steps;

            // Advance checkpoint by exactly the keys this dispatch covered.
            current = crate::gpu::convert::scalar_add_be(&current, batch);
            scanned_keys += batch;

            if stop_flag.load(Ordering::Relaxed) {
                break;
            }

            // ── rotation ──────────────────────────────────────────────────
            // Park after GPU_ROTATION_BUDGET keys scanned *this claim* and let
            // the loop claim a fresh pending chunk.  `scanned_keys` resets
            // every claim, so this is a per-claim budget exactly like the CPU
            // worker's.  Releasing also keeps the hub's resume position fresher
            // than the 30s heartbeat would on its own.
            if scanned_keys >= GPU_ROTATION_BUDGET {
                break;
            }
        }

        // ── report the outcome ────────────────────────────────────────────
        if lease_lost {
            continue; // hub already parked our chunk; nothing to finalize
        }
        if solved {
            // 别的 worker 已命中：hub 已 solved，无需 release/park——下轮 claim
            // 读到 solved 即退出。
            continue;
        }

        if hit {
            // Win: report the hit to the hub (`/api/win` — hub 落 win 记录 + 置
            // solved), then stop (the scan already set stop_flag + hit_flag).
            if let Err(e) = client.win(chunk_id, worker_id) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[remote] win failed ({e}) — hub will reclaim the chunk"
                    ));
                }
            }
            return;
        }

        // Fully scanned (current >= end) is `done` regardless of why the loop
        // stopped (stop flag / GPU error / rotation): parking an empty
        // `[end, end)` chunk back at the hub would leave it pending forever
        // instead of finishing it.  A chunk that ends exactly on the rotation
        // budget boundary lands here via this path.
        let done = !crate::gpu::convert::be_lt(&current, &end_bytes);
        if done {
            // Whole range scanned — finished.
            if let Err(e) = client.done(chunk_id, worker_id) {
                if !is_lease_lost(&e) {
                    term_line(&format!(
                        "[remote] done failed ({e}) — hub will reclaim the chunk"
                    ));
                }
            }
        } else {
            // Parked (rotation budget, SIGINT, or GPU error): send the resume
            // position.  Forward scanning → `current` = next key to scan.
            if let Err(e) = client.release(
                chunk_id,
                worker_id,
                Some(hex_encode_key(&current)),
                None,
            ) {
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
