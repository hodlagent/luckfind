//! luckfind — Bitcoin dormant address lottery (Rust / libsecp256k1 reimplementation).
//!
//! On Apple M4 this sustains ~1.4M keys/sec in a release build with 5 worker
//! threads — the bottleneck is the secp256k1 point-add (~2.4µs/key, 99% of
//! pipeline wall-clock), NOT the hash160 (~25ns/key).  A GPU port must
//! accelerate the EC op to move the needle; GPU hashing alone would not.

mod puzzles;
mod args;
mod config;
mod btc;
mod progress;
mod report;
mod workers;
mod puzzle;
mod remote;
mod gpu;
mod framework;
#[cfg(feature = "cuda")]
mod cuda;
// heartbeat logic lives in workers.rs::run (a separate ticker thread).

use std::path::Path;
use std::time::{Duration, Instant};

use sha2::Digest;

use args::Cli;

fn main() {
    let cli = Cli::parse();

    // ── config file ─────────────────────────────────────────────────────────
    // Explicit `--config <path>` wins; otherwise a file is auto-discovered in
    // the current working directory (`luckfind.toml`, then `config.toml`).
    // Precedence: CLI flags > config file > built-in defaults.
    let mut cfg = crate::config::Config::default();
    if let Some(ref cfg_path) = cli.config {
        match crate::config::Config::load(cfg_path) {
            Ok(c) => {
                eprintln!("  [config] loaded {}", cfg_path.display());
                cfg = c;
            }
            Err(e) => {
                eprintln!("[config] {e}");
                std::process::exit(2);
            }
        }
    } else {
        match crate::config::Config::discover() {
            crate::config::Discovery::Loaded(path, c) => {
                eprintln!("  [config] loaded {}", path.display());
                cfg = c;
            }
            crate::config::Discovery::Failed(path, e) => {
                eprintln!("[config] {}: {e}", path.display());
                std::process::exit(2);
            }
            crate::config::Discovery::None => {}
        }
    }

    // Resolve the scan mode.  An explicit `--puzzle` / `--remote` flag always
    // wins over the config file; otherwise the file's `mode` selects the mode
    // (absent → random, the default).
    let mode = match cfg.mode.as_deref() {
        Some(s) => match crate::config::Mode::parse(s) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[config] {e}");
                std::process::exit(2);
            }
        },
        None => crate::config::Mode::Random,
    };

    // Puzzle worklist path and remote hub URL, each resolved CLI-over-config.
    // The required config fields are validated only when the file actually
    // selects that mode.
    let puzzle_path: Option<String> = if cli.puzzle.is_some() {
        cli.puzzle.clone()
    } else if cli.remote.is_none() && mode == crate::config::Mode::Puzzle {
        match &cfg.puzzle.database {
            Some(db) => Some(db.clone()),
            None => {
                eprintln!("[config] mode = \"puzzle\" requires `[puzzle] database = \"<path>\"`");
                std::process::exit(2);
            }
        }
    } else {
        None
    };

    let remote_url: Option<String> = if cli.remote.is_some() {
        cli.remote.clone()
    } else if cli.puzzle.is_none() && mode == crate::config::Mode::Remote {
        match &cfg.remote.uri {
            Some(uri) => Some(uri.clone()),
            None => {
                eprintln!("[config] mode = \"remote\" requires `[remote] uri = \"<url>\"`");
                std::process::exit(2);
            }
        }
    } else {
        None
    };

    // Rotation (reclaim) budgets for puzzle mode, resolved CLI-over-config.
    // A value of 0 disables rotation (`None` = scan chunks to completion).
    let rotate_keys = resolve_rotate(cli.cpu_rotate_keys, cfg.cpu_rotate_keys, 1u64 << 27);
    let gpu_rotate_keys = resolve_rotate(cli.gpu_rotate_keys, cfg.gpu_rotate_keys, 1u64 << 31);

    // CPU worker availability + load, from the config `[cpu]` section.
    // `enabled` toggles CPU scanning entirely (default true); `load` is the
    // proportion (0.1..=1.0) of the resolved worker thread count that actually
    // runs — the base is `--workers`, defaulting to all logical CPUs, so the
    // default run uses `num_cpus × load` threads (1.0 = all cores).  The load
    // value is validated only while CPU is enabled — when disabled it is
    // unused.
    let cpu_enabled = cfg.cpu.enabled.unwrap_or(true);
    let cpu_load = if cpu_enabled {
        let l = cfg.cpu.load.unwrap_or(1.0);
        if !(0.1..=1.0).contains(&l) {
            eprintln!("[config] [cpu] load must be in 0.1..=1.0, got {l}");
            std::process::exit(2);
        }
        l
    } else {
        1.0
    };
    let cpu_workers = if cpu_enabled {
        ((cli.workers() as f64) * cpu_load).ceil().max(1.0) as usize
    } else {
        0
    };

    if cli.bench {
        bench(cli.workers(), cli.duration.unwrap_or(5.0));
        return;
    }

    if cli.profile {
        profile();
        return;
    }

    // ── GPU backend resolution (before the remote/puzzle/--bench branches:
    //    remote and puzzle mode both need the resolved framework too).  `auto`
    //    probes CUDA first — CUDA exists because WebGPU's NVIDIA support is
    //    limited — then WebGPU.  The config `[gpu] enabled` key gates the whole
    //    thing: when disabled we skip probing entirely and run CPU-only, so the
    //    GPU is used for collision only when `gpu.enabled = true` AND a device
    //    is actually available.
    let mut framework = cli.gpu_framework;
    let gpu_enabled = cfg.gpu.enabled.unwrap_or(true);
    let gpu_available = if !gpu_enabled {
        eprintln!("  [GPU] disabled in config (`[gpu] enabled = false`) — running CPU-only.");
        // Keep the "framework is never Auto downstream" invariant; no GPU
        // workers spawn because gpu_available = false.
        framework = crate::framework::GpuFramework::WebGpu;
        false
    } else {
        let avail = match framework {
            crate::framework::GpuFramework::WebGpu => probe_webgpu(),
            crate::framework::GpuFramework::Cuda => probe_cuda(),
            crate::framework::GpuFramework::Auto => {
                // Platform policy: Windows/Linux prefer CUDA (NVIDIA); macOS
                // goes straight to WebGPU (Metal) — CUDA does not apply there.
                #[cfg(any(target_os = "windows", target_os = "linux"))]
                {
                    eprintln!("  [GPU] auto: probing backends (CUDA preferred)…");
                    if probe_cuda() {
                        framework = crate::framework::GpuFramework::Cuda;
                        true
                    } else if probe_webgpu() {
                        framework = crate::framework::GpuFramework::WebGpu;
                        true
                    } else {
                        // No accelerator found; keep a displayable value, worker
                        // runs CPU-only (gpu_available = false below).
                        framework = crate::framework::GpuFramework::WebGpu;
                        false
                    }
                }
                #[cfg(target_os = "macos")]
                {
                    eprintln!("  [GPU] auto: macOS — selecting WebGPU (Metal)…");
                    if probe_webgpu() {
                        framework = crate::framework::GpuFramework::WebGpu;
                        true
                    } else {
                        framework = crate::framework::GpuFramework::WebGpu;
                        false
                    }
                }
                #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
                {
                    framework = crate::framework::GpuFramework::WebGpu;
                    probe_webgpu()
                }
            }
        };
        if !avail {
            eprintln!(
                "  [GPU] No {} device available — running CPU-only.",
                framework
            );
        }
        avail
    };

    // A run with neither backend is a configuration mistake — surface it rather
    // than silently scanning nothing.
    if !cpu_enabled {
        if gpu_available {
            eprintln!("  [CPU] disabled in config (`[cpu] enabled = false`) — running GPU-only.");
        } else {
            eprintln!("  [CPU] disabled in config (`[cpu] enabled = false`).");
        }
    }
    if cpu_workers == 0 && !gpu_available {
        eprintln!(
            "[config] no CPU workers and no GPU available — nothing to scan. \
             Set `[cpu] enabled = true` and/or ensure `[gpu] enabled = true` with a usable device."
        );
        std::process::exit(2);
    }

    // ── remote mode: claim chunks over HTTP from a LAN hub (lan-hub) ────────
    // `--puzzle` and `--remote` are mutually exclusive (clap conflicts_with).
    // The hub holds the SQLite worklist and is the single writer; workers claim
    // chunks over HTTP.  When a GPU device was resolved above, a dedicated GPU
    // worker thread claims + dense-tiles chunks alongside the CPU workers.
    if let Some(ref remote_url) = remote_url {
        let (stats, _matches) = remote::run(
            remote_url,
            cli.worker_id(),
            cpu_workers,
            cli.heartbeat,
            Some(Path::new(&cli.output_dir)),
            framework,
            gpu_available,
        );
        // progress 由 remote::run 打印；aman_<TS>.txt 已在 run() 内落盘。
        let _ = stats;
        return;
    }

    // ── puzzle mode: deterministic sub-range scan from a worklist file ────────
    // Supports `.db` (SQLite, the runtime format) and `.json` (one-time import;
    // a `.db` sibling is created on first run and used for all future saves).
    if let Some(ref puzzle_path) = puzzle_path {
        // 旋转预算（reclaim count）来自 CLI/配置文件解析后的 rotate_keys 与
        // gpu_rotate_keys（见上方解析），分别控制 CPU / GPU worker 每个 claim
        // 扫描多少 keys 后停放当前子区间并随机重新选择。
        let (stats, _matches) = puzzle::run(
            Path::new(puzzle_path),
            cpu_workers,
            cli.heartbeat,
            rotate_keys,
            gpu_rotate_keys,
            Some(Path::new(&cli.output_dir)),
            framework,
            gpu_available,
        );
        // progress 由 puzzle::run 打印；aman_<TS>.txt 已在 run() 内落盘（先于 sqlite）。
        let _ = stats;
        return;
    }

    // Default mode: lottery against the embedded 77-puzzle set
    // (range-constrained key generation in [2^70, 2^160)).
    let ps = puzzles::puzzle_set();
    eprintln!("  🧩 Loaded {} puzzles, key space [2^70, 2^160)", ps.len());
    let target = workers::ScanTarget::PuzzleSet(ps);

    let start = Instant::now();
    let limits = workers::RuntimeLimits {
        duration_secs: cli.duration.map(|m| m * 60.0),
        heartbeat_secs: cli.heartbeat,
    };

    let (stats, matches) = workers::run(cpu_workers, target, limits, gpu_available, framework);

    println!();
    report::final_report(&stats, &matches, &start);
    report::flush_match_files(&matches, Some(Path::new(&cli.output_dir)));
}

/// Probe CUDA availability (feature-gated: without the `cuda` feature the
/// backend cannot be used at all).  On success, logs the full device inventory
/// — a custom PC with a mainboard GPU plus several discrete cards shows each
/// one, and `workers::run` / `puzzle::run` / `remote::run` spawn one worker per
/// device.
fn probe_cuda() -> bool {
    #[cfg(feature = "cuda")]
    {
        // `probe()` prints a helpful message when the CUDA kernel was not
        // compiled at build time (nvcc/cl.exe unavailable).
        let ok = crate::cuda::CudaScanner::probe();
        if ok {
            let names = crate::cuda::CudaScanner::device_names();
            eprintln!(
                "  [GPU] {} CUDA device(s) detected -- enabling one GPU worker per device.",
                names.len()
            );
            for (i, name) in names.iter().enumerate() {
                eprintln!("  [GPU]   device #{i}: {name}");
            }
        }
        ok
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("  [GPU] CUDA support not compiled in -- rebuild with --features cuda.");
        false
    }
}

/// Probe WebGPU availability.
fn probe_webgpu() -> bool {
    let ok = crate::gpu::GpuContext::new_blocking(0).is_ok();
    if ok {
        eprintln!("  [GPU] WebGPU device detected -- enabling GPU lottery worker.");
    }
    ok
}

/// Resolve a puzzle-mode rotation (reclaim) budget with CLI-over-config
/// precedence.  A value of 0 disables rotation — returned as `None`, which
/// tells the worker to scan chunks to completion per claim.
fn resolve_rotate(cli: Option<u64>, cfg: Option<u64>, default: u64) -> Option<u64> {
    let v = cli.or(cfg).unwrap_or(default);
    if v == 0 {
        None
    } else {
        Some(v)
    }
}

#[inline(never)]
fn bench(n_workers: usize, secs: f64) {
    eprintln!("  ⚡ BENCHMARK — {:.0}s burn-in", secs);
    eprintln!("  Workers : {}", n_workers);
    let start = Instant::now();
    let counter = std::sync::atomic::AtomicU64::new(0);
    let counter = &counter;
    let deadline = start + Duration::from_secs_f64(secs);

    std::thread::scope(|s| {
        for _ in 0..n_workers {
            s.spawn(|| use_secp256k1(counter, deadline));
        }
    });

    let total = counter.load(std::sync::atomic::Ordering::Relaxed);
    let rate = total as f64 / secs;
    eprintln!("  Keys    : {}", workers::fmt_comma(total));
    eprintln!("  Rate    : {} keys/s", workers::fmt_comma(rate as u64));
}

/// Stage-by-stage throughput profile of the scan hot path.
///
/// Answers "is hash160 the bottleneck?" — the go/no-go for a GPU port.
/// Each stage burns ~5s in a tight loop and reports its own throughput,
/// so we can see the SHA256 / RIPEMD160 / hash160-combined / point-add
/// split and decide whether GPU hashing is worth the engineering cost.
#[inline(never)]
fn profile() {
    const SECS: f64 = 5.0;

    eprintln!("  🔬 PIPELINE PROFILE — {:.0}s per stage", SECS);
    eprintln!("  (Apple Silicon SHA256 asm backend: ON)\n");

    // Helper: each stage gets its own fresh relative deadline so the stages run
    // back-to-back without one stage's wall-clock debt starving the next.
    let burn = |f: &mut dyn FnMut()| -> f64 {
        let dl = Instant::now() + Duration::from_secs_f64(SECS);
        let mut n = 0u64;
        while Instant::now() < dl {
            f();
            n += 1;
        }
        n as f64 / SECS
    };

    // Representative inputs: a real compressed pubkey and a real SHA256 digest.
    let pk_bytes: [u8; 33] = [
        0x02, 0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE,
        0x87, 0x0B, 0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x82, 0xD9, 0x59, 0xF2, 0x81,
        0x5B, 0x16, 0xF8, 0x17, 0x98,
    ];
    let sha_bytes: [u8; 32] = [
        0x91, 0x42, 0xAD, 0x99, 0x79, 0x5C, 0x9E, 0x77, 0x7B, 0xB1, 0xC2, 0xA2, 0x73, 0x9E,
        0x9E, 0xE7, 0xE0, 0xC7, 0xC5, 0xE0, 0xC7, 0xC5, 0xE0, 0xC7, 0xC5, 0xE0, 0xC7, 0xC5,
        0xE0, 0xC7, 0xC5, 0xE0,
    ];

    // ── SHA256 (33-byte compressed pubkey → 32-byte digest) ─────────────────
    let sha_rate = burn(&mut || {
        let _ = sha2::Sha256::digest(pk_bytes);
    });

    // ── RIPEMD160 (32-byte digest → 20-byte hash) ───────────────────────────
    let ripemd_rate = burn(&mut || {
        let _ = ripemd::Ripemd160::digest(sha_bytes);
    });

    // ── hash160 combined (SHA256 → RIPEMD160, the actual per-key op) ─────────
    let combined_rate = burn(&mut || {
        let _ = crate::btc::hash160(&pk_bytes);
    });

    // ── point add (pk + G, the non-hash half of the hot path) ───────────────
    let pk = secp256k1::PublicKey::from_slice(&crate::btc::GENERATOR_COMPRESSED).unwrap();
    let g = crate::btc::generator_public_key();
    let pt_rate = burn(&mut || {
        let _ = pk.combine(&g).unwrap_or(pk);
    });

    eprintln!(
        "  {:<20} {:>12} hashes/s",
        "SHA256 (33B):",
        workers::fmt_comma(sha_rate as u64)
    );
    eprintln!(
        "  {:<20} {:>12} hashes/s",
        "RIPEMD160 (32B):",
        workers::fmt_comma(ripemd_rate as u64)
    );
    eprintln!(
        "  {:<20} {:>12} hashes/s",
        "hash160 combined:",
        workers::fmt_comma(combined_rate as u64)
    );
    eprintln!(
        "  {:<20} {:>12} adds/s",
        "point-add (pk+G):",
        workers::fmt_comma(pt_rate as u64)
    );
    // Per-key wall-clock split: each key costs 1× hash160 + 1× point-add.
    // Convert rates to seconds-per-op, then weight by each stage's share of the
    // total per-key cost.  This is the correct GPU-go/no-go signal.
    let combined_cost = 1.0 / combined_rate;
    let pt_cost = 1.0 / pt_rate;
    let hash_and_pt = combined_cost + pt_cost;
    let hash_share = combined_cost / hash_and_pt * 100.0;

    eprintln!();
    eprintln!("  hash160 = SHA256 + RIPEMD160; per-key cost = 1× hash160 + 1× point-add.");
    eprintln!(
        "  Time/key        : hash160 {:.0}ns + point-add {:.0}ns = ~{:.0}ns",
        combined_cost * 1e9,
        pt_cost * 1e9,
        hash_and_pt * 1e9
    );
    eprintln!(
        "  Pipeline split  : {:.0}% point-add / {:.0}% hashing",
        100.0 - hash_share,
        hash_share
    );
    if pt_cost > combined_cost {
        eprintln!("  ➤ point-add dominates — a GPU port must accelerate the EC op, not just hashing.");
    } else {
        eprintln!("  ➤ hashing dominates — GPU hashing would move the needle.");
    }
}

fn use_secp256k1(counter: &std::sync::atomic::AtomicU64, deadline: Instant) {
    let secp = secp256k1::Secp256k1::new();
    use rand::TryRng;
    let mut local = 0u64;

    // Mirror the real lottery worker's hot path EXACTLY:
    //   1. One random starting key.
    //   2. One full scalar mult `pk = sk * G` up front.
    //   3. Tight loop: hash160 check + point-add advance `pk = pk + G`.
    //
    // This is what makes the benchmark representative of actual scan rate.
    let mut buf = [0u8; 32];
    rand::rngs::SysRng
        .try_fill_bytes(&mut buf)
        .expect("OS entropy source always available");
    let mut sk = loop {
        if !(buf.iter().all(|b| *b == 0) || buf.iter().all(|b| *b == 0xff)) {
            if let Ok(k) = secp256k1::SecretKey::from_byte_array(buf) {
                break k;
            }
        }
        rand::rngs::SysRng
            .try_fill_bytes(&mut buf)
            .expect("OS entropy source always available");
    };

    let one = secp256k1::Scalar::from_be_bytes({
        let mut b = [0u8; 32];
        b[31] = 1;
        b
    })
    .expect("scalar 1 valid");

    let point_g = crate::btc::generator_public_key();
    let mut pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);

    while Instant::now() < deadline {
        let pk_c = pk.serialize();
        let _h = crate::btc::hash160(&pk_c);

        // advance both in lock-step: sk + 1 (mod n), pk + G (point add).
        // `.unwrap_or(pk)` guards the astronomically-unlikely sk == n-1 case
        // (pk would be the point at infinity) without branching in the hot
        // path — keeps the loop body branchless and tight.
        sk = sk.add_tweak(&one).unwrap_or(sk);
        pk = pk.combine(&point_g).unwrap_or(pk);
        local += 1;
    }
    counter.fetch_add(local, std::sync::atomic::Ordering::Relaxed);
}
