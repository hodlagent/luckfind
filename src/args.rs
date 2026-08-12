use clap::Parser;

pub use crate::framework::GpuFramework;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "luckfind",
    version = "0.2.0",
    about = "Bitcoin dormant address lottery — Rust secp256k1 scanner",
    long_about = "Scans random secp256k1 private keys, derives P2PKH addresses, \
                  and checks them against a built-in list of dormant addresses."
)]
pub struct Cli {
    /// Runtime duration in minutes.  Forever if unset.
    #[arg(short = 'd', long)]
    pub duration: Option<f64>,

    /// Worker thread count.  Defaults to half the logical-CPU count.
    #[arg(short = 'w', long)]
    pub workers: Option<usize>,

    /// CPU load target per worker (0.0-1.0).  1.0 = no throttling.
    #[arg(short = 'l', long, default_value = "1.0")]
    pub load: f64,

    /// Directory for match output files.
    #[arg(short = 'o', long, default_value = ".")]
    pub output_dir: String,

    /// Seconds between heartbeat lines (default 10.0).
    #[arg(short = 'H', long, default_value = "10.0")]
    pub heartbeat: f64,

    /// Run a 5-second benchmark burn-in and exit.
    #[arg(long)]
    pub bench: bool,

    /// Profile the scan pipeline stage-by-stage (SHA256 / RIPEMD160 / hash160
    /// combined / point-add) and exit.  Answers "is hash160 the bottleneck?" —
    /// the go/no-go for a GPU port.  Burns each stage for ~5s.
    #[arg(long)]
    pub profile: bool,

    /// Path to puzzle worklist file (`.json` or `.db`).
    ///
    /// In puzzle mode the binary:
    ///   - reads the worklist (subdivided sub-ranges, each with `current_hex`,
    ///     `end_hex`, `status`; the old `start_hex`/`range_bits` format is
    ///     auto-migrated on load),
    ///   - picks a random *pending* sub-range per worker (splitting it when
    ///     under the cap — 随机挑选 + 随机拆分策略),
    ///   - scans its keys sequentially from `current_hex` → `end_hex` (key += 1),
    ///   - 每处理 2^31 个 keys 就停放当前子区间并重新选择（旋转策略），
    ///   - marks the sub-range `finished` once the whole range is done,
    ///   - on SIGINT (Ctrl+C) writes the current scanning position back into
    ///     the worklist so the next run resumes exactly from there.
    ///
    /// Format handling:
    ///   - `.db` (SQLite): loaded directly; all saves go here.
    ///   - `.json`: if a sibling `{n}.db` already exists, the DB is loaded
    ///     (JSON is ignored — it was only the initial import); otherwise the
    ///     JSON is loaded and a `.db` is created for all future saves.
    ///     JSON is thus a one-time import format.
    ///
    /// Hex values that are shorter than 64 chars are left-padded to a full
    /// 32-byte key (high bytes zero-filled).
    #[arg(long)]
    pub puzzle: Option<String>,

    /// GPU framework to use for acceleration.
    /// "auto" (default): probe CUDA first (NVIDIA-preferred), then WebGPU,
    /// then CPU-only.  "webgpu": cross-platform via wgpu (Metal/Vulkan/DX12).
    /// "cuda": NVIDIA-only, uses CUDA kernels for better utilization.
    #[arg(long = "gpu_framework", default_value = "auto")]
    pub gpu_framework: GpuFramework,
}

impl Cli {
    pub fn parse() -> Self {
        <Self as clap::Parser>::parse()
    }

    pub fn workers(&self) -> usize {
        self.workers
            .unwrap_or_else(|| std::cmp::max(1, num_cpus::get() / 2))
    }
}
