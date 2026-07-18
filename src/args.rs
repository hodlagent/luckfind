use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(
    name = "luckfind",
    version = "0.2.0",
    about = "Bitcoin dormant address lottery — Rust secp256k1 scanner",
    long_about = "Scans random secp256k1 private keys, derives P2PKH addresses, \
                  and checks them against a built-in list of dormant addresses."
)]
pub struct Cli {
    /// Runtime duration in minutes (single value or `min~max` range).  Forever if unset.
    #[arg(short = 'd', long)]
    pub duration: Option<f64>,

    /// Worker thread count.  Defaults to half the logical-CPU count.
    #[arg(short = 'w', long)]
    pub workers: Option<usize>,

    /// CPU load target per worker (0.0-1.0).  1.0 = no throttling.
    #[arg(short = 'l', long, default_value = "1.0")]
    pub load: f64,

    /// Path to candidate JSON file (default: built-in 78-address list).
    #[arg(short = 'a', long)]
    pub addrs: Option<String>,

    /// `builtin` (78 built-in addresses) or `file` (`--addrs`).
    /// Defaults to `file` when `--addrs` is given, else `builtin`.
    #[arg(short = 's', long, value_enum)]
    pub source: Option<Source>,

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

    /// Path to puzzle JSON worklist file (e.g. a btcpuzzle #76 range-split).
    ///
    /// In puzzle mode the binary:
    ///   - reads the JSON (subdivided sub-ranges, each with `start_hex`,
    ///     `range_bits`, `status`),
    ///   - picks a random *pending* sub-range per worker,
    ///   - scans its keys sequentially from `start` → `end` (key += 1),
    ///   - marks the sub-range `finished` once the whole range is done,
    ///   - on SIGINT (Ctrl+C) writes the current scanning position back into
    ///     the JSON so the next run resumes exactly from there.
    ///
    /// Hex values that are shorter than 64 chars are left-padded to a full
    /// 32-byte key (high bytes zero-filled).
    #[arg(long)]
    pub puzzle: Option<String>,
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum Source {
    Builtin,
    File,
}

pub struct RuntimeLimits {
    pub duration_secs:  Option<f64>,
    pub heartbeat_secs: f64,
    #[allow(dead_code)]
    pub load_target:    f64,    // reserved for future load-throttle impl
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
