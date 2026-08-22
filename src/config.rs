//! TOML configuration file support.
//!
//! The config file is located either explicitly (`--config <path>`) or, when
//! the flag is omitted, auto-discovered in the current working directory
//! (`luckfind.toml`, then `config.toml`).
//!
//! Precedence, highest first:
//!   1. command-line flags (`--puzzle`, `--remote`, `--cpu-rotate-keys`, …)
//!   2. the config file
//!   3. built-in defaults
//!
//! The file is entirely optional and every key is optional — it supplies
//! defaults for settings the flags omit, so the same file can drive a headless
//! worker while an explicit CLI flag still overrides it for a one-off run.
//!
//! ```toml
//! # mode = "random" (default) | "puzzle" | "remote"
//! mode = "puzzle"
//!
//! # per-claim rotation (reclaim) budgets; 0 disables rotation
//! cpu_rotate_keys = 134217728    # 2^27 (default)
//! gpu_rotate_keys = 2147483648   # 2^31 (default)
//!
//! # CPU workers: `enabled` toggles CPU scanning (default true); `load` is the
//! # proportion (0.1..=1.0) of the resolved `--workers` thread count to run
//! # (default 1.0 = all threads).
//! [cpu]
//! enabled = true
//! load = 1.0
//!
//! # GPU collision scanning: engaged only when `enabled` (default true) AND a
//! # GPU device is actually available.
//! [gpu]
//! enabled = true
//!
//! [puzzle]                        # required when mode = "puzzle"
//! database = "bin/71.db"
//!
//! [remote]                        # required when mode = "remote"
//! uri = "http://192.168.1.10:42069"
//! ```

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;

/// Scan mode, selected by the config file's `mode` key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Default.  Lottery scan against the embedded 77-puzzle set.
    Random,
    /// Deterministic sub-range scan from a worklist database (`[puzzle] database`).
    Puzzle,
    /// Claim chunks over HTTP from a LAN hub (`[remote] uri`).
    Remote,
}

impl Mode {
    /// Parse a `mode` string (trimmed, case-insensitive).
    pub fn parse(s: &str) -> anyhow::Result<Mode> {
        match s.trim().to_ascii_lowercase().as_str() {
            "random" | "lottery" => Ok(Mode::Random),
            "puzzle" => Ok(Mode::Puzzle),
            "remote" => Ok(Mode::Remote),
            other => Err(anyhow::anyhow!(
                "unknown mode {other:?} (expected \"random\", \"puzzle\", or \"remote\")"
            )),
        }
    }
}

/// On-disk configuration.  All fields optional; absent keys fall back to
/// CLI flags then built-in defaults.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Scan mode; defaults to [`Mode::Random`] when absent.
    pub mode: Option<String>,
    /// Per-claim CPU rotation (reclaim) budget in puzzle mode.  `0` disables.
    pub cpu_rotate_keys: Option<u64>,
    /// Per-claim GPU rotation (reclaim) budget in puzzle mode.  `0` disables.
    pub gpu_rotate_keys: Option<u64>,
    /// `[cpu]` section — CPU worker availability and load.
    #[serde(default)]
    pub cpu: CpuSection,
    /// `[gpu]` section — GPU availability for collision scanning.
    #[serde(default)]
    pub gpu: GpuSection,
    /// `[puzzle]` section — required when `mode = "puzzle"`.
    #[serde(default)]
    pub puzzle: PuzzleSection,
    /// `[remote]` section — required when `mode = "remote"`.
    #[serde(default)]
    pub remote: RemoteSection,
}

/// `[cpu]` section.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct CpuSection {
    /// Whether CPU workers run at all.  Absent ⇒ `true`.
    pub enabled: Option<bool>,
    /// Proportion (0.1..=1.0) of the resolved `--workers` thread count that
    /// actually runs.  `1.0` = all threads (default).  Absent ⇒ `1.0`.
    pub load: Option<f64>,
}

/// `[gpu]` section.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct GpuSection {
    /// Whether the GPU is used for collision scanning.  The GPU is engaged
    /// only when this is `true` (absent ⇒ `true`) AND a device is actually
    /// available.
    pub enabled: Option<bool>,
}

/// `[puzzle]` section.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct PuzzleSection {
    /// Path to the worklist (`.db` runtime format, or `.json` one-time import).
    pub database: Option<String>,
}

/// `[remote]` section.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(default)]
pub struct RemoteSection {
    /// Hub base URL, e.g. `http://192.168.1.10:42069`.
    pub uri: Option<String>,
}

impl Config {
    /// Load and parse a TOML config file.
    pub fn load(path: &Path) -> anyhow::Result<Config> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let cfg: Config = toml::from_str(&text)
            .with_context(|| format!("parsing config {}", path.display()))?;
        Ok(cfg)
    }
}

/// Candidate filenames auto-discovered in the current working directory when
/// `--config` is not given, in priority order.
pub const CANDIDATES: [&str; 2] = ["luckfind.toml", "config.toml"];

/// Outcome of [`Config::discover`].
#[derive(Debug)]
pub enum Discovery {
    /// No candidate file exists in the current working directory.
    None,
    /// A candidate exists and parsed successfully.
    Loaded(PathBuf, Config),
    /// A candidate exists but failed to parse.
    Failed(PathBuf, anyhow::Error),
}

impl Config {
    /// Auto-discover a config file in the current working directory:
    /// `luckfind.toml` first, then `config.toml`.  Returns [`Discovery::None`]
    /// when neither exists.  Discovery is cwd-relative, so relative
    /// `database` paths in the file resolve the same way a plain run does.
    pub fn discover() -> Discovery {
        Self::discover_in(Path::new("."))
    }

    /// [`Config::discover`] restricted to a specific directory (used by tests).
    fn discover_in(dir: &Path) -> Discovery {
        for name in CANDIDATES {
            let path = dir.join(name);
            if path.exists() {
                return match Self::load(&path) {
                    Ok(cfg) => Discovery::Loaded(path, cfg),
                    Err(e) => Discovery::Failed(path, e),
                };
            }
        }
        Discovery::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_parse() {
        assert_eq!(Mode::parse("puzzle").unwrap(), Mode::Puzzle);
        assert_eq!(Mode::parse("  RANDOM ").unwrap(), Mode::Random);
        assert_eq!(Mode::parse("lottery").unwrap(), Mode::Random);
        assert_eq!(Mode::parse("remote").unwrap(), Mode::Remote);
        assert!(Mode::parse("banana").is_err());
    }

    #[test]
    fn empty_config_is_default() {
        let cfg: Config = toml::from_str("").unwrap();
        assert!(cfg.mode.is_none());
        assert!(cfg.cpu_rotate_keys.is_none());
        assert!(cfg.gpu_rotate_keys.is_none());
        assert!(cfg.cpu.enabled.is_none());
        assert!(cfg.cpu.load.is_none());
        assert!(cfg.gpu.enabled.is_none());
        assert!(cfg.puzzle.database.is_none());
        assert!(cfg.remote.uri.is_none());
    }

    #[test]
    fn parses_full_config() {
        let toml_str = r#"
            mode = "puzzle"
            cpu_rotate_keys = 134217728
            gpu_rotate_keys = 2147483648

            [cpu]
            enabled = false
            load = 0.5

            [gpu]
            enabled = true

            [puzzle]
            database = "bin/71.db"

            [remote]
            uri = "http://192.168.1.10:42069"
        "#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.mode.as_deref(), Some("puzzle"));
        assert_eq!(cfg.cpu_rotate_keys, Some(134217728));
        assert_eq!(cfg.gpu_rotate_keys, Some(2147483648));
        assert_eq!(cfg.cpu.enabled, Some(false));
        assert_eq!(cfg.cpu.load, Some(0.5));
        assert_eq!(cfg.gpu.enabled, Some(true));
        assert_eq!(cfg.puzzle.database.as_deref(), Some("bin/71.db"));
        assert_eq!(cfg.remote.uri.as_deref(), Some("http://192.168.1.10:42069"));
    }

    #[test]
    fn cpu_gpu_sections_are_optional() {
        // A config with neither section still deserializes with defaults.
        let cfg: Config = toml::from_str("mode = \"random\"").unwrap();
        assert!(cfg.cpu.enabled.is_none());
        assert!(cfg.cpu.load.is_none());
        assert!(cfg.gpu.enabled.is_none());

        // Setting only `[gpu] enabled` leaves `[cpu]` at defaults.
        let cfg: Config = toml::from_str("[gpu]\nenabled = false").unwrap();
        assert_eq!(cfg.gpu.enabled, Some(false));
        assert!(cfg.cpu.enabled.is_none());
        assert!(cfg.cpu.load.is_none());
    }

    #[test]
    fn rotation_zero_round_trips() {
        // 0 means "disable rotation" (resolved to `None` in main); confirm the
        // parse round-trips the value.
        let cfg: Config = toml::from_str("cpu_rotate_keys = 0").unwrap();
        assert_eq!(cfg.cpu_rotate_keys, Some(0));
    }

    /// Per-test scratch dir under the OS temp dir (unique per test+process).
    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("luckfind_cfg_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn discover_returns_none_when_absent() {
        let dir = tmp_dir("empty");
        assert!(matches!(Config::discover_in(&dir), Discovery::None));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discover_finds_config_toml() {
        let dir = tmp_dir("find");
        std::fs::write(dir.join("config.toml"), "mode = \"puzzle\"\ncpu_rotate_keys = 42\n").unwrap();
        match Config::discover_in(&dir) {
            Discovery::Loaded(found, cfg) => {
                assert_eq!(found, dir.join("config.toml"));
                assert_eq!(cfg.mode.as_deref(), Some("puzzle"));
                assert_eq!(cfg.cpu_rotate_keys, Some(42));
            }
            _ => panic!("expected Discovery::Loaded"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discover_prefers_luckfind_toml() {
        let dir = tmp_dir("order");
        std::fs::write(dir.join("config.toml"), "cpu_rotate_keys = 1\n").unwrap();
        std::fs::write(dir.join("luckfind.toml"), "cpu_rotate_keys = 2\n").unwrap();
        match Config::discover_in(&dir) {
            Discovery::Loaded(found, cfg) => {
                assert_eq!(found, dir.join("luckfind.toml"));
                assert_eq!(cfg.cpu_rotate_keys, Some(2));
            }
            _ => panic!("expected Discovery::Loaded"),
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn discover_failed_on_bad_toml() {
        let dir = tmp_dir("bad");
        std::fs::write(dir.join("config.toml"), "not valid toml ===").unwrap();
        assert!(matches!(Config::discover_in(&dir), Discovery::Failed(..)));
        std::fs::remove_dir_all(&dir).ok();
    }
}
