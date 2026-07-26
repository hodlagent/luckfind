//! luckfind — Bitcoin dormant address lottery (Rust / libsecp256k1 reimplementation).
//!
//! Exposes `btc` addrs helpers so integration tests can validate address
//! derivation against known vectors without running the full lottery binary.

pub mod puzzles;
pub mod btc;
pub mod progress;
pub mod gpu;
pub mod workers;

// Re-export key types for integration tests and external consumers.
pub use puzzles::{PuzzleRange, PuzzleSet};
pub use workers::ScanTarget;

// workers / report / puzzle intentionally private — only the binary uses them.
