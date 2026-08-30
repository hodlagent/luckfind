//! luckfind — Bitcoin dormant address lottery (Rust / libsecp256k1 reimplementation).
//!
//! Library-facing pieces for integration tests and external consumers:
//! the embedded puzzle table (`puzzles`), address derivation (`btc`),
//! TOML config + the `[btc]` check switches (`config`), shared progress
//! counters (`progress`), the worker-pool types (`workers`), and the wgpu
//! backend (`gpu`).
//!
//! The binary-only modules (`args`, `report`, `puzzle`) stay private — they
//! only make sense inside `main`.

pub mod puzzles;
pub mod btc;
pub mod config;
pub mod progress;
pub mod gpu;
pub mod workers;
pub mod framework;
#[cfg(feature = "cuda")]
pub mod cuda;

// Re-export key types for integration tests and external consumers.
pub use puzzles::{PuzzleRange, PuzzleSet};
pub use workers::ScanTarget;
