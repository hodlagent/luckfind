//! CUDA-accelerated secp256k1 scanner backend for NVIDIA GPUs.
//!
//! Uses the `cust` crate for CUDA driver API bindings.  The kernel source is in
//! `src/cuda/kernel.cu` and is compiled to PTX at build time via `build.rs`
//! (when nvcc is available).
//!
//! The public interface mirrors `crate::gpu::GpuScanner` so the lottery worker
//! can use either backend interchangeably.

pub mod scanner;
pub mod lottery;

pub use scanner::CudaScanner;

/// Number of parallel GPU threads — matches the WebGPU backend.
pub const NUM_GPU_THREADS: u32 = 100_000;

/// Workgroup size for CUDA kernel — 128 threads/block.
pub const WORKGROUP_SIZE: u32 = 128;

/// Maximum retained matches between readbacks.
pub const MAX_MATCHES: u32 = 256;
