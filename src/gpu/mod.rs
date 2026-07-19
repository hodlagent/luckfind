//! GPU-accelerated secp256k1 scanner backend (Metal via wgpu + naga WGSL→MSL).
//!
//! 100k independent GPU threads each walk `P += G` with per-step hash160
//! comparison against a candidate set of Bitcoin addresses.  The bottleneck
//! is the secp256k1 point-add (~2.4µs/key on CPU); the GPU parallelizes this
//! 100k-way for a target throughput of 80-150 Mkeys/s on M4.
//!
//! Architecture mirrors the kangaroo ECDLP solver (`/Users/jerin/projects/kangaroo`):
//! we reuse its `field.wgsl` (mod-p arithmetic) and `curve.wgsl` (Jacobian
//! point ops) verbatim, then add a `luckfind.wgsl` kernel that does sequential
//! stepping + SHA256/RIPEMD160 instead of kangaroo's random-jump + DP mechanism.

pub mod context;
pub mod pipeline;
pub mod buffers;
pub mod scanner;
pub mod convert;

pub use context::GpuContext;
pub use scanner::{GpuScanner, GpuMatchOutput};

/// The number of parallel GPU threads (independent P+=G walkers).
///
/// This is the single knob for GPU parallelism — change it here and it
/// propagates everywhere: bind group sizing, dispatch count, init buffer.
pub const NUM_GPU_THREADS: u32 = 100_000;

/// Matches the WGSL `Config` layout exactly. 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuConfig {
    pub num_threads: u32,
    pub steps_per_call: u32,
    pub num_candidates: u32,
    /// Keys advanced per shader step (`P += stride·G`, `scalar += stride`).
    /// 1 = lottery (step = G); N = puzzle dense-tiling (step = N·G).
    pub stride: u32,
}

/// Matches the WGSL `GeneratorPoint` layout. 64 bytes.
/// Uploaded from Rust to avoid hand-transcribing secp256k1 constants into WGSL.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GeneratorPoint {
    pub gx: [u32; 8],
    pub gy: [u32; 8],
}

/// 192 bytes — matches WGSL `GpuState` (3× Jacobian + scalar + step point).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuState {
    pub x: [u32; 8],
    pub y: [u32; 8],
    pub z: [u32; 8],
    pub scalar: [u32; 8],
    /// Per-walker affine step point (LE limbs). Lottery: = G. Puzzle: = N·G.
    pub step_px: [u32; 8],
    pub step_py: [u32; 8],
}

const _: [(); 192] = [(); std::mem::size_of::<GpuState>()];
