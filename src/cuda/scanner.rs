//! CUDA scanner — init, dispatch loop, match readback via the `cust` crate.
//!
//! Mirrors `crate::gpu::scanner::GpuScanner` interface so the lottery worker
//! can dispatch to either WebGPU or CUDA transparently.

use anyhow::{Context, Result};

use crate::gpu::{GpuMatchOutput, GpuState, GpuConfig};
use crate::cuda::{NUM_GPU_THREADS, WORKGROUP_SIZE, MAX_MATCHES};

use cust::context::{Context as CudaContext, ContextFlags};
use cust::device::{Device, DeviceAttribute};
use cust::memory::{CopyDestination, DeviceBox, DeviceBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};
use cust::prelude::*;
use bytemuck::Zeroable;

/// CUDA scanner — manages device memory and launches the luckfind kernel.
pub struct CudaScanner {
    // `context`/`device` are never read directly but MUST stay alive for the
    // lifetime of the scanner (dropping them invalidates all device pointers).
    // DROP ORDER MATTERS: fields are dropped in declaration order, and cust
    // requires the CUDA context to outlive every stream/module/buffer, so the
    // resources are declared FIRST and `device`/`context` LAST.  Putting the
    // context first (as initially written) destroyed it while the stream and
    // device buffers were still alive, crashing with STATUS_ACCESS_VIOLATION
    // at exit (0xc0000005) whenever a scanner was dropped.
    stream: Stream,
    module: Module,
    device_states: DeviceBuffer<GpuState>,
    device_candidates: DeviceBuffer<[u32; 5]>,
    device_matches: DeviceBuffer<GpuMatchOutput>,
    device_match_count: DeviceBox<u32>,
    device_config: DeviceBox<GpuConfig>,
    /// Host-side copy of the config (the kernel takes it by value; DeviceBox
    /// is not deref-able in cust 0.3.2).
    config: GpuConfig,
    /// Stride (keys per shader step). 1 = lottery; N = puzzle dense-tiling.
    pub stride: u32,
    /// Number of active candidate slots.
    pub num_candidates: u32,
    /// Whether the kernel checks the compressed (33-byte) pubkey (0/1).
    /// Mirrors `[btc] check_compressed_pk`; default 1.
    pub check_compressed_pk: u32,
    /// Whether the kernel checks the uncompressed (65-byte) pubkey (0/1).
    /// Mirrors `[btc] check_uncompressed_pk`; default 1.
    pub check_uncompressed_pk: u32,
    /// Steps per dispatch call.
    pub steps_per_call: u32,
    /// Initial scalar per thread (LE limbs), for test verification.
    initial_scalars: Vec<[u32; 8]>,
    pub total_ops: u64,
    /// Device name for logging.
    device_name: String,
    /// CUDA device ordinal this scanner runs on (0-based).
    device_index: u32,
    // Dropped LAST (see struct doc): device before context.
    #[allow(dead_code)]
    device: Device,
    #[allow(dead_code)]
    context: CudaContext,
}

impl CudaScanner {
    /// Initialize CUDA scanner with the given candidate hash160s on a specific
    /// device ordinal (0-based, as reported by `nvidia-smi` / `Device::num_devices`).
    ///
    /// Each scanner owns its own CUDA context for `device_index`; a worker thread
    /// per device is the supported multi-GPU arrangement (cust docs: "Users can
    /// simply make new contexts for every thread with no concern").
    pub fn new_on_device(candidates: &[[u32; 5]], device_index: u32) -> Result<Self> {
        // Initialize CUDA
        cust::init(cust::CudaFlags::empty())
            .context("CUDA initialization failed — is the NVIDIA driver installed?")?;

        let device = Device::get_device(device_index)
            .context(format!("No CUDA-capable device at index {device_index}"))?;
        let device_name = device.name()?;

        let context = CudaContext::new(device.clone())
            .context("Failed to create CUDA context")?;
        context.set_flags(ContextFlags::SCHED_AUTO)
            .context("Failed to set CUDA context flags")?;

        let stream = Stream::new(StreamFlags::DEFAULT, None)
            .context("Failed to create CUDA stream")?;

        // Load PTX module (compiled from kernel.cu at build time)
        let ptx = Self::get_ptx();
        let (ptx_ver, ptx_arch) = Self::ptx_header(ptx);
        let cc = Self::compute_capability(device.clone());
        let module = Module::from_ptx(ptx, &[]).map_err(|e| {
            anyhow::anyhow!(
                "Failed to load CUDA PTX module on {device_name} (compute capability {cc}).\n\
                 Embedded PTX: {ptx_ver} targeting {ptx_arch}\n\
                 Underlying driver error: {e}\n\
                 This usually means the NVIDIA driver on this machine is too old to JIT-\
                 compile the PTX ISA this binary was built with.  Update the driver \
                 (https://www.nvidia.com/drivers) and retry.  If it still fails, rebuild \
                 luckfind with an older CUDA toolkit, or set CUDA_ARCH to match this GPU."
            )
        })?;

        // Allocate device memory
        let device_states = DeviceBuffer::<GpuState>::zeroed(NUM_GPU_THREADS as usize)
            .context("Failed to allocate device states buffer")?;

        // Candidates buffer: 78 slots fixed
        let mut cand_buffer = vec![[0u32; 5]; 78];
        for (i, c) in candidates.iter().enumerate().take(78) {
            cand_buffer[i] = *c;
        }
        let device_candidates = DeviceBuffer::from_slice(&cand_buffer)
            .context("Failed to allocate device candidates buffer")?;

        let device_matches = DeviceBuffer::<GpuMatchOutput>::zeroed(MAX_MATCHES as usize)
            .context("Failed to allocate device matches buffer")?;

        let device_match_count = DeviceBox::new(&0u32)
            .context("Failed to allocate device match count")?;

        let gpu_config = GpuConfig {
            num_threads: NUM_GPU_THREADS,
            steps_per_call: 1,
            num_candidates: candidates.len().min(78) as u32,
            stride: 1,
            check_compressed_pk: 1,   // default: both serialisations checked
            check_uncompressed_pk: 1,
            _pad: 0,
            _pad2: 0,
        };
        let device_config = DeviceBox::new(&gpu_config)
            .context("Failed to allocate device config")?;

        Ok(Self {
            stream,
            module,
            device_states,
            device_candidates,
            device_matches,
            device_match_count,
            device_config,
            config: gpu_config,
            stride: 1,
            num_candidates: candidates.len().min(78) as u32,
            check_compressed_pk: 1,
            check_uncompressed_pk: 1,
            steps_per_call: 1,
            initial_scalars: Vec::new(),
            total_ops: 0,
            device_name,
            device_index,
            device,
            context,
        })
    }

    /// Get the CUDA device ordinal this scanner runs on.
    pub fn device_index(&self) -> u32 {
        self.device_index
    }

    /// Get the device name for logging.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Number of CUDA-capable devices visible to the driver (0 = none/disabled).
    ///
    /// Gated on `cuda_compiled` (set by build.rs only when nvcc actually
    /// produced a real PTX) so a stub PTX from a failed nvcc run never reports
    /// phantom devices.  Quiet by design — callers decide how to log it.
    pub fn device_count() -> u32 {
        #[cfg(not(cuda_compiled))]
        {
            0
        }
        #[cfg(cuda_compiled)]
        {
            if cust::init(cust::CudaFlags::empty()).is_err() {
                return 0;
            }
            Device::num_devices().unwrap_or(0)
        }
    }

    /// Names of all CUDA-capable devices, in device-ordinal order.
    ///
    /// Used at startup to log the multi-GPU inventory ("识别多张显卡").  Empty
    /// when CUDA is disabled or no device is reachable.
    pub fn device_names() -> Vec<String> {
        #[cfg(not(cuda_compiled))]
        {
            Vec::new()
        }
        #[cfg(cuda_compiled)]
        {
            if cust::init(cust::CudaFlags::empty()).is_err() {
                return Vec::new();
            }
            let n = Device::num_devices().unwrap_or(0);
            (0..n)
                .filter_map(|i| Device::get_device(i).ok())
                .filter_map(|d| d.name().ok())
                .collect()
        }
    }

    /// Probe whether any CUDA device is available without creating a full
    /// scanner.  True when ≥1 device is reachable; per-device `new_on_device`
    /// still fails individually (e.g. a too-old GPU that cannot JIT the PTX).
    ///
    /// Gated on `cuda_compiled` (set by build.rs only when nvcc actually
    /// produced a real PTX).  Without the gate, a stub PTX from a failed nvcc
    /// run would pass the device probe and the worker would only fail later at
    /// kernel launch, with a confusing "device detected" message upstream.
    pub fn probe() -> bool {
        #[cfg(not(cuda_compiled))]
        {
            eprintln!("  [GPU] CUDA kernel was not compiled at build time (nvcc or cl.exe unavailable) — CUDA backend disabled.");
            eprintln!("  [GPU] Install the CUDA toolkit and rebuild from a Visual Studio Developer Command Prompt.");
            return false;
        }
        #[cfg(cuda_compiled)]
        {
            Self::device_count() > 0
        }
    }

    /// Get the PTX string for the kernel.
    /// The PTX is embedded at compile time by build.rs.
    fn get_ptx() -> &'static str {
        include_str!(concat!(env!("OUT_DIR"), "/luckfind.ptx"))
    }

    /// Extract the PTX ISA version and target arch from the embedded header,
    /// e.g. (".version 8.7", ".target sm_89").  Used to diagnose driver-vs-PTX
    /// compatibility when `cuModuleLoadData` fails.
    fn ptx_header(ptx: &str) -> (String, String) {
        let grab = |prefix: &str| -> String {
            ptx.lines()
                .find_map(|l| {
                    l.trim().strip_prefix(prefix).map(|rest| {
                        let v = rest.trim();
                        if v.is_empty() {
                            format!("{prefix}?")
                        } else {
                            format!("{prefix}{v}")
                        }
                    })
                })
                .unwrap_or_else(|| format!("{prefix}?"))
        };
        (grab(".version"), grab(".target"))
    }

    /// Human-readable device compute capability ("8.9"), or "?" if unreadable.
    fn compute_capability(device: Device) -> String {
        let major = device.clone().get_attribute(DeviceAttribute::ComputeCapabilityMajor);
        let minor = device.get_attribute(DeviceAttribute::ComputeCapabilityMinor);
        match (major, minor) {
            (Ok(m), Ok(n)) => format!("{m}.{n}"),
            _ => "?".to_string(),
        }
    }

    /// Initialize states with random starting points inside the puzzle key space.
    pub fn init_random(&mut self, puzzle_set: &crate::puzzles::PuzzleSet) -> Result<()> {
        use rayon::prelude::*;

        let n = NUM_GPU_THREADS as usize;
        let secp = secp256k1::Secp256k1::new();

        let results: Vec<([u32; 8], GpuState)> = (0..n)
            .into_par_iter()
            .map(|_| {
                let mut buf = [0u8; 32];
                let idx = puzzle_set.pick_random_puzzle();
                let range = &puzzle_set.ranges()[idx];
                puzzle_set.generate_key_in_range(range, &mut buf);
                let sk = secp256k1::SecretKey::from_byte_array(buf)
                    .expect("generate_key_in_range always produces a valid key in [2^70, 2^160)");
                let pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);
                let encoded = pk.serialize_uncompressed();
                let mut x = [0u8; 32];
                let mut y = [0u8; 32];
                x.copy_from_slice(&encoded[1..33]);
                y.copy_from_slice(&encoded[33..65]);
                let (step_px, step_py) =
                    crate::gpu::convert::stride_step_point(self.stride);
                (
                    crate::gpu::convert::scalar_be_to_limbs(&buf),
                    GpuState {
                        x: crate::gpu::convert::be_bytes_to_limbs(&x),
                        y: crate::gpu::convert::be_bytes_to_limbs(&y),
                        z: [1, 0, 0, 0, 0, 0, 0, 0],
                        scalar: crate::gpu::convert::scalar_be_to_limbs(&buf),
                        step_px,
                        step_py,
                    },
                )
            })
            .collect();

        let initial_scalars: Vec<[u32; 8]> = results.iter().map(|(s, _)| *s).collect();
        let states: Vec<GpuState> = results.into_iter().map(|(_, st)| st).collect();

        self.device_states.copy_from(&states)
            .context("Failed to copy states to device")?;
        self.initial_scalars = initial_scalars;
        Ok(())
    }

    /// Deterministic seeding for puzzle mode.
    pub fn seed_range(&mut self, start_be: [u8; 32]) -> Result<()> {
        use rayon::prelude::*;

        let n = NUM_GPU_THREADS as usize;
        let secp = secp256k1::Secp256k1::new();
        let stride = self.stride;
        let (step_px, step_py) = crate::gpu::convert::stride_step_point(stride);

        let results: Vec<([u8; 32], GpuState)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let sk_be = crate::gpu::convert::scalar_add_be(&start_be, i as u64);
                let sk = secp256k1::SecretKey::from_byte_array(sk_be).expect("start+i < n");
                let pk = secp256k1::PublicKey::from_secret_key(&secp, &sk);
                let encoded = pk.serialize_uncompressed();
                let mut x = [0u8; 32];
                let mut y = [0u8; 32];
                x.copy_from_slice(&encoded[1..33]);
                y.copy_from_slice(&encoded[33..65]);
                (
                    sk_be,
                    GpuState {
                        x: crate::gpu::convert::be_bytes_to_limbs(&x),
                        y: crate::gpu::convert::be_bytes_to_limbs(&y),
                        z: [1, 0, 0, 0, 0, 0, 0, 0],
                        scalar: crate::gpu::convert::scalar_be_to_limbs(&sk_be),
                        step_px,
                        step_py,
                    },
                )
            })
            .collect();

        let initial_scalars: Vec<[u32; 8]> = results.iter().map(|(s, _)| {
            crate::gpu::convert::scalar_be_to_limbs(s)
        }).collect();
        let states: Vec<GpuState> = results.into_iter().map(|(_, st)| st).collect();

        self.device_states.copy_from(&states)
            .context("Failed to copy states to device")?;
        self.initial_scalars = initial_scalars;
        Ok(())
    }

    /// Upload config to device and keep the host-side copy for the launch.
    fn upload_config(&mut self) -> Result<()> {
        let config = GpuConfig {
            num_threads: NUM_GPU_THREADS,
            steps_per_call: self.steps_per_call,
            num_candidates: self.num_candidates,
            stride: self.stride,
            check_compressed_pk: self.check_compressed_pk,
            check_uncompressed_pk: self.check_uncompressed_pk,
            _pad: 0,
            _pad2: 0,
        };
        self.config = config;
        self.device_config.copy_from(&config)
            .context("Failed to upload config to device")?;
        Ok(())
    }

    /// Reset match count on device.
    fn reset_match_count(&mut self) -> Result<()> {
        self.device_match_count.copy_from(&0u32)
            .context("Failed to reset match count")?;
        Ok(())
    }

    /// Dispatch one kernel launch and read back matches.
    pub fn step(&mut self) -> Result<Vec<GpuMatchOutput>> {
        self.upload_config()?;
        self.reset_match_count()?;

        // Launch kernel
        let function = self.module.get_function("luckfind_kernel")
            .context("Failed to find luckfind_kernel function")?;

        let blocks = (NUM_GPU_THREADS + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

        // The launch! macro requires a plain `ident` (not a field expression),
        // so bind the stream to a local first.  The kernel takes the config by
        // value and match_count by pointer.
        let stream = &self.stream;
        let config = self.config;
        let match_count = self.device_match_count.as_device_ptr();

        unsafe {
            launch!(
                function<<<blocks, WORKGROUP_SIZE, 0, stream>>>(
                    config,
                    self.device_states.as_device_ptr(),
                    self.device_candidates.as_device_ptr(),
                    self.device_matches.as_device_ptr(),
                    match_count
                )
            ).context("CUDA kernel launch failed")?;
        }

        self.stream.synchronize()
            .context("CUDA stream synchronize failed")?;

        self.total_ops += NUM_GPU_THREADS as u64 * self.steps_per_call as u64;

        // Read back match count
        let mut match_count = 0u32;
        self.device_match_count.copy_to(&mut match_count)
            .context("Failed to read back match count")?;

        let match_count = match_count.min(MAX_MATCHES);
        if match_count == 0 {
            return Ok(Vec::new());
        }

        // Read back matches (device buffer is fixed at MAX_MATCHES slots;
        // copy the whole buffer then truncate to the count written).
        let mut matches = vec![GpuMatchOutput::zeroed(); MAX_MATCHES as usize];
        self.device_matches.copy_to(&mut matches)
            .context("Failed to read back matches")?;
        matches.truncate(match_count as usize);

        Ok(matches)
    }

    /// Initial scalar for thread i (test helper).
    #[allow(dead_code)]
    pub fn initial_scalar_bytes(&self, i: usize) -> [u32; 8] {
        self.initial_scalars[i]
    }

    /// Read back all states (for testing).
    #[allow(dead_code)]
    pub fn readback_states(&self) -> Result<Vec<GpuState>> {
        let mut states = vec![GpuState::zeroed(); NUM_GPU_THREADS as usize];
        self.device_states.copy_to(&mut states)
            .context("Failed to read back states")?;
        Ok(states)
    }
}

impl Drop for CudaScanner {
    fn drop(&mut self) {
        // Ensure all CUDA work is complete before dropping
        let _ = self.stream.synchronize();
    }
}
