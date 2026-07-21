//! GPU scanner — init, dispatch loop, calibration, match readback.
//!
//! Adapted from kangaroo's `solver.rs`: stripped of DP table / jump table logic.
//! Keeps: calibration loop (~120ms target), dispatch + readback pattern.

use super::buffers::GpuBuffers;
use super::context::GpuContext;
use super::pipeline::LuckfindPipeline;
use super::{GpuConfig, GpuState, NUM_GPU_THREADS};
use anyhow::Result;

pub struct GpuScanner {
    ctx: GpuContext,
    pipeline: LuckfindPipeline,
    buffers: GpuBuffers,
    bind_group: wgpu::BindGroup,
    pub steps_per_call: u32,
    /// Stride (keys per shader step). 1 = lottery; N = puzzle dense-tiling.
    pub stride: u32,
    /// Number of active candidate slots (1 = puzzle single target; 78 = lottery set).
    pub num_candidates: u32,
    /// Initial scalar per thread (LE limbs), for CPU-side match verification.
    initial_scalars: Vec<[u32; 8]>,
    pub total_ops: u64,
    /// Milliseconds to sleep after each dispatch (throttles GPU to leave
    /// headroom for display rendering; 0 = run at full speed).
    pub sleep_ms: u64,
}

impl GpuScanner {
    pub fn new(ctx: GpuContext, candidates: &[[u32; 5]]) -> Result<Self> {
        let pipeline = LuckfindPipeline::new(&ctx)?;
        let layout = pipeline.pipeline.get_bind_group_layout(0);

        // Generator point G is hardcoded in the shader as GX/GY constants.
        // Candidates are [[u32;5]] (LE u32 words), matching RIPEMD160 output format.
        let candidates_gpu: Vec<[u32; 5]> = candidates.to_vec();

        let buffers = GpuBuffers::new(&ctx, NUM_GPU_THREADS, &candidates_gpu)?;
        let bind_group = buffers.bind_group(&layout);

        Ok(Self {
            ctx,
            pipeline,
            buffers,
            bind_group,
            steps_per_call: 1,   // 1 step × 100k threads = 100k keys/dispatch (avoids TDR)
            stride: 1,           // lottery default; puzzle sets this to NUM_GPU_THREADS
            num_candidates: 78,  // lottery default; puzzle sets this to 1
            initial_scalars: Vec::new(),
            total_ops: 0,
            #[cfg(target_os = "macos")]
            sleep_ms: 0,         // macOS: no throttle needed
            #[cfg(not(target_os = "macos"))]
            sleep_ms: 24,        // Windows/Linux: throttle to leave headroom for display
        })
    }

    /// Initialize states with random starting points.
    ///
    /// For each of NUM_GPU_THREADS threads: generate random sk, compute pk = sk×G
    /// (one scalar mult per thread, parallelized with rayon), convert to Jacobian (z=1).
    #[allow(dead_code)]
    pub fn init_random(&mut self, _seed: u64) -> Result<()> {
        use rand::TryRng;
        use rayon::prelude::*;

        let n = NUM_GPU_THREADS as usize;
        let secp = secp256k1::Secp256k1::new();

        // Parallel generate (scalar_limbs, GpuState) pairs.
        let results: Vec<([u32; 8], GpuState)> = (0..n)
            .into_par_iter()
            .map(|_| {
                let mut buf = [0u8; 32];
                loop {
                    rand::rngs::SysRng
                        .try_fill_bytes(&mut buf)
                        .expect("OS entropy");
                    if let Ok(sk) = secp256k1::SecretKey::from_byte_array(buf) {
                        let pk =
                            secp256k1::PublicKey::from_secret_key(&secp, &sk);
                        let encoded = pk.serialize_uncompressed();
                        let mut x = [0u8; 32];
                        let mut y = [0u8; 32];
                        x.copy_from_slice(&encoded[1..33]);
                        y.copy_from_slice(&encoded[33..65]);
                        let (step_px, step_py) =
                            crate::gpu::convert::stride_step_point(self.stride);
                        return (
                            crate::gpu::convert::scalar_be_to_limbs(&buf),
                            GpuState {
                                x: crate::gpu::convert::be_bytes_to_limbs(&x),
                                y: crate::gpu::convert::be_bytes_to_limbs(&y),
                                z: [1, 0, 0, 0, 0, 0, 0, 0],
                                scalar: crate::gpu::convert::scalar_be_to_limbs(&buf),
                                step_px,
                                step_py,
                            },
                        );
                    }
                }
            })
            .collect();

        let initial_scalars: Vec<[u32; 8]> = results.iter().map(|(s, _)| *s).collect();
        let states: Vec<GpuState> = results.into_iter().map(|(_, st)| st).collect();

        self.ctx
            .queue()
            .write_buffer(&self.buffers.states, 0, bytemuck::cast_slice(&states));
        self.initial_scalars = initial_scalars;
        Ok(())
    }

    /// Deterministic seeding for puzzle mode: walker `i` starts at key
    /// `start + i` (scalar `start+i`, point `(start+i)·G`).  With stride = N
    /// each walker advances N keys/step, so the N walkers partition the chunk
    /// `[start, start + N·steps_per_call)` with zero overlap.  Parallelized
    /// with rayon (same cost model as init_random: one scalar mult per walker,
    /// paid once per chunk).
    ///
    /// `start_be` is the chunk's current scan position, big-endian [u8; 32].
    pub fn seed_range(&mut self, start_be: [u8; 32]) -> Result<()> {
        use rayon::prelude::*;
        let n = NUM_GPU_THREADS as usize;
        let secp = secp256k1::Secp256k1::new();
        let stride = self.stride;
        assert!(stride > 0, "stride must be set before seed_range");
        let (step_px, step_py) = crate::gpu::convert::stride_step_point(stride);

        let results: Vec<([u8; 32], GpuState)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let sk_be = crate::gpu::convert::scalar_add_be(&start_be, i as u64);
                let sk =
                    secp256k1::SecretKey::from_byte_array(sk_be).expect("start+i < n");
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

        self.ctx
            .queue()
            .write_buffer(&self.buffers.states, 0, bytemuck::cast_slice(&states));
        self.initial_scalars = initial_scalars;
        Ok(())
    }

    /// Upload config buffer.
    fn upload_config(&self) -> Result<()> {
        let config = GpuConfig {
            num_threads: NUM_GPU_THREADS,
            steps_per_call: self.steps_per_call,
            num_candidates: self.num_candidates,
            stride: self.stride,
        };
        self.ctx
            .queue()
            .write_buffer(&self.buffers.config, 0, bytemuck::bytes_of(&config));
        Ok(())
    }

    /// Dispatch one compute pass and read back matches.
    pub fn step(&mut self) -> Result<Vec<GpuMatchOutput>> {
        self.upload_config()?;

        // WORKGROUP_SIZE matches the shader's @workgroup_size(WORKGROUP_SIZE)
        // override in pipeline.rs (128 — must equal shared-array width in the
        // shader's batch-inversion scratch).
        let workgroups = (NUM_GPU_THREADS + 127) / 128;

        // Create a fresh staging buffer for each dispatch.  We cannot reuse a
        // staging buffer after a failed map_async because wgpu 28 leaves its
        // internal mapping state dirty, causing "Buffer is already mapped"
        // panics on reuse.
        let matches_byte_size = self.buffers.matches_byte_size();
        let staging_size = 4u64 + matches_byte_size;
        let staging = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: staging_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.ctx.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Luckfind Encoder") },
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Luckfind Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        // Copy match_count + matches → fresh staging.
        encoder.copy_buffer_to_buffer(
            &self.buffers.match_count,
            0,
            &staging,
            0,
            4,
        );
        encoder.copy_buffer_to_buffer(
            &self.buffers.matches,
            0,
            &staging,
            4,
            matches_byte_size,
        );
        self.ctx.queue().submit([encoder.finish()]);

        self.total_ops += NUM_GPU_THREADS as u64 * self.steps_per_call as u64;

        // Read back (consumes the staging buffer).
        let matches = self.readback_matches(&staging)?;

        // Reset match_count for next dispatch.
        self.ctx
            .queue()
            .write_buffer(&self.buffers.match_count, 0, &[0u8; 4]);

        // Throttle: sleep to leave GPU headroom for display rendering.
        if self.sleep_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(self.sleep_ms));
        }

        Ok(matches)
    }

    /// Read back matches from a fresh staging buffer.
    ///
    /// The staging buffer is consumed (dropped) after this call.  On success
    /// we explicitly unmap before dropping; on failure the buffer is already
    /// destroyed at the wgpu level so we just propagate the error.
    fn readback_matches(&self, staging: &wgpu::Buffer) -> Result<Vec<GpuMatchOutput>> {
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).ok();
        });
        self.ctx
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| anyhow::anyhow!("GPU poll failed: {e:?}"))?;

        rx.recv()
            .map_err(|e| anyhow::anyhow!("map callback dropped: {e}"))?
            .map_err(|e| anyhow::anyhow!("buffer map failed: {e:?}"))?;

        let data = slice.get_mapped_range();
        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let count = count.min(256);
        let mut matches = Vec::with_capacity(count);
        let match_size = std::mem::size_of::<GpuMatchOutput>();
        for i in 0..count {
            let offset = 4 + i * match_size;
            let bytes = &data[offset..offset + match_size];
            matches.push(bytemuck::pod_read_unaligned(bytes));
        }
        drop(data);
        staging.unmap();

        Ok(matches)
    }

    /// Initial scalar (LE limbs) for thread `i` — for test verification.
    #[allow(dead_code)]
    pub fn initial_scalar_bytes(&self, i: usize) -> [u32; 8] {
        self.initial_scalars[i]
    }

    /// Set thread `tid`'s initial state directly (test helper).
    #[allow(dead_code)]
    pub fn set_initial_state(
        &mut self,
        tid: usize,
        scalar_le: [u32; 8],
        point: GpuState,
    ) -> Result<()> {
        self.initial_scalars[tid] = scalar_le;
        // Read all states, patch slot tid, write back.
        let mut states = self.readback_states()?;
        states[tid] = point;
        self.ctx
            .queue()
            .write_buffer(&self.buffers.states, 0, bytemuck::cast_slice(&states));
        Ok(())
    }

    /// Read back all states (for testing). Copies states buffer → staging → CPU.
    #[allow(dead_code)]
    pub fn readback_states(&self) -> anyhow::Result<Vec<GpuState>> {
        let num = self.buffers.num_threads();
        let states_size = num as u64 * std::mem::size_of::<GpuState>() as u64;
        let staging = self.ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("states-staging"),
            size: states_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.ctx.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("readback-encoder") },
        );
        encoder.copy_buffer_to_buffer(&self.buffers.states, 0, &staging, 0, states_size);
        self.ctx.queue().submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            tx.send(res).unwrap();
        });
        self.ctx
            .device()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| anyhow::anyhow!("GPU poll failed: {e:?}"))?;
        rx.recv()
            .map_err(|e| anyhow::anyhow!("map callback dropped: {e}"))?
            .map_err(|e| anyhow::anyhow!("buffer map failed: {e:?}"))?;

        let data = slice.get_mapped_range();
        let mut states = Vec::with_capacity(num as usize);
        let state_size = std::mem::size_of::<GpuState>();
        for i in 0..num as usize {
            let offset = i * state_size;
            states.push(bytemuck::pod_read_unaligned(
                &data[offset..offset + state_size],
            ));
        }
        drop(data);
        staging.unmap();
        Ok(states)
    }
}

/// Output of a GPU-reported match.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMatchOutput {
    pub scalar: [u32; 8],
    pub pubkey_x: [u32; 8],
    pub pubkey_y: [u32; 8],
    pub hash160: [u32; 5],
    pub candidate_index: u32,
    pub thread_id: u32,
    pub _padding: u32,
}

const _: [(); 128] = [(); std::mem::size_of::<GpuMatchOutput>()];
