//! GPU buffer management — config, states, candidates, matches, match_count.
//!
//! Simplified from kangaroo's `gpu/buffers.rs`: no double-buffering (matches
//! are rare; we read back once per dispatch). 5 buffers.
//!
//! Note: the staging buffer is NOT stored here because after a failed
//! `map_async` the buffer's internal mapping state is left dirty (wgpu 28
//! does not reset `mc.mapped_range` on failure), which would cause
//! "Buffer is already mapped" panics on reuse. Instead, a fresh staging
//! buffer is created for each dispatch.

use super::{GpuConfig, GpuContext, GpuMatchOutput, GpuState};
use anyhow::Result;
use std::sync::Arc;

/// Maximum number of matches we retain between readbacks.
pub(crate) const MAX_MATCHES: u32 = 256;

pub struct GpuBuffers {
    device: Arc<wgpu::Device>,
    pub config: wgpu::Buffer,
    pub states: wgpu::Buffer,
    pub candidates: wgpu::Buffer,
    pub matches: wgpu::Buffer,
    pub match_count: wgpu::Buffer,
}

impl GpuBuffers {
    pub fn new(
        ctx: &GpuContext,
        num_threads: u32,
        candidates: &[[u32; 5]],
    ) -> Result<Self> {
        let device = ctx.device();

        let config = ctx.create_buffer::<GpuConfig>(
            "config",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            1,
        )?;

        let states = ctx.create_buffer::<GpuState>(
            "states",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            num_threads as u64,
        )?;

        let candidates_buf = ctx.create_buffer_init(
            "candidates",
            wgpu::BufferUsages::STORAGE,
            candidates,
        );

        let matches = ctx.create_buffer::<GpuMatchOutput>(
            "matches",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            MAX_MATCHES as u64,
        )?;

        let match_count = ctx.create_buffer::<u32>(
            "match_count",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            1,
        )?;

        Ok(Self {
            device,
            config,
            states,
            candidates: candidates_buf,
            matches,
            match_count,
        })
    }

    #[allow(dead_code)]
    pub fn num_threads(&self) -> u32 {
        (self.states.size() / size_of::<GpuState>() as u64) as u32
    }

    pub fn matches_byte_size(&self) -> u64 {
        (MAX_MATCHES as u64) * size_of::<GpuMatchOutput>() as u64
    }

    /// Build the bind group for the luckfind pipeline.
    pub fn bind_group(&self, layout: &wgpu::BindGroupLayout) -> wgpu::BindGroup {
        self.device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Luckfind Bind Group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.config.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.states.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.candidates.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.matches.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.match_count.as_entire_binding(),
                    },
                ],
            })
    }
}
