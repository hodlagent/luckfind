//! Compute pipeline setup for the luckfind GPU kernel.

use super::GpuContext;
use anyhow::Result;
use std::sync::Arc;
use wgpu::ComputePipeline;

/// Luckfind compute pipeline. wgpu types are Arc-wrapped, so Clone is cheap.
#[derive(Clone)]
pub struct LuckfindPipeline {
    pub pipeline: Arc<ComputePipeline>,
}

impl LuckfindPipeline {
    pub fn new(ctx: &GpuContext) -> Result<Self> {
        // Shader modules are concatenated in dependency order: field/curve ops
        // first, then sha256/ripemd160 compressors (from kangaroo v0.1.0), then a
        // thin glue layer exposing the sha256_block/ripemd160_block interface the
        // kernel expects, then the kernel itself.
        let field = include_str!("../shaders/field.wgsl");
        let curve = include_str!("../shaders/curve.wgsl");
        let sha256 = include_str!("../shaders/sha256.wgsl");
        let ripemd160 = include_str!("../shaders/ripemd160.wgsl");
        let hash_glue = include_str!("../shaders/hash_glue.wgsl");
        let kernel = include_str!("../shaders/luckfind.wgsl");

        let constants = [("WORKGROUP_SIZE", 128.0)];
        let shader = ctx.create_shader_module(
            "Luckfind Shader",
            &[field, curve, sha256, ripemd160, hash_glue, kernel],
        );
        let bind_group_layout = bind_group_layout(ctx);
        let pipeline_layout = ctx.device().create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Luckfind Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            },
        );
        let pipeline = ctx
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Luckfind Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    zero_initialize_workgroup_memory: true,
                },
                cache: None,
            });

        Ok(Self {
            pipeline: Arc::new(pipeline),
        })
    }
}

/// The bind group layout: 5 bindings matching `luckfind.wgsl`.
fn bind_group_layout(ctx: &GpuContext) -> wgpu::BindGroupLayout {
    ctx.device()
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Luckfind Bind Group Layout"),
            entries: &[
                // 0: Config (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: states (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: candidates (storage, read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: matches (storage, read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: match_count (storage, atomic)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
}
