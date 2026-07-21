//! GPU context — cross-platform device management via wgpu 28 + pollster.
//!
//! Adapted from kangaroo's `gpu_crypto/context.rs`.  Uses `Backends::all()` so
//! it works on Metal (macOS), Vulkan (Windows/Linux), and DX12 (Windows).
//! Keeps: `create_shader_module` (3-source WGSL concatenation), `create_buffer`,
//! `create_buffer_init`, device/queue.

use anyhow::{Context, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct GpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    adapter_info: wgpu::AdapterInfo,
    /// Queried device limits (for future feature / `--gpu-list` output).
    #[allow(dead_code)]
    limits: wgpu::Limits,
}

/// A discovered GPU device (for `--gpu-list`).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub backend: wgpu::Backend,
}

impl GpuContext {
    /// Initialize the GPU backend on the given adapter index.
    /// Uses all available backends for the platform (Metal on macOS, Vulkan/DX12 on Windows/Linux).
    pub async fn new(device_index: u32) -> Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all()).await;
        let adapter = adapters
            .into_iter()
            .nth(device_index as usize)
            .context("GPU device index out of range")?;

        let adapter_info = adapter.get_info();
        Self::from_adapter(adapter, adapter_info).await
    }

    /// Synchronous convenience wrapper (luckfind doesn't need an async runtime).
    pub fn new_blocking(device_index: u32) -> Result<Self> {
        pollster::block_on(Self::new(device_index))
    }

    async fn from_adapter(adapter: wgpu::Adapter, adapter_info: wgpu::AdapterInfo) -> Result<Self> {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("luckfind-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .context("Failed to create GPU device")?;

        let limits = device.limits();
        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
            limits,
        })
    }

    pub fn device_name(&self) -> &str {
        &self.adapter_info.name
    }

    #[allow(dead_code)]
    pub fn backend(&self) -> wgpu::Backend {
        self.adapter_info.backend
    }

    #[allow(dead_code)]
    pub fn max_workgroup_size(&self) -> u32 {
        self.limits.max_compute_workgroup_size_x
    }

    /// Create an uninitialized buffer of `count` elements of type `T`.
    pub fn create_buffer<T: bytemuck::Pod>(
        &self,
        label: &str,
        usage: wgpu::BufferUsages,
        count: u64,
    ) -> Result<wgpu::Buffer> {
        let element_size = std::mem::size_of::<T>() as u64;
        let size = count
            .checked_mul(element_size)
            .context("Buffer size overflow")?;
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        }))
    }

    /// Create a buffer initialized with `data`.
    pub fn create_buffer_init<T: bytemuck::Pod>(
        &self,
        label: &str,
        usage: wgpu::BufferUsages,
        data: &[T],
    ) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage,
            })
    }

    /// Create a shader module by concatenating multiple WGSL sources.
    pub fn create_shader_module(&self, label: &str, sources: &[&str]) -> wgpu::ShaderModule {
        let source = sources.join("\n\n");
        self.device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            })
    }

    pub fn device(&self) -> Arc<wgpu::Device> {
        self.device.clone()
    }

    pub fn queue(&self) -> Arc<wgpu::Queue> {
        self.queue.clone()
    }
}

/// Enumerate available GPU devices (for `--gpu-list`).
#[allow(dead_code)]
pub fn enumerate_gpus() -> Vec<GpuDeviceInfo> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    pollster::block_on(async {
        instance
            .enumerate_adapters(wgpu::Backends::all())
            .await
            .into_iter()
            .map(|a| {
                let info = a.get_info();
                GpuDeviceInfo {
                    name: info.name,
                    backend: info.backend,
                }
            })
            .collect()
    })
}
