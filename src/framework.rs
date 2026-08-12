//! GPU framework selection types shared between the binary and library.

/// GPU framework backend selection.
///
/// `Auto` (the CLI default) resolves at startup: CUDA is probed first (it
/// exists specifically because WebGPU's support for NVIDIA GPUs is limited),
/// then WebGPU, then CPU-only.  The resolved value — never `Auto` — is what
/// flows into `workers::run` / `puzzle::run`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuFramework {
    Auto,
    WebGpu,
    Cuda,
}

impl std::str::FromStr for GpuFramework {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "webgpu" => Ok(Self::WebGpu),
            "cuda" => Ok(Self::Cuda),
            other => Err(format!(
                "Unknown GPU framework: {other}. Use 'auto', 'webgpu' or 'cuda'."
            )),
        }
    }
}

impl std::fmt::Display for GpuFramework {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::WebGpu => write!(f, "webgpu"),
            Self::Cuda => write!(f, "cuda"),
        }
    }
}
