// Minimal CUDA device declarations for kernel.cu.
//
// Replaces <cuda_runtime.h> so PTX generation does not drag in the host C++
// standard library.  With very new MSVC STL versions (e.g. VS 2026 / MSVC
// 14.51 against CUDA 11.8) cuda_runtime.h pulls in type_traits etc. which the
// CUDA 11.x frontend cannot parse (STL1002 / "operator may not be a static
// member"), and the CUDA toolkit version checks reject the host compiler.
//
// Only the execution-configuration built-ins are needed — threadIdx, blockIdx
// and blockDim (declared here via device_launch_parameters.h) — plus
// __syncthreads(), which is a compiler builtin.  The kernel's match counter is
// incremented with inline PTX (`atom.add.u32`) rather than the atomicAdd
// builtin, because atomicAdd is never name-resolved under these minimal
// headers and nvcc would emit an extern call that the driver JIT cannot load.
//
// Note: this is incomplete by design — it is NOT a drop-in replacement for
// the CUDA runtime headers and must only be used by compute-only kernels.

#ifndef LUCKFIND_MINIMAL_CUDA_H
#define LUCKFIND_MINIMAL_CUDA_H

#if !defined(__CUDACC__) || defined(__CUDACC_RTC__)
#error "minimal_cuda.h is only valid for nvcc-compiled device code (__CUDACC__)"
#endif

#include <device_launch_parameters.h>

#endif // LUCKFIND_MINIMAL_CUDA_H