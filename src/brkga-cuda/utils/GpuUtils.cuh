#ifndef BOX_UTILS_GPUUTILS_CUH
#define BOX_UTILS_GPUUTILS_CUH

#include "GpuUtils.hpp"

#include <cuda_runtime.h>

namespace box {
namespace gpu {
template <typename F, typename... Args>
void vectorized(cudaStream_t stream,
                F kernel,
                uint n,
                uint threadsPerBlock,
                Args&... args) {
  kernel<<<gpu::blocks(n, threadsPerBlock), threadsPerBlock, 0, stream>>>(
      args...);
  CUDA_CHECK_LAST();
}
}  // namespace gpu
}  // namespace box

#endif  // BOX_UTILS_GPUUTILS_CUH
