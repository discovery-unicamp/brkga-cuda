#include "BasicTypes.hpp"
#include "Chromosome.hpp"
#include "utils/GpuUtils.hpp"

const unsigned TILE_DIM = 32;
const unsigned BLOCK_ROWS = 8;

namespace box {
template <class T>
__global__ void copyKernel(T* dst,
                           const Chromosome<T>* src,
                           const uint n,
                           const uint m) {
  const auto x = blockIdx.x * TILE_DIM + threadIdx.x;
  const auto y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (uint i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    if (y + i < n && x < m) dst[(y + i) * m + x] = src[y + i][x];
}

template <class T>
void Chromosome<T>::copy(cudaStream_t stream,
                         T* to,
                         const Chromosome<T>* chromosomes,
                         const uint n,
                         const uint chromosomeLength) {
  const dim3 grid((unsigned)((chromosomeLength + TILE_DIM - 1) / TILE_DIM),
                  (unsigned)((n + TILE_DIM - 1) / TILE_DIM));
  const dim3 block(TILE_DIM, BLOCK_ROWS);
  copyKernel<<<grid, block, 0, stream>>>(to, chromosomes, n, chromosomeLength);
  CUDA_CHECK_LAST();
}
}  // namespace box
