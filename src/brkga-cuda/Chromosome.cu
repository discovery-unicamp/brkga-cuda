#include "Chromosome.hpp"
#include "CudaError.cuh"

const unsigned TILE_DIM = 32;
const unsigned BLOCK_ROWS = 8;

template <class T>
__global__ void copyKernel(T* dst,
                           const box::Chromosome<T>* src,
                           const unsigned n,
                           const unsigned m) {
  const unsigned x = blockIdx.x * TILE_DIM + threadIdx.x;
  const unsigned y = blockIdx.y * TILE_DIM + threadIdx.y;
  for (unsigned i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    if (y + i < n && x < m) dst[(y + i) * m + x] = src[y + i][x];
}

template <class T>
void box::Chromosome<T>::copy(cudaStream_t stream,
                              T* to,
                              const Chromosome<T>* chromosomes,
                              const unsigned n,
                              const unsigned chromosomeLength) {
  const dim3 grid((chromosomeLength + TILE_DIM - 1) / TILE_DIM,
                  (n + TILE_DIM - 1) / TILE_DIM);
  const dim3 block(TILE_DIM, BLOCK_ROWS);
  copyKernel<<<grid, block, 0, stream>>>(to, chromosomes, n, chromosomeLength);
  CUDA_CHECK_LAST();
}
