#include "CudaError.cuh"
#include "CudaUtils.hpp"
#include "Logger.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cctype>

__global__ void deviceIota(unsigned* arr, unsigned n) {
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) arr[i] = i;
}

void box::cuda::iota(cudaStream_t stream, unsigned* arr, unsigned n) {
  constexpr auto threads = 256;
  box::logger::debug("iota on", n, "elements to array", (void*)arr, "on stream",
                     (void*)stream, "using", threads, "threads");
  deviceIota<<<1, threads, 0, stream>>>(arr, n);
  CUDA_CHECK_LAST();
}

__global__ void deviceIotaMod(unsigned* arr, unsigned n, unsigned k) {
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) arr[i] = i % k;
}

void box::cuda::iotaMod(cudaStream_t stream,
                        unsigned* arr,
                        unsigned n,
                        unsigned k) {
  constexpr auto threads = 256;
  box::logger::debug("iotaMod on", n, "elements mod", k, "to array", (void*)arr,
                     "on stream", (void*)stream, "using", threads, "threads");
  deviceIotaMod<<<1, threads, 0, stream>>>(arr, n, k);
  CUDA_CHECK_LAST();
}

auto box::cuda::_detail::CachedAllocator::allocate(std::size_t nbytes)
    -> byte* {
  byte* ptr = nullptr;

  auto iterFree = freeMem.find(nbytes);
  if (iterFree == freeMem.end()) {
    box::logger::debug("No memory found for", nbytes, "bytes");
    ptr = alloc<byte>(nullptr, nbytes);
  } else {
    box::logger::debug("Reuse", iterFree->first, "bytes of the requested",
                       nbytes, "bytes");
    assert(nbytes <= iterFree->first);
    nbytes = iterFree->first;
    ptr = iterFree->second;
    freeMem.erase(iterFree);
  }

  box::logger::debug("Allocated cached memory", (void*)ptr);
  allocMem.emplace(ptr, nbytes);
  return ptr;
}

void box::cuda::_detail::CachedAllocator::deallocate(byte* ptr, std::size_t) {
  box::logger::debug("Save", (void*)ptr, "to the cache");
  auto iterAlloc = allocMem.find(ptr);
  assert(iterAlloc != allocMem.end());

  auto nbytes = iterAlloc->second;
  freeMem.emplace(nbytes, ptr);
  allocMem.erase(iterAlloc);
}

void box::cuda::_detail::CachedAllocator::free() {
  box::logger::debug("Free", freeMem.size(), "unused memory in the cache");
  for (auto pair : freeMem) box::cuda::free(nullptr, pair.second);
  freeMem.clear();
}

box::cuda::_detail::CachedAllocator box::cuda::_detail::cachedAllocator;

// Defined by the bb_segsort implementation.
template <class Key, class Value>
void bbSegSort(Key*, Value*, std::size_t, std::size_t);

void box::cuda::segSort(cudaStream_t stream,
                        float* dKeys,
                        unsigned* dValues,
                        std::size_t segCount,
                        std::size_t segSize) {
  // FIXME We need to block the host
  cuda::sync(stream);
  bbSegSort(dKeys, dValues, segCount, segSize);
  CUDA_CHECK_LAST();
}

box::cuda::Timer::Timer() {
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  reset();
}

box::cuda::Timer::~Timer() {
  // noexcept = no CUDA_CHECK
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void box::cuda::Timer::reset() {
  CUDA_CHECK(cudaEventRecord(start));
}

float box::cuda::Timer::milliseconds() {
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = -1;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}
