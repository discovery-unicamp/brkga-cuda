#include "../BasicTypes.hpp"
#include "../Logger.hpp"
#include "GpuUtils.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cctype>

// Defined by the bb_segsort implementation.
template <class Key, class Value>
void bbSegSort(Key*, Value*, std::size_t, std::size_t);

namespace box {
__global__ void deviceIota(uint* arr, uint n) {
  for (uint i = threadIdx.x; i < n; i += blockDim.x) arr[i] = i;
}

void gpu::iota(cudaStream_t stream, uint* arr, uint n) {
  constexpr auto threads = 256;
  logger::debug("iota on", n, "elements to array", (void*)arr, "on stream",
                (void*)stream, "using", threads, "threads");
  deviceIota<<<1, threads, 0, stream>>>(arr, n);
  CUDA_CHECK_LAST();
}

template <class T>
__global__ void deviceIotaMod(T* arr, uint n, uint k) {
  for (uint i = threadIdx.x; i < n; i += blockDim.x) arr[i] = i % k;
}

template <class T>
void iotaMod(cudaStream_t stream, T* arr, uint n, uint k) {
  constexpr auto threads = 256;
  logger::debug("iotaMod on", n, "elements mod", k, "to array", (void*)arr,
                "on stream", (void*)stream, "using", threads, "threads");
  deviceIotaMod<<<1, threads, 0, stream>>>(arr, n, k);
  CUDA_CHECK_LAST();
}

void gpu::iotaMod(cudaStream_t stream, uint* arr, uint n, uint k) {
  iotaMod(stream, arr, n, k);
}

void gpu::iotaMod(cudaStream_t stream, ulong* arr, uint n, uint k) {
  iotaMod(stream, arr, n, k);
}

auto gpu::_detail::CachedAllocator::allocate(std::size_t nbytes) -> byte* {
  byte* ptr = nullptr;

  auto iterFree = freeMem.find(nbytes);
  if (iterFree == freeMem.end()) {
    logger::debug("No memory found for", nbytes, "bytes");
    ptr = alloc<byte>(nullptr, nbytes);
  } else {
    logger::debug("Reuse", iterFree->first, "bytes of the requested", nbytes,
                  "bytes");
    assert(nbytes <= iterFree->first);
    nbytes = iterFree->first;
    ptr = iterFree->second;
    freeMem.erase(iterFree);
  }

  logger::debug("Allocated cached memory", (void*)ptr);
  allocMem.emplace(ptr, nbytes);
  return ptr;
}

void gpu::_detail::CachedAllocator::deallocate(byte* ptr, std::size_t) {
  logger::debug("Save", (void*)ptr, "to the cache");
  auto iterAlloc = allocMem.find(ptr);
  assert(iterAlloc != allocMem.end());

  auto nbytes = iterAlloc->second;
  freeMem.emplace(nbytes, ptr);
  allocMem.erase(iterAlloc);
}

void gpu::_detail::CachedAllocator::free() {
  logger::debug("Free", freeMem.size(), "unused memory in the cache");
  for (auto pair : freeMem) gpu::free(nullptr, pair.second);
  freeMem.clear();
}

gpu::_detail::CachedAllocator gpu::_detail::cachedAllocator;

void gpu::segSort(cudaStream_t stream,
                  Gene* dKeys,
                  GeneIndex* dValues,
                  std::size_t segCount,
                  std::size_t segSize) {
  // FIXME We need to block the host
  gpu::sync(stream);
  bbSegSort(dKeys, dValues, segCount, segSize);
  CUDA_CHECK_LAST();
}

gpu::Timer::Timer() {
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  reset();
}

gpu::Timer::~Timer() {
  // noexcept = no CUDA_CHECK
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

void gpu::Timer::reset() {
  CUDA_CHECK(cudaEventRecord(start));
}

float gpu::Timer::milliseconds() {
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = -1;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  return ms;
}

float gpu::Timer::seconds() {
  return milliseconds() / 1000;
}
}  // namespace box
