#include "bb_segsort.h"

#include <cuda_runtime.h>

#include <cctype>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#define cuCheck(status)                                     \
  if (status != cudaSuccess) {                              \
    throw std::runtime_error(std::string("bb-segsort: ")    \
                             + cudaGetErrorString(status)); \
  }

static std::vector<int> seg;
static int* dSeg = nullptr;
static std::size_t dSegLength = 0;

template <class Key, class Value>
void bbSegSort(Key* keys,
               Value* values,
               std::size_t segCount,
               std::size_t segSize) {
  if (segSize > std::numeric_limits<int>::max())
    throw std::invalid_argument("Segment is too big! (>= 2^31)");
  if (segCount * segSize > std::numeric_limits<int>::max())
    throw std::invalid_argument("Total size is too big! (>= 2^31)");

  // Cache the segments to avoid re-creating/re-calculating them
  seg.reserve(segCount);
  if (seg.size() >= 2 && seg[1] != (int)segSize) {
    seg.resize(1);
  } else if (seg.empty()) {
    seg.push_back(0);
  }

  // Check if the segments were updated or is in the first call
  if (seg.size() < segCount || dSegLength == 0) {
    std::size_t start = seg.size();
    seg.resize(segCount);
    for (std::size_t i = start; i < segCount; ++i)
      seg[i] = seg[i - 1] + (int)segSize;

    // Copy to device only if updated or in the first call
    if (dSegLength < seg.size()) {
      cuCheck(cudaFree(dSeg));
      cuCheck(cudaMalloc(&dSeg, seg.size() * sizeof(int)));
      dSegLength = seg.size();
    }
    cuCheck(cudaMemcpy(dSeg, seg.data(), seg.size() * sizeof(int),
                       cudaMemcpyHostToDevice));
  }

  bb_segsort(keys, values, (int)(segCount * segSize), dSeg, (int)segCount);
  cuCheck(cudaPeekAtLastError());
}

// Define the required interfaces for brkga-cuda
template void bbSegSort(float*, unsigned*, std::size_t, std::size_t);
