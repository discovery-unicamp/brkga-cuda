#include "../Brkga.hpp"
#include "../Chromosome.hpp"
#include "../CudaError.cuh"
#include "../CudaUtils.hpp"
#include "../Decoder.hpp"
#include "../Logger.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// FIXME this is an experimental feature

template <class T>
__global__ void copyChromosome(T* dst,
                               const unsigned index,
                               const T* src,
                               const unsigned chromosomeLength,
                               const unsigned* fitnessIdx) {
  const auto k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= chromosomeLength) return;

  const auto sortedIndex = fitnessIdx[index];
  dst[k] = src[sortedIndex * chromosomeLength + k];
}

__global__ void copyFitness(float* fitness,
                            unsigned index,
                            float* dFitness,
                            unsigned* dFitnessIdx) {
  const auto sortedIndex = dFitnessIdx[index];
  *fitness = dFitness[sortedIndex];
}

template <class T>
__global__ void copyToDevice(T* dst,
                             const unsigned index,
                             const T* src,
                             const unsigned chromosomeLength,
                             const unsigned* fitnessIdx) {
  const auto k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= chromosomeLength) return;

  const auto sortedIndex = fitnessIdx[index];
  dst[sortedIndex * chromosomeLength + k] = src[k];
}

template <class T>
__host__ __device__ void setupBlock(unsigned j,
                                    box::Chromosome<T>* wrapper,
                                    T* chromosomes,
                                    const unsigned* blocks,
                                    unsigned blockSize,
                                    unsigned chromosomeSize,
                                    unsigned id) {
  const auto b = blocks[j];
  const auto l = b * blockSize;
  const auto r = l + blockSize;  // Overflow will never happen here
  assert(l < chromosomeSize);
  wrapper[j] =
      box::Chromosome<float>(chromosomes, chromosomeSize, /* base: */ id,
                             /* guide: */ (id ^ 1), l, r);
}

template <class T>
__global__ void buildBlocksKernel(box::Chromosome<T>* wrapper,
                                  T* chromosomes,
                                  const unsigned* blocks,
                                  unsigned blockSize,
                                  unsigned chromosomeSize,
                                  unsigned id) {
  const auto j = blockIdx.x * blockDim.x + threadIdx.x;
  setupBlock(j, wrapper, chromosomes, blocks, blockSize, chromosomeSize, id);
}

template <class T>
void buildBlocks(unsigned n,
                 box::Chromosome<T>* wrapper,
                 T* chromosomes,
                 const unsigned* blocks,
                 unsigned blockSize,
                 unsigned chromosomeSize,
                 unsigned id) {
  for (unsigned j = 0; j < n; ++j)
    setupBlock(j, wrapper, chromosomes, blocks, blockSize, chromosomeSize, id);
}

std::vector<float> box::Brkga::pathRelink(const unsigned blockSize,
                                          const unsigned base,
                                          const unsigned guide) {
  assert(blockSize > 0);
  box::logger::debug("Running Path Relink with", base, "and", guide);

  auto dChromosomes = cuda::alloc<float>(nullptr, 2 * chromosomeSize);
  copyChromosome<<<cuda::blocks(chromosomeSize, threadsPerBlock),
                   threadsPerBlock>>>(dChromosomes, base, dPopulation.get(),
                                      chromosomeSize, dFitnessIdx.get());
  CUDA_CHECK_LAST();
  copyChromosome<<<cuda::blocks(chromosomeSize, threadsPerBlock),
                   threadsPerBlock>>>(dChromosomes + chromosomeSize, guide,
                                      dPopulation.get(), chromosomeSize,
                                      dFitnessIdx.get());
  CUDA_CHECK_LAST();

  std::vector<float> chromosomes(2 * chromosomeSize);
  cuda::copy2h(nullptr, chromosomes.data(), dChromosomes, 2 * chromosomeSize);
  cuda::sync();

  std::vector<float> bestGenes(chromosomes.begin(),
                               chromosomes.begin() + chromosomeSize);

  auto* dBestFitness = cuda::alloc<float>(nullptr, 1);
  copyFitness<<<1, 1>>>(dBestFitness, base, dFitness.get(), dFitnessIdx.get());
  CUDA_CHECK_LAST();
  float bestFitness = -1e30f;
  cuda::copy2h(nullptr, &bestFitness, dBestFitness, 1);
  cuda::free(nullptr, dBestFitness);
  box::logger::debug("Starting Path Relink with:", bestFitness);

  const auto numberOfSegments = (chromosomeSize + blockSize - 1) / blockSize;
  box::logger::debug("Number of blocks to process:", numberOfSegments);
  std::vector<unsigned> blocks(numberOfSegments);
  std::iota(blocks.begin(), blocks.end(), 0);

  fitness.resize(numberOfSegments);

  unsigned* dBlocks = nullptr;
  float* dFitness = nullptr;
  if (!decodeType.onCpu()) {
    dBlocks = cuda::alloc<unsigned>(nullptr, numberOfSegments);
    dFitness = cuda::alloc<float>(nullptr, numberOfSegments);
  }

  unsigned id = 0;
  for (unsigned i = numberOfSegments; i > 0; --i) {
    if (decodeType.onCpu()) {
      buildBlocks(i, populationWrapper, chromosomes.data(), blocks.data(),
                  blockSize, chromosomeSize, id);
      decoder->decode(i, populationWrapper, fitness.data());
    } else {
      cuda::copy2d(streams[0], dChromosomes, chromosomes.data(),
                   chromosomes.size());
      cuda::copy2d(streams[0], dBlocks, blocks.data(), i);
      buildBlocksKernel<<<1, i, 0, streams[0]>>>(populationWrapper,
                                                 dChromosomes, dBlocks,
                                                 blockSize, chromosomeSize, id);
      decoder->decode(streams[0], i, populationWrapper, dFitness);
      cuda::copy2h(streams[0], fitness.data(), dFitness, i);
      cuda::sync(streams[0]);
    }

    unsigned bestIdx = 0;
    for (unsigned j = 1; j < i; ++j) {
      if (fitness[j] < fitness[bestIdx]) bestIdx = j;
    }
    box::logger::debug("PR moved to:", fitness[bestIdx],
                       "-- incumbent:", bestFitness);

    const auto baseBegin = chromosomes.begin() + id * chromosomeSize;
    const auto guideBegin = chromosomes.begin() + (id ^ 1) * chromosomeSize;

    const auto changeOffset = blocks[bestIdx] * blockSize;
    const auto bs = std::min(chromosomeSize - changeOffset, blockSize);
    auto itFrom = guideBegin + changeOffset;
    auto itTo = baseBegin + changeOffset;
    std::copy(itFrom, itFrom + bs, itTo);

    if (fitness[bestIdx] < bestFitness) {
      bestFitness = fitness[bestIdx];
      std::copy(baseBegin, baseBegin + chromosomeSize, bestGenes.begin());
    }

    std::swap(blocks[bestIdx], blocks[i - 1]);  // "Erase" the block used
    id ^= 1;  // "Swap" the base and the guide chromosome
  }

  box::logger::debug("Path Relink finished with:", bestFitness);

  cuda::free(nullptr, dChromosomes);
  cuda::free(nullptr, dBlocks);
  return bestGenes;
}

void box::Brkga::runPathRelink(unsigned blockSize,
                               const std::vector<PathRelinkPair>& pairList) {
  // TODO move the chromosomes based on the order of their fitness (for GPU)
  //   is it really necessary to move now?
  // TODO ensure population wrapper has enough capacity
  // TODO can we implement this for the permutation without sorting every time?

  box::logger::debug("Run Path Relink between", pairList.size(), "pairs");

  // FIXME add support to permutation
  assert(decodeType.chromosome());

  if (blockSize < 1)
    throw std::invalid_argument("Invalid block size: "
                                + std::to_string(blockSize));
  if (blockSize >= chromosomeSize)
    throw std::invalid_argument(
        "Block size should be less than the chromosome size otherwise"
        " Path Relink will do nothing");

  for (const auto& pair : pairList) {
    if (pair.basePopulationId >= numberOfPopulations)
      throw std::invalid_argument("Invalid base population");
    if (pair.guidePopulationId >= numberOfPopulations)
      throw std::invalid_argument("Invalid guide population");
    if (pair.baseChromosomeId >= populationSize)
      throw std::invalid_argument("Invalid base chromosome");
    if (pair.guideChromosomeId >= populationSize)
      throw std::invalid_argument("Invalid guide chromosome");
  }

  std::vector<unsigned> insertedCount(numberOfPopulations, 0);
  auto dChromosomes = cuda::alloc<float>(nullptr, chromosomeSize);

  for (const auto& pair : pairList) {
    const auto base =
        pair.basePopulationId * populationSize + pair.baseChromosomeId;
    const auto guide =
        pair.guidePopulationId * populationSize + pair.guideChromosomeId;

    const auto bestGenes = pathRelink(blockSize, base, guide);

    // TODO use hamming distance/kendall tau to check if it should be included?
    //   maybe give the user a method to filter "duplicated" chromosomes with
    //   those methods

    ++insertedCount[pair.basePopulationId];
    assert(insertedCount[pair.basePopulationId] < populationSize - eliteSize);
    const auto replacedChromosomeIndex =
        populationSize - insertedCount[pair.basePopulationId];

    box::logger::debug("Copying the chromosome found back to the device");
    cuda::copy2d(nullptr, dChromosomes, bestGenes.data(), chromosomeSize);
    copyToDevice<<<cuda::blocks(chromosomeSize, threadsPerBlock),
                   threadsPerBlock>>>(
        dPopulation.row(pair.basePopulationId), replacedChromosomeIndex,
        dChromosomes, chromosomeSize, dFitnessIdx.row(pair.basePopulationId));
    CUDA_CHECK_LAST();
  }

  cuda::free(nullptr, dChromosomes);

  // FIXME should decode only the new chromosomes, not the population
  updateFitness();
  box::logger::debug("The Path Relink has finished");
}
