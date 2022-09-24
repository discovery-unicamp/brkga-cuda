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

std::vector<float> box::Brkga::pathRelink(const unsigned blockSize,
                                          const unsigned base,
                                          const unsigned guide) {
  box::logger::debug("Running Path Relink with", base, "and", guide);

  auto dChromosome = cuda::alloc<float>(nullptr, 2 * chromosomeSize);
  copyChromosome<<<cuda::blocks(chromosomeSize, threadsPerBlock),
                   threadsPerBlock>>>(dChromosome, base, dPopulation.get(),
                                      chromosomeSize, dFitnessIdx.get());
  CUDA_CHECK_LAST();
  copyChromosome<<<cuda::blocks(chromosomeSize, threadsPerBlock),
                   threadsPerBlock>>>(dChromosome + chromosomeSize, guide,
                                      dPopulation.get(), chromosomeSize,
                                      dFitnessIdx.get());
  CUDA_CHECK_LAST();

  std::vector<float> chromosomes(2 * chromosomeSize);
  cuda::copy2h(nullptr, chromosomes.data(), dChromosome, 2 * chromosomeSize);
  cuda::sync();

  float bestFitness = -1;
  std::vector<float> bestGenes(chromosomes.begin(),
                               chromosomes.begin() + chromosomeSize);

  populationWrapper[0] = Chromosome<float>(bestGenes.data(), chromosomeSize, 0);
  decoder->decode(1, populationWrapper, &bestFitness);
  assert(bestFitness > 0);
  box::logger::debug("Starting Path Relink with:", bestFitness);

  const auto numberOfSegments = (chromosomeSize + blockSize - 1) / blockSize;
  box::logger::debug("Number of blocks to process:", numberOfSegments);
  std::vector<unsigned> blocks(numberOfSegments);
  std::iota(blocks.begin(), blocks.end(), 0);

  unsigned id = 0;
  for (unsigned i = numberOfSegments; i > 0; --i) {
    for (unsigned j = 0; j < i; ++j) {
      const auto b = blocks[j];
      const auto l = b * blockSize;
      const auto r = l + blockSize;  // Overflow will never happen here
      assert(l < chromosomeSize);
      populationWrapper[j] =
          Chromosome<float>(chromosomes.data(), chromosomeSize, /* base: */ id,
                            /* guide: */ (id ^ 1), l, r);
    }

    decoder->decode(i, populationWrapper, fitness.data());

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

  cuda::free(nullptr, dChromosome);
  return bestGenes;
}

void box::Brkga::runPathRelink(unsigned blockSize,
                               const std::vector<PathRelinkPair>& pairList) {
  // TODO move the chromosomes based on the order of their fitness (for GPU)
  //   is it really necessary to move now?
  // TODO ensure population wrapper has enough capacity
  // TODO can we implement this for the permutation without sorting every time?

  box::logger::debug("Run Path Relink between", pairList.size(), "pairs");

  // FIXME add support to gpu/permutation
  assert(decodeType.onCpu());
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
  auto dChromosome = cuda::alloc<float>(nullptr, chromosomeSize);

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
    cuda::copy2d(nullptr, dChromosome, bestGenes.data(), chromosomeSize);
    copyToDevice<<<cuda::blocks(chromosomeSize, threadsPerBlock),
                   threadsPerBlock>>>(
        dPopulation.row(pair.basePopulationId), replacedChromosomeIndex,
        dChromosome, chromosomeSize, dFitnessIdx.row(pair.basePopulationId));
    CUDA_CHECK_LAST();
  }

  cuda::free(nullptr, dChromosome);

  // FIXME should decode only the new chromosomes, not the population
  updateFitness();
  box::logger::debug("The Path Relink has finished");
}
