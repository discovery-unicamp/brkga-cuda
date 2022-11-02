#include "../BasicTypes.hpp"
#include "../Brkga.hpp"
#include "../Chromosome.hpp"
#include "../Decoder.hpp"
#include "../Logger.hpp"
#include "../utils/GpuUtils.hpp"

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

namespace box {
template <class T>
__global__ void copyChromosome(T* dst,
                               const uint index,
                               const T* src,
                               const uint chromosomeLength,
                               const uint* fitnessIdx) {
  const auto k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= chromosomeLength) return;

  const auto sortedIndex = fitnessIdx[index];
  dst[k] = src[sortedIndex * chromosomeLength + k];
}

__global__ void copyFitness(box::Fitness* fitness,
                            uint index,
                            box::Fitness* dFitness,
                            uint* dFitnessIdx) {
  const auto sortedIndex = dFitnessIdx[index];
  *fitness = dFitness[sortedIndex];
}

template <class T>
__global__ void copyToDevice(T* dst,
                             const uint index,
                             const T* src,
                             const uint chromosomeLength,
                             const uint* fitnessIdx) {
  const auto k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= chromosomeLength) return;

  const auto sortedIndex = fitnessIdx[index];
  dst[sortedIndex * chromosomeLength + k] = src[k];
}

template <class T>
__host__ __device__ void setupBlock(uint j,
                                    Chromosome<T>* wrapper,
                                    T* chromosomes,
                                    const uint* blocks,
                                    uint blockSize,
                                    uint chromosomeLength,
                                    uint id) {
  const auto b = blocks[j];
  const auto l = b * blockSize;
  const auto r = l + blockSize;  // Overflow will never happen here
  assert(l < chromosomeLength);
  wrapper[j] = Chromosome<Gene>(chromosomes, chromosomeLength, /* base: */ id,
                                /* guide: */ (id ^ 1), l, r);
}

template <class T>
__global__ void buildBlocksKernel(Chromosome<T>* wrapper,
                                  T* chromosomes,
                                  const uint* blocks,
                                  uint blockSize,
                                  uint chromosomeLength,
                                  uint id) {
  const auto j = blockIdx.x * blockDim.x + threadIdx.x;
  setupBlock(j, wrapper, chromosomes, blocks, blockSize, chromosomeLength, id);
}

template <class T>
void buildBlocks(uint n,
                 Chromosome<T>* wrapper,
                 T* chromosomes,
                 const uint* blocks,
                 uint blockSize,
                 uint chromosomeLength,
                 uint id) {
  for (uint j = 0; j < n; ++j)
    setupBlock(j, wrapper, chromosomes, blocks, blockSize, chromosomeLength,
               id);
}

std::vector<Gene> Brkga::pathRelink(const uint base, const uint guide) {
  logger::debug("Running Path Relink with", base, "and", guide);

  auto dChromosomes = gpu::alloc<Gene>(nullptr, 2 * config.chromosomeLength());
  copyChromosome<<<gpu::blocks(config.chromosomeLength(), config.gpuThreads()),
                   config.gpuThreads()>>>(dChromosomes, base, dPopulation.get(),
                                          config.chromosomeLength(),
                                          dFitnessIdx.get());
  CUDA_CHECK_LAST();
  copyChromosome<<<gpu::blocks(config.chromosomeLength(), config.gpuThreads()),
                   config.gpuThreads()>>>(
      dChromosomes + config.chromosomeLength(), guide, dPopulation.get(),
      config.chromosomeLength(), dFitnessIdx.get());
  CUDA_CHECK_LAST();

  std::vector<Gene> chromosomes(2 * config.chromosomeLength());
  gpu::copy2h(nullptr, chromosomes.data(), dChromosomes,
              2 * config.chromosomeLength());
  gpu::sync();

  std::vector<Gene> bestGenes(chromosomes.begin(),
                              chromosomes.begin() + config.chromosomeLength());

  auto* dBestFitness = gpu::alloc<box::Fitness>(nullptr, 1);
  copyFitness<<<1, 1>>>(dBestFitness, base, dFitness.get(), dFitnessIdx.get());
  CUDA_CHECK_LAST();
  box::Fitness bestFitness = -1e30f;
  gpu::copy2h(nullptr, &bestFitness, dBestFitness, 1);
  gpu::free(nullptr, dBestFitness);
  logger::debug("Starting PR with:", bestFitness);

  const auto numberOfSegments =
      (config.chromosomeLength() + config.pathRelinkBlockSize() - 1)
      / config.pathRelinkBlockSize();
  logger::debug("Number of blocks to process:", numberOfSegments);
  std::vector<uint> blocks(numberOfSegments);
  std::iota(blocks.begin(), blocks.end(), 0);

  fitness.resize(numberOfSegments);

  uint* dBlocks = nullptr;
  box::Fitness* dFitnessPtr = nullptr;
  if (!config.decodeType().onCpu()) {
    dBlocks = gpu::alloc<uint>(nullptr, numberOfSegments);
    dFitnessPtr = gpu::alloc<box::Fitness>(nullptr, numberOfSegments);
  }

  uint id = 0;
  for (uint i = numberOfSegments; i > 0; --i) {
    if (config.decodeType().onCpu()) {
      buildBlocks(i, populationWrapper, chromosomes.data(), blocks.data(),
                  config.pathRelinkBlockSize(), config.chromosomeLength(), id);
      config.decoder()->decode(i, populationWrapper, fitness.data());
    } else {
      gpu::copy2d(streams[0], dChromosomes, chromosomes.data(),
                  chromosomes.size());
      gpu::copy2d(streams[0], dBlocks, blocks.data(), i);
      buildBlocksKernel<<<1, (unsigned)i, 0, streams[0]>>>(
          populationWrapper, dChromosomes, dBlocks,
          config.pathRelinkBlockSize(), config.chromosomeLength(), id);
      config.decoder()->decode(streams[0], i, populationWrapper, dFitnessPtr);
      gpu::copy2h(streams[0], fitness.data(), dFitnessPtr, i);
      gpu::sync(streams[0]);
    }

    uint bestIdx = 0;
    for (uint j = 1; j < i; ++j) {
      if (fitness[j] < fitness[bestIdx]) bestIdx = j;
    }
    logger::debug("PR moved to:", fitness[bestIdx],
                  format(Separator(""), "(incumbent: ", bestFitness, ")"));

    const auto baseBegin = chromosomes.begin() + id * config.chromosomeLength();
    const auto guideBegin =
        chromosomes.begin() + (id ^ 1) * config.chromosomeLength();

    const auto changeOffset = blocks[bestIdx] * config.pathRelinkBlockSize();
    const auto bs = std::min(config.chromosomeLength() - changeOffset,
                             config.pathRelinkBlockSize());
    auto itFrom = guideBegin + changeOffset;
    auto itTo = baseBegin + changeOffset;
    std::copy(itFrom, itFrom + bs, itTo);

    if (fitness[bestIdx] < bestFitness) {
      bestFitness = fitness[bestIdx];
      std::copy(baseBegin, baseBegin + config.chromosomeLength(),
                bestGenes.begin());
    }

    std::swap(blocks[bestIdx], blocks[i - 1]);  // "Erase" the block used
    id ^= 1;  // "Swap" the base and the guide chromosome
  }

  logger::debug("Path Relink finished with:", bestFitness);

  gpu::free(nullptr, dChromosomes);
  gpu::free(nullptr, dBlocks);
  return bestGenes;
}

void Brkga::runPathRelink(const std::vector<PathRelinkPair>& pairList) {
  // TODO move the chromosomes based on the order of their fitness (for GPU)
  //   is it really necessary to move now?
  // TODO ensure population wrapper has enough capacity
  // FIXME add support to permutation
  // TODO can we implement this for the permutation without sorting every time?
  logger::debug("Run Path Relink between", pairList.size(), "pairs");
  if (config.pathRelinkBlockSize() == 0)
    throw InvalidArgument("Block size wasn't defined", __FUNCTION__);

  for (const auto& pair : pairList) {
    InvalidArgument::max(
        Arg<uint>(pair.basePopulationId, "base population"),
        Arg<uint>(config.numberOfPopulations() - 1, "#populations - 1"),
        __FUNCTION__);
    InvalidArgument::max(
        Arg<uint>(pair.guidePopulationId, "guide population"),
        Arg<uint>(config.numberOfPopulations() - 1, "#populations - 1"),
        __FUNCTION__);
    InvalidArgument::max(
        Arg<uint>(pair.baseChromosomeId, "base chromosome"),
        Arg<uint>(config.populationSize() - 1, "|population| - 1"),
        __FUNCTION__);
    InvalidArgument::max(
        Arg<uint>(pair.guideChromosomeId, "guide chromosome"),
        Arg<uint>(config.populationSize() - 1, "|population| - 1"),
        __FUNCTION__);
  }

  std::vector<uint> insertedCount(config.numberOfPopulations(), 0);
  auto dChromosomes = gpu::alloc<Gene>(nullptr, config.chromosomeLength());

  for (const auto& pair : pairList) {
    const auto base =
        pair.basePopulationId * config.populationSize() + pair.baseChromosomeId;
    const auto guide = pair.guidePopulationId * config.populationSize()
                       + pair.guideChromosomeId;

    const auto bestGenes = pathRelink(base, guide);

    // TODO use hamming distance/kendall tau to check if it should be included?
    //   maybe give the user a method to filter "duplicated" chromosomes with
    //   those methods

    ++insertedCount[pair.basePopulationId];
    assert(insertedCount[pair.basePopulationId]
           < config.populationSize() - config.numberOfElites());
    const auto replacedChromosomeIndex =
        config.populationSize() - insertedCount[pair.basePopulationId];

    logger::debug("Copying the chromosome found back to the device");
    gpu::copy2d(nullptr, dChromosomes, bestGenes.data(),
                config.chromosomeLength());
    copyToDevice<<<gpu::blocks(config.chromosomeLength(), config.gpuThreads()),
                   config.gpuThreads()>>>(
        dPopulation.row(pair.basePopulationId), replacedChromosomeIndex,
        dChromosomes, config.chromosomeLength(),
        dFitnessIdx.row(pair.basePopulationId));
    CUDA_CHECK_LAST();
  }

  gpu::free(nullptr, dChromosomes);

  // FIXME should decode only the new chromosomes, not the population
  updateFitness();
  logger::debug("The Path Relink has finished");
}
}  // namespace box
