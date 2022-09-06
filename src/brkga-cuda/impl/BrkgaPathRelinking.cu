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
#include <utility>
#include <vector>

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

DecodedChromosome box::Brkga::pathRelinking(const unsigned base,
                                            const unsigned guide) {
  box::logger::debug("Running Path Relinking with", base, "and", guide);

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

  DecodedChromosome best;
  best.fitness = -1;
  best.genes = std::vector<float>(chromosomes.begin(),
                                  chromosomes.begin() + chromosomeSize);

  populationWrapper[0] =
      Chromosome<float>(best.genes.data(), chromosomeSize, 0);
  decoder->decode(1, populationWrapper, &best.fitness);
  assert(best.fitness > 0);
  box::logger::debug("Starting IPR with:", best.fitness);

  const unsigned blockSize = std::max(1u, (unsigned)(chromosomeSize * 0.15));
  if (blockSize >= chromosomeSize)
    throw std::invalid_argument(
        "Block size should be less than the chromosome size otherwise"
        " Path Relinking will do nothing");

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
      assert(j < numberOfPopulations * populationSize);  // overflow!
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
                       "-- incumbent:", best.fitness);

    const auto baseBegin = chromosomes.begin() + id * chromosomeSize;
    const auto guideBegin = chromosomes.begin() + (id ^ 1) * chromosomeSize;

    const auto changeOffset = blocks[bestIdx] * blockSize;
    const auto bs = std::min(chromosomeSize - changeOffset, blockSize);
    auto itFrom = guideBegin + changeOffset;
    auto itTo = baseBegin + changeOffset;
    std::copy(itFrom, itFrom + bs, itTo);

    if (fitness[bestIdx] < best.fitness) {
      best.fitness = fitness[bestIdx];
      std::copy(baseBegin, baseBegin + chromosomeSize, best.genes.begin());
    }

    std::swap(blocks[bestIdx], blocks[i - 1]);  // "erase" the block used
    id ^= 1;  // "Swap" the base and the guide chromosome
  }

  box::logger::debug("IPR finished with:", best.fitness);

  cuda::free(nullptr, dChromosomes);
  return best;
}

void box::Brkga::runPathRelinking() {
  // TODO move the chromosomes based on the order of their fitness (for GPU)
  //   is it really necessary to move now?
  // TODO ensure population wrapper has enough capacity
  // TODO can we implement this for the permutation without sorting every time?

  // For a while
  assert(decodeType.onCpu());
  assert(decodeType.chromosome());

  const auto geneCount = populationSize * chromosomeSize;
  std::mt19937 rng(0);
  std::uniform_int_distribution<unsigned> uid(0, eliteSize - 1);

  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    const auto base = uid(rng) + p * geneCount;
    const auto guide = uid(rng) + (p + 1) % numberOfPopulations * geneCount;

    DecodedChromosome best = pathRelinking(base, guide);

    // TODO use hamming distance/kendall tau to check if it should be included?
    //   maybe give the user a method to filter "duplicated" chromsomes with
    //   those methods
    box::logger::debug("Copying the chromosome found back to the device");
    auto dChromosomes = cuda::alloc<float>(nullptr, chromosomeSize);
    cuda::copy2d(nullptr, dChromosomes, best.genes.data(), chromosomeSize);
    const auto replacedChromosomeIndex = populationSize - 1;
    copyToDevice<<<cuda::blocks(chromosomeSize, threadsPerBlock),
                   threadsPerBlock>>>(dPopulation.row(p),
                                      replacedChromosomeIndex, dChromosomes,
                                      chromosomeSize, dFitnessIdx.row(p));
    CUDA_CHECK_LAST();

    cuda::free(nullptr, dChromosomes);
    box::logger::debug("Finished processing the population", p);
  }

  updateFitness();
  box::logger::debug("The (implicit) Path Relinking has finished");
}
