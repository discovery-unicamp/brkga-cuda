#include "Brkga.hpp"
#include "Chromosome.hpp"
#include "CudaError.cuh"
#include "CudaUtils.hpp"
#include "Decoder.hpp"
#include "Logger.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <numeric>
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

void box::Brkga::runPathRelinking(
    std::vector<std::pair<unsigned, unsigned>> pairs,
    unsigned blockSize) {
  if (blockSize >= chromosomeSize)
    throw std::invalid_argument(
        "Block size should be less than the chromosome size otherwise"
        " Path Relinking will do nothing");
  for (const auto& pair : pairs) {
    const auto base = pair.first;
    const auto guide = pair.second;
    if (base >= populationSize)
      throw std::invalid_argument("Base chromosome " + std::to_string(base)
                                  + " is out of range");
    if (guide >= populationSize)
      throw std::invalid_argument("Guide chromosome " + std::to_string(guide)
                                  + " is out of range");
  }

  // TODO move the chromosomes based on the order of their fitness (for GPU)
  // TODO ensure population wrapper has enough capacity
  // TODO can we implement this for the permutation without sorting every time?

  assert(decodeType.chromosome());
  const auto numberOfSegments = (chromosomeSize + blockSize - 1) / blockSize;
  logger::debug("Number of blocks to process:", numberOfSegments);

  assert((unsigned)pairs.size() < populationSize - mutantsSize - eliteSize);

  if (decodeType.onCpu()) {
    std::vector<unsigned> blocks(numberOfSegments);
    for (unsigned p = 0; p < numberOfPopulations; ++p) {
      unsigned k = 0;
      for (const auto& pair : pairs) {
        const auto base = pair.first;
        const auto guide = pair.second;
        logger::debug("Running Path Relinking with", base, "and", guide,
                      "of population", p);

        auto dChromosomes = cuda::alloc<float>(nullptr, 2 * chromosomeSize);
        copyChromosome<<<cuda::blocks(chromosomeSize, threadsPerBlock),
                         threadsPerBlock>>>(dChromosomes, base,
                                            dPopulation.row(p), chromosomeSize,
                                            dFitnessIdx.row(p));
        CUDA_CHECK_LAST();
        copyChromosome<<<cuda::blocks(chromosomeSize, threadsPerBlock),
                         threadsPerBlock>>>(dChromosomes + chromosomeSize,
                                            guide, dPopulation.row(p),
                                            chromosomeSize, dFitnessIdx.row(p));
        CUDA_CHECK_LAST();

        std::vector<float> chromosomes(2 * chromosomeSize);
        cuda::copy2h(nullptr, chromosomes.data(), dChromosomes,
                     2 * chromosomeSize);
        cuda::sync();

        float currentFitness = -1;
        populationWrapper[0] =
            Chromosome<float>(chromosomes.data(), chromosomeSize, 0);
        decoder->decode(1, populationWrapper, &currentFitness);
        assert(currentFitness > 0);
        logger::debug("Starting IPR with:", currentFitness);

        std::iota(blocks.begin(), blocks.end(), 0);
        bool foundAnyImprovement = false;
        for (unsigned i = numberOfSegments; i > 0; --i) {
          for (unsigned j = 0; j < i; ++j) {
            const auto b = blocks[j];
            const auto l = b * blockSize;
            const auto r = l + blockSize;
            assert(l < chromosomeSize);
            assert(j < numberOfPopulations * populationSize);  // overflow!
            populationWrapper[j] = Chromosome<float>(
                chromosomes.data(), chromosomeSize, /* base: */ 0,
                /* guide: */ 1, l, r);
          }

          decoder->decode(i, populationWrapper, fitness.data());

          unsigned best = 0;
          for (unsigned j = 1; j < i; ++j) {
            if (fitness[j] < fitness[best]) best = j;
          }
          logger::debug("Next best solution:", fitness[best]);
          if (fitness[best] >= currentFitness) break;  // No further improvement
          foundAnyImprovement = true;
          logger::debug("Improved:", currentFitness, "=>", fitness[best]);

          currentFitness = fitness[best];
          const auto b = blocks[best];
          const auto l = b * blockSize;
          const auto r = l + blockSize;
          for (unsigned j = l; j < r; ++j)
            chromosomes[j] = chromosomes[j + chromosomeSize];

          std::swap(blocks[best], blocks[i - 1]);  // "erase" the block used
        }

        if (!foundAnyImprovement) {
          logger::debug("The Path Relinking couldn't improve base", base,
                        "using", guide, "as guide");
          continue;
        }

        logger::debug("Copying back to the device");
        cuda::copy2d(nullptr, dChromosomes, chromosomes.data(), chromosomeSize);
        ++k;  // One more good solution
        const auto replacedChromosomeIndex = populationSize - k;
        copyToDevice<<<cuda::blocks(chromosomeSize, threadsPerBlock),
                       threadsPerBlock>>>(dPopulation.row(p),
                                          replacedChromosomeIndex, dChromosomes,
                                          chromosomeSize, dFitnessIdx.row(p));
        CUDA_CHECK_LAST();

        cuda::free(nullptr, dChromosomes);
      }

      logger::debug("Finished processing the population", p);
    }

    updateFitness();
    logger::debug("The (implicit) Path Relinking has finished");
  } else {
    logger::error("Only CPU & chromosome decoder is supported");
    abort();
  }
}