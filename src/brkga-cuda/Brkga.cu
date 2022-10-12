#include "Brkga.hpp"
#include "BrkgaConfiguration.hpp"
#include "BrkgaFilter.hpp"
#include "CudaError.cuh"
#include "DecodeType.hpp"
#include "Decoder.hpp"
#include "Logger.hpp"
#include "except/InvalidArgument.hpp"
#include "utils/GpuUtils.hpp"

#include <curand.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

box::Brkga::Brkga(
    const BrkgaConfiguration& config,
    const std::vector<std::vector<std::vector<float>>>& initialPopulation)
    : dPopulation(config.numberOfPopulations,
                  config.populationSize * config.chromosomeLength),
      dPopulationTemp(config.numberOfPopulations,
                      config.populationSize * config.chromosomeLength),
      populationWrapper(nullptr),
      dFitness(config.numberOfPopulations, config.populationSize),
      dFitnessIdx(config.numberOfPopulations, config.populationSize),
      dPermutations(config.numberOfPopulations,
                    config.populationSize * config.chromosomeLength),
      dRandomEliteParent(config.numberOfPopulations, config.populationSize),
      dRandomParent(config.numberOfPopulations, config.populationSize) {
  CUDA_CHECK_LAST();
  logger::debug("Building BoxBrkga with", config.numberOfPopulations,
                "populations, each with", config.populationSize,
                "chromosomes of length", config.chromosomeLength);
  logger::debug("Selected decoder:", config.decodeType.str());

  if (initialPopulation.size() > config.numberOfPopulations) {
    throw InvalidArgument(
        "Initial population cannot have more chromosomes than population size",
        __FUNCTION__);
  }

  // TODO save only the configuration class
  decoder = config.decoder;
  numberOfPopulations = config.numberOfPopulations;
  populationSize = config.populationSize;
  chromosomeSize = config.chromosomeLength;
  eliteSize = config.eliteCount;
  mutantsSize = config.mutantsCount;
  rhoe = config.rhoe;
  decodeType = config.decodeType;
  threadsPerBlock = config.threadsPerBlock;

  decoder->setConfiguration(&config);

  static_assert(sizeof(Chromosome<float>) == sizeof(Chromosome<unsigned>));
  const auto totalChromosomes = numberOfPopulations * populationSize;
  populationWrapper =
      decodeType.onCpu()
          ? new Chromosome<float>[totalChromosomes]
          : gpu::alloc<Chromosome<float>>(nullptr, totalChromosomes);

  // One stream for each population
  streams.resize(numberOfPopulations);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    streams[p] = gpu::allocStream();

  logger::debug("Building random generator with seed", config.seed);
  std::mt19937 rng(config.seed);
  std::uniform_int_distribution<std::mt19937::result_type> uid;
  generators.resize(numberOfPopulations);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    generators[p] = gpu::allocRandomGenerator(uid(rng));

  // FIXME generate according to this log
  // logger::debug("Use", initialPopulation.size(),
  //               "provided populations and generate",
  //               numberOfPopulations - initialPopulation.size());

  // TODO handle population with number of chromosomes != population size

  if (initialPopulation.empty()) {
    logger::debug("Building the initial populations");
    for (unsigned p = 0; p < numberOfPopulations; ++p)
      gpu::random(streams[p], generators[p], dPopulation.row(p),
                  populationSize * chromosomeSize);
  } else {
    logger::debug("Using the provided initial populations");
    assert(initialPopulation.size() == numberOfPopulations);  // FIXME
    for (unsigned p = 0; p < numberOfPopulations; ++p) {
      std::vector<float> chromosomes;
      for (unsigned i = 0; i < initialPopulation[p].size(); ++i) {
        const auto& ch = initialPopulation[p][i];
        assert(ch.size() == chromosomeSize);
        chromosomes.insert(chromosomes.end(), ch.begin(), ch.end());
      }
      gpu::copy2d(streams[p], dPopulation.row(p), chromosomes.data(),
                  chromosomes.size());
    }
  }

  updateFitness();
  logger::debug("Brkga was configured successfully");
}

box::Brkga::~Brkga() {
  for (unsigned p = 0; p < numberOfPopulations; ++p) gpu::free(generators[p]);
  for (unsigned p = 0; p < numberOfPopulations; ++p) gpu::free(streams[p]);

  if (decodeType.onCpu()) {
    delete[] populationWrapper;
  } else {
    gpu::free(nullptr, populationWrapper);
  }
}

__global__ void evolveCopyElite(float* population,
                                const float* previousPopulation,
                                const unsigned* dFitnessIdx,
                                const unsigned eliteSize,
                                const unsigned chromosomeSize) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= eliteSize * chromosomeSize) return;

  const auto permutations = tid / chromosomeSize;
  const auto geneIdx = tid % chromosomeSize;
  const auto eliteIdx = dFitnessIdx[permutations];
  population[permutations * chromosomeSize + geneIdx] =
      previousPopulation[eliteIdx * chromosomeSize + geneIdx];

  // The fitness was already sorted with dFitnessIdx.
}

__global__ void evolveMate(float* population,
                           const float* previousPopulation,
                           const unsigned* dFitnessIdx,
                           const float* randomEliteParent,
                           const float* randomParent,
                           const unsigned populationSize,
                           const unsigned eliteSize,
                           const unsigned mutantsSize,
                           const unsigned chromosomeSize,
                           const float rhoe) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (populationSize - eliteSize - mutantsSize) * chromosomeSize)
    return;

  const auto permutations = eliteSize + tid / chromosomeSize;
  const auto geneIdx = tid % chromosomeSize;

  // On rare cases, the generator will return 1 in the random parent variables.
  // Thus, we check the index and decrease it to avoid index errors.
  unsigned parentIdx;
  if (population[permutations * chromosomeSize + geneIdx] < rhoe) {
    // Elite parent
    parentIdx = (unsigned)(randomEliteParent[permutations] * eliteSize);
    if (parentIdx == eliteSize) --parentIdx;
  } else {
    // Non-elite parent
    parentIdx =
        (unsigned)(eliteSize
                   + randomParent[permutations] * (populationSize - eliteSize));
    if (parentIdx == populationSize) --parentIdx;
  }

  const auto parent = dFitnessIdx[parentIdx];
  population[permutations * chromosomeSize + geneIdx] =
      previousPopulation[parent * chromosomeSize + geneIdx];
}

void box::Brkga::evolve() {
  logger::debug("Selecting the parents for the evolution");
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    gpu::random(streams[p], generators[p], dPopulationTemp.row(p),
                populationSize * chromosomeSize);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    gpu::random(streams[p], generators[p], dRandomEliteParent.row(p),
                populationSize - mutantsSize);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    gpu::random(streams[p], generators[p], dRandomParent.row(p),
                populationSize - mutantsSize);

  logger::debug("Copying the elites to the next generation");
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    evolveCopyElite<<<gpu::blocks(eliteSize * chromosomeSize, threadsPerBlock),
                      threadsPerBlock, 0, streams[p]>>>(
        dPopulationTemp.row(p), dPopulation.row(p), dFitnessIdx.row(p),
        eliteSize, chromosomeSize);
  }
  CUDA_CHECK_LAST();

  logger::debug("Mating pairs of the population");
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    const auto blocks =
        gpu::blocks((populationSize - eliteSize - mutantsSize) * chromosomeSize,
                    threadsPerBlock);
    evolveMate<<<blocks, threadsPerBlock, 0, streams[p]>>>(
        dPopulationTemp.row(p), dPopulation.row(p), dFitnessIdx.row(p),
        dRandomEliteParent.row(p), dRandomParent.row(p), populationSize,
        eliteSize, mutantsSize, chromosomeSize, rhoe);
  }
  CUDA_CHECK_LAST();

  // The mutants were generated in the "parent selection" above.

  // Saves the new generation.
  std::swap(dPopulation, dPopulationTemp);

  updateFitness();
  logger::debug("A new generation of the population was created");
}

void box::Brkga::updateFitness() {
  logger::debug("Updating the population fitness");

  sortChromosomesGenes();

  if (decodeType.onCpu()) {
    logger::debug("Copying data to host");
    if (decodeType.chromosome()) {
      population.resize(numberOfPopulations * populationSize * chromosomeSize);
      for (unsigned p = 0; p < numberOfPopulations; ++p) {
        gpu::copy2h(streams[p],
                    population.data() + p * populationSize * chromosomeSize,
                    dPopulation.row(p), populationSize * chromosomeSize);
      }
    } else {
      permutations.resize(numberOfPopulations * populationSize
                          * chromosomeSize);
      for (unsigned p = 0; p < numberOfPopulations; ++p) {
        gpu::copy2h(streams[p],
                    permutations.data() + p * populationSize * chromosomeSize,
                    dPermutations.row(p), populationSize * chromosomeSize);
      }
    }

    fitness.resize(numberOfPopulations * populationSize);
  }

  logger::debug("Calling the decoder");
  if (decodeType.allAtOnce()) {
    logger::debug("Decoding all at once");
    syncStreams();
  }

  // FIXME this method will also decode the elites, which didn't change

  const auto n =
      (decodeType.allAtOnce() ? numberOfPopulations : 1) * populationSize;
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    if (decodeType.onCpu()) {
      gpu::sync(streams[p]);
      if (decodeType.chromosome()) {
        auto* wrap = wrapCpu(population.data(), p, n);
        logger::debug("Entering CPU-chromosome decoder");
        decoder->decode(n, wrap, fitness.data() + p * n);
      } else {
        auto* wrap = wrapCpu(permutations.data(), p, n);
        logger::debug("Entering CPU-permutation decoder");
        decoder->decode(n, wrap, fitness.data() + p * n);
      }
      logger::debug("The decoder has finished");

      logger::debug("Copying fitness back to device");
      gpu::copy2d(streams[p], dFitness.row(p), fitness.data() + p * n, n);
    } else {
      if (decodeType.chromosome()) {
        auto* wrap = wrapGpu(dPopulation.get(), p, n);
        logger::debug("Entering GPU-chromosome decoder");
        decoder->decode(streams[p], n, wrap, dFitness.row(p));
      } else {
        auto* wrap = wrapGpu(dPermutations.get(), p, n);
        logger::debug("Entering GPU-permutation decoder");
        decoder->decode(streams[p], n, wrap, dFitness.row(p));
      }
      CUDA_CHECK_LAST();
      logger::debug("The decoder kernel call has finished");
    }

    // Cannot sort all chromosomes since they come from different populations
    if (decodeType.allAtOnce()) {
      gpu::sync(streams[0]);  // To avoid sort starting before decoder
      for (unsigned q = 0; q < numberOfPopulations; ++q) {
        gpu::iota(streams[q], dFitnessIdx.row(q), populationSize);
        gpu::sortByKey(streams[q], dFitness.row(q), dFitnessIdx.row(q),
                       populationSize);
      }
      break;
    }

    gpu::iota(streams[p], dFitnessIdx.row(p), populationSize);
    gpu::sortByKey(streams[p], dFitness.row(p), dFitnessIdx.row(p),
                   populationSize);
  }

  logger::debug("Decoding step has finished");
}

template <class T>
auto box::Brkga::wrapCpu(T* pop, unsigned popId, unsigned n) -> Chromosome<T>* {
  logger::debug("Wrapping population", popId);
  pop += popId * n * chromosomeSize;
  auto* wrap = ((Chromosome<T>*)populationWrapper) + popId * n;

  for (unsigned k = 0; k < n; ++k) {
    wrap[k] = Chromosome<T>(pop, chromosomeSize, k);
  }
  return wrap;
}

template <class T>
__global__ void initWrapper(box::Chromosome<T>* dWrapper,
                            T* dPopulation,
                            unsigned columnCount,
                            unsigned n) {
  const auto k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  dWrapper[k] = box::Chromosome<T>(dPopulation, columnCount, k);
}

template <class T>
auto box::Brkga::wrapGpu(T* pop, unsigned popId, unsigned n) -> Chromosome<T>* {
  // TODO this will not work for transposed matrices with the `all` decoder
  pop += popId * n * chromosomeSize;
  auto* wrap = ((Chromosome<T>*)populationWrapper) + popId * n;

  const auto blocks = gpu::blocks(n, threadsPerBlock);
  initWrapper<<<blocks, threadsPerBlock, 0, streams[popId]>>>(
      wrap, pop, chromosomeSize, n);
  return wrap;
}

void box::Brkga::sortChromosomesGenes() {
  if (decodeType.chromosome()) return;
  logger::debug("Sorting the chromosomes for sorted decode");

  for (unsigned p = 0; p < numberOfPopulations; ++p)
    gpu::iotaMod(streams[p], dPermutations.row(p),
                 populationSize * chromosomeSize, chromosomeSize);

  // Copy to temp memory since the sort modifies the original array.
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    gpu::copy(streams[p], dPopulationTemp.row(p), dPopulation.row(p),
              populationSize * chromosomeSize);

  // FIXME We should sort each chromosome on its own thread to avoid
  //  synchonization.
  syncStreams();
  gpu::segSort(streams[0], dPopulationTemp.get(), dPermutations.get(),
               numberOfPopulations * populationSize, chromosomeSize);
  gpu::sync(streams[0]);
}

/**
 * Exchanges the best chromosomes between the populations.
 *
 * This method replaces the worsts chromosomes by the elite ones of the other
 * populations.
 *
 * @param population The population to exchange.
 * @param chromosomeSize To size of the chromosomes.
 * @param populationSize The number of chromosomes on each population.
 * @param numberOfPopulations The nuber of populations.
 * @param dFitnessIdx The order of the chromosomes, increasing by fitness.
 * @param count The number of elites to copy.
 */
__global__ void deviceExchangeElite(float* population,
                                    unsigned chromosomeSize,
                                    unsigned populationSize,
                                    unsigned numberOfPopulations,
                                    unsigned* dFitnessIdx,
                                    unsigned count) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chromosomeSize) return;

  for (unsigned i = 0; i < numberOfPopulations; ++i)
    for (unsigned j = 0; j < numberOfPopulations; ++j)
      if (i != j)  // don't duplicate chromosomes
        for (unsigned k = 0; k < count; ++k) {
          // Position of the bad chromosome to be replaced
          // Note that `j < i` is due the condition above
          // Over the iterations of each population, `p` will be:
          //  `size - 1`, `size - 2`, ...
          const auto p = populationSize - (i - (j < i)) * count - k - 1;

          // Global position of source/destination chromosomes
          const auto src =
              i * populationSize + dFitnessIdx[i * populationSize + k];
          const auto dest =
              j * populationSize + dFitnessIdx[j * populationSize + p];

          // Copy the chromosome
          population[dest * chromosomeSize + tid] =
              population[src * chromosomeSize + tid];
        }
}

void box::Brkga::exchangeElite(unsigned count) {
  logger::debug("Sharing the", count, "best chromosomes of each one of the",
                numberOfPopulations, "populations");

  InvalidArgument::range(Arg<unsigned>(count, "exchange count"),
                         Arg<unsigned>(1),
                         Arg<unsigned>(populationSize / numberOfPopulations,
                                       "population / #populations"),
                         3 /* closed */, __FUNCTION__);

  syncStreams();

  const auto blocks = gpu::blocks(chromosomeSize, threadsPerBlock);
  deviceExchangeElite<<<blocks, threadsPerBlock>>>(
      dPopulation.get(), chromosomeSize, populationSize, numberOfPopulations,
      dFitnessIdx.get(), count);
  CUDA_CHECK_LAST();
  gpu::sync();

  updateFitness();
}

std::vector<bool> box::Brkga::compareChromosomes(
    const std::vector<PathRelinkPair>& ids,
    const FilterBase& cmp) {
  std::vector<bool> equal;
  equal.reserve(ids.size());

  std::vector<float> ch1(chromosomeSize);
  std::vector<float> ch2(chromosomeSize);
  for (const auto& id : ids) {
    gpu::copy2h(nullptr, ch1.data(),
                dPopulation.row(id.basePopulationId)
                    + id.baseChromosomeId * chromosomeSize,
                chromosomeSize);
    gpu::copy2h(nullptr, ch2.data(),
                dPopulation.row(id.guidePopulationId)
                    + id.guideChromosomeId * chromosomeSize,
                chromosomeSize);

    equal.push_back(cmp(Chromosome<float>(ch1.data(), chromosomeSize, 0),
                        Chromosome<float>(ch2.data(), chromosomeSize, 0)));
  }

  return equal;
}

std::vector<float> box::Brkga::getBestChromosome() {
  auto bestIdx = getBest();
  auto bestPopulation = bestIdx.first;
  auto bestChromosome = bestIdx.second;

  std::vector<float> best(chromosomeSize);
  gpu::copy2h(nullptr, best.data(),
              dPopulation.row(bestPopulation) + bestChromosome * chromosomeSize,
              chromosomeSize);

  return best;
}

std::vector<unsigned> box::Brkga::getBestPermutation() {
  if (decodeType.chromosome())
    throw InvalidArgument("The chromosome decoder has no permutation",
                          __FUNCTION__);

  auto bestIdx = getBest();
  auto bestPopulation = bestIdx.first;
  auto bestChromosome = bestIdx.second;

  // Copy the best chromosome
  std::vector<unsigned> best(chromosomeSize);
  gpu::copy2h(
      streams[0], best.data(),
      dPermutations.row(bestPopulation) + bestChromosome * chromosomeSize,
      chromosomeSize);
  gpu::sync(streams[0]);

  return best;
}

float box::Brkga::getBestFitness() {
  auto bestPopulation = getBest().first;
  float bestFitness = -1;
  gpu::copy2h(streams[0], &bestFitness, dFitness.row(bestPopulation), 1);
  gpu::sync(streams[0]);
  return bestFitness;
}

std::pair<unsigned, unsigned> box::Brkga::getBest() {
  logger::debug("Searching for the best population/chromosome");

  const unsigned chromosomesToCopy = 1;
  std::vector<float> bestFitness(numberOfPopulations, -1);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    gpu::copy2h(streams[p], &bestFitness[p], dFitness.row(p),
                chromosomesToCopy);

  // Find the best population
  unsigned bestPopulation = 0;
  for (unsigned p = 1; p < numberOfPopulations; ++p) {
    gpu::sync(streams[p]);
    if (bestFitness[p] < bestFitness[bestPopulation]) bestPopulation = p;
  }

  // Get the index of the best chromosome
  unsigned bestChromosome = (unsigned)-1;
  gpu::copy2h(streams[0], &bestChromosome, dFitnessIdx.row(bestPopulation),
              chromosomesToCopy);
  gpu::sync(streams[0]);

  logger::debug("Best fitness:", bestFitness[bestPopulation], "on population",
                bestPopulation, "and chromosome", bestChromosome);

  return {bestPopulation, bestChromosome};
}

std::vector<DecodedChromosome> box::Brkga::getPopulation(unsigned p) {
  std::vector<float> hFitness(populationSize);
  box::gpu::copy2h(streams[p], hFitness.data(), dFitness.row(p),
                   populationSize);

  std::vector<unsigned> hFitnessIdx(populationSize);
  box::gpu::copy2h(streams[p], hFitnessIdx.data(), dFitnessIdx.row(p),
                   populationSize);

  std::vector<float> hChromosomes(populationSize * chromosomeSize);
  box::gpu::copy2h(streams[p], hChromosomes.data(), dPopulation.row(p),
                   populationSize * chromosomeSize);

  std::vector<DecodedChromosome> decoded;
  for (unsigned i = 0; i < populationSize; ++i) {
    const auto ptr = hChromosomes.begin() + hFitnessIdx[i] * chromosomeSize;
    decoded.push_back(DecodedChromosome{
        fitness[i], std::vector<float>(ptr, ptr + chromosomeSize)});
  }

  return decoded;
}

void box::Brkga::syncStreams() {
  for (unsigned p = 0; p < numberOfPopulations; ++p) gpu::sync(streams[p]);
}
