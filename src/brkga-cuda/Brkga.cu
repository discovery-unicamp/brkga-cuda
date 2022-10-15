#include "Brkga.hpp"
#include "BrkgaConfiguration.hpp"
#include "Comparator.hpp"
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
    const BrkgaConfiguration& _config,
    const std::vector<std::vector<std::vector<float>>>& initialPopulation)
    : config(_config),
      dPopulation(config.numberOfPopulations(),
                  config.populationSize() * config.chromosomeLength()),
      dPopulationTemp(config.numberOfPopulations(),
                      config.populationSize() * config.chromosomeLength()),
      populationWrapper(nullptr),
      dFitness(config.numberOfPopulations(), config.populationSize()),
      dFitnessIdx(config.numberOfPopulations(), config.populationSize()),
      dPermutations(config.numberOfPopulations(),
                    config.populationSize() * config.chromosomeLength()),
      dRandomEliteParent(config.numberOfPopulations(), config.populationSize()),
      dRandomParent(config.numberOfPopulations(), config.populationSize()) {
  CUDA_CHECK_LAST();
  logger::debug("Building BoxBrkga with", config.numberOfPopulations(),
                "populations, each with", config.populationSize(),
                "chromosomes of length", config.chromosomeLength());
  logger::debug("Selected decoder:", config.decodeType().str());

  if (initialPopulation.size() > config.numberOfPopulations()) {
    throw InvalidArgument(
        "Initial population cannot have more chromosomes than population size",
        __FUNCTION__);
  }

  config.decoder()->setConfiguration(&config);

  static_assert(sizeof(Chromosome<float>) == sizeof(Chromosome<unsigned>));
  const auto totalChromosomes =
      config.numberOfPopulations() * config.populationSize();
  populationWrapper =
      config.decodeType().onCpu()
          ? new Chromosome<float>[totalChromosomes]
          : gpu::alloc<Chromosome<float>>(nullptr, totalChromosomes);

  // One stream for each population
  streams.resize(config.numberOfPopulations());
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p)
    streams[p] = gpu::allocStream();

  logger::debug("Building random generator with seed", config.seed());
  std::mt19937 rng(config.seed());
  std::uniform_int_distribution<std::mt19937::result_type> uid;
  generators.resize(config.numberOfPopulations());
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p)
    generators[p] = gpu::allocRandomGenerator(uid(rng));

  // FIXME generate according to this log
  // logger::debug("Use", initialPopulation.size(),
  //               "provided populations and generate",
  //               config.numberOfPopulations() - initialPopulation.size());

  // TODO handle population with number of chromosomes != population size

  if (initialPopulation.empty()) {
    logger::debug("Building the initial populations");
    for (unsigned p = 0; p < config.numberOfPopulations(); ++p)
      gpu::random(streams[p], generators[p], dPopulation.row(p),
                  config.populationSize() * config.chromosomeLength());
  } else {
    logger::debug("Using the provided initial populations");
    assert(initialPopulation.size() == config.numberOfPopulations());  // FIXME
    for (unsigned p = 0; p < config.numberOfPopulations(); ++p) {
      std::vector<float> chromosomes;
      for (unsigned i = 0; i < initialPopulation[p].size(); ++i) {
        const auto& ch = initialPopulation[p][i];
        assert(ch.size() == config.chromosomeLength());
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
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p)
    gpu::free(generators[p]);
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p)
    gpu::free(streams[p]);

  if (config.decodeType().onCpu()) {
    delete[] populationWrapper;
  } else {
    gpu::free(nullptr, populationWrapper);
  }
}

__global__ void evolveCopyElite(float* population,
                                const float* previousPopulation,
                                const unsigned* dFitnessIdx,
                                const unsigned numberOfElites,
                                const unsigned chromosomeLength) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfElites * chromosomeLength) return;

  const auto permutations = tid / chromosomeLength;
  const auto geneIdx = tid % chromosomeLength;
  const auto eliteIdx = dFitnessIdx[permutations];
  population[permutations * chromosomeLength + geneIdx] =
      previousPopulation[eliteIdx * chromosomeLength + geneIdx];

  // The fitness was already sorted with dFitnessIdx.
}

__global__ void evolveMate(float* population,
                           const float* previousPopulation,
                           const unsigned* dFitnessIdx,
                           const float* randomEliteParent,
                           const float* randomParent,
                           const unsigned populationSize,
                           const unsigned numberOfElites,
                           const unsigned numberOfMutants,
                           const unsigned chromosomeLength,
                           const float rhoe) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid
      >= (populationSize - numberOfElites - numberOfMutants) * chromosomeLength)
    return;

  const auto permutations = numberOfElites + tid / chromosomeLength;
  const auto geneIdx = tid % chromosomeLength;

  // On rare cases, the generator will return 1 in the random parent variables.
  // Thus, we check the index and decrease it to avoid index errors.
  unsigned parentIdx;
  if (population[permutations * chromosomeLength + geneIdx] < rhoe) {
    // Elite parent
    parentIdx = (unsigned)(randomEliteParent[permutations] * numberOfElites);
    if (parentIdx == numberOfElites) --parentIdx;
  } else {
    // Non-elite parent
    parentIdx = (unsigned)(numberOfElites
                           + randomParent[permutations]
                                 * (populationSize - numberOfElites));
    if (parentIdx == populationSize) --parentIdx;
  }

  const auto parent = dFitnessIdx[parentIdx];
  population[permutations * chromosomeLength + geneIdx] =
      previousPopulation[parent * chromosomeLength + geneIdx];
}

void box::Brkga::evolve() {
  logger::debug("Selecting the parents for the evolution");
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p)
    gpu::random(streams[p], generators[p], dPopulationTemp.row(p),
                config.populationSize() * config.chromosomeLength());
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p)
    gpu::random(streams[p], generators[p], dRandomEliteParent.row(p),
                config.populationSize() - config.numberOfMutants());
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p)
    gpu::random(streams[p], generators[p], dRandomParent.row(p),
                config.populationSize() - config.numberOfMutants());

  logger::debug("Copying the elites to the next generation");
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p) {
    evolveCopyElite<<<gpu::blocks(
                          config.numberOfElites() * config.chromosomeLength(),
                          config.gpuThreads()),
                      config.gpuThreads(), 0, streams[p]>>>(
        dPopulationTemp.row(p), dPopulation.row(p), dFitnessIdx.row(p),
        config.numberOfElites(), config.chromosomeLength());
  }
  CUDA_CHECK_LAST();

  logger::debug("Mating pairs of the population");
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p) {
    const auto blocks =
        gpu::blocks((config.populationSize() - config.numberOfElites()
                     - config.numberOfMutants())
                        * config.chromosomeLength(),
                    config.gpuThreads());
    evolveMate<<<blocks, config.gpuThreads(), 0, streams[p]>>>(
        dPopulationTemp.row(p), dPopulation.row(p), dFitnessIdx.row(p),
        dRandomEliteParent.row(p), dRandomParent.row(p),
        config.populationSize(), config.numberOfElites(),
        config.numberOfMutants(), config.chromosomeLength(), config.rhoe());
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

  if (config.decodeType().onCpu()) {
    logger::debug("Copying data to host");
    if (config.decodeType().chromosome()) {
      population.resize(config.numberOfPopulations() * config.populationSize()
                        * config.chromosomeLength());
      for (unsigned p = 0; p < config.numberOfPopulations(); ++p) {
        gpu::copy2h(
            streams[p],
            population.data()
                + p * config.populationSize() * config.chromosomeLength(),
            dPopulation.row(p),
            config.populationSize() * config.chromosomeLength());
      }
    } else {
      permutations.resize(config.numberOfPopulations() * config.populationSize()
                          * config.chromosomeLength());
      for (unsigned p = 0; p < config.numberOfPopulations(); ++p) {
        gpu::copy2h(
            streams[p],
            permutations.data()
                + p * config.populationSize() * config.chromosomeLength(),
            dPermutations.row(p),
            config.populationSize() * config.chromosomeLength());
      }
    }

    fitness.resize(config.numberOfPopulations() * config.populationSize());
  }

  logger::debug("Calling the config.decoder()");
  if (config.decodeType().allAtOnce()) {
    logger::debug("Decoding all at once");
    syncStreams();
  }

  // FIXME this method will also decode the elites, which didn't change

  const auto n =
      (config.decodeType().allAtOnce() ? config.numberOfPopulations() : 1)
      * config.populationSize();
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p) {
    if (config.decodeType().onCpu()) {
      gpu::sync(streams[p]);
      if (config.decodeType().chromosome()) {
        auto* wrap = wrapCpu(population.data(), p, n);
        logger::debug("Entering CPU-chromosome config.decoder()");
        config.decoder()->decode(n, wrap, fitness.data() + p * n);
      } else {
        auto* wrap = wrapCpu(permutations.data(), p, n);
        logger::debug("Entering CPU-permutation config.decoder()");
        config.decoder()->decode(n, wrap, fitness.data() + p * n);
      }
      logger::debug("The config.decoder() has finished");

      logger::debug("Copying fitness back to device");
      gpu::copy2d(streams[p], dFitness.row(p), fitness.data() + p * n, n);
    } else {
      if (config.decodeType().chromosome()) {
        auto* wrap = wrapGpu(dPopulation.get(), p, n);
        logger::debug("Entering GPU-chromosome config.decoder()");
        config.decoder()->decode(streams[p], n, wrap, dFitness.row(p));
      } else {
        auto* wrap = wrapGpu(dPermutations.get(), p, n);
        logger::debug("Entering GPU-permutation config.decoder()");
        config.decoder()->decode(streams[p], n, wrap, dFitness.row(p));
      }
      CUDA_CHECK_LAST();
      logger::debug("The config.decoder() kernel call has finished");
    }

    // Cannot sort all chromosomes since they come from different populations
    if (config.decodeType().allAtOnce()) {
      gpu::sync(streams[0]);  // To avoid sort starting before config.decoder()
      for (unsigned q = 0; q < config.numberOfPopulations(); ++q) {
        gpu::iota(streams[q], dFitnessIdx.row(q), config.populationSize());
        gpu::sortByKey(streams[q], dFitness.row(q), dFitnessIdx.row(q),
                       config.populationSize());
      }
      break;
    }

    gpu::iota(streams[p], dFitnessIdx.row(p), config.populationSize());
    gpu::sortByKey(streams[p], dFitness.row(p), dFitnessIdx.row(p),
                   config.populationSize());
  }

  logger::debug("Decoding step has finished");
}

template <class T>
auto box::Brkga::wrapCpu(T* pop, unsigned popId, unsigned n) -> Chromosome<T>* {
  logger::debug("Wrapping population", popId);
  pop += popId * n * config.chromosomeLength();
  auto* wrap = ((Chromosome<T>*)populationWrapper) + popId * n;

  for (unsigned k = 0; k < n; ++k) {
    wrap[k] = Chromosome<T>(pop, config.chromosomeLength(), k);
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
  // TODO this will not work for transposed matrices with the `all`
  // config.decoder()
  pop += popId * n * config.chromosomeLength();
  auto* wrap = ((Chromosome<T>*)populationWrapper) + popId * n;

  const auto blocks = gpu::blocks(n, config.gpuThreads());
  initWrapper<<<blocks, config.gpuThreads(), 0, streams[popId]>>>(
      wrap, pop, config.chromosomeLength(), n);
  return wrap;
}

void box::Brkga::sortChromosomesGenes() {
  if (config.decodeType().chromosome()) return;
  logger::debug("Sorting the chromosomes for sorted decode");

  for (unsigned p = 0; p < config.numberOfPopulations(); ++p)
    gpu::iotaMod(streams[p], dPermutations.row(p),
                 config.populationSize() * config.chromosomeLength(),
                 config.chromosomeLength());

  // Copy to temp memory since the sort modifies the original array.
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p)
    gpu::copy(streams[p], dPopulationTemp.row(p), dPopulation.row(p),
              config.populationSize() * config.chromosomeLength());

  // FIXME We should sort each chromosome on its own thread to avoid
  //  synchonization.
  syncStreams();
  gpu::segSort(streams[0], dPopulationTemp.get(), dPermutations.get(),
               config.numberOfPopulations() * config.populationSize(),
               config.chromosomeLength());
  gpu::sync(streams[0]);
}

/**
 * Exchanges the best chromosomes between the populations.
 *
 * This method replaces the worsts chromosomes by the elite ones of the other
 * populations.
 *
 * @param population The population to exchange.
 * @param chromosomeLength To size of the chromosomes.
 * @param populationSize The number of chromosomes on each population.
 * @param numberOfPopulations The nuber of populations.
 * @param dFitnessIdx The order of the chromosomes, increasing by fitness.
 * @param count The number of elites to copy.
 */
__global__ void deviceExchangeElite(float* population,
                                    unsigned chromosomeLength,
                                    unsigned populationSize,
                                    unsigned numberOfPopulations,
                                    unsigned* dFitnessIdx,
                                    unsigned count) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chromosomeLength) return;

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
          population[dest * chromosomeLength + tid] =
              population[src * chromosomeLength + tid];
        }
}

void box::Brkga::exchangeElites() {
  if (config.numberOfElitesToExchange() == 0
      || config.numberOfPopulations() == 1) {
    logger::warning("Ignoring operation: exchange",
                    config.numberOfElitesToExchange(), "elite(s) between",
                    config.numberOfPopulations(), "population(s)");
    return;
  }
  logger::debug("Sharing the", config.numberOfElitesToExchange(),
                "best chromosomes of each one of the",
                config.numberOfPopulations(), "populations");

  syncStreams();

  const auto blocks =
      gpu::blocks(config.chromosomeLength(), config.gpuThreads());
  deviceExchangeElite<<<blocks, config.gpuThreads()>>>(
      dPopulation.get(), config.chromosomeLength(), config.populationSize(),
      config.numberOfPopulations(), dFitnessIdx.get(),
      config.numberOfElitesToExchange());
  CUDA_CHECK_LAST();
  gpu::sync();

  updateFitness();
}

std::vector<float> box::Brkga::getBestChromosome() {
  auto bestIdx = getBest();
  auto bestPopulation = bestIdx.first;
  auto bestChromosome = bestIdx.second;

  std::vector<float> best(config.chromosomeLength());
  gpu::copy2h(nullptr, best.data(),
              dPopulation.row(bestPopulation)
                  + bestChromosome * config.chromosomeLength(),
              config.chromosomeLength());

  return best;
}

std::vector<unsigned> box::Brkga::getBestPermutation() {
  if (config.decodeType().chromosome())
    throw InvalidArgument("The chromosome config.decoder() has no permutation",
                          __FUNCTION__);

  auto bestIdx = getBest();
  auto bestPopulation = bestIdx.first;
  auto bestChromosome = bestIdx.second;

  // Copy the best chromosome
  std::vector<unsigned> best(config.chromosomeLength());
  gpu::copy2h(streams[0], best.data(),
              dPermutations.row(bestPopulation)
                  + bestChromosome * config.chromosomeLength(),
              config.chromosomeLength());
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
  std::vector<float> bestFitness(config.numberOfPopulations(), -1);
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p)
    gpu::copy2h(streams[p], &bestFitness[p], dFitness.row(p),
                chromosomesToCopy);

  // Find the best population
  unsigned bestPopulation = 0;
  for (unsigned p = 1; p < config.numberOfPopulations(); ++p) {
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
  std::vector<float> hFitness(config.populationSize());
  box::gpu::copy2h(streams[p], hFitness.data(), dFitness.row(p),
                   config.populationSize());

  std::vector<unsigned> hFitnessIdx(config.populationSize());
  box::gpu::copy2h(streams[p], hFitnessIdx.data(), dFitnessIdx.row(p),
                   config.populationSize());

  std::vector<float> hChromosomes(config.populationSize()
                                  * config.chromosomeLength());
  box::gpu::copy2h(streams[p], hChromosomes.data(), dPopulation.row(p),
                   config.populationSize() * config.chromosomeLength());

  std::vector<DecodedChromosome> decoded;
  for (unsigned i = 0; i < config.populationSize(); ++i) {
    const auto ptr =
        hChromosomes.begin() + hFitnessIdx[i] * config.chromosomeLength();
    decoded.push_back(DecodedChromosome{
        fitness[i], std::vector<float>(ptr, ptr + config.chromosomeLength())});
  }

  return decoded;
}

void box::Brkga::syncStreams() {
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p)
    gpu::sync(streams[p]);
}
