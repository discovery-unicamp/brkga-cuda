#include "BasicTypes.hpp"
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

#ifndef NDEBUG
#warning Compiling without the NDEBUG flag makes the algorithm VERY SLOW!
#endif  // NDEBUG

namespace box {
/// Initialize @p n @p states with given @p seed as suggested in:
/// https://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-overview
__global__ void initializeCurandStates(uint n,
                                       curandState_t* states,
                                       const uint seed) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  curand_init(seed, tid, 0, &states[tid]);
}

Brkga::Brkga(
    const BrkgaConfiguration& _config,
    const std::vector<std::vector<std::vector<Gene>>>& initialPopulation)
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
      dParent(config.numberOfPopulations(),
              config.populationSize() * config.numberOfParents()),
      dRandomStates(config.numberOfPopulations(), config.populationSize()) {
  if (initialPopulation.size() > config.numberOfPopulations()) {
    logger::warning(
        "Ignoring", initialPopulation.size() - config.numberOfPopulations(),
        "populations in initial population since it is greater than the number "
        "of populations in the configuration",
        format(Separator(""), "(", config.numberOfPopulations(), ")"));
  }
  for (uint p = 0; p < initialPopulation.size(); ++p) {
    if (initialPopulation[p].size() > config.populationSize()) {
      logger::warning("Ignoring",
                      initialPopulation[p].size() - config.populationSize(),
                      "chromosomes in the initial population", p,
                      "since it is greater than the number of chromosomes in "
                      "the configuration",
                      format(Separator(""), "(", config.populationSize(), ")"));
    }
    for (uint k = 0; k < initialPopulation[p].size(); ++k) {
      const auto& chromosome = initialPopulation[p][k];
      InvalidArgument::diff(
          Arg<uint>((uint)chromosome.size(), format("length of the chromosome",
                                                    k, "in the population", p)),
          Arg<uint>(config.chromosomeLength(), "|chromosome|"), BOX_FUNCTION);
    }
  }

  config.decoder()->setConfiguration(&config);

  static_assert(sizeof(Chromosome<Gene>) == sizeof(Chromosome<GeneIndex>));
  const auto totalChromosomes =
      config.numberOfPopulations() * config.populationSize();
  populationWrapper =
      config.decodeType().onCpu()
          ? new Chromosome<Gene>[totalChromosomes]
          : gpu::alloc<Chromosome<Gene>>(nullptr, totalChromosomes);

  // One stream for each population
  streams.resize(config.numberOfPopulations());
  for (uint p = 0; p < config.numberOfPopulations(); ++p)
    streams[p] = gpu::allocStream();

  logger::debug("Building random generator with seed", config.seed());
  std::mt19937 rng(config.seed());
  std::uniform_int_distribution<std::mt19937::result_type> uid;
  generators.resize(config.numberOfPopulations());
  for (uint p = 0; p < config.numberOfPopulations(); ++p)
    generators[p] = gpu::allocRandomGenerator(uid(rng));

  // FIXME generate according to this log
  // logger::debug("Use", initialPopulation.size(),
  //               "provided populations and generate",
  //               config.numberOfPopulations() - initialPopulation.size());

  // TODO handle population with number of chromosomes != population size

  if (initialPopulation.empty()) {
    logger::debug("Building the initial populations");
    for (uint p = 0; p < config.numberOfPopulations(); ++p)
      gpu::random(streams[p], generators[p], dPopulation.row(p),
                  config.populationSize() * config.chromosomeLength());
  } else {
    logger::debug("Using the provided initial populations");
    assert(initialPopulation.size() == config.numberOfPopulations());  // FIXME
    for (uint p = 0; p < config.numberOfPopulations(); ++p) {
      std::vector<Gene> chromosomes;
      for (uint i = 0; i < initialPopulation[p].size(); ++i) {
        const auto& ch = initialPopulation[p][i];
        assert(ch.size() == config.chromosomeLength());
        chromosomes.insert(chromosomes.end(), ch.begin(), ch.end());
      }
      gpu::copy2d(streams[p], dPopulation.row(p), chromosomes.data(),
                  chromosomes.size());
    }
  }

  initializeCurandStates<<<gpu::blocks(totalChromosomes, config.gpuThreads()),
                           config.gpuThreads()>>>(
      totalChromosomes, dRandomStates.get(), config.seed());

  updateFitness();
  logger::debug("Brkga was configured successfully");
}

Brkga::~Brkga() {
  for (uint p = 0; p < config.numberOfPopulations(); ++p)
    gpu::free(generators[p]);
  for (uint p = 0; p < config.numberOfPopulations(); ++p) gpu::free(streams[p]);

  if (config.decodeType().onCpu()) {
    delete[] populationWrapper;
  } else {
    gpu::free(nullptr, populationWrapper);
  }
}

/// @brief Select k distinct values from range [a, b)
/// @param sample The output array of select values
/// @param k The number of values to select
/// @param a The lowest possible value, inclusive
/// @param b The highest possible value, exclusive
/// @param state The RNG state
/// @warning Undefined behaviour if k > b - a -- assert fails if enabled
__device__ void rangeSample(uint* sample,
                            uint k,
                            uint a,
                            uint b,
                            curandState_t* state) {
  assert(k <= b - a);
  b -= a;
  for (uint i = 0; i < k; ++i) {
    const auto r = curand_uniform(state);
    auto x = (uint)ceilf(r * (b - i)) - 1 + a;
    assert(x >= a);
    uint j;
    for (j = 0; j < i && x >= sample[j]; ++j) ++x;
    assert(x < a + b);
    for (j = i; j != 0 && x < sample[j - 1]; --j) sample[j] = sample[j - 1];
    sample[j] = x;
  }
}

__global__ void selectParents(uint* dParent,
                              const uint n,
                              const uint numberOfParents,
                              const uint numberOfEliteParents,
                              const uint populationSize,
                              const uint numberOfElites,
                              curandState_t* state) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  const auto nonEliteParents = numberOfParents - numberOfEliteParents;
  rangeSample(dParent + tid * numberOfParents, numberOfEliteParents, 0,
              numberOfElites, &state[tid]);
  rangeSample(dParent + tid * numberOfParents + numberOfEliteParents,
              nonEliteParents, numberOfElites, populationSize, &state[tid]);

#ifndef NDEBUG
  const auto* start = dParent + tid * numberOfParents;
  for (uint i = 1; i < numberOfParents; ++i) assert(start[i] > start[i - 1]);
  assert(start[numberOfEliteParents - 1] < numberOfElites);
  assert(start[numberOfParents - 1] < populationSize);
#endif  // NDEBUG
}

__global__ void evolveCopyElite(Gene* population,
                                const Gene* previousPopulation,
                                const uint* dFitnessIdx,
                                const uint numberOfElites,
                                const uint chromosomeLength) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfElites * chromosomeLength) return;

  const auto chromosome = tid / chromosomeLength;
  const auto gene = tid % chromosomeLength;
  const auto eliteIdx = dFitnessIdx[chromosome];
  population[chromosome * chromosomeLength + gene] =
      previousPopulation[eliteIdx * chromosomeLength + gene];

  // The fitness was already sorted with dFitnessIdx.
}

// TODO test if it is better to process many genes in the same thread
__global__ void evolveMate(Gene* population,
                           const Gene* previousPopulation,
                           const uint* dFitnessIdx,
                           const uint* dParent,
                           const uint numberOfParents,
                           const float* dBias,
                           const uint populationSize,
                           const uint numberOfElites,
                           const uint numberOfMutants,
                           const uint chromosomeLength) {
  extern __shared__ char sharedMemory[];

  auto* bias = (float*)sharedMemory;
  for (uint i = threadIdx.x; i < numberOfParents; i += blockDim.x)
    bias[i] = dBias[i];

  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid
      >= (populationSize - numberOfElites - numberOfMutants) * chromosomeLength)
    return;

  __syncthreads();  // sync to ensure bias was initialized

  const auto chromosome = numberOfElites + tid / chromosomeLength;
  assert(chromosome < populationSize);
  const auto gene = tid % chromosomeLength;
  assert(bias[numberOfParents - 1] > 0);
  const auto toss = population[chromosome * chromosomeLength + gene]
                    * bias[numberOfParents - 1];
  assert(toss > 0);  // curand doesn't return 0.0

  uint parentIdx = 0;
  while (toss > bias[parentIdx]) {
    assert(parentIdx < numberOfParents);
    ++parentIdx;
  }

  assert(dParent[chromosome * numberOfParents + parentIdx]
         < (parentIdx < 1 ? numberOfElites : populationSize));
  parentIdx = dParent[chromosome * numberOfParents + parentIdx];
  assert(parentIdx < populationSize);
  parentIdx = dFitnessIdx[parentIdx];
  assert(parentIdx < populationSize);
  population[chromosome * chromosomeLength + gene] =
      previousPopulation[parentIdx * chromosomeLength + gene];
}

void Brkga::evolve() {
  logger::debug("Preparing to perform evolution");
  // Used as toss in the roulette.
  for (uint p = 0; p < config.numberOfPopulations(); ++p) {
    gpu::random(streams[p], generators[p], dPopulationTemp.row(p),
                config.populationSize() * config.chromosomeLength());
  }

  // Select parents for crossover.
  // TODO generate only for the ones that will be updated, i.e., ignore elites
  //   and mutants.
  for (uint p = 0; p < config.numberOfPopulations(); ++p) {
    selectParents<<<gpu::blocks(config.populationSize(), config.gpuThreads()),
                    config.gpuThreads(), 0, streams[p]>>>(
        dParent.row(p), config.populationSize(), config.numberOfParents(),
        config.numberOfEliteParents(), config.populationSize(),
        config.numberOfElites(), dRandomStates.row(p));
  }
  CUDA_CHECK_LAST();

  logger::debug("Copying the elites to the next generation");
  for (uint p = 0; p < config.numberOfPopulations(); ++p) {
    evolveCopyElite<<<gpu::blocks(
                          config.numberOfElites() * config.chromosomeLength(),
                          config.gpuThreads()),
                      config.gpuThreads(), 0, streams[p]>>>(
        dPopulationTemp.row(p), dPopulation.row(p), dFitnessIdx.row(p),
        config.numberOfElites(), config.chromosomeLength());
  }
  CUDA_CHECK_LAST();

  logger::debug("Mating pairs of the population");
  // Copy bias every time since it may be outdated
  auto* dBias = gpu::alloc<float>(nullptr, config.numberOfParents());
  gpu::copy2d(nullptr, dBias, config.bias().data(), config.numberOfParents());

  for (uint p = 0; p < config.numberOfPopulations(); ++p) {
    const auto blocks =
        gpu::blocks((config.populationSize() - config.numberOfElites()
                     - config.numberOfMutants())
                        * config.chromosomeLength(),
                    config.gpuThreads());
    const auto sharedMemory = config.numberOfParents() * sizeof(float);
    evolveMate<<<blocks, config.gpuThreads(), sharedMemory, streams[p]>>>(
        dPopulationTemp.row(p), dPopulation.row(p), dFitnessIdx.row(p),
        dParent.row(p), config.numberOfParents(), dBias,
        config.populationSize(), config.numberOfElites(),
        config.numberOfMutants(), config.chromosomeLength());
  }
  CUDA_CHECK_LAST();
  gpu::free(nullptr, dBias);

  // The mutants were generated in the "parent selection" above.

  // Saves the new generation.
  std::swap(dPopulation, dPopulationTemp);

  updateFitness();
  logger::debug("A new generation of the population was created");
}

void Brkga::updateFitness() {
  logger::debug("Updating the population fitness");

  if (!config.decodeType().chromosome()) sortChromosomesGenes();

  if (config.decodeType().onCpu()) {
    logger::debug("Copying data to host");
    if (config.decodeType().chromosome()) {
      population.resize(config.numberOfPopulations() * config.populationSize()
                        * config.chromosomeLength());
      for (uint p = 0; p < config.numberOfPopulations(); ++p) {
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
      for (uint p = 0; p < config.numberOfPopulations(); ++p) {
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

  logger::debug("Calling the decoder");
  if (config.decodeType().allAtOnce()) {
    logger::debug("Decoding all at once");
    syncStreams();
  }

  // FIXME this method will also decode the elites, which didn't change

  const auto n =
      (config.decodeType().allAtOnce() ? config.numberOfPopulations() : 1)
      * config.populationSize();
  for (uint p = 0; p < config.numberOfPopulations(); ++p) {
    if (config.decodeType().onCpu()) {
      gpu::sync(streams[p]);
      if (config.decodeType().chromosome()) {
        auto* wrap = wrapCpu(population.data(), p, n);
        logger::debug("Entering CPU-chromosome decoder");
        config.decoder()->decode(n, wrap, fitness.data() + p * n);
      } else {
        auto* wrap = wrapCpu(permutations.data(), p, n);
        logger::debug("Entering CPU-permutation decoder");
        config.decoder()->decode(n, wrap, fitness.data() + p * n);
      }
      logger::debug("The decoder has finished");

      logger::debug("Copying fitness back to device");
      gpu::copy2d(streams[p], dFitness.row(p), fitness.data() + p * n, n);
    } else {
      if (config.decodeType().chromosome()) {
        auto* wrap = wrapGpu(dPopulation.get(), p, n);
        logger::debug("Entering GPU-chromosome decoder");
        config.decoder()->decode(streams[p], n, wrap, dFitness.row(p));
      } else {
        auto* wrap = wrapGpu(dPermutations.get(), p, n);
        logger::debug("Entering GPU-permutation decoder");
        config.decoder()->decode(streams[p], n, wrap, dFitness.row(p));
      }
      CUDA_CHECK_LAST();
      logger::debug("The decoder kernel call has finished");
    }

    // Cannot sort all chromosomes since they come from different populations
    if (config.decodeType().allAtOnce()) {
      gpu::sync(streams[0]);  // To avoid sort starting before decoder
      for (uint q = 0; q < config.numberOfPopulations(); ++q) {
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
auto Brkga::wrapCpu(T* pop, uint popId, uint n) -> Chromosome<T>* {
  logger::debug("Wrapping population", popId);
  pop += popId * n * config.chromosomeLength();
  auto* wrap = ((Chromosome<T>*)populationWrapper) + popId * n;

  for (uint k = 0; k < n; ++k) {
    wrap[k] = Chromosome<T>(pop, config.chromosomeLength(), k);
  }
  return wrap;
}

template <class T>
__global__ void initWrapper(Chromosome<T>* dWrapper,
                            T* dPopulation,
                            uint columnCount,
                            uint n) {
  const auto k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  dWrapper[k] = Chromosome<T>(dPopulation, columnCount, k);
}

template <class T>
auto Brkga::wrapGpu(T* pop, uint popId, uint n) -> Chromosome<T>* {
  // TODO this will not work for transposed matrices with the `all` decoder
  pop += popId * n * config.chromosomeLength();
  auto* wrap = ((Chromosome<T>*)populationWrapper) + popId * n;

  const auto blocks = gpu::blocks(n, config.gpuThreads());
  initWrapper<<<blocks, config.gpuThreads(), 0, streams[popId]>>>(
      wrap, pop, config.chromosomeLength(), n);
  return wrap;
}

void Brkga::sortChromosomesGenes() {
  logger::debug("Sorting the chromosomes for sorted decode");

  for (uint p = 0; p < config.numberOfPopulations(); ++p)
    gpu::iotaMod(streams[p], dPermutations.row(p),
                 config.populationSize() * config.chromosomeLength(),
                 config.chromosomeLength());

  // Copy to temp memory since the sort modifies the original array.
  for (uint p = 0; p < config.numberOfPopulations(); ++p)
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
__global__ void deviceExchangeElite(Gene* population,
                                    uint chromosomeLength,
                                    uint populationSize,
                                    uint numberOfPopulations,
                                    uint* dFitnessIdx,
                                    uint count) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chromosomeLength) return;

  for (uint i = 0; i < numberOfPopulations; ++i)
    for (uint j = 0; j < numberOfPopulations; ++j)
      if (i != j)  // don't duplicate chromosomes
        for (uint k = 0; k < count; ++k) {
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

void Brkga::exchangeElites() {
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

std::vector<Gene> Brkga::getBestChromosome() {
  auto bestIdx = getBest();
  auto bestPopulation = bestIdx.first;
  auto bestChromosome = bestIdx.second;

  std::vector<Gene> best(config.chromosomeLength());
  gpu::copy2h(nullptr, best.data(),
              dPopulation.row(bestPopulation)
                  + bestChromosome * config.chromosomeLength(),
              config.chromosomeLength());

  return best;
}

std::vector<GeneIndex> Brkga::getBestPermutation() {
  if (config.decodeType().chromosome())
    throw InvalidArgument("The chromosome decoder has no permutation",
                          BOX_FUNCTION);

  auto bestIdx = getBest();
  auto bestPopulation = bestIdx.first;
  auto bestChromosome = bestIdx.second;

  // Copy the best chromosome
  std::vector<GeneIndex> best(config.chromosomeLength());
  gpu::copy2h(streams[0], best.data(),
              dPermutations.row(bestPopulation)
                  + bestChromosome * config.chromosomeLength(),
              config.chromosomeLength());
  gpu::sync(streams[0]);

  return best;
}

void Brkga::localSearch(std::function<void(Gene*)> method) {
  const auto totalChromosomes =
      config.numberOfPopulations() * config.populationSize();
  const auto totalGenes = totalChromosomes * config.chromosomeLength();
  const auto genesPerPopulation =
      config.populationSize() * config.chromosomeLength();

  population.resize(totalGenes);

  for (uint i = 0; i < config.numberOfPopulations(); ++i) {
    gpu::copy2h(streams[i], population.data() + i * genesPerPopulation,
                dPopulation.row(i), genesPerPopulation);
  }
  for (uint i = 0; i < config.numberOfPopulations(); ++i) {
    gpu::sync(streams[i]);
    for (uint j = 0; j < config.populationSize(); ++j) {
      const auto offset =
          i * genesPerPopulation + j * config.chromosomeLength();
      method(population.data() + offset);
    }

    gpu::copy2d(streams[i], dPopulation.row(i),
                population.data() + i * genesPerPopulation, genesPerPopulation);
  }

  updateFitness();
}

void Brkga::localSearch(std::function<void(GeneIndex*)> method) {
  const auto totalChromosomes =
      config.numberOfPopulations() * config.populationSize();
  const auto totalGenes = totalChromosomes * config.chromosomeLength();
  const auto genesPerPopulation =
      config.populationSize() * config.chromosomeLength();

  population.resize(totalGenes);
  permutations.resize(totalGenes);

  // If the decode type is permutation, then the chromosomes are already sorted
  if (config.decodeType().chromosome()) sortChromosomesGenes();

  logger::debug("Running user provided local search");

  for (uint i = 0; i < config.numberOfPopulations(); ++i) {
    gpu::copy2h(streams[i], permutations.data() + i * genesPerPopulation,
                dPermutations.row(i), genesPerPopulation);
  }

#ifdef _OPENMP
  for (uint i = 0; i < config.numberOfPopulations(); ++i) gpu::sync(streams[i]);

#pragma omp parallel for if (config.ompThreads() > 1) \
    collapse(2) default(shared) num_threads(config.ompThreads())
#endif  // _OPENMP
  for (uint i = 0; i < config.numberOfPopulations(); ++i) {
#ifndef _OPENMP
    gpu::sync(streams[i]);
#endif  // _OPENMP

    for (uint j = 0; j < config.populationSize(); ++j) {
      const auto offset =
          i * genesPerPopulation + j * config.chromosomeLength();

      method(permutations.data() + offset);

      auto* perm = permutations.data() + offset;
      auto* chr = population.data() + offset;
      auto gene = (Gene)0;
      const auto geneInc = (Gene)1 / (Gene)config.chromosomeLength();
      for (uint k = 0; k < config.chromosomeLength(); ++k) {
        chr[perm[k]] = gene;
        gene += geneInc;
      }
    }

#ifndef _OPENMP
    gpu::copy2d(streams[i], dPopulation.row(i),
                population.data() + i * genesPerPopulation, genesPerPopulation);
#endif  // _OPENMP
  }

#ifdef _OPENMP
  for (uint i = 0; i < config.numberOfPopulations(); ++i) {
    gpu::copy2d(streams[i], dPopulation.row(i),
                population.data() + i * genesPerPopulation, genesPerPopulation);
  }
#endif  // _OPENMP

  updateFitness();
}

Fitness Brkga::getBestFitness() {
  const auto bestPopulation = getBest().first;
  Fitness bestFitness;
  gpu::copy2h(streams[0], &bestFitness, dFitness.row(bestPopulation), 1);
  gpu::sync(streams[0]);
  return bestFitness;
}

std::pair<uint, uint> Brkga::getBest() {
  logger::debug("Searching for the best population/chromosome");

  const uint chromosomesToCopy = 1;
  std::vector<Fitness> bestFitness(config.numberOfPopulations(), -1);
  for (uint p = 0; p < config.numberOfPopulations(); ++p)
    gpu::copy2h(streams[p], &bestFitness[p], dFitness.row(p),
                chromosomesToCopy);

  // Find the best population
  uint bestPopulation = 0;
  for (uint p = 1; p < config.numberOfPopulations(); ++p) {
    gpu::sync(streams[p]);
    if (bestFitness[p] < bestFitness[bestPopulation]) bestPopulation = p;
  }

  // Get the index of the best chromosome
  uint bestChromosome = -1u;
  gpu::copy2h(streams[0], &bestChromosome, dFitnessIdx.row(bestPopulation),
              chromosomesToCopy);
  gpu::sync(streams[0]);

  logger::debug("Best fitness:", bestFitness[bestPopulation], "on population",
                bestPopulation, "and chromosome", bestChromosome);

  return {bestPopulation, bestChromosome};
}

std::vector<DecodedChromosome> Brkga::getPopulation(uint p) {
  std::vector<Fitness> hFitness(config.populationSize());
  gpu::copy2h(streams[p], hFitness.data(), dFitness.row(p),
              config.populationSize());

  std::vector<uint> hFitnessIdx(config.populationSize());
  gpu::copy2h(streams[p], hFitnessIdx.data(), dFitnessIdx.row(p),
              config.populationSize());

  std::vector<Gene> hChromosomes(config.populationSize()
                                 * config.chromosomeLength());
  gpu::copy2h(streams[p], hChromosomes.data(), dPopulation.row(p),
              config.populationSize() * config.chromosomeLength());

  std::vector<DecodedChromosome> decoded;
  for (uint i = 0; i < config.populationSize(); ++i) {
    const auto ptr =
        hChromosomes.begin() + hFitnessIdx[i] * config.chromosomeLength();
    decoded.push_back(DecodedChromosome{
        fitness[i], std::vector<Gene>(ptr, ptr + config.chromosomeLength())});
  }

  return decoded;
}

void Brkga::syncStreams() {
  for (uint p = 0; p < config.numberOfPopulations(); ++p) gpu::sync(streams[p]);
}
}  // namespace box
