#include "Brkga.hpp"
#include "BrkgaConfiguration.hpp"
#include "CudaError.cuh"
#include "CudaUtils.hpp"
#include "DecodeType.hpp"
#include "Decoder.hpp"
#include "Logger.hpp"

#include <curand.h>

#include <algorithm>
#include <cassert>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

box::Brkga::Brkga(const BrkgaConfiguration& config)
    : dPopulation(config.numberOfPopulations,
                  config.populationSize * config.chromosomeLength),
      dPopulationTemp(config.numberOfPopulations,
                      config.populationSize * config.chromosomeLength),
      dFitness(config.numberOfPopulations, config.populationSize),
      dFitnessIdx(config.numberOfPopulations, config.populationSize),
      dPermutations(config.numberOfPopulations,
                    config.populationSize * config.chromosomeLength),
      dRandomEliteParent(config.numberOfPopulations, config.populationSize),
      dRandomParent(config.numberOfPopulations, config.populationSize) {
  CUDA_CHECK_LAST();
  // TODO save only the configuration class
  decoder = config.decoder;
  numberOfPopulations = config.numberOfPopulations;
  populationSize = config.populationSize;
  numberOfChromosomes = numberOfPopulations * populationSize;
  numberOfGenes = numberOfChromosomes * config.chromosomeLength;
  chromosomeSize = config.chromosomeLength;
  eliteSize = config.eliteCount;
  mutantsSize = config.mutantsCount;
  rhoe = config.rhoe;
  decodeType = config.decodeType;
  threadsPerBlock = config.threadsPerBlock;

  decoder->setConfiguration(&config);

  // One stream for each population
  streams.resize(numberOfPopulations);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    streams[p] = cuda::allocStream();

  logger::debug("Building random generator with seed", config.seed);
  std::mt19937 rng(config.seed);
  std::uniform_int_distribution<std::mt19937::result_type> uid;
  generators.resize(numberOfPopulations);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    generators[p] = cuda::allocRandomGenerator(uid(rng));

  logger::debug("Building the initial populations");
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::random(streams[p], generators[p], dPopulation.row(p),
                 populationSize * chromosomeSize);

  updateFitness();
  logger::debug("Brkga was configured successfully");
}

box::Brkga::~Brkga() {
  for (unsigned p = 0; p < numberOfPopulations; ++p) cuda::free(generators[p]);
  for (unsigned p = 0; p < numberOfPopulations; ++p) cuda::free(streams[p]);
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
    cuda::random(streams[p], generators[p], dPopulationTemp.row(p),
                 populationSize * chromosomeSize);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::random(streams[p], generators[p], dRandomEliteParent.row(p),
                 populationSize - mutantsSize);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::random(streams[p], generators[p], dRandomParent.row(p),
                 populationSize - mutantsSize);

  logger::debug("Copying the elites to the next generation");
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    evolveCopyElite<<<cuda::blocks(eliteSize * chromosomeSize, threadsPerBlock),
                      threadsPerBlock, 0, streams[p]>>>(
        dPopulationTemp.row(p), dPopulation.row(p), dFitnessIdx.row(p),
        eliteSize, chromosomeSize);
  }
  CUDA_CHECK_LAST();

  logger::debug("Mating pairs of the population");
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    const auto blocks = cuda::blocks(
        (populationSize - eliteSize - mutantsSize) * chromosomeSize,
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
        cuda::copy2h(streams[p],
                     population.data() + p * populationSize * chromosomeSize,
                     dPopulation.row(p), populationSize * chromosomeSize);
      }
    } else {
      permutations.resize(numberOfPopulations * populationSize
                          * chromosomeSize);
      for (unsigned p = 0; p < numberOfPopulations; ++p) {
        cuda::copy2h(streams[p],
                     permutations.data() + p * populationSize * chromosomeSize,
                     dPermutations.row(p), populationSize * chromosomeSize);
      }
    }

    fitness.resize(numberOfPopulations * populationSize * chromosomeSize);
  }

  logger::debug("Calling the decoder");
  if (decodeType.allAtOnce()) {
    syncStreams();
    logger::debug("Decoding all at once");

    const auto totalChromosomes = numberOfPopulations * populationSize;
    if (decodeType.onCpu()) {
      if (decodeType.chromosome()) {
        decoder->decode(totalChromosomes, population.data(), fitness.data());
      } else {
        decoder->decode(totalChromosomes, permutations.data(), fitness.data());
      }

      for (unsigned p = 0; p < numberOfPopulations; ++p)
        cuda::copy2d(streams[p], dFitness.row(p),
                     fitness.data() + p * populationSize, populationSize);
    } else {
      if (decodeType.chromosome()) {
        decoder->decode(streams[0], totalChromosomes, dPopulation.get(),
                        dFitness.get());
      } else {
        decoder->decode(streams[0], totalChromosomes, dPermutations.get(),
                        dFitness.get());
      }
      CUDA_CHECK_LAST();
      cuda::sync(streams[0]);
    }

    for (unsigned p = 0; p < numberOfPopulations; ++p)
      cuda::iota(streams[p], dFitnessIdx.row(p), populationSize);
    for (unsigned p = 0; p < numberOfPopulations; ++p)
      cuda::sortByKey(streams[p], dFitness.row(p), dFitnessIdx.row(p),
                      populationSize);
  } else {
    logger::debug("Decoding by population");
    for (unsigned p = 0; p < numberOfPopulations; ++p) {
      if (decodeType.onCpu()) {
        cuda::sync(streams[p]);
        if (decodeType.chromosome()) {
          decoder->decode(
              populationSize,
              population.data() + p * populationSize * chromosomeSize,
              fitness.data() + p * populationSize);
        } else {
          decoder->decode(
              populationSize,
              permutations.data() + p * populationSize * chromosomeSize,
              fitness.data() + p * populationSize);
        }

        cuda::copy2d(streams[p], dFitness.row(p),
                     fitness.data() + p * populationSize, populationSize);
      } else {
        if (decodeType.chromosome()) {
          decoder->decode(streams[p], populationSize, dPopulation.row(p),
                          dFitness.row(p));
        } else {
          decoder->decode(streams[p], populationSize, dPermutations.row(p),
                          dFitness.row(p));
        }
        CUDA_CHECK_LAST();
      }

      cuda::iota(streams[p], dFitnessIdx.row(p), populationSize);
      cuda::sortByKey(streams[p], dFitness.row(p), dFitnessIdx.row(p),
                      populationSize);
    }
  }

  logger::debug("Decoding step has finished");
}

void box::Brkga::sortChromosomesGenes() {
  if (decodeType.chromosome()) return;
  logger::debug("Sorting the chromosomes for sorted decode");

  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::iotaMod(streams[p], dPermutations.row(p),
                  populationSize * chromosomeSize, chromosomeSize);

  // Copy to temp memory since the sort modifies the original array.
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::copy(streams[p], dPopulationTemp.row(p), dPopulation.row(p),
               populationSize * chromosomeSize);

  // FIXME We should sort each chromosome on its own thread to avoid
  //  synchonization.
  syncStreams();
  cuda::segSort(streams[0], dPopulationTemp.get(), dPermutations.get(),
                numberOfChromosomes, chromosomeSize);
  cuda::sync(streams[0]);
}

void box::Brkga::decodePopulation(unsigned p) {
  logger::debug("Decode population", p, "with", decodeType.str(), "decoder");

  // if (decodeType == DecodeType::DEVICE) {
  //   decoder->decode(streams[p], populationSize, dPopulation.row(p),
  //                   dFitness.row(p));
  //   CUDA_CHECK_LAST();
  // } else if (decodeType == DecodeType::DEVICE_SORTED) {
  //   decoder->decode(streams[p], populationSize, dPermutations.row(p),
  //                   dFitness.row(p));
  //   CUDA_CHECK_LAST();
  // } else if (decodeType == DecodeType::HOST) {
  //   decoder->decode(populationSize, population[p].data(), fitness[p].data());
  // } else if (decodeType == DecodeType::HOST_SORTED) {
  //   decoder->decode(populationSize, permutations[p].data(),
  //   fitness[p].data());
  // } else {
  //   throw std::domain_error("Function decode type is unknown");
  // }

  logger::debug("Finished the decoder of the population", p);
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

  if (count > eliteSize)
    throw std::range_error("Exchange count is greater than elite size.");
  if (count * numberOfPopulations > populationSize) {
    throw std::range_error(
        "Exchange count will replace the entire population: it should be at "
        "most [population size] / [number of populations] ("
        + std::to_string(populationSize / numberOfPopulations) + ").");
  }

  syncStreams();

  const auto blocks = cuda::blocks(chromosomeSize, threadsPerBlock);
  deviceExchangeElite<<<blocks, threadsPerBlock>>>(
      dPopulation.get(), chromosomeSize, populationSize, numberOfPopulations,
      dFitnessIdx.get(), count);
  CUDA_CHECK_LAST();
  cuda::sync();

  updateFitness();
}

std::vector<float> box::Brkga::getBestChromosome() {
  auto bestIdx = getBest();
  auto bestPopulation = bestIdx.first;
  auto bestChromosome = bestIdx.second;

  std::vector<float> best(chromosomeSize);
  cuda::copy2h(
      nullptr, best.data(),
      dPopulation.row(bestPopulation) + bestChromosome * chromosomeSize,
      chromosomeSize);

  return best;
}

std::vector<unsigned> box::Brkga::getBestPermutation() {
  if (!decodeType.chromosome())
    throw std::runtime_error("Only sorted decodes can get the sorted indices");

  auto bestIdx = getBest();
  auto bestPopulation = bestIdx.first;
  auto bestChromosome = bestIdx.second;

  // Copy the best chromosome
  std::vector<unsigned> best(chromosomeSize);
  cuda::copy2h(
      streams[0], best.data(),
      dPermutations.row(bestPopulation) + bestChromosome * chromosomeSize,
      chromosomeSize);
  cuda::sync(streams[0]);

  return best;
}

float box::Brkga::getBestFitness() {
  auto bestPopulation = getBest().first;
  float bestFitness = -1;
  cuda::copy2h(streams[0], &bestFitness, dFitness.row(bestPopulation), 1);
  cuda::sync(streams[0]);
  return bestFitness;
}

std::pair<unsigned, unsigned> box::Brkga::getBest() {
  logger::debug("Searching for the best population/chromosome");

  const unsigned chromosomesToCopy = 1;
  std::vector<float> bestFitness(numberOfPopulations, -1);
  for (unsigned p = 0; p < numberOfPopulations; ++p)
    cuda::copy2h(streams[p], &bestFitness[p], dFitness.row(p),
                 chromosomesToCopy);

  // Find the best population
  unsigned bestPopulation = 0;
  for (unsigned p = 1; p < numberOfPopulations; ++p) {
    cuda::sync(streams[p]);
    if (bestFitness[p] < bestFitness[bestPopulation]) bestPopulation = p;
  }

  // Get the index of the best chromosome
  unsigned bestChromosome = (unsigned)-1;
  cuda::copy2h(streams[0], &bestChromosome, dFitnessIdx.row(bestPopulation),
               chromosomesToCopy);
  cuda::sync(streams[0]);

  logger::debug("Best fitness:", bestFitness[bestPopulation], "on population",
                bestPopulation, "and chromosome", bestChromosome);

  return {bestPopulation, bestChromosome};
}

void box::Brkga::syncStreams() {
  for (unsigned p = 0; p < numberOfPopulations; ++p) cuda::sync(streams[p]);
}
