#include "../Brkga.hpp"
#include "../Comparator.hpp"
#include "../Chromosome.hpp"
#include "../Logger.hpp"
#include "../utils/GpuUtils.hpp"

#include <set>
#include <vector>

__global__ void copySorted(float* sortedPopulation,
                           const unsigned* fitnessIdx,
                           const float* population,
                           unsigned numberOfPopulations,
                           unsigned populationSize,
                           unsigned chromosomeLength) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfPopulations * populationSize) return;

  const auto p = tid / populationSize;
  const auto c = tid % populationSize;
  const float* from =
      population
      + (p * populationSize + fitnessIdx[p * populationSize + c])
            * chromosomeLength;
  float* to = sortedPopulation + tid * chromosomeLength;

  for (unsigned i = 0; i < chromosomeLength; ++i) to[i] = from[i];
}

void box::Brkga::printStatus() {
  logger::debug("Copy chromosomes sorted");
  copySorted<<<gpu::blocks(
                   config.numberOfPopulations() * config.populationSize(),
                   config.gpuThreads()),
               config.gpuThreads()>>>(
      dPopulationTemp.get(), dFitnessIdx.get(), dPopulation.get(),
      config.numberOfPopulations(), config.populationSize(),
      config.chromosomeLength());
  CUDA_CHECK_LAST();
  gpu::sync();

  logger::debug("Copy data to host");
  assert(decodeType.chromosome());
  population.resize(config.numberOfPopulations() * config.populationSize()
                    * config.chromosomeLength());
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p) {
    gpu::copy2h(streams[p],
                population.data()
                    + p * config.populationSize() * config.chromosomeLength(),
                dPopulation.row(p),
                config.populationSize() * config.chromosomeLength());
  }
  syncStreams();

  logger::debug("Print info");
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p) {
    unsigned k = 0;
    for (unsigned i = 0; i < config.numberOfElites(); i += k) {
      for (k = 1; i + k < config.populationSize(); ++k) {
        const float* ci = population.data()
                          + (p * config.populationSize() + i + k - 1)
                                * config.chromosomeLength();
        const float* ck =
            population.data()
            + (p * config.populationSize() + i + k) * config.chromosomeLength();

        const float eps = 1e-6f;
        bool eq = true;
        for (unsigned j = 0; j < config.chromosomeLength(); ++j) {
          if (std::abs(ci[j] - ck[j]) >= eps) {
            eq = false;
            break;
          }
        }

        if (!eq) break;
      }

      if (k > 2) {
        logger::warning("Found", k, "equal chromosomes on population", p);
      }
    }
  }
}

void box::Brkga::removeSimilarElites(const ComparatorBase& filter) {
  logger::debug("Removing duplicated chromosomes");

  assert(decodeType.chromosome());
  logger::debug("Copying data to host");

  // FIXME this block was duplicated
  population.resize(config.numberOfPopulations() * config.populationSize()
                    * config.chromosomeLength());
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p) {
    gpu::copy2h(streams[p],
                population.data()
                    + p * config.populationSize() * config.chromosomeLength(),
                dPopulation.row(p),
                config.populationSize() * config.chromosomeLength());
  }

  // TODO should i update the fitness too?
  // fitness.resize(config.numberOfPopulations() * config.populationSize());
  // for (unsigned p = 0; p < config.numberOfPopulations(); ++p) {
  //   gpu::copy2h(streams[p], fitness.data() + p * config.populationSize(),
  //                dFitness.row(p), config.populationSize());
  // }

  std::vector<unsigned> fitnessIdx(
      config.numberOfPopulations() * config.populationSize(), -1u);
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p) {
    gpu::copy2h(streams[p], fitnessIdx.data() + p * config.populationSize(),
                dFitnessIdx.row(p), config.populationSize());
  }

  syncStreams();

  unsigned duplicatedCount = 0;
  // const float badFitness = 1e18;  // TODO replace by the worst fitness *
  // factor

  std::vector<box::Chromosome<float>> elites(config.numberOfElites());
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p) {
    const auto offset = p * config.populationSize();

    for (unsigned i = 0; i < config.numberOfElites(); ++i) {
      elites[i] = Chromosome<float>(
          population.data() + offset * config.chromosomeLength(),
          config.chromosomeLength(), fitnessIdx[offset + i]);
    }

    unsigned popDuplicatedCount = 0;
    std::vector<bool> remove(config.numberOfElites(), false);
    for (unsigned i = 0; i < config.numberOfElites(); ++i) {
      if (remove[i]) {
        popDuplicatedCount += 1;
        // fitness[fitnessIdx[offset + i]] = badFitness;
        continue;
      }
      for (unsigned j = i + 1; j < config.numberOfElites(); ++j)
        remove[j] = remove[j] || filter(elites[i], elites[j]);
    }

    if (popDuplicatedCount == 0) continue;
    duplicatedCount += popDuplicatedCount;

    unsigned k = 0;
    std::vector<unsigned> removedIdx;
    for (unsigned i = 0; i < config.populationSize(); ++i) {
      if (remove[i]) {
        removedIdx.push_back(fitnessIdx[i]);
      } else {
        fitnessIdx[k] = fitnessIdx[i];
        ++k;
      }
    }
    // TODO is this enough?
    for (unsigned idx : removedIdx) {
      fitnessIdx[k] = idx;
      ++k;
    }
    assert((unsigned)std::set<unsigned>(fitnessIdx.begin(), fitnessIdx.end())
               .size()
           == config.populationSize());
  }

  logger::debug("Copying data to device");
  for (unsigned p = 0; p < config.numberOfPopulations(); ++p) {
    gpu::copy2d(streams[p], dFitnessIdx.row(p),
                fitnessIdx.data() + p * config.populationSize(),
                config.populationSize());
  }

  logger::debug("Removed", duplicatedCount, "duplicated chromosomes");
}
