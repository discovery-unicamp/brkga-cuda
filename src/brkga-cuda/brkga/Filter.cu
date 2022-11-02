#include "../BasicTypes.hpp"
#include "../Brkga.hpp"
#include "../Chromosome.hpp"
#include "../Comparator.hpp"
#include "../Logger.hpp"
#include "../utils/GpuUtils.hpp"

#include <set>
#include <vector>

namespace box {
__global__ void copySorted(Gene* sortedPopulation,
                           const uint* fitnessIdx,
                           const Gene* population,
                           uint numberOfPopulations,
                           uint populationSize,
                           uint chromosomeLength) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfPopulations * populationSize) return;

  const auto p = tid / populationSize;
  const auto c = tid % populationSize;
  const auto* from = population
                     + (p * populationSize + fitnessIdx[p * populationSize + c])
                           * chromosomeLength;
  auto* to = sortedPopulation + tid * chromosomeLength;

  for (uint i = 0; i < chromosomeLength; ++i) to[i] = from[i];
}

void Brkga::printStatus() {
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
  assert(config.decodeType().chromosome());
  population.resize(config.numberOfPopulations() * config.populationSize()
                    * config.chromosomeLength());
  for (uint p = 0; p < config.numberOfPopulations(); ++p) {
    gpu::copy2h(streams[p],
                population.data()
                    + p * config.populationSize() * config.chromosomeLength(),
                dPopulation.row(p),
                config.populationSize() * config.chromosomeLength());
  }
  syncStreams();

  logger::debug("Print info");
  for (uint p = 0; p < config.numberOfPopulations(); ++p) {
    uint k = 0;
    for (uint i = 0; i < config.numberOfElites(); i += k) {
      for (k = 1; i + k < config.populationSize(); ++k) {
        const auto* ci = population.data()
                         + (p * config.populationSize() + i + k - 1)
                               * config.chromosomeLength();
        const auto* ck =
            population.data()
            + (p * config.populationSize() + i + k) * config.chromosomeLength();

        const float eps = 1e-6f;
        bool eq = true;
        for (uint j = 0; j < config.chromosomeLength(); ++j) {
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

void Brkga::removeSimilarElites(const ComparatorBase& comparator) {
  logger::debug("Removing duplicated chromosomes");

  // FIXME this block was duplicated
  population.resize(config.numberOfPopulations() * config.populationSize()
                    * config.chromosomeLength());
  for (uint p = 0; p < config.numberOfPopulations(); ++p) {
    gpu::copy2h(streams[p],
                population.data()
                    + p * config.populationSize() * config.chromosomeLength(),
                dPopulation.row(p),
                config.populationSize() * config.chromosomeLength());
  }

  // TODO should i update the fitness too?
  // fitness.resize(config.numberOfPopulations() * config.populationSize());
  // for (uint p = 0; p < config.numberOfPopulations(); ++p) {
  //   gpu::copy2h(streams[p], fitness.data() + p * config.populationSize(),
  //                dFitness.row(p), config.populationSize());
  // }

  std::vector<uint> fitnessIdx(
      config.numberOfPopulations() * config.populationSize(), -1u);
  for (uint p = 0; p < config.numberOfPopulations(); ++p) {
    gpu::copy2h(streams[p], fitnessIdx.data() + p * config.populationSize(),
                dFitnessIdx.row(p), config.populationSize());
  }

  syncStreams();

  uint duplicatedCount = 0;
  // TODO replace by the worst fitness * factor
  // const float badFitness = 1e18;

  std::vector<Chromosome<Gene>> elites(config.numberOfElites());
  for (uint p = 0; p < config.numberOfPopulations(); ++p) {
    logger::debug("Pruning population", p);
    const auto offset = p * config.populationSize();

    for (uint i = 0; i < config.numberOfElites(); ++i) {
      elites[i] = Chromosome<Gene>(
          population.data() + offset * config.chromosomeLength(),
          config.chromosomeLength(), fitnessIdx[offset + i]);
    }

    uint k = 0;
    std::vector<uint> removedIdx;
    std::vector<bool> remove(config.numberOfElites(), false);
    for (uint i = 0; i < config.numberOfElites(); ++i) {
      if (remove[i]) {
        // fitness[fitnessIdx[offset + i]] = badFitness;
        removedIdx.push_back(fitnessIdx[offset + i]);
        continue;
      }

      fitnessIdx[offset + k] = fitnessIdx[offset + i];
      ++k;
      for (uint j = i + 1; j < config.numberOfElites(); ++j)
        remove[j] = remove[j] || comparator(elites[i], elites[j]);
    }
    if (removedIdx.empty()) continue;
    duplicatedCount += (uint)removedIdx.size();

    // TODO is this enough?
    for (uint i = config.numberOfElites(); i < config.populationSize(); ++i) {
      fitnessIdx[offset + k] = fitnessIdx[offset + i];
      ++k;
    }
    for (uint idx : removedIdx) {
      fitnessIdx[offset + k] = idx;
      ++k;
    }
    assert(k == config.populationSize());
    assert((uint)std::set<uint>(
               fitnessIdx.begin() + offset,
               fitnessIdx.begin() + offset + config.populationSize())
               .size()
           == config.populationSize());
  }

  logger::debug("Copying data to device");
  for (uint p = 0; p < config.numberOfPopulations(); ++p) {
    gpu::copy2d(streams[p], dFitnessIdx.row(p),
                fitnessIdx.data() + p * config.populationSize(),
                config.populationSize());
  }

  logger::debug("Removed", duplicatedCount, "duplicated chromosomes");
}
}  // namespace box
