#include "../Brkga.hpp"
#include "../BrkgaFilter.hpp"
#include "../Chromosome.hpp"
#include "../CudaUtils.hpp"
#include "../Logger.hpp"

#include <set>
#include <vector>

__global__ void copySorted(float* sortedPopulation,
                           const unsigned* fitnessIdx,
                           const float* population,
                           unsigned numberOfPopulations,
                           unsigned populationSize,
                           unsigned chromosomeSize) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numberOfPopulations * populationSize) return;

  const auto p = tid / populationSize;
  const auto c = tid % populationSize;
  const float* from =
      population
      + (p * populationSize + fitnessIdx[p * populationSize + c])
            * chromosomeSize;
  float* to = sortedPopulation + tid * chromosomeSize;

  for (unsigned i = 0; i < chromosomeSize; ++i) to[i] = from[i];
}

void box::Brkga::printStatus() {
  logger::debug("Copy chromosomes sorted");
  copySorted<<<cuda::blocks(numberOfPopulations * populationSize,
                            threadsPerBlock),
               threadsPerBlock>>>(dPopulationTemp.get(), dFitnessIdx.get(),
                                  dPopulation.get(), numberOfPopulations,
                                  populationSize, chromosomeSize);
  CUDA_CHECK_LAST();
  cuda::sync();

  logger::debug("Copy data to host");
  assert(decodeType.chromosome());
  population.resize(numberOfPopulations * populationSize * chromosomeSize);
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    cuda::copy2h(streams[p],
                 population.data() + p * populationSize * chromosomeSize,
                 dPopulation.row(p), populationSize * chromosomeSize);
  }
  syncStreams();

  logger::debug("Print info");
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    unsigned k = 0;
    for (unsigned i = 0; i < eliteSize; i += k) {
      for (k = 1; i + k < populationSize; ++k) {
        const float* ci = population.data()
                          + (p * populationSize + i + k - 1) * chromosomeSize;
        const float* ck =
            population.data() + (p * populationSize + i + k) * chromosomeSize;

        const float eps = 1e-6f;
        bool eq = true;
        for (unsigned j = 0; j < chromosomeSize; ++j) {
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

void box::Brkga::removeSimilarElites(const FilterBase& filter) {
  logger::debug("Removing duplicated chromosomes");

  assert(decodeType.chromosome());
  logger::debug("Copying data to host");

  // FIXME this block was duplicated
  population.resize(numberOfPopulations * populationSize * chromosomeSize);
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    cuda::copy2h(streams[p],
                 population.data() + p * populationSize * chromosomeSize,
                 dPopulation.row(p), populationSize * chromosomeSize);
  }

  // TODO should i update the fitness too?
  // fitness.resize(numberOfPopulations * populationSize);
  // for (unsigned p = 0; p < numberOfPopulations; ++p) {
  //   cuda::copy2h(streams[p], fitness.data() + p * populationSize,
  //                dFitness.row(p), populationSize);
  // }

  std::vector<unsigned> fitnessIdx(numberOfPopulations * populationSize, -1u);
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    cuda::copy2h(streams[p], fitnessIdx.data() + p * populationSize,
                 dFitnessIdx.row(p), populationSize);
  }

  syncStreams();

  unsigned duplicatedCount = 0;
  // const float badFitness = 1e18;  // TODO replace by the worst fitness *
  // factor

  std::vector<box::Chromosome<float>> elites(eliteSize);
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    const auto offset = p * populationSize;

    for (unsigned i = 0; i < eliteSize; ++i) {
      elites[i] = Chromosome<float>(population.data() + offset * chromosomeSize,
                                    chromosomeSize, fitnessIdx[offset + i]);
    }

    unsigned popDuplicatedCount = 0;
    std::vector<bool> remove(eliteSize, false);
    for (unsigned i = 0; i < eliteSize; ++i) {
      if (remove[i]) {
        popDuplicatedCount += 1;
        // fitness[fitnessIdx[offset + i]] = badFitness;
        continue;
      }
      for (unsigned j = i + 1; j < eliteSize; ++j)
        remove[j] = remove[j] || filter(elites[i], elites[j]);
    }

    if (popDuplicatedCount == 0) continue;
    duplicatedCount += popDuplicatedCount;

    unsigned k = 0;
    std::vector<unsigned> removedIdx;
    for (unsigned i = 0; i < populationSize; ++i) {
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
           == populationSize);
  }

  logger::debug("Copying data to device");
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    cuda::copy2d(streams[p], dFitnessIdx.row(p),
                 fitnessIdx.data() + p * populationSize, populationSize);
  }

  logger::debug("Removed", duplicatedCount, "duplicated chromosomes");
}
