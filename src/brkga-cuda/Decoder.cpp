#include "Decoder.hpp"

#include "BrkgaConfiguration.hpp"
#include "Chromosome.hpp"
#include "except/NotImplemented.hpp"

float box::Decoder::decode(const Chromosome<float>&) const {
  throw NotImplemented(__PRETTY_FUNCTION__);
}

float box::Decoder::decode(const Chromosome<unsigned>&) const {
  throw NotImplemented(__PRETTY_FUNCTION__);
}

void box::Decoder::decode(unsigned numberOfChromosomes,
                          const Chromosome<float>* chromosomes,
                          float* fitness) const {
  try {
#ifdef _OPENMP
#pragma omp parallel for if (config->ompThreads > 1) default(shared) \
    num_threads(config->ompThreads)
#endif
    for (unsigned i = 0; i < numberOfChromosomes; ++i) {
      fitness[i] = decode(chromosomes[i]);
    }
  } catch (NotImplemented&) {
    std::throw_with_nested(NotImplemented(__PRETTY_FUNCTION__));
  }
}

void box::Decoder::decode(unsigned numberOfPermutations,
                          const Chromosome<unsigned>* permutations,
                          float* fitness) const {
  try {
#ifdef _OPENMP
#pragma omp parallel for if (config->ompThreads > 1) default(shared) \
    num_threads(config->ompThreads)
#endif
    for (unsigned i = 0; i < numberOfPermutations; ++i) {
      fitness[i] = decode(permutations[i]);
    }
  } catch (NotImplemented&) {
    std::throw_with_nested(NotImplemented(__PRETTY_FUNCTION__));
  }
}

void box::Decoder::decode(cudaStream_t,
                          unsigned,
                          const Chromosome<float>*,
                          float*) const {
  throw NotImplemented(__PRETTY_FUNCTION__);
}

void box::Decoder::decode(cudaStream_t,
                          unsigned,
                          const Chromosome<unsigned>*,
                          float*) const {
  throw NotImplemented(__PRETTY_FUNCTION__);
}
