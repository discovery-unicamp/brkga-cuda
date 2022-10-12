#include "Decoder.hpp"

#include "BrkgaConfiguration.hpp"
#include "Chromosome.hpp"
#include "except/InvalidArgument.hpp"
#include "except/NotImplemented.hpp"

namespace box {
void Decoder::setConfiguration(const BrkgaConfiguration* newConfig) {
  InvalidArgument::null(Arg<const BrkgaConfiguration*>(newConfig),
                        __FUNCTION__);
  config = newConfig;
}

float Decoder::decode(const Chromosome<float>&) const {
  throw NotImplemented(__PRETTY_FUNCTION__);
}

float Decoder::decode(const Chromosome<unsigned>&) const {
  throw NotImplemented(__PRETTY_FUNCTION__);
}

void Decoder::decode(unsigned numberOfChromosomes,
                     const Chromosome<float>* chromosomes,
                     float* fitness) const {
  try {
#ifdef _OPENMP
#pragma omp parallel for if (config->ompThreads() > 1) default(shared) \
    num_threads(config->ompThreads())
#endif
    for (unsigned i = 0; i < numberOfChromosomes; ++i) {
      fitness[i] = decode(chromosomes[i]);
    }
  } catch (NotImplemented&) {
    std::throw_with_nested(NotImplemented(__PRETTY_FUNCTION__));
  }
}

void Decoder::decode(unsigned numberOfPermutations,
                     const Chromosome<unsigned>* permutations,
                     float* fitness) const {
  try {
#ifdef _OPENMP
#pragma omp parallel for if (config->ompThreads() > 1) default(shared) \
    num_threads(config->ompThreads())
#endif
    for (unsigned i = 0; i < numberOfPermutations; ++i) {
      fitness[i] = decode(permutations[i]);
    }
  } catch (NotImplemented&) {
    std::throw_with_nested(NotImplemented(__PRETTY_FUNCTION__));
  }
}

void Decoder::decode(cudaStream_t,
                     unsigned,
                     const Chromosome<float>*,
                     float*) const {
  throw NotImplemented(__PRETTY_FUNCTION__);
}

void Decoder::decode(cudaStream_t,
                     unsigned,
                     const Chromosome<unsigned>*,
                     float*) const {
  throw NotImplemented(__PRETTY_FUNCTION__);
}
}  // namespace box
