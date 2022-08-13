#ifndef SRC_BRKGA_INSTANCE_HPP
#define SRC_BRKGA_INSTANCE_HPP

#include "BrkgaConfiguration.hpp"
#include "Chromosome.hpp"

#include <cuda_runtime.h>

namespace box {
class Decoder {
public:
  Decoder() : config(nullptr) {}
  Decoder(const Decoder&) = default;
  Decoder(Decoder&&) = default;
  Decoder& operator=(const Decoder&) = default;
  Decoder& operator=(Decoder&&) = default;

  virtual ~Decoder() = default;

  inline void setConfiguration(const BrkgaConfiguration* newConfig) {
    config = newConfig;
  }

  virtual float decode(const Chromosome<float>& chromosome) const;

  virtual float decode(const Chromosome<unsigned>& permutation) const;

  virtual void decode(unsigned numberOfChromosomes,
                      const Chromosome<float>* chromosomes,
                      float* fitness) const;

  virtual void decode(unsigned numberOfPermutations,
                      const Chromosome<unsigned>* permutations,
                      float* fitness) const;

  virtual void decode(cudaStream_t stream,
                      unsigned numberOfChromosomes,
                      const Chromosome<float>* dChromosomes,
                      float* dFitness) const;

  virtual void decode(cudaStream_t stream,
                      unsigned numberOfPermutations,
                      const Chromosome<unsigned>* dPermutations,
                      float* dFitness) const;

protected:
  const box::BrkgaConfiguration* config;
};
}  // namespace box

#endif  // SRC_BRKGA_INSTANCE_HPP
