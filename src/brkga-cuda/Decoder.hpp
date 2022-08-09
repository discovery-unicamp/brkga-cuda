#ifndef SRC_BRKGA_INSTANCE_HPP
#define SRC_BRKGA_INSTANCE_HPP

#include "BrkgaConfiguration.hpp"

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

  virtual float decode(const float* chromosome) const;

  virtual float decode(const unsigned* permutation) const;

  virtual void decode(unsigned numberOfChromosomes,
                      const float* chromosomes,
                      float* fitness) const;

  virtual void decode(unsigned numberOfPermutations,
                      const unsigned* permutations,
                      float* fitness) const;

  virtual void decode(cudaStream_t stream,
                      unsigned numberOfChromosomes,
                      const float* dChromosomes,
                      float* dFitness) const;

  virtual void decode(cudaStream_t stream,
                      unsigned numberOfPermutations,
                      const unsigned* dPermutations,
                      float* dFitness) const;

protected:
  const box::BrkgaConfiguration* config;
};
}  // namespace box

#endif  // SRC_BRKGA_INSTANCE_HPP
