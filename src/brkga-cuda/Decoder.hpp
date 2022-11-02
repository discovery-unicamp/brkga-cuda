#ifndef SRC_BRKGA_INSTANCE_HPP
#define SRC_BRKGA_INSTANCE_HPP

#include "BasicTypes.hpp"
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

  void setConfiguration(const BrkgaConfiguration* newConfig);

  virtual Fitness decode(const Chromosome<Gene>& chromosome) const;

  virtual Fitness decode(const Chromosome<GeneIndex>& permutation) const;

  virtual void decode(uint numberOfChromosomes,
                      const Chromosome<Gene>* chromosomes,
                      Fitness* fitness) const;

  virtual void decode(uint numberOfPermutations,
                      const Chromosome<GeneIndex>* permutations,
                      Fitness* fitness) const;

  virtual void decode(cudaStream_t stream,
                      uint numberOfChromosomes,
                      const Chromosome<Gene>* dChromosomes,
                      Fitness* dFitness) const;

  virtual void decode(cudaStream_t stream,
                      uint numberOfPermutations,
                      const Chromosome<GeneIndex>* dPermutations,
                      Fitness* dFitness) const;

protected:
  const box::BrkgaConfiguration* config;
};
}  // namespace box

#endif  // SRC_BRKGA_INSTANCE_HPP
