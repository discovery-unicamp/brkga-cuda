#ifndef BRKGACUDA_BRKGACONFIGURATION_HPP
#define BRKGACUDA_BRKGACONFIGURATION_HPP

#include "DecodeType.hpp"

namespace box {
class Decoder;

// FIXME some parameters are not used inside the algorithm; should we keep them?

class BrkgaConfiguration {
public:
  class Builder {
  public:
    Builder& decoder(Decoder* d);
    Builder& threadsPerBlock(unsigned k);
    Builder& ompThreads(unsigned k);
    Builder& numberOfPopulations(unsigned n);
    Builder& populationSize(unsigned n);
    Builder& chromosomeLength(unsigned n);
    Builder& eliteCount(unsigned n);
    Builder& eliteFactor(float p);
    Builder& mutantsCount(unsigned n);
    Builder& mutantsFactor(float p);
    Builder& rhoe(float r);
    Builder& seed(unsigned s);
    Builder& decodeType(DecodeType dt);

    BrkgaConfiguration build() const;

  private:
    Decoder* _decoder = nullptr;
    unsigned _threadsPerBlock = 0;
    unsigned _ompThreads = 1;
    unsigned _numberOfPopulations = 0;
    unsigned _populationSize = 0;
    unsigned _chromosomeLength = 0;
    unsigned _eliteCount = 0;
    unsigned _mutantsCount = 0;
    float _rhoe = 0;
    unsigned _seed = 0;
    DecodeType _decodeType;
  };

  virtual ~BrkgaConfiguration() = default;

  [[nodiscard]] inline float getMutantsProbability() const {
    return (float)mutantsCount / (float)populationSize;
  }

  [[nodiscard]] inline float getEliteProbability() const {
    return (float)eliteCount / (float)populationSize;
  }

  // TODO make private
  Decoder* decoder;
  DecodeType decodeType;  ///< @see DecodeType.hpp
  unsigned threadsPerBlock;  ///< number threads per block in CUDA
  unsigned ompThreads;  ///< number of threads to use on OpenMP
  unsigned numberOfPopulations;  ///< number of independent populations
  unsigned populationSize;  ///< size of the population
  unsigned chromosomeLength;  ///< the length of the chromosome to be generated
  unsigned eliteCount;  ///< proportion of elite population
  unsigned mutantsCount;  ///< proportion of mutant population
  float rhoe;  ///< probability that child gets an allele from elite parent
  unsigned seed;  ///< the seed to use in the algorithm

private:
  friend Builder;

  BrkgaConfiguration() {}
};
}  // namespace box

#endif  // BRKGACUDA_BRKGACONFIGURATION_HPP
