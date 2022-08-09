#ifndef BRKGACUDA_BRKGACONFIGURATION_HPP
#define BRKGACUDA_BRKGACONFIGURATION_HPP

#include "DecodeType.hpp"

namespace box {
class Decoder;

class BrkgaConfiguration {
public:
  class Builder {
  public:
    Builder& decoder(Decoder* d);
    Builder& threadsPerBlock(unsigned k);
    Builder& ompThreads(unsigned k);
    Builder& generations(unsigned n);
    Builder& exchangeBestInterval(unsigned k);
    Builder& exchangeBestCount(unsigned n);
    Builder& numberOfPopulations(unsigned n);
    Builder& populationSize(unsigned n);
    Builder& chromosomeLength(unsigned n);
    Builder& eliteCount(unsigned n);
    Builder& eliteProportion(float p);
    Builder& mutantsCount(unsigned n);
    Builder& mutantsProportion(float p);
    Builder& rhoe(float r);
    Builder& seed(unsigned s);
    Builder& decodeType(DecodeType dt);

    BrkgaConfiguration build() const;

  private:
    Decoder* _decoder = nullptr;
    unsigned _generations = 0;
    unsigned _threadsPerBlock = 0;
    unsigned _ompThreads = 0;  // TODO default to the number of threads of the CPU
    unsigned _exchangeBestInterval = 0;
    unsigned _exchangeBestCount = 0;
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

  // these members are just for convenience as they aren't used by the main
  // algorithm
  unsigned generations;  ///< the number of generations of the population
  unsigned exchangeBestInterval;  ///< steps to exchange best individuals
  unsigned exchangeBestCount;  ///< number of individuals to exchange

private:
  friend Builder;

  BrkgaConfiguration() {}
};
}  // namespace box

#endif  // BRKGACUDA_BRKGACONFIGURATION_HPP
