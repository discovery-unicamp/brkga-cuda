#ifndef BRKGACUDA_BRKGACONFIGURATION_HPP
#define BRKGACUDA_BRKGACONFIGURATION_HPP

#include "DecodeType.hpp"

namespace box {
class Decoder;

// TODO add docstring to the methods

/// Configuration of the BRKGA algorithm
class BrkgaConfiguration {
public:
  class Builder {
  public:
    Builder();
    ~Builder();

    Builder& decoder(Decoder* d);
    Builder& decodeType(DecodeType dt);
    Builder& ompThreads(unsigned k);
    Builder& gpuThreads(unsigned k);
    Builder& numberOfPopulations(unsigned n);
    Builder& populationSize(unsigned n);
    Builder& chromosomeLength(unsigned n);
    Builder& numberOfElites(unsigned n);
    Builder& elitePercentage(float p);
    Builder& numberOfMutants(unsigned n);
    Builder& mutantPercentage(float p);
    Builder& rhoe(float r);
    Builder& numberOfElitesToExchange(unsigned k);
    Builder& seed(unsigned s);

    BrkgaConfiguration build();

  private:
    BrkgaConfiguration* config;
  };

  virtual ~BrkgaConfiguration() = default;

  inline Decoder* decoder() { return _decoder; }
  inline const Decoder* decoder() const { return _decoder; }
  inline DecodeType decodeType() const { return _decodeType; }
  inline unsigned numberOfPopulations() const { return _numberOfPopulations; }
  inline unsigned populationSize() const { return _populationSize; }
  inline unsigned chromosomeLength() const { return _chromosomeLength; }
  inline float rhoe() const { return _rhoe; }

  inline unsigned numberOfElites() const { return _numberOfElites; }
  inline float elitePercentage() const {
    return (float)_numberOfElites / (float)_populationSize;
  }

  inline unsigned numberOfMutants() const { return _numberOfMutants; }
  inline float mutantPercentage() const {
    return (float)_numberOfMutants / (float)_populationSize;
  }

  inline unsigned numberOfElitesToExchange() const {
    return _numberOfElitesToExchange;
  }

  inline unsigned seed() const { return _seed; }
  inline unsigned ompThreads() const { return _ompThreads; }
  inline unsigned gpuThreads() const { return _gpuThreads; }

  void setRhoe(float r);
  void setNumberOfElites(unsigned n);
  void setElitePercentage(float p);
  void setNumberOfMutants(unsigned n);
  void setMutantPercentage(float p);
  void setNumberOfElitesToExchange(unsigned k);
  void setOmpThreads(unsigned k);
  void setGpuThreads(unsigned k);

private:
  friend Builder;

  BrkgaConfiguration()
      : _decoder(nullptr),
        _decodeType(),
        _numberOfPopulations(0),
        _populationSize(0),
        _chromosomeLength(0),
        _rhoe(0),
        _numberOfElites(0),
        _numberOfMutants(0),
        _numberOfElitesToExchange(0),
        _seed(0),
        _ompThreads(1),
        _gpuThreads(0) {}

  Decoder* _decoder;  /// The decoder implementation
  DecodeType _decodeType;  /// @see DecodeType.hpp
  unsigned _numberOfPopulations;  /// Number of independent populations
  unsigned _populationSize;  /// Size/#chromosomes of each population
  unsigned _chromosomeLength;  /// The length of the chromosomes
  float _rhoe;  /// Probability that child gets an allele from the elite parent
  unsigned _numberOfElites;  /// #elites in the population
  unsigned _numberOfMutants;  /// #mutants in the population
  unsigned _numberOfElitesToExchange;  /// #elites for @ref exchangeElites
  unsigned _seed;  /// The seed to use
  unsigned _ompThreads;  /// #threads to use on OpenMP
  unsigned _gpuThreads;  /// #threads per block for CUDA kernels
};
}  // namespace box

#endif  // BRKGACUDA_BRKGACONFIGURATION_HPP
