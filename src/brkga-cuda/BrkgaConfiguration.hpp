#ifndef BRKGACUDA_BRKGACONFIGURATION_HPP
#define BRKGACUDA_BRKGACONFIGURATION_HPP

#include "BasicTypes.hpp"
#include "Bias.hpp"
#include "DecodeType.hpp"

#include <vector>

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
    Builder& ompThreads(uint k);
    Builder& gpuThreads(uint k);
    Builder& numberOfPopulations(uint n);
    Builder& populationSize(uint n);
    Builder& chromosomeLength(uint n);
    Builder& numberOfElites(uint n);
    Builder& elitePercentage(float p);
    Builder& numberOfMutants(uint n);
    Builder& mutantPercentage(float p);
    Builder& parents(const std::vector<float>& bias, uint numberOfEliteParents);
    Builder& parents(uint n, Bias biasType, uint numberOfEliteParents);
    Builder& numberOfElitesToExchange(uint k);
    Builder& pathRelinkBlockSize(uint k);
    Builder& seed(uint s);

    BrkgaConfiguration build();

  private:
    BrkgaConfiguration* config;
  };

  virtual ~BrkgaConfiguration() = default;

  inline Decoder* decoder() { return _decoder; }
  inline const Decoder* decoder() const { return _decoder; }
  inline DecodeType decodeType() const { return _decodeType; }
  inline uint numberOfPopulations() const { return _numberOfPopulations; }
  inline uint populationSize() const { return _populationSize; }
  inline uint chromosomeLength() const { return _chromosomeLength; }
  inline uint numberOfParents() const { return (uint)_bias.size(); }
  inline uint numberOfEliteParents() const { return _numberOfEliteParents; }
  inline const std::vector<float>& bias() const { return _bias; }
  inline uint numberOfElites() const { return _numberOfElites; }
  inline uint numberOfMutants() const { return _numberOfMutants; }
  inline uint numberOfElitesToExchange() const {
    return _numberOfElitesToExchange;
  }
  inline uint pathRelinkBlockSize() const { return _pathRelinkBlockSize; }
  inline uint seed() const { return _seed; }
  inline uint ompThreads() const { return _ompThreads; }
  inline unsigned gpuThreads() const { return (unsigned)_gpuThreads; }

  void setBias(const std::vector<float>& bias, uint numberOfEliteParents);
  void setBias(Bias biasType, uint numberOfEliteParents);
  void setNumberOfElites(uint n);
  void setElitePercentage(float p);
  void setNumberOfMutants(uint n);
  void setMutantPercentage(float p);
  void setNumberOfElitesToExchange(uint k);
  void setPathRelinkBlockSize(uint k);
  void setOmpThreads(uint k);
  void setGpuThreads(uint k);

private:
  friend Builder;

  BrkgaConfiguration()
      : _decoder(nullptr),
        _decodeType(),
        _numberOfPopulations(0),
        _populationSize(0),
        _chromosomeLength(0),
        _numberOfEliteParents(0),
        _bias(),
        _numberOfElites(0),
        _numberOfMutants(0),
        _numberOfElitesToExchange(0),
        _pathRelinkBlockSize(0),
        _seed(0),
        _ompThreads(1),
        _gpuThreads(0) {}

  Decoder* _decoder;  /// The decoder implementation
  DecodeType _decodeType;  /// @see DecodeType.hpp
  uint _numberOfPopulations;  /// Number of independent populations
  uint _populationSize;  /// Size/#chromosomes of each population
  uint _chromosomeLength;  /// The length of the chromosomes
  uint _numberOfEliteParents;  /// Number of elite parents for mating
  std::vector<float> _bias;  /// Probability to select the i-th parent on mating
  uint _numberOfElites;  /// #elites in the population
  uint _numberOfMutants;  /// #mutants in the population
  uint _numberOfElitesToExchange;  /// #elites for @ref exchangeElites
  uint _pathRelinkBlockSize;  /// Block size for @ref pathRelink
  uint _seed;  /// The seed to use
  uint _ompThreads;  /// #threads to use on OpenMP
  uint _gpuThreads;  /// #threads per block for CUDA kernels
};
}  // namespace box

#endif  // BRKGACUDA_BRKGACONFIGURATION_HPP
