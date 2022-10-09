#include "BrkgaConfiguration.hpp"

#include "Logger.hpp"
#include "except/InvalidArgument.hpp"

namespace box {
BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::decoder(Decoder* d) {
  InvalidArgument::null("Decoder", d, __FUNCTION__);
  _decoder = d;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::threadsPerBlock(
    unsigned k) {
  InvalidArgument::range("Threads per block", k, 1u, 1024u,
                         3 /* closed range */, __FUNCTION__);
  _threadsPerBlock = k;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::ompThreads(
    unsigned k) {
#ifndef _OPENMP
  if (k > 1)
    throw std::logic_error(format(
        "OpenMP wasn't enabled; cannot set the number of threads to ", k));
#endif  //_OPENMP
  InvalidArgument::min("OpenMP threads", k, 1u, __FUNCTION__);
  _ompThreads = k;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::numberOfPopulations(
    unsigned n) {
  InvalidArgument::min("Number of populations", n, 1u, __FUNCTION__);
  _numberOfPopulations = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::populationSize(
    unsigned n) {
  InvalidArgument::min("Population size", n, 3u, __FUNCTION__);
  _populationSize = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::chromosomeLength(
    unsigned n) {
  InvalidArgument::min("Chromosome length", n, 2u, __FUNCTION__);
  _chromosomeLength = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::eliteCount(
    unsigned n) {
  InvalidArgument::range("Number of elites", n, 1u,
                         _populationSize - _mutantsCount, 2 /* start closed */,
                         __FUNCTION__);
  _eliteCount = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::eliteFactor(float p) {
  InvalidArgument::range("Elite proportion", p, 0.0f, 1.0f, 0 /* open range */,
                         __FUNCTION__);
  return eliteCount((unsigned)(p * (float)_populationSize));
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::mutantsCount(
    unsigned n) {
  InvalidArgument::range("Number of mutants", n, 1u,
                         _populationSize - _eliteCount, 2 /* start closed */,
                         __FUNCTION__);
  _mutantsCount = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::mutantsFactor(
    float p) {
  InvalidArgument::range("Mutant proportion", p, 0.0f, 1.0f, 0 /* open range */,
                         __FUNCTION__);
  return mutantsCount((unsigned)(p * (float)_populationSize));
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::rhoe(float r) {
  InvalidArgument::range("Rhoe", r, 0.5f, 1.0f, 0 /* open range */,
                         __FUNCTION__);
  _rhoe = r;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::seed(unsigned s) {
  _seed = s;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::decodeType(
    DecodeType dt) {
  _decodeType = dt;
  return *this;
}

BrkgaConfiguration BrkgaConfiguration::Builder::build() const {
  if (_decoder == nullptr)
    throw InvalidArgument("Decoder wasn't set", __FUNCTION__);
  if (_threadsPerBlock == 0)
    throw InvalidArgument("Threads per block wasn't set", __FUNCTION__);
  if (_numberOfPopulations == 0)
    throw InvalidArgument("Number of populations wasn't set", __FUNCTION__);
  if (_populationSize == 0)
    throw InvalidArgument("Population size wasn't set", __FUNCTION__);
  if (_chromosomeLength == 0)
    throw InvalidArgument("Chromosome length wasn't set", __FUNCTION__);
  if (_eliteCount == 0)
    throw InvalidArgument("Elite count wasn't set", __FUNCTION__);
  if (_mutantsCount == 0)
    throw InvalidArgument("Mutants count wasn't set", __FUNCTION__);
  if (_rhoe < 1e-7f) throw InvalidArgument("Rhoe wasn't set", __FUNCTION__);

  BrkgaConfiguration config;
  config.decoder = _decoder;
  config.threadsPerBlock = _threadsPerBlock;
  config.ompThreads = _ompThreads;
  config.numberOfPopulations = _numberOfPopulations;
  config.populationSize = _populationSize;
  config.chromosomeLength = _chromosomeLength;
  config.eliteCount = _eliteCount;
  config.mutantsCount = _mutantsCount;
  config.rhoe = _rhoe;
  config.seed = _seed;
  config.decodeType = _decodeType;

  return config;
}
}  // namespace box
