#include "BrkgaConfiguration.hpp"

#include "Logger.hpp"
#include "except/InvalidArgument.hpp"

namespace box {
BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::decoder(Decoder* d) {
  InvalidArgument::null(Arg<Decoder*>(d, "decoder"), __FUNCTION__);
  _decoder = d;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::threadsPerBlock(
    unsigned k) {
  InvalidArgument::range(Arg<unsigned>(k, "threads per block"),
                         Arg<unsigned>(1), Arg<unsigned>(1024),
                         3 /* closed range */, __FUNCTION__);
  _threadsPerBlock = k;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::ompThreads(
    unsigned k) {
#ifndef _OPENMP
  if (k > 1)
    throw std::logic_error(format(
        "OpenMP wasn't enabled; cannot set the number of threads to", k));
#endif  //_OPENMP
  InvalidArgument::min(Arg<unsigned>(k, "OpenMP threads"), Arg<unsigned>(1),
                       __FUNCTION__);
  _ompThreads = k;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::numberOfPopulations(
    unsigned n) {
  InvalidArgument::min(Arg<unsigned>(n, "#populations"), Arg<unsigned>(1),
                       __FUNCTION__);
  _numberOfPopulations = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::populationSize(
    unsigned n) {
  InvalidArgument::min(Arg<unsigned>(n, "population size"), Arg<unsigned>(3),
                       __FUNCTION__);
  _populationSize = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::chromosomeLength(
    unsigned n) {
  InvalidArgument::min(Arg<unsigned>(n, "chromosome length"), Arg<unsigned>(2),
                       __FUNCTION__);
  _chromosomeLength = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::eliteCount(
    unsigned n) {
  if (_populationSize == 0)
    throw InvalidArgument(
        "You should define the population size before #elites", __FUNCTION__);
  InvalidArgument::range(
      Arg<unsigned>(n, "#elites"), Arg<unsigned>(1),
      Arg<unsigned>(_populationSize - _mutantsCount, "population - #mutants"),
      2 /* start closed */, __FUNCTION__);
  _eliteCount = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::eliteFactor(float p) {
  if (_populationSize == 0)
    throw InvalidArgument("You should define the population size before elite%",
                          __FUNCTION__);
  InvalidArgument::range(Arg<float>(p, "elite%"), Arg<float>(0), Arg<float>(1),
                         0 /* open range */, __FUNCTION__);
  return eliteCount((unsigned)(p * (float)_populationSize));
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::mutantsCount(
    unsigned n) {
  if (_populationSize == 0)
    throw InvalidArgument(
        "You should define the population size before #mutants", __FUNCTION__);
  InvalidArgument::range(
      Arg<unsigned>(n, "#mutants"), Arg<unsigned>(1),
      Arg<unsigned>(_populationSize - _eliteCount, "population - #elites"),
      2 /* start closed */, __FUNCTION__);
  _mutantsCount = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::mutantsFactor(
    float p) {
  if (_populationSize == 0)
    throw InvalidArgument(
        "You should define the population size before mutant%", __FUNCTION__);
  InvalidArgument::range(Arg<float>(p, "mutant%"), Arg<float>(0), Arg<float>(1),
                         0 /* open range */, __FUNCTION__);
  return mutantsCount((unsigned)(p * (float)_populationSize));
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::rhoe(float r) {
  InvalidArgument::range(Arg<float>(r, "rhoe"), Arg<float>(.5f), Arg<float>(1),
                         0 /* open range */, __FUNCTION__);
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
    throw InvalidArgument("#populations wasn't set", __FUNCTION__);
  if (_populationSize == 0)
    throw InvalidArgument("Population size wasn't set", __FUNCTION__);
  if (_chromosomeLength == 0)
    throw InvalidArgument("Chromosome length wasn't set", __FUNCTION__);
  if (_eliteCount == 0)
    throw InvalidArgument("#elites wasn't set", __FUNCTION__);
  if (_mutantsCount == 0)
    throw InvalidArgument("#mutants wasn't set", __FUNCTION__);
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
