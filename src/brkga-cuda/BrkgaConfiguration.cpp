#include "BrkgaConfiguration.hpp"

#include "Logger.hpp"
#include "except/InvalidArgument.hpp"

const unsigned MAX_GPU_THREADS = 1024;

namespace box {
BrkgaConfiguration::Builder::Builder() : config(new BrkgaConfiguration) {}

BrkgaConfiguration::Builder::~Builder() {
  delete config;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::decoder(Decoder* d) {
  InvalidArgument::null(Arg<Decoder*>(d, "decoder"), __FUNCTION__);
  config->_decoder = d;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::decodeType(
    DecodeType dt) {
  config->_decodeType = dt;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::numberOfPopulations(
    unsigned n) {
  InvalidArgument::min(Arg<unsigned>(n, "#populations"), Arg<unsigned>(1),
                       __FUNCTION__);
  config->_numberOfPopulations = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::populationSize(
    unsigned n) {
  InvalidArgument::min(Arg<unsigned>(n, "population size"), Arg<unsigned>(3),
                       __FUNCTION__);
  config->_populationSize = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::chromosomeLength(
    unsigned n) {
  InvalidArgument::min(Arg<unsigned>(n, "chromosome length"), Arg<unsigned>(2),
                       __FUNCTION__);
  config->_chromosomeLength = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::numberOfElites(
    unsigned n) {
  if (config->_populationSize == 0)
    throw InvalidArgument(
        "You should define the population size before #elites", __FUNCTION__);
  config->setNumberOfElites(n);
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::elitePercentage(
    float p) {
  if (config->_populationSize == 0)
    throw InvalidArgument("You should define the population size before elite%",
                          __FUNCTION__);
  config->setElitePercentage(p);
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::numberOfMutants(
    unsigned n) {
  if (config->_populationSize == 0)
    throw InvalidArgument(
        "You should define the population size before #mutants", __FUNCTION__);
  config->setNumberOfMutants(n);
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::mutantPercentage(
    float p) {
  if (config->_populationSize == 0)
    throw InvalidArgument(
        "You should define the population size before mutant%", __FUNCTION__);
  return numberOfMutants((unsigned)(p * (float)config->_populationSize));
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::rhoe(float r) {
  config->setRhoe(r);
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::seed(unsigned s) {
  config->_seed = s;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::ompThreads(
    unsigned k) {
  config->setOmpThreads(k);
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::gpuThreads(
    unsigned k) {
  config->setGpuThreads(k);
  return *this;
}

BrkgaConfiguration BrkgaConfiguration::Builder::build() {
  if (config->_decoder == nullptr)
    throw InvalidArgument("Decoder wasn't set", __FUNCTION__);
  if (config->_numberOfPopulations == 0)
    throw InvalidArgument("#populations wasn't set", __FUNCTION__);
  if (config->_populationSize == 0)
    throw InvalidArgument("Population size wasn't set", __FUNCTION__);
  if (config->_chromosomeLength == 0)
    throw InvalidArgument("Chromosome length wasn't set", __FUNCTION__);
  if (config->_numberOfElites == 0)
    throw InvalidArgument("#elites wasn't set", __FUNCTION__);
  if (config->_numberOfMutants == 0)
    throw InvalidArgument("#mutants wasn't set", __FUNCTION__);
  if (config->_rhoe < 0.5f)
    throw InvalidArgument("Rhoe wasn't set", __FUNCTION__);
  if (config->_gpuThreads == 0)
    throw InvalidArgument("Threads per block wasn't set", __FUNCTION__);
  return *config;
}

void BrkgaConfiguration::setNumberOfElites(unsigned n) {
  InvalidArgument::range(Arg<unsigned>(n, "#elites"), Arg<unsigned>(1),
                         Arg<unsigned>(_populationSize - _numberOfMutants,
                                       "population - #mutants"),
                         2 /* start closed */, __FUNCTION__);
  _numberOfElites = n;
}

void BrkgaConfiguration::setElitePercentage(float p) {
  InvalidArgument::range(Arg<float>(p, "elite%"), Arg<float>(0), Arg<float>(1),
                         0 /* open range */, __FUNCTION__);
  setNumberOfElites((unsigned)(p * (float)_populationSize));
}

void BrkgaConfiguration::setNumberOfMutants(unsigned n) {
  InvalidArgument::range(
      Arg<unsigned>(n, "#mutants"), Arg<unsigned>(1),
      Arg<unsigned>(_populationSize - _numberOfElites, "population - #elites"),
      2 /* start closed */, __FUNCTION__);
  _numberOfMutants = n;
}

void BrkgaConfiguration::setMutantPercentage(float p) {
  InvalidArgument::range(Arg<float>(p, "mutant%"), Arg<float>(0), Arg<float>(1),
                         0 /* open range */, __FUNCTION__);
  setNumberOfMutants((unsigned)(p * (float)_populationSize));
}

void BrkgaConfiguration::setRhoe(float r) {
  InvalidArgument::range(Arg<float>(r, "rhoe"), Arg<float>(.5f), Arg<float>(1),
                         0 /* open range */, __FUNCTION__);
  _rhoe = r;
}

void BrkgaConfiguration::setOmpThreads(unsigned k) {
#ifndef _OPENMP
  if (k > 1)
    throw std::logic_error(format(
        "OpenMP wasn't enabled; cannot set the number of threads to", k));
#endif  //_OPENMP
  InvalidArgument::min(Arg<unsigned>(k, "OpenMP threads"), Arg<unsigned>(1),
                       __FUNCTION__);
  _ompThreads = k;
}

void BrkgaConfiguration::setGpuThreads(unsigned k) {
  InvalidArgument::range(Arg<unsigned>(k, "gpu threads"), Arg<unsigned>(1),
                         Arg<unsigned>(MAX_GPU_THREADS, "CUDA limit"),
                         3 /* closed range */, __FUNCTION__);
  _gpuThreads = k;
}
}  // namespace box
