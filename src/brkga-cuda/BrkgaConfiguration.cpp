#include "BrkgaConfiguration.hpp"

#include "Logger.hpp"
#include "except/InvalidArgument.hpp"

#include <cmath>
#include <functional>

const unsigned MAX_GPU_THREADS = 1024;

namespace box {
BrkgaConfiguration::Builder::Builder() : config(new BrkgaConfiguration) {}

BrkgaConfiguration::Builder::~Builder() {
  delete config;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::decoder(Decoder* d) {
  InvalidArgument::null(Arg<Decoder*>(d, "decoder"), BOX_FUNCTION);
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
                       BOX_FUNCTION);
  config->_numberOfPopulations = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::populationSize(
    unsigned n) {
  InvalidArgument::min(Arg<unsigned>(n, "|population|"), Arg<unsigned>(3),
                       BOX_FUNCTION);
  config->_populationSize = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::chromosomeLength(
    unsigned n) {
  InvalidArgument::min(Arg<unsigned>(n, "|chromosome|"), Arg<unsigned>(2),
                       BOX_FUNCTION);
  config->_chromosomeLength = n;
  return *this;
}

std::vector<float> processBias(std::vector<float> bias) {
  InvalidArgument::min(Arg<float>(bias[0], "bias[0]"), Arg<float>(0.1f),
                       BOX_FUNCTION);
  for (unsigned i = 1; i < bias.size(); ++i) {
    InvalidArgument::max(
        Arg<float>(bias[i], format(Separator(""), "bias[", i, "]")),
        Arg<float>(bias[i - 1], format(Separator(""), "bias[", i - 1, "]")),
        BOX_FUNCTION);
  }

  for (unsigned i = 1; i < bias.size(); ++i) bias[i] += bias[i - 1];
  return bias;
}

std::vector<float> buildBias(unsigned n, Bias biasType) {
  std::function<float(unsigned)> generator;
  switch (biasType) {
    case Bias::CONSTANT:
      generator = [n](unsigned) { return 1 / (float)n; };
      break;

    case Bias::LINEAR:
      generator = [](unsigned i) { return 1 / (float)(i + 1); };
      break;

    case Bias::QUADRATIC:
      generator = [](unsigned i) { return 1 / powf((float)(i + 1), 2); };
      break;

    case Bias::CUBIC:
      generator = [](unsigned i) { return 1 / powf((float)(i + 1), 3); };
      break;

    case Bias::EXPONENTIAL:
      generator = [](unsigned i) { return expf(-(float)i); };
      break;

    case Bias::LOGARITHM:
      generator = [](unsigned i) { return 1 / logf((float)(i + 2)); };
      break;

    default:
      throw std::logic_error("Missing implementation for bias");
  }

  std::vector<float> bias(n);
  for (unsigned i = 0; i < n; ++i) bias[i] = generator(i);
  return bias;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::parents(
    const std::vector<float>& bias,
    unsigned numberOfElites) {
  InvalidArgument::range(Arg<unsigned>((unsigned)bias.size(), "#parents"),
                         Arg<unsigned>(2),
                         Arg<unsigned>(config->_populationSize, "|population|"),
                         3 /* closed */, BOX_FUNCTION);
  InvalidArgument::range(Arg<unsigned>(numberOfElites, "#elite parents"),
                         Arg<unsigned>(1),
                         Arg<unsigned>((unsigned)bias.size(), "#parents"),
                         2 /* start closed */, BOX_FUNCTION);

  config->_bias = processBias(bias);
  config->_numberOfEliteParents = numberOfElites;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::parents(
    unsigned n,
    Bias biasType,
    unsigned numberOfElites) {
  return parents(buildBias(n, biasType), numberOfElites);
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::numberOfElites(
    unsigned n) {
  if (config->_populationSize == 0)
    throw InvalidArgument(
        "You should define the population size before #elites", BOX_FUNCTION);
  config->setNumberOfElites(n);
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::elitePercentage(
    float p) {
  if (config->_populationSize == 0)
    throw InvalidArgument("You should define the population size before elite%",
                          BOX_FUNCTION);
  config->setElitePercentage(p);
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::numberOfMutants(
    unsigned n) {
  if (config->_populationSize == 0)
    throw InvalidArgument(
        "You should define the population size before #mutants", BOX_FUNCTION);
  config->setNumberOfMutants(n);
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::mutantPercentage(
    float p) {
  if (config->_populationSize == 0)
    throw InvalidArgument(
        "You should define the population size before mutant%", BOX_FUNCTION);
  return numberOfMutants((unsigned)(p * (float)config->_populationSize));
}

BrkgaConfiguration::Builder&
BrkgaConfiguration::Builder::numberOfElitesToExchange(unsigned k) {
  config->setNumberOfElitesToExchange(k);
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::pathRelinkBlockSize(
    unsigned k) {
  config->setPathRelinkBlockSize(k);
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
    throw InvalidArgument("Decoder wasn't set", BOX_FUNCTION);
  if (config->_numberOfPopulations == 0)
    throw InvalidArgument("#populations wasn't set", BOX_FUNCTION);
  if (config->_populationSize == 0)
    throw InvalidArgument("Population size wasn't set", BOX_FUNCTION);
  if (config->_chromosomeLength == 0)
    throw InvalidArgument("Chromosome length wasn't set", BOX_FUNCTION);
  if (config->_numberOfElites == 0)
    throw InvalidArgument("#elites wasn't set", BOX_FUNCTION);
  if (config->_numberOfMutants == 0)
    throw InvalidArgument("#mutants wasn't set", BOX_FUNCTION);
  if (config->_bias.empty())
    throw InvalidArgument("Bias wasn't set", BOX_FUNCTION);
  if (config->_gpuThreads == 0)
    throw InvalidArgument("Threads per block wasn't set", BOX_FUNCTION);
  return *config;
}

void BrkgaConfiguration::setBias(const std::vector<float>& bias,
                                 unsigned numberOfEliteParents) {
  InvalidArgument::diff(Arg<unsigned>((unsigned)bias.size(), "bias size"),
                        Arg<unsigned>((unsigned)_bias.size(), "current size"),
                        BOX_FUNCTION);
  _bias = processBias(bias);
  _numberOfEliteParents = numberOfEliteParents;
}

void BrkgaConfiguration::setBias(Bias biasType, unsigned numberOfEliteParents) {
  setBias(buildBias((unsigned)_bias.size(), biasType), numberOfEliteParents);
}

void BrkgaConfiguration::setNumberOfElites(unsigned n) {
  InvalidArgument::range(Arg<unsigned>(n, "#elites"), Arg<unsigned>(1),
                         Arg<unsigned>(_populationSize - _numberOfMutants,
                                       "|population| - #mutants"),
                         2 /* start closed */, BOX_FUNCTION);
  _numberOfElites = n;
}

void BrkgaConfiguration::setElitePercentage(float p) {
  InvalidArgument::range(Arg<float>(p, "elite%"), Arg<float>(0), Arg<float>(1),
                         0 /* open */, BOX_FUNCTION);
  setNumberOfElites((unsigned)(p * (float)_populationSize));
}

void BrkgaConfiguration::setNumberOfMutants(unsigned n) {
  InvalidArgument::range(Arg<unsigned>(n, "#mutants"), Arg<unsigned>(1),
                         Arg<unsigned>(_populationSize - _numberOfElites,
                                       "|population| - #elites"),
                         2 /* start closed */, BOX_FUNCTION);
  _numberOfMutants = n;
}

void BrkgaConfiguration::setMutantPercentage(float p) {
  InvalidArgument::range(Arg<float>(p, "mutant%"), Arg<float>(0), Arg<float>(1),
                         0 /* open */, BOX_FUNCTION);
  setNumberOfMutants((unsigned)(p * (float)_populationSize));
}

void BrkgaConfiguration::setNumberOfElitesToExchange(unsigned k) {
  InvalidArgument::range(Arg<unsigned>(k, "exchange count"), Arg<unsigned>(0),
                         Arg<unsigned>(_numberOfElites, "#elites"),
                         3 /* closed */, BOX_FUNCTION);
  InvalidArgument::range(Arg<unsigned>(k, "exchange count"), Arg<unsigned>(0),
                         Arg<unsigned>(_populationSize / _numberOfPopulations,
                                       "|population| / #populations"),
                         3 /* closed */, BOX_FUNCTION);
  _numberOfElitesToExchange = k;
}

void BrkgaConfiguration::setPathRelinkBlockSize(unsigned k) {
  InvalidArgument::range(Arg<unsigned>(k, "pr block size"), Arg<unsigned>(1),
                         Arg<unsigned>(_chromosomeLength, "|chromosome|"),
                         2 /* start closed */, BOX_FUNCTION);
  _pathRelinkBlockSize = k;
}

void BrkgaConfiguration::setOmpThreads(unsigned k) {
#ifndef _OPENMP
  if (k > 1)
    throw std::logic_error(format(
        "OpenMP wasn't enabled; cannot set the number of threads to", k));
#endif  //_OPENMP
  InvalidArgument::min(Arg<unsigned>(k, "OpenMP threads"), Arg<unsigned>(1),
                       BOX_FUNCTION);
  _ompThreads = k;
}

void BrkgaConfiguration::setGpuThreads(unsigned k) {
  InvalidArgument::range(Arg<unsigned>(k, "gpu threads"), Arg<unsigned>(1),
                         Arg<unsigned>(MAX_GPU_THREADS, "CUDA limit"),
                         3 /* closed */, BOX_FUNCTION);
  _gpuThreads = k;
}
}  // namespace box
