#include "BrkgaConfiguration.hpp"

#include "BasicTypes.hpp"
#include "Logger.hpp"
#include "except/InvalidArgument.hpp"

#include <cmath>
#include <functional>

const uint MAX_GPU_THREADS = 1024;

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
    uint n) {
  InvalidArgument::min(Arg<uint>(n, "#populations"), Arg<uint>(1),
                       BOX_FUNCTION);
  config->_numberOfPopulations = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::populationSize(
    uint n) {
  InvalidArgument::min(Arg<uint>(n, "|population|"), Arg<uint>(3),
                       BOX_FUNCTION);
  config->_populationSize = n;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::chromosomeLength(
    uint n) {
  InvalidArgument::min(Arg<uint>(n, "|chromosome|"), Arg<uint>(2),
                       BOX_FUNCTION);
  config->_chromosomeLength = n;
  return *this;
}

std::vector<float> processBias(std::vector<float> bias) {
  InvalidArgument::min(Arg<float>(bias[0], "bias[0]"), Arg<float>(0.1f),
                       BOX_FUNCTION);
  for (uint i = 1; i < bias.size(); ++i) {
    InvalidArgument::max(
        Arg<float>(bias[i], format(Separator(""), "bias[", i, "]")),
        Arg<float>(bias[i - 1], format(Separator(""), "bias[", i - 1, "]")),
        BOX_FUNCTION);
  }

  for (uint i = 1; i < bias.size(); ++i) bias[i] += bias[i - 1];
  return bias;
}

std::vector<float> buildBias(uint n, Bias biasType) {
  std::function<float(uint)> generator;
  switch (biasType) {
    case Bias::CONSTANT:
      generator = [n](uint) { return 1 / (float)n; };
      break;

    case Bias::LINEAR:
      generator = [](uint i) { return 1 / (float)(i + 1); };
      break;

    case Bias::QUADRATIC:
      generator = [](uint i) { return 1 / powf((float)(i + 1), 2); };
      break;

    case Bias::CUBIC:
      generator = [](uint i) { return 1 / powf((float)(i + 1), 3); };
      break;

    case Bias::EXPONENTIAL:
      generator = [](uint i) { return expf(-(float)i); };
      break;

    case Bias::LOGARITHM:
      generator = [](uint i) { return 1 / logf((float)(i + 2)); };
      break;

    default:
      throw std::logic_error("Missing implementation for bias");
  }

  std::vector<float> bias(n);
  for (uint i = 0; i < n; ++i) bias[i] = generator(i);
  return bias;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::parents(
    const std::vector<float>& bias,
    uint numberOfElites) {
  InvalidArgument::range(Arg<uint>((uint)bias.size(), "#parents"), Arg<uint>(2),
                         Arg<uint>(config->_populationSize, "|population|"),
                         3 /* closed */, BOX_FUNCTION);
  InvalidArgument::range(Arg<uint>(numberOfElites, "#elite parents"),
                         Arg<uint>(1), Arg<uint>((uint)bias.size(), "#parents"),
                         2 /* start closed */, BOX_FUNCTION);

  config->_bias = processBias(bias);
  config->_numberOfEliteParents = numberOfElites;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::parents(
    uint n,
    Bias biasType,
    uint numberOfElites) {
  return parents(buildBias(n, biasType), numberOfElites);
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::numberOfElites(
    uint n) {
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
    uint n) {
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
  return numberOfMutants((uint)(p * (float)config->_populationSize));
}

BrkgaConfiguration::Builder&
BrkgaConfiguration::Builder::numberOfElitesToExchange(uint k) {
  config->setNumberOfElitesToExchange(k);
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::pathRelinkBlockSize(
    uint k) {
  config->setPathRelinkBlockSize(k);
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::seed(uint s) {
  config->_seed = s;
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::ompThreads(uint k) {
  config->setOmpThreads(k);
  return *this;
}

BrkgaConfiguration::Builder& BrkgaConfiguration::Builder::gpuThreads(uint k) {
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
                                 uint numberOfEliteParents) {
  InvalidArgument::diff(Arg<uint>((uint)bias.size(), "bias size"),
                        Arg<uint>((uint)_bias.size(), "current size"),
                        BOX_FUNCTION);
  _bias = processBias(bias);
  _numberOfEliteParents = numberOfEliteParents;
}

void BrkgaConfiguration::setBias(Bias biasType, uint numberOfEliteParents) {
  setBias(buildBias((uint)_bias.size(), biasType), numberOfEliteParents);
}

void BrkgaConfiguration::setNumberOfElites(uint n) {
  InvalidArgument::range(
      Arg<uint>(n, "#elites"), Arg<uint>(1),
      Arg<uint>(_populationSize - _numberOfMutants, "|population| - #mutants"),
      2 /* start closed */, BOX_FUNCTION);
  _numberOfElites = n;
}

void BrkgaConfiguration::setElitePercentage(float p) {
  InvalidArgument::range(Arg<float>(p, "elite%"), Arg<float>(0), Arg<float>(1),
                         0 /* open */, BOX_FUNCTION);
  setNumberOfElites((uint)(p * (float)_populationSize));
}

void BrkgaConfiguration::setNumberOfMutants(uint n) {
  InvalidArgument::range(
      Arg<uint>(n, "#mutants"), Arg<uint>(1),
      Arg<uint>(_populationSize - _numberOfElites, "|population| - #elites"),
      2 /* start closed */, BOX_FUNCTION);
  _numberOfMutants = n;
}

void BrkgaConfiguration::setMutantPercentage(float p) {
  InvalidArgument::range(Arg<float>(p, "mutant%"), Arg<float>(0), Arg<float>(1),
                         0 /* open */, BOX_FUNCTION);
  setNumberOfMutants((uint)(p * (float)_populationSize));
}

void BrkgaConfiguration::setNumberOfElitesToExchange(uint k) {
  InvalidArgument::range(Arg<uint>(k, "exchange count"), Arg<uint>(0),
                         Arg<uint>(_numberOfElites, "#elites"), 3 /* closed */,
                         BOX_FUNCTION);
  InvalidArgument::range(Arg<uint>(k, "exchange count"), Arg<uint>(0),
                         Arg<uint>(_populationSize / _numberOfPopulations,
                                   "|population| / #populations"),
                         3 /* closed */, BOX_FUNCTION);
  _numberOfElitesToExchange = k;
}

void BrkgaConfiguration::setPathRelinkBlockSize(uint k) {
  InvalidArgument::range(Arg<uint>(k, "pr block size"), Arg<uint>(1),
                         Arg<uint>(_chromosomeLength, "|chromosome|"),
                         2 /* start closed */, BOX_FUNCTION);
  _pathRelinkBlockSize = k;
}

void BrkgaConfiguration::setOmpThreads(uint k) {
#ifndef _OPENMP
  if (k > 1)
    throw std::logic_error(format(
        "OpenMP wasn't enabled; cannot set the number of threads to", k));
#endif  //_OPENMP
  InvalidArgument::min(Arg<uint>(k, "OpenMP threads"), Arg<uint>(1),
                       BOX_FUNCTION);
  _ompThreads = k;
}

void BrkgaConfiguration::setGpuThreads(uint k) {
  InvalidArgument::range(Arg<uint>(k, "gpu threads"), Arg<uint>(1),
                         Arg<uint>(MAX_GPU_THREADS, "CUDA limit"),
                         3 /* closed */, BOX_FUNCTION);
  _gpuThreads = k;
}
}  // namespace box
