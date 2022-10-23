#include "PathRelinkPair.hpp"

#include "Chromosome.hpp"
#include "Logger.hpp"
#include "except/InvalidArgument.hpp"
#include "except/NotImplemented.hpp"

#include <random>

// FIXME use the provided seed?
std::mt19937 rng(0);

namespace box {
std::vector<PathRelinkPair> PathRelinkPair::bestElites(
    const BrkgaConfiguration& config,
    float* population,  // TODO add support to GPU
    unsigned k,
    const ComparatorBase& similar) {
  InvalidArgument::range(Arg<unsigned>(k, "k"), Arg<unsigned>(1),
                         Arg<unsigned>(config.numberOfElites(), "#elites"),
                         3 /* closed */, __FUNCTION__);

  logger::debug("Build Path Relink pairs with the best", k, "elites");

  std::vector<PathRelinkPair> pairs;
  pairs.reserve(config.numberOfPopulations() * k);

  for (unsigned p = 0; p < config.numberOfPopulations(); ++p) {
    const auto p1 = (p + 1) % config.numberOfPopulations();
    const auto pOff = p * config.populationSize();
    const auto p1Off = p1 * config.populationSize();
    for (unsigned base = 0; base < k; ++base)
      for (unsigned guide = 0; guide < config.numberOfElites(); ++guide)
        if (!similar(Chromosome<float>(population, config.chromosomeLength(),
                                       pOff + base),
                     Chromosome<float>(population, config.chromosomeLength(),
                                       p1Off + guide))) {
          pairs.emplace_back(p, base, p1, guide);
          break;
        }
  }

  if (pairs.empty())
    logger::debug("No distinct pairs found on any population");
  return pairs;
}

std::vector<PathRelinkPair> PathRelinkPair::randomElites(
    const BrkgaConfiguration&,
    float*,
    unsigned,
    const ComparatorBase&) {
  // FIXME
  throw NotImplemented(__PRETTY_FUNCTION__);
}
}  // namespace box
