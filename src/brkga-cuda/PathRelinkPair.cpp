#include "PathRelinkPair.hpp"

#include "BasicTypes.hpp"
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
    Gene* population,  // TODO add support to GPU
    uint k,
    const ComparatorBase& similar) {
  InvalidArgument::range(Arg<uint>(k, "k"), Arg<uint>(1),
                         Arg<uint>(config.numberOfElites(), "#elites"),
                         3 /* closed */, __FUNCTION__);

  logger::debug("Build Path Relink pairs with the best", k, "elites");

  std::vector<PathRelinkPair> pairs;
  pairs.reserve(config.numberOfPopulations() * k);

  for (uint p = 0; p < config.numberOfPopulations(); ++p) {
    const auto p1 = (p + 1) % config.numberOfPopulations();
    const auto pOff = p * config.populationSize();
    const auto p1Off = p1 * config.populationSize();
    for (uint base = 0; base < k; ++base)
      for (uint guide = 0; guide < config.numberOfElites(); ++guide)
        if (!similar(Chromosome<Gene>(population, config.chromosomeLength(),
                                      pOff + base),
                     Chromosome<Gene>(population, config.chromosomeLength(),
                                      p1Off + guide))) {
          pairs.emplace_back(p, base, p1, guide);
          break;
        }
  }

  if (pairs.empty()) logger::debug("No distinct pairs found on any population");
  return pairs;
}

std::vector<PathRelinkPair> PathRelinkPair::randomElites(
    const BrkgaConfiguration&,
    Gene*,
    uint,
    const ComparatorBase&) {
  // FIXME
  throw NotImplemented(__PRETTY_FUNCTION__);
}
}  // namespace box
