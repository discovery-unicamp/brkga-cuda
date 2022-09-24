#include "Logger.hpp"
#include "PathRelinkPair.hpp"

#include <random>

// FIXME use the provided seed?
std::mt19937 rng(0);

namespace box {
std::vector<PathRelinkPair> PathRelinkPair::bestElites(
    unsigned numberOfPopulations,
    unsigned numberOfElites,
    unsigned k) {
  if (k < 1) throw std::invalid_argument("k should be at least 2");
  if (k > numberOfElites)
    throw std::invalid_argument("k should not exceed the number of elites");

  logger::debug("Build Path Relink pairs with the best", k, "elites");

  std::vector<PathRelinkPair> prPairs;
  prPairs.reserve(numberOfPopulations * k);

  std::vector<unsigned> bases(k);
  std::vector<unsigned> guides(k);
  for (unsigned p = 0; p < numberOfPopulations; ++p) {
    for (unsigned i = 0; i < k; ++i) {
      bases[i] = i;
      guides[i] = i;
    }

    const auto p1 = (p + 1) % numberOfPopulations;
    for (unsigned i = 0; i < k; ++i) {
      std::uniform_int_distribution<unsigned> uid(0, k - i - 1);
      const auto b = uid(rng);
      const auto g = uid(rng);
      prPairs.emplace_back(p, bases[b], p1, guides[g]);
      std::swap(bases[b], bases.back());
      std::swap(guides[g], guides.back());
    }
  }

  return prPairs;
}

std::vector<PathRelinkPair> PathRelinkPair::randomElites(unsigned,
                                                         unsigned,
                                                         unsigned) {
  // FIXME
  logger::error(__PRETTY_FUNCTION__, "wasn't implemented");
  abort();
}
}  // namespace box
