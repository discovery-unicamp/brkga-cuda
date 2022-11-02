#ifndef BOX_PATH_RELINK_PAIR_HPP
#define BOX_PATH_RELINK_PAIR_HPP

#include "BasicTypes.hpp"
#include "BrkgaConfiguration.hpp"
#include "Comparator.hpp"

#include <vector>

namespace box {
/**
 * Pair of chromosomes to run the Path Relink algorithm.
 *
 * The newly created chromosome will be inserted in the same population of the
 * base chromosome.
 */
class PathRelinkPair {
public:
  // TODO how to check if two permutations are different?

  /**
   * Selects the @p k best elites of each population to run the Path Relink.
   *
   * Each of the best @p k elites of the population p are paired with the first
   * elite of the population (p + 1) % @p config.numberOfPopulations for which
   * @p similar returns false. If no such chromosome exists, then the current
   * elite will be ignored.
   */
  static std::vector<PathRelinkPair> bestElites(
      const BrkgaConfiguration& config,
      Gene* population,
      uint k,
      const ComparatorBase& similar);

  /**
   * Selects @p k random elites of each population to run the Path Relink.
   *
   * Each of the @p k random elites of the population p are paired with a random
   * elite of the population (p + 1) % @p config.numberOfPopulations for which
   * @p similar returns false.
   */
  /// NOT IMPLEMENTED YET.
  static std::vector<PathRelinkPair> randomElites(
      const BrkgaConfiguration& config,
      Gene* population,
      uint k,
      const ComparatorBase& similar);

  PathRelinkPair() : PathRelinkPair(-1u, -1u, -1u, -1u) {}

  PathRelinkPair(uint _basePopulationId,
                 uint _baseChromosomeId,
                 uint _guidePopulationId,
                 uint _guideChromosomeId)
      : basePopulationId(_basePopulationId),
        baseChromosomeId(_baseChromosomeId),
        guidePopulationId(_guidePopulationId),
        guideChromosomeId(_guideChromosomeId) {}

  uint basePopulationId;
  uint baseChromosomeId;
  uint guidePopulationId;
  uint guideChromosomeId;
};
}  // namespace box

#endif  // BOX_PATH_RELINK_PAIR_HPP
