#ifndef BOX_PATH_RELINK_PAIR_HPP
#define BOX_PATH_RELINK_PAIR_HPP

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
  // TODO how to check if the chromosomes are different?

  /**
   * Selects the \p k best elites of each population to run the Path Relink.
   *
   * The elites of the population p are paired with the ones in the population
   * (p + 1) % \p numberOfPopulations randomly.
   *
   * \param numberOfPopulations The number of populations in the BRKGA.
   * \param numberOfElites The number of elites in the BRKGA.
   * \param k The number of elites to take.
   * \return A vector with the generated pairs.
   */
  static std::vector<PathRelinkPair> bestElites(unsigned numberOfPopulations,
                                                unsigned numberOfElites,
                                                unsigned k);

  /**
   * Selects \p k random elites of each population to run the Path Relink.
   *
   * The elites of the population p are paired with the ones in the population
   * (p + 1) % \p numberOfPopulations randomly.
   *
   * \param numberOfPopulations The number of populations in the BRKGA.
   * \param numberOfElites The number of elites in the BRKGA.
   * \param k The number of elites to take.
   * \return A vector with the generated pairs.
   */
  static std::vector<PathRelinkPair> randomElites(unsigned numberOfPopulations,
                                                  unsigned numberOfElites,
                                                  unsigned k);

  PathRelinkPair() : PathRelinkPair(-1u, -1u, -1u, -1u) {}

  PathRelinkPair(unsigned _basePopulationId,
                 unsigned _baseChromosomeId,
                 unsigned _guidePopulationId,
                 unsigned _guideChromosomeId)
      : basePopulationId(_basePopulationId),
        baseChromosomeId(_baseChromosomeId),
        guidePopulationId(_guidePopulationId),
        guideChromosomeId(_guideChromosomeId) {}

  unsigned basePopulationId;
  unsigned baseChromosomeId;
  unsigned guidePopulationId;
  unsigned guideChromosomeId;
};
}  // namespace box

#endif  // BOX_PATH_RELINK_PAIR_HPP
