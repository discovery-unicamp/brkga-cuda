#include "../Brkga.hpp"
#include "../BrkgaConfiguration.hpp"
#include "../Chromosome.hpp"
#include "../Decoder.hpp"
#include "../PathRelinkPair.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

using box::BrkgaConfiguration;
using std::begin;
using std::end;

typedef std::vector<float> Chromosome;
typedef float fitness_t;
typedef float gene_t;

const auto MINIMIZE = false;

const float FITNESS_T_MIN = -1e30f;
const float FITNESS_T_MAX = 1e30f;

std::chrono::system_clock::time_point pr_start_time;

std::mt19937 rng(0);  // FIXME

class KendallTauDistance {
public:
  /// Default constructor.
  KendallTauDistance() {}

  /// Default destructor.
  virtual ~KendallTauDistance() {}

  /**
   * \brief Computes the Kendall Tau distance between two vectors.
   * \param vector1 first vector
   * \param vector2 second vector
   */
  virtual double distance(const std::vector<gene_t>& vector1,
                          const std::vector<gene_t>& vector2) {
    if (vector1.size() != vector2.size())
      throw std::runtime_error(
          "The size of the vector must "
          "be the same!");

    const std::size_t size = vector1.size();

    std::vector<std::pair<gene_t, std::size_t>> pairs_v1;
    std::vector<std::pair<gene_t, std::size_t>> pairs_v2;

    pairs_v1.reserve(size);
    std::size_t rank = 0;
    for (const auto& v : vector1) pairs_v1.emplace_back(v, ++rank);

    pairs_v2.reserve(size);
    rank = 0;
    for (const auto& v : vector2) pairs_v2.emplace_back(v, ++rank);

    std::sort(begin(pairs_v1), end(pairs_v1));
    std::sort(begin(pairs_v2), end(pairs_v2));

    unsigned disagreements = 0;
    for (std::size_t i = 0; i < size - 1; ++i) {
      for (std::size_t j = i + 1; j < size; ++j) {
        if ((pairs_v1[i].second < pairs_v1[j].second
             && pairs_v2[i].second > pairs_v2[j].second)
            || (pairs_v1[i].second > pairs_v1[j].second
                && pairs_v2[i].second < pairs_v2[j].second))
          ++disagreements;
      }
    }

    return double(disagreements);
  }

  /**
   * \brief Returns true if the changing of `key1` by `key2` affects
   *        the solution.
   * \param key1 the first key
   * \param key2 the second key
   */
  virtual bool affectSolution(const gene_t key1, const gene_t key2) {
    return fabs(key1 - key2) > 1e-6;
  }

  /**
   * \brief Returns true if the changing of the blocks of keys `v1` by the
   *        blocks of keys `v2` affects the solution.
   *
   * \param v1_begin begin of the first blocks of keys
   * \param v2_begin begin of the first blocks of keys
   * \param block_size number of keys to be considered.
   *
   * \todo (ceandrade): implement this properly.
   */
  virtual bool affectSolution(std::vector<gene_t>::const_iterator v1_begin,
                              std::vector<gene_t>::const_iterator v2_begin,
                              const std::size_t block_size) {
    return block_size == 1 ? affectSolution(*v1_begin, *v2_begin) : true;
  }
};

template <class T>
inline bool close_enough(T a, T b) {
  return std::abs(a - b) < 1e-6;
}

inline box::Chromosome<box::Gene> castChromosome(Chromosome& chr) {
  return box::Chromosome<box::Gene>(chr.data(), (unsigned)chr.size(), 0);
}

void permutatioBasedPathRelink(
    BrkgaConfiguration& config,
    Chromosome& chr1,
    Chromosome& chr2,
    // std::shared_ptr<DistanceFunctionBase> /*non-used*/,
    std::pair<fitness_t, Chromosome>& best_found,
    // unsigned /*non-used block_size*/,
    long max_time,
    double percentage) {
  const unsigned PATH_SIZE = unsigned(percentage * config.chromosomeLength());

  std::set<unsigned> remaining_indices;
  for (unsigned i = 0; i < chr1.size(); ++i) remaining_indices.insert(i);

  struct DecodeStruct {
  public:
    Chromosome chr;
    fitness_t fitness;
    unsigned key_index;
    unsigned pos1;
    unsigned pos2;
    DecodeStruct()
        : chr(), fitness(FITNESS_T_MAX), key_index(0), pos1(0), pos2(0) {}
  };

  // Allocate memory for the candidates.
  std::vector<DecodeStruct> candidates_left(chr1.size());
  std::vector<DecodeStruct> candidates_right(chr1.size());

  for (auto& cand : candidates_left) cand.chr.resize(chr1.size());

  for (auto& cand : candidates_right) cand.chr.resize(chr1.size());

  Chromosome* base = &chr1;
  Chromosome* guide = &chr2;
  std::vector<DecodeStruct>* candidates_base = &candidates_left;
  std::vector<DecodeStruct>* candidates_guide = &candidates_right;

  std::vector<unsigned> chr1_indices(chr1.size());
  std::vector<unsigned> chr2_indices(chr1.size());
  std::vector<unsigned>* base_indices = &chr1_indices;
  std::vector<unsigned>* guide_indices = &chr2_indices;

  // Create and order the indices.
  std::vector<std::pair<double, unsigned>> sorted(chr1.size());

  for (unsigned j = 0; j < 2; ++j) {
    for (unsigned i = 0; i < base->size(); ++i)
      sorted[i] = std::pair<double, unsigned>((*base)[i], i);

    std::sort(begin(sorted), end(sorted));
    for (unsigned i = 0; i < base->size(); ++i)
      (*base_indices)[i] = sorted[i].second;

    swap(base, guide);
    swap(base_indices, guide_indices);
  }

  base = &chr1;
  guide = &chr2;
  base_indices = &chr1_indices;
  guide_indices = &chr2_indices;

#ifdef _OPENMP
#pragma omp parallel for num_threads(config.ompThreads())
#endif
  for (unsigned i = 0; i < candidates_left.size(); ++i) {
    std::copy(begin(*base), end(*base), begin(candidates_left[i].chr));
  }

#ifdef _OPENMP
#pragma omp parallel for num_threads(config.ompThreads())
#endif
  for (unsigned i = 0; i < candidates_right.size(); ++i) {
    std::copy(begin(*guide), end(*guide), begin(candidates_right[i].chr));
  }

  const bool sense = MINIMIZE;

  unsigned iterations = 0;
  while (!remaining_indices.empty()) {
    unsigned position_in_base;
    unsigned position_in_guide;

    auto it_idx = remaining_indices.begin();
    for (unsigned i = 0; i < remaining_indices.size(); ++i) {
      position_in_base = (*base_indices)[*it_idx];
      position_in_guide = (*guide_indices)[*it_idx];

      if (position_in_base == position_in_guide) {
        it_idx = remaining_indices.erase(it_idx);
        --i;
        continue;
      }

      (*candidates_base)[i].key_index = *it_idx;
      (*candidates_base)[i].pos1 = position_in_base;
      (*candidates_base)[i].pos2 = position_in_guide;

      if (sense)
        (*candidates_base)[i].fitness = FITNESS_T_MIN;
      else
        (*candidates_base)[i].fitness = FITNESS_T_MAX;
      ++it_idx;
    }

    if (remaining_indices.size() == 0) break;

    // Decode the candidates.
    volatile bool times_up = false;
#ifdef _OPENMP
#pragma omp parallel for num_threads(config.ompThreads()) shared(times_up) \
    schedule(static, 1)
#endif
    for (unsigned i = 0; i < remaining_indices.size(); ++i) {
      if (times_up) continue;

      std::swap((*candidates_base)[i].chr[(*candidates_base)[i].pos1],
                (*candidates_base)[i].chr[(*candidates_base)[i].pos2]);

      (*candidates_base)[i].fitness =
          config.decoder()->decode(castChromosome((*candidates_base)[i].chr));

      std::swap((*candidates_base)[i].chr[(*candidates_base)[i].pos1],
                (*candidates_base)[i].chr[(*candidates_base)[i].pos2]);

      const auto elapsed_seconds =
          std::chrono::duration_cast<std::chrono::seconds>(
              std::chrono::system_clock::now() - pr_start_time)
              .count();
      if (elapsed_seconds > max_time) times_up = true;
    }

    // Locate the best candidate
    unsigned best_key_index = 0;
    unsigned best_index = -1u;

    fitness_t best_value;
    if (sense)
      best_value = FITNESS_T_MIN;
    else
      best_value = FITNESS_T_MAX;

    for (unsigned i = 0; i < remaining_indices.size(); ++i) {
      if ((best_value < (*candidates_base)[i].fitness && sense)
          || (best_value > (*candidates_base)[i].fitness && !sense)) {
        best_index = i;
        best_key_index = (*candidates_base)[i].key_index;
        best_value = (*candidates_base)[i].fitness;
      }
    }
    assert(best_index != -1u);

    position_in_base = (*base_indices)[best_key_index];
    position_in_guide = (*guide_indices)[best_key_index];

    // Commit the best exchange in all candidates.
    // The last will not be used.
    for (unsigned i = 0; i < remaining_indices.size() - 1; ++i) {
      std::swap((*candidates_base)[i].chr[position_in_base],
                (*candidates_base)[i].chr[position_in_guide]);
    }

    std::swap((*base_indices)[position_in_base],
              (*base_indices)[position_in_guide]);

    // Hold, if it is the best found until now
    if ((sense && best_found.first < best_value)
        || (!sense && best_found.first > best_value)) {
      const auto& best_chr = (*candidates_base)[best_index].chr;
      best_found.first = best_value;
      copy(begin(best_chr), end(best_chr), begin(best_found.second));
    }

    std::swap(base_indices, guide_indices);
    std::swap(candidates_base, candidates_guide);
    remaining_indices.erase(best_key_index);

    // Is time to stop?
    const auto elapsed_seconds =
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now() - pr_start_time)
            .count();

    if ((elapsed_seconds > max_time) || (iterations++ > PATH_SIZE)) break;
  }
}

void pathRelink(BrkgaConfiguration& config,
                // PathRelinking::Type pr_type,  => permutation
                // PathRelinking::Selection pr_selection,  => best
                // std::shared_ptr<DistanceFunctionBase> dist,  => kendall
                float* population,
                unsigned* order,
                float* fitness,
                unsigned number_pairs,
                double minimum_distance,
                // unsigned block_size,
                long max_time,
                double percentage) {
  if (percentage < 1e-6 || percentage > 1.0) {
    std::stringstream ss;
    ss << __PRETTY_FUNCTION__ << ", line " << __LINE__ << ": "
       << "Percentage/size of path relinking invalid, current: " << percentage;
    throw std::range_error(ss.str());
  }

  if (max_time <= 0) max_time = std::numeric_limits<long>::max();

  std::unique_ptr<KendallTauDistance> dist(new KendallTauDistance);
  Chromosome initial_solution(config.chromosomeLength());
  Chromosome guiding_solution(config.chromosomeLength());

  const auto getChromosome = [&](unsigned p, unsigned k) {
    return population + p * config.populationSize() * config.chromosomeLength()
           + order[p * config.populationSize() + k] * config.chromosomeLength();
  };
  const auto getFitness = [&](unsigned p, unsigned k) {
    return fitness[p * config.populationSize() + k];
  };

  // Perform path relinking between elite chromosomes from different
  // populations. This is done in a circular fashion.
  bool path_relinking_possible = false;
  std::deque<std::pair<unsigned, unsigned>> index_pairs;

  // Keep track of the time.
  pr_start_time = std::chrono::system_clock::now();

  for (unsigned pop_count = 0; pop_count < config.numberOfPopulations();
       ++pop_count) {
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                               std::chrono::system_clock::now() - pr_start_time)
                               .count();
    if (elapsed_seconds > max_time) break;

    unsigned pop_base = pop_count;
    unsigned pop_guide = pop_count + 1;
    bool found_pair = false;

    // If we have just one population, we take the both solution from it.
    if (config.numberOfPopulations() == 1) {
      pop_base = pop_guide = 0;
      pop_count = config.numberOfPopulations();
    }
    // If we have two populations, perform just one path relinking.
    else if (config.numberOfPopulations() == 2) {
      pop_count = config.numberOfPopulations();
    }

    // Do the circular thing.
    if (pop_guide == config.numberOfPopulations()) pop_guide = 0;

    index_pairs.clear();
    for (unsigned i = 0; i < config.numberOfElites(); ++i)
      for (unsigned j = 0; j < config.numberOfElites(); ++j)
        index_pairs.emplace_back(std::make_pair(i, j));

    unsigned tested_pairs_count = 0;
    if (number_pairs == 0) number_pairs = (unsigned)index_pairs.size();

    while (!index_pairs.empty() && tested_pairs_count < number_pairs
           && elapsed_seconds < max_time) {
      const auto index = 0;
      // const auto index = (pr_selection ==
      // PathRelinking::Selection::BESTSOLUTION
      //                         ? 0
      //                         : randInt(index_pairs.size() - 1, rng));

      const auto pos1 = index_pairs[index].first;
      const auto pos2 = index_pairs[index].second;

      const auto* ptr1 = getChromosome(pop_base, pos1);
      const std::vector<float> chr1(ptr1, ptr1 + config.chromosomeLength());
      // const std::vector<float> chr1 =
      // current[pop_base]->population[current[pop_base]->fitness[pos1].second];

      const auto* ptr2 = getChromosome(pop_guide, pos2);
      const std::vector<float> chr2(ptr2, ptr2 + config.chromosomeLength());
      // const std::vector<float> chr2 =
      // current[pop_guide]->population[current[pop_base]->fitness[pos2].second];
      // TODO shouldn't it be pop_guide?        ^^^^^^^^

      if (dist->distance(chr1, chr2) >= minimum_distance - 1e-6) {
        copy(begin(chr1), end(chr1), begin(initial_solution));
        copy(begin(chr2), end(chr2), begin(guiding_solution));
        found_pair = true;
        break;
      }

      index_pairs.erase(begin(index_pairs) + index);
      ++tested_pairs_count;
      elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now() - pr_start_time)
                            .count();
    }

    path_relinking_possible |= found_pair;

    // The elite sets are too homogeneous, we cannot do
    // a good path relinking. Let's try other populations.
    if (!found_pair) continue;

    // Create a empty solution.
    std::pair<fitness_t, Chromosome> best_found;
    best_found.second.resize(config.chromosomeLength(), 0.0);

    const bool sense = MINIMIZE;
    if (sense)
      best_found.first = FITNESS_T_MIN;
    else
      best_found.first = FITNESS_T_MAX;

    const auto fence = best_found.first;

    // Perform the path relinking.
    // if (pr_type == PathRelinking::Type::DIRECT) {
    //   directPathRelink(initial_solution, guiding_solution, dist, best_found,
    //                    block_size, max_time, percentage);
    // } else {
    permutatioBasedPathRelink(config, initial_solution, guiding_solution,
                              best_found, max_time, percentage);
    // }

    // **NOTE:** is fitness_t contains float types, so the comparison
    // `best_found.first == fence` may be unfase. Therefore, we use
    // helper functions that define the correct behavior at compilation
    // time.
    if (close_enough(best_found.first, fence)) continue;

    // Re-decode and apply local search if the config.decoder() are able to do
    // it.
    best_found.first =
        config.decoder()->decode(castChromosome(best_found.second));

    // Now, check if the best solution found is really good.
    // If it is the best, overwrite the worse solution in the population.
    bool include_in_population =
        (sense && best_found.first > getFitness(pop_base, 0))
        || (!sense && best_found.first < getFitness(pop_base, 0));

    // If not the best, but is better than the worst elite member, check
    // if the distance between this solution and all elite members
    // is at least minimum_distance.
    if (!include_in_population
        && ((sense
             && best_found.first
                    > getFitness(pop_base, config.numberOfElites() - 1))
            || (!sense
                && best_found.first
                       < getFitness(pop_base, config.numberOfElites() - 1)))) {
      include_in_population = true;
      for (unsigned i = 0; i < config.numberOfElites(); ++i) {
        const auto* ptr = getChromosome(pop_base, i);
        std::vector<float> chr(ptr, ptr + config.chromosomeLength());
        if (dist->distance(best_found.second, chr) < minimum_distance - 1e-6) {
          include_in_population = false;
          break;
        }
      }
    }

    if (include_in_population) {
      box::logger::debug("including chromosome with fitness", best_found.first);
      std::copy(begin(best_found.second), end(best_found.second),
                getChromosome(pop_base, config.populationSize() - 1));
    }
  }
}

namespace box {
void Brkga::runPathRelink(const std::vector<PathRelinkPair>&) {
  logger::debug("Running Path Relink from C. E. Andrade");
  const auto previous = getBestFitness();

  const auto totalChromosomes =
      config.numberOfPopulations() * config.populationSize();
  const auto geneCount = totalChromosomes * config.chromosomeLength();

  gpu::copy2h(nullptr, population.data(), dPopulation.get(), geneCount);

  std::vector<uint> fitnessIdx(totalChromosomes);
  gpu::copy2h(nullptr, fitnessIdx.data(), dFitnessIdx.get(), fitnessIdx.size());

  fitness.resize(totalChromosomes);
  gpu::copy2h(nullptr, fitness.data(), dFitness.get(), fitness.size());

  const auto minDistancePercentage = 0.09;
  const auto n = config.chromosomeLength();
  const auto minDistance =
      (double)((long)n * (n - 1) / 2) * minDistancePercentage;
  const auto maxTime = 24;
  const auto maxIterationsPct = 1.0;
  gpu::sync();
  ::pathRelink(config, population.data(), fitnessIdx.data(), fitness.data(),
               /* number pairs: */ 0, minDistance, maxTime, maxIterationsPct);

  gpu::copy2d(nullptr, dPopulation.get(), population.data(), geneCount);
  gpu::sync();
  updateFitness();

  const auto current = getBestFitness();
  if (current < previous) {
    logger::debug("Improved", previous, "to", current);
  } else {
    logger::debug("No improvement:", current);
  }
}
}  // namespace box
