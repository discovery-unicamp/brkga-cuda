#ifndef BRKGACUDA_BRKGA_HPP
#define BRKGACUDA_BRKGA_HPP

#include "BrkgaConfiguration.hpp"
#include "Chromosome.hpp"
#include "Comparator.hpp"
#include "PathRelinkPair.hpp"
#include "utils/GpuUtils.hpp"

#include <curand.h>
#include <curand_kernel.h>

#include <utility>
#include <vector>

// FIXME create a clear declaration of this structure
struct DecodedChromosome {
  float fitness;
  std::vector<float> genes;  // FIXME create a typedef for the gene type
};

namespace box {
class DecodeType;
class Decoder;

// FIXME use this typedef
// typedef float Fitness;
// typedef float Gene;

/// Implements the BRKGA algorithm for GPUs
class Brkga {
public:
  // FIXME remove this method
  void printStatus();

  // FIXME update how we handle the initialPopulation
  // FIXME add option to set device chromosomes
  // TODO how to expose a chromosome to the decoder and another for interaction?
  /// Construct a new Brkga object.
  Brkga(const BrkgaConfiguration& config,
        const std::vector<std::vector<std::vector<float>>>& initialPopulation =
            {});

  /// Releases memory.
  ~Brkga();

  /// Evolve the population to the next generation.
  void evolve();

  // TODO add support to device objects
  /// For each pair of elites, remove the worse of them if @p similar is true.
  void removeSimilarElites(const ComparatorBase& similar);

  /// Copy the elites from/to all populations, except from/to itself.
  void exchangeElites();

  /// Use @p selectMethod to define the pairs to run the Path Relink algorithm.
  template <typename Method, typename... Args>
  inline void runPathRelink(const Method& selectMethod, const Args&... args) {
    logger::debug("Using a method to select pairs for path relink");
    const auto n = config.numberOfPopulations() * config.populationSize()
                   * config.chromosomeLength();
    population.resize(n);
    gpu::copy2h(nullptr, population.data(), dPopulation.get(), n);
    runPathRelink(selectMethod(config, population.data(), args...));
  }

  /// Run the Path Relink algorithm between pairs of chromosomes.
  /// @warning This method doesn't use the permutation decoder. You should
  ///   implement the chromosome decoder otherwise an exception will be thrown.
  void runPathRelink(const std::vector<PathRelinkPair>& pairList);

  // TODO how to handle any type provided by the user without templates?
  /// Get the fitness of the best chromosome found so far.
  float getBestFitness();

  /// Get the genes of the best chromosome found so far.
  std::vector<float> getBestChromosome();

  /// Get the permutation of the best chromosome found so far.
  /// @throw `std::runtime_error` If the decode type is a non-sorted one.
  std::vector<unsigned> getBestPermutation();

  std::vector<DecodedChromosome> getPopulation(unsigned p);

  BrkgaConfiguration config;  /// The parameters of the algorithm

private:
  std::pair<unsigned, unsigned> getBest();

  /// Sync all streams (except the default) with the host
  void syncStreams();

  /// Call the decode method to the population @p p
  void decodePopulation(unsigned p);

  /// Sorts the indices of the chromosomes in case of sorted decode
  void sortChromosomesGenes();

  /**
   * Ensures the fitness is sorted.
   *
   * This operation should be executed after updating the chromosomes.
   */
  void updateFitness();

  std::vector<float> pathRelink(unsigned base, unsigned guide);

  template <class T>
  Chromosome<T>* wrapCpu(T* pop, unsigned popId, unsigned n);

  template <class T>
  Chromosome<T>* wrapGpu(T* pop, unsigned popId, unsigned n);

  gpu::Matrix<float> dPopulation;  /// All the chromosomes
  std::vector<float> population;  /// All chromosomes, but on CPU
  gpu::Matrix<float> dPopulationTemp;  /// Temp memory for chromosomes
  Chromosome<float>* populationWrapper;  /// Wrapper for the decoder

  gpu::Matrix<float> dFitness;  /// The (sorted) fitness of each chromosome
  std::vector<float> fitness;  /// All fitness, but on CPU
  gpu::Matrix<unsigned> dFitnessIdx;
  gpu::Matrix<unsigned> dPermutations;  /// Indices of the genes when sorted
  std::vector<unsigned> permutations;  /// All permutations, but on CPU

  gpu::Matrix<unsigned> dParent;  /// The parent in the crossover
  gpu::Matrix<curandState_t> dRandomStates;  /// RNG to select parents

  std::vector<cudaStream_t> streams;  /// The streams to process the populations
  std::vector<curandGenerator_t> generators;  /// Random generators
};
}  // namespace box

#endif  // BRKGACUDA_BRKGA_HPP
