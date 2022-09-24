#ifndef BRKGACUDA_BRKGA_HPP
#define BRKGACUDA_BRKGA_HPP

#include "BrkgaConfiguration.hpp"
#include "BrkgaFilter.hpp"
#include "Chromosome.hpp"
#include "CudaUtils.hpp"
#include "PathRelinkPair.hpp"

#include <curand.h>  // TODO check if this header is required here

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

class Brkga {
public:
  // FIXME remove this method
  void printStatus();

  // FIXME update how we handle the initialPopulation
  // FIXME add option to set device chromosomes
  // TODO how to expose a chromosome to the decoder and another for interaction?
  /// Construct a new Brkga object.
  Brkga(const BrkgaConfiguration& config,
        const std::vector<std::vector<std::vector<float>>>& initialPopulation);

  /// Releases memory.
  ~Brkga();

  /// Evolve the population to the next generation.
  void evolve();

  /// For each pair of elites, remove the worse of them if \p filter is true.
  /// TODO add support to device objects
  void removeSimilarElites(const FilterBase& filter);

  /**
   * Copy the elites from/to all populations.
   *
   * This method will simply copy the \p count elites from one population to all
   * the others. It will not copy to the same population, which avoids
   * generating duplicated chromosomes.
   *
   * This operation blocks the CPU until it is finished.
   *
   * \param count The number of elites to copy from each population.
   */
  void exchangeElite(unsigned count);

  // FIXME this is temporary
  // @{
  /// Run the Path Relink algorithm between pairs of chromosomes.
  template <typename F, typename... Args>
  inline void runPathRelink(unsigned blockSize,
                            const F& selectMethod,
                            const Args&... args) {
    runPathRelink(blockSize,
                  selectMethod(numberOfPopulations, eliteSize, args...));
  }

  void runPathRelink(unsigned blockSize,
                     const std::vector<PathRelinkPair>& pairList);

  std::vector<float> pathRelink(unsigned blockSize,
                                unsigned base,
                                unsigned guide);
  // @} # temporary code

  // TODO how to handle any type provided by the user without templates?
  /// Get the fitness of the best chromosome found so far.
  float getBestFitness();

  /// Get the genes of the best chromosome found so far.
  std::vector<float> getBestChromosome();

  /// Get the permutation of the best chromosome found so far.
  /// \throw `std::runtime_error` If the decode type is a non-sorted one.
  std::vector<unsigned> getBestPermutation();

  std::vector<DecodedChromosome> getPopulation(unsigned p);

private:
  std::pair<unsigned, unsigned> getBest();

  /// Sync all streams (except the default) with the host
  void syncStreams();

  /**
   * Call the decode method to the population `p`.
   *
   * \param p The index of the population to decode.
   */
  void decodePopulation(unsigned p);

  /// Sorts the indices of the chromosomes in case of sorted decode
  void sortChromosomesGenes();

  /**
   * Ensures the fitness is sorted.
   *
   * This operation should be executed after each change to any chromosome.
   */
  void updateFitness();

  template <class T>
  Chromosome<T>* wrapCpu(T* pop, unsigned popId, unsigned n);

  template <class T>
  Chromosome<T>* wrapGpu(T* pop, unsigned popId, unsigned n);

  /// The main stream to run the operations independently
  constexpr static cudaStream_t defaultStream = nullptr;

  Decoder* decoder;  /// The decoder of the problem

  cuda::Matrix<float> dPopulation;  /// All the chromosomes
  std::vector<float> population;  /// All chromosomes, but on CPU
  cuda::Matrix<float> dPopulationTemp;  /// Temp memory for chromosomes
  Chromosome<float>* populationWrapper;  /// Wrapper for the decoder

  cuda::Matrix<float> dFitness;  /// The (sorted) fitness of each chromosome
  std::vector<float> fitness;  /// All fitness, but on CPU
  cuda::Matrix<unsigned> dFitnessIdx;
  cuda::Matrix<unsigned> dPermutations;  /// Indices of the genes when sorted
  std::vector<unsigned> permutations;  /// All permutations, but on CPU

  cuda::Matrix<float> dRandomEliteParent;  /// The elite parent
  cuda::Matrix<float> dRandomParent;  /// The non-elite parent

  unsigned chromosomeSize;  /// The size of each chromosome
  unsigned populationSize;  /// The size of each population
  unsigned eliteSize;  /// The number of elites in the population
  unsigned mutantsSize;  /// The number of mutants in the population
  unsigned numberOfPopulations;  /// The number of populations
  float rhoe;  /// The bias to accept the elite chromosome
  DecodeType decodeType;  /// The decode method
  std::vector<cudaStream_t> streams;  /// The streams to process the populations
  std::vector<curandGenerator_t> generators;  /// Random generators

  unsigned threadsPerBlock;  /// Number of device threads to use
};
}  // namespace box

#endif  // BRKGACUDA_BRKGA_HPP
