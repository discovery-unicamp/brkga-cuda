#ifndef BRKGACUDA_BRKGA_HPP
#define BRKGACUDA_BRKGA_HPP

#include "BrkgaConfiguration.hpp"
#include "Chromosome.hpp"
#include "CudaUtils.hpp"

#include <curand.h>  // TODO check if this header is required here

#include <utility>
#include <vector>

namespace box {
class DecodeType;
class Decoder;

class Brkga {
public:
  /**
   * Construct a new Brkga object.
   *
   * @param config The configuration to run the algorithm.
   */
  Brkga(const BrkgaConfiguration& config);

  /// Releases memory
  ~Brkga();

  /**
   * Evolve the population to the next generation.
   */
  void evolve();

  /**
   * Copy the elites from/to all populations.
   *
   * This method will simply copy the @p count elites from one population to all
   * the others. It will not copy to the same population, which avoids
   * generating duplicated chromsomes.
   *
   * This operation blocks the CPU until it is finished.
   *
   * @param count The number of elites to copy from each population.
   */
  void exchangeElite(unsigned count);

  /**
   * Run the (implicit) Path Relinking algorithm between pairs of chromosomes.
   *
   * Divides the base and theguide into the blocks [0, @p blockSize),
   * [@p blockSize, 2 * @p blockSize), ..., and replaces the best block on the
   * base with the corresponding block on theguide. The method is
   * repeated until it doesn't improve the solution.
   *
   * @param pairs The chromosomes to use as base (.first) and guide (.second).
   * @param blockSize The number of genes to be changed in a single operation.
   */
  void runPathRelinking(std::vector<std::pair<unsigned, unsigned>> pairs,
                        unsigned blockSize);

  /**
   * Get the fitness of the best chromosome found so far.
   *
   * This operation blocks the CPU until it is finished.
   *
   * @return The fitness of the best chromsome.
   */
  float getBestFitness();

  /**
   * Get the best chromosome.
   *
   * This operation blocks the CPU until it is finished.
   *
   * @return The best chromsome.
   */
  std::vector<float> getBestChromosome();

  /**
   * Get the best chromosome when sorted.
   *
   * This operation blocks the CPU until it is finished.
   *
   * @return The best chromsome when sorted.
   * @throw `std::runtime_error` If the decode type is a non-sorted one.
   */
  std::vector<unsigned> getBestPermutation();

private:
  std::pair<unsigned, unsigned> getBest();

  /// Sync all streams (except the default) with the host
  void syncStreams();

  /**
   * Call the decode method to the population `p`.
   *
   * @param p The index of the population to decode.
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

  /// The main stream to run the operations indenpendently
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
