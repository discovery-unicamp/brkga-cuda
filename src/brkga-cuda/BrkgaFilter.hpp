#ifndef BOX_BRKGA_FILTER_HPP
#define BOX_BRKGA_FILTER_HPP

#include "Chromosome.hpp"
#include "except/InvalidArgument.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace box {
// FIXME Handle these filter inside the device
/**
 * Filter to determine if two chromosomes are very similar.
 *
 * With this class we aim to be flexible while removing duplicated chromosomes
 * in the population.
 */
// TODO rename to compare (or something like that)
class FilterBase {
public:
  /**
   * Initializes the data used by the filter.
   * \param _chromosomeLength The number of genes in the chromosome.
   * \param _similarityLimit % of equal genes to consider the chromosomes equal.
   */
  inline FilterBase(unsigned _chromosomeLength, float _similarityLimit)
      : chromosomeLength(_chromosomeLength),
        minDiffGenes(
            (unsigned)((float)chromosomeLength * (1.0 - _similarityLimit))) {
    InvalidArgument::range(Arg<float>(_similarityLimit, "Similarity"),
                           Arg<float>(0), Arg<float>(1), 1 /* end closed */,
                           __FUNCTION__);

    // The condition above isn't enough?
    assert(minDiffGenes > 0);
    assert(minDiffGenes < chromosomeLength);
  }

  /// Determines if two chromosomes are equal.
  virtual bool operator()(const Chromosome<float>& lhs,
                          const Chromosome<float>& rhs) const;

  /// Determines if two genes are equal.
  virtual bool isEqual(float lhs, float rhs) const = 0;

protected:
  unsigned chromosomeLength;
  unsigned minDiffGenes;
};

/**
 * Determine if two chromosomes are very similar comparing each gene.
 *
 * The filter assumes that two genes are equal if their absolute difference
 * doesn't exceed the epsilon, i.e.:
 * \code{.cpp}
 * isEqual[i] = abs(lhs[i] - rhs[i]) < eps
 * \endcode
 */
class EpsilonFilter : public FilterBase {
public:
  /**
   * Initializes the data used by the filter.
   * \param _chromosomeLength The number of genes in the chromosome.
   * \param _similarityLimit % of equal genes to consider the chromosomes equal.
   * \param _eps The range to consider two genes equal. Defaults to 1e-7f.
   */
  inline EpsilonFilter(unsigned _chromosomeLength,
                       float _similarityLimit,
                       float _eps = 1e-7f)
      : FilterBase(_chromosomeLength, _similarityLimit), eps(_eps) {
    InvalidArgument::range(Arg<float>(eps, "epsilon"), Arg<float>(0),
                           Arg<float>(1), 0 /* open range */, __FUNCTION__);
  }

  inline bool isEqual(float lhs, float rhs) const {
    return std::abs(lhs - rhs) < this->eps;
  }

private:
  float eps;
};

/**
 * Determine if two chromosomes are very similar using Hamming distance.
 *
 * The filter assumes that two genes are equal if their comparison against the
 * threshold lead to the same value -- similar to the Hamming distance --, i.e.:
 * \code{.cpp}
 * isEqual[i] = (lhs[i] < threshold) == (rhs[i] < threshold)
 * \endcode
 */
class ThresholdFilter : public FilterBase {
public:
  /**
   * Initializes the data used by the filter.
   * \param _chromosomeLength The number of genes in the chromosome.
   * \param _similarityLimit % of equal genes to consider the chromosomes equal.
   * \param _threshold The range to consider two genes equal. Defaults to 1e-7f.
   */
  inline ThresholdFilter(unsigned _chromosomeLength,
                         float _similarityLimit,
                         float _threshold)
      : FilterBase(_chromosomeLength, _similarityLimit), threshold(_threshold) {
    InvalidArgument::range(Arg<float>(threshold, "threshold"), Arg<float>(0),
                           Arg<float>(1), 0 /* open range */, __FUNCTION__);
  }

  inline bool isEqual(float lhs, float rhs) const {
    return (lhs < this->threshold) == (rhs < this->threshold);
  }

private:
  float threshold;
};
}  // namespace box

#endif  // BOX_BRKGA_FILTER_HPP
