#ifndef BOX_BRKGA_FILTER_HPP
#define BOX_BRKGA_FILTER_HPP

#include "Chromosome.hpp"
#include "except/InvalidArgument.hpp"

#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace box {
// TODO Handle these filters inside the device
// TODO How to hide the inclusion of params like chromosomeLength?
/// Filter to determine if two chromosomes are very similar.
class ComparatorBase {
public:
  inline ComparatorBase(unsigned _chromosomeLength, unsigned _minDiffGenes)
      : chromosomeLength(_chromosomeLength), minDiffGenes(_minDiffGenes) {
    InvalidArgument::range(Arg<unsigned>(minDiffGenes, "minimum difference"),
                           Arg<unsigned>(0), Arg<unsigned>(chromosomeLength),
                           1 /* end closed */, BOX_FUNCTION);
  }

  inline ComparatorBase(unsigned _chromosomeLength, float _similarityLimit)
      : ComparatorBase(
          _chromosomeLength,
          (unsigned)((float)_chromosomeLength * (1 - _similarityLimit))) {}

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
class EpsilonComparator : public ComparatorBase {
public:
  inline EpsilonComparator(unsigned _chromosomeLength,
                           unsigned _minDiffGenes,
                           float _eps = 1e-7f)
      : ComparatorBase(_chromosomeLength, _minDiffGenes), eps(_eps) {
    InvalidArgument::range(Arg<float>(eps, "epsilon"), Arg<float>(0),
                           Arg<float>(1), 0 /* open range */, BOX_FUNCTION);
  }

  inline EpsilonComparator(unsigned _chromosomeLength,
                           float _similarityLimit,
                           float _eps = 1e-7f)
      : ComparatorBase(_chromosomeLength, _similarityLimit), eps(_eps) {
    InvalidArgument::range(Arg<float>(eps, "epsilon"), Arg<float>(0),
                           Arg<float>(1), 0 /* open range */, BOX_FUNCTION);
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
class ThresholdComparator : public ComparatorBase {
public:
  inline ThresholdComparator(unsigned _chromosomeLength,
                             unsigned _minDiffGenes,
                             float _threshold)
      : ComparatorBase(_chromosomeLength, _minDiffGenes),
        threshold(_threshold) {
    InvalidArgument::range(Arg<float>(threshold, "threshold"), Arg<float>(0),
                           Arg<float>(1), 0 /* open range */, BOX_FUNCTION);
  }

  inline ThresholdComparator(unsigned _chromosomeLength,
                             float _similarityLimit,
                             float _threshold)
      : ComparatorBase(_chromosomeLength, _similarityLimit),
        threshold(_threshold) {
    InvalidArgument::range(Arg<float>(threshold, "threshold"), Arg<float>(0),
                           Arg<float>(1), 0 /* open range */, BOX_FUNCTION);
  }

  inline bool isEqual(float lhs, float rhs) const {
    return (lhs < this->threshold) == (rhs < this->threshold);
  }

private:
  float threshold;
};
}  // namespace box

#endif  // BOX_BRKGA_FILTER_HPP
