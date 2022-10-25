#ifndef BOX_COMPARATOR_HPP
#define BOX_COMPARATOR_HPP

#include "Chromosome.hpp"
#include "except/InvalidArgument.hpp"
#include "except/NotImplemented.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace box {
// TODO Handle these comparators inside the device
// TODO How to hide the inclusion of params like chromosomeLength?
/// Check if two chromosomes are very similar.
class ComparatorBase {
public:
  inline ComparatorBase(unsigned _chromosomeLength, unsigned _minEqualGenes)
      : ComparatorBase(_chromosomeLength,
                       (float)_minEqualGenes / (float)_chromosomeLength) {}

  inline ComparatorBase(unsigned _chromosomeLength, float _similarity)
      : chromosomeLength(_chromosomeLength), similarity(_similarity) {
    InvalidArgument::range(Arg<float>(similarity, "similarity"), Arg<float>(0),
                           Arg<float>(1), 1 /* end closed */, BOX_FUNCTION);
  }

  /// Check if two chromosomes are very similar.
  virtual bool operator()(const Chromosome<float>& lhs,
                          const Chromosome<float>& rhs) const;

  /// Check if two genes are equal.
  virtual bool isEqual(float lhs, float rhs) const = 0;

protected:
  unsigned chromosomeLength;
  float similarity;  /// The minimum % to consider two chromosomes equal
};

/**
 * Check for similarity if two genes have absolute value less than an epsilon.
 *
 * The comparator assumes that two genes are equal if their absolute difference
 * doesn't exceed the epsilon, i.e.:
 * \code{.cpp}
 * isEqual[i] = abs(lhs[i] - rhs[i]) < eps
 * \endcode
 */
class EpsilonComparator : public ComparatorBase {
public:
  inline EpsilonComparator(unsigned _chromosomeLength,
                           unsigned _minEqualGenes,
                           float _eps = 1e-7f)
      : ComparatorBase(_chromosomeLength, _minEqualGenes), eps(_eps) {
    InvalidArgument::range(Arg<float>(eps, "epsilon"), Arg<float>(0),
                           Arg<float>(1), 0 /* open range */, BOX_FUNCTION);
  }

  inline EpsilonComparator(unsigned _chromosomeLength,
                           float _similarity,
                           float _eps = 1e-7f)
      : ComparatorBase(_chromosomeLength, _similarity), eps(_eps) {
    InvalidArgument::range(Arg<float>(eps, "epsilon"), Arg<float>(0),
                           Arg<float>(1), 0 /* open range */, BOX_FUNCTION);
  }

  inline bool isEqual(float lhs, float rhs) const override {
    return std::abs(lhs - rhs) < this->eps;
  }

private:
  float eps;
};

/**
 * Check for similarity when both genes compare equal with a threshold.
 *
 * The comparator assumes that two genes are equal if their comparison against
 * the threshold have the same result, i.e.:
 * \code{.cpp}
 * isEqual[i] = (lhs[i] < threshold) == (rhs[i] < threshold)
 * \endcode
 */
class ThresholdComparator : public ComparatorBase {
public:
  inline ThresholdComparator(unsigned _chromosomeLength,
                             unsigned _minEqualGenes,
                             float _threshold)
      : ComparatorBase(_chromosomeLength, _minEqualGenes),
        threshold(_threshold) {
    InvalidArgument::range(Arg<float>(threshold, "threshold"), Arg<float>(0),
                           Arg<float>(1), 0 /* open range */, BOX_FUNCTION);
  }

  inline ThresholdComparator(unsigned _chromosomeLength,
                             float _similarity,
                             float _threshold)
      : ComparatorBase(_chromosomeLength, _similarity), threshold(_threshold) {
    InvalidArgument::range(Arg<float>(threshold, "threshold"), Arg<float>(0),
                           Arg<float>(1), 0 /* open range */, BOX_FUNCTION);
  }

  inline bool isEqual(float lhs, float rhs) const override {
    return (lhs < this->threshold) == (rhs < this->threshold);
  }

private:
  float threshold;
};

/**
 * Check for similarity using the Kendall's tau distance.
 *
 * This comparator first sort the chromosomes before performing the evaluation.
 * The comparator counts the number of inversions of the genes and uses that to
 * compare against the given similarity. Two genes u and v has an inversion if u
 * is to left of v in lhs and u is to the right of v in rhs. There will be at
 * most n * (n - 1) / 2 inversions, where n = |chromosome|.
 * Therefore, the chromosomes are considered very similar if:
 *   #inversions / (n * (n - 1) / 2) >= similarity
 */
class KendallTauComparator : public ComparatorBase {
public:
  inline KendallTauComparator(unsigned _chromosomeLength, float _similarity)
      : ComparatorBase(_chromosomeLength, _similarity) {}

  bool operator()(const Chromosome<float>& lhs,
                  const Chromosome<float>& rhs) const override;

  inline bool isEqual(float, float) const override {
    throw NotImplemented(format(__PRETTY_FUNCTION__,
                                "doesn't work with the inversions comparator"));
  }
};
}  // namespace box

#endif  // BOX_COMPARATOR_HPP
