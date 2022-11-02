#include "Comparator.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

namespace box {
bool ComparatorBase::operator()(const Chromosome<Gene>& lhs,
                                const Chromosome<Gene>& rhs) const {
  unsigned equal = 0;
  const auto minNumberToConsiderEqual =
      (unsigned)ceil(similarity * (float)chromosomeLength);
  for (unsigned i = 0; i < chromosomeLength; ++i) {
    if (isEqual(lhs[i], rhs[i])) {
      ++equal;
      if (equal >= minNumberToConsiderEqual) return true;
    }
  }
  return false;
}

bool KendallTauComparator::operator()(const Chromosome<Gene>& lhs0,
                                      const Chromosome<Gene>& rhs0) const {
  const auto n = chromosomeLength;

  auto sorted = [n](const Chromosome<Gene>& chromosome) {
    std::vector<unsigned> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::sort(permutation.begin(), permutation.end(),
              [&chromosome](unsigned i, unsigned j) {
                return chromosome[i] < chromosome[j];
              });
    return permutation;
  };
  const auto lhs = sorted(lhs0);
  const auto rhs = sorted(rhs0);

  // Set rhs to the sequence listed in lhs.
  std::vector<unsigned> a(n);
  for (unsigned i = 0; i < n; ++i) a[lhs[i]] = rhs[i];

  /*
   * h = number of inversions calculated using BIT (Fenwick Tree)
   * k = n * (n - 1) / 2 - h
   * tau = (2 * k - n * (n - 1) / 2) / (n * (n - 1) / 2)
   *     = 2 * k / (n * (n - 1) / 2) - 1  ==> tau \in [-1, 1]
   *     = k / (n * (n - 1) / 2)  ==> tau \in [0, 1]
   *
   *                   tau >= similarity
   * k / (n * (n - 1) / 2) >= similarity
   *                     k >= ceil(similarity * (n * (n - 1) / 2))
   */

  const auto maxValue = (unsigned long)n * (n - 1) / 2;
  const auto minNumberToConsiderEqual =
      (unsigned long)ceil((double)maxValue * similarity);

  unsigned long h = 0;
  std::vector<unsigned> bit(n + 1, 0);
  for (unsigned i = n - 1; i != -1u; --i) {
    for (unsigned j = a[i] + 1; j; j -= j & -j) h += bit[j];
    const auto k = n * (n - 1) / 2 - h;  // k is decreasing
    if (k < minNumberToConsiderEqual) return false;

    // Will not overflow if n < ~4e7 since there are at most O(n lg n) updates
    for (unsigned j = a[i] + 1; j <= n; j += j & -j) ++bit[j];
  }

  return true;
}
}  // namespace box
