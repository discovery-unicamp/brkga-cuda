#include "Comparator.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

namespace box {
bool ComparatorBase::operator()(const Chromosome<float>& lhs,
                                const Chromosome<float>& rhs) const {
  unsigned diff = 0;
  const auto minDiff = (unsigned)(similarity * (float)chromosomeLength);
  for (unsigned i = 0; i < chromosomeLength; ++i) {
    if (!isEqual(lhs[i], rhs[i])) {
      ++diff;
      if (diff > minDiff) return false;
    }
  }
  return true;
}

bool InversionsComparator::operator()(const Chromosome<float>& lhs0,
                                      const Chromosome<float>& rhs0) const {
  const auto n = chromosomeLength;

  auto sorted = [n](const Chromosome<float>& chromosome) {
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

  // Calculate the number of inversions using BIT (Fenwick Tree).
  const auto minDiff =
      (unsigned long)((float)((unsigned long)n * (n - 1) / 2) * similarity);
  unsigned long diff = 0;
  std::vector<unsigned> bit(n + 1, 0);
  for (unsigned i = n - 1; i != -1u; --i) {
    for (unsigned k = a[i] + 1; k; k -= k & -k) diff += bit[k];
    if (diff >= minDiff) return false;

    // Will not overflow if n < ~4e7 since there are at most O(n lg n) updates
    for (unsigned k = a[i] + 1; k <= n; k += k & -k) ++bit[k];
  }

  return true;
}
}  // namespace box
