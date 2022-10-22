#include "Comparator.hpp"

namespace box {
bool ComparatorBase::operator()(const Chromosome<float>& lhs,
                            const Chromosome<float>& rhs) const {
  unsigned diff = 0;
  for (unsigned i = 0; i < this->chromosomeLength; ++i) {
    if (!this->isEqual(lhs[i], rhs[i])) {
      diff += 1;
      if (diff > this->minDiffGenes) return false;
    }
  }
  return true;
}
}  // namespace box