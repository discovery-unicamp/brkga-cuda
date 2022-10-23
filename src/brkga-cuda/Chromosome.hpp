#ifndef BOX_BRKGA_CHROMOSOME_HPP
#define BOX_BRKGA_CHROMOSOME_HPP 1

#include <cuda_runtime.h>

#include <cassert>
#include <type_traits>

namespace box {
template <class T>
class Chromosome {
public:
  static_assert(std::is_same<T, float>::value
                    || std::is_same<T, unsigned>::value,
                "Chromosome can only be float or unsigned");

  __host__ __device__ inline Chromosome() : Chromosome(nullptr, 0, 0) {}

  __host__ __device__ inline Chromosome(T* _population,
                                        unsigned _columnCount,
                                        unsigned _chromosomeIndex,
                                        unsigned _guideIndex = (unsigned)-1,
                                        unsigned _guideStart = 0,
                                        unsigned _guideEnd = 0)
      : population(_population),
        columnCount(_columnCount),
        chromosomeIndex(_chromosomeIndex),
        guideIndex(_guideIndex),
        guideStart(_guideStart),
        guideEnd(_guideEnd),
        guidePopulation(nullptr) {
    assert(chromosomeIndex != guideIndex);
    assert(guideEnd < (1u << (8 * sizeof(unsigned) - 1)));

// TODO define GPU specific code to access the transposed matrix
#ifndef __CUDA_ARCH__
    // Add an offset to the first gene of the chromosome
    guidePopulation = _population + guideIndex * columnCount;
    population += chromosomeIndex * columnCount;
#endif

    if (guideStart >= guideEnd) {
      guideStart = guideEnd = 0;
      guidePopulation = nullptr;
    } else {
      guideEnd -= guideStart;
    }
  }

  __host__ __device__ inline T operator[](unsigned i) const {
    // ** gl = guideStart and gr = guideEnd **
    // gl <= i && i < gr == i - gl < gr - gl:
    // i - gl will overflow if i < gl and then i - gl < gr - gl will be false
    // => this only works if gl <= gr < 2^(#bits(typeof(gr)) - 1)
    // => also gr - gl was already performed in the constructor
#ifdef __CUDA_ARCH__
    return *(population
             + (i - guideStart < guideEnd ? guideIndex : chromosomeIndex)
                   * columnCount
             + i);
#else
    return *((i - guideStart < guideEnd ? guidePopulation : population) + i);
#endif
  }

  static void copy(cudaStream_t stream,
                   T* to,
                   const Chromosome<T>* chromosomes,
                   const unsigned n,
                   const unsigned chromosomeLength);

private:
  T* population;
  unsigned columnCount;
  unsigned chromosomeIndex;
  unsigned guideIndex;
  unsigned guideStart;
  unsigned guideEnd;
  T* guidePopulation;  // Used to speedup the population access on the CPU
};

template class Chromosome<float>;
template class Chromosome<unsigned>;
}  // namespace box

#endif  // BOX_BRKGA_CHROMOSOME_HPP
