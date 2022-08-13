#ifndef BOX_BRKGA_CHROMOSOME_HPP
#define BOX_BRKGA_CHROMOSOME_HPP 1

#include <cuda_runtime.h>

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
                                        unsigned _chromosomeIndex)
      : population(_population),
        columnCount(_columnCount),
        chromosomeIndex(_chromosomeIndex) {
// TODO define GPU specific code to access the transposed matrix
#ifndef CUDA_ARCH
    // Add an offset to the first gene of the chromosome
    population += chromosomeIndex * columnCount;
#endif
  }

  __host__ __device__ inline T operator[](unsigned gene) const {
#ifdef CUDA_ARCH
    return population[chromosomeIndex * columnCount + gene];
#else
    return population[gene];
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
};

template class Chromosome<float>;
template class Chromosome<unsigned>;
}  // namespace box

#endif  // BOX_BRKGA_CHROMOSOME_HPP
