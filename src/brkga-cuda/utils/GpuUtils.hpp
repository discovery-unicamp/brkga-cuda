#ifndef BOX_UTILS_GPUUTILS_HPP
#define BOX_UTILS_GPUUTILS_HPP

#include "../BasicTypes.hpp"
#include "../Logger.hpp"
#include "../except/CudaError.hpp"
#include "../except/InvalidArgument.hpp"

#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <cctype>
#include <map>
#include <stdexcept>

#define CUDA_CHECK(cmd) \
  box::CudaError::_check((cmd), __FILE__, __LINE__, __PRETTY_FUNCTION__)
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaPeekAtLastError())

namespace box {
/// C++ wrapper for operations in the device.
namespace gpu {
/// Synchronize the host with the main stream in the device.
inline void sync() {
  logger::debug("Sync with the main stream");
  CUDA_CHECK(cudaDeviceSynchronize());
}

/// Synchronize the host with the specified stream.
inline void sync(cudaStream_t stream) {
  logger::debug("Sync with stream", (void*)stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

/// Set the maximum memory allocated on the heap to @p maxBytes bytes.
inline void setMaxHeapSize(std::size_t maxBytes) {
  logger::debug("Increase heap limit to", maxBytes, "bytes");
  CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, maxBytes));
}

/**
 * Allocates contiguous memory on the device.
 *
 * This works like the `new[]` operator.
 *
 * @tparam T The memory type.
 * @param n The number of elements to allocate.
 * @return The allocated memory.
 */
template <class T>
inline T* alloc(cudaStream_t stream, std::size_t n) {
  logger::debug("Allocating", n, "elements of", sizeof(T), "bytes");
  T* ptr = nullptr;
  CUDA_CHECK(cudaMallocAsync(&ptr, n * sizeof(T), stream));
  return ptr;
}

/**
 * Releases memory from the device.
 *
 * This works like the `delete[]` operator.
 *
 * @param ptr The pointer to the memory to free.
 */
template <class T>
inline void free(cudaStream_t stream, T* ptr) {
  logger::debug("Free", (void*)ptr);
  CUDA_CHECK(cudaFreeAsync(ptr, stream));
}

/// Creates a new stream.
inline cudaStream_t allocStream() {
  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  return stream;
}

/// Releases an allocated stream.
inline void free(cudaStream_t stream) {
  logger::debug("Free stream", (void*)stream);
  CUDA_CHECK(cudaStreamDestroy(stream));
}

/// Creates a new random number generator.
inline curandGenerator_t allocRandomGenerator(
    ulong seed,
    curandRngType_t type = CURAND_RNG_PSEUDO_DEFAULT) {
  curandGenerator_t generator = nullptr;
  curandCreateGenerator(&generator, type);
  curandSetPseudoRandomGeneratorSeed(generator, seed);
  CUDA_CHECK_LAST();
  return generator;
}

/// Releases an allocated random generator.
inline void free(curandGenerator_t generator) {
  logger::debug("Free generator", (void*)generator);
  curandDestroyGenerator(generator);
  CUDA_CHECK_LAST();
}

/**
 * Copy data in a contiguous memory in the device to another.
 * @tparam T The memory type.
 * @param stream The stream to run.
 * @param dest The destination memory.
 * @param src The source memory.
 * @param n The number of elements to copy.
 */
template <class T>
inline void copy(cudaStream_t stream, T* dest, const T* src, std::size_t n) {
  logger::debug("Copy", n, "elements from", (void*)src, "(device) to",
                (void*)dest, "(device) on stream", (void*)stream);
  CUDA_CHECK(cudaMemcpyAsync(dest, src, n * sizeof(T), cudaMemcpyDeviceToDevice,
                             stream));
}

/**
 * Copy data in a contiguous memory in the host to another in the device.
 * @tparam T The memory type.
 * @param stream The stream to run.
 * @param dest The destination memory in the device.
 * @param src The source memory in the host.
 * @param n The number of elements to copy.
 */
template <class T>
inline void copy2d(cudaStream_t stream, T* dest, const T* src, std::size_t n) {
  logger::debug("Copy", n, "elements of", sizeof(T), "bytes from", (void*)src,
                "(host) to", (void*)dest, "(device) on stream", (void*)stream);
  CUDA_CHECK(cudaMemcpyAsync(dest, src, n * sizeof(T), cudaMemcpyHostToDevice,
                             stream));
}

/**
 * Copy data in a contiguous memory in the device to another in the host.
 * @tparam T The memory type.
 * @param stream The stream to run.
 * @param dest The destination memory in the host.
 * @param src The source memory in the device.
 * @param n The number of elements to copy.
 */
template <class T>
inline void copy2h(cudaStream_t stream, T* dest, const T* src, std::size_t n) {
  logger::debug("Copy", n, "elements from", (void*)src, "(device) to",
                (void*)dest, "(host) on stream", (void*)stream);
  CUDA_CHECK(cudaMemcpyAsync(dest, src, n * sizeof(T), cudaMemcpyDeviceToHost,
                             stream));
}

/**
 * Returns the number of blocks required to process `n` items.
 * @param n The number of items to process.
 * @param threads The desired number of threads.
 * @return The minimum number of blocks `k` s.t. `k * threads >= n`.
 */
[[nodiscard]] inline constexpr unsigned blocks(uint n, uint threads) {
  return (unsigned)((n + threads - 1) / threads);
}

/**
 * Sets the sequence `0, 1, ..., n-1` to an array.
 * @param stream The stream to process.
 * @param arr The array to store the sequence
 * @param n The size of the array.
 */
void iota(cudaStream_t stream, uint* arr, uint n);

/// @brief Builds the sequence `0, 1, ..., k-1, 0, 1, ...`.
/// @param stream The stream to run on.
/// @param arr The output array were the values will be stored.
/// @param n The length of the array.
/// @param k The period of the sequence.
void iotaMod(cudaStream_t stream, uint* arr, uint n, uint k);

/// @brief Builds the sequence `0, 1, ..., k-1, 0, 1, ...`.
/// @param stream The stream to run on.
/// @param arr The output array were the values will be stored.
/// @param n The length of the array.
/// @param k The period of the sequence.
void iotaMod(cudaStream_t stream, ulong* arr, uint n, uint k);

/// @brief Generate random values in range (0, 1].
/// @param stream The stream to run on.
/// @param generator The generator to use.
/// @param arr The output array were the values will be stored.
/// @param n The length of the array.
inline void random(cudaStream_t stream,
                   curandGenerator_t generator,
                   float* arr,
                   std::size_t n) {
  curandSetStream(generator, stream);
  curandGenerateUniform(generator, arr, n);
  CUDA_CHECK_LAST();
}

/// @brief Generate random values in range (0, 1].
/// @param stream The stream to run on.
/// @param generator The generator to use.
/// @param arr The output array were the values will be stored.
/// @param n The length of the array.
inline void random(cudaStream_t stream,
                   curandGenerator_t generator,
                   double* arr,
                   std::size_t n) {
  curandSetStream(generator, stream);
  curandGenerateUniformDouble(generator, arr, n);
  CUDA_CHECK_LAST();
}

namespace _detail {
/// @brief Allocator that doesn't free the memory between allocations.
/// Inspired by: https://stackoverflow.com/a/48671517/10111328.
class CachedAllocator {
public:
  typedef char byte;
  typedef byte value_type;

  CachedAllocator() = default;
  CachedAllocator(const CachedAllocator&) = delete;
  CachedAllocator(CachedAllocator&&) = delete;
  CachedAllocator& operator=(const CachedAllocator&) = delete;
  CachedAllocator& operator=(CachedAllocator&&) = delete;

  ~CachedAllocator() {
    try {
      free();
    } catch (CudaError& e) {
      // "driver shutting down" error
    }
  }

  /// Allocates or reuse a memory of at least @p nbytes bytes.
  byte* allocate(std::size_t nbytes);

  /// Saves @p ptr to `freeMem`.
  void deallocate(byte* ptr, std::size_t);

  /// Releases memory in `freeMem`.
  void free();

private:
  std::multimap<std::size_t, byte*> freeMem;  /// Allocated but unused memory
  std::map<byte*, std::size_t> allocMem;  /// Allocated and used memory
};

extern CachedAllocator cachedAllocator;  // FIXME will this lead to seg-fault?
}  // namespace _detail

/**
 * Sorts the array of keys and values based on the keys.
 * @tparam Key The key type.
 * @tparam Value The value type.
 * @param stream The stream to process.
 * @param keys The keys used to compare (and also sorted).
 * @param values The values to sort.
 * @param n The length of the arrays.
 */
template <class Key, class Value>
inline void sortByKey(cudaStream_t stream,
                      Key* keys,
                      Value* values,
                      std::size_t n) {
  thrust::device_ptr<Key> keysPtr(keys);
  thrust::device_ptr<Value> valuesPtr(values);
  thrust::sort_by_key(thrust::cuda::par(_detail::cachedAllocator).on(stream),
                      keysPtr, keysPtr + n, valuesPtr);
  CUDA_CHECK_LAST();
}

/**
 * Sorts the segments `[0, step)`, `[step, 2 * step)`, and so on.
 *
 * Both the keys and the values are sorted on the process.
 *
 * This method blocks the host until the kernel finishes.
 *
 * @param stream The stream to process.
 * @param dKeys The (mutable) key to use on comparator.
 * @param dValues The values to sort.
 * @param size The size of the arrays.
 * @param step The size of the segments to sort.
 * @throw std::invalid_argument if @p size is not a multiple of @p step.
 * @throw std::invalid_argument if @p size doesn't fit 31 bit integer.
 */
void segSort(cudaStream_t stream,
             Gene* dKeys,
             GeneIndex* dValues,
             std::size_t size,
             std::size_t step);

class Timer {
public:
  Timer();
  ~Timer();

  void reset();
  float milliseconds();
  float seconds();

private:
  cudaEvent_t start;
  cudaEvent_t stop;
};

template <class T>
class Matrix {
public:
  inline Matrix(std::size_t _nrows, std::size_t _ncols)
      : nrows(_nrows),
        ncols(_ncols),
        matrix(alloc<T>(nullptr, nrows * ncols)) {}

  Matrix(const Matrix&) = delete;

  inline Matrix(Matrix&& that)
      : nrows(that.nrows), ncols(that.ncols), matrix(that.matrix) {
    that.matrix = nullptr;
  }

  ~Matrix() { free(nullptr, matrix); }

  inline T* get() { return matrix; }

  inline T* row(std::size_t k) {
    InvalidArgument::max(Arg<std::size_t>(k, "row"),
                         Arg<std::size_t>(nrows - 1, "last row"), __FUNCTION__);
    return matrix + k * ncols;
  }

  inline void swap(Matrix& that) {
    std::swap(nrows, that.nrows);
    std::swap(ncols, that.ncols);
    std::swap(matrix, that.matrix);
  }

private:
  std::size_t nrows;
  std::size_t ncols;
  T* matrix;
};
}  // namespace gpu
}  // namespace box

namespace std {
template <class T>
inline void swap(box::gpu::Matrix<T>& lhs, box::gpu::Matrix<T>& rhs) {
  lhs.swap(rhs);
}
}  // namespace std

#endif  // BOX_UTILS_GPUUTILS_HPP
