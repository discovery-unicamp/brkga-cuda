#ifndef BRKGA_CUDA_CUDAERROR_CUH
#define BRKGA_CUDA_CUDAERROR_CUH

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace box {
class CudaError : public std::runtime_error {
public:
  static inline void _check(const cudaError_t status,
                            const char* file,
                            const int line,
                            const char* func) {
    if (status != cudaSuccess) throw CudaError(status, file, line, func);
  }

private:
  CudaError(const cudaError_t status,
            const char* file,
            const int line,
            const char* func)
      : std::runtime_error(std::string(file) + ":" + std::to_string(line)
                           + ": On " + func + ": "
                           + cudaGetErrorString(status)) {}
};
}  // namespace box

#define CUDA_CHECK(cmd) \
  box::CudaError::_check((cmd), __FILE__, __LINE__, __PRETTY_FUNCTION__)
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaPeekAtLastError())

#endif  // BRKGA_CUDA_CUDAERROR_CUH
