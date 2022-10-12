#ifndef BOX_EXCEPT_CUDAERROR_HPP
#define BOX_EXCEPT_CUDAERROR_HPP

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

#endif  // BOX_EXCEPT_CUDAERROR_HPP
