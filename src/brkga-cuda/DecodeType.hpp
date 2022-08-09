#ifndef BRKGA_CUDA_DECODE_TYPE_HPP
#define BRKGA_CUDA_DECODE_TYPE_HPP

#include <stdexcept>
#include <string>

namespace box {
class DecodeType {
public:
  [[nodiscard]] static DecodeType fromString(const std::string& str);

  DecodeType() : _cpu(true), _chromosome(true), _all(false) {}

  DecodeType(bool onCpu, bool chromosome, bool allAtOnce)
      : _cpu(onCpu), _chromosome(chromosome), _all(allAtOnce) {}

  [[nodiscard]] inline bool onCpu() const { return _cpu; }

  [[nodiscard]] inline bool chromosome() const { return _chromosome; }

  [[nodiscard]] inline bool allAtOnce() const { return _all; }

  [[nodiscard]] std::string str() const;

private:
  bool _cpu;
  bool _chromosome;
  bool _all;
};
}  // namespace box

#endif  // BRKGA_CUDA_DECODE_TYPE_HPP
