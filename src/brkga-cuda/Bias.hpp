#ifndef BOX_BIAS_HPP
#define BOX_BIAS_HPP

#include <string>

namespace box {
/// Probability to copy an allele from parent i (0 <= i < #parents)
enum Bias {
  /// bias(i) = 1 / #parents
  CONSTANT,

  /// bias(i) = 1 / (i + 1)
  LINEAR,

  /// bias(i) = 1 / (i + 1)^2
  QUADRATIC,

  /// bias(i) = 1 / (i + 1)^3
  CUBIC,

  /// bias(i) = 1 / exp(i)
  EXPONENTIAL,

  /// bias(i) = 1 / log(i + 2)
  LOGARITHM,
};

/// Convert @p bias to the enum @ref Bias
/// @throws @ref InvalidArgument if @p bias is invalid
Bias biasFromString(const std::string& bias);
}  // namespace box

#endif  // BOX_BIAS_HPP
