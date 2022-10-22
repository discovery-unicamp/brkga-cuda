#ifndef BOX_BIAS_HPP
#define BOX_BIAS_HPP

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
}  // namespace box

#endif  // BOX_BIAS_HPP
