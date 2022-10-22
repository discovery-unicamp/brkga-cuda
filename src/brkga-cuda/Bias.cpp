#include "Bias.hpp"

#include "except/InvalidArgument.hpp"

namespace box {
Bias biasFromString(const std::string& bias) {
  if (bias == "CONSTANT") return CONSTANT;
  if (bias == "LINEAR") return LINEAR;
  if (bias == "QUADRATIC") return QUADRATIC;
  if (bias == "CUBIC") return CUBIC;
  if (bias == "EXPONENTIAL") return EXPONENTIAL;
  if (bias == "LOGARITHM") return LOGARITHM;
  throw InvalidArgument(format("Unknown bias:", bias), BOX_FUNCTION);
}
}  // namespace box
