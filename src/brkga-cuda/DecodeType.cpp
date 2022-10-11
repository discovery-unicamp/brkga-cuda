#include "DecodeType.hpp"

#include "Logger.hpp"
#include "except/InvalidArgument.hpp"
#include "utils/StringUtils.hpp"

#include <cctype>
#include <string>

namespace box {
bool contains(const std::string& str, const std::string& pattern) {
  return str.find(pattern) != std::string::npos;
}

DecodeType DecodeType::fromString(const std::string& str) {
  logger::debug("Parsing decoder:", str);

  bool cpu = contains(str, "cpu");
  bool chromosome = !contains(str, "permutation");
  bool allAtOnce = contains(str, "all");
  auto dt = DecodeType(cpu, chromosome, allAtOnce);
  if (dt.str() != str)
    throw InvalidArgument(format(Separator(""), "Invalid decoder: ", str,
                                 "; did you mean ", dt.str(), "?"),
                          __FUNCTION__);

  return dt;
}

std::string DecodeType::str() const {
  std::string str = _all ? "all-" : "";
  str += _cpu ? "cpu" : "gpu";
  str += _chromosome ? "" : "-permutation";
  return str;
}
}  // namespace box
