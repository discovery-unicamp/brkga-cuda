#include "DecodeType.hpp"

#include "Logger.hpp"

#include <cctype>
#include <string>

bool contains(const std::string& str, const std::string& pattern) {
  return str.find(pattern) != std::string::npos;
}

box::DecodeType box::DecodeType::fromString(const std::string& str) {
  box::logger::debug("Parsing decoder:", str);

  bool cpu = contains(str, "cpu");
  bool chromosome = !contains(str, "permutation");
  bool allAtOnce = contains(str, "all");
  auto dt = DecodeType(cpu, chromosome, allAtOnce);
  if (dt.str() != str)
    throw std::runtime_error("Invalid decoder: " + str + "; did you mean "
                             + dt.str() + "?");

  return dt;
}

std::string box::DecodeType::str() const {
  std::string str = _all ? "all-" : "";
  str += _cpu ? "cpu" : "gpu";
  str += _chromosome ? "" : "-permutation";
  return str;
}
