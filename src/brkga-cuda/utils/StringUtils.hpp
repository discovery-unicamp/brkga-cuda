#ifndef STRING_UTILS_HPP
#define STRING_UTILS_HPP

#include <sstream>
#include <string>

namespace box {
template <typename... T>
std::string format(const T&... args) {
  std::stringstream ss;
  (void)std::initializer_list<int>{(ss << args, 0)...};
  return ss.str();
}

template <typename... T>
std::string formats(const std::string& separator, const T&... args) {
  std::stringstream ss;
  bool noSep = true;
  (void)std::initializer_list<bool>{
      (ss << (noSep ? "" : separator) << args, noSep = false)...};
  return ss.str();
}
}  // namespace box

#endif  // STRING_UTILS_HPP
