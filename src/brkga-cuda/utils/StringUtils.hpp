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
}  // namespace box

#endif  // STRING_UTILS_HPP
