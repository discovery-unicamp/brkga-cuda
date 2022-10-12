#ifndef STRING_UTILS_HPP
#define STRING_UTILS_HPP

#include <sstream>
#include <string>

namespace box {
struct Separator {
  inline Separator(const char* _separator) : separator(_separator) {}
  const char* separator;
};

template <typename... T>
inline std::string format(const Separator& sep, const T&... args) {
  std::stringstream ss;
  bool noSep = true;
  (void)std::initializer_list<bool>{
      (ss << (noSep ? "" : sep.separator) << args, noSep = false)...};
  return ss.str();
}

template <typename... T>
inline std::string format(const T&... args) {
  return format(Separator(" "), args...);
}
}  // namespace box

#endif  // STRING_UTILS_HPP
