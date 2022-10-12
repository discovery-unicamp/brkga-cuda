#ifndef STRING_UTILS_HPP
#define STRING_UTILS_HPP

#include <sstream>
#include <string>

namespace box {
/// Separator used in @ref format
struct Separator {
  inline Separator(const char* _separator) : separator(_separator) {}
  const char* separator;
};

/// Returns @p args as a string separated by @p sep
template <typename... T>
inline std::string format(const Separator& sep, const T&... args) {
  std::stringstream ss;
  bool noSep = true;
  (void)std::initializer_list<bool>{
      (ss << (noSep ? "" : sep.separator) << args, noSep = false)...};
  return ss.str();
}

/// Returns @p args as a string separated by spaces
template <typename... T>
inline std::string format(const T&... args) {
  return format(Separator(" "), args...);
}

/// Returns the current time according to ISO 8601 with 6 decimal places
std::string nowIso();
}  // namespace box

#endif  // STRING_UTILS_HPP
