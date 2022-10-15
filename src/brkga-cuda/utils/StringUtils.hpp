#ifndef STRING_UTILS_HPP
#define STRING_UTILS_HPP

#include <chrono>
#include <ctime>
#include <iomanip>
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
inline std::string nowIso() {
  auto now = std::chrono::system_clock::now();
  auto seconds = std::chrono::system_clock::to_time_t(now);
  auto nowSeconds = std::chrono::system_clock::from_time_t(seconds);
  auto microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(now - nowSeconds)
          .count();

  std::stringstream ss;
  ss << std::put_time(gmtime(&seconds), "%FT%T") << '.' << std::fixed
     << std::setw(6) << std::setfill('0') << microseconds;
  return ss.str();
}
}  // namespace box

#endif  // STRING_UTILS_HPP
