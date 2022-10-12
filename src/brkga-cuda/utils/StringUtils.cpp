#include "StringUtils.hpp"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>

namespace box {
std::string nowIso() {
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
