#ifndef BRKGA_CUDA_LOGGER_HPP
#define BRKGA_CUDA_LOGGER_HPP

#ifndef LOG_LEVEL
#define LOG_LEVEL box::logger::_LogType::WARNING
#endif  // LOG_LEVEL

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace box {
template <class T>
std::string str(const std::vector<T>& v, const std::string& sep = ", ") {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(6);
  bool flag = false;
  ss << '[';
  for (auto& x : v) {
    (flag ? ss << sep : ss) << x;
    flag = true;
  }
  ss << ']';
  return ss.str();
}

namespace logger {
static std::ostream* stream = &std::clog;

enum _LogType { NONE = 0, ERROR, WARNING, INFO, DEBUG };

static const char* RESET = "\033[0m";
static const char* RED = "\033[31m";
static const char* GREEN = "\033[32m";
static const char* YELLOW = "\033[33m";
static const char* BLUE = "\033[34m";

inline void _log_impl(std::ostream&) {}

template <class T, class... U>
inline void _log_impl(std::ostream& out, const T& x, const U&... y) {
  out << ' ' << x;
  _log_impl(out, y...);
}

template <class... T>
inline void _log(std::ostream& out,
                 const char* color,
                 const char* type,
                 const T&... x) {
  out << color << type;
  _log_impl(out, x...);
  out << RESET << std::endl;  // Use std::endl to avoid missing any log.
}

template <class... T>
inline void error(const T&... args) {
  if (LOG_LEVEL >= ERROR) _log(*stream, RED, "[  ERROR]", args...);
}

template <class... T>
inline void warning(const T&... args) {
  if (LOG_LEVEL >= WARNING) _log(*stream, YELLOW, "[WARNING]", args...);
}

template <class... T>
inline void info(const T&... args) {
  if (LOG_LEVEL >= INFO) _log(*stream, GREEN, "[   INFO]", args...);
}

template <class... T>
inline void debug(const T&... args) {
  if (LOG_LEVEL >= DEBUG) _log(*stream, BLUE, "[  DEBUG]", args...);
}
}  // namespace logger
}  // namespace box

#endif  // BRKGA_CUDA_LOGGER_HPP
