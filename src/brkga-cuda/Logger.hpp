#ifndef BRKGA_CUDA_LOGGER_HPP
#define BRKGA_CUDA_LOGGER_HPP

#ifndef LOG_LEVEL
#define LOG_LEVEL box::logger::_LogType::WARNING
#endif  // LOG_LEVEL

#include <iostream>
#include <utility>
#include <vector>

namespace box {
namespace logger {
static std::ostream* logStream = &std::clog;

enum _LogType { NONE = 0, ERROR, WARNING, INFO, DEBUG };

static const char* RESET = "\033[0m";
static const char* BLACK = "\033[0;30m";
static const char* CYAN = "\033[0;36m";
static const char* RED = "\033[0;31m";
static const char* YELLOW = "\033[0;33m";

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
  if (LOG_LEVEL >= ERROR) _log(*logStream, RED, "[  ERROR]", args...);
}

template <class... T>
inline void warning(const T&... args) {
  if (LOG_LEVEL >= WARNING) _log(*logStream, YELLOW, "[WARNING]", args...);
}

template <class... T>
inline void info(const T&... args) {
  if (LOG_LEVEL >= INFO) _log(*logStream, CYAN, "[   INFO]", args...);
}

template <class... T>
inline void debug(const T&... args) {
  if (LOG_LEVEL >= DEBUG) _log(*logStream, BLACK, "[  DEBUG]", args...);
}
}  // namespace logger
}  // namespace box

#endif  // BRKGA_CUDA_LOGGER_HPP
