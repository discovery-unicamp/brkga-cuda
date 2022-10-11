#ifndef BRKGA_CUDA_LOGGER_HPP
#define BRKGA_CUDA_LOGGER_HPP

#ifndef LOG_LEVEL
#define LOG_LEVEL box::logger::_LogType::WARNING
#endif  // LOG_LEVEL

#include "utils/StringUtils.hpp"

#include <iostream>
#include <string>

namespace box {
namespace logger {
static std::ostream* logStream = &std::clog;

enum _LogType { NONE = 0, ERROR, WARNING, INFO, DEBUG };

static const char* RESET = "\033[0m";
static const char* LINE_ABOVE = "\033[F";
static const char* BLACK = "\033[0;30m";
static const char* CYAN = "\033[0;36m";
static const char* RED = "\033[0;31m";
static const char* YELLOW = "\033[0;33m";

template <class... T>
inline void _log(std::ostream& out,
                 const char* config,
                 const char* type,
                 const T&... x) {
  out << config << format(type, x...) << RESET << std::endl;
}

template <class... T>
inline void pbar(double completed,
                 unsigned length,
                 bool begin,
                 const T&... args) {
  if (completed < 0)
    completed = 0.0;
  else if (completed > 1)
    completed = 1.0;

  char str[10];
  snprintf(str, sizeof(str), "[%6.1lf%%]", 100 * completed);

  const auto filled = (unsigned)(completed * length + 1e-6);
  std::string progress(filled, '=');
  if (filled < length) {
    progress += ">";
    progress += std::string(length - filled - 1, ' ');
  }
  progress = "[" + progress + "]";

  auto config = (begin ? std::string() : std::string(LINE_ABOVE)) + CYAN;
  _log(*logStream, config.data(), str, progress, args...);
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
