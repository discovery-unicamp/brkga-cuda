#ifndef BOX_EXCEPT_INVALIDARGUMENT_HPP
#define BOX_EXCEPT_INVALIDARGUMENT_HPP

#include "../utils/StringUtils.hpp"

#include <stdexcept>
#include <string>

namespace box {
class InvalidArgument : public std::runtime_error {
public:
  typedef std::runtime_error Super;

  template <class T>
  static void null(const std::string& name,
                   const T* arg,
                   const std::string& func) {
    if (arg == nullptr) throw InvalidArgument(format(name, " is null"), func);
  }

  template <class T>
  static void min(const std::string& name,
                  const T& arg,
                  const T& minValue,
                  const std::string& func) {
    if (arg < minValue)
      throw InvalidArgument(format(name, " is less than ", minValue, ": ", arg),
                            func);
  }

  template <class T>
  static void range(const std::string& name,
                    const T& arg,
                    const T& begin,
                    const T& end,
                    unsigned type,
                    const std::string& func) {
    const bool startClosed = type & 2;
    const bool endClosed = type & 1;
    if ((startClosed ? arg < begin : arg <= begin)
        || (endClosed ? arg > end : arg >= end)) {
      throw InvalidArgument(
          format(name, " is out of range ", (startClosed ? "[" : "("), begin,
                 " ", end, (endClosed ? "]" : ")"), ": ", arg),
          func);
    }
  }

  InvalidArgument(const std::string& msg, const std::string& func)
      : Super(format(msg, " (on ", func, ")")) {}
};
}  // namespace box

#endif  // BOX_EXCEPT_INVALIDARGUMENT_HPP
