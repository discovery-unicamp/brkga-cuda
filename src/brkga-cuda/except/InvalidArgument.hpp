#ifndef BOX_EXCEPT_INVALIDARGUMENT_HPP
#define BOX_EXCEPT_INVALIDARGUMENT_HPP

#include "../utils/StringUtils.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace box {
template <class T>
struct Arg {
  inline Arg(const T& _value, const std::string& _name = "")
      : value(_value), name(_name) {}

  inline std::string str() const {
    if (name.empty()) return format(value);
    return format(Separator(""), name, " (", value, ")");
  }

  const T& value;
  const std::string& name;
};

class InvalidArgument : public std::runtime_error {
public:
  typedef std::runtime_error Super;

  template <class T>
  static inline void null(const Arg<T*>& arg, const std::string& func) {
    if (arg.value == nullptr)
      throw InvalidArgument(format(arg.name, "is null"), func);
  }

  template <class T>
  static inline void min(const Arg<T>& arg,
                         const Arg<T>& min,
                         const std::string& func) {
    if (arg.value < min.value)
      throw InvalidArgument(format(arg.str(), "is less than", min.str()), func);
  }

  template <class T>
  static inline void max(const Arg<T>& arg,
                         const Arg<T>& max,
                         const std::string& func) {
    if (arg.value > max.value)
      throw InvalidArgument(format(arg.str(), "is greater than", max.str()),
                            func);
  }

  template <class T>
  static inline void range(const Arg<T>& arg,
                           const Arg<T>& min,
                           const Arg<T>& max,
                           unsigned type,
                           const std::string& func) {
    const bool startClosed = type & 2;
    const bool endClosed = type & 1;
    if ((startClosed ? arg.value < min.value : arg.value <= min.value)
        || (endClosed ? arg.value > max.value : arg.value >= max.value)) {
      throw InvalidArgument(
          format(arg.str(), "is out of range",
                 format(Separator(""), (startClosed ? '[' : '('), min.str(),
                        ", ", max.str(), (endClosed ? ']' : ')'))),
          func);
    }
  }

  inline InvalidArgument(const std::string& msg, const std::string& func)
      : Super(format(Separator(""), msg, " [function ", func, "]")) {}
};
}  // namespace box

#endif  // BOX_EXCEPT_INVALIDARGUMENT_HPP
