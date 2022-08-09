#ifndef BRKGACUDA_EXCEPT_NOTIMPLEMENTED_HPP
#define BRKGACUDA_EXCEPT_NOTIMPLEMENTED_HPP

#include <stdexcept>
#include <string>

namespace box {
class NotImplemented : public std::logic_error {
public:
  NotImplemented(const std::string& func)
      : std::logic_error("Function `" + func + "` wasn't implemented") {}
};
}  // namespace box

#endif  // BRKGACUDA_EXCEPT_NOTIMPLEMENTED_HPP
