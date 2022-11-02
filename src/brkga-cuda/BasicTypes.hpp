#ifndef BOX_BASICTYPES_HPP
#define BOX_BASICTYPES_HPP

#include <cstdint>

#ifdef byte
#warning Macro `byte` is being undefined
#undef byte
#endif  // byte

#ifdef uint
#warning Macro `uint` is being undefined
#undef uint
#endif  // uint

#ifdef ulong
#warning Macro `ulong` is being undefined
#undef ulong
#endif  // ulong

#ifdef float_t
#warning Macro `float_t` is being undefined
#undef float_t
#endif  // float_t

namespace box {
typedef std::int8_t byte;
typedef std::uint32_t uint;
typedef std::uint64_t ulong;
typedef float float_t;

typedef float_t Gene;
typedef ulong GeneIndex;
typedef float_t Fitness;  // FIXME how to handle user type without template?
}  // namespace box

#endif  // BOX_BASICTYPES_HPP
