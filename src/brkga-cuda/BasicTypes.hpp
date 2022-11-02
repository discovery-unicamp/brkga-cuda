#ifndef BOX_BASICTYPES_HPP
#define BOX_BASICTYPES_HPP

#include <cstdint>

#ifdef Byte
#warning Macro `Byte` is being undefined
#undef Byte
#endif  // Byte

#ifdef UInt
#warning Macro `UInt` is being undefined
#undef UInt
#endif  // UInt

#ifdef ULong
#warning Macro `ULong` is being undefined
#undef ULong
#endif  // ULong

#ifdef Float
#warning Macro `Float` is being undefined
#undef Float
#endif  // Float

namespace box {
typedef std::int8_t Byte;
typedef std::uint32_t UInt;
typedef std::uint64_t ULong;
typedef float Float;

typedef Float Gene;
typedef UInt GeneIndex;
typedef Float Fitness;  // FIXME how to handle user type without template?
}  // namespace box

#endif  // BOX_BASICTYPES_HPP
