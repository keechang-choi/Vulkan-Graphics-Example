/*

flag template
Reference: Vulkan-Hpp enums

for the enum class, use template specialization
to set isBitmask of FlagTraits to be true

*/

#pragma once

// std
#include <type_traits>

// NOTE: namespace matters?
namespace vgeu {

template <typename FlagBitsType>
struct FlagTraits {
  static constexpr bool isBitmask = false;
};

template <typename BitType>
class Flags {
 public:
  using MaskType = typename std::underlying_type<BitType>::type;

  constexpr Flags() noexcept : mask_(0) {}
  constexpr Flags(BitType bit) noexcept : mask_(static_cast<MaskType>(bit)) {}
  constexpr Flags(Flags<BitType> const& rhs) noexcept = default;
  constexpr explicit Flags(MaskType flags) noexcept : mask_(flags) {}
  constexpr bool operator==(Flags<BitType> const& rhs) const noexcept {
    return mask_ == rhs.mask_;
  }
  constexpr bool operator!=(Flags<BitType> const& rhs) const noexcept {
    return mask_ != rhs.mask_;
  }
  constexpr Flags<BitType> operator&(Flags<BitType> const& rhs) const noexcept {
    return Flags<BitType>(mask_ & rhs.mask_);
  }
  constexpr Flags<BitType> operator|(Flags<BitType> const& rhs) const noexcept {
    return Flags<BitType>(mask_ | rhs.mask_);
  }
  constexpr Flags<BitType> operator^(Flags<BitType> const& rhs) const noexcept {
    return Flags<BitType>(mask_ ^ rhs.mask_);
  }
  explicit constexpr operator bool() const noexcept { return !!mask_; }
  explicit constexpr operator MaskType() const noexcept { return mask_; }

 private:
  MaskType mask_;
};

// bitwise operators on BitType
template <
    typename BitType,
    typename std::enable_if<FlagTraits<BitType>::isBitmask, bool>::type = true>
inline constexpr Flags<BitType> operator&(BitType lhs, BitType rhs) noexcept {
  return Flags<BitType>(lhs) & rhs;
}

template <
    typename BitType,
    typename std::enable_if<FlagTraits<BitType>::isBitmask, bool>::type = true>
inline constexpr Flags<BitType> operator|(BitType lhs, BitType rhs) noexcept {
  return Flags<BitType>(lhs) | rhs;
}

template <
    typename BitType,
    typename std::enable_if<FlagTraits<BitType>::isBitmask, bool>::type = true>
inline constexpr Flags<BitType> operator^(BitType lhs, BitType rhs) noexcept {
  return Flags<BitType>(lhs) ^ rhs;
}

}  // namespace vgeu