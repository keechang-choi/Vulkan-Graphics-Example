#include "vge_base.hpp"

// libs
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>
namespace vge {
VgeBase::VgeBase() { vk::raii::Context context; }
VgeBase::~VgeBase() {}
}  // namespace vge