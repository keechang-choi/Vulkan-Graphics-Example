#pragma once

// libs
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

namespace vge {
class VgeBase {};
}  // namespace vge

#define VULKAN_EXAMPLE_BASE()
int main(int argc, char** argv) {
  vk::raii::Context context;

  return 0;
}