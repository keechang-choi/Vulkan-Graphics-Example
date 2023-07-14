#include "vge_base.hpp"

#include "vgeu_utils.hpp"

// libs
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

// std
#include <iostream>
namespace vge {
VgeBase::VgeBase() { std::cout << "Created: Vulkan Example Base" << std::endl; }
VgeBase::~VgeBase() {}

bool VgeBase::initVulkan() {
  vk::raii::Context context;
  vk::raii::Instance instance = vgeu::createInstance(context, title, title);
  return true;
}
}  // namespace vge