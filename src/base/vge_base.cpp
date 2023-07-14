#include "vge_base.hpp"

#include "vgeu_utils.hpp"

// libs
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

// std
#include <iostream>
#include <memory>
namespace vge {
VgeBase::VgeBase() { std::cout << "Created: Vulkan Example Base" << std::endl; }
VgeBase::~VgeBase() {}

bool VgeBase::initVulkan() {
  // NOTE: shoud be created before instance for getting required extensions;
  vgeuWindow = std::make_unique<vgeu::VgeuWindow>(WIDTH, HEIGHT, title);
  context = std::make_unique<vk::raii::Context>();
  instance = vgeu::createInstance(*context, title, title);
  if (vgeu::enableValidationLayers) {
    debugUtilsMessenger = vgeu::setupDebugMessenger(instance);
  }
  // select gpu
  physicalDevice = vk::raii::PhysicalDevices(instance).front();
  getEnabledExtensions();
  std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
      physicalDevice.getQueueFamilyProperties();

  queueFamilyIndices = vgeu::findQueueFamilyIndices(
      queueFamilyProperties,
      vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute);

  device = vgeu::createLogicalDevice(
      physicalDevice, queueFamilyIndices, enabledDeviceExtensions,
      &enabledFeatures, deviceCreatepNextChain, true,
      vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute);
  return true;
}

void VgeBase::getEnabledExtensions(){};
void VgeBase::prepare() {
  VkSurfaceKHR surface_;

  vgeuWindow->createWindowSurface(static_cast<VkInstance>(*instance),
                                  &surface_);
  surface = vk::raii::SurfaceKHR(instance, surface_);
}

}  // namespace vge