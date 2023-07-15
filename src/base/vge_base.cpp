#include "vge_base.hpp"

#include "vgeu_utils.hpp"

// libs
// #include <Vulkan-Hpp/vulkan/vulkan.hpp>
// #include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

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

  queue = vk::raii::Queue(device, queueFamilyIndices.graphics, 0);
  vk::CommandPoolCreateInfo cmdPoolCI(
      vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      queueFamilyIndices.graphics);
  commandPool = vk::raii::CommandPool(device, cmdPoolCI);
  depthFormat = vgeu::pickDepthFormat(physicalDevice, requiresStencil);

  semaphores.presentComplete =
      vk::raii::Semaphore(device, vk::SemaphoreCreateInfo());
  semaphores.renderComplete =
      vk::raii::Semaphore(device, vk::SemaphoreCreateInfo());

  // surface
  VkSurfaceKHR surface_;
  vgeuWindow->createWindowSurface(static_cast<VkInstance>(*instance),
                                  &surface_);
  surface = vk::raii::SurfaceKHR(instance, surface_);

  return true;
}

void VgeBase::getEnabledExtensions(){};
void VgeBase::prepare() {
  // NOTE: first graphicsQueue supports present?
  swapChainData = std::make_unique<vgeu::SwapChainData>(
      physicalDevice, device, surface, vgeuWindow->getExtent(),
      vk::ImageUsageFlagBits::eColorAttachment |
          vk::ImageUsageFlagBits::eTransferSrc,
      nullptr, queueFamilyIndices.graphics, queueFamilyIndices.graphics);
}

}  // namespace vge