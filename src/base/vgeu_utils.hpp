#pragma once

/*

refence: sample util code in vulkan-hpp
https://github.com/KhronosGroup/Vulkan-Hpp


*/

// libs
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

// std
#include <memory>
#include <string>
#include <vector>

namespace vgeu {
// initVulkan - device
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

struct QueueFamilyIndices {
  uint32_t graphics;
  uint32_t compute;
  uint32_t transfer;
};

vk::raii::Instance createInstance(const vk::raii::Context& context,
                                  const std::string& appName,
                                  const std::string& engineName,
                                  uint32_t apiVersion = VK_API_VERSION_1_0);
vk::DebugUtilsMessengerCreateInfoEXT createDebugCreateInfo();
bool checkValidationLayerSupport(const vk::raii::Context& context);
std::vector<const char*> getRequiredExtensions();
vk::raii::DebugUtilsMessengerEXT setupDebugMessenger(
    vk::raii::Instance& instance);

vk::raii::Device createLogicalDevice(
    const vk::raii::PhysicalDevice& physicalDevice,
    const QueueFamilyIndices& queueFamilyIndices,
    const std::vector<const char*>& extensions = {},
    const vk::PhysicalDeviceFeatures* physicalDeviceFeatures = nullptr,
    const void* pNext = nullptr, bool useSwapChain = true,
    vk::QueueFlags requestedQueueTypes = vk::QueueFlagBits::eGraphics |
                                         vk::QueueFlagBits::eCompute);

QueueFamilyIndices findQueueFamilyIndices(
    const std::vector<vk::QueueFamilyProperties>& queueFamilyProperties,
    vk::QueueFlags requestedQueueTypes);

uint32_t getQueueFamilyIndex(
    const std::vector<vk::QueueFamilyProperties>& queueFamilyProperties,
    vk::QueueFlagBits queueFlag);

vk::Format pickDepthFormat(const vk::raii::PhysicalDevice& physicalDevice,
                           bool requiresStencil);

struct SwapChainData {
  SwapChainData(const vk::raii::PhysicalDevice& physicalDevice,
                const vk::raii::Device& device,
                const vk::raii::SurfaceKHR& surface, const vk::Extent2D& extent,
                vk::ImageUsageFlags usage,
                const vk::raii::SwapchainKHR* pOldSwapchain,
                uint32_t graphicsQueueFamilyIndex,
                uint32_t presentQueueFamilyIndex);

  vk::Format colorFormat;
  vk::raii::SwapchainKHR swapChain = nullptr;
  std::vector<vk::Image> images;
  std::vector<vk::raii::ImageView> imageViews;
};

template <class T>
VULKAN_HPP_INLINE constexpr const T& clamp(const T& v, const T& lo,
                                           const T& hi) {
  return v < lo ? lo : hi < v ? hi : v;
}

vk::SurfaceFormatKHR pickSurfaceFormat(
    std::vector<vk::SurfaceFormatKHR> const& formats);
vk::PresentModeKHR pickPresentMode(
    std::vector<vk::PresentModeKHR> const& presentModes);
}  // namespace vgeu