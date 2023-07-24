#pragma once

/*

refence: sample util code in vulkan-hpp
https://github.com/KhronosGroup/Vulkan-Hpp

*/

// libs
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

// std
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vgeu {
// initVulkan - device

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

struct QueueFamilyIndices {
  uint32_t graphics;
  uint32_t compute;
  uint32_t transfer;
};

vk::raii::Instance createInstance(const vk::raii::Context& context,
                                  const std::string& appName,
                                  const std::string& engineName,
                                  bool enableValidationLayers,
                                  uint32_t apiVersion = VK_API_VERSION_1_0);
vk::DebugUtilsMessengerCreateInfoEXT createDebugCreateInfo();
bool checkValidationLayerSupport(const vk::raii::Context& context);
std::vector<const char*> getRequiredExtensions(bool enableValidationLayers);
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
  SwapChainData(std::nullptr_t) {}

  vk::Extent2D swapChainExtent;
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

struct ImageData {
  ImageData(vk::raii::PhysicalDevice const& physicalDevice,
            vk::raii::Device const& device, vk::Format format_,
            vk::Extent2D const& extent, vk::ImageTiling tiling,
            vk::ImageUsageFlags usage, vk::ImageLayout initialLayout,
            vk::MemoryPropertyFlags memoryProperties,
            vk::ImageAspectFlags aspectMask);

  ImageData(std::nullptr_t) {}

  vk::Format format;
  vk::raii::Image image = nullptr;
  vk::raii::DeviceMemory deviceMemory = nullptr;
  vk::raii::ImageView imageView = nullptr;
};

vk::raii::DeviceMemory allocateDeviceMemory(
    const vk::raii::Device& device,
    const vk::PhysicalDeviceMemoryProperties& memoryProperties,
    const vk::MemoryRequirements& memoryRequirements,
    vk::MemoryPropertyFlags memoryPropertyFlags);

uint32_t findMemoryType(
    const vk::PhysicalDeviceMemoryProperties& memoryProperties,
    uint32_t typeBits, vk::MemoryPropertyFlags requirementsMask);

vk::raii::RenderPass createRenderPass(
    const vk::raii::Device& device, vk::Format colorFormat,
    vk::Format depthFormat,
    vk::AttachmentLoadOp loadOp = vk::AttachmentLoadOp::eClear,
    vk::ImageLayout colorFinalLayout = vk::ImageLayout::ePresentSrcKHR);

std::vector<vk::raii::Framebuffer> createFramebuffers(
    const vk::raii::Device& device, const vk::raii::RenderPass& renderPass,
    const std::vector<vk::raii::ImageView>& imageViews,
    const vk::raii::ImageView* pDepthImageView, const vk::Extent2D& extent);

std::vector<char> readFile(const std::string& filepath);
vk::raii::ShaderModule createShaderModule(const vk::raii::Device& device,
                                          std::vector<char>& code);

template <typename Func>
void oneTimeSubmit(const vk::raii::CommandBuffer& commandBuffer,
                   const vk::raii::Queue& queue, const Func& func) {
  commandBuffer.begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
  func(commandBuffer);
  commandBuffer.end();
  vk::SubmitInfo submitInfo(nullptr, nullptr, *commandBuffer);
  queue.submit(submitInfo, nullptr);
  queue.waitIdle();
}

template <typename Func>
void oneTimeSubmit(const vk::raii::Device& device,
                   const vk::raii::CommandPool& commandPool,
                   const vk::raii::Queue& queue, const Func& func) {
  vk::raii::CommandBuffers commandBuffers(
      device, {*commandPool, vk::CommandBufferLevel::ePrimary, 1});
  oneTimeSubmit(commandBuffers.front(), queue, func);
}

}  // namespace vgeu