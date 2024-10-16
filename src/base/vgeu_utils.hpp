#pragma once

/*

refence: sample util code in vulkan-hpp
https://github.com/KhronosGroup/Vulkan-Hpp

*/

//
#include "vgeu_buffer.hpp"

// libs
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

// std
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace vgeu {
// from: https://stackoverflow.com/a/57595105
template <typename T, typename... Rest>
void hashCombine(std::size_t& seed, const T& v, const Rest&... rest) {
  if constexpr (sizeof(size_t) >= 8) {
    seed ^= std::hash<T>{}(v) + 0x9e3779b97f4a7c15 + (seed << 6) + (seed >> 2);
  } else {
    seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  (hashCombine(seed, rest), ...);
};

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
bool isDeviceExtensionSupported(
    const std::vector<std::string>& supportedDeviceExtensions,
    const std::string& extension);
vk::raii::DebugUtilsMessengerEXT setupDebugMessenger(
    vk::raii::Instance& instance);

vk::raii::Device createLogicalDevice(
    const vk::raii::PhysicalDevice& physicalDevice,
    const QueueFamilyIndices& queueFamilyIndices,
    const std::vector<std::string>& supportedDeviceExtensions,
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

// NOTE:  Deprecated. Use VMA image
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

// create on command buffer from the pool
// record, submit, then wait for the queue to be idle.
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

// transition by image memory barrier.
// internally select stage, accessMask.
void setImageLayout(const vk::raii::CommandBuffer& commandBuffer,
                    vk::Image image, vk::Format format,
                    vk::ImageSubresourceRange imageSubresourceRange,
                    vk::ImageLayout oldImageLayout,
                    vk::ImageLayout newImageLayout);

// transition by image memory barrier with base and count for mipLevels
void setImageLayout(const vk::raii::CommandBuffer& commandBuffer,
                    vk::Image image, vk::Format format, uint32_t baseMipLevel,
                    uint32_t levelCount, vk::ImageLayout oldImageLayout,
                    vk::ImageLayout newImageLayout);

// get aligned size of uniform/storage buffer
size_t padBufferSize(const vk::raii::PhysicalDevice physicalDevice,
                     size_t originalSize, bool isUniformType);

// mainly used for ownership release and acquire
void addQueueFamilyOwnershipTransferBarriers(
    uint32_t srcQueueFamilyIndex, uint32_t dstQueueFamilyIndex,
    const vk::raii::CommandBuffer& cmdBuffer,
    const std::vector<const vgeu::VgeuBuffer*>& targetBufferPtrs,
    vk::AccessFlags srcAccessMask, vk::AccessFlags dstAccessMask,
    vk::PipelineStageFlags srcStageMask, vk::PipelineStageFlags dstStageMask);

// mainly used for compute dispatch execution order
void addComputeToComputeBarriers(
    const vk::raii::CommandBuffer& cmdBuffer,
    const std::vector<const vgeu::VgeuBuffer*>& targetBufferPtrs);

}  // namespace vgeu