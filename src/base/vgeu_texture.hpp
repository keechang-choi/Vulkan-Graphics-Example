
/*
texture loading class based on stb_image

base code from
https://github.com/SaschaWillems/Vulkan/blob/master/base/VulkanTexture.h
*/

// modified existing structure to fit in RAII paradigm.

#pragma once

//
#include "vgeu_buffer.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

// std
#include <memory>

namespace vgeu {

class Texture {
 public:
  std::unique_ptr<vgeu::VgeuImage> vgeuImage;
  vk::ImageLayout imageLayout{};
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t mipLevels = 0;
  uint32_t layerCount = 0;
  vk::DescriptorImageInfo descriptorInfo{};
  vk::raii::Sampler sampler = nullptr;

  Texture() = default;
  // empty texture
  Texture(const vk::raii::Device& device, VmaAllocator allocator,
          const vk::raii::Queue& transferQueue,
          const vk::raii::CommandPool& commandPool);

  void generateMipmaps(const vk::raii::CommandBuffer& cmdBuffer);
  void createEmptyTexture(const vk::raii::Device& device,
                          VmaAllocator allocator,
                          const vk::raii::Queue& transferQueue,
                          const vk::raii::CommandPool& commandPool);
  // NOTE: use mipLevels
  void createSampler(const vk::raii::Device& device);
  // NOTE: used at the end of fromglTFImage()
  void updateDescriptorInfo();
};

}  // namespace vgeu