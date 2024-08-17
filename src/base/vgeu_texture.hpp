
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
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>

// std
#include <memory>

namespace vgeu {

bool isKtx(const tinygltf::Image& gltfImage);
bool loadImageDataFunc(tinygltf::Image* gltfImage, const int imageIndex,
                       std::string* error, std::string* warning, int req_width,
                       int req_height, const unsigned char* bytes, int size,
                       void* userData);
bool loadImageDataFuncEmpty(tinygltf::Image* image, const int imageIndex,
                            std::string* error, std::string* warning,
                            int req_width, int req_height,
                            const unsigned char* bytes, int size,
                            void* userData);

struct Texture {
  std::unique_ptr<vgeu::VgeuImage> vgeuImage;
  vk::ImageLayout imageLayout{};
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t mipLevels = 0;
  uint32_t layerCount = 0;
  vk::DescriptorImageInfo descriptorInfo{};
  vk::raii::Sampler sampler = nullptr;

  // fromglTFImage
  Texture(const tinygltf::Image& gltfimage, const vk::raii::Device& device,
          VmaAllocator allocator, const vk::raii::Queue& transferQueue,
          const vk::raii::CommandPool& commandPool);
  // empty texture
  Texture(const vk::raii::Device& device, VmaAllocator allocator,
          const vk::raii::Queue& transferQueue,
          const vk::raii::CommandPool& commandPool);

  // TODO: move into gltf using inheritance
  void fromglTFImage(const tinygltf::Image& gltfimage,
                     const vk::raii::Device& device, VmaAllocator allocator,
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