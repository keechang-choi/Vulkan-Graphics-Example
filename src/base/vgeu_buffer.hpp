#pragma once

// libs
#include <vulkan/vulkan.h>
//
// #define VMA_VULKAN_VERSION 1003000
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

namespace vgeu {

// custom VMA allcator for RAII paradigm(for destruction order)
class VgeuAllocator {
 public:
  VgeuAllocator(VkDevice device, VkPhysicalDevice physicalDevice,
                VkInstance instance, uint32_t apiVersion);
  ~VgeuAllocator();

  VgeuAllocator(const VgeuAllocator&) = delete;
  VgeuAllocator& operator=(const VgeuAllocator&) = delete;

  VmaAllocator getAllocator() const { return allocator; };
  VmaAllocator operator*() const { return allocator; };

 private:
  // NOTE: allocator should be free after all members destruction,
  // not first(in Base destructor)
  // Also, should be initialized with nullptr or check not null when destruct.
  VmaAllocator allocator = nullptr;
};

class VgeuBuffer {
 public:
  VgeuBuffer(VmaAllocator allocator, vk::DeviceSize instanceSize,
             uint32_t instanceCount, vk::BufferUsageFlags usageFlags,
             VmaMemoryUsage memUsage,
             VmaAllocationCreateFlags allocCreateFlags);
  ~VgeuBuffer();

  VgeuBuffer(const VgeuBuffer&) = delete;
  VgeuBuffer& operator=(const VgeuBuffer&) = delete;

  vk::DescriptorBufferInfo descriptorInfo(vk::DeviceSize size = VK_WHOLE_SIZE,
                                          vk::DeviceSize offset = 0);

  vk::Buffer getBuffer() const { return buffer; }
  void* getMappedData() const { return allocInfo.pMappedData; }
  vk::DeviceSize getBufferSize() const { return bufferSize; }
  uint32_t getInstanceCount() const { return instanceCount; }

 private:
  VmaAllocator allocator = VK_NULL_HANDLE;
  vk::DeviceSize instanceSize;
  uint32_t instanceCount;
  vk::DeviceSize bufferSize;
  vk::Buffer buffer = nullptr;
  VmaAllocation alloc = VK_NULL_HANDLE;
  VmaAllocationInfo allocInfo{};
};
class VgeuImage {
 public:
  // device for image view
  VgeuImage(const vk::raii::Device& device, VmaAllocator allocator,
                     vk::Format format, const vk::Extent2D& extent,
                     vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                     vk::ImageLayout initialLayout, VmaMemoryUsage memUsage,
                     VmaAllocationCreateFlags allocCreateFlags,
                     vk::ImageAspectFlags aspectMask, uint32_t mipLevels));
  ~VgeuImage();

  VgeuImage(const VgeuImage&) = delete;
  VgeuImage& operator=(const VgeuImage&) = delete;

  vk::Image getImage() const { return image; }
  const vk::raii::ImageView& getImageView() const { return imageView; }

 private:
  VmaAllocator allocator = VK_NULL_HANDLE;
  vk::Format format;
  vk::Image image = nullptr;
  vk::raii::ImageView imageView = nullptr;
  VmaAllocation alloc = VK_NULL_HANDLE;
  VmaAllocationInfo allocInfo{};
};

}  // namespace vgeu