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

}  // namespace vgeu