#pragma once

// libs
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

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

 private:
  VmaAllocator allocator = VK_NULL_HANDLE;
  vk::DeviceSize instanceSize;
  uint32_t instanceCount;
  vk::DeviceSize bufferSize;
  vk::Buffer buffer = nullptr;
  VmaAllocation alloc = VK_NULL_HANDLE;
};

}  // namespace vgeu