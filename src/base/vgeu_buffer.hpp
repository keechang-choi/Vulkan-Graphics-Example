#pragma once

// libs
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

namespace vgeu {

// TODO: buffer Data
// buffer, deviceMemory,
class VgeuBuffer {
 public:
  VgeuBuffer(const vk::raii::PhysicalDevice& physicalDevice,
             const vk::raii::Device& device, vk::DeviceSize instanceSize,
             uint32_t instanceCount, vk::BufferUsageFlags usageFlags,
             vk::MemoryPropertyFlags propertyFlags,
             vk::DeviceSize minOffsetAligment = 1);
  ~VgeuBuffer();

  VgeuBuffer(const VgeuBuffer&) = delete;
  VgeuBuffer& operator=(const VgeuBuffer&) = delete;

  void map(vk::DeviceSize size = VK_WHOLE_SIZE, vk::DeviceSize offset = 0);
  void unmap();

  void writeToBuffer(void* data, vk::DeviceSize size = VK_WHOLE_SIZE,
                     vk::DeviceSize offset = 0);
  vk::Result flush(vk::DeviceSize size = VK_WHOLE_SIZE,
                   vk::DeviceSize offset = 0);
  vk::DescriptorBufferInfo descriptorInfo(vk::DeviceSize size = VK_WHOLE_SIZE,
                                          vk::DeviceSize offset = 0);
  vk::Result invalidate(vk::DeviceSize size = VK_WHOLE_SIZE,
                        vk::DeviceSize offset = 0);
  // TODO: getters

 private:
  static vk::DeviceSize getAlignment(vk::DeviceSize instanceSize,
                                     vk::DeviceSize minOffsetAlignment);
  vk::DeviceSize instanceSize;
  uint32_t instanceCount;
  vk::DeviceSize alignmentSize;
  vk::DeviceSize bufferSize;
  vk::raii::Buffer buffer = nullptr;
  vk::raii::DeviceMemory memory = nullptr;
  void* mapped = nullptr;
  vk::BufferUsageFlags usageFlags;
  vk::MemoryPropertyFlags memoryPropertyFlags;
};

}  // namespace vgeu