#define VMA_IMPLEMENTATION
#include "vgeu_buffer.hpp"

// libs
#include <vulkan/vulkan.h>

#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>
#include <cassert>
namespace vgeu {

VgeuBuffer::VgeuBuffer(VmaAllocator allocator, vk::DeviceSize instanceSize,
                       uint32_t instanceCount, vk::BufferUsageFlags usageFlags,
                       VmaMemoryUsage memUsage,
                       VmaAllocationCreateFlags allocCreateFlags)
    : allocator(allocator),
      instanceSize(instanceSize),
      instanceCount(instanceCount),
      bufferSize(instanceCount * instanceSize) {
  VkBufferCreateInfo vkBufferCI{};
  vkBufferCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  vkBufferCI.size = bufferSize;
  vkBufferCI.usage = static_cast<VkBufferUsageFlags>(usageFlags);
  vkBufferCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo allocCI{};
  allocCI.usage = memUsage;
  allocCI.flags = allocCreateFlags;

  VkBuffer vkBuffer = VK_NULL_HANDLE;
  VmaAllocationInfo allocInfo{};
  VkResult result = vmaCreateBuffer(allocator, &vkBufferCI, &allocCI, &vkBuffer,
                                    &alloc, &allocInfo);
  assert(result == VK_SUCCESS && "VMA ERROR: failed to create buffer.");
  buffer = vk::Buffer(vkBuffer);
}
VgeuBuffer::~VgeuBuffer() {
  vmaDestroyBuffer(allocator, VkBuffer(buffer), alloc);
}
vk::DescriptorBufferInfo VgeuBuffer::descriptorInfo(vk::DeviceSize size,
                                                    vk::DeviceSize offset) {
  return vk::DescriptorBufferInfo(buffer, offset, size);
}
}  // namespace vgeu