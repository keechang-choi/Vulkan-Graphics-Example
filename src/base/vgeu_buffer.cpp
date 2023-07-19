#include "vgeu_buffer.hpp"

namespace vgeu {

VgeuBuffer::VgeuBuffer(const vk::raii::PhysicalDevice& physicalDevice,
                       const vk::raii::Device& device,
                       vk::DeviceSize instanceSize, uint32_t instanceCount,
                       vk::BufferUsageFlags usageFlags,
                       vk::MemoryPropertyFlags propertyFlags,
                       vk::DeviceSize minOffsetAligment)
    : instanceSize(instanceSize),
      instanceCount(instanceCount),
      alignmentSize(getAlignment(instanceSize, minOffsetAligment)),
      bufferSize(instanceCount * alignmentSize),
      buffer(device, vk::BufferCreateInfo({}, bufferSize, usageFlags)),
      usageFlags(usageFlags),
      memoryPropertyFlags(memoryPropertyFlags) {
  buffer.bindMemory(*memory, 0);
}

VgeuBuffer::~VgeuBuffer() { unmap(); }

void VgeuBuffer::map(vk::DeviceSize size, vk::DeviceSize offset) {
  // TODO: assert buffer and memory created.
  // need to check empty or not -> unique_ptr or optioanl;
  assert(VkBuffer(*buffer) && VkDeviceMemory(*memory) &&
         "Called map on buffer before create");
  mapped = memory.mapMemory(offset, size, {});
}

vk::Result VgeuBuffer::flush(vk::DeviceSize size, vk::DeviceSize offset) {
  buffer.getDevice().flushMappedMemoryRanges(1, nullptr);
  VkBuffer b;
  vk::Buffer bb(b);
}
}  // namespace vgeu