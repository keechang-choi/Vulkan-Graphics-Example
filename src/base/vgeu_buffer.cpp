// from vma docs
#define VMA_IMPLEMENTATION
#include "vgeu_buffer.hpp"

// libs
#include <vulkan/vulkan.h>

#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>
#include <cassert>

// std
#include <iostream>

namespace vgeu {

VgeuAllocator::VgeuAllocator(VkDevice device, VkPhysicalDevice physicalDevice,
                             VkInstance instance, uint32_t apiVersion) {
  // create VMA global allocator
  VmaAllocatorCreateInfo allocatorCI{};
  allocatorCI.physicalDevice = physicalDevice;
  allocatorCI.device = device;
  allocatorCI.instance = instance;
  allocatorCI.vulkanApiVersion = apiVersion;
  // TOOD: check vma flags
  // for higher version Vulkan API
  VmaVulkanFunctions vulkanFunctions = {};
  vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
  allocatorCI.pVulkanFunctions = &vulkanFunctions;

  VkResult result = vmaCreateAllocator(&allocatorCI, &allocator);
  assert(result == VK_SUCCESS && "VMA allocator create Error");
}

VgeuAllocator::~VgeuAllocator() {
  if (allocator != VK_NULL_HANDLE) {
    vmaDestroyAllocator(allocator);
    allocator = VK_NULL_HANDLE;
  }
}

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
  VkResult result = vmaCreateBuffer(allocator, &vkBufferCI, &allocCI, &vkBuffer,
                                    &alloc, &allocInfo);
  assert(result == VK_SUCCESS && "VMA ERROR: failed to create buffer.");
  buffer = vk::Buffer(vkBuffer);
}
VgeuBuffer::~VgeuBuffer() {
  vmaDestroyBuffer(allocator, static_cast<VkBuffer>(buffer), alloc);
}
vk::DescriptorBufferInfo VgeuBuffer::descriptorInfo(vk::DeviceSize size,
                                                    vk::DeviceSize offset) {
  return vk::DescriptorBufferInfo(buffer, offset, size);
}

VgeuImage::VgeuImage(const vk::raii::Device& device, VmaAllocator allocator,
                     vk::Format format, const vk::Extent2D& extent,
                     vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                     vk::ImageLayout initialLayout,
                     vk::ImageAspectFlags aspectMask, VmaMemoryUsage memUsage,
                     VmaAllocationCreateFlags allocCreateFlags)
    : allocator(allocator) {
  VkImageCreateInfo vkImageCI{};
  vkImageCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  vkImageCI.flags = 0;
  vkImageCI.imageType = VK_IMAGE_TYPE_2D;
  vkImageCI.format = static_cast<VkFormat>(format);
  vkImageCI.extent.width = extent.width;
  vkImageCI.extent.height = extent.height;
  vkImageCI.extent.depth = 1;
  vkImageCI.mipLevels = 1;
  vkImageCI.arrayLayers = 1;
  vkImageCI.samples = VK_SAMPLE_COUNT_1_BIT;
  vkImageCI.tiling = static_cast<VkImageTiling>(tiling);
  vkImageCI.usage = static_cast<VkImageUsageFlags>(usage);
  vkImageCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  vkImageCI.initialLayout = static_cast<VkImageLayout>(initialLayout);

  VmaAllocationCreateInfo allocCI{};
  allocCI.usage = memUsage;
  allocCI.flags = allocCreateFlags;

  VkImage vkImage = VK_NULL_HANDLE;
  VkResult result = vmaCreateImage(allocator, &vkImageCI, &allocCI, &vkImage,
                                   &alloc, &allocInfo);
  assert(result == VK_SUCCESS && "VMA ERROR: failed to create image.");
  image = vk::Image(vkImage);

  imageView = vk::raii::ImageView(
      device, vk::ImageViewCreateInfo(vk::ImageViewCreateFlags(), image,
                                      vk::ImageViewType::e2D, format, {},
                                      {aspectMask, 0, 1, 0, 1}));
}

VgeuImage::~VgeuImage() {
  // std::cout << "Call: VgeuImage Destructor" << std::endl;
  vmaDestroyImage(allocator, static_cast<VkImage>(image), alloc);
}

}  // namespace vgeu