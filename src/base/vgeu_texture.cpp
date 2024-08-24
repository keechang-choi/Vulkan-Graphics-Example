#include "vgeu_texture.hpp"

#include "vgeu_utils.hpp"

// libs
// #define STB_IMAGE_IMPLEMENTATION
// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include <stb_image.h>

namespace vgeu {

Texture::Texture(const vk::raii::Device& device, VmaAllocator allocator,
                 const vk::raii::Queue& transferQueue,
                 const vk::raii::CommandPool& commandPool) {
  createEmptyTexture(device, allocator, transferQueue, commandPool);
}

void Texture::generateMipmaps(const vk::raii::CommandBuffer& cmdBuffer) {
  uint32_t mipWidth = width;
  uint32_t mipHeight = height;
  for (uint32_t i = 1; i < mipLevels; i++) {
    setImageLayout(cmdBuffer, vgeuImage->getImage(), vgeuImage->getFormat(),
                   i - 1, 1, vk::ImageLayout::eTransferDstOptimal,
                   vk::ImageLayout::eTransferSrcOptimal);
    uint32_t nextMipWidth = mipWidth > 1 ? mipWidth / 2 : 1u;
    uint32_t nextMipHeight = mipHeight > 1 ? mipHeight / 2 : 1u;

    vk::ImageBlit region(
        vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i - 1, 0,
                                   1},
        {
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{static_cast<int>(mipWidth),
                         static_cast<int>(mipHeight), 1},
        },
        vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i, 0, 1},
        {
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{static_cast<int>(nextMipWidth),
                         static_cast<int>(nextMipHeight), 1},
        });

    cmdBuffer.blitImage(
        vgeuImage->getImage(), vk::ImageLayout::eTransferSrcOptimal,
        vgeuImage->getImage(), vk::ImageLayout::eTransferDstOptimal, region,
        vk::Filter::eLinear);
    setImageLayout(cmdBuffer, vgeuImage->getImage(), vgeuImage->getFormat(),
                   i - 1, 1, vk::ImageLayout::eTransferSrcOptimal,
                   vk::ImageLayout::eShaderReadOnlyOptimal);
    mipWidth = nextMipWidth;
    mipHeight = nextMipHeight;
  }
  setImageLayout(cmdBuffer, vgeuImage->getImage(), vgeuImage->getFormat(),
                 mipLevels - 1, 1, vk::ImageLayout::eTransferDstOptimal,
                 vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Texture::createEmptyTexture(const vk::raii::Device& device,
                                 VmaAllocator allocator,
                                 const vk::raii::Queue& transferQueue,
                                 const vk::raii::CommandPool& commandPool) {
  width = 1;
  height = 1;
  layerCount = 1;
  mipLevels = 1;
  unsigned char buffer = 0u;

  {
    vgeu::VgeuBuffer stagingBuffer(
        allocator, 4, width * height, vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);

    std::memcpy(stagingBuffer.getMappedData(), &buffer,
                stagingBuffer.getBufferSize());

    vgeuImage = std::make_unique<VgeuImage>(
        device, allocator, vk::Format::eR8G8B8A8Unorm,
        vk::Extent2D(width, height), vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::ImageLayout::eUndefined, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
        VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        vk::ImageAspectFlagBits::eColor, mipLevels);

    // NOTE: 0 for buffer packed tightly
    vk::BufferImageCopy region(
        0, 0, 0,
        vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        vk::Offset3D{0, 0, 0}, vk::Extent3D{width, height, 1});
    // layout transition
    oneTimeSubmit(device, commandPool, transferQueue,
                  [&](const vk::raii::CommandBuffer& cmdBuffer) {
                    setImageLayout(cmdBuffer, vgeuImage->getImage(),
                                   vgeuImage->getFormat(), 0, mipLevels,
                                   vk::ImageLayout::eUndefined,
                                   vk::ImageLayout::eTransferDstOptimal);
                    cmdBuffer.copyBufferToImage(
                        stagingBuffer.getBuffer(), vgeuImage->getImage(),
                        vk::ImageLayout::eTransferDstOptimal, region);
                    setImageLayout(cmdBuffer, vgeuImage->getImage(),
                                   vgeuImage->getFormat(), 0, mipLevels,
                                   vk::ImageLayout::eTransferDstOptimal,
                                   vk::ImageLayout::eShaderReadOnlyOptimal);
                  });
    imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  }
  createSampler(device);
  updateDescriptorInfo();
}

void Texture::createSampler(const vk::raii::Device& device) {
  // NOTE: maxAnisotorpy fixed. may get it from physical Device.
  vk::SamplerCreateInfo samplerCI(
      vk::SamplerCreateFlags{}, vk::Filter::eLinear, vk::Filter::eLinear,
      vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, 0.f,
      true, 8.0f, false, vk::CompareOp::eNever, 0.f,
      static_cast<float>(mipLevels), vk::BorderColor::eFloatOpaqueWhite);

  sampler = vk::raii::Sampler(device, samplerCI);
}

void Texture::updateDescriptorInfo() {
  descriptorInfo.sampler = *sampler;
  descriptorInfo.imageView = *vgeuImage->getImageView();
  descriptorInfo.imageLayout = imageLayout;
}
}  // namespace vgeu