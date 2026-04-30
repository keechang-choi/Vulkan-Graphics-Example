#include "vgeu_ibl.hpp"

#include "vgeu_utils.hpp"

// libs
#include <stb_image.h>

#include <glm/gtc/matrix_transform.hpp>

// std
#include <cassert>
#include <cmath>
#include <cstring>

namespace vgeu {

namespace {
struct CapturePushConstants {
  glm::mat4 mvp;
};
struct SkyboxPushConstants {
  glm::mat4 view;
  glm::mat4 projection;
  float lod = 0.0f;
};
}  // namespace

IBLBaker::IBLBaker(const vk::raii::Device& device, VmaAllocator allocator,
                   const vk::raii::Queue& transferQueue,
                   const vk::raii::CommandPool& commandPool)
    : device_(device),
      allocator_(allocator),
      transferQueue_(transferQueue),
      commandPool_(commandPool) {
  createSamplersAndCaptureMatrices();
}

void IBLBaker::createSamplersAndCaptureMatrices() {
  vk::SamplerCreateInfo samplerCI(
      {}, vk::Filter::eLinear, vk::Filter::eLinear,
      vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge, 0.f, false, 1.f, false,
      vk::CompareOp::eNever, 0.f,
      static_cast<float>(static_cast<uint32_t>(std::floor(std::log2(512))) + 1),
      vk::BorderColor::eFloatOpaqueWhite);
  iblSampler_ = vk::raii::Sampler(device_, samplerCI);

  vk::SamplerCreateInfo hdrSamplerCI(
      {}, vk::Filter::eLinear, vk::Filter::eLinear,
      vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge, 0.f, false, 1.f, false,
      vk::CompareOp::eNever, 0.f,
      static_cast<float>(static_cast<uint32_t>(std::floor(std::log2(512))) + 1),
      vk::BorderColor::eFloatOpaqueWhite);
  hdrSampler_ = vk::raii::Sampler(device_, hdrSamplerCI);

  captureProj_ = glm::perspective(glm::radians(90.f), 1.f, 0.1f, 512.f);
  captureViews_ = std::vector<glm::mat4>{
      // +X
      glm::lookAt(glm::vec3(0), glm::vec3(1, 0, 0), glm::vec3(0, -1, 0)),
      // -X
      glm::lookAt(glm::vec3(0), glm::vec3(-1, 0, 0), glm::vec3(0, -1, 0)),
      // +Y
      glm::lookAt(glm::vec3(0), glm::vec3(0, 1, 0), glm::vec3(0, 0, 1)),
      // -Y
      glm::lookAt(glm::vec3(0), glm::vec3(0, -1, 0), glm::vec3(0, 0, -1)),
      // +Z
      glm::lookAt(glm::vec3(0), glm::vec3(0, 0, 1), glm::vec3(0, -1, 0)),
      // -Z
      glm::lookAt(glm::vec3(0), glm::vec3(0, 0, -1), glm::vec3(0, -1, 0)),
  };
}

void IBLBaker::loadHdr(const std::string& path) {
  int w, h, c;
  float* pixels = stbi_loadf(path.c_str(), &w, &h, &c, 4);
  assert(pixels && "Failed to load HDR file");

  vk::DeviceSize size = static_cast<vk::DeviceSize>(w) * h * 4 * sizeof(float);

  vgeu::VgeuBuffer staging(
      allocator_, size, 1, vk::BufferUsageFlagBits::eTransferSrc,
      VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);
  std::memcpy(staging.getMappedData(), pixels, size);
  stbi_image_free(pixels);

  hdrTexture_ = std::make_unique<vgeu::VgeuImage>(
      device_, allocator_, vk::Format::eR32G32B32A32Sfloat,
      vk::Extent2D{(uint32_t)w, (uint32_t)h}, vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
      vk::ImageLayout::eUndefined, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
      vk::ImageAspectFlagBits::eColor, 1);

  vk::BufferImageCopy region(
      0, 0, 0,
      vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      vk::Offset3D{0, 0, 0}, vk::Extent3D{(uint32_t)w, (uint32_t)h, 1});

  vgeu::oneTimeSubmit(
      device_, commandPool_, transferQueue_,
      [&](const vk::raii::CommandBuffer& cmd) {
        vgeu::setImageLayout(
            cmd, hdrTexture_->getImage(), vk::Format::eR32G32B32A32Sfloat, 0, 1,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        cmd.copyBufferToImage(staging.getBuffer(), hdrTexture_->getImage(),
                              vk::ImageLayout::eTransferDstOptimal, region);
        vgeu::setImageLayout(cmd, hdrTexture_->getImage(),
                             vk::Format::eR32G32B32A32Sfloat, 0, 1,
                             vk::ImageLayout::eTransferDstOptimal,
                             vk::ImageLayout::eShaderReadOnlyOptimal);
      });
}

void IBLBaker::buildEnvCubemap(const IBLBakeConfig& config) {
  const vk::Format fmt = vk::Format::eR16G16B16A16Sfloat;
  const uint32_t dim = config.envCubemapSize;
  const uint32_t numMips =
      static_cast<uint32_t>(std::floor(std::log2(dim))) + 1;

  vk::raii::PipelineCache pipelineCache(device_, vk::PipelineCacheCreateInfo{});

  envCubemap_ = std::make_unique<vgeu::VgeuImage>(
      device_, allocator_, fmt, vk::Extent2D{dim, dim},
      vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
          vk::ImageUsageFlagBits::eTransferSrc,
      vk::ImageLayout::eUndefined, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
      vk::ImageAspectFlagBits::eColor, numMips, true /*isCubemap*/);

  auto offscreenImg = std::make_unique<vgeu::VgeuImage>(
      device_, allocator_, fmt, vk::Extent2D{dim, dim},
      vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eColorAttachment |
          vk::ImageUsageFlagBits::eTransferSrc,
      vk::ImageLayout::eUndefined, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
      vk::ImageAspectFlagBits::eColor, 1);

  vk::AttachmentDescription attDesc(
      {}, fmt, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
      vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
      vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
      vk::ImageLayout::eColorAttachmentOptimal);
  vk::AttachmentReference colorRef(0, vk::ImageLayout::eColorAttachmentOptimal);
  vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {},
                                 colorRef);
  std::array<vk::SubpassDependency, 2> deps{
      vk::SubpassDependency(VK_SUBPASS_EXTERNAL, 0,
                            vk::PipelineStageFlagBits::eBottomOfPipe,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::AccessFlagBits::eMemoryRead,
                            vk::AccessFlagBits::eColorAttachmentWrite,
                            vk::DependencyFlagBits::eByRegion),
      vk::SubpassDependency(0, VK_SUBPASS_EXTERNAL,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits::eBottomOfPipe,
                            vk::AccessFlagBits::eColorAttachmentWrite,
                            vk::AccessFlagBits::eMemoryRead,
                            vk::DependencyFlagBits::eByRegion)};
  auto captureRenderPass = vk::raii::RenderPass(
      device_, vk::RenderPassCreateInfo({}, attDesc, subpass, deps));

  vk::ImageView offscreenView = *offscreenImg->getImageView();
  auto captureFBO = vk::raii::Framebuffer(
      device_, vk::FramebufferCreateInfo({}, *captureRenderPass, offscreenView,
                                         dim, dim, 1));

  vk::DescriptorSetLayoutBinding hdrBinding(
      0, vk::DescriptorType::eCombinedImageSampler, 1,
      vk::ShaderStageFlagBits::eFragment);
  auto hdrDSL = vk::raii::DescriptorSetLayout(
      device_, vk::DescriptorSetLayoutCreateInfo({}, hdrBinding));

  std::vector<vk::DescriptorPoolSize> poolSizes{
      {vk::DescriptorType::eCombinedImageSampler, 1}};
  auto capturePool = vk::raii::DescriptorPool(
      device_,
      vk::DescriptorPoolCreateInfo(
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes));

  auto hdrDS =
      std::move(vk::raii::DescriptorSets(device_, vk::DescriptorSetAllocateInfo(
                                                      *capturePool, *hdrDSL))
                    .front());

  auto hdrImgInfo = hdrTexture_->descriptorImageInfo(
      *hdrSampler_, vk::ImageLayout::eShaderReadOnlyOptimal);
  device_.updateDescriptorSets(
      vk::WriteDescriptorSet(*hdrDS, 0, 0,
                             vk::DescriptorType::eCombinedImageSampler,
                             hdrImgInfo, nullptr),
      nullptr);

  vk::PushConstantRange pcRange(vk::ShaderStageFlagBits::eVertex, 0,
                                sizeof(CapturePushConstants));
  std::vector<vk::DescriptorSetLayout> setLayouts{*hdrDSL};
  auto capturePipelineLayout = vk::raii::PipelineLayout(
      device_, vk::PipelineLayoutCreateInfo({}, setLayouts, pcRange));

  auto vertCode =
      vgeu::readFile(config.commonShadersPath + "/equirect.vert.spv");
  auto fragCode =
      vgeu::readFile(config.commonShadersPath + "/equirect.frag.spv");
  auto vertMod = vgeu::createShaderModule(device_, vertCode);
  auto fragMod = vgeu::createShaderModule(device_, fragCode);

  std::array<vk::PipelineShaderStageCreateInfo, 2> stages{
      vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                                        *vertMod, "main"),
      vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment,
                                        *fragMod, "main")};
  vk::PipelineVertexInputStateCreateInfo emptyVI{};
  vk::PipelineInputAssemblyStateCreateInfo iaCI(
      {}, vk::PrimitiveTopology::eTriangleList);
  vk::PipelineRasterizationStateCreateInfo rasCI(
      {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
      vk::FrontFace::eCounterClockwise, false, 0, 0, 0, 1.f);
  vk::PipelineColorBlendAttachmentState blendAtt(
      false, {}, {}, {}, {}, {}, {},
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
  vk::PipelineColorBlendStateCreateInfo cbCI({}, false, {}, blendAtt);
  vk::PipelineDepthStencilStateCreateInfo dsCI({}, false, false,
                                               vk::CompareOp::eLessOrEqual);
  vk::PipelineViewportStateCreateInfo vpCI({}, 1, nullptr, 1, nullptr);
  vk::PipelineMultisampleStateCreateInfo msCI({}, vk::SampleCountFlagBits::e1);
  std::array<vk::DynamicState, 2> dynStates{vk::DynamicState::eViewport,
                                            vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dynCI({}, dynStates);

  auto capturePipeline = vk::raii::Pipeline(
      device_, pipelineCache,
      vk::GraphicsPipelineCreateInfo(
          {}, stages, &emptyVI, &iaCI, nullptr, &vpCI, &rasCI, &msCI, &dsCI,
          &cbCI, &dynCI, *capturePipelineLayout, *captureRenderPass));

  vgeu::oneTimeSubmit(
      device_, commandPool_, transferQueue_,
      [&](const vk::raii::CommandBuffer& cmd) {
        vgeu::setImageLayout(
            cmd, envCubemap_->getImage(), fmt,
            vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0,
                                      numMips, 0, 6},
            vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        vgeu::setImageLayout(cmd, offscreenImg->getImage(), fmt, 0, 1,
                             vk::ImageLayout::eUndefined,
                             vk::ImageLayout::eColorAttachmentOptimal);

        vk::ClearValue clearVal;
        clearVal.color = vk::ClearColorValue(0.f, 0.f, 0.f, 1.f);

        for (uint32_t f = 0; f < 6; ++f) {
          CapturePushConstants pc;
          pc.mvp = captureProj_ * captureViews_[f];

          cmd.beginRenderPass(
              vk::RenderPassBeginInfo(*captureRenderPass, *captureFBO,
                                      vk::Rect2D({}, vk::Extent2D{dim, dim}),
                                      clearVal),
              vk::SubpassContents::eInline);
          cmd.setViewport(
              0, vk::Viewport(0.f, 0.f, (float)dim, (float)dim, 0.f, 1.f));
          cmd.setScissor(0, vk::Rect2D({}, vk::Extent2D{dim, dim}));
          cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *capturePipeline);
          cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                 *capturePipelineLayout, 0, {*hdrDS}, nullptr);
          cmd.pushConstants<CapturePushConstants>(
              *capturePipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, pc);
          cmd.draw(36, 1, 0, 0);
          cmd.endRenderPass();

          vgeu::setImageLayout(cmd, offscreenImg->getImage(), fmt, 0, 1,
                               vk::ImageLayout::eColorAttachmentOptimal,
                               vk::ImageLayout::eTransferSrcOptimal);

          vk::ImageCopy copyRegion(
              vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0,
                                         1},
              vk::Offset3D{0, 0, 0},
              vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, f,
                                         1},
              vk::Offset3D{0, 0, 0}, vk::Extent3D{dim, dim, 1});
          cmd.copyImage(offscreenImg->getImage(),
                        vk::ImageLayout::eTransferSrcOptimal,
                        envCubemap_->getImage(),
                        vk::ImageLayout::eTransferDstOptimal, copyRegion);

          vgeu::setImageLayout(cmd, offscreenImg->getImage(), fmt, 0, 1,
                               vk::ImageLayout::eTransferSrcOptimal,
                               vk::ImageLayout::eColorAttachmentOptimal);
        }

        vgeu::setImageLayout(cmd, envCubemap_->getImage(), fmt,
                             vk::ImageSubresourceRange{
                                 vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6},
                             vk::ImageLayout::eTransferDstOptimal,
                             vk::ImageLayout::eTransferSrcOptimal);
      });

  vgeu::oneTimeSubmit(
      device_, commandPool_, transferQueue_,
      [&](const vk::raii::CommandBuffer& cmd) {
        // mip[0] is in eTransferSrcOptimal (transitioned at end of first
        // submit) mips[1..N-1] are in eTransferDstOptimal
        uint32_t mipWidth = dim, mipHeight = dim;
        for (uint32_t m = 1; m < numMips; ++m) {
          uint32_t nextW = mipWidth > 1 ? mipWidth / 2 : 1u;
          uint32_t nextH = mipHeight > 1 ? mipHeight / 2 : 1u;

          // Blit from mip[m-1] (SRC) to mip[m] (DST)
          vk::ImageBlit blit(
              vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, m - 1,
                                         0, 6},
              {vk::Offset3D{0, 0, 0},
               vk::Offset3D{(int)mipWidth, (int)mipHeight, 1}},
              vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, m, 0,
                                         6},
              {vk::Offset3D{0, 0, 0}, vk::Offset3D{(int)nextW, (int)nextH, 1}});
          cmd.blitImage(
              envCubemap_->getImage(), vk::ImageLayout::eTransferSrcOptimal,
              envCubemap_->getImage(), vk::ImageLayout::eTransferDstOptimal,
              blit, vk::Filter::eLinear);

          // mip[m-1] is done as source — transition to SHADER_READ_ONLY
          vgeu::setImageLayout(
              cmd, envCubemap_->getImage(), fmt,
              vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, m - 1,
                                        1, 0, 6},
              vk::ImageLayout::eTransferSrcOptimal,
              vk::ImageLayout::eShaderReadOnlyOptimal);

          if (m < numMips - 1) {
            // Prepare mip[m] as SRC for the next blit iteration
            vgeu::setImageLayout(
                cmd, envCubemap_->getImage(), fmt,
                vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, m, 1,
                                          0, 6},
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eTransferSrcOptimal);
          } else {
            // Last mip: DST -> SHADER_READ_ONLY
            vgeu::setImageLayout(
                cmd, envCubemap_->getImage(), fmt,
                vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, m, 1,
                                          0, 6},
                vk::ImageLayout::eTransferDstOptimal,
                vk::ImageLayout::eShaderReadOnlyOptimal);
          }

          mipWidth = nextW;
          mipHeight = nextH;
        }
      });
}

void IBLBaker::buildIrradianceMap(const IBLBakeConfig& config) {
  const vk::Format fmt = vk::Format::eR16G16B16A16Sfloat;
  const uint32_t dim = config.irradianceSize;
  // Irradiance is already a smooth low-frequency signal; only mip 0 is baked.
  // Using numMips>1 would leave higher mips uninitialized -> garbage reads.
  const uint32_t numMips = 1;

  vk::raii::PipelineCache pipelineCache(device_, vk::PipelineCacheCreateInfo{});

  irradianceMap_ = std::make_unique<vgeu::VgeuImage>(
      device_, allocator_, fmt, vk::Extent2D{dim, dim},
      vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
      vk::ImageLayout::eUndefined, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
      vk::ImageAspectFlagBits::eColor, numMips, true);

  auto offscreenImg = std::make_unique<vgeu::VgeuImage>(
      device_, allocator_, fmt, vk::Extent2D{dim, dim},
      vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eColorAttachment |
          vk::ImageUsageFlagBits::eTransferSrc,
      vk::ImageLayout::eUndefined, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
      vk::ImageAspectFlagBits::eColor, 1);

  vk::AttachmentDescription attDesc(
      {}, fmt, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
      vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
      vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
      vk::ImageLayout::eColorAttachmentOptimal);
  vk::AttachmentReference colorRef(0, vk::ImageLayout::eColorAttachmentOptimal);
  vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {},
                                 colorRef);
  std::array<vk::SubpassDependency, 2> deps{
      vk::SubpassDependency(VK_SUBPASS_EXTERNAL, 0,
                            vk::PipelineStageFlagBits::eBottomOfPipe,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::AccessFlagBits::eMemoryRead,
                            vk::AccessFlagBits::eColorAttachmentWrite,
                            vk::DependencyFlagBits::eByRegion),
      vk::SubpassDependency(0, VK_SUBPASS_EXTERNAL,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits::eBottomOfPipe,
                            vk::AccessFlagBits::eColorAttachmentWrite,
                            vk::AccessFlagBits::eMemoryRead,
                            vk::DependencyFlagBits::eByRegion)};
  auto captureRenderPass = vk::raii::RenderPass(
      device_, vk::RenderPassCreateInfo({}, attDesc, subpass, deps));
  vk::ImageView offscreenView = *offscreenImg->getImageView();
  auto captureFBO = vk::raii::Framebuffer(
      device_, vk::FramebufferCreateInfo({}, *captureRenderPass, offscreenView,
                                         dim, dim, 1));

  vk::DescriptorSetLayoutBinding envBinding(
      0, vk::DescriptorType::eCombinedImageSampler, 1,
      vk::ShaderStageFlagBits::eFragment);
  auto envDSL = vk::raii::DescriptorSetLayout(
      device_, vk::DescriptorSetLayoutCreateInfo({}, envBinding));

  std::vector<vk::DescriptorPoolSize> poolSizes{
      {vk::DescriptorType::eCombinedImageSampler, 1}};
  auto capturePool = vk::raii::DescriptorPool(
      device_,
      vk::DescriptorPoolCreateInfo(
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1, poolSizes));
  auto envDS =
      std::move(vk::raii::DescriptorSets(device_, vk::DescriptorSetAllocateInfo(
                                                      *capturePool, *envDSL))
                    .front());

  auto envImgInfo = envCubemap_->descriptorImageInfo(
      *iblSampler_, vk::ImageLayout::eShaderReadOnlyOptimal);
  device_.updateDescriptorSets(
      vk::WriteDescriptorSet(*envDS, 0, 0,
                             vk::DescriptorType::eCombinedImageSampler,
                             envImgInfo, nullptr),
      nullptr);

  // vertex: mat4 mvp at offset 0 (64 bytes)
  // fragment: IrradiancePush (numSamples, useJitter) at offset 64 (8 bytes)
  std::array<vk::PushConstantRange, 2> pcRanges{
      vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex, 0,
                            sizeof(CapturePushConstants)),
      vk::PushConstantRange(vk::ShaderStageFlagBits::eFragment,
                            sizeof(CapturePushConstants),
                            2u * sizeof(uint32_t))};
  std::vector<vk::DescriptorSetLayout> setLayouts{*envDSL};
  auto capturePipelineLayout = vk::raii::PipelineLayout(
      device_, vk::PipelineLayoutCreateInfo({}, setLayouts, pcRanges));

  auto vertCode =
      vgeu::readFile(config.commonShadersPath + "/equirect.vert.spv");
  auto fragCode =
      vgeu::readFile(config.commonShadersPath + "/irradiance.frag.spv");
  auto vertMod = vgeu::createShaderModule(device_, vertCode);
  auto fragMod = vgeu::createShaderModule(device_, fragCode);
  std::array<vk::PipelineShaderStageCreateInfo, 2> stages{
      vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                                        *vertMod, "main"),
      vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment,
                                        *fragMod, "main")};
  vk::PipelineVertexInputStateCreateInfo emptyVI{};
  vk::PipelineInputAssemblyStateCreateInfo iaCI(
      {}, vk::PrimitiveTopology::eTriangleList);
  vk::PipelineRasterizationStateCreateInfo rasCI(
      {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
      vk::FrontFace::eCounterClockwise, false, 0, 0, 0, 1.f);
  vk::PipelineColorBlendAttachmentState blendAtt(
      false, {}, {}, {}, {}, {}, {},
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
  vk::PipelineColorBlendStateCreateInfo cbCI({}, false, {}, blendAtt);
  vk::PipelineDepthStencilStateCreateInfo dsCI({}, false, false,
                                               vk::CompareOp::eLessOrEqual);
  vk::PipelineViewportStateCreateInfo vpCI({}, 1, nullptr, 1, nullptr);
  vk::PipelineMultisampleStateCreateInfo msCI({}, vk::SampleCountFlagBits::e1);
  std::array<vk::DynamicState, 2> dynStates{vk::DynamicState::eViewport,
                                            vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dynCI({}, dynStates);
  auto capturePipeline = vk::raii::Pipeline(
      device_, pipelineCache,
      vk::GraphicsPipelineCreateInfo(
          {}, stages, &emptyVI, &iaCI, nullptr, &vpCI, &rasCI, &msCI, &dsCI,
          &cbCI, &dynCI, *capturePipelineLayout, *captureRenderPass));

  struct IrradiancePush {
    uint32_t numSamples;
    uint32_t useJitter;
  };

  vgeu::oneTimeSubmit(
      device_, commandPool_, transferQueue_,
      [&](const vk::raii::CommandBuffer& cmd) {
        vgeu::setImageLayout(
            cmd, irradianceMap_->getImage(), fmt,
            vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0,
                                      numMips, 0, 6},
            vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        vgeu::setImageLayout(cmd, offscreenImg->getImage(), fmt, 0, 1,
                             vk::ImageLayout::eUndefined,
                             vk::ImageLayout::eColorAttachmentOptimal);

        IrradiancePush push{config.irradianceSamples,
                            config.useJitter ? 1u : 0u};
        vk::ClearValue clearVal;
        clearVal.color = vk::ClearColorValue(0.f, 0.f, 0.f, 1.f);

        // fragment push constants are the same for all faces
        cmd.pushConstants<IrradiancePush>(*capturePipelineLayout,
                                          vk::ShaderStageFlagBits::eFragment,
                                          sizeof(CapturePushConstants), push);

        for (uint32_t f = 0; f < 6; ++f) {
          CapturePushConstants pc;
          pc.mvp = captureProj_ * captureViews_[f];

          cmd.beginRenderPass(
              vk::RenderPassBeginInfo(*captureRenderPass, *captureFBO,
                                      vk::Rect2D({}, vk::Extent2D{dim, dim}),
                                      clearVal),
              vk::SubpassContents::eInline);
          cmd.setViewport(0,
                          vk::Viewport(0, 0, (float)dim, (float)dim, 0.f, 1.f));
          cmd.setScissor(0, vk::Rect2D({}, vk::Extent2D{dim, dim}));
          cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *capturePipeline);
          cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                 *capturePipelineLayout, 0, {*envDS}, nullptr);
          cmd.pushConstants<CapturePushConstants>(
              *capturePipelineLayout, vk::ShaderStageFlagBits::eVertex, 0, pc);
          cmd.draw(36, 1, 0, 0);
          cmd.endRenderPass();

          vgeu::setImageLayout(cmd, offscreenImg->getImage(), fmt, 0, 1,
                               vk::ImageLayout::eColorAttachmentOptimal,
                               vk::ImageLayout::eTransferSrcOptimal);
          vk::ImageCopy copyRegion(
              vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0,
                                         1},
              {},
              vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, f,
                                         1},
              {}, vk::Extent3D{dim, dim, 1});
          cmd.copyImage(offscreenImg->getImage(),
                        vk::ImageLayout::eTransferSrcOptimal,
                        irradianceMap_->getImage(),
                        vk::ImageLayout::eTransferDstOptimal, copyRegion);
          vgeu::setImageLayout(cmd, offscreenImg->getImage(), fmt, 0, 1,
                               vk::ImageLayout::eTransferSrcOptimal,
                               vk::ImageLayout::eColorAttachmentOptimal);
        }

        vgeu::setImageLayout(
            cmd, irradianceMap_->getImage(), fmt,
            vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0,
                                      numMips, 0, 6},
            vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal);
      });
}

void IBLBaker::buildPrefilteredMap(const IBLBakeConfig&) {
  // Task 8에서 이식
}

void IBLBaker::buildBrdfLut(const IBLBakeConfig&) {
  // Task 9에서 이식
}

void IBLBaker::bake(const IBLBakeConfig& config) {
  loadHdr(config.hdrPath);
  buildEnvCubemap(config);
  buildIrradianceMap(config);
  buildPrefilteredMap(config);
  buildBrdfLut(config);
}

void IBLBaker::rebakeFiltering(const IBLBakeConfig& config) {
  buildIrradianceMap(config);
  buildPrefilteredMap(config);
}

// Skybox 구현은 Task 11에서

}  // namespace vgeu
