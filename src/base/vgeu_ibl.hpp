#pragma once

#include "vgeu_buffer.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>

// std
#include <memory>
#include <string>
#include <vector>

namespace vgeu {

struct IBLBakeConfig {
  std::string hdrPath;
  std::string commonShadersPath;
  uint32_t envCubemapSize = 512;
  uint32_t irradianceSize = 64;
  uint32_t prefilteredSize = 128;
  uint32_t brdfLutSize = 512;
  uint32_t irradianceSamples = 2048;
  uint32_t prefilteredSamples = 1024;
  bool useJitter = true;
};

class IBLBaker {
public:
  IBLBaker(const vk::raii::Device& device, VmaAllocator allocator,
           const vk::raii::Queue& transferQueue,
           const vk::raii::CommandPool& commandPool);

  // Full bake: HDR -> envCubemap -> irradiance -> prefiltered -> BRDF LUT.
  void bake(const IBLBakeConfig& config);

  // Re-bake irradiance + prefiltered only (e.g., on jitter toggle).
  void rebakeFiltering(const IBLBakeConfig& config);

  const VgeuImage& envCubemap() const { return *envCubemap_; }
  const VgeuImage& irradianceMap() const { return *irradianceMap_; }
  const VgeuImage& prefilteredMap() const { return *prefilteredMap_; }
  const VgeuImage& brdfLut() const { return *brdfLut_; }
  const vk::raii::Sampler& iblSampler() const { return iblSampler_; }
  const vk::raii::Sampler& hdrSampler() const { return hdrSampler_; }

private:
  void createSamplersAndCaptureMatrices();
  void loadHdr(const std::string& path);
  void buildEnvCubemap(const IBLBakeConfig&);
  void buildIrradianceMap(const IBLBakeConfig&);
  void buildPrefilteredMap(const IBLBakeConfig&);
  void buildBrdfLut(const IBLBakeConfig&);

  const vk::raii::Device& device_;
  VmaAllocator allocator_;
  const vk::raii::Queue& transferQueue_;
  const vk::raii::CommandPool& commandPool_;

  std::unique_ptr<VgeuImage> hdrTexture_;
  std::unique_ptr<VgeuImage> envCubemap_;
  std::unique_ptr<VgeuImage> irradianceMap_;
  std::unique_ptr<VgeuImage> prefilteredMap_;
  std::unique_ptr<VgeuImage> brdfLut_;
  vk::raii::Sampler iblSampler_ = nullptr;
  vk::raii::Sampler hdrSampler_ = nullptr;
  glm::mat4 captureProj_;
  std::vector<glm::mat4> captureViews_;
};

class Skybox {
public:
  Skybox(const vk::raii::Device& device,
         const vk::raii::PipelineCache& pipelineCache,
         const vk::raii::DescriptorPool& descPool,
         const vk::raii::RenderPass& renderPass,
         const std::string& commonShadersPath, const IBLBaker& iblBaker,
         uint32_t maxFramesInFlight);

  // Bind pipeline + descriptor + push constants and issue draw.
  void draw(const vk::raii::CommandBuffer& cmd, uint32_t frameIndex,
            const glm::mat4& view, const glm::mat4& proj, float lod = 0.0f);

private:
  const vk::raii::Device& device_;
  vk::raii::DescriptorSetLayout descSetLayout_ = nullptr;
  vk::raii::PipelineLayout pipelineLayout_ = nullptr;
  vk::raii::Pipeline pipeline_ = nullptr;
  std::vector<vk::raii::DescriptorSet> descriptorSets_;
};

}  // namespace vgeu
