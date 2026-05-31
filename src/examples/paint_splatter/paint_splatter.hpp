#pragma once

#include "vge_base.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

// std
#include <memory>
#include <vector>

namespace vge {

struct CanvasVertex {
  float position[3];
  float uv[2];
};

// GlobalUbo: camera matrices + canvas info
// Layout (std140-compatible):
//   projection  mat4  -- 0
//   view        mat4  -- 64
//   inverseView mat4  -- 128
//   canvasInfo  vec4  -- 192
//                        208 total
struct GlobalUbo {
  glm::mat4 projection{1.f};   // -- 0 --
  glm::mat4 view{1.f};         // -- 64 --
  glm::mat4 inverseView{1.f};  // -- 128 --
  glm::vec4 canvasInfo{
      0.f};  // -- 192 -- xy=half-extent, z=world size, w unused
  // -- 208 --
};
static_assert(sizeof(GlobalUbo) == 208, "GlobalUbo std140 size");

// Intentionally empty for M1; simulation/spoid knobs are added in later
// milestones.
struct Options {};

class VgeExample : public VgeBase {
public:
  VgeExample();
  ~VgeExample();

  void initVulkan() override;
  void getEnabledFeatures() override;
  void prepare() override;
  void render() override;
  void viewChanged() override;
  void setupCommandLineParser(CLI::App& app) override;
  void onUpdateUIOverlay() override;
  void buildCommandBuffers() override;

  void setOptions(const std::optional<Options>& o) {
    if (o) opts = *o;
  }

  Options opts{};

private:
  void createVertexBuffer();
  void createIndexBuffer();
  void createUniformBuffers();
  void createDescriptorSetLayout();
  void createDescriptorPool();
  void createDescriptorSets();
  void createPipelines();
  void updateGlobalUbo();
  void draw();

  // geometry
  std::unique_ptr<vgeu::VgeuBuffer> vertexBuffer;
  std::unique_ptr<vgeu::VgeuBuffer> indexBuffer;

  // per-frame uniform buffers
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> uniformBuffers;
  GlobalUbo globalUbo{};

  // descriptors
  std::vector<vk::raii::DescriptorSet> descriptorSets;
  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;

  // pipeline
  vk::raii::Pipeline pipeline = nullptr;

  static constexpr float kCanvasWorld = 4.0f;
};

}  // namespace vge
