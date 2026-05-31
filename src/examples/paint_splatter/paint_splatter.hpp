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

// Particle: std430, 64 bytes
//   pos      vec4  -- 0  -- xyz position, w=1
//   vel      vec4  -- 16 -- xyz velocity, w=wetness/alpha
//   predict  vec4  -- 32 -- xyz predicted x*, w=lambda
//   color    vec4  -- 48 -- rgba paint color (a=concentration)
//                     64
struct Particle {
  glm::vec4 pos;      // -- 0 --
  glm::vec4 vel;      // -- 16 --
  glm::vec4 predict;  // -- 32 --
  glm::vec4 color;    // -- 48 --
  // -- 64 --
};
static_assert(sizeof(Particle) == 64, "Particle std430 size");

// ComputeUbo: std140, 48 bytes
//   dt            float  -- 0
//   particleCount uint   -- 4
//   gravity       float  -- 8
//   _pad0         float  -- 12
//   canvasMin     vec4   -- 16
//   canvasMax     vec4   -- 32
//                           48
struct ComputeUbo {
  float dt;                // -- 0 --
  uint32_t particleCount;  // -- 4 --
  float gravity;           // -- 8 --
  float _pad0;             // -- 12 --
  glm::vec4 canvasMin;  // -- 16 -- xyz world-min of fluid domain, w=cell size h
  glm::vec4 canvasMax;  // -- 32 -- xyz world-max, w unused
  // -- 48 --
};
static_assert(sizeof(ComputeUbo) == 48, "ComputeUbo std140 size");

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

  // ---- particle SSBO helpers ----
  void createParticleBuffers();
  void createParticlePipeline();

  // ---- compute helpers ----
  void prepareCompute();
  void createComputeDescriptorSetLayout();
  void createComputeDescriptorSets();
  void createComputePipeline();
  void updateComputeUbo();
  void buildComputeCommandBuffers();

  // geometry
  std::unique_ptr<vgeu::VgeuBuffer> vertexBuffer;
  std::unique_ptr<vgeu::VgeuBuffer> indexBuffer;

  // per-frame uniform buffers (GlobalUbo)
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> uniformBuffers;
  GlobalUbo globalUbo{};

  // descriptors (canvas / GlobalUbo)
  std::vector<vk::raii::DescriptorSet> descriptorSets;
  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;

  // pipeline (canvas quad)
  vk::raii::Pipeline pipeline = nullptr;

  // ---- particle SSBO ----
  static constexpr uint32_t kMaxParticles = 1u << 16;  // 65536
  uint32_t numParticles = 0;
  // per-frame device-local SSBOs (written by compute, read as vertex buffer)
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> particleBuffers;

  // particle debug renderer (point list)
  vk::raii::Pipeline particlePipeline = nullptr;
  bool showParticles = true;

  // ---- graphics sync semaphores (signalled by graphics, waited by compute)
  // ----
  struct {
    uint32_t queueFamilyIndex = 0;
    std::vector<vk::raii::Semaphore> semaphores;
  } graphics;

  // ---- compute ----
  struct {
    uint32_t queueFamilyIndex = 0;
    vk::raii::Queue queue = nullptr;
    vk::raii::CommandPool cmdPool = nullptr;
    vk::raii::CommandBuffers cmdBuffers = nullptr;
    std::vector<vk::raii::Semaphore> semaphores;

    // per-frame uniform buffers (ComputeUbo)
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> uniformBuffers;
    ComputeUbo ubo{};

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline pipeline = nullptr;
  } compute;

  static constexpr float kCanvasWorld = 4.0f;
};

}  // namespace vge
