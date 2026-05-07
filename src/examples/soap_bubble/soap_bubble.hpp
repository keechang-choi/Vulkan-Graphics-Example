#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "vge_base.hpp"
#include "vgeu_gltf.hpp"
#include "vgeu_ibl.hpp"
#include "vgeu_texture.hpp"

namespace vge {

struct BgInstance {
  std::string path;  // assetsPath-relative
  glm::vec3 translate;
  glm::vec3 eulerDeg;   // degrees
  float scale;          // uniform
  glm::vec3 baseColor;  // linear RGB
  bool enabled;
};

struct BgPushConstant {
  glm::mat4 model;
  glm::vec4 baseColor;
};
static_assert(sizeof(BgPushConstant) == 80,
              "BgPushConstant size must be 80 bytes");

struct Options {
  // Thin Film
  float thicknessMin = 200.f;  // nm
  float thicknessMax = 800.f;
  float n1 = 1.0f;
  float n2 = 1.33f;
  float n3 = 1.0f;
  int32_t spectralSamples = 16;
  // Thickness Source
  int32_t thicknessMode = 0;  // 0=Texture, 1=Procedural
  float gravityStrength = 1.0f;
  float noiseScale = 2.0f;
  // Animation
  bool useAnimation = false;
  float driftSpeed = 0.2f;
  // Surface
  float roughness = 0.0f;
  // R/T Debug
  int32_t rtMode = 0;  // 0=both, 1=R-only, 2=T-only
  // IBL / Env
  float iblExposure = 4.5f;
  float iblGamma = 2.2f;
  bool useJitter = true;
  float skyboxLod = 0.0f;
  // Debug
  bool showThicknessHeatmap = false;
  bool showFresnelOnly = false;
  bool showNormal = false;
  // Background scene (hardcoded N-extensible; per-instance enable togglable)
  std::vector<BgInstance> backgrounds = {
      {"/models/apple/food_apple_01_4k.gltf", glm::vec3(-1.5f, 0.3f, 1.5f),
       glm::vec3(0.f, 25.f, 0.f), 1.5f, glm::vec3(0.85f, 0.18f, 0.18f), true},
      {"/models/fox/Fox.gltf", glm::vec3(1.6f, -0.2f, 1.2f),
       glm::vec3(0.f, -20.f, 0.f), 0.015f, glm::vec3(0.95f, 0.62f, 0.20f),
       true},
      {"/models/sphere/smooth_sphere.gltf", glm::vec3(0.0f, -1.5f, 2.0f),
       glm::vec3(0.f, 0.f, 0.f), 0.6f, glm::vec3(0.30f, 0.55f, 0.85f), true},
      {"/models/dutch_ship_medium_1k/dutch_ship_medium_1k.gltf",
       glm::vec3(0.0f, 1.4f, 2.5f), glm::vec3(0.f, 180.f, 15.f), 0.5f,
       glm::vec3(0.55f, 0.42f, 0.30f), true},
  };
  // Model: "helmet" (DamagedHelmet) or "sphere"
  std::string model = "helmet";
};

struct GlobalsUbo {
  glm::mat4 view{1.f};
  glm::mat4 projection{1.f};
  glm::mat4 model{1.f};
  glm::vec4 viewPos{0.f};
};

struct BubbleParamsUbo {
  float thicknessMin;
  float thicknessMax;
  float n1;
  float n2;
  // -- 16 byte boundary --
  float n3;
  int32_t spectralSamples;
  int32_t thicknessMode;
  float gravityStrength;
  // -- 16 --
  float noiseScale;
  int32_t useAnimation;
  float driftSpeed;
  float roughness;
  // -- 16 --
  float iblExposure;
  float iblGamma;
  float time;
  int32_t rtMode;
  // -- 16 --
  int32_t showThicknessHeatmap;
  int32_t showFresnelOnly;
  int32_t showNormal;
  int32_t _pad0;
};

class VgeExample : public VgeBase {
public:
  VgeExample();
  ~VgeExample();
  void setupCommandLineParser(CLI::App& app) override;
  void setOptions(const std::optional<Options>& opts);

  void initVulkan() override;
  void getEnabledExtensions() override;
  void getEnabledFeatures() override;
  void prepare() override;
  void render() override;
  void viewChanged() override;
  void onUpdateUIOverlay() override;

  void loadAssets();
  void prepareIBL();
  void prepareUniformBuffers();
  void setupDescriptors();
  void preparePipelines();
  void buildCommandBuffers() override;
  void draw();

  void updateGlobalsUbo();
  void updateBubbleParamsUbo();

  Options opts{};

  // IBL
  std::unique_ptr<vgeu::IBLBaker> iblBaker;
  std::unique_ptr<vgeu::Skybox> skybox;
  vgeu::IBLBakeConfig iblConfig;

  // Bubble model
  std::shared_ptr<vgeu::glTF::Model> bubbleModel;

  // Background scene
  std::vector<std::shared_ptr<vgeu::glTF::Model>> bgModels;

  // Height texture (separate Texture2D, used for thickness modulation)
  std::unique_ptr<vgeu::Texture2D> heightTexture;

  // Uniform buffers (per-frame)
  struct UniformBuffers {
    std::unique_ptr<vgeu::VgeuBuffer> globals;
    std::unique_ptr<vgeu::VgeuBuffer> bubbleParams;
  };
  std::vector<UniformBuffers> uniformBuffers;

  GlobalsUbo globalsUbo;
  BubbleParamsUbo bubbleParamsUbo;

  // Pipeline
  vk::raii::DescriptorSetLayout globalsSetLayout = nullptr;
  vk::raii::DescriptorSetLayout bubbleParamsSetLayout = nullptr;
  vk::raii::DescriptorSetLayout heightTexSetLayout = nullptr;
  vk::raii::DescriptorSetLayout envSetLayout = nullptr;
  vk::raii::PipelineLayout bubblePipelineLayout = nullptr;
  vk::raii::Pipeline bubblePipeline = nullptr;

  std::vector<vk::raii::DescriptorSet> globalsDescSets;
  std::vector<vk::raii::DescriptorSet> bubbleParamsDescSets;
  vk::raii::DescriptorSet heightTexDescSet = nullptr;
  std::vector<vk::raii::DescriptorSet> envDescSets;

  // Background pipeline / descriptor handles
  vk::raii::DescriptorSetLayout bgIrradianceSetLayout = nullptr;
  vk::raii::PipelineLayout bgPipelineLayout = nullptr;
  vk::raii::Pipeline bgPipeline = nullptr;
  std::vector<vk::raii::DescriptorSet> bgIrradianceDescSets;
};

}  // namespace vge
