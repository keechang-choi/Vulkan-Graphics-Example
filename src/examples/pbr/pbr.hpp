#include "vge_base.hpp"
#include "vgeu_gltf.hpp"
#include "vgeu_texture.hpp"

#include <vector>

namespace vge {

#define MAX_LIGHTS 10

struct Options {
  float moveSpeed = 10.f;
  glm::mat4 cameraView{1.f};
  int32_t debugDisplayTarget = 0;
  int32_t numTargets = 10;
  float farClamp = 50.f;
  int32_t modelNumX = 4;
  int32_t modelNumZ = 4;
  float spacingX = 4.f;
  float spacingZ = 4.f;
  // light animation
  bool animateLights = true;
  float rotationSpeed = 1.0f;
  float orbitRadius = 8.0f;
  float orbitHeight = -3.0f;
  float spriteSize = 0.3f;
  int32_t numLights = 6;
};

struct DynamicUboElt {
  glm::mat4 modelMatrix{1.f};
  glm::vec4 modelColor{0.f};
};

struct UniformDataOffscreen {
  glm::mat4 projection{1.f};
  glm::mat4 view{1.f};
};

struct Light {
  glm::vec4 position;
  glm::vec3 color;
  float radius;
};

// NOTE: alignment. size = 10*32 + 16 + 4*5 = 352 bytes
struct alignas(64) UniformDataComposition {
  Light lights[MAX_LIGHTS];
  glm::vec4 viewPos;
  int debugDisplayTarget{0};
  int numLights;
  float nearPlane;
  float farPlane;
  float farClamp;
};

struct ModelInstance {
  std::shared_ptr<vgeu::glTF::Model> model;
  std::string name;
  bool isBone = false;
  int animationIndex = -1;
  float animationTime = 0.f;
  vgeu::TransformComponent transform;
  ModelInstance(){};
  ModelInstance(const ModelInstance& o) = delete;
  ModelInstance& operator=(const ModelInstance& other) = delete;
  ModelInstance(ModelInstance&& other);
  ModelInstance& operator=(ModelInstance&& other);
};

struct SpritePushConstants {
  float spriteSize = 0.3f;
};

struct SpecializationData {
  uint32_t displayTargetIndex;
};

class VgeExample : public VgeBase {
 public:
  VgeExample();
  ~VgeExample();
  virtual void setupCommandLineParser(CLI::App& app);
  void setOptions(const std::optional<Options>& opts);

  virtual void initVulkan();
  virtual void getEnabledExtensions();
  virtual void getEnabledFeatures();
  virtual void prepare();
  virtual void render();
  virtual void viewChanged();
  virtual void onUpdateUIOverlay();

  void loadAssets();
  void setupDynamicUbo();
  void prepareOffScreenFrameBuffer();
  std::unique_ptr<vgeu::VgeuImage> createAttachment(vk::Format format,
                                                    vk::ImageUsageFlags usage);
  void prepareUniformBuffers();
  void setupDescriptors();
  void preparePipelines();

  void updateUboComposition();
  void updateUboOffScreen();
  void buildCommandBuffers();
  void draw();

  void addModelInstance(ModelInstance&& newInstance);
  const std::vector<size_t>& findInstances(const std::string& name);

  Options opts{};

  std::vector<ModelInstance> modelInstances;
  std::unordered_map<std::string, std::vector<size_t>> instanceMap;

  UniformDataOffscreen uniformDataOffscreen;
  std::vector<DynamicUboElt> dynamicUbo;
  size_t alignedSizeDynamicUboElt = 0;
  UniformDataComposition uniformDataComposition;

  struct UniformBuffers {
    std::unique_ptr<vgeu::VgeuBuffer> dynamic;
    std::unique_ptr<vgeu::VgeuBuffer> offScreen;
    std::unique_ptr<vgeu::VgeuBuffer> composition;
  };
  std::vector<UniformBuffers> uniformBuffers;

  struct {
    vk::raii::Pipeline offScreen = nullptr;
    vk::raii::Pipeline composition = nullptr;
    std::vector<vk::raii::Pipeline> displayTargets;
    vk::raii::Pipeline sprite = nullptr;
  } pipelines;

  vk::raii::PipelineLayout pipelineLayoutOffScreen = nullptr;
  vk::raii::PipelineLayout pipelineLayoutComposition = nullptr;
  vk::raii::PipelineLayout pipelineLayoutSprite = nullptr;

  struct {
    std::vector<vk::raii::DescriptorSet> dynamicUboDescriptorSets;
    std::vector<vk::raii::DescriptorSet> offScreenUboDescriptorSets;
    std::vector<vk::raii::DescriptorSet> composition;
    std::vector<vk::raii::DescriptorSet> sprite;
  } descriptorSets;

  vk::raii::DescriptorSetLayout compositionDescriptorSetLayout = nullptr;
  vk::raii::DescriptorSetLayout offScreenUboDescriptorSetLayout = nullptr;
  vk::raii::DescriptorSetLayout dynamicUboDescriptorSetLayout = nullptr;
  vk::raii::DescriptorSetLayout spriteLightDescriptorSetLayout = nullptr;

  struct FrameBuffer {
    uint32_t width, height;
    std::vector<vk::raii::Framebuffer> frameBuffers;
    std::vector<std::unique_ptr<vgeu::VgeuImage>> position, normal, albedo, arm, emissive;
    std::vector<std::unique_ptr<vgeu::VgeuImage>> depth;
    const size_t numAttachments = 6;
    vk::raii::RenderPass renderPass = nullptr;
    std::vector<bool> isFirstFrame;
  } offScreenFrameBuf;

  vk::raii::Sampler colorSampler = nullptr;

  // Light animation accumulator
  float lightAnimTime = 0.f;
};

}  // namespace vge
