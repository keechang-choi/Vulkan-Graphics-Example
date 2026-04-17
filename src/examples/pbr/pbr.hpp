#include <vector>

#include "vge_base.hpp"
#include "vgeu_gltf.hpp"
#include "vgeu_texture.hpp"

namespace vge {

#define MAX_LIGHTS 10

constexpr int kMaxGridSize = 8;  // max metallic/roughness grid dimension

struct Options {
  float moveSpeed = 10.f;
  glm::mat4 cameraView{1.f};
  int32_t debugDisplayTarget = 0;
  int32_t numTargets = 12;
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
  float lightIntensity = 1.0f;
  bool showDebugViews = true;
  bool useSpheres = false;
  bool helmetPbrOverride =
      true;  // true = grid metallic/roughness, false = model texture
  bool useMaterial = false;
  float heightScale = 0.03f;
  std::array<float, 4> sphereAlbedo = {1.0f, 1.0f, 1.0f, 1.0f};
  bool useDirectionalLight = false;
  std::array<float, 3> dirLightDir = {
      0.f, -1.f, -1.f};  // world-space direction toward light
  float ambientStrength = 0.03f;
  bool useIBL = false;
  float iblExposure = 4.5f;
  float iblGamma = 2.2f;
  float skyboxLod = 0.0f;
  // per-texel phi jitter in IBL filtering; toggled via ImGui (re-bakes IBL)
  bool useJitter = true;
};

struct DynamicUboElt {
  glm::mat4 modelMatrix{1.f};
  glm::vec4 modelColor{0.f};
  glm::vec4 pbrOverride{
      0.f};  // x=metallic, y=roughness, z=useOverride(0/1), w=unused
};

struct UniformDataOffscreen {
  glm::mat4 projection{1.f};
  glm::mat4 view{1.f};
  glm::vec4 viewPos{0.f};
  float heightScale{0.05f};
};

struct Light {
  glm::vec4 position;
  glm::vec3 color;
  float radius;
};

// NOTE: alignment. size = 10*32 + 16 + 4*5 + 16 + 4*2 = 384 bytes
struct alignas(64) UniformDataComposition {
  Light lights[MAX_LIGHTS];
  glm::vec4 viewPos;
  int debugDisplayTarget{0};
  int numLights;
  float nearPlane;
  float farPlane;
  float farClamp;
  int useDirectionalLight{0};
  float ambientStrength{0.03f};
  glm::vec2 _pad;
  alignas(16) glm::vec4
      dirLightDir;  // xyz = direction toward light (normalized), w = unused
  glm::vec3 dirLightColor;  // pre-multiplied with intensity
  float _pad2;
  int useIBL{0};
  int _pad3[3];
};

struct ModelInstance {
  std::shared_ptr<vgeu::glTF::Model> model;
  std::string name;
  bool isBone = false;
  int animationIndex = -1;
  float animationTime = 0.f;
  vgeu::TransformComponent transform;
  enum class SceneMode { kModelOnly, kSphereOnly, kSphereWithMaterial };
  SceneMode sceneMode = SceneMode::kModelOnly;
  int gridI =
      -1;  // row index in metallic/roughness grid (-1 = not a grid instance)
  int gridJ = -1;  // col index in metallic/roughness grid
  ModelInstance() {};
  ModelInstance(const ModelInstance& o) = delete;
  ModelInstance& operator=(const ModelInstance& other) = delete;
  ModelInstance(ModelInstance&& other);
  ModelInstance& operator=(ModelInstance&& other);
};

struct CapturePushConstants {
  glm::mat4 mvp;
};

struct SkyboxPushConstants {
  glm::mat4 view;        // translation 제거한 view
  glm::mat4 projection;  // 128바이트
  float lod = 0.0f;      // env cubemap LOD (0 = full res, 9 = most blurred)
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

  std::unique_ptr<vgeu::VgeuImage> createDummyTexture(
      std::array<uint8_t, 4> rgba);
  void updateDynamicUbo();

  void addModelInstance(ModelInstance&& newInstance);
  const std::vector<size_t>& findInstances(const std::string& name);

  Options opts{};
  int prevModelNumX = -1;
  int prevModelNumZ = -1;
  int gridRebuildCountdown = 0;

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
    std::vector<std::unique_ptr<vgeu::VgeuImage>> position, normal, albedo, arm,
        emissive, heightAttach;
    std::vector<std::unique_ptr<vgeu::VgeuImage>> depth;
    const size_t numAttachments = 7;
    vk::raii::RenderPass renderPass = nullptr;
    std::vector<bool> isFirstFrame;
  } offScreenFrameBuf;

  vk::raii::Sampler colorSampler = nullptr;

  // Dummy textures for sphere pass (1x1 pixels)
  std::unique_ptr<vgeu::VgeuImage> sphereDummyAlbedo;
  std::unique_ptr<vgeu::VgeuImage> sphereDummyNormal;
  std::unique_ptr<vgeu::VgeuImage> sphereDummyMetRough;
  std::unique_ptr<vgeu::VgeuImage> sphereDummyEmissive;
  vk::raii::DescriptorSetLayout sphereImageSetLayout = nullptr;
  vk::raii::DescriptorSet sphereDummyDescriptorSet = nullptr;

  // Height map (set 4): pirate-gold uses real texture, others use white dummy
  // (no parallax)
  std::unique_ptr<vgeu::Texture2D> pirateGoldHeightTexture;
  std::unique_ptr<vgeu::VgeuImage> sphereDummyHeight;
  vk::raii::DescriptorSetLayout heightMapDescriptorSetLayout = nullptr;
  vk::raii::DescriptorSet pirateGoldHeightDescriptorSet = nullptr;
  vk::raii::DescriptorSet sphereDummyHeightDescriptorSet = nullptr;

  // Light animation accumulator
  float lightAnimTime = 0.f;

  // IBL 텍스처 (startup 1회 생성)
  std::unique_ptr<vgeu::VgeuImage> hdrTexture;
  std::unique_ptr<vgeu::VgeuImage> envCubemap;
  std::unique_ptr<vgeu::VgeuImage> irradianceMap;
  std::unique_ptr<vgeu::VgeuImage> prefilteredMap;
  std::unique_ptr<vgeu::VgeuImage> brdfLut;
  vk::raii::Sampler iblSampler = nullptr;
  vk::raii::Sampler hdrSampler = nullptr;
  glm::mat4 captureProj;
  std::vector<glm::mat4> captureViews;

  // IBL descriptor (composition set=1)
  vk::raii::DescriptorSetLayout iblDescriptorSetLayout = nullptr;
  std::vector<vk::raii::DescriptorSet> iblDescriptorSets;

  // Skybox
  vk::raii::Pipeline skyboxPipeline = nullptr;
  vk::raii::PipelineLayout skyboxPipelineLayout = nullptr;
  vk::raii::DescriptorSetLayout skyboxDescriptorSetLayout = nullptr;
  std::vector<vk::raii::DescriptorSet> skyboxDescriptorSets;
  std::unique_ptr<vgeu::VgeuBuffer> skyboxUboBuffer;  // per-frame, view+proj

  // IBL 생성 함수
  void prepareIBL();
  void loadHdrTexture();
  void buildEnvCubemap();
  void buildIrradianceMap();
  void buildPrefilteredMap();
  // Waits idle, re-runs buildIrradianceMap + buildPrefilteredMap (both pick up
  // opts.useJitter), and rewrites IBL descriptor sets.
  void rebuildIBLFiltering();
  void buildBrdfLut();
  void prepareSkyboxPipeline();
  void prepareSkyboxDescriptors();
};

}  // namespace vge
