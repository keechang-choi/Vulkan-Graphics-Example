#include "vge_base.hpp"

//
#include "vgeu_gltf.hpp"

// std
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace vge {

struct GlobalUbo {
  glm::mat4 projection{1.f};
  glm::mat4 view{1.f};
  glm::mat4 inverseView{1.f};
  // NOTE: alignment
  // tailSize, tailIntensity, tailFadeOut
  glm::vec4 tailInfo{0.f};
  glm::vec2 screenDim;
  // point min size, max size
  glm::vec2 pointSize{1.f, 64.f};
};

// NOTE: for current animation implementation,
// each instance need its own uniformBuffers
struct ModelInstance {
  std::shared_ptr<vgeu::glTF::Model> model;
  std::string name;
  bool isBone = false;
  int animationIndex = -1;
  float animationTime = 0.f;
};

struct DynamicUboElt {
  glm::mat4 modelMatrix{1.f};
  glm::vec4 modelColor{0.f};
  uint32_t numVertices;
  uint32_t numIndices;
};

struct Particle {
  glm::vec4 pos;
  glm::vec4 vel;
  glm::vec4 pk[4];
  glm::vec4 vk[4];
  glm::vec4 attractionWeight;
};

struct SpecializationData {
  uint32_t sharedDataSize;
  uint32_t integrator;
  uint32_t integrateStep;
  uint32_t localSizeX;
};

struct Options {
  int32_t numParticles{1024};
  int32_t numAttractors{6};
  std::vector<std::vector<float>> colors = {
      {5.f / 255.f, 12.f / 255.f, 129.f / 255.f, 1.f},
      {202.f / 255.f, 42.f / 255.f, 1.f / 255.f, 1.f},
      {41.f / 255.f, 86.f / 255.f, 143.f / 255.f, 1.f},
      {161.f / 255.f, 40.f / 255.f, 48.f / 255.f, 1.f},
      {1.f / 255.f, 75.f / 255.f, 255.f / 255.f, 1.f},
      {246.f / 255.f, 7.f / 255.f, 9.f / 255.f, 1.f}};
  float coefficientDeltaTime = 0.05f;
  float rotationVelocity = 50.0f;
  float gravity = 0.02f;
  float power = 1.f;
  float soften = 0.001f;
  int32_t tailSize = 300;
  float tailSampleTime = 0.1;
  int32_t integrator = 1;
  float moveSpeed = 10.f;
  float lineWidth = 1.0f;
  int32_t attractionType = 0;
  int32_t bindingModel = 4;
  float pointSize[2] = {1.f, 128.f};
  int32_t desiredSharedDataSize = 256u;
  float animationSpeed = 0.1;
  float tailIntensity = 1.0;
  float tailFadeOut = 2.0;
};

struct AnimatedVertex {
  glm::vec4 pos;
  glm::vec4 normal;
  glm::vec4 tangent;
};
class VgeExample : public VgeBase {
 public:
  VgeExample();
  ~VgeExample();
  virtual void initVulkan();
  virtual void getEnabledFeatures();
  virtual void render();
  virtual void prepare();
  virtual void viewChanged();
  virtual void setupCommandLineParser(CLI::App& app);
  virtual void onUpdateUIOverlay();

  // to separate cmd line init and restart variable
  void setupCommandLineParser(CLI::App& app, Options& opts);
  // copy option values to member variables
  void initFromOptions();

  void loadAssets();
  void createDescriptorPool();

  void createVertexSCI();
  void createStorageBuffers();
  void createUniformBuffers();

  // graphics resources
  void prepareGraphics();
  void createDescriptorSetLayout();
  void createDescriptorSets();
  void createPipelines();

  // compute resources
  void prepareCompute();

  void draw();
  void buildCommandBuffers();
  void buildComputeCommandBuffers();

  void setupDynamicUbo();
  // NOTE: moved to utils
  size_t padUniformBufferSize(size_t originalSize);
  void updateGraphicsUbo();
  void updateComputeUbo();
  void updateDynamicUbo();
  void updateTailData();

  void addModelInstance(const ModelInstance& newInstance);
  const std::vector<size_t>& findInstances(const std::string& name);

  void setOptions(const std::optional<Options>& opts);

  struct VertexInfos {
    vk::PipelineVertexInputStateCreateInfo vertexInputSCI;
    std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
    std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
  } vertexInfos;
  struct {
    // NOTE: movable element;
    uint32_t queueFamilyIndex;
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> globalUniformBuffers;
    GlobalUbo globalUbo;
    std::vector<vk::raii::DescriptorSet> globalUboDescriptorSets;
    vk::raii::DescriptorSetLayout globalUboDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline pipeline = nullptr;
    std::vector<vk::raii::Semaphore> semaphores;
  } graphics;

  struct {
    uint32_t queueFamilyIndex;
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> storageBuffers;
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> uniformBuffers;
    vk::raii::Queue queue = nullptr;
    vk::raii::CommandPool cmdPool = nullptr;
    vk::raii::CommandBuffers cmdBuffers = nullptr;
    std::vector<vk::raii::Semaphore> semaphores;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    std::vector<vk::raii::Pipeline> pipelineCalculate;
    vk::raii::Pipeline pipelineIntegrate = nullptr;

    vk::raii::DescriptorSetLayout skinDescriptorSetLayout = nullptr;
    // each frames in flight, each model
    std::vector<std::vector<vk::raii::DescriptorSet>> skinDescriptorSets;

    // each model, each skins
    std::vector<std::vector<vgeu::glTF::MeshMatricesData>> skinMatricesData;
    // each frames in flight, each model
    std::vector<std::vector<std::unique_ptr<vgeu::VgeuBuffer>>>
        skinMatricesBuffers;

    // each frames in flight, each model
    std::vector<std::vector<std::unique_ptr<vgeu::VgeuBuffer>>>
        animatedVertexBuffers;

    // for compute animation
    vk::raii::Pipeline pipelineModelAnimate = nullptr;

    vk::raii::Pipeline pipelineModelCalculate = nullptr;
    vk::raii::Pipeline pipelineModelIntegrate = nullptr;
    // TOOD: check alignment for std140
    struct computeUbo {
      glm::vec4 clickData;
      float dt;
      uint32_t particleCount;
      float gravity;
      float power;
      float soften;
      float tailTimer;
      uint32_t tailSize;
    } ubo;

  } compute;

  std::vector<ModelInstance> modelInstances;
  std::unordered_map<std::string, std::vector<size_t>> instanceMap;

  std::vector<DynamicUboElt> dynamicUbo;
  size_t alignedSizeDynamicUboElt = 0;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> dynamicUniformBuffers;
  std::vector<vk::raii::DescriptorSet> dynamicUboDescriptorSets;
  vk::raii::DescriptorSetLayout dynamicUboDescriptorSetLayout = nullptr;

  float animationTime = 0.f;
  float animationLastTime = 0.f;
  uint32_t numParticles = 1024u * 4u * 6u;
  const uint32_t kMaxNumParticles = 1024u * 1024u * 4u;
  uint32_t numAttractors = 6u;
  uint32_t integrator = 1u;
  float rotationVelocity = 50.f;
  float gravity = 0.02f;
  float power = 1.0f;
  float soften = 0.001;

  // vertex buffer ->
  // NOTE: to use this data for vertex buffer,
  // change also vertexSCI to be consistent with offsetof()
  struct TailElt {
    // xyz,w=packedColor
    glm::vec4 pos{0.f};
  };
  std::vector<TailElt> tailData;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> tailBuffers;
  std::vector<uint32_t> tailIndices;
  std::unique_ptr<vgeu::VgeuBuffer> tailIndexBuffer;

  vk::raii::Pipeline tailPipeline = nullptr;
  float tailTimer = -1.f;
  size_t tailSize = 30;
  float tailSampleTime = 0.05f;
  VertexInfos tailVertexInfos;

  Options opts{};
  int values_offset = 0;
  float dist = 0.f;
  std::vector<float> values = std::vector<float>(1000, dist);
  float max_dist = -std::numeric_limits<float>::max();
  float min_dist = std::numeric_limits<float>::max();
  float energy = -27e7f;
  float max_energy = -std::numeric_limits<float>::max();
  float min_energy = std::numeric_limits<float>::max();
  std::vector<float> energyValues = std::vector<float>(1000, energy);

  uint32_t attractionType = 0u;
  uint32_t desiredSharedDataSize = 256u;
  uint32_t sharedDataSize;
};
}  // namespace vge
