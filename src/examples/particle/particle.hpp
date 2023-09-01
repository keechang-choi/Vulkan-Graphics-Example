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
  glm::vec4 lightPos{0.f};
  glm::mat4 inverseView{1.f};
  glm::vec2 screenDim;
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
};

struct Particle {
  glm::vec4 pos;
  glm::vec4 vel;
  glm::vec4 pk[4];
  glm::vec4 vk[4];
};

struct SpecializationData {
  uint32_t sharedDataSize;
  float gravity;
  float power;
  float soften;
  uint32_t rkStep;
};

class VgeExample : public VgeBase {
 public:
  VgeExample();
  ~VgeExample();
  virtual void initVulkan();
  virtual void getEnabledExtensions();
  virtual void render();
  virtual void prepare();
  virtual void viewChanged();
  virtual void setupCommandLineParser(CLI::App& app);

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
  size_t padUniformBufferSize(size_t originalSize);
  void updateGraphicsUbo();
  void updateComputeUbo();
  void updateDynamicUbo();
  void updateTailSSBO();

  void addModelInstance(const ModelInstance& newInstance);
  const std::vector<size_t>& findInstances(const std::string& name);

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
    struct computeUbo {
      float dt;
      uint32_t particleCount;
    } ubo;

  } compute;

  std::vector<ModelInstance> modelInstances;
  std::unordered_map<std::string, std::vector<size_t>> instanceMap;

  // TODO: move those into compute resource
  std::vector<DynamicUboElt> dynamicUbo;
  size_t alignedSizeDynamicUboElt = 0;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> dynamicUniformBuffers;
  std::vector<vk::raii::DescriptorSet> dynamicUboDescriptorSets;
  vk::raii::DescriptorSetLayout dynamicUboDescriptorSetLayout = nullptr;

  float animationTime = 0.f;
  float animationLastTime = 0.f;
  uint32_t numParticles = 1024u * 4u * 6u;
  uint32_t numAttractors = 6u;
  uint32_t integrateStep = 1;
  float rotationVelocity = 50.f;
  float gravity = 0.02f;
  float power = 1.0f;
  float soften = 0.1;

  // vertex buffer ->
  // TODO: to use this data for vertex buffer,
  // change also vertexSCI to be consistent with offsetof()
  struct TailElt {
    // xyz color
    glm::vec4 pos{0.f};
    // color
    // glm::vec4 vel{0.f};
  };
  // numParticles * tailSize * 4(xyz, color)
  std::vector<std::list<glm::vec4>> tails;
  std::vector<TailElt> tailsData;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> tailBuffers;
  std::unique_ptr<vgeu::VgeuBuffer> tailIndexBuffer;

  vk::raii::Pipeline tailPipeline = nullptr;
  float tailTimer = -1.f;
  size_t tailSize = 300;
  float tailSampleTime = 0.1f;
  VertexInfos tailVertexInfos;
};
}  // namespace vge
