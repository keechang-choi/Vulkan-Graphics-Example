#include "vge_base.hpp"

//
#include "vgeu_gltf.hpp"

// std
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
};

class VgeExample : public VgeBase {
 public:
  VgeExample();
  ~VgeExample();
  void initVulkan();
  void getEnabledExtensions();
  void render();
  void prepare();
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

  void viewChanged();
  void setupDynamicUbo();
  size_t padUniformBufferSize(size_t originalSize);
  void updateGraphicsUbo();
  void updateComputeUbo();
  void updateDynamicUbo();

  void addModelInstance(const ModelInstance& newInstance);
  const std::vector<size_t>& findInstances(const std::string& name);

  struct {
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
    vk::raii::Semaphore semaphore = nullptr;
  } graphics;

  struct {
    uint32_t queueFamilyIndex;
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> storageBuffers;
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> uniformBuffers;
    vk::raii::Queue queue = nullptr;
    vk::raii::CommandPool cmdPool = nullptr;
    vk::raii::CommandBuffers cmdBuffers = nullptr;
    vk::raii::Semaphore semaphore = nullptr;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline pipelineCalculate = nullptr;
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
  uint32_t numParticles = 1024u * 6u;
};
}  // namespace vge
