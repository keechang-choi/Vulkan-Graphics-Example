#include "vge_base.hpp"

//
#include "vgeu_gltf.hpp"

// std
#include <memory>
#include <unordered_map>
namespace vge {

struct GlobalUbo {
  glm::mat4 projection{1.f};
  glm::mat4 view{1.f};
  glm::vec4 lightPos{0.f};
  glm::mat4 inverseView{1.f};
};

struct ModelInstance {
  std::shared_ptr<vgeu::glTF::Model> model;
  uint32_t id = 0u;
  bool isBone = false;
};

struct DynamicUboElt {
  glm::mat4 modelMatrix{1.f};
  glm::vec4 modelColor{0.f};
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
  void createUniformBuffers();
  void createDescriptorSetLayout();
  void createDescriptorPool();
  void createDescriptorSets();
  void createPipelines();
  void draw();
  void drawSkeleton();
  void buildCommandBuffers();
  void viewChanged();
  void setupDynamicUbo();
  size_t padUniformBufferSize(size_t originalSize);

  // NOTE: movable element;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> globalUniformBuffers;
  GlobalUbo globalUbo;
  // vk::raii::DescriptorSets?
  std::vector<vk::raii::DescriptorSet> globalUboDescriptorSets;

  std::vector<ModelInstance> modelInstances;

  std::vector<DynamicUboElt> dynamicUbo;
  size_t alignedSizeDynamicUboElt = 0;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> dynamicUniformBuffers;
  std::vector<vk::raii::DescriptorSet> dynamicUboDescriptorSets;

  vk::raii::DescriptorSetLayout globalUboDescriptorSetLayout = nullptr;
  vk::raii::DescriptorSetLayout dynamicUboDescriptorSetLayout = nullptr;

  vk::raii::PipelineLayout pipelineLayout = nullptr;

  struct {
    vk::raii::Pipeline phong = nullptr;
    vk::raii::Pipeline wireframe = nullptr;
  } pipelines;
};
}  // namespace vge
