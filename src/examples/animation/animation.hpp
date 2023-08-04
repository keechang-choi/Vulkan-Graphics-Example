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

struct DynamicUboElt {
  glm::mat4 model{1.f};
  glm::mat4 joints[63];
};

struct ModelInstance {
  std::shared_ptr<vgeu::glTF::Model> model;
  glm::mat4 modelMatrix{1.f};
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
  void buildCommandBuffers();
  void viewChanged();

  std::unique_ptr<vgeu::glTF::Model> scene;

  // NOTE: movable element;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> globalUniformBuffers;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> dynamicUniformBuffers;

  struct {
    GlobalUbo globalUbo;
    std::vector<DynamicUboElt> dynamicUbo;
  } Ubo;

  // vk::raii::DescriptorSets?
  std::vector<vk::raii::DescriptorSet> descriptorSets;

  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;

  struct {
    vk::raii::Pipeline phong = nullptr;
    vk::raii::Pipeline toon = nullptr;
    vk::raii::Pipeline wireframe = nullptr;
  } pipelines;
};
}  // namespace vge
