#include "vge_base.hpp"

//
#include "vgeu_gltf.hpp"

// std
#include <memory>
#include <unordered_map>
namespace vge {

struct GlobalUbo {
  glm::mat4 projection{1.f};
  glm::mat4 model{1.f};
  glm::mat4 view{1.f};
  glm::vec4 lightPos{0.f};
  glm::mat4 normalMatrix{1.f};
  glm::mat4 inverseView{1.f};
};
struct Options {};

class VgeExample : public VgeBase {
 public:
  VgeExample();
  ~VgeExample();
  void initVulkan();
  void getEnabledFeatures();
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
  void setOptions(const std::optional<Options>& opts){};

  std::unique_ptr<vgeu::glTF::Model> scene;

  // NOTE: movable element;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> uniformBuffers;
  GlobalUbo globalUbo;
  // vk::raii::DescriptorSets?
  std::vector<vk::raii::DescriptorSet> descriptorSets;

  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;

  struct {
    vk::raii::Pipeline phong = nullptr;
    vk::raii::Pipeline toon = nullptr;
    vk::raii::Pipeline wireframe = nullptr;
  } pipelines;
  Options opts{};
};
}  // namespace vge
