#include "vge_base.hpp"

//
#include "vgeu_gltf.hpp"

// std
#include <memory>
#include <unordered_map>
namespace vge {
struct Vertex {
  float position[3];
  float color[3];
};

struct GlobalUbo {
  glm::mat4 projection{1.f};
  glm::mat4 model{1.f};
  glm::mat4 view{1.f};
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
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> uniformBuffers;
  GlobalUbo globalUbo;
  // vk::raii::DescriptorSets?
  std::vector<vk::raii::DescriptorSet> descriptorSets;

  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline pipeline = nullptr;
};
}  // namespace vge
