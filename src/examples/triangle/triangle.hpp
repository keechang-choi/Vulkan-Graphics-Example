#include "vge_base.hpp"

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
struct Options {};
class VgeExample : public VgeBase {
 public:
  VgeExample();
  ~VgeExample();
  void initVulkan();
  void render();
  void prepare();
  void createUniformBuffers();
  void createVertexBuffer();
  void createIndexBuffer();
  void createDescriptorSetLayout();
  void createDescriptorPool();
  void createDescriptorSets();
  void createPipelines();
  void draw();
  void buildCommandBuffers();
  void viewChanged();
  void setOptions(const std::optional<Options>& opts){};

  // NOTE: movable element;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> uniformBuffers;
  GlobalUbo globalUbo;
  // vk::raii::DescriptorSets?
  std::vector<vk::raii::DescriptorSet> descriptorSets;

  std::unique_ptr<vgeu::VgeuBuffer> vertexBuffer;
  std::unique_ptr<vgeu::VgeuBuffer> indexBuffer;
  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline pipeline = nullptr;

  Options opts{};
};
}  // namespace vge
