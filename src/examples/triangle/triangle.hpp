#include "vge_base.hpp"

// std
#include <memory>
namespace vge {
struct Vertex {
  float position[3];
  float color[3];
};

struct Ubo {
  glm::mat4 projection;
  glm::mat4 model;
  glm::mat4 view;
};
class VgeExample : public VgeBase {
 public:
  VgeExample();
  ~VgeExample();
  void render();
  void prepare();
  void createUniformBuffers();
  void createVertexBuffer();
  void createIndexBuffer();
  void createDescriptorSetLayout();
  void createDescriptorPool();
  void createDescriptorSets();
  void createPipelines();

  // NOTE: movable element;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> uniformBuffers;
  std::unique_ptr<vgeu::VgeuBuffer> vertexBuffer;
  std::unique_ptr<vgeu::VgeuBuffer> indexBuffer;
  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline pipeline = nullptr;
  vk::raii::DescriptorSet descriptorSet = nullptr;
};
}  // namespace vge
