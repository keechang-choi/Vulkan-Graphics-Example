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
  const uint32_t MAX_CONCURRENT_FRAMES = 2;
  VgeExample();
  ~VgeExample();
  void render();
  void prepare();
  void createUniformBuffers();

  std::vector<vgeu::VgeuBuffer> uniformBuffers;
  std::unique_ptr<vgeu::VgeuBuffer> vertexBuffer;
};
}  // namespace vge
