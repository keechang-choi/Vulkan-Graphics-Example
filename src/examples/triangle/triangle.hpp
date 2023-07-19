#include "vge_base.hpp"

// std
#include <memory>
namespace vge {
struct Vertex {
  float position[3];
  float color[3];
};

class VgeExample : public VgeBase {
 public:
  VgeExample();
  ~VgeExample();
  void render();
  void prepare();
  std::unique_ptr<vgeu::VgeuBuffer> vertexBuffer;
};
}  // namespace vge
