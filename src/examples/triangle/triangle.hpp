#include "vge_base.hpp"

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
};
}  // namespace vge
