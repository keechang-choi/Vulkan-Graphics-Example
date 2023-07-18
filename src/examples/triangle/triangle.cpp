#include "vge_base.hpp"

namespace vge {
class VgeExample : public VgeBase {
 public:
  VgeExample() : VgeBase() {}
  ~VgeExample() {}
  void render() {}
};
}  // namespace vge

VULKAN_EXAMPLE_MAIN()