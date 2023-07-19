#include "triangle.hpp"

namespace vge {
VgeExample::VgeExample() : VgeBase() {
  title = "First Triangle Example";
  // camera setup
}
VgeExample::~VgeExample() {}
void VgeExample::render() {}
void VgeExample::prepare() {
  VgeBase::prepare();
  std::vector<Vertex> vertices{
      {{1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
      {{-1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
      {{0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
  };

  vertexBuffer = std::make_unique<vgeu::VgeuBuffer>(
      globalAllocator, sizeof(Vertex), static_cast<uint32_t>(vertices.size()),
      vk::BufferUsageFlagBits::eVertexBuffer |
          vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);
}
}  // namespace vge

VULKAN_EXAMPLE_MAIN()