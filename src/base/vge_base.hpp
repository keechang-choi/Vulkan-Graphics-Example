#pragma once

// libs
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

#include "vgeu_window.hpp"
namespace vge {
class VgeBase {
 public:
  static constexpr uint32_t WIDTH = 1280;
  static constexpr uint32_t HEIGHT = 720;

  VgeBase();
  ~VgeBase();

  // TODO:
  // initVulkan
  // virtual prepare
  // renderLoop

  std::string title = "Vulkan Example KC";
  std::string name = "vulkanExample";

 private:
  vgeu::VgeuWindow vgeuWindow{WIDTH, HEIGHT, title};
};
}  // namespace vge

#define VULKAN_EXAMPLE_MAIN()       \
  int main(int argc, char** argv) { \
    vge::VgeExample vgeExample{};   \
    return 0;                       \
  }
