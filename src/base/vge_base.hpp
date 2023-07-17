#pragma once

// libs
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

#include "vgeu_utils.hpp"
#include "vgeu_window.hpp"
namespace vge {
class VgeBase {
 public:
  static constexpr uint32_t WIDTH = 1280;
  static constexpr uint32_t HEIGHT = 720;

  VgeBase();
  ~VgeBase();

  // TODO:

  // initVulkan vk instance
  // device, queue, sema
  bool initVulkan();
  virtual void getEnabledExtensions();

  // virtual prepare vk resources
  // swapchain, commandPool, commandBuffers, Synch Primitives
  // DepthStencil, RenderPass, PipelineCache, FrameBuffer
  // UI overlay
  virtual void prepare();
  // renderLoop

  std::string title = "Vulkan Example KC";
  std::string name = "vulkanExample";

 private:
  std::unique_ptr<vk::raii::Context> context;
  vk::raii::Instance instance = nullptr;
  vk::raii::DebugUtilsMessengerEXT debugUtilsMessenger = nullptr;
  vk::raii::PhysicalDevice physicalDevice = nullptr;
  vk::PhysicalDeviceFeatures enabledFeatures{};
  std::vector<const char*> enabledDeviceExtensions;
  vk::raii::Device device = nullptr;
  vgeu::QueueFamilyIndices queueFamilyIndices;
  void* deviceCreatepNextChain = nullptr;
  vk::raii::Queue queue = nullptr;
  vk::raii::CommandPool commandPool = nullptr;
  bool requiresStencil{false};
  vk::Format depthFormat;
  struct {
    vk::raii::Semaphore presentComplete = nullptr;
    vk::raii::Semaphore renderComplete = nullptr;
  } semaphores;

  vk::raii::SurfaceKHR surface = nullptr;
  std::unique_ptr<vgeu::VgeuWindow> vgeuWindow;

  // prepare
  std::unique_ptr<vgeu::SwapChainData> swapChainData;
  vk::raii::CommandPool cmdPool = nullptr;
  vk::raii::CommandBuffers drawCmdBuffers = nullptr;
  // TOOD: swapchain related resources need to be recreated.
  // when window resize.
  std::vector<vk::raii::Fence> waitFences;
  vgeu::ImageData depthStencil = nullptr;
};
}  // namespace vge

#define VULKAN_EXAMPLE_MAIN()       \
  int main(int argc, char** argv) { \
    vge::VgeExample vgeExample{};   \
    vgeExample.initVulkan();        \
    vgeExample.prepare();           \
    return 0;                       \
  }
