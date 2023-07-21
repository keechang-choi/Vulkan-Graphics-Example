#pragma once

#include "vgeu_buffer.hpp"
#include "vgeu_camera.hpp"
#include "vgeu_keyboard_movement_controller.hpp"
#include "vgeu_utils.hpp"
#include "vgeu_window.hpp"

// libs
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

// std
#include <chrono>
#include <stdexcept>

namespace vge {
class VgeBase {
 public:
  const uint32_t MAX_CONCURRENT_FRAMES = 2;

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
  // msg handle if necessary
  // resize and frameTime
  // view update
  // render, camera update, UI update,
  void renderLoop();

  virtual void windowResized();
  virtual void viewChanged();
  virtual void render() = 0;
  virtual void buildCommandBuffers();
  std::string getShadersPath() { return "../shaders"; }
  void prepareFrame();
  void submitFrame();

  uint32_t width = 1280;
  uint32_t height = 1080;
  std::string title = "Vulkan Example KC";
  std::string name = "vulkanExample";
  uint32_t apiVersion = VK_API_VERSION_1_3;
  float frameTimer = 1.0f;
  bool prepared = false;
  bool resized = false;
  bool viewUpdated = false;
  float timer = 0.0f;
  float timerSpeed = 0.25f;
  bool paused = false;
  vgeu::VgeuCamera camera;

 protected:
  VmaAllocator globalAllocator;
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

  // synchronization objects

  std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
  std::vector<vk::raii::Semaphore> renderCompleteSemaphores;
  std::vector<vk::raii::Fence> waitFences;

  vk::raii::SurfaceKHR surface = nullptr;
  std::unique_ptr<vgeu::VgeuWindow> vgeuWindow;

  // prepare
  std::unique_ptr<vgeu::SwapChainData> swapChainData;
  // NOTE: order matters cmd pool and cmd buffers
  vk::raii::CommandPool cmdPool = nullptr;
  // vector
  vk::raii::CommandBuffers drawCmdBuffers = nullptr;
  vgeu::ImageData depthStencil = nullptr;
  vk::raii::RenderPass renderPass = nullptr;
  vk::raii::PipelineCache pipelineCache = nullptr;
  std::vector<vk::raii::Framebuffer> frameBuffers;
  vk::raii::DescriptorPool descriptorPool = nullptr;

  uint32_t frameCounter = 0;
  uint32_t lastFPS = 0;
  std::chrono::time_point<std::chrono::high_resolution_clock> lastTimestamp,
      tPrevEnd;

  uint32_t currentFrameIndex = 0;
  uint32_t currentImageIndex;

 private:
  uint32_t destWidth;
  uint32_t destHeight;
  void windowResize();
};
}  // namespace vge

#define VULKAN_EXAMPLE_MAIN()             \
  int main(int argc, char** argv) {       \
    try {                                 \
      vge::VgeExample vgeExample{};       \
      vgeExample.initVulkan();            \
      vgeExample.prepare();               \
      vgeExample.renderLoop();            \
    } catch (const std::exception& e) {   \
      std::cerr << e.what() << std::endl; \
      return EXIT_FAILURE;                \
    }                                     \
    return 0;                             \
  }
