#include "vge_base.hpp"

#include "vgeu_utils.hpp"

// libs
// #include <Vulkan-Hpp/vulkan/vulkan.hpp>
// #include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

// std
#include <chrono>
#include <iostream>
#include <memory>
namespace vge {
VgeBase::VgeBase() { std::cout << "Created: Vulkan Example Base" << std::endl; }
VgeBase::~VgeBase() {}

bool VgeBase::initVulkan() {
  // NOTE: shoud be created before instance for getting required extensions;
  vgeuWindow = std::make_unique<vgeu::VgeuWindow>(width, height, title);

  context = std::make_unique<vk::raii::Context>();
  instance = vgeu::createInstance(*context, title, title);
  if (vgeu::enableValidationLayers) {
    debugUtilsMessenger = vgeu::setupDebugMessenger(instance);
  }

  // select gpu
  physicalDevice = vk::raii::PhysicalDevices(instance).front();
  getEnabledExtensions();
  std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
      physicalDevice.getQueueFamilyProperties();

  queueFamilyIndices = vgeu::findQueueFamilyIndices(
      queueFamilyProperties,
      vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute);

  device = vgeu::createLogicalDevice(
      physicalDevice, queueFamilyIndices, enabledDeviceExtensions,
      &enabledFeatures, deviceCreatepNextChain, true,
      vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute);

  queue = vk::raii::Queue(device, queueFamilyIndices.graphics, 0);
  vk::CommandPoolCreateInfo cmdPoolCI(
      vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      queueFamilyIndices.graphics);
  commandPool = vk::raii::CommandPool(device, cmdPoolCI);
  depthFormat = vgeu::pickDepthFormat(physicalDevice, requiresStencil);

  semaphores.presentComplete =
      vk::raii::Semaphore(device, vk::SemaphoreCreateInfo());
  semaphores.renderComplete =
      vk::raii::Semaphore(device, vk::SemaphoreCreateInfo());

  // surface
  VkSurfaceKHR surface_;
  vgeuWindow->createWindowSurface(static_cast<VkInstance>(*instance),
                                  &surface_);
  surface = vk::raii::SurfaceKHR(instance, surface_);

  return true;
}

void VgeBase::getEnabledExtensions(){};
void VgeBase::prepare() {
  std::cout << "Call: prepare" << std::endl;
  // NOTE: first graphicsQueue supports present?
  swapChainData = std::make_unique<vgeu::SwapChainData>(
      physicalDevice, device, surface, vgeuWindow->getExtent(),
      vk::ImageUsageFlagBits::eColorAttachment |
          vk::ImageUsageFlagBits::eTransferSrc,
      nullptr, queueFamilyIndices.graphics, queueFamilyIndices.graphics);

  // create command pool
  vk::CommandPoolCreateInfo cmdPoolCI(
      vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      queueFamilyIndices.graphics);
  cmdPool = vk::raii::CommandPool(device, cmdPoolCI);

  // create command buffer
  vk::CommandBufferAllocateInfo cmdBufferAI(
      *cmdPool, vk::CommandBufferLevel::ePrimary,
      static_cast<uint32_t>(swapChainData->images.size()));
  drawCmdBuffers = vk::raii::CommandBuffers(device, cmdBufferAI);

  // synch primitives
  vk::FenceCreateInfo fenceCI(vk::FenceCreateFlagBits::eSignaled);
  waitFences.reserve(drawCmdBuffers.size());
  for (int i = 0; i < drawCmdBuffers.size(); i++) {
    waitFences.emplace_back(device, fenceCI);
  }
  depthStencil = vgeu::ImageData(
      physicalDevice, device, depthFormat, vgeuWindow->getExtent(),
      vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eDepthStencilAttachment,
      vk::ImageLayout::eUndefined, vk::MemoryPropertyFlagBits::eDeviceLocal,
      vk::ImageAspectFlagBits::eDepth);
  renderPass =
      vgeu::createRenderPass(device, swapChainData->colorFormat, depthFormat);

  pipelineCache =
      vk::raii::PipelineCache(device, vk::PipelineCacheCreateInfo());

  // CHECK: vector move assigment operator and validity
  frameBuffers = vgeu::createFramebuffers(
      device, renderPass, swapChainData->imageViews, &depthStencil.imageView,
      vgeuWindow->getExtent());

  // UI overlay
}

void VgeBase::renderLoop() {
  std::cout << "Call: render loop" << std::endl;
  destWidth = width;
  destHeight = height;
  while (!vgeuWindow->shouldClose()) {
    glfwPollEvents();
    auto tStart = std::chrono::high_resolution_clock::now();
    if (viewUpdated) {
      viewUpdated = false;
      viewChanged();
    }
    render();
    frameCounter++;
    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tDiff =
        std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    frameTimer = tDiff / 1000.0f;
    // TODO: camera update

    // Convert to clamped timer value
    if (!paused) {
      timer += timerSpeed * frameTimer;
      if (timer > 1.0) {
        timer -= 1.0f;
      }
    }
    float fpsTimer =
        std::chrono::duration<double, std::milli>(tEnd - lastTimestamp).count();
    if (fpsTimer > 1000.0f) {
      lastFPS = (float)frameCounter * (1000.0f / fpsTimer);
      frameCounter = 0;
      lastTimestamp = tEnd;
    }
    // TODO: UI overlay update
  }
  device.waitIdle();
}

void VgeBase::windowResize() {
  if (!prepared) {
    return;
  }
  prepared = false;
  resized = true;
  device.waitIdle();

  // TODO: dest size shoud be handled earier
  vk::Extent2D extent = vgeuWindow->getExtent();
  destWidth = extent.width;
  destWidth = extent.height;

  // recreate swpchain
  // NOTE: unique_ptr move assignment
  swapChainData = std::make_unique<vgeu::SwapChainData>(
      physicalDevice, device, surface, vgeuWindow->getExtent(),
      vk::ImageUsageFlagBits::eColorAttachment |
          vk::ImageUsageFlagBits::eTransferSrc,
      &(swapChainData->swapChain), queueFamilyIndices.graphics,
      queueFamilyIndices.graphics);
  // recreate framebuffers
  depthStencil = vgeu::ImageData(
      physicalDevice, device, depthFormat, vgeuWindow->getExtent(),
      vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eDepthStencilAttachment,
      vk::ImageLayout::eUndefined, vk::MemoryPropertyFlagBits::eDeviceLocal,
      vk::ImageAspectFlagBits::eDepth);
  frameBuffers = vgeu::createFramebuffers(
      device, renderPass, swapChainData->imageViews, &depthStencil.imageView,
      vgeuWindow->getExtent());

  // TODO: UI overlay resize

  // recreate Command buffers
  vk::CommandBufferAllocateInfo cmdBufferAI(
      *cmdPool, vk::CommandBufferLevel::ePrimary,
      static_cast<uint32_t>(swapChainData->images.size()));
  drawCmdBuffers = vk::raii::CommandBuffers(device, cmdBufferAI);

  // recreate synchobjects
  waitFences.clear();
  vk::FenceCreateInfo fenceCI(vk::FenceCreateFlagBits::eSignaled);
  waitFences.reserve(drawCmdBuffers.size());
  for (int i = 0; i < drawCmdBuffers.size(); i++) {
    waitFences.emplace_back(device, fenceCI);
  }

  device.waitIdle();

  // TODO: camera aspect ratio update
}
void VgeBase::windowResized() {}
void VgeBase::viewChanged() {}

}  // namespace vge