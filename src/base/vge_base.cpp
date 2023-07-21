#include "vge_base.hpp"

#include "vgeu_utils.hpp"

// libs
// #include <Vulkan-Hpp/vulkan/vulkan.hpp>
// #include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

#include <vulkan/vulkan_core.h>

// std
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <tuple>
namespace vge {
VgeBase::VgeBase() { std::cout << "Created: Vulkan Example Base" << std::endl; }
VgeBase::~VgeBase() {
  // need to destroy non-RAII object created
  if (globalAllocator != VK_NULL_HANDLE) {
    vmaDestroyAllocator(globalAllocator);
    globalAllocator = VK_NULL_HANDLE;
  }
}

bool VgeBase::initVulkan() {
  // NOTE: shoud be created before instance for getting required extensions;
  vgeuWindow = std::make_unique<vgeu::VgeuWindow>(width, height, title);
  // NOTE: all vk::raii class have no copy assignment operator.
  // -> omit std::move
  context = std::make_unique<vk::raii::Context>();
  instance = vgeu::createInstance(*context, title, title, apiVersion);

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
  // TODO: check flag transient.
  vk::CommandPoolCreateInfo cmdPoolCI(
      vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      queueFamilyIndices.graphics);
  commandPool = vk::raii::CommandPool(device, cmdPoolCI);
  depthFormat = vgeu::pickDepthFormat(physicalDevice, requiresStencil);

  // surface
  VkSurfaceKHR surface_;
  vgeuWindow->createWindowSurface(static_cast<VkInstance>(*instance),
                                  &surface_);
  surface = vk::raii::SurfaceKHR(instance, surface_);

  // create VMA global allocator
  VmaAllocatorCreateInfo allocatorCI{};
  allocatorCI.physicalDevice = VkPhysicalDevice(*physicalDevice);
  allocatorCI.device = VkDevice(*device);
  allocatorCI.instance = VkInstance(*instance);
  allocatorCI.vulkanApiVersion = apiVersion;
  // TOOD: check vma flags
  // for higher version Vulkan API
  VmaVulkanFunctions vulkanFunctions = {};
  vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
  allocatorCI.pVulkanFunctions = &vulkanFunctions;

  VkResult result = vmaCreateAllocator(&allocatorCI, &globalAllocator);
  assert(result == VK_SUCCESS && "VMA allocator create Error");

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
  std::cout << "SwapChain image count: " << swapChainData->images.size()
            << std::endl;

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

  // create command pool
  vk::CommandPoolCreateInfo cmdPoolCI(
      vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      queueFamilyIndices.graphics);
  cmdPool = vk::raii::CommandPool(device, cmdPoolCI);

  // NOTE: using frames in flight ,not the image count

  // create command buffer
  vk::CommandBufferAllocateInfo cmdBufferAI(
      *cmdPool, vk::CommandBufferLevel::ePrimary, MAX_CONCURRENT_FRAMES);
  drawCmdBuffers = vk::raii::CommandBuffers(device, cmdBufferAI);

  // synch primitives

  presentCompleteSemaphores.reserve(MAX_CONCURRENT_FRAMES);
  renderCompleteSemaphores.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
    renderCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
  }

  vk::FenceCreateInfo fenceCI(vk::FenceCreateFlagBits::eSignaled);
  waitFences.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    waitFences.emplace_back(device, fenceCI);
  }

  // UI overlay
}

void VgeBase::renderLoop() {
  std::cout << "Call: render loop" << std::endl;
  destWidth = width;
  destHeight = height;

  vgeu::TransformComponent viewerTransform{};
  viewerTransform.translation = camera.getPosition();
  viewerTransform.rotation = camera.getRotationYXZ();
  vgeu::KeyBoardMovementController cameraController{};

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
    // camera update
    if (cameraController.moveInPlaneXZ(vgeuWindow->getGLFWwindow(), frameTimer,
                                       viewerTransform)) {
      viewUpdated = true;
    }
    camera.setViewYXZ(viewerTransform.translation, viewerTransform.rotation);

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
    tPrevEnd = tEnd;
    // TODO: UI overlay update
  }
  device.waitIdle();
}

void VgeBase::windowResize() {
  // TODO: dest size shoud be handled earier
  vk::Extent2D extent = vgeuWindow->getExtent();
  destWidth = extent.width;
  destWidth = extent.height;

  if (!prepared) {
    return;
  }
  prepared = false;
  resized = true;
  device.waitIdle();

  width = destWidth;
  height = destHeight;
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

  device.waitIdle();

  windowResized();
  viewChanged();
  prepared = true;
}
void VgeBase::windowResized() {}
void VgeBase::viewChanged() {
  // camera aspect ratio update
  if (width > 0.f && height > 0.f) {
    camera.setAspectRatio(width / height);
  }
}
void VgeBase::prepareFrame() {
  vk::Result result;

  std::tie(result, currentImageIndex) =
      swapChainData->swapChain.acquireNextImage(
          std::numeric_limits<uint64_t>::max(),
          *presentCompleteSemaphores[currentFrameIndex]);
  // std::cout << "swapchain acquired image index : " << currentImageIndex
  //           << std::endl;
  if ((result == vk::Result::eErrorOutOfDateKHR) ||
      (result == vk::Result::eSuboptimalKHR)) {
    if (result == vk::Result::eErrorOutOfDateKHR) {
      windowResize();
    }
    return;
  } else {
    assert((result == vk::Result::eSuccess) &&
           "failed to acquire swap chain image!");
  }
}

// submit presentation queue
void VgeBase::submitFrame() {
  vk::PresentInfoKHR presentInfoKHR(
      *renderCompleteSemaphores[currentFrameIndex], *swapChainData->swapChain,
      currentImageIndex);
  vk::Result result = queue.presentKHR(presentInfoKHR);

  currentFrameIndex = (currentFrameIndex + 1) % MAX_CONCURRENT_FRAMES;

  if ((result == vk::Result::eErrorOutOfDateKHR) ||
      (result == vk::Result::eSuboptimalKHR)) {
    windowResize();
    if (result == vk::Result::eErrorOutOfDateKHR) {
      return;
    }
  } else {
    assert((result == vk::Result::eSuccess) &&
           "failed to acquire swap chain image!");
  }
  // queue wait idle
  // queue.waitIdle();
}
void VgeBase::buildCommandBuffers() {}

}  // namespace vge