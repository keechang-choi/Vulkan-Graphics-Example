#include "vge_base.hpp"

#include "vgeu_utils.hpp"

// libs
#include <CLI11/CLI11.hpp>
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

// std
#include <cassert>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <tuple>

namespace vge {
VgeBase::VgeBase() {
  std::cout << "FileSystem::CurrentPath: " << std::filesystem::current_path()
            << std::endl;
  std::cout << "Created: Vulkan Example Base" << std::endl;
}
VgeBase::~VgeBase() {
  // need to destroy non-RAII object created
  // NOTE: but order matters. members will be destroyed after this block.
}

void VgeBase::initVulkan() {
  // NOTE: shoud be created before instance for getting required extensions;
  vgeuWindow = std::make_unique<vgeu::VgeuWindow>(width, height, title);
  // NOTE: all vk::raii class have no copy assignment operator.
  // -> omit std::move
  context = std::make_unique<vk::raii::Context>();
  instance = vgeu::createInstance(*context, title, title, settings.validation,
                                  apiVersion);

  if (settings.validation) {
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
  // TODO: check flag transient for performance.
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

  globalAllocator = std::make_unique<vgeu::VgeuAllocator>(
      static_cast<VkDevice>(*device),
      static_cast<VkPhysicalDevice>(*physicalDevice),
      static_cast<VkInstance>(*instance), apiVersion);
}

void VgeBase::getEnabledExtensions(){};
void VgeBase::prepare() {
  std::cout << "Call: prepare" << std::endl;
  // NOTE: first graphicsQueue supports present?
  swapChainData = std::make_unique<vgeu::SwapChainData>(
      physicalDevice, device, surface, vk::Extent2D(width, height),
      vk::ImageUsageFlagBits::eColorAttachment |
          vk::ImageUsageFlagBits::eTransferSrc,
      nullptr, queueFamilyIndices.graphics, queueFamilyIndices.graphics);
  std::cout << "SwapChain image count: " << swapChainData->images.size()
            << std::endl;

  depthStencil = std::make_unique<vgeu::VgeuImage>(
      device, globalAllocator->getAllocator(), depthFormat,
      swapChainData->swapChainExtent, vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eDepthStencilAttachment,
      vk::ImageLayout::eUndefined, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
      vk::ImageAspectFlagBits::eDepth, 1);

  renderPass =
      vgeu::createRenderPass(device, swapChainData->colorFormat, depthFormat);

  pipelineCache =
      vk::raii::PipelineCache(device, vk::PipelineCacheCreateInfo());

  // CHECK: vector move assigment operator and validity
  frameBuffers = vgeu::createFramebuffers(
      device, renderPass, swapChainData->imageViews,
      &depthStencil->getImageView(), swapChainData->swapChainExtent);

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
  if (settings.overlay) {
    uiOverlay = std::make_unique<vgeu::UIOverlay>(
        device, vgeuWindow->getGLFWwindow(), instance, queue, physicalDevice,
        renderPass, pipelineCache, commandPool,
        std::max(2u, MAX_CONCURRENT_FRAMES));
  }
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

    // UI overlay update
    updateUIOverlay();

    // NOTE: submitting cmd should be called after ui render()
    render();
    frameCounter++;
    auto tEnd = std::chrono::high_resolution_clock::now();
    auto tDiff =
        std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    frameTimer = tDiff / 1000.0f;
    paused = vgeuWindow->isPaused();

    if (!paused) {
      // camera update
      if (cameraController.moveInPlaneXZ(vgeuWindow->getGLFWwindow(),
                                         frameTimer, viewerTransform)) {
        viewUpdated = true;
      }

      camera.setViewYXZ(viewerTransform.translation, viewerTransform.rotation);
    }
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
  }
  device.waitIdle();
}

void VgeBase::windowResize() {
  // TODO: handle dest size for other platforms.
  vk::Extent2D extent = vgeuWindow->getExtent();
  destWidth = extent.width;
  destHeight = extent.height;

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
      physicalDevice, device, surface, vk::Extent2D(width, height),
      vk::ImageUsageFlagBits::eColorAttachment |
          vk::ImageUsageFlagBits::eTransferSrc,
      &(swapChainData->swapChain), queueFamilyIndices.graphics,
      queueFamilyIndices.graphics);

  // recreate framebuffers
  depthStencil = std::make_unique<vgeu::VgeuImage>(
      device, globalAllocator->getAllocator(), depthFormat,
      swapChainData->swapChainExtent, vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eDepthStencilAttachment,
      vk::ImageLayout::eUndefined, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
      vk::ImageAspectFlagBits::eDepth, 1);

  frameBuffers = vgeu::createFramebuffers(
      device, renderPass, swapChainData->imageViews,
      &depthStencil->getImageView(), swapChainData->swapChainExtent);

  if (width > 0 && height > 0) {
    if (settings.overlay) {
      uiOverlay->resize(width, height);
    }
  }

  device.waitIdle();

  // camera aspect ratio update
  if (width > 0 && height > 0) {
    camera.setAspectRatio(static_cast<float>(width) /
                          static_cast<float>(height));
  }
  windowResized();
  viewChanged();
  prepared = true;
}
void VgeBase::windowResized() {}
void VgeBase::viewChanged() {}
void VgeBase::prepareFrame() {
  // NOTE: eErrorOutOfDateKHR raise exceptions in vulkan-hpp
  // https://github.com/KhronosGroup/Vulkan-Hpp/issues/599
  vk::Result result;
  try {
    std::tie(result, currentImageIndex) =
        swapChainData->swapChain.acquireNextImage(
            std::numeric_limits<uint64_t>::max(),
            *presentCompleteSemaphores[currentFrameIndex]);
  } catch (const vk::OutOfDateKHRError& e) {
    result = vk::Result::eErrorOutOfDateKHR;
    // NOTE: if fails, no image is acquired according to spec docs.
  }

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
  // NOTE: eErrorOutOfDateKHR raise exceptions in vulkan-hpp
  // https://github.com/KhronosGroup/Vulkan-Hpp/issues/599
  vk::Result result;
  try {
    result = queue.presentKHR(presentInfoKHR);
  } catch (const vk::OutOfDateKHRError& e) {
    result = vk::Result::eErrorOutOfDateKHR;
  }

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

void VgeBase::updateUIOverlay() {
  if (!settings.overlay) return;
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  // demo for test
  ImGui::ShowDemoWindow();

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
  ImGui::SetNextWindowPos(
      ImVec2(10 * uiOverlay->getScale(), 10 * uiOverlay->getScale()));
  ImGui::Begin("Vulkan Example", nullptr,
               ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize |
                   ImGuiWindowFlags_NoMove);
  ImGui::TextUnformatted(title.c_str());
  ImGui::TextUnformatted(physicalDevice.getProperties().deviceName);
  ImGui::Text("%.2f ms/frame (%.1d fps)", (1000.0f / lastFPS), lastFPS);

  ImGui::PushItemWidth(110.0f * uiOverlay->getScale());
  onUpdateUIOverlay();
  ImGui::PopItemWidth();

  ImGui::End();
  ImGui::PopStyleVar();

  ImGui::Render();
  uiOverlay->update();
}

void VgeBase::drawUI(const vk::raii::CommandBuffer& commandBuffer) {
  if (settings.overlay && uiOverlay->isVisible()) {
    uiOverlay->draw(commandBuffer);
  }
}

void VgeBase::onUpdateUIOverlay() {}

std::string VgeBase::getShadersPath() {
  std::filesystem::path p = "../shaders";
  return std::filesystem::absolute(p).string();
}
std::string VgeBase::getAssetsPath() {
  std::filesystem::path p = "../assets";
  return std::filesystem::absolute(p).string();
}

void VgeBase::setupCommandLineParser(CLI::App& app) {
  // dummy flag for empty arg "",
  // (some vscode launch inputs may produce empty string arg)
  bool emptyArg{false};
  app.add_flag("--vs,", emptyArg,
               "Dummy flag to prevent vscode launching with empty string arg");
  app.add_option("-v, --validation", settings.validation,
                 "Enable/Disable Validation Layer")
      ->capture_default_str();
  app.add_option("--width", width, "Window Width")->capture_default_str();
  app.add_option("--height", height, "Window Height")->capture_default_str();
  app.add_option("-f, --frame", MAX_CONCURRENT_FRAMES, "Max frames in-flight")
      ->capture_default_str();
}

}  // namespace vge