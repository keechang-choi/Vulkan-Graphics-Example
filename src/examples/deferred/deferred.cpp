#include "deferred.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtx/matrix_query.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

// std
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <unordered_set>

namespace vge {
VgeExample::VgeExample() : VgeBase() { title = "Cloth Example"; }
VgeExample::~VgeExample() {}
void VgeExample::setupCommandLineParser(CLI::App& app) {}
void VgeExample::setOptions(const std::optional<Options>& opts) {
  if (opts.has_value()) {
    this->opts = opts.value();
    // overwrite cli args for restart run
    cameraController.moveSpeed = this->opts.moveSpeed;
  } else {
    // save cli args for initial run
  }
}

void VgeExample::initVulkan() {
  cameraController.moveSpeed = opts.moveSpeed;
  // camera setup
  if (glm::isIdentity(opts.cameraView, 1e-6f)) {
    camera.setViewTarget(glm::vec3{0.f, -10.f, -20.f},
                         glm::vec3{0.f, 0.f, 0.f});
  } else {
    camera.setViewMatrix(opts.cameraView);
  }
  camera.setPerspectiveProjection(
      glm::radians(60.f),
      static_cast<float>(width) / (static_cast<float>(height)), 0.1f, 256.f);
  // NOTE: coordinate space in world

  VgeBase::initVulkan();
}

void VgeExample::getEnabledExtensions() {}
void VgeExample::getEnabledFeatures() {
  enabledFeatures.samplerAnisotropy =
      physicalDevice.getFeatures().samplerAnisotropy;
  enabledFeatures.fillModeNonSolid =
      physicalDevice.getFeatures().fillModeNonSolid;
}
void VgeExample::prepare() {
  VgeBase::prepare();
  loadAssets();
  prepareOffScreenFrameBuffer();
  prepareUniformBuffers();
  setupDescriptors();
  preparePipelines();
  prepared = true;
}

void VgeExample::render() {
  if (!prepared) {
    return;
  }
  // update ubo

  buildCommandBuffers();
  buildDefferredCommandBuffers();

  // draw
}

void VgeExample::viewChanged() {}
void VgeExample::onUpdateUIOverlay() {}

void VgeExample::loadAssets() {
  // NOTE: no flip or preTransform for animation and skinning
  vgeu::FileLoadingFlags glTFLoadingFlags =
      vgeu::FileLoadingFlagBits::kPreMultiplyVertexColors;
  // | vgeu::FileLoadingFlagBits::kPreTransformVertices;
  //| vgeu::FileLoadingFlagBits::kFlipY;
  std::shared_ptr<vgeu::glTF::Model> damagedHelmet;

  damagedHelmet = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  damagedHelmet->loadFromFile(
      getAssetsPath() + "/models/DamagedHelmet/glTF/DamagedHelmet.gltf",
      glTFLoadingFlags);

  {
    ModelInstance modelInstance{};
    modelInstance.model = damagedHelmet;
    modelInstance.name = "damagedHelmet";
    addModelInstance(std::move(modelInstance));
  }
}

std::unique_ptr<vgeu::VgeuImage> VgeExample::createAttachment(
    vk::Format format, vk::ImageUsageFlagBits usage) {
  return std::make_unique<vgeu::VgeuImage>(
      device, globalAllocator->getAllocator(), vk::Format::eR16G16B16A16Sfloat,
      vk::Extent2D{offScreenFrameBuf.width, offScreenFrameBuf.height},
      vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eColorAttachment,
      vk::ImageLayout::eUndefined, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
      vk::ImageAspectFlagBits::eDepth, 1);
}

void VgeExample::prepareOffScreenFrameBuffer() {
  offScreenFrameBuf.width = 2048;
  offScreenFrameBuf.height = 2048;

  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    offScreenFrameBuf.position.push_back(
        std::move(createAttachment(vk::Format::eR16G16B16A16Sfloat,
                                   vk::ImageUsageFlagBits::eColorAttachment)));
    offScreenFrameBuf.normal.push_back(
        std::move(createAttachment(vk::Format::eR16G16B16A16Sfloat,
                                   vk::ImageUsageFlagBits::eColorAttachment)));
    offScreenFrameBuf.albedo.push_back(
        std::move(createAttachment(vk::Format::eR16G16B16A16Sfloat,
                                   vk::ImageUsageFlagBits::eColorAttachment)));
    offScreenFrameBuf.depth.push_back(std::move(createAttachment(
        depthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment)));
  }
  // render pass creation, subpass dependency
  std::vector<vk::AttachmentDescription> attachmentDescriptions;
  // position, normal, albedo, depth
  for (uint32_t i = 0; i < 4; i++) {
    vk::ImageLayout finalLayout;
    if (i == 3) {
      finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
    } else {
      finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }
    attachmentDescriptions.emplace_back(
        vk::AttachmentDescriptionFlags(), vk::Format::eUndefined,
        vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
        vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined,
        finalLayout);
  }
  attachmentDescriptions[0].format = offScreenFrameBuf.position[0]->getFormat();
  attachmentDescriptions[1].format = offScreenFrameBuf.normal[0]->getFormat();
  attachmentDescriptions[2].format = offScreenFrameBuf.albedo[0]->getFormat();
  attachmentDescriptions[3].format = offScreenFrameBuf.depth[0]->getFormat();
  std::vector<vk::AttachmentReference> colorReferences;
  colorReferences.emplace_back(0, vk::ImageLayout::eColorAttachmentOptimal);
  colorReferences.emplace_back(1, vk::ImageLayout::eColorAttachmentOptimal);
  colorReferences.emplace_back(2, vk::ImageLayout::eColorAttachmentOptimal);
  vk::AttachmentReference depthReference(
      3, vk::ImageLayout::eDepthAttachmentOptimal);
  vk::SubpassDescription subpassDescription(
      vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, {},
      colorReferences, {}, &depthReference);

  // subpass dependencies for layout transition
  std::vector<vk::SubpassDependency> dependencies;
  dependencies.emplace_back(VK_SUBPASS_EXTERNAL, 0u,
                            vk::PipelineStageFlagBits::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits::eLateFragmentTests,
                            vk::PipelineStageFlagBits::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits::eLateFragmentTests,
                            vk::AccessFlagBits::eDepthStencilAttachmentWrite,
                            vk::AccessFlagBits::eDepthStencilAttachmentWrite |
                                vk::AccessFlagBits::eDepthStencilAttachmentRead,
                            vk::DependencyFlags());
  dependencies.emplace_back(VK_SUBPASS_EXTERNAL, 0u,
                            vk::PipelineStageFlagBits::eBottomOfPipe,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::AccessFlagBits::eMemoryRead,
                            vk::AccessFlagBits::eColorAttachmentWrite |
                                vk::AccessFlagBits::eColorAttachmentRead,
                            vk::DependencyFlags());
  // transition for lighting render pass after geometry pass
  dependencies.emplace_back(0u, VK_SUBPASS_EXTERNAL,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits::eBottomOfPipe,
                            vk::AccessFlagBits::eColorAttachmentWrite |
                                vk::AccessFlagBits::eColorAttachmentRead,
                            vk::AccessFlagBits::eMemoryRead,
                            vk::DependencyFlags());
  vk::RenderPassCreateInfo renderPassCreateInfo(
      vk::RenderPassCreateFlags(), attachmentDescriptions, subpassDescription,
      dependencies);
  offScreenFrameBuf.renderPass =
      vk::raii::RenderPass(device, renderPassCreateInfo);
  // frame buffer creation

  std::vector<vk::raii::Framebuffer> framebuffers;
  framebuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    std::array<vk::ImageView, 4> attachments{};
    attachments[0] = *offScreenFrameBuf.position[i]->getImageView();
    attachments[1] = *offScreenFrameBuf.normal[i]->getImageView();
    attachments[2] = *offScreenFrameBuf.albedo[i]->getImageView();
    attachments[3] = *offScreenFrameBuf.depth[i]->getImageView();

    vk::FramebufferCreateInfo framebufferCreateInfo(
        vk::FramebufferCreateFlags(), *renderPass, attachments,
        offScreenFrameBuf.width, offScreenFrameBuf.height, 1);
    framebuffers.push_back(
        vk::raii::Framebuffer(device, framebufferCreateInfo));
  }
  // sampler creation
}
void VgeExample::prepareUniformBuffers() {}
void VgeExample::setupDescriptors() {}
void VgeExample::preparePipelines() {}

void VgeExample::buildCommandBuffers() {}
void VgeExample::buildDefferredCommandBuffers() {}

void VgeExample::draw() {
  prepareFrame();

  // offscreen rendering
  // scene rendering

  submitFrame();
}

void VgeExample::addModelInstance(ModelInstance&& newInstance) {
  size_t instanceIdx = modelInstances.size();
  modelInstances.push_back(std::move(newInstance));
  instanceMap[newInstance.name].push_back(instanceIdx);
}

const std::vector<size_t>& VgeExample::findInstances(const std::string& name) {
  assert(instanceMap.find(name) != instanceMap.end() &&
         "failed to find instance by name.");
  return instanceMap.at(name);
}

ModelInstance::ModelInstance(ModelInstance&& other) {
  model = other.model;
  simpleModel = other.simpleModel;
  name = other.name;
  isBone = other.isBone;
  animationIndex = other.animationIndex;
  animationTime = other.animationTime;
  transform = other.transform;
}

ModelInstance& ModelInstance::operator=(ModelInstance&& other) {
  model = other.model;
  simpleModel = other.simpleModel;
  name = other.name;
  isBone = other.isBone;
  animationIndex = other.animationIndex;
  animationTime = other.animationTime;
  transform = other.transform;
  return *this;
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()