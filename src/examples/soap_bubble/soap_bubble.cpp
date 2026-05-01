#include "soap_bubble.hpp"

#include "vgeu_utils.hpp"

namespace vge {

VgeExample::VgeExample() : VgeBase() { title = "soap_bubble"; }
VgeExample::~VgeExample() {}

void VgeExample::setupCommandLineParser(CLI::App& app) {}

void VgeExample::setOptions(const std::optional<Options>& o) {
  if (o) opts = *o;
}

void VgeExample::initVulkan() {
  camera.setViewTarget(glm::vec3{0.f, 0.f, -3.f}, glm::vec3{0.f, 0.f, 0.f});
  camera.setPerspectiveProjection(
      glm::radians(60.f),
      static_cast<float>(width) / static_cast<float>(height), 0.1f, 256.f);
  VgeBase::initVulkan();
}
void VgeExample::getEnabledExtensions() {}
void VgeExample::getEnabledFeatures() {
  enabledFeatures.samplerAnisotropy = VK_TRUE;
}

void VgeExample::prepare() {
  VgeBase::prepare();
  loadAssets();
  prepareIBL();
  prepareUniformBuffers();
  setupDescriptors();
  preparePipelines();
  prepared = true;
}

void VgeExample::loadAssets() {}

void VgeExample::prepareIBL() {
  iblConfig.hdrPath =
      getAssetsPath() + "/textures/hdr/tree_lined_driveway_4k.hdr";
  iblConfig.commonShadersPath = getShadersPath() + "/common";
  iblConfig.useJitter = opts.useJitter;
  iblBaker = std::make_unique<vgeu::IBLBaker>(
      device, globalAllocator->getAllocator(), queue, commandPool);
  iblBaker->bake(iblConfig);
}

void VgeExample::prepareUniformBuffers() {}

void VgeExample::setupDescriptors() {
  // Minimal pool: only skybox for now (MAX_CONCURRENT_FRAMES CIS descriptors)
  std::vector<vk::DescriptorPoolSize> poolSizes{
      {vk::DescriptorType::eCombinedImageSampler,
       1u * MAX_CONCURRENT_FRAMES /*skybox*/}};
  descriptorPool = vk::raii::DescriptorPool(
      device,
      vk::DescriptorPoolCreateInfo(
          vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
          MAX_CONCURRENT_FRAMES /*skybox*/, poolSizes));

  skybox = std::make_unique<vgeu::Skybox>(
      device, pipelineCache, descriptorPool, renderPass,
      iblConfig.commonShadersPath, *iblBaker, MAX_CONCURRENT_FRAMES);
}

void VgeExample::preparePipelines() {}

void VgeExample::buildCommandBuffers() {
  const auto& cmd = drawCmdBuffers[currentFrameIndex];
  cmd.begin({});

  std::array<vk::ClearValue, 2> clearValues;
  clearValues[0].color = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
  clearValues[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

  vk::RenderPassBeginInfo rpBegin(*renderPass, *frameBuffers[currentImageIndex],
                                  {{0, 0}, {width, height}}, clearValues);
  cmd.beginRenderPass(rpBegin, vk::SubpassContents::eInline);

  cmd.setViewport(
      0, vk::Viewport(0.f, 0.f, (float)width, (float)height, 0.f, 1.f));
  cmd.setScissor(0, vk::Rect2D({0, 0}, {width, height}));

  skybox->draw(cmd, currentFrameIndex, camera.getView(),
               camera.getProjection(), opts.skyboxLod);

  drawUI(cmd);
  cmd.endRenderPass();
  cmd.end();
}

void VgeExample::draw() {
  {
    vk::Result result = device.waitForFences(*waitFences[currentFrameIndex],
                                             VK_TRUE, UINT64_MAX);
    assert(result != vk::Result::eTimeout);
    device.resetFences(*waitFences[currentFrameIndex]);
  }
  prepareFrame();
  buildCommandBuffers();
  {
    vk::PipelineStageFlags waitStage(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    vk::SubmitInfo submitInfo(*presentCompleteSemaphores[currentFrameIndex],
                              waitStage, *drawCmdBuffers[currentFrameIndex],
                              *renderCompleteSemaphores[currentFrameIndex]);
    queue.submit(submitInfo, *waitFences[currentFrameIndex]);
  }
  submitFrame();
}

void VgeExample::render() {
  if (!prepared) return;
  draw();
}

void VgeExample::viewChanged() {}

void VgeExample::onUpdateUIOverlay() {}

void VgeExample::updateGlobalsUbo() {}
void VgeExample::updateBubbleParamsUbo() {}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()
