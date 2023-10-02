#include "pipelines.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

// std
#include <array>
#include <cstring>
#include <limits>
#include <memory>

namespace vge {
VgeExample::VgeExample() : VgeBase() { title = "Pipelines Example"; }
VgeExample::~VgeExample() {}

void VgeExample::initVulkan() {
  // ubo setup
  globalUbo.model = glm::scale(glm::mat4(1.0), glm::vec3(10.f));
  globalUbo.normalMatrix = glm::mat4(glm::inverse(glm::mat3(globalUbo.model)));
  // NOTE: check coordinate space
  globalUbo.lightPos = glm::vec4(-2.f, -4.f, -2.f, 0.f);
  // camera setup
  camera.setViewTarget(glm::vec3{-2.f, -2.f, -5.f}, glm::vec3{0.f, 0.f, 0.f});
  camera.setPerspectiveProjection(
      glm::radians(60.f),
      (static_cast<float>(width) / 3.f) / static_cast<float>(height), 0.1f,
      256.f);
  VgeBase::initVulkan();
}

void VgeExample::getEnabledExtensions() {
  enabledFeatures.samplerAnisotropy =
      physicalDevice.getFeatures().samplerAnisotropy;
  enabledFeatures.fillModeNonSolid =
      physicalDevice.getFeatures().fillModeNonSolid;
  enabledFeatures.wideLines = physicalDevice.getFeatures().wideLines;
}

void VgeExample::prepare() {
  VgeBase::prepare();
  loadAssets();
  createUniformBuffers();
  createDescriptorSetLayout();
  createDescriptorPool();
  createDescriptorSets();
  createPipelines();
  prepared = true;
}
void VgeExample::loadAssets() {
  vgeu::FileLoadingFlags glTFLoadingFlags =
      vgeu::FileLoadingFlagBits::kPreTransformVertices |
      vgeu::FileLoadingFlagBits::kPreMultiplyVertexColors |
      vgeu::FileLoadingFlagBits::kFlipY;

  scene = std::make_unique<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  scene->loadFromFile(getAssetsPath() + "/models/apple-vertex-color.gltf",
                      glTFLoadingFlags);
}

void VgeExample::createUniformBuffers() {
  // NOTE: prevent move during vector element creation.
  uniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    uniformBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(GlobalUbo), 1,
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT));
    std::memcpy(uniformBuffers[i]->getMappedData(), &globalUbo,
                sizeof(GlobalUbo));
  }
}

void VgeExample::createDescriptorSetLayout() {
  // set 0
  vk::DescriptorSetLayoutBinding layoutBinding(
      0, vk::DescriptorType::eUniformBuffer, 1,
      vk::ShaderStageFlagBits::eVertex);
  vk::DescriptorSetLayoutCreateInfo layoutCI({}, 1, &layoutBinding);
  descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutCI);

  // set 1
  std::vector<vk::DescriptorSetLayout> setLayouts;
  setLayouts.push_back(*descriptorSetLayout);
  setLayouts.push_back(*scene->descriptorSetLayoutImage);

  vk::PipelineLayoutCreateInfo pipelineLayoutCI({}, setLayouts);
  pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutCI);
}

void VgeExample::createDescriptorPool() {
  std::vector<vk::DescriptorPoolSize> poolSizes;
  poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer,
                         MAX_CONCURRENT_FRAMES);
  // NOTE: need to check flag
  vk::DescriptorPoolCreateInfo descriptorPoolCI(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      MAX_CONCURRENT_FRAMES, poolSizes);
  descriptorPool = vk::raii::DescriptorPool(device, descriptorPoolCI);
}

void VgeExample::createDescriptorSets() {
  vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                          *descriptorSetLayout);

  descriptorSets.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    // NOTE: move descriptor set
    descriptorSets.push_back(
        std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
  }

  std::vector<vk::DescriptorBufferInfo> bufferInfos;
  bufferInfos.reserve(uniformBuffers.size());

  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  writeDescriptorSets.reserve(uniformBuffers.size());
  for (int i = 0; i < uniformBuffers.size(); i++) {
    // copy
    bufferInfos.push_back(uniformBuffers[i]->descriptorInfo());
    // NOTE: ArrayProxyNoTemporaries has no T rvalue constructor.
    writeDescriptorSets.emplace_back(*descriptorSets[i], 0, 0,
                                     vk::DescriptorType::eUniformBuffer,
                                     nullptr, bufferInfos.back());
  }
  device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

void VgeExample::createPipelines() {
  vk::PipelineVertexInputStateCreateInfo vertexInputSCI =
      vgeu::glTF::Vertex::getPipelineVertexInputState({
          vgeu::glTF::VertexComponent::kPosition,
          vgeu::glTF::VertexComponent::kNormal,
          vgeu::glTF::VertexComponent::kUV,
          vgeu::glTF::VertexComponent::kColor,
      });

  vk::PipelineInputAssemblyStateCreateInfo inputAssemblySCI(
      vk::PipelineInputAssemblyStateCreateFlags(),
      vk::PrimitiveTopology::eTriangleList);

  vk::PipelineViewportStateCreateInfo viewportSCI(
      vk::PipelineViewportStateCreateFlags(), 1, nullptr, 1, nullptr);

  vk::PipelineRasterizationStateCreateInfo rasterizationSCI(
      vk::PipelineRasterizationStateCreateFlags(), false, false,
      vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
      vk::FrontFace::eCounterClockwise, false, 0.0f, 0.0f, 0.0f, 1.0f);

  vk::PipelineMultisampleStateCreateInfo multisampleSCI(
      vk::PipelineMultisampleStateCreateFlags(), vk::SampleCountFlagBits::e1);

  vk::StencilOpState stencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep,
                                    vk::StencilOp::eKeep,
                                    vk::CompareOp::eAlways);
  vk::PipelineDepthStencilStateCreateInfo depthStencilSCI(
      vk::PipelineDepthStencilStateCreateFlags(), true, true,
      vk::CompareOp::eLessOrEqual, false, false, stencilOpState,
      stencilOpState);

  vk::PipelineColorBlendAttachmentState colorBlendAttachmentState(
      false, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
      vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);

  vk::PipelineColorBlendStateCreateInfo colorBlendSCI(
      vk::PipelineColorBlendStateCreateFlags(), false, vk::LogicOp::eNoOp,
      colorBlendAttachmentState, {{1.0f, 1.0f, 1.0f, 1.0f}});

  std::array<vk::DynamicState, 2> dynamicStates = {vk::DynamicState::eViewport,
                                                   vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dynamicSCI(
      vk::PipelineDynamicStateCreateFlags(), dynamicStates);

  auto vertCode =
      vgeu::readFile(getShadersPath() + "/pipelines/phong.vert.spv");
  auto fragCode =
      vgeu::readFile(getShadersPath() + "/pipelines/phong.frag.spv");
  // NOTE: after pipeline creation, shader modules can be destroyed.
  vk::raii::ShaderModule vertShaderModule =
      vgeu::createShaderModule(device, vertCode);
  vk::raii::ShaderModule fragShaderModule =
      vgeu::createShaderModule(device, fragCode);

  std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStageCIs{
      vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                        vk::ShaderStageFlagBits::eVertex,
                                        *vertShaderModule, "main", nullptr),
      vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                        vk::ShaderStageFlagBits::eFragment,
                                        *fragShaderModule, "main", nullptr),
  };

  // base and derivatives
  vk::GraphicsPipelineCreateInfo graphicsPipelineCI(
      vk::PipelineCreateFlagBits::eAllowDerivatives, shaderStageCIs,
      &vertexInputSCI, &inputAssemblySCI, nullptr, &viewportSCI,
      &rasterizationSCI, &multisampleSCI, &depthStencilSCI, &colorBlendSCI,
      &dynamicSCI, *pipelineLayout, *renderPass);

  pipelines.phong =
      vk::raii::Pipeline(device, pipelineCache, graphicsPipelineCI);

  {
    graphicsPipelineCI.flags = vk::PipelineCreateFlagBits::eDerivative;
    graphicsPipelineCI.basePipelineHandle = *pipelines.phong;
    // only handle or index
    graphicsPipelineCI.basePipelineIndex = -1;

    vertCode = vgeu::readFile(getShadersPath() + "/pipelines/toon.vert.spv");
    fragCode = vgeu::readFile(getShadersPath() + "/pipelines/toon.frag.spv");
    // NOTE: after pipeline creation, shader modules can be destroyed.
    vertShaderModule = vgeu::createShaderModule(device, vertCode);
    fragShaderModule = vgeu::createShaderModule(device, fragCode);
    shaderStageCIs[0] = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex,
        *vertShaderModule, "main", nullptr);
    shaderStageCIs[1] = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "main", nullptr);
    pipelines.toon =
        vk::raii::Pipeline(device, pipelineCache, graphicsPipelineCI);
  }
  if (enabledFeatures.fillModeNonSolid) {
    rasterizationSCI.polygonMode = vk::PolygonMode::eLine;
    vertCode =
        vgeu::readFile(getShadersPath() + "/pipelines/wireframe.vert.spv");
    fragCode =
        vgeu::readFile(getShadersPath() + "/pipelines/wireframe.frag.spv");
    // NOTE: after pipeline creation, shader modules can be destroyed.
    vertShaderModule = vgeu::createShaderModule(device, vertCode);
    fragShaderModule = vgeu::createShaderModule(device, fragCode);
    shaderStageCIs[0] = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex,
        *vertShaderModule, "main", nullptr);
    shaderStageCIs[1] = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "main", nullptr);
    pipelines.wireframe =
        vk::raii::Pipeline(device, pipelineCache, graphicsPipelineCI);
  }
}

void VgeExample::render() {
  if (!prepared) {
    return;
  }
  // CHECK: ubo update frequency.
  globalUbo.view = camera.getView();
  globalUbo.projection = camera.getProjection();
  globalUbo.inverseView = camera.getInverseView();
  std::memcpy(uniformBuffers[currentFrameIndex]->getMappedData(), &globalUbo,
              sizeof(GlobalUbo));

  draw();
}

void VgeExample::draw() {
  // std::cout << "Call: draw() - " << currentFrameIndex << std::endl;
  vk::Result result =
      device.waitForFences(*waitFences[currentFrameIndex], VK_TRUE,
                           std::numeric_limits<uint64_t>::max());
  assert(result != vk::Result::eTimeout && "Timed out: waitFence");

  device.resetFences(*waitFences[currentFrameIndex]);

  prepareFrame();

  // draw cmds recording or command buffers should be built already.
  buildCommandBuffers();

  vk::PipelineStageFlags waitDstStageMask(
      vk::PipelineStageFlagBits::eColorAttachmentOutput);
  // NOTE: parameter array type has no r value constructor
  vk::SubmitInfo submitInfo(*presentCompleteSemaphores[currentFrameIndex],
                            waitDstStageMask,
                            *drawCmdBuffers[currentFrameIndex],
                            *renderCompleteSemaphores[currentFrameIndex]);

  queue.submit(submitInfo, *waitFences[currentFrameIndex]);
  submitFrame();
}

void VgeExample::buildCommandBuffers() {
  // reset cmd buffers - commandPool flag

  // begin cmd buffer
  drawCmdBuffers[currentFrameIndex].begin({});
  // begin render pass
  std::array<vk::ClearValue, 2> clearValues;
  clearValues[0].color = vk::ClearColorValue(0.2f, 0.2f, 0.2f, 0.2f);
  clearValues[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);
  // NOTE: use swapChainData->swapChainExtent, height instead of direct
  // vgeuWindow->getExtent()
  vk::RenderPassBeginInfo renderPassBeginInfo(
      *renderPass, *frameBuffers[currentImageIndex],
      vk::Rect2D(vk::Offset2D(0, 0), swapChainData->swapChainExtent),
      clearValues);
  // NOTE: no secondary cmd buffers
  drawCmdBuffers[currentFrameIndex].beginRenderPass(
      renderPassBeginInfo, vk::SubpassContents::eInline);

  // set viewport and scissors

  drawCmdBuffers[currentFrameIndex].setScissor(
      0, vk::Rect2D(vk::Offset2D(0, 0), swapChainData->swapChainExtent));

  // bind ubo descriptor
  drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0,
      {*descriptorSets[currentFrameIndex]}, nullptr);
  // bind vertex buffer
  // bind index buffer
  scene->bindBuffers(drawCmdBuffers[currentFrameIndex]);

  // Left view
  {
    drawCmdBuffers[currentFrameIndex].setViewport(
        0, vk::Viewport(
               0.0f, 0.0f,
               static_cast<float>(swapChainData->swapChainExtent.width) / 3.f,
               static_cast<float>(swapChainData->swapChainExtent.height), 0.0f,
               1.0f));
    // bind pipeline
    drawCmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eGraphics, *pipelines.phong);
    // draw indexed
    scene->draw(currentFrameIndex, drawCmdBuffers[currentFrameIndex],
                vgeu::RenderFlagBits::kBindImages, *pipelineLayout, 1);
  }

  // Center view
  {
    drawCmdBuffers[currentFrameIndex].setViewport(
        0, vk::Viewport(
               static_cast<float>(swapChainData->swapChainExtent.width) / 3.f,
               0.0f,
               static_cast<float>(swapChainData->swapChainExtent.width) / 3.f,
               static_cast<float>(swapChainData->swapChainExtent.height), 0.0f,
               1.0f));
    // bind pipeline
    drawCmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eGraphics, *pipelines.toon);
    // draw indexed
    scene->draw(currentFrameIndex, drawCmdBuffers[currentFrameIndex],
                vgeu::RenderFlagBits::kBindImages, *pipelineLayout, 1);
  }

  if (enabledFeatures.wideLines) {
    drawCmdBuffers[currentFrameIndex].setLineWidth(2.f);
  }
  if (enabledFeatures.fillModeNonSolid) {
    drawCmdBuffers[currentFrameIndex].setViewport(
        0, vk::Viewport(
               static_cast<float>(swapChainData->swapChainExtent.width) / 3.f *
                   2.f,
               0.0f,
               static_cast<float>(swapChainData->swapChainExtent.width) / 3.f,
               static_cast<float>(swapChainData->swapChainExtent.height), 0.0f,
               1.0f));
    // bind pipeline
    drawCmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eGraphics, *pipelines.wireframe);
    // draw indexed
    scene->draw(currentFrameIndex, drawCmdBuffers[currentFrameIndex],
                vgeu::RenderFlagBits::kBindImages, *pipelineLayout, 1);
  }

  // UI overlay draw
  drawUI(drawCmdBuffers[currentFrameIndex]);

  // end renderpass
  drawCmdBuffers[currentFrameIndex].endRenderPass();

  // end command buffer
  drawCmdBuffers[currentFrameIndex].end();
}
void VgeExample::viewChanged() {
  // std::cout << "Call: viewChanged()" << std::endl;
  camera.setAspectRatio((static_cast<float>(width) / 3.f) /
                        static_cast<float>(height));
  // NOTE: moved updating ubo into render() to use frameindex.
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()