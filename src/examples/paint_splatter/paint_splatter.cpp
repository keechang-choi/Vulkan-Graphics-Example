#include "paint_splatter.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

// std
#include <array>
#include <cstring>
#include <iostream>
#include <limits>

namespace vge {

VgeExample::VgeExample() : VgeBase() { title = "Paint Splatter Example"; }
VgeExample::~VgeExample() {}

void VgeExample::setupCommandLineParser(CLI::App& app) {
  VgeBase::setupCommandLineParser(app);
}

void VgeExample::getEnabledFeatures() {}

void VgeExample::initVulkan() {
  // World convention (consistent with engine): screen-up = world -Y, so the
  // canvas floor (X-Z plane at y=0) is viewed from the -Y side (above) and -Z
  // (behind). Gravity will pull +Y; spoids/fluid live at y<0 (above the floor).
  camera.setViewTarget(glm::vec3{0.f, -4.f, -4.f}, glm::vec3{0.f, 0.f, 0.f});
  camera.setPerspectiveProjection(
      glm::radians(60.f),
      static_cast<float>(width) / static_cast<float>(height), 0.1f, 256.f);
  VgeBase::initVulkan();
}

void VgeExample::prepare() {
  VgeBase::prepare();
  createVertexBuffer();
  createIndexBuffer();
  createUniformBuffers();
  createDescriptorSetLayout();
  createDescriptorPool();
  createDescriptorSets();
  createPipelines();
  prepared = true;
}

// ---------------------------------------------------------------------------
// Geometry: X-Z plane quad at y=0, side length kCanvasWorld, UV [0..1]
// ---------------------------------------------------------------------------
void VgeExample::createVertexBuffer() {
  const float h = kCanvasWorld * 0.5f;
  // positions in X-Z plane, y=0; UVs cover [0,1]x[0,1]
  std::vector<CanvasVertex> vertices{
      {{-h, 0.f, -h}, {0.f, 0.f}},
      {{h, 0.f, -h}, {1.f, 0.f}},
      {{h, 0.f, h}, {1.f, 1.f}},
      {{-h, 0.f, h}, {0.f, 1.f}},
  };

  vgeu::VgeuBuffer stagingBuffer(
      globalAllocator->getAllocator(), sizeof(CanvasVertex),
      static_cast<uint32_t>(vertices.size()),
      vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);
  std::memcpy(stagingBuffer.getMappedData(), vertices.data(),
              stagingBuffer.getBufferSize());

  // Device-local destination: filled once via the staging copy below, never
  // CPU-mapped, so request DEVICE_PREFER with no host-access flags.
  vertexBuffer = std::make_unique<vgeu::VgeuBuffer>(
      globalAllocator->getAllocator(), sizeof(CanvasVertex),
      static_cast<uint32_t>(vertices.size()),
      vk::BufferUsageFlagBits::eVertexBuffer |
          vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

  vgeu::oneTimeSubmit(
      device, commandPool, queue,
      [&](const vk::raii::CommandBuffer& cmdBuffer) {
        cmdBuffer.copyBuffer(
            stagingBuffer.getBuffer(), vertexBuffer->getBuffer(),
            vk::BufferCopy(0, 0, stagingBuffer.getBufferSize()));
      });
}

void VgeExample::createIndexBuffer() {
  // Two triangles: 0-1-2, 0-2-3
  std::vector<uint32_t> indices{0, 1, 2, 0, 2, 3};

  vgeu::VgeuBuffer stagingBuffer(
      globalAllocator->getAllocator(), sizeof(uint32_t),
      static_cast<uint32_t>(indices.size()),
      vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);
  std::memcpy(stagingBuffer.getMappedData(), indices.data(),
              stagingBuffer.getBufferSize());

  // Device-local destination (see createVertexBuffer): staging-filled only.
  indexBuffer = std::make_unique<vgeu::VgeuBuffer>(
      globalAllocator->getAllocator(), sizeof(uint32_t),
      static_cast<uint32_t>(indices.size()),
      vk::BufferUsageFlagBits::eIndexBuffer |
          vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);

  vgeu::oneTimeSubmit(
      device, commandPool, queue,
      [&](const vk::raii::CommandBuffer& cmdBuffer) {
        cmdBuffer.copyBuffer(
            stagingBuffer.getBuffer(), indexBuffer->getBuffer(),
            vk::BufferCopy(0, 0, stagingBuffer.getBufferSize()));
      });
}

void VgeExample::createUniformBuffers() {
  uniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
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
  // set=0, binding=0: GlobalUbo (vertex stage)
  vk::DescriptorSetLayoutBinding layoutBinding(
      0, vk::DescriptorType::eUniformBuffer, 1,
      vk::ShaderStageFlagBits::eVertex);
  vk::DescriptorSetLayoutCreateInfo layoutCI({}, 1, &layoutBinding);
  descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutCI);

  vk::PipelineLayoutCreateInfo pipelineLayoutCI({}, *descriptorSetLayout);
  pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutCI);
}

void VgeExample::createDescriptorPool() {
  std::vector<vk::DescriptorPoolSize> poolSizes;
  poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer,
                         MAX_CONCURRENT_FRAMES);
  vk::DescriptorPoolCreateInfo descriptorPoolCI(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      MAX_CONCURRENT_FRAMES, poolSizes);
  descriptorPool = vk::raii::DescriptorPool(device, descriptorPoolCI);
}

void VgeExample::createDescriptorSets() {
  vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                          *descriptorSetLayout);
  descriptorSets.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    descriptorSets.push_back(
        std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
  }

  std::vector<vk::DescriptorBufferInfo> bufferInfos;
  bufferInfos.reserve(uniformBuffers.size());
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  writeDescriptorSets.reserve(uniformBuffers.size());

  for (uint32_t i = 0; i < static_cast<uint32_t>(uniformBuffers.size()); i++) {
    bufferInfos.push_back(uniformBuffers[i]->descriptorInfo());
    writeDescriptorSets.emplace_back(*descriptorSets[i], 0, 0,
                                     vk::DescriptorType::eUniformBuffer,
                                     nullptr, bufferInfos.back());
  }
  device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

void VgeExample::createPipelines() {
  auto vertCode =
      vgeu::readFile(getShadersPath() + "/paint_splatter/canvas.vert.spv");
  auto fragCode =
      vgeu::readFile(getShadersPath() + "/paint_splatter/canvas.frag.spv");

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

  // Vertex input: binding 0 = CanvasVertex (position vec3 + uv vec2)
  vk::VertexInputBindingDescription vertexInputBindingDescription(
      0, sizeof(CanvasVertex));

  std::vector<vk::VertexInputAttributeDescription>
      vertexInputAttributeDescriptions;
  vertexInputAttributeDescriptions.emplace_back(
      0, 0, vk::Format::eR32G32B32Sfloat, offsetof(CanvasVertex, position));
  vertexInputAttributeDescriptions.emplace_back(1, 0, vk::Format::eR32G32Sfloat,
                                                offsetof(CanvasVertex, uv));

  vk::PipelineVertexInputStateCreateInfo vertexInputSCI(
      vk::PipelineVertexInputStateCreateFlags(), vertexInputBindingDescription,
      vertexInputAttributeDescriptions);

  vk::PipelineInputAssemblyStateCreateInfo inputAssemblySCI(
      vk::PipelineInputAssemblyStateCreateFlags(),
      vk::PrimitiveTopology::eTriangleList);

  vk::PipelineViewportStateCreateInfo viewportSCI(
      vk::PipelineViewportStateCreateFlags(), 1, nullptr, 1, nullptr);

  // Back-face cull OFF so quad is visible from both sides
  vk::PipelineRasterizationStateCreateInfo rasterizationSCI(
      vk::PipelineRasterizationStateCreateFlags(), false, false,
      vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
      vk::FrontFace::eCounterClockwise, false, 0.0f, 0.0f, 0.0f, 1.0f);

  vk::PipelineMultisampleStateCreateInfo multisampleSCI(
      vk::PipelineMultisampleStateCreateFlags(), vk::SampleCountFlagBits::e1);

  vk::StencilOpState stencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep,
                                    vk::StencilOp::eKeep,
                                    vk::CompareOp::eAlways);
  // Depth test on
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

  vk::GraphicsPipelineCreateInfo graphicsPipelineCI(
      vk::PipelineCreateFlags(), shaderStageCIs, &vertexInputSCI,
      &inputAssemblySCI, nullptr, &viewportSCI, &rasterizationSCI,
      &multisampleSCI, &depthStencilSCI, &colorBlendSCI, &dynamicSCI,
      *pipelineLayout, *renderPass);

  pipeline = vk::raii::Pipeline(device, pipelineCache, graphicsPipelineCI);
}

// ---------------------------------------------------------------------------
// Per-frame UBO update
// ---------------------------------------------------------------------------
void VgeExample::updateGlobalUbo() {
  globalUbo.projection = camera.getProjection();
  globalUbo.view = camera.getView();
  globalUbo.inverseView = camera.getInverseView();
  globalUbo.canvasInfo =
      glm::vec4(kCanvasWorld * 0.5f, kCanvasWorld * 0.5f, kCanvasWorld, 0.f);
  std::memcpy(uniformBuffers[currentFrameIndex]->getMappedData(), &globalUbo,
              sizeof(GlobalUbo));
}

// ---------------------------------------------------------------------------
// render / draw
// ---------------------------------------------------------------------------
void VgeExample::render() {
  if (!prepared) return;
  updateGlobalUbo();
  draw();
}

void VgeExample::draw() {
  vk::Result result =
      device.waitForFences(*waitFences[currentFrameIndex], VK_TRUE,
                           std::numeric_limits<uint64_t>::max());
  assert(result != vk::Result::eTimeout && "Timed out: waitFence");

  device.resetFences(*waitFences[currentFrameIndex]);

  prepareFrame();
  buildCommandBuffers();

  vk::PipelineStageFlags waitDstStageMask(
      vk::PipelineStageFlagBits::eColorAttachmentOutput);
  vk::SubmitInfo submitInfo(*presentCompleteSemaphores[currentFrameIndex],
                            waitDstStageMask,
                            *drawCmdBuffers[currentFrameIndex],
                            *renderCompleteSemaphores[currentFrameIndex]);

  queue.submit(submitInfo, *waitFences[currentFrameIndex]);
  submitFrame();
}

void VgeExample::buildCommandBuffers() {
  drawCmdBuffers[currentFrameIndex].begin({});

  // Mid-gray clear
  std::array<vk::ClearValue, 2> clearValues;
  clearValues[0].color = vk::ClearColorValue(0.5f, 0.5f, 0.5f, 1.0f);
  clearValues[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

  vk::RenderPassBeginInfo renderPassBeginInfo(
      *renderPass, *frameBuffers[currentImageIndex],
      vk::Rect2D(vk::Offset2D(0, 0), swapChainData->swapChainExtent),
      clearValues);
  drawCmdBuffers[currentFrameIndex].beginRenderPass(
      renderPassBeginInfo, vk::SubpassContents::eInline);

  // Viewport + scissor
  drawCmdBuffers[currentFrameIndex].setViewport(
      0, vk::Viewport(0.0f, 0.0f,
                      static_cast<float>(swapChainData->swapChainExtent.width),
                      static_cast<float>(swapChainData->swapChainExtent.height),
                      0.0f, 1.0f));
  drawCmdBuffers[currentFrameIndex].setScissor(
      0, vk::Rect2D(vk::Offset2D(0, 0), swapChainData->swapChainExtent));

  // Bind pipeline + set=0 UBO
  drawCmdBuffers[currentFrameIndex].bindPipeline(
      vk::PipelineBindPoint::eGraphics, *pipeline);
  drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0,
      {*descriptorSets[currentFrameIndex]}, nullptr);

  // Bind canvas geometry
  drawCmdBuffers[currentFrameIndex].bindVertexBuffers(
      0, {vertexBuffer->getBuffer()}, {0});
  drawCmdBuffers[currentFrameIndex].bindIndexBuffer(indexBuffer->getBuffer(), 0,
                                                    vk::IndexType::eUint32);

  // Draw quad (6 indices)
  drawCmdBuffers[currentFrameIndex].drawIndexed(indexBuffer->getInstanceCount(),
                                                1, 0, 0, 0);

  // ImGui overlay
  drawUI(drawCmdBuffers[currentFrameIndex]);

  drawCmdBuffers[currentFrameIndex].endRenderPass();
  drawCmdBuffers[currentFrameIndex].end();
}

void VgeExample::viewChanged() {
  camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
  // NOTE: UBO update deferred to render() to use correct frame index.
}

// ---------------------------------------------------------------------------
// ImGui panel skeleton
// ---------------------------------------------------------------------------
void VgeExample::onUpdateUIOverlay() {
  if (ImGui::CollapsingHeader("Paint Splatter",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    // Read-only info
    ImGui::Text("Frame time : %.3f ms", frameTimer * 1000.0f);

    glm::vec3 camPos = camera.getPosition();
    ImGui::Text("Camera pos : (%.2f, %.2f, %.2f)", camPos.x, camPos.y,
                camPos.z);

    // Placeholder button — no-op, just logs to stdout
    if (uiOverlay->button("Save (no-op)")) {
      std::cout << "[paint_splatter] Save button pressed (no-op)\n";
    }
  }
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()
