#include "triangle.hpp"

// std
#include <array>
#include <cstring>
#include <glm/gtc/matrix_transform.hpp>
#include <limits>

namespace vge {
VgeExample::VgeExample() : VgeBase() {
  title = "First Triangle Example";
  // camera setup
  camera.setViewTarget(glm::vec3{-2.f, -2.f, -5.f}, glm::vec3{0.f, 0.f, 0.f});
  camera.setPerspectiveProjection(
      glm::radians(60.f),
      static_cast<float>(width) / static_cast<float>(height), 0.1f, 256.f);
}
VgeExample::~VgeExample() {}

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

void VgeExample::createUniformBuffers() {
  // NOTE: prevent move during vector element creation.
  uniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    uniformBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator, sizeof(GlobalUbo), 1,
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT));
    std::memcpy(uniformBuffers[i]->getMappedData(), &globalUbo,
                sizeof(GlobalUbo));
  }
}

void VgeExample::createVertexBuffer() {
  std::vector<Vertex> vertices{
      {{1.0f, 1.0f, 0.1f}, {1.0f, 0.0f, 0.0f}},
      {{-1.0f, 1.0f, 0.2f}, {0.0f, 1.0f, 0.0f}},
      {{0.0f, -1.0f, 0.3f}, {0.0f, 0.0f, 1.0f}},
  };

  vgeu::VgeuBuffer stagingBuffer(
      globalAllocator, sizeof(Vertex), static_cast<uint32_t>(vertices.size()),
      vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);

  std::memcpy(stagingBuffer.getMappedData(), vertices.data(),
              stagingBuffer.getBufferSize());

  vertexBuffer = std::make_unique<vgeu::VgeuBuffer>(
      globalAllocator, sizeof(Vertex), static_cast<uint32_t>(vertices.size()),
      vk::BufferUsageFlagBits::eVertexBuffer |
          vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);

  // single Time command
  // begin, copy , end, submit, waitIdle queue.
  vgeu::oneTimeSubmit(
      device, commandPool, queue,
      [&](const vk::raii::CommandBuffer& cmdBuffer) {
        cmdBuffer.copyBuffer(
            stagingBuffer.getBuffer(), vertexBuffer->getBuffer(),
            vk::BufferCopy(0, 0, stagingBuffer.getBufferSize()));
      });
}

void VgeExample::createIndexBuffer() {
  std::vector<uint32_t> indices{0, 1, 2};

  vgeu::VgeuBuffer stagingBuffer(
      globalAllocator, sizeof(uint32_t), static_cast<uint32_t>(indices.size()),
      vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);

  std::memcpy(stagingBuffer.getMappedData(), indices.data(),
              stagingBuffer.getBufferSize());

  indexBuffer = std::make_unique<vgeu::VgeuBuffer>(
      globalAllocator, sizeof(uint32_t), static_cast<uint32_t>(indices.size()),
      vk::BufferUsageFlagBits::eIndexBuffer |
          vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);

  // single Time command copy
  vgeu::oneTimeSubmit(
      device, commandPool, queue,
      [&](const vk::raii::CommandBuffer& cmdBuffer) {
        cmdBuffer.copyBuffer(
            stagingBuffer.getBuffer(), indexBuffer->getBuffer(),
            vk::BufferCopy(0, 0, stagingBuffer.getBufferSize()));
      });
}

void VgeExample::createDescriptorSetLayout() {
  // binding 0
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
  auto vertCode =
      vgeu::readFile(getShadersPath() + "/triangle/simple_shader.vert.spv");
  auto fragCode =
      vgeu::readFile(getShadersPath() + "/triangle/simple_shader.frag.spv");
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
  //   // NOTE: brace init list flags may cause template argument deducing fail
  //   std::vector<vk::PipelineShaderStageCreateInfo> pipelineShaderStageCI;
  //   pipelineShaderStageCI.emplace_back(vk::PipelineShaderStageCreateFlags{},
  //                                      vk::ShaderStageFlagBits::eVertex,
  //                                      *vertShaderModule, "main");
  //   pipelineShaderStageCI.push_back(vk::PipelineShaderStageCreateInfo(
  //       {}, vk::ShaderStageFlagBits::eVertex, *vertShaderModule, "main"));

  vk::VertexInputBindingDescription vertexInputBindingDescription(
      0, sizeof(Vertex));

  std::vector<vk::VertexInputAttributeDescription>
      vertexInputAttributeDescriptions;
  vertexInputAttributeDescriptions.emplace_back(
      0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position));
  vertexInputAttributeDescriptions.emplace_back(
      1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color));

  vk::PipelineVertexInputStateCreateInfo vertexInputSCI(
      vk::PipelineVertexInputStateCreateFlags(), vertexInputBindingDescription,
      vertexInputAttributeDescriptions);

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

  vk::GraphicsPipelineCreateInfo graphicsPipelineCI(
      vk::PipelineCreateFlags(), shaderStageCIs, &vertexInputSCI,
      &inputAssemblySCI, nullptr, &viewportSCI, &rasterizationSCI,
      &multisampleSCI, &depthStencilSCI, &colorBlendSCI, &dynamicSCI,
      *pipelineLayout, *renderPass);

  pipeline = vk::raii::Pipeline(device, pipelineCache, graphicsPipelineCI);
}

void VgeExample::render() {
  if (!prepared) {
    return;
  }
  // CHECK: ubo update frequency.
  globalUbo.view = camera.getView();
  globalUbo.projection = camera.getProjection();
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
  drawCmdBuffers[currentFrameIndex].setViewport(
      0, vk::Viewport(0.0f, 0.0f,
                      static_cast<float>(swapChainData->swapChainExtent.width),
                      static_cast<float>(swapChainData->swapChainExtent.height),
                      0.0f, 1.0f));
  drawCmdBuffers[currentFrameIndex].setScissor(
      0, vk::Rect2D(vk::Offset2D(0, 0), swapChainData->swapChainExtent));

  // bind pipeline
  drawCmdBuffers[currentFrameIndex].bindPipeline(
      vk::PipelineBindPoint::eGraphics, *pipeline);

  // bind ubo descriptor
  drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0,
      {*descriptorSets[currentFrameIndex]}, nullptr);

  // bind vertex buffer
  drawCmdBuffers[currentFrameIndex].bindVertexBuffers(
      0, {vertexBuffer->getBuffer()}, {0});

  // bind index buffer
  drawCmdBuffers[currentFrameIndex].bindIndexBuffer(indexBuffer->getBuffer(), 0,
                                                    vk::IndexType::eUint32);

  // draw indexed
  drawCmdBuffers[currentFrameIndex].drawIndexed(indexBuffer->getInstanceCount(),
                                                1, 0, 0, 0);

  // UI overlay draw
  drawUI(drawCmdBuffers[currentFrameIndex]);

  // end renderpass
  drawCmdBuffers[currentFrameIndex].endRenderPass();

  // end command buffer
  drawCmdBuffers[currentFrameIndex].end();
}
void VgeExample::viewChanged() {
  // std::cout << "Call: viewChanged()" << std::endl;
  camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
  // NOTE: moved updating ubo into render() to use frameindex.
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()