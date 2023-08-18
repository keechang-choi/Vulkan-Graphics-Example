#include "particle.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
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
VgeExample::VgeExample() : VgeBase() { title = "Particle Example"; }
VgeExample::~VgeExample() {}

void VgeExample::initVulkan() {
  // camera setup
  camera.setViewTarget(glm::vec3{0.f, -6.f, -10.f}, glm::vec3{0.f, 0.f, 0.f});
  camera.setPerspectiveProjection(
      glm::radians(60.f),
      static_cast<float>(width) / (static_cast<float>(height)), 0.1f, 256.f);
  // NOTE: coordinate space in world
  graphics.globalUbo.lightPos = glm::vec4(20.f, -10.f, -10.f, 0.f);
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
  createDescriptorPool();
  prepareGraphics();
  prepareCompute();
  prepared = true;
}
void VgeExample::prepareGraphics() {
  createStorageBuffers();
  setupDynamicUbo();
  createUniformBuffers();
  createDescriptorSetLayout();
  createDescriptorSets();
  createPipelines();
}

void VgeExample::prepareCompute() {}

void VgeExample::loadAssets() {
  // NOTE: no flip or preTransform for animation and skinning
  vgeu::FileLoadingFlags glTFLoadingFlags =
      vgeu::FileLoadingFlagBits::kPreMultiplyVertexColors;
  // | vgeu::FileLoadingFlagBits::kPreTransformVertices;
  //| vgeu::FileLoadingFlagBits::kFlipY;
  std::shared_ptr<vgeu::glTF::Model> fox;
  fox = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  fox->loadFromFile(getAssetsPath() + "/models/fox-normal/fox-normal.gltf",
                    glTFLoadingFlags);

  {
    ModelInstance modelInstance{};
    modelInstance.model = fox;
    modelInstance.name = "fox1";
    modelInstance.animationIndex = 2;
    addModelInstance(modelInstance);
  }

  {
    ModelInstance modelInstance{};
    modelInstance.model = fox;
    modelInstance.name = "fox1-1";
    modelInstance.animationIndex = 2;
    addModelInstance(modelInstance);
  }
}

void VgeExample::createStorageBuffers() {
  std::vector<glm::vec3> attractor = {
      glm::vec3(5.0f, 0.0f, 0.0f), glm::vec3(-5.0f, 0.0f, 0.0f),
      glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(0.0f, -5.0f, 0.0f),
      glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, -5.0f),
  };

  std::vector<Particle> particles;
  std::default_random_engine rndEngine;
  rndEngine.seed(1111);
  std::normal_distribution<float> normalDist(0.0f, 1.0f);

  for (size_t i = 0; i < attractor.size(); i++) {
    uint32_t numParticlesPerAttractor = numParticles / attractor.size();
    if (i == attractor.size() - 1) {
      numParticlesPerAttractor =
          numParticles - numParticlesPerAttractor * (attractor.size() - 1);
    }
    for (size_t j = 0; j < numParticlesPerAttractor; j++) {
      Particle& particle = particles.emplace_back();
      glm::vec3 position;
      glm::vec3 velocity;
      float mass;
      float colorOffset =
          static_cast<float>(i) / static_cast<float>(attractor.size());
      if (j == 0) {
        position = attractor[i] * 1.5f;
        velocity = glm::vec3{0.f};
        mass = 1000.0f;
      } else {
        position = attractor[i] + glm::vec3{
                                      normalDist(rndEngine),
                                      normalDist(rndEngine),
                                      normalDist(rndEngine),
                                  } * 0.75f;

        velocity =
            glm::vec3{
                normalDist(rndEngine),
                normalDist(rndEngine),
                normalDist(rndEngine),
            } *
            0.025f;
        mass = (normalDist(rndEngine) * 0.5f + 0.5f) * 75.f;
      }
      particle.pos = glm::vec4(position, mass);
      particle.vel = glm::vec4(velocity, colorOffset);
    }
  }

  compute.ubo.particleCount = particles.size();
  // ssbo staging
  {
    vgeu::VgeuBuffer stagingBuffer(
        globalAllocator->getAllocator(), sizeof(Particle), particles.size(),
        vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    std::memcpy(stagingBuffer.getMappedData(), particles.data(),
                stagingBuffer.getBufferSize());
    compute.storageBuffers.resize(MAX_CONCURRENT_FRAMES);
    for (size_t i = 0; i < compute.storageBuffers.size(); i++) {
      compute.storageBuffers[i] = std::make_unique<vgeu::VgeuBuffer>(
          globalAllocator->getAllocator(), sizeof(Particle), particles.size(),
          vk::BufferUsageFlagBits::eVertexBuffer |
              vk::BufferUsageFlagBits::eTransferSrc |
              vk::BufferUsageFlagBits::eStorageBuffer,
          VMA_MEMORY_USAGE_AUTO,
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
              VMA_ALLOCATION_CREATE_MAPPED_BIT);
    }
    vgeu::oneTimeSubmit(
        device, commandPool, queue,
        [&](const vk::raii::CommandBuffer& cmdBuffer) {
          for (size_t i = 0; i < compute.storageBuffers.size(); i++) {
            cmdBuffer.copyBuffer(
                stagingBuffer.getBuffer(),
                compute.storageBuffers[i]->getBuffer(),
                vk::BufferCopy(0, 0, stagingBuffer.getBufferSize()));
          }
          // TODO: pipeline barrier to the compute queue?
        });
  }
  // TODO: vertex binding and attribute descriptions
}

void VgeExample::setupDynamicUbo() {
  const float foxScale = 0.03f;
  glm::vec3 up{0.f, -1.f, 0.f};
  dynamicUbo.resize(modelInstances.size());
  {
    size_t instanceIndex = findInstances("fox1")[0];
    dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, glm::vec3{-3.f, 0.f, 0.f});
    dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        dynamicUbo[instanceIndex].modelMatrix, glm::radians(180.f), up);
    // FlipY manually
    dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{foxScale, -foxScale, foxScale});
    dynamicUbo[instanceIndex].modelColor = glm::vec4{1.0f, 0.f, 0.f, 0.3f};
  }
  {
    size_t instanceIndex = findInstances("fox1-1")[0];
    dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, glm::vec3{3.f, 0.f, 0.f});
    dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        dynamicUbo[instanceIndex].modelMatrix, glm::radians(0.f), up);
    // FlipY manually
    dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{foxScale, -foxScale, foxScale});
    dynamicUbo[instanceIndex].modelColor = glm::vec4{0.0f, 0.f, 1.f, 0.3f};
  }
}

void VgeExample::createUniformBuffers() {
  alignedSizeDynamicUboElt = padUniformBufferSize(sizeof(DynamicUboElt));
  // NOTE: prevent move during vector element creation.
  graphics.globalUniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
  dynamicUniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    graphics.globalUniformBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(GlobalUbo), 1,
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT));
    std::memcpy(graphics.globalUniformBuffers[i]->getMappedData(),
                &graphics.globalUbo, sizeof(GlobalUbo));

    dynamicUniformBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), alignedSizeDynamicUboElt,
        dynamicUbo.size(), vk::BufferUsageFlagBits::eUniformBuffer,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT));
    for (size_t j = 0; j < dynamicUbo.size(); j++) {
      std::memcpy(
          static_cast<char*>(dynamicUniformBuffers[i]->getMappedData()) +
              j * alignedSizeDynamicUboElt,
          &dynamicUbo[j], alignedSizeDynamicUboElt);
    }
  }
}

void VgeExample::createDescriptorSetLayout() {
  std::vector<vk::DescriptorSetLayout> setLayouts;

  // set 0
  {
    vk::DescriptorSetLayoutBinding layoutBinding(
        0, vk::DescriptorType::eUniformBuffer, 1,
        vk::ShaderStageFlagBits::eVertex);
    vk::DescriptorSetLayoutCreateInfo layoutCI({}, 1, &layoutBinding);
    graphics.globalUboDescriptorSetLayout =
        vk::raii::DescriptorSetLayout(device, layoutCI);
    setLayouts.push_back(*graphics.globalUboDescriptorSetLayout);
  }

  // set 1
  {
    vk::DescriptorSetLayoutBinding layoutBinding(
        0, vk::DescriptorType::eUniformBufferDynamic, 1,
        vk::ShaderStageFlagBits::eVertex);
    vk::DescriptorSetLayoutCreateInfo layoutCI({}, 1, &layoutBinding);
    dynamicUboDescriptorSetLayout =
        vk::raii::DescriptorSetLayout(device, layoutCI);
    setLayouts.push_back(*dynamicUboDescriptorSetLayout);
  }

  // set 2
  // TODO: need to improve structure. descriptorSetLayout per model
  setLayouts.push_back(*modelInstances[0].model->descriptorSetLayoutImage);

  // set3
  setLayouts.push_back(*modelInstances[0].model->descriptorSetLayoutUbo);

  vk::PipelineLayoutCreateInfo pipelineLayoutCI({}, setLayouts);
  graphics.pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutCI);
}

void VgeExample::createDescriptorPool() {
  std::vector<vk::DescriptorPoolSize> poolSizes;
  poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer,
                         MAX_CONCURRENT_FRAMES);
  poolSizes.emplace_back(vk::DescriptorType::eUniformBufferDynamic,
                         MAX_CONCURRENT_FRAMES);
  // NOTE: need to check flag
  vk::DescriptorPoolCreateInfo descriptorPoolCI(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      MAX_CONCURRENT_FRAMES * 2, poolSizes);
  descriptorPool = vk::raii::DescriptorPool(device, descriptorPoolCI);
}

void VgeExample::createDescriptorSets() {
  // global UBO
  {
    vk::DescriptorSetAllocateInfo allocInfo(
        *descriptorPool, *graphics.globalUboDescriptorSetLayout);
    graphics.globalUboDescriptorSets.reserve(MAX_CONCURRENT_FRAMES);
    for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      // NOTE: move descriptor set
      graphics.globalUboDescriptorSets.push_back(
          std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
    }
    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(graphics.globalUniformBuffers.size());
    std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
    writeDescriptorSets.reserve(graphics.globalUniformBuffers.size());
    for (int i = 0; i < graphics.globalUniformBuffers.size(); i++) {
      // copy
      bufferInfos.push_back(graphics.globalUniformBuffers[i]->descriptorInfo());
      // NOTE: ArrayProxyNoTemporaries has no T rvalue constructor.
      writeDescriptorSets.emplace_back(*graphics.globalUboDescriptorSets[i], 0,
                                       0, vk::DescriptorType::eUniformBuffer,
                                       nullptr, bufferInfos.back());
    }
    device.updateDescriptorSets(writeDescriptorSets, nullptr);
  }
  // dynamic UBO
  {
    vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                            *dynamicUboDescriptorSetLayout);
    dynamicUboDescriptorSets.reserve(MAX_CONCURRENT_FRAMES);
    for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      dynamicUboDescriptorSets.push_back(
          std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
    }
    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(dynamicUniformBuffers.size());
    std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
    writeDescriptorSets.reserve(dynamicUniformBuffers.size());
    for (int i = 0; i < dynamicUniformBuffers.size(); i++) {
      // NOTE: descriptorBufferInfo range be alignedSizeDynamicUboElt
      bufferInfos.push_back(dynamicUniformBuffers[i]->descriptorInfo(
          alignedSizeDynamicUboElt, 0));
      writeDescriptorSets.emplace_back(
          *dynamicUboDescriptorSets[i], 0, 0,
          vk::DescriptorType::eUniformBufferDynamic, nullptr,
          bufferInfos.back());
    }
    device.updateDescriptorSets(writeDescriptorSets, nullptr);
  }
}

void VgeExample::createPipelines() {
  vk::PipelineVertexInputStateCreateInfo vertexInputSCI =
      vgeu::glTF::Vertex::getPipelineVertexInputState({
          vgeu::glTF::VertexComponent::kPosition,
          vgeu::glTF::VertexComponent::kNormal,
          vgeu::glTF::VertexComponent::kUV,
          vgeu::glTF::VertexComponent::kColor,
          vgeu::glTF::VertexComponent::kJoint0,
          vgeu::glTF::VertexComponent::kWeight0,
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

  std::array<vk::DynamicState, 3> dynamicStates = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor,
      vk::DynamicState::eLineWidth,
  };
  vk::PipelineDynamicStateCreateInfo dynamicSCI(
      vk::PipelineDynamicStateCreateFlags(), dynamicStates);

  auto vertCode =
      vgeu::readFile(getShadersPath() + "/animation/phong.vert.spv");
  auto fragCode =
      vgeu::readFile(getShadersPath() + "/animation/phong.frag.spv");
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
      &dynamicSCI, *graphics.pipelineLayout, *renderPass);

  graphics.pipeline =
      vk::raii::Pipeline(device, pipelineCache, graphicsPipelineCI);
}

void VgeExample::render() {
  if (!prepared) {
    return;
  }
  if (!paused) {
    animationTime += frameTimer;
  }
  draw();
  animationLastTime = animationTime;
}

void VgeExample::draw() {
  // std::cout << "Call: draw() - " << currentFrameIndex << std::endl;
  vk::Result result =
      device.waitForFences(*waitFences[currentFrameIndex], VK_TRUE,
                           std::numeric_limits<uint64_t>::max());
  assert(result != vk::Result::eTimeout && "Timed out: waitFence");

  device.resetFences(*waitFences[currentFrameIndex]);

  prepareFrame();

  // update uniform buffers;
  updateUniforms();

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
      vk::PipelineBindPoint::eGraphics, *graphics.pipelineLayout, 0 /*set 0*/,
      {*graphics.globalUboDescriptorSets[currentFrameIndex]}, nullptr);

  // Top view
  {
    drawCmdBuffers[currentFrameIndex].setLineWidth(1.f);
    drawCmdBuffers[currentFrameIndex].setViewport(
        0,
        vk::Viewport(0.0f, 0.0f,
                     static_cast<float>(swapChainData->swapChainExtent.width),
                     static_cast<float>(swapChainData->swapChainExtent.height),
                     0.0f, 1.0f));
    // bind pipeline
    drawCmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eGraphics, *graphics.pipeline);

    // draw all instances including model based and bones.
    for (size_t instanceIdx = 0; instanceIdx < modelInstances.size();
         instanceIdx++) {
      const auto& modelInstance = modelInstances[instanceIdx];
      if (modelInstance.isBone) {
        continue;
      }
      // bind dynamic
      drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics, *graphics.pipelineLayout,
          1 /*set 1*/, {*dynamicUboDescriptorSets[currentFrameIndex]},
          alignedSizeDynamicUboElt * instanceIdx);
      // bind vertex buffer
      // bind index buffer
      modelInstance.model->bindBuffers(drawCmdBuffers[currentFrameIndex]);
      // draw indexed
      modelInstance.model->draw(
          currentFrameIndex, drawCmdBuffers[currentFrameIndex],
          vgeu::RenderFlagBits::kBindImages, *graphics.pipelineLayout,
          2u /*set 2*/, 3 /*set 3*/);
    }
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
  camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
  // NOTE: moved updating ubo into render() to use frameindex.
}

size_t VgeExample::padUniformBufferSize(size_t originalSize) {
  size_t minUboAlignment =
      physicalDevice.getProperties().limits.minUniformBufferOffsetAlignment;
  size_t alignedSize = originalSize;
  if (minUboAlignment > 0) {
    alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
  }
  return alignedSize;
}

// update UniformBuffers for currentFrameIndex
void VgeExample::updateUniforms() {
  float animationTimer = animationTime - animationLastTime;
  // CHECK: ubo update frequency.
  graphics.globalUbo.view = camera.getView();
  graphics.globalUbo.projection = camera.getProjection();
  graphics.globalUbo.inverseView = camera.getInverseView();
  std::memcpy(graphics.globalUniformBuffers[currentFrameIndex]->getMappedData(),
              &graphics.globalUbo, sizeof(GlobalUbo));

  // model move

  glm::vec3 up{0.f, -1.f, 0.f};
  // deg per sec;
  float rotationVelocity = 30.f;
  {
    size_t instanceIndex = findInstances("fox1")[0];
    dynamicUbo[instanceIndex].modelMatrix =
        glm::rotate(glm::mat4{1.f},
                    glm::radians(rotationVelocity) * animationTimer, up) *
        dynamicUbo[instanceIndex].modelMatrix;
  }
  {
    size_t instanceIndex = findInstances("fox1-1")[0];
    dynamicUbo[instanceIndex].modelMatrix =
        glm::rotate(glm::mat4{1.f},
                    glm::radians(rotationVelocity) * animationTimer, up) *
        dynamicUbo[instanceIndex].modelMatrix;
  }
  // update animation joint matrices for each shared model
  {
    std::unordered_set<const vgeu::glTF::Model*> updatedSharedModelSet;
    for (auto& modelInstance : modelInstances) {
      if (updatedSharedModelSet.find(modelInstance.model.get()) !=
          updatedSharedModelSet.end()) {
        continue;
      }
      updatedSharedModelSet.insert(modelInstance.model.get());
      modelInstance.animationTime += animationTimer;
      modelInstance.model->updateAnimation(currentFrameIndex,
                                           modelInstance.animationIndex,
                                           modelInstance.animationTime, true);
    }
  }

  // update all dynamicUbo elements
  for (size_t j = 0; j < dynamicUbo.size(); j++) {
    std::memcpy(static_cast<char*>(
                    dynamicUniformBuffers[currentFrameIndex]->getMappedData()) +
                    j * alignedSizeDynamicUboElt,
                &dynamicUbo[j], alignedSizeDynamicUboElt);
  }
}

void VgeExample::addModelInstance(const ModelInstance& newInstance) {
  size_t instanceIdx = modelInstances.size();
  modelInstances.push_back(newInstance);
  instanceMap[newInstance.name].push_back(instanceIdx);
}

const std::vector<size_t>& VgeExample::findInstances(const std::string& name) {
  assert(instanceMap.find(name) != instanceMap.end() &&
         "failed to find instance by name.");
  return instanceMap.at(name);
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()