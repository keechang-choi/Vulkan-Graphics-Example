#include "animation.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

// std
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_set>

namespace vge {
VgeExample::VgeExample() : VgeBase() { title = "Animation Example"; }
VgeExample::~VgeExample() {}

void VgeExample::initVulkan() {
  // camera setup
  camera.setViewTarget(glm::vec3{0.f, -4.f, -7.f}, glm::vec3{0.f, 0.f, 0.f});
  camera.setPerspectiveProjection(
      glm::radians(60.f),
      static_cast<float>(width) / (static_cast<float>(height) / 2.f), 0.1f,
      256.f);
  // TODO: check coordinate space
  globalUbo.lightPos = glm::vec4(20.f, -10.f, -10.f, 0.f);
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
  setupDynamicUbo();
  createUniformBuffers();
  createDescriptorSetLayout();
  createDescriptorPool();
  createDescriptorSets();
  createPipelines();
  prepared = true;
}
void VgeExample::loadAssets() {
  vgeu::FileLoadingFlags glTFLoadingFlags =
      vgeu::FileLoadingFlagBits::kPreMultiplyVertexColors |
      vgeu::FileLoadingFlagBits::kFlipY;
  std::shared_ptr<vgeu::glTF::Model> fox;
  fox = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  fox->loadFromFile(getAssetsPath() + "/models/fox/Fox.gltf", glTFLoadingFlags);

  {
    ModelInstance& modelInstance = modelInstances.emplace_back();
    modelInstance.model = fox;
    modelInstance.id = modelInstances.size() - 1;
  }

  {
    ModelInstance& modelInstance = modelInstances.emplace_back();
    modelInstance.model = fox;
    modelInstance.id = modelInstances.size() - 1;
  }

  std::shared_ptr<vgeu::glTF::Model> bone = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  bone->loadFromFile(getAssetsPath() + "/models/bone3.gltf",
                     vgeu::FileLoadingFlags{});

  std::vector<std::vector<glm::mat4>> jointMatrices;
  fox->getSkeletonMatrices(jointMatrices);
  for (const auto& jointMatricesEachSkin : jointMatrices) {
    for (const auto& jointMatrix : jointMatricesEachSkin) {
      ModelInstance& modelInstance = modelInstances.emplace_back();
      modelInstance.model = bone;
      modelInstance.id = modelInstances.size() - 1;
      modelInstance.isBone = true;
    }
  }
}
void VgeExample::setupDynamicUbo() {
  dynamicUbo.resize(modelInstances.size());
  dynamicUbo[0].modelMatrix =
      glm::translate(glm::mat4{1.f}, glm::vec3{-3.f, 0.f, 0.f});
  dynamicUbo[0].modelMatrix =
      glm::rotate(dynamicUbo[0].modelMatrix, glm::radians(140.f),
                  glm::vec3{0.f, -1.f, 0.f});
  dynamicUbo[0].modelMatrix =
      glm::scale(dynamicUbo[0].modelMatrix, glm::vec3(.03f));

  dynamicUbo[1].modelMatrix =
      glm::translate(glm::mat4{1.f}, glm::vec3{3.f, 0.f, 0.f});
  // dynamicUbo[1].modelMatrix =
  //     glm::rotate(dynamicUbo[1].modelMatrix, glm::radians(140.f + 180.f),
  //                 glm::vec3{0.f, -1.f, 0.f});
  dynamicUbo[1].modelMatrix =
      glm::scale(dynamicUbo[1].modelMatrix, glm::vec3(.03f));

  // 2~25 -> bones for instance 0
  {
    size_t boneInstanceIdx = 2;
    std::vector<std::vector<glm::mat4>> jointMatrices;
    modelInstances[0].model->getSkeletonMatrices(jointMatrices);
    glm::mat4 boneAxisChange{1.f};
    boneAxisChange[1] = glm::vec4{0.f, 0.f, 1.f, 0.f};
    boneAxisChange[2] = glm::vec4{0.f, 1.f, 0.f, 0.f};
    glm::mat4 flipY{1.f};
    flipY[1][1] = -1.f;
    for (const auto& jointMatricesEachSkin : jointMatrices) {
      for (size_t i = 0; i < jointMatricesEachSkin.size(); i++) {
        const auto& jointMatrix = jointMatricesEachSkin[i];

        dynamicUbo[boneInstanceIdx].modelColor = glm::vec4{1.f, 1.f, 1.f, 1.f};
        dynamicUbo[boneInstanceIdx].modelMatrix =
            dynamicUbo[0].modelMatrix * jointMatrix;
        boneInstanceIdx++;
      }
    }

    {
      // test
      std::cout << "---------test------------" << std::endl;
      glm::quat q0(0.7071054698831242,
                   glm::vec3{-0.7071080924875391, 0.0, 0.0});
      glm::mat4 t1 = glm::translate(
          glm::mat4{1.f}, glm::vec3{0, 26.748403549194336, 42.93817138671875});
      glm::quat q1(0.695481840425441,
                   glm::vec3{0.12769094176175547, -0.6954820192393762,
                             -0.12769022650601444});
      glm::mat4 m1 = t1 * glm::toMat4(q1);
      glm::vec4 head{0, 26.748403549194336, 42.93817138671875, 1.f};
      glm::vec4 tail{0, 23.227, 52.205, 1.f};
      std::cout << glm::to_string(m1 * head) << std::endl;
      std::cout << glm::to_string(m1 * tail) << std::endl;
      std::cout << "---------test------------" << std::endl;
    }
    {
      glm::mat4 joint0{1.f};
      {
        // NOTE: constructor gets wxyz
        // but internal order is xyzw(for make_quat)
        glm::quat q(0.7071054698831242,
                    glm::vec3{-0.7071080924875391, 0.0, 0.0});
        // NOTE:: same as glm::Mat4
        joint0 = glm::toMat4(q);
        // std::cout << glm::to_string(joint0) << std::endl;
      }
      dynamicUbo[2].modelColor = glm::vec4{1.f, 0.f, 0.f, 1.f};
      dynamicUbo[2].modelMatrix =
          dynamicUbo[0].modelMatrix * flipY * joint0 * boneAxisChange;
      glm::mat4 joint1{1.f};
      joint1 = glm::translate(
          joint1, glm::vec3{0, 26.748403549194336, 42.93817138671875});

      glm::quat q(0.695481840425441,
                  glm::vec3{0.12769094176175547, -0.6954820192393762,
                            -0.12769022650601444});
      joint1 = joint1 * glm::toMat4(q);

      std::cout << glm::to_string(joint1) << std::endl;
      std::cout << glm::to_string(boneAxisChange) << std::endl;
      std::cout << glm::to_string(joint1 * boneAxisChange) << std::endl;
      std::cout << glm::to_string(joint0 * boneAxisChange) << std::endl;
      dynamicUbo[3].modelColor = glm::vec4{0.f, 1.f, 0.f, 1.f};
      dynamicUbo[3].modelMatrix =
          dynamicUbo[0].modelMatrix * flipY * joint0 * joint1;
      dynamicUbo[4].modelColor = glm::vec4{0.f, 0.f, 1.f, 1.f};
      dynamicUbo[4].modelMatrix =
          dynamicUbo[0].modelMatrix * flipY * joint0 * joint1 * boneAxisChange;
      dynamicUbo[5].modelColor = glm::vec4{1.f, 1.f, 0.f, 1.f};
      dynamicUbo[5].modelMatrix = dynamicUbo[0].modelMatrix * flipY * joint0 *
                                  glm::toMat4(q) * boneAxisChange;
      dynamicUbo[6].modelColor = glm::vec4{0.f, 1.f, 1.f, 1.f};
      dynamicUbo[6].modelMatrix =
          dynamicUbo[0].modelMatrix * flipY * glm::toMat4(q) * boneAxisChange;

      dynamicUbo[7].modelColor = glm::vec4{1.f, 0.f, 1.f, 1.f};
      dynamicUbo[7].modelMatrix = dynamicUbo[0].modelMatrix * glm::toMat4(q);

      dynamicUbo[8].modelColor = glm::vec4{0.1f, 0.2f, 0.4f, 1.f};
      dynamicUbo[8].modelMatrix =
          dynamicUbo[0].modelMatrix * flipY * glm::toMat4(q);

      dynamicUbo[9].modelColor = glm::vec4{0.1f, 0.4f, 0.2f, 1.f};
      dynamicUbo[9].modelMatrix = dynamicUbo[0].modelMatrix * flipY * joint1;

      dynamicUbo[10].modelColor = glm::vec4{0.8f, 0.3f, 0.2f, 1.f};
      dynamicUbo[10].modelMatrix =
          dynamicUbo[0].modelMatrix * flipY *
          glm::translate(glm::mat4{1.f},
                         glm::vec3{0, 26.748403549194336, 42.93817138671875});
    }
  }
}

void VgeExample::createUniformBuffers() {
  alignedSizeDynamicUboElt = padUniformBufferSize(sizeof(DynamicUboElt));
  // NOTE: prevent move during vector element creation.
  globalUniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
  dynamicUniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    globalUniformBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(GlobalUbo), 1,
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT));
    std::memcpy(globalUniformBuffers[i]->getMappedData(), &globalUbo,
                sizeof(GlobalUbo));

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
    globalUboDescriptorSetLayout =
        vk::raii::DescriptorSetLayout(device, layoutCI);
    setLayouts.push_back(*globalUboDescriptorSetLayout);
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
  setLayouts.push_back(*modelInstances[0].model->descriptorSetLayoutUbo);

  vk::PipelineLayoutCreateInfo pipelineLayoutCI({}, setLayouts);
  pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutCI);
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
    vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                            *globalUboDescriptorSetLayout);
    globalUboDescriptorSets.reserve(MAX_CONCURRENT_FRAMES);
    for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      // NOTE: move descriptor set
      globalUboDescriptorSets.push_back(
          std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
    }
    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(globalUniformBuffers.size());
    std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
    writeDescriptorSets.reserve(globalUniformBuffers.size());
    for (int i = 0; i < globalUniformBuffers.size(); i++) {
      // copy
      bufferInfos.push_back(globalUniformBuffers[i]->descriptorInfo());
      // NOTE: ArrayProxyNoTemporaries has no T rvalue constructor.
      writeDescriptorSets.emplace_back(*globalUboDescriptorSets[i], 0, 0,
                                       vk::DescriptorType::eUniformBuffer,
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
      &dynamicSCI, *pipelineLayout, *renderPass);

  pipelines.phong =
      vk::raii::Pipeline(device, pipelineCache, graphicsPipelineCI);

  if (enabledFeatures.fillModeNonSolid) {
    rasterizationSCI.polygonMode = vk::PolygonMode::eLine;
    vertCode =
        vgeu::readFile(getShadersPath() + "/animation/wireframe.vert.spv");
    fragCode =
        vgeu::readFile(getShadersPath() + "/animation/wireframe.frag.spv");
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
      vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0 /*set 0*/,
      {*globalUboDescriptorSets[currentFrameIndex]}, nullptr);

  // Top view
  {
    drawCmdBuffers[currentFrameIndex].setLineWidth(1.f);
    drawCmdBuffers[currentFrameIndex].setViewport(
        0, vk::Viewport(
               0.0f, 0.0f,
               static_cast<float>(swapChainData->swapChainExtent.width),
               static_cast<float>(swapChainData->swapChainExtent.height) / 2.f,
               0.0f, 1.0f));
    // bind pipeline
    drawCmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eGraphics, *pipelines.phong);

    for (auto& modelInstance : modelInstances) {
      if (modelInstance.isBone) {
        continue;
      }
      // bind dynamic
      drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics, *pipelineLayout, 1 /*set 1*/,
          {*dynamicUboDescriptorSets[currentFrameIndex]},
          alignedSizeDynamicUboElt * modelInstance.id);
      // bind vertex buffer
      // bind index buffer
      modelInstance.model->bindBuffers(drawCmdBuffers[currentFrameIndex]);
      // draw indexed
      modelInstance.model->draw(currentFrameIndex,
                                drawCmdBuffers[currentFrameIndex],
                                vgeu::RenderFlagBits::kBindImages,
                                *pipelineLayout, 2u /*set 2*/, 3u /*set 3*/);
    }
  }
  // Bottom view
  if (enabledFeatures.fillModeNonSolid) {
    drawCmdBuffers[currentFrameIndex].setViewport(
        0, vk::Viewport(
               0.0f,
               static_cast<float>(swapChainData->swapChainExtent.height) / 2.f,
               static_cast<float>(swapChainData->swapChainExtent.width),
               static_cast<float>(swapChainData->swapChainExtent.height) / 2.f,
               0.0f, 1.0f));
    // bind pipeline
    drawCmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eGraphics, *pipelines.wireframe);
    for (auto& modelInstance : modelInstances) {
      // bind dynamic
      drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics, *pipelineLayout, 1 /*set 1*/,
          {*dynamicUboDescriptorSets[currentFrameIndex]},
          alignedSizeDynamicUboElt * modelInstance.id);
      // bind vertex buffer
      // bind index buffer
      modelInstance.model->bindBuffers(drawCmdBuffers[currentFrameIndex]);
      if (enabledFeatures.wideLines) {
        float lineWidth = 1.f;
        if (modelInstance.isBone) {
          lineWidth = 3.f;
        }
        drawCmdBuffers[currentFrameIndex].setLineWidth(lineWidth);
      }
      // draw indexed
      modelInstance.model->draw(currentFrameIndex,
                                drawCmdBuffers[currentFrameIndex],
                                vgeu::RenderFlagBits::kBindImages,
                                *pipelineLayout, 2u /*set 2*/, 3u /*set 3*/);
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
  camera.setAspectRatio(static_cast<float>(width) /
                        (static_cast<float>(height) / 2.f));
  // NOTE: moved updating ubo into render() to use frameindex.
}

void VgeExample::drawSkeleton() {
  // uniforms?
  // TODO: decide use how many uniform buffers.
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

void VgeExample::updateUniforms() {
  // CHECK: ubo update frequency.
  globalUbo.view = camera.getView();
  globalUbo.projection = camera.getProjection();
  globalUbo.inverseView = camera.getInverseView();
  std::memcpy(globalUniformBuffers[currentFrameIndex]->getMappedData(),
              &globalUbo, sizeof(GlobalUbo));

  // update animation joint matrices for each shared model
  {
    std::unordered_set<const vgeu::glTF::Model*> updatedSharedModelSet;
    for (auto& modelInstance : modelInstances) {
      if (updatedSharedModelSet.find(modelInstance.model.get()) !=
          updatedSharedModelSet.end()) {
        continue;
      }
      updatedSharedModelSet.insert(modelInstance.model.get());
      modelInstance.animationTime += frameTimer;
      modelInstance.model->updateAnimation(currentFrameIndex,
                                           modelInstance.animationIndex,
                                           modelInstance.animationTime, true);
    }
  }
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()