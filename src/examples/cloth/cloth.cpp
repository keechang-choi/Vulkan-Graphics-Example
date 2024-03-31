#include "cloth.hpp"

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

namespace {
float packColor(uint8_t r, uint8_t g, uint8_t b) {
  return r * 1.0f + g * 256.0f + b * 256.0f * 256.0f;
}
glm::vec3 unpackColor(float f) {
  glm::vec3 color;
  color.b = std::floor(f / 256.0f / 256.0f);
  color.g = std::floor((f - color.b * 256.0f * 256.0f) / 256.0f);
  color.r = std::floor(f - color.b * 256.0f * 256.0f - color.g * 256.0f);
  // each field in 0.0 ~ 255.0
  return color;
}

std::optional<glm::vec3> rayPlaneIntersection(glm::vec3 rayStart,
                                              glm::vec3 rayDir,
                                              glm::vec3 planeNormal,
                                              glm::vec3 planePoint) {
  float denom = glm::dot(rayDir, planeNormal);
  if (std::abs(denom) < 1e-3f) {
    return std::nullopt;
  }
  float t = glm::dot((planePoint - rayStart), planeNormal) / denom;
  if (t >= 0) {
    return rayStart + rayDir * t;
  }
  return std::nullopt;
}

std::pair<glm::vec3, glm::vec3> getRayStartAndDir(const glm::vec2 mousePos,
                                                  const glm::vec2 windowSize,
                                                  glm::mat4 inverseProjView) {
  glm::vec2 normalizedMousePos{2.f * mousePos.x / windowSize.x - 1.f,
                               2.f * mousePos.y / windowSize.y - 1.f};

  glm::vec4 rayStart =
      inverseProjView * glm::vec4(normalizedMousePos, 0.f, 1.f);
  rayStart /= rayStart.w;
  glm::vec4 rayEnd = inverseProjView * glm::vec4(normalizedMousePos, 0.1f, 1.f);
  rayEnd /= rayEnd.w;

  glm::vec3 rayDir(glm::normalize(rayEnd - rayStart));
  return std::make_pair(glm::vec3(rayStart), rayDir);
}

}  // namespace
namespace vge {
VgeExample::VgeExample() : VgeBase() { title = "Particle Example"; }
VgeExample::~VgeExample() {}

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

void VgeExample::getEnabledExtensions() {
  enabledDeviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);

  {
    auto features2 =
        physicalDevice
            .getFeatures2<vk::PhysicalDeviceFeatures2,
                          vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT>();
    vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT supportedFeatures =
        features2.get<vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT>();
    if (supportedFeatures.shaderBufferFloat32AtomicAdd) {
      atomicFloatFeatures.shaderBufferFloat32AtomicAdd = true;
    } else if (supportedFeatures.shaderBufferFloat32Atomics) {
      atomicFloatFeatures.shaderBufferFloat32Atomics = true;
    } else {
      std::cerr << "both shaderBufferFloat32AtomicAdd and "
                   "shaderBufferFloat32Atomics not supported.";
    }
  }
  deviceCreatepNextChain = &atomicFloatFeatures;
}

void VgeExample::getEnabledFeatures() {
  enabledFeatures.samplerAnisotropy =
      physicalDevice.getFeatures().samplerAnisotropy;
  enabledFeatures.fillModeNonSolid =
      physicalDevice.getFeatures().fillModeNonSolid;
  enabledFeatures.wideLines = physicalDevice.getFeatures().wideLines;
}

void VgeExample::prepare() {
  VgeBase::prepare();
  graphics.queueFamilyIndex = queueFamilyIndices.graphics;
  compute.queueFamilyIndex = queueFamilyIndices.compute;
  prepareCommon();
  prepareGraphics();
  prepareCompute();
  {
    common.ownershipTransferBufferPtrs.resize(MAX_CONCURRENT_FRAMES);
    compute.calculateBufferPtrs.resize(MAX_CONCURRENT_FRAMES);
    for (auto frameIndex = 0; frameIndex < MAX_CONCURRENT_FRAMES;
         frameIndex++) {
      for (const auto& animatedVertexBuffer :
           common.animatedVertexBuffers[currentFrameIndex]) {
        common.ownershipTransferBufferPtrs[frameIndex].push_back(
            animatedVertexBuffer.get());
      }
      for (size_t instanceIdx = 0; instanceIdx < modelInstances.size();
           instanceIdx++) {
        const auto& modelInstance = modelInstances[instanceIdx];
        if (!modelInstance.clothModel) {
          continue;
        }
        common.ownershipTransferBufferPtrs[frameIndex].push_back(
            modelInstance.clothModel->getRenderSBPtr(frameIndex));
        compute.calculateBufferPtrs[frameIndex].push_back(
            modelInstance.clothModel->getCalculateSBPtr(frameIndex));
      }
    }
  }
  setupClothSSBO();

  prepared = true;
}

void VgeExample::prepareCommon() {
  loadAssets();
  // use model's descriptor set layout
  createDescriptorSetLayout();

  // use modelInstances size
  // TODO: add maximum cloth model size
  createDescriptorPool();

  // NOTE: use global descriptor pool for cloth
  initClothModels();
  // use model instance size
  createStorageBuffers();
  // use model instance size
  setupDynamicUbo();
  // dynamic ubo need to be set
  createUniformBuffers();
  createDescriptorSets();
}

void VgeExample::createStorageBuffers() {
  // animated vertex ssbo wo transfer and mapped ptr
  common.animatedVertexBuffers.resize(MAX_CONCURRENT_FRAMES);
  for (auto i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    common.animatedVertexBuffers[i].reserve(modelInstances.size());
    for (size_t j = 0; j < modelInstances.size(); j++) {
      common.animatedVertexBuffers[i].push_back(
          std::move(std::make_unique<vgeu::VgeuBuffer>(
              globalAllocator->getAllocator(), sizeof(AnimatedVertex),
              modelInstances[j].getVertexCount(),
              vk::BufferUsageFlagBits::eStorageBuffer |
                  vk::BufferUsageFlagBits::eVertexBuffer,
              VMA_MEMORY_USAGE_AUTO,
              VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT)));
    }
  }
}

void VgeExample::createUniformBuffers() {
  common.alignedSizeDynamicUboElt =
      vgeu::padBufferSize(physicalDevice, sizeof(DynamicUboElt), true);
  common.dynamicUniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    common.dynamicUniformBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), common.alignedSizeDynamicUboElt,
        common.dynamicUbo.size(), vk::BufferUsageFlagBits::eUniformBuffer,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT));
    for (size_t j = 0; j < common.dynamicUbo.size(); j++) {
      std::memcpy(
          static_cast<char*>(common.dynamicUniformBuffers[i]->getMappedData()) +
              j * common.alignedSizeDynamicUboElt,
          &common.dynamicUbo[j], common.alignedSizeDynamicUboElt);
    }
  }
}

void VgeExample::createDescriptorSetLayout() {
  {
    std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
    layoutBindings.emplace_back(0 /*binding*/,
                                vk::DescriptorType::eUniformBufferDynamic, 1,
                                vk::ShaderStageFlagBits::eAll);
    vk::DescriptorSetLayoutCreateInfo layoutCI({}, layoutBindings);
    common.dynamicUboDescriptorSetLayout =
        vk::raii::DescriptorSetLayout(device, layoutCI);
  }

  // descriptor set layout for cloth vertex
  // descriptor set layout for cloth constraints
  // since it needs to be created before load Asset,
  // but load Asset shoud be done before create compute descriptor set layout
  // TODO: pull out model descriptor set layout from the model class
  // and re-order load asset call after create descriptor set layout
  {
    std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
    // calculation particles
    layoutBindings.emplace_back(0 /*binding*/,
                                vk::DescriptorType::eStorageBuffer, 1,
                                vk::ShaderStageFlagBits::eCompute);
    // render particles
    layoutBindings.emplace_back(1 /*binding*/,
                                vk::DescriptorType::eStorageBuffer, 1,
                                vk::ShaderStageFlagBits::eAll);
    // TODO: fix descriptor pool size
    //  calculation particles prev frame
    layoutBindings.emplace_back(2 /*binding*/,
                                vk::DescriptorType::eStorageBuffer, 1,
                                vk::ShaderStageFlagBits::eCompute);
    vk::DescriptorSetLayoutCreateInfo layoutCI({}, layoutBindings);
    common.particleDescriptorSetLayout =
        vk::raii::DescriptorSetLayout(device, layoutCI);
  }

  {
    std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
    layoutBindings.emplace_back(0 /*binding*/,
                                vk::DescriptorType::eStorageBuffer, 1,
                                vk::ShaderStageFlagBits::eCompute);
    vk::DescriptorSetLayoutCreateInfo layoutCI({}, layoutBindings);
    compute.constraintDescriptorSetLayout =
        vk::raii::DescriptorSetLayout(device, layoutCI);
  }
}

void VgeExample::createDescriptorSets() {
  vk::DescriptorSetAllocateInfo allocInfo(
      *descriptorPool, *common.dynamicUboDescriptorSetLayout);
  common.dynamicUboDescriptorSets.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    common.dynamicUboDescriptorSets.push_back(
        std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
  }
  std::vector<vk::DescriptorBufferInfo> bufferInfos;
  bufferInfos.reserve(common.dynamicUniformBuffers.size());
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  writeDescriptorSets.reserve(common.dynamicUniformBuffers.size());
  for (int i = 0; i < common.dynamicUniformBuffers.size(); i++) {
    // NOTE: descriptorBufferInfo range be alignedSizeDynamicUboElt
    bufferInfos.push_back(common.dynamicUniformBuffers[i]->descriptorInfo(
        common.alignedSizeDynamicUboElt, 0));
    writeDescriptorSets.emplace_back(*common.dynamicUboDescriptorSets[i], 0, 0,
                                     vk::DescriptorType::eUniformBufferDynamic,
                                     nullptr, bufferInfos.back());
  }
  device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

void VgeExample::initClothModels() {
  vgeu::FileLoadingFlags glTFLoadingFlags =
      vgeu::FileLoadingFlagBits::kPreMultiplyVertexColors;
  std::vector<vgeu::glTF::Vertex> modelVertices;
  std::vector<uint32_t> indices;
  // loading cloth models
  std::shared_ptr<vgeu::glTF::Model> koreanFlag;
  koreanFlag = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  koreanFlag->additionalBufferUsageFlags =
      vk::BufferUsageFlagBits::eStorageBuffer;
  koreanFlag->loadFromFile(getAssetsPath() + "/models/koreanFlag/k-flag.gltf",
                           glTFLoadingFlags, 1.0f, &modelVertices, &indices);

  {
    ModelInstance modelInstance{};
    modelInstance.model = koreanFlag;
    modelInstance.name = "koreanFlag1";
    auto& clothModel = modelInstance.clothModel;
    clothModel = std::make_unique<Cloth>(
        device, globalAllocator->getAllocator(), queue,
        graphics.queueFamilyIndex, compute.queueFamilyIndex, commandPool,
        descriptorPool, common.particleDescriptorSetLayout,
        compute.constraintDescriptorSetLayout, MAX_CONCURRENT_FRAMES);

    float kFlagScale = 10.f;
    glm::mat4 translateMat =
        glm::translate(glm::mat4{1.f}, glm::vec3{0.f, -10.f, 0.f});
    glm::mat4 rotateMat{1.f};
    // FlipY manually
    // NOTE: not only flip Y, also need flip z to preserve orientation
    glm::mat4 scaleMat = glm::scale(
        glm::mat4{1.f}, glm::vec3{kFlagScale, -kFlagScale, -kFlagScale});

    clothModel->initParticlesData(modelVertices, indices, translateMat,
                                  rotateMat, scaleMat);
    // model specific infos
    clothModel->initDistConstraintsData(150, 100);

    addModelInstance(std::move(modelInstance));
  }
}
void VgeExample::setupClothSSBO() {
  // NOTE: cloth initialize use model matrix as initial transform
  updateDynamicUbo();
  vgeu::oneTimeSubmit(
      device, compute.cmdPool, compute.queue,
      [&](const vk::raii::CommandBuffer& cmdBuffer) {
        for (auto frameIndex = 0; frameIndex < MAX_CONCURRENT_FRAMES;
             frameIndex++) {
          // acquire from transfer
          vgeu::addQueueFamilyOwnershipTransferBarriers(
              graphics.queueFamilyIndex, compute.queueFamilyIndex, cmdBuffer,
              compute.calculateBufferPtrs[frameIndex], vk::AccessFlags{},
              vk::AccessFlagBits::eShaderWrite,
              vk::PipelineStageFlagBits::eTopOfPipe,
              vk::PipelineStageFlagBits::eComputeShader);

          for (size_t instanceIdx = 0; instanceIdx < modelInstances.size();
               instanceIdx++) {
            const auto& modelInstance = modelInstances[instanceIdx];
            if (!modelInstance.clothModel) {
              continue;
            }
            // NOTE: not using but for validation errors
            cmdBuffer.pushConstants<ComputePushConstantsData>(
                *compute.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                compute.pc);
            cmdBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, *compute.pipelineLayout,
                0 /*set*/, *compute.descriptorSets[frameIndex], nullptr);
            cmdBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, *compute.pipelineLayout,
                3 /*set*/, *compute.skinDescriptorSets[frameIndex][instanceIdx],
                nullptr);
            cmdBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, *compute.pipelineLayout,
                5 /*set*/,
                modelInstance.clothModel->getConstraintDescriptorSet(),
                nullptr);

            cmdBuffer.bindPipeline(
                vk::PipelineBindPoint::eCompute,
                *compute.pipelines.pipelinesCloth[static_cast<uint32_t>(
                    ComputeType::kInitializeParticles)]);
            // NOTE: bind imported cloth model
            modelInstance.model->bindSSBO(cmdBuffer, *compute.pipelineLayout,
                                          1 /*set*/);
            cmdBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, *compute.pipelineLayout,
                2 /*set*/, {*common.dynamicUboDescriptorSets[frameIndex]},
                common.alignedSizeDynamicUboElt * instanceIdx);
            cmdBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eCompute, *compute.pipelineLayout,
                4 /*set*/,
                modelInstance.clothModel->getParticleDescriptorSet(frameIndex),
                nullptr);
            cmdBuffer.dispatch(modelInstance.clothModel->getNumTriangles() * 3 /
                                       sharedDataSize +
                                   1,
                               1, 1);
          }
        }
      });
  // remove initial model matrix after setup in compute shader.
  for (size_t instanceIdx = 0; instanceIdx < modelInstances.size();
       instanceIdx++) {
    const auto& modelInstance = modelInstances[instanceIdx];
    if (!modelInstance.clothModel) {
      continue;
    }
    common.dynamicUbo[instanceIdx].modelMatrix = glm::mat4{1.f};
  }
  // compute initial rest length from model
  vgeu::oneTimeSubmit(
      device, compute.cmdPool, compute.queue,
      [&](const vk::raii::CommandBuffer& cmdBuffer) {
        for (size_t instanceIdx = 0; instanceIdx < modelInstances.size();
             instanceIdx++) {
          const auto& modelInstance = modelInstances[instanceIdx];
          if (!modelInstance.clothModel) {
            continue;
          }
          // NOTE: not using but for validation errors
          cmdBuffer.pushConstants<ComputePushConstantsData>(
              *compute.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
              compute.pc);
          cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                       *compute.pipelineLayout, 0 /*set*/,
                                       *compute.descriptorSets[0], nullptr);
          modelInstance.model->bindSSBO(cmdBuffer, *compute.pipelineLayout,
                                        1 /*set*/);
          cmdBuffer.bindDescriptorSets(
              vk::PipelineBindPoint::eCompute, *compute.pipelineLayout,
              2 /*set*/, {*common.dynamicUboDescriptorSets[0]},
              common.alignedSizeDynamicUboElt * instanceIdx);
          cmdBuffer.bindDescriptorSets(
              vk::PipelineBindPoint::eCompute, *compute.pipelineLayout,
              3 /*set*/, *compute.skinDescriptorSets[0][instanceIdx], nullptr);

          cmdBuffer.bindPipeline(
              vk::PipelineBindPoint::eCompute,
              *compute.pipelines.pipelinesCloth[static_cast<uint32_t>(
                  ComputeType::kInitializeConstraints)]);

          cmdBuffer.bindDescriptorSets(
              vk::PipelineBindPoint::eCompute, *compute.pipelineLayout,
              4 /*set*/, modelInstance.clothModel->getParticleDescriptorSet(0),
              nullptr);
          cmdBuffer.bindDescriptorSets(
              vk::PipelineBindPoint::eCompute, *compute.pipelineLayout,
              5 /*set*/, modelInstance.clothModel->getConstraintDescriptorSet(),
              nullptr);
          cmdBuffer.dispatch(
              modelInstance.clothModel->getNumConstraints() / sharedDataSize +
                  1,
              1, 1);
        }
      });
}

void VgeExample::prepareGraphics() {
  createGraphicsUniformBuffers();
  createGraphicsDescriptorSetLayout();
  createGraphicsDescriptorSets();
  createVertexSCI();
  createGraphicsPipelines();
}

void VgeExample::createComputeUniformBuffers() {
  // compute UBO
  compute.uniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    compute.uniformBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(compute.ubo), 1,
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT));
    std::memcpy(compute.uniformBuffers[i]->getMappedData(), &compute.ubo,
                sizeof(compute.ubo));
  }
}

void VgeExample::createComputeDescriptorSetLayout() {
  std::vector<vk::DescriptorSetLayout> setLayouts;

  // set 0 ubo
  {
    std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
    layoutBindings.emplace_back(0 /* binding */,
                                vk::DescriptorType::eUniformBuffer, 1,
                                vk::ShaderStageFlagBits::eCompute);
    vk::DescriptorSetLayoutCreateInfo layoutCI(
        vk::DescriptorSetLayoutCreateFlags{}, layoutBindings);
    compute.descriptorSetLayout =
        vk::raii::DescriptorSetLayout(device, layoutCI);
    setLayouts.push_back(*compute.descriptorSetLayout);
  }

  // set 1 input model vertex ssbo
  setLayouts.push_back(*modelInstances[0].model->descriptorSetLayoutVertex);

  // set 2 dynamic ubo
  { setLayouts.push_back(*common.dynamicUboDescriptorSetLayout); }

  // set 3 skin ssbo
  {
    std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
    // skin SSBO for all model instances
    layoutBindings.emplace_back(0 /*binding*/,
                                vk::DescriptorType::eStorageBuffer, 1,
                                vk::ShaderStageFlagBits::eCompute);
    // animated vertex ssbo for all model instances
    layoutBindings.emplace_back(1 /*binding*/,
                                vk::DescriptorType::eStorageBuffer, 1,
                                vk::ShaderStageFlagBits::eCompute);
    vk::DescriptorSetLayoutCreateInfo layoutCI({}, layoutBindings);
    compute.skinDescriptorSetLayout =
        vk::raii::DescriptorSetLayout(device, layoutCI);
    setLayouts.push_back(*compute.skinDescriptorSetLayout);
  }

  // set 4 particle ssbo
  setLayouts.push_back(*common.particleDescriptorSetLayout);
  // set 5 constraint ssbo
  setLayouts.push_back(*compute.constraintDescriptorSetLayout);

  // push constants
  vk::PushConstantRange pcRange(vk::ShaderStageFlagBits::eCompute, 0u,
                                sizeof(ComputePushConstantsData));
  vk::PipelineLayoutCreateInfo pipelineLayoutCI({}, setLayouts, pcRange);
  compute.pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutCI);
}

void VgeExample::createComputeDescriptorSets() {
  // ubo descriptor set
  {
    vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                            *compute.descriptorSetLayout);
    compute.descriptorSets.reserve(MAX_CONCURRENT_FRAMES);
    for (size_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      compute.descriptorSets.push_back(
          std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
    }

    std::vector<vk::DescriptorBufferInfo> uniformBufferInfos;
    uniformBufferInfos.reserve(compute.descriptorSets.size());
    for (size_t i = 0; i < compute.descriptorSets.size(); i++) {
      uniformBufferInfos.push_back(compute.uniformBuffers[i]->descriptorInfo());
    }
    std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
    writeDescriptorSets.reserve(compute.descriptorSets.size());
    for (int i = 0; i < compute.descriptorSets.size(); i++) {
      writeDescriptorSets.emplace_back(
          *compute.descriptorSets[i], 0 /*binding*/, 0,
          vk::DescriptorType::eUniformBuffer, nullptr, uniformBufferInfos[i]);
    }
    device.updateDescriptorSets(writeDescriptorSets, nullptr);
  }

  // skin descriptor set
  {
    vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                            *compute.skinDescriptorSetLayout);
    compute.skinDescriptorSets.resize(MAX_CONCURRENT_FRAMES);
    for (size_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      compute.skinDescriptorSets[i].reserve(modelInstances.size());
      for (size_t j = 0; j < modelInstances.size(); j++) {
        compute.skinDescriptorSets[i].push_back(
            std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
      }
    }

    std::vector<std::vector<vk::DescriptorBufferInfo>> skinMatricesBufferInfos;
    skinMatricesBufferInfos.resize(compute.skinDescriptorSets.size());
    std::vector<std::vector<vk::DescriptorBufferInfo>>
        animatedVertexBufferInfos;
    animatedVertexBufferInfos.resize(compute.skinDescriptorSets.size());
    for (size_t i = 0; i < compute.skinDescriptorSets.size(); i++) {
      skinMatricesBufferInfos.reserve(compute.skinDescriptorSets[i].size());
      animatedVertexBufferInfos.reserve(compute.skinDescriptorSets[i].size());
      for (size_t j = 0; j < compute.skinDescriptorSets[i].size(); j++) {
        skinMatricesBufferInfos[i].push_back(
            compute.skinMatricesBuffers[i][j]->descriptorInfo());
        animatedVertexBufferInfos[i].push_back(
            common.animatedVertexBuffers[i][j]->descriptorInfo());
      }
    }
    std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
    writeDescriptorSets.reserve(compute.skinDescriptorSets.size() *
                                compute.skinDescriptorSets[0].size());
    for (size_t i = 0; i < compute.skinDescriptorSets.size(); i++) {
      for (size_t j = 0; j < compute.skinDescriptorSets[i].size(); j++) {
        writeDescriptorSets.emplace_back(
            *compute.skinDescriptorSets[i][j], 0 /*binding*/, 0,
            vk::DescriptorType::eStorageBuffer, nullptr,
            skinMatricesBufferInfos[i][j]);
        writeDescriptorSets.emplace_back(
            *compute.skinDescriptorSets[i][j], 1 /*binding*/, 0,
            vk::DescriptorType::eStorageBuffer, nullptr,
            animatedVertexBufferInfos[i][j]);
      }
    }
    device.updateDescriptorSets(writeDescriptorSets, nullptr);
  }
}

void VgeExample::createComputePipelines() {
  uint32_t maxComputeSharedMemorySize =
      physicalDevice.getProperties().limits.maxComputeSharedMemorySize;

  sharedDataSize = std::min(
      desiredSharedDataSize,
      static_cast<uint32_t>(maxComputeSharedMemorySize / sizeof(glm::vec4)));

  collisionWorkGroupSize = std::min(
      desiredCollisionWorkGroupSize,
      physicalDevice.getProperties().limits.maxComputeWorkGroupSize[2]);
  // NOTE x*y*1 <= 1024 (maxComputeWorkGroupInvocations)
  assert(sharedDataSize * collisionWorkGroupSize <=
         physicalDevice.getProperties().limits.maxComputeWorkGroupInvocations);

  SpecializationData specializationData{};
  specializationData.sharedDataSize = sharedDataSize;
  specializationData.computeType = 0u;
  specializationData.localSizeX = sharedDataSize;
  specializationData.localSizeY = 1u;
  specializationData.localSizeZ = 1u;

  std::vector<vk::SpecializationMapEntry> specializationMapEntries;
  specializationMapEntries.emplace_back(
      0u, offsetof(SpecializationData, sharedDataSize), sizeof(uint32_t));
  specializationMapEntries.emplace_back(
      1u, offsetof(SpecializationData, computeType), sizeof(uint32_t));
  specializationMapEntries.emplace_back(
      2u, offsetof(SpecializationData, localSizeX), sizeof(uint32_t));
  specializationMapEntries.emplace_back(
      3u, offsetof(SpecializationData, localSizeY), sizeof(uint32_t));
  specializationMapEntries.emplace_back(
      4u, offsetof(SpecializationData, localSizeZ), sizeof(uint32_t));

  {
    // compute animation
    auto compCode =
        vgeu::readFile(getShadersPath() + "/cloth/model_animate.comp.spv");
    vk::raii::ShaderModule compIntegrateShaderModule =
        vgeu::createShaderModule(device, compCode);
    vk::SpecializationInfo specializationInfo(
        specializationMapEntries,
        vk::ArrayProxyNoTemporaries<const SpecializationData>(
            specializationData));
    vk::PipelineShaderStageCreateInfo computeShaderStageCI(
        vk::PipelineShaderStageCreateFlags{}, vk::ShaderStageFlagBits::eCompute,
        *compIntegrateShaderModule, "main", &specializationInfo);
    vk::ComputePipelineCreateInfo computePipelineCI(vk::PipelineCreateFlags{},
                                                    computeShaderStageCI,
                                                    *compute.pipelineLayout);
    compute.pipelines.pipelineModelAnimate =
        vk::raii::Pipeline(device, pipelineCache, computePipelineCI);
  }
  {
    vk::raii::ShaderModule compClothShaderModule = nullptr;
    if (atomicFloatFeatures.shaderBufferFloat32AtomicAdd) {
      auto compClothCode =
          vgeu::readFile(getShadersPath() + "/cloth/cloth.comp.spv");
      compClothShaderModule = vgeu::createShaderModule(device, compClothCode);
    } else {
      // TODO: resolve duplicated codes
      auto compClothCode = vgeu::readFile(
          getShadersPath() + "/cloth/cloth_no_atomic_add.comp.spv");
      compClothShaderModule = vgeu::createShaderModule(device, compClothCode);
    }

    // TODO: change specialization data for each type
    for (auto i = 0; i <= static_cast<uint32_t>(ComputeType::kUpdateNormals);
         i++) {
      // compute cloth
      specializationData.computeType = i;
      switch (static_cast<ComputeType>(i)) {
        case ComputeType::kSolveCollision:
          specializationData.localSizeX = sharedDataSize;
          specializationData.localSizeY = collisionWorkGroupSize;
          specializationData.localSizeZ = 1u;
          break;
        default:
          specializationData.localSizeX = sharedDataSize;
          specializationData.localSizeY = 1;
          specializationData.localSizeZ = 1;
      }
      vk::SpecializationInfo specializationInfo(
          specializationMapEntries,
          vk::ArrayProxyNoTemporaries<const SpecializationData>(
              specializationData));
      vk::PipelineShaderStageCreateInfo computeShaderStageCI(
          vk::PipelineShaderStageCreateFlags{},
          vk::ShaderStageFlagBits::eCompute, *compClothShaderModule, "main",
          &specializationInfo);
      vk::ComputePipelineCreateInfo computePipelineCI(vk::PipelineCreateFlags{},
                                                      computeShaderStageCI,
                                                      *compute.pipelineLayout);
      compute.pipelines.pipelinesCloth.emplace_back(device, pipelineCache,
                                                    computePipelineCI);
    }
  }
}

void VgeExample::prepareCompute() {
  // create ubo
  createComputeUniformBuffers();
  // create ssbo
  createComputeStorageBuffers();
  // create queue
  compute.queue = vk::raii::Queue(device, compute.queueFamilyIndex, 0);
  // create descriptorSetLayout
  // create pipelineLayout
  createComputeDescriptorSetLayout();

  // dynamic UBO descriptorSet -> common
  // skin ssbo descriptorSets
  createComputeDescriptorSets();

  // create pipelines
  createComputePipelines();

  // create commandPool
  {
    vk::CommandPoolCreateInfo cmdPoolCI(
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        compute.queueFamilyIndex);
    compute.cmdPool = vk::raii::CommandPool(device, cmdPoolCI);
  }

  // create commandBuffer
  {
    vk::CommandBufferAllocateInfo cmdBufferAI(*compute.cmdPool,
                                              vk::CommandBufferLevel::ePrimary,
                                              MAX_CONCURRENT_FRAMES);
    compute.cmdBuffers = vk::raii::CommandBuffers(device, cmdBufferAI);
    compute.firstCompute.resize(MAX_CONCURRENT_FRAMES);
    for (auto i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      compute.firstCompute[i] = true;
    }
  }

  // create semaphore for compute-graphics sync
  {
    std::vector<vk::Semaphore> semaphoresToSignal;
    semaphoresToSignal.reserve(MAX_CONCURRENT_FRAMES);
    compute.semaphores.ready.reserve(MAX_CONCURRENT_FRAMES);
    for (size_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      vk::raii::Semaphore& semaphore = compute.semaphores.ready.emplace_back(
          device, vk::SemaphoreCreateInfo());
      semaphoresToSignal.push_back(*semaphore);
    }
    // initial signaled
    vk::SubmitInfo submitInfo({}, {}, {}, semaphoresToSignal);
    compute.queue.submit(submitInfo);
    compute.queue.waitIdle();
  }
  {
    compute.semaphores.complete.reserve(MAX_CONCURRENT_FRAMES);
    for (size_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      compute.semaphores.complete.emplace_back(device,
                                               vk::SemaphoreCreateInfo());
    }
  }
}

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
  // NOTE: additional flag for compute animation
  fox->additionalBufferUsageFlags = vk::BufferUsageFlagBits::eStorageBuffer;
  fox->loadFromFile(getAssetsPath() + "/models/fox-normal/fox-normal.gltf",
                    glTFLoadingFlags);
  {
    ModelInstance modelInstance{};
    modelInstance.model = fox;
    modelInstance.name = "fox0";
    modelInstance.animationIndex = 0;
    addModelInstance(std::move(modelInstance));
  }

  std::shared_ptr<vgeu::glTF::Model> fox1;
  fox1 = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  fox1->additionalBufferUsageFlags = vk::BufferUsageFlagBits::eStorageBuffer;
  fox1->loadFromFile(getAssetsPath() + "/models/fox-normal/fox-normal.gltf",
                     glTFLoadingFlags);
  {
    ModelInstance modelInstance{};
    modelInstance.model = fox1;
    modelInstance.name = "fox1";
    modelInstance.animationIndex = 2;
    addModelInstance(std::move(modelInstance));
  }

  // NOTE: different animation to fox1
  std::shared_ptr<vgeu::glTF::Model> fox2;
  fox2 = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  fox2->additionalBufferUsageFlags = vk::BufferUsageFlagBits::eStorageBuffer;
  fox2->loadFromFile(getAssetsPath() + "/models/fox-normal/fox-normal.gltf",
                     glTFLoadingFlags);
  {
    ModelInstance modelInstance{};
    modelInstance.model = fox2;
    modelInstance.name = "fox2";
    modelInstance.animationIndex = 1;
    addModelInstance(std::move(modelInstance));
  }

  std::shared_ptr<vgeu::glTF::Model> apple;
  apple = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  apple->additionalBufferUsageFlags = vk::BufferUsageFlagBits::eStorageBuffer;
  apple->loadFromFile(getAssetsPath() + "/models/apple/food_apple_01_4k.gltf",
                      glTFLoadingFlags);
  {
    ModelInstance modelInstance{};
    modelInstance.model = apple;
    modelInstance.name = "apple1";
    addModelInstance(std::move(modelInstance));
  }

  std::shared_ptr<SimpleModel> circle = std::make_shared<SimpleModel>(
      device, globalAllocator->getAllocator(), queue, commandPool);
  circle->setNgon(32, {1.0f, 1.0f, 1.0f, 1.f});

  std::shared_ptr<SimpleModel> rectLines = std::make_shared<SimpleModel>(
      device, globalAllocator->getAllocator(), queue, commandPool);
  {
    std::vector<glm::vec4> positions{
        {0.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 0.0f, 1.0f},
    };
    std::vector<uint32_t> indices{0, 1, 1, 2, 2, 3, 3, 0};
    rectLines->setLineList(positions, indices, {1.f, 1.f, 1.f, 1.f});
  }
  for (auto i = 1; i <= 4; i++) {
    ModelInstance modelInstance{};
    modelInstance.simpleModel = rectLines;
    modelInstance.name = "rectLines" + std::to_string(i);
    addModelInstance(std::move(modelInstance));
  }

  std::shared_ptr<SimpleModel> circleLines = std::make_shared<SimpleModel>(
      device, globalAllocator->getAllocator(), queue, commandPool);
  {
    std::vector<glm::vec4> positions;
    std::vector<uint32_t> indices;
    int n = 64;
    for (auto i = 0; i < n; i++) {
      auto& pos = positions.emplace_back();
      pos.x = cos(glm::two_pi<float>() / static_cast<float>(n) *
                  static_cast<float>(i));
      pos.y = sin(glm::two_pi<float>() / static_cast<float>(n) *
                  static_cast<float>(i));
      indices.push_back(i);
      indices.push_back((i + 1) % n);
    }
    circleLines->setLineList(positions, indices, {1.f, 1.f, 1.f, 1.f});
  }

  std::shared_ptr<SimpleModel> singleLine = std::make_shared<SimpleModel>(
      device, globalAllocator->getAllocator(), queue, commandPool);
  {
    std::vector<glm::vec4> positions{
        {0.0f, 0.0f, 0.0f, 1.0f},
        {0.0f, 0.0f, 1.0f, 1.0f},
    };
    std::vector<uint32_t> indices{0, 1};
    singleLine->setLineList(positions, indices, {1.f, 1.f, 1.f, 1.f});
  }

  // NOTE: facing +z ccw
  std::shared_ptr<SimpleModel> centerCircle = std::make_shared<SimpleModel>(
      device, globalAllocator->getAllocator(), queue, commandPool);
  centerCircle->setNgon(6, {1.0f, 1.0f, 1.0f, 1.f}, true);

  std::shared_ptr<SimpleModel> quad = std::make_shared<SimpleModel>(
      device, globalAllocator->getAllocator(), queue, commandPool);
  quad->setNgon(4, {0.5f, 0.5f, 0.5f, 0.f});
  {
    ModelInstance modelInstance{};
    modelInstance.simpleModel = quad;
    modelInstance.name = "quad1";
    addModelInstance(std::move(modelInstance));
  }
  {
    ModelInstance modelInstance{};
    modelInstance.simpleModel = quad;
    modelInstance.name = "quad2";
    addModelInstance(std::move(modelInstance));
  }
}

void VgeExample::createComputeStorageBuffers() {
  // use loaded model to create skin ssbo

  compute.skinMatricesData.resize(modelInstances.size());
  // TODO: optimize for models without skin
  for (size_t i = 0; i < modelInstances.size(); i++) {
    if (modelInstances[i].model)
      modelInstances[i].model->getSkinMatrices(compute.skinMatricesData[i]);
    else {
      compute.skinMatricesData[i].resize(1);
    }
  }
  compute.skinMatricesBuffers.resize(MAX_CONCURRENT_FRAMES);
  for (auto i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    compute.skinMatricesBuffers[i].reserve(modelInstances.size());
    for (size_t j = 0; j < modelInstances.size(); j++) {
      // mapped
      compute.skinMatricesBuffers[i].push_back(
          std::move(std::make_unique<vgeu::VgeuBuffer>(
              globalAllocator->getAllocator(),
              sizeof(vgeu::glTF::MeshMatricesData),
              compute.skinMatricesData[j].size(),
              vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_AUTO,
              VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                  VMA_ALLOCATION_CREATE_MAPPED_BIT)));
    }
  }
}

void VgeExample::createVertexSCI() {
  // vertex binding and attribute descriptions
  graphics.simpleVertexInfos.bindingDescriptions.emplace_back(
      0 /*binding*/, sizeof(SimpleModel::Vertex), vk::VertexInputRate::eVertex);

  graphics.simpleVertexInfos.attributeDescriptions.emplace_back(
      0 /*location*/, 0 /* binding */, vk::Format::eR32G32B32A32Sfloat,
      offsetof(SimpleModel::Vertex, pos));
  graphics.simpleVertexInfos.attributeDescriptions.emplace_back(
      1 /*location*/, 0 /* binding */, vk::Format::eR32G32B32A32Sfloat,
      offsetof(SimpleModel::Vertex, normal));
  graphics.simpleVertexInfos.attributeDescriptions.emplace_back(
      2 /*location*/, 0 /* binding */, vk::Format::eR32G32B32A32Sfloat,
      offsetof(SimpleModel::Vertex, color));
  graphics.simpleVertexInfos.attributeDescriptions.emplace_back(
      3 /*location*/, 0 /* binding */, vk::Format::eR32G32Sfloat,
      offsetof(SimpleModel::Vertex, uv));

  graphics.simpleVertexInfos.vertexInputSCI =
      vk::PipelineVertexInputStateCreateInfo(
          vk::PipelineVertexInputStateCreateFlags{},
          graphics.simpleVertexInfos.bindingDescriptions,
          graphics.simpleVertexInfos.attributeDescriptions);

  // vertex binding and attribute descriptions
  graphics.animatedVertexInfos.bindingDescriptions.emplace_back(
      0 /*binding*/, sizeof(AnimatedVertex), vk::VertexInputRate::eVertex);

  graphics.animatedVertexInfos.attributeDescriptions.emplace_back(
      0 /*location*/, 0 /* binding */, vk::Format::eR32G32B32A32Sfloat,
      offsetof(AnimatedVertex, pos));
  graphics.animatedVertexInfos.attributeDescriptions.emplace_back(
      1 /*location*/, 0 /* binding */, vk::Format::eR32G32B32A32Sfloat,
      offsetof(AnimatedVertex, normal));
  graphics.animatedVertexInfos.attributeDescriptions.emplace_back(
      2 /*location*/, 0 /* binding */, vk::Format::eR32G32B32A32Sfloat,
      offsetof(AnimatedVertex, color));
  graphics.animatedVertexInfos.attributeDescriptions.emplace_back(
      3 /*location*/, 0 /* binding */, vk::Format::eR32G32B32A32Sfloat,
      offsetof(AnimatedVertex, tangent));
  graphics.animatedVertexInfos.attributeDescriptions.emplace_back(
      4 /*location*/, 0 /* binding */, vk::Format::eR32G32Sfloat,
      offsetof(AnimatedVertex, uv));

  graphics.animatedVertexInfos.vertexInputSCI =
      vk::PipelineVertexInputStateCreateInfo(
          vk::PipelineVertexInputStateCreateFlags{},
          graphics.animatedVertexInfos.bindingDescriptions,
          graphics.animatedVertexInfos.attributeDescriptions);

  // vertex binding and attribute descriptions
  graphics.clothVertexInfos.bindingDescriptions.emplace_back(
      0 /*binding*/, sizeof(ParticleRender), vk::VertexInputRate::eVertex);

  graphics.clothVertexInfos.attributeDescriptions.emplace_back(
      0 /*location*/, 0 /* binding */, vk::Format::eR32G32B32A32Sfloat,
      offsetof(ParticleRender, pos));
  graphics.clothVertexInfos.attributeDescriptions.emplace_back(
      1 /*location*/, 0 /* binding */, vk::Format::eR32G32B32A32Sfloat,
      offsetof(ParticleRender, normal));
  graphics.clothVertexInfos.attributeDescriptions.emplace_back(
      2 /*location*/, 0 /* binding */, vk::Format::eR32G32Sfloat,
      offsetof(ParticleRender, uv));

  graphics.clothVertexInfos.vertexInputSCI =
      vk::PipelineVertexInputStateCreateInfo(
          vk::PipelineVertexInputStateCreateFlags{},
          graphics.clothVertexInfos.bindingDescriptions,
          graphics.clothVertexInfos.attributeDescriptions);
}

void VgeExample::setupDynamicUbo() {
  const float foxScale = 0.05f;
  glm::vec3 up{0.f, -1.f, 0.f};
  glm::vec3 right{1.f, 0.f, 0.f};
  glm::vec3 forward{0.f, 0.f, 1.f};
  float quadScale = 20.f;

  common.dynamicUbo.resize(modelInstances.size());
  {
    size_t instanceIndex = findInstances("fox0")[0];
    common.dynamicUbo[instanceIndex].modelMatrix = glm::translate(
        glm::mat4{1.f}, glm::vec3{quadScale * 1.5f - 6.f, 0.f, 0.f});
    common.dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        common.dynamicUbo[instanceIndex].modelMatrix, glm::radians(180.f), up);
    // FlipY manually
    common.dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(common.dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{foxScale, -foxScale, foxScale});
    common.dynamicUbo[instanceIndex].modelColor =
        glm::vec4{1.0f, 0.f, 0.f, 0.3f};
  }
  {
    size_t instanceIndex = findInstances("fox1")[0];
    common.dynamicUbo[instanceIndex].modelMatrix = glm::translate(
        glm::mat4{1.f}, glm::vec3{quadScale * 1.5f + 6.f, 0.f, 0.f});
    common.dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        common.dynamicUbo[instanceIndex].modelMatrix, glm::radians(0.f), up);
    // FlipY manually
    common.dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(common.dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{foxScale, -foxScale, foxScale});
    common.dynamicUbo[instanceIndex].modelColor =
        glm::vec4{0.0f, 0.f, 1.f, 0.3f};
  }

  {
    size_t instanceIndex = findInstances("fox2")[0];
    common.dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, glm::vec3{3.f, 0.f, 0.f});
    common.dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        common.dynamicUbo[instanceIndex].modelMatrix, glm::radians(180.f), up);
    // FlipY manually
    common.dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(common.dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{foxScale, -foxScale, foxScale});
    common.dynamicUbo[instanceIndex].modelColor =
        glm::vec4{0.f, 1.f, 1.f, 0.3f};
  }
  {
    float appleScale = 40.f;
    size_t instanceIndex = findInstances("apple1")[0];
    common.dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, glm::vec3{-3.f, 0.f, 0.f});
    // FlipY manually
    common.dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(common.dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{appleScale, -appleScale, appleScale});
  }
  {
    size_t instanceIndex = findInstances("koreanFlag1")[0];
    // NOTE: transformation moved into cloth initialization

    // -1 alpha for wire frame color
    common.dynamicUbo[instanceIndex].modelColor =
        glm::vec4{0.f, 0.f, 1.f, -1.f};
  }

  {
    size_t instanceIndex = findInstances("quad1")[0];
    common.dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, glm::vec3{0.f, 0.f, 0.f});
    common.dynamicUbo[instanceIndex].modelMatrix =
        glm::rotate(common.dynamicUbo[instanceIndex].modelMatrix,
                    glm::radians(90.f), right);
    common.dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(common.dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{quadScale, quadScale, quadScale});
    // default
    common.dynamicUbo[instanceIndex].modelColor = glm::vec4{0.f};
  }
  {
    size_t instanceIndex = findInstances("quad2")[0];
    common.dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, glm::vec3{quadScale * 1.5f, 0.f, 0.f});
    common.dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        common.dynamicUbo[instanceIndex].modelMatrix, glm::radians(45.f), up);
    common.dynamicUbo[instanceIndex].modelMatrix =
        glm::rotate(common.dynamicUbo[instanceIndex].modelMatrix,
                    glm::radians(90.f), right);
    common.dynamicUbo[instanceIndex].modelMatrix = glm::scale(
        common.dynamicUbo[instanceIndex].modelMatrix,
        glm::vec3{quadScale * 0.5f * sqrt(2.f), quadScale * 0.5f * sqrt(2.f),
                  quadScale * 0.5f * sqrt(2.f)});
    // default
    common.dynamicUbo[instanceIndex].modelColor = glm::vec4{0.f};
  }

  float rectScale = quadScale * sqrt(2.f);
  for (auto i = 0; i < 4; i++) {
    glm::vec3 tr(quadScale * cos(i * glm::half_pi<float>()), 0.f,
                 quadScale * sin(i * glm::half_pi<float>()));
    size_t instanceIndex =
        findInstances("rectLines" + std::to_string(i + 1))[0];
    common.dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, tr);
    common.dynamicUbo[instanceIndex].modelMatrix =
        glm::rotate(common.dynamicUbo[instanceIndex].modelMatrix,
                    glm::radians(((i + 1) % 4) * 90.f + 45.f), up);
    common.dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(common.dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{rectScale, -rectScale * 0.5f, rectScale});
    // default
    common.dynamicUbo[instanceIndex].modelColor = glm::vec4{0.f};
  }

  for (size_t instanceIdx = 0; instanceIdx < modelInstances.size();
       instanceIdx++) {
    const auto& modelInstance = modelInstances[instanceIdx];
    if (!modelInstance.clothModel) {
      continue;
    }
    common.dynamicUbo[instanceIdx].modelMatrix =
        modelInstance.clothModel->getInitialTransform();
  }
}

void VgeExample::createGraphicsUniformBuffers() {
  // graphics UBO
  graphics.globalUniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    graphics.globalUniformBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(GlobalUbo), 1,
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT));
    std::memcpy(graphics.globalUniformBuffers[i]->getMappedData(),
                &graphics.globalUbo, sizeof(GlobalUbo));
  }
}

void VgeExample::createDescriptorPool() {
  std::vector<vk::DescriptorPoolSize> poolSizes;
  poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer,
                         MAX_CONCURRENT_FRAMES * 2);
  poolSizes.emplace_back(vk::DescriptorType::eUniformBufferDynamic,
                         MAX_CONCURRENT_FRAMES);
  poolSizes.emplace_back(vk::DescriptorType::eStorageBuffer,
                         MAX_CONCURRENT_FRAMES * modelInstances.size() * 2 +
                             (MAX_CONCURRENT_FRAMES + 1) * kMaxNumClothModels);
  // NOTE: need to check flag
  vk::DescriptorPoolCreateInfo descriptorPoolCI(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      /*set globalUBO, dynamicUBO, computeUbo*/
      MAX_CONCURRENT_FRAMES * 3 +
          /*skin & animated vertex ssbo*/
          MAX_CONCURRENT_FRAMES * modelInstances.size() +
          /* calSB, renSB would be in a same set*/
          MAX_CONCURRENT_FRAMES * kMaxNumClothModels +
          /* constraint descriptor set*/
          kMaxNumClothModels,
      poolSizes);
  descriptorPool = vk::raii::DescriptorPool(device, descriptorPoolCI);
}

void VgeExample::createGraphicsDescriptorSetLayout() {
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

  // NOTE: used for external animation in compute and
  // used for lighting in graphics.
  // currently only for graphics
  // set 1
  {
    vk::DescriptorSetLayoutBinding layoutBinding(
        0, vk::DescriptorType::eUniformBufferDynamic, 1,
        vk::ShaderStageFlagBits::eAll);
    vk::DescriptorSetLayoutCreateInfo layoutCI({}, 1, &layoutBinding);
    common.dynamicUboDescriptorSetLayout =
        vk::raii::DescriptorSetLayout(device, layoutCI);
    setLayouts.push_back(*common.dynamicUboDescriptorSetLayout);
  }

  // set 2
  // TODO: need to improve structure. descriptorSetLayout per model
  setLayouts.push_back(*modelInstances[0].model->descriptorSetLayoutImage);

  vk::PipelineLayoutCreateInfo pipelineLayoutCI({}, setLayouts);
  graphics.pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutCI);
}

void VgeExample::createGraphicsDescriptorSets() {
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
}

void VgeExample::createGraphicsPipelines() {
  vk::PipelineVertexInputStateCreateInfo vertexInputSCI =
      graphics.animatedVertexInfos.vertexInputSCI;

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
      vk::PipelineDepthStencilStateCreateFlags(), true /*depthTestEnable*/,
      true, vk::CompareOp::eLessOrEqual, false, false, stencilOpState,
      stencilOpState);

  vk::PipelineColorBlendAttachmentState colorBlendAttachmentState(
      true, vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha,
      vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eOne,
      vk::BlendOp::eAdd,
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

  auto vertCode = vgeu::readFile(getShadersPath() + "/cloth/phong.vert.spv");
  auto fragCode = vgeu::readFile(getShadersPath() + "/cloth/phong.frag.spv");
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

  graphics.pipelines.pipelinePhong =
      vk::raii::Pipeline(device, pipelineCache, graphicsPipelineCI);
  {
    vertexInputSCI.setVertexBindingDescriptions(
        graphics.clothVertexInfos.bindingDescriptions);
    vertexInputSCI.setVertexAttributeDescriptions(
        graphics.clothVertexInfos.attributeDescriptions);
    vertCode = vgeu::readFile(getShadersPath() + "/cloth/cloth.vert.spv");
    // NOTE: share fragment shader w/ phong shader.
    // NOTE: after pipeline creation, shader modules can be destroyed.
    vertShaderModule = vgeu::createShaderModule(device, vertCode);
    shaderStageCIs[0] = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex,
        *vertShaderModule, "main", nullptr);
    graphics.pipelines.pipelineCloth =
        vk::raii::Pipeline(device, pipelineCache, graphicsPipelineCI);
  }
  {
    vertexInputSCI.setVertexBindingDescriptions(
        graphics.animatedVertexInfos.bindingDescriptions);
    vertexInputSCI.setVertexAttributeDescriptions(
        graphics.animatedVertexInfos.attributeDescriptions);
    rasterizationSCI.polygonMode = vk::PolygonMode::eLine;
    vertCode = vgeu::readFile(getShadersPath() + "/cloth/wireframe.vert.spv");
    fragCode = vgeu::readFile(getShadersPath() + "/cloth/wireframe.frag.spv");
    // NOTE: after pipeline creation, shader modules can be destroyed.
    vertShaderModule = vgeu::createShaderModule(device, vertCode);
    fragShaderModule = vgeu::createShaderModule(device, fragCode);
    shaderStageCIs[0] = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex,
        *vertShaderModule, "main", nullptr);
    shaderStageCIs[1] = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "main", nullptr);
    graphics.pipelines.pipelineWireMesh =
        vk::raii::Pipeline(device, pipelineCache, graphicsPipelineCI);
  }
  {
    vertexInputSCI.setVertexBindingDescriptions(
        graphics.clothVertexInfos.bindingDescriptions);
    vertexInputSCI.setVertexAttributeDescriptions(
        graphics.clothVertexInfos.attributeDescriptions);
    vertCode = vgeu::readFile(getShadersPath() + "/cloth/wireCloth.vert.spv");
    // NOTE: share fragment shader w/ phong shader.
    // NOTE: after pipeline creation, shader modules can be destroyed.
    vertShaderModule = vgeu::createShaderModule(device, vertCode);
    shaderStageCIs[0] = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex,
        *vertShaderModule, "main", nullptr);
    graphics.pipelines.pipelineWireCloth =
        vk::raii::Pipeline(device, pipelineCache, graphicsPipelineCI);
  }
  {
    rasterizationSCI.polygonMode = vk::PolygonMode::eFill;
    vertexInputSCI.setVertexBindingDescriptions(
        graphics.simpleVertexInfos.bindingDescriptions);
    vertexInputSCI.setVertexAttributeDescriptions(
        graphics.simpleVertexInfos.attributeDescriptions);
    vertCode = vgeu::readFile(getShadersPath() + "/cloth/simpleMesh.vert.spv");
    fragCode = vgeu::readFile(getShadersPath() + "/cloth/simpleMesh.frag.spv");
    // NOTE: after pipeline creation, shader modules can be destroyed.
    vertShaderModule = vgeu::createShaderModule(device, vertCode);
    fragShaderModule = vgeu::createShaderModule(device, fragCode);
    shaderStageCIs[0] = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex,
        *vertShaderModule, "main", nullptr);
    shaderStageCIs[1] = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "main", nullptr);
    graphics.pipelines.pipelineSimpleMesh =
        vk::raii::Pipeline(device, pipelineCache, graphicsPipelineCI);
  }
  {
    inputAssemblySCI.topology = vk::PrimitiveTopology::eLineList;
    vertCode = vgeu::readFile(getShadersPath() + "/cloth/simpleLine.vert.spv");
    fragCode = vgeu::readFile(getShadersPath() + "/cloth/simpleLine.frag.spv");
    // NOTE: after pipeline creation, shader modules can be destroyed.
    vertShaderModule = vgeu::createShaderModule(device, vertCode);
    fragShaderModule = vgeu::createShaderModule(device, fragCode);
    shaderStageCIs[0] = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex,
        *vertShaderModule, "main", nullptr);
    shaderStageCIs[1] = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "main", nullptr);
    graphics.pipelines.pipelineSimpleLine =
        vk::raii::Pipeline(device, pipelineCache, graphicsPipelineCI);
  }
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
  updateDynamicUbo();
  updateComputeUbo();
  updateGraphicsUbo();

  //  compute recording and submitting
  {
    buildComputeCommandBuffers();
    vk::PipelineStageFlags computeWaitDstStageMask(
        vk::PipelineStageFlagBits::eComputeShader);
    vk::SubmitInfo computeSubmitInfo(
        *compute.semaphores.ready[currentFrameIndex], computeWaitDstStageMask,
        *compute.cmdBuffers[currentFrameIndex],
        *compute.semaphores.complete[currentFrameIndex]);
    compute.queue.submit(computeSubmitInfo);
  }

  {
    // draw cmds recording or command buffers should be built already.
    buildCommandBuffers();

    std::vector<vk::PipelineStageFlags> graphicsWaitDstStageMasks{
        vk::PipelineStageFlagBits::eVertexInput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
    };

    std::vector<vk::Semaphore> graphicsWaitSemaphores{
        *compute.semaphores.complete[currentFrameIndex],
        *presentCompleteSemaphores[currentFrameIndex],
    };

    std::vector<vk::Semaphore> graphicsSignalSemaphore{
        *compute.semaphores.ready[currentFrameIndex],
        *renderCompleteSemaphores[currentFrameIndex],
    };

    // NOTE: parameter array type has no r value constructor
    vk::SubmitInfo graphicsSubmitInfo(
        graphicsWaitSemaphores, graphicsWaitDstStageMasks,
        *drawCmdBuffers[currentFrameIndex], graphicsSignalSemaphore);

    queue.submit(graphicsSubmitInfo, *waitFences[currentFrameIndex]);
  }
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

  //  acquire barrier compute -> graphics
  {
    vgeu::addQueueFamilyOwnershipTransferBarriers(
        compute.queueFamilyIndex, graphics.queueFamilyIndex,
        drawCmdBuffers[currentFrameIndex],
        common.ownershipTransferBufferPtrs[currentFrameIndex],
        vk::AccessFlags{}, vk::AccessFlagBits::eVertexAttributeRead,
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eVertexInput);
  }

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

  {
    drawCmdBuffers[currentFrameIndex].setLineWidth(1.f);
    drawCmdBuffers[currentFrameIndex].setViewport(
        0,
        vk::Viewport(0.0f, 0.0f,
                     static_cast<float>(swapChainData->swapChainExtent.width),
                     static_cast<float>(swapChainData->swapChainExtent.height),
                     0.0f, 1.0f));
    if (opts.renderWireMesh) {
      drawCmdBuffers[currentFrameIndex].bindPipeline(
          vk::PipelineBindPoint::eGraphics,
          *graphics.pipelines.pipelineWireMesh);
    } else {
      // bind pipeline
      drawCmdBuffers[currentFrameIndex].bindPipeline(
          vk::PipelineBindPoint::eGraphics, *graphics.pipelines.pipelinePhong);
    }
    // draw all instances including model based and bones.
    for (size_t instanceIdx = 0; instanceIdx < modelInstances.size();
         instanceIdx++) {
      const auto& modelInstance = modelInstances[instanceIdx];
      if (!modelInstance.model || modelInstance.clothModel) {
        continue;
      }

      // bind dynamic
      drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics, *graphics.pipelineLayout,
          1 /*set 1*/, {*common.dynamicUboDescriptorSets[currentFrameIndex]},
          common.alignedSizeDynamicUboElt * instanceIdx);
      // bind vertex buffer
      vk::DeviceSize offset(0);
      drawCmdBuffers[currentFrameIndex].bindVertexBuffers(
          0,
          common.animatedVertexBuffers[currentFrameIndex][instanceIdx]
              ->getBuffer(),
          offset);
      // bind index buffer
      modelInstance.model->bindIndexBufferOnly(
          drawCmdBuffers[currentFrameIndex]);
      // draw indexed
      modelInstance.model->draw(currentFrameIndex,
                                drawCmdBuffers[currentFrameIndex],
                                vgeu::RenderFlagBits::kBindImages,
                                *graphics.pipelineLayout, 2u /*set 2*/);
    }
  }
  // draw cloth models
  if (opts.renderWireMesh) {
    drawCmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eGraphics,
        *graphics.pipelines.pipelineWireCloth);
  } else {
    drawCmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eGraphics, *graphics.pipelines.pipelineCloth);
  }
  for (size_t instanceIdx = 0; instanceIdx < modelInstances.size();
       instanceIdx++) {
    const auto& modelInstance = modelInstances[instanceIdx];
    if (!modelInstance.clothModel) {
      continue;
    }
    // bind dynamic
    drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, *graphics.pipelineLayout, 1 /*set 1*/,
        {*common.dynamicUboDescriptorSets[currentFrameIndex]},
        common.alignedSizeDynamicUboElt * instanceIdx);
    // bind vertex buffer
    modelInstance.clothModel->bindVertexBuffer(
        drawCmdBuffers[currentFrameIndex], currentFrameIndex);
    // bind first loaded material textures
    drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, *graphics.pipelineLayout, 2 /*set 1*/,
        {modelInstance.model->getMaterialDescriptor(0)}, nullptr);
    // draw w/o indexed
    drawCmdBuffers[currentFrameIndex].draw(
        modelInstance.clothModel->getNumTriangles() * 3, 1, 0, 0);
  }

  // draw simple models
  for (auto instanceIdx = 0; instanceIdx < modelInstances.size();
       instanceIdx++) {
    const auto& modelInstance = modelInstances[instanceIdx];
    if (!modelInstance.simpleModel) {
      continue;
    }
    if (modelInstance.simpleModel->isLines) {
      drawCmdBuffers[currentFrameIndex].setLineWidth(opts.lineWidth);
      // simpleLines
      drawCmdBuffers[currentFrameIndex].bindPipeline(
          vk::PipelineBindPoint::eGraphics,
          *graphics.pipelines.pipelineSimpleLine);
    } else {
      // simpleMesh
      drawCmdBuffers[currentFrameIndex].bindPipeline(
          vk::PipelineBindPoint::eGraphics,
          *graphics.pipelines.pipelineSimpleMesh);
    }
    // bind dynamic
    drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, *graphics.pipelineLayout, 1 /*set 1*/,
        {*common.dynamicUboDescriptorSets[currentFrameIndex]},
        common.alignedSizeDynamicUboElt * instanceIdx);
    vk::DeviceSize offset(0);
    drawCmdBuffers[currentFrameIndex].bindVertexBuffers(
        0, modelInstance.simpleModel->vertexBuffer->getBuffer(), offset);
    drawCmdBuffers[currentFrameIndex].bindIndexBuffer(
        modelInstance.simpleModel->indexBuffer->getBuffer(), offset,
        vk::IndexType::eUint32);

    drawCmdBuffers[currentFrameIndex].drawIndexed(
        modelInstance.simpleModel->indexBuffer->getInstanceCount(), 1, 0, 0, 0);
  }

  // UI overlay draw
  drawUI(drawCmdBuffers[currentFrameIndex]);

  // end renderpass
  drawCmdBuffers[currentFrameIndex].endRenderPass();

  // release graphics -> compute
  {
    vgeu::addQueueFamilyOwnershipTransferBarriers(
        graphics.queueFamilyIndex, compute.queueFamilyIndex,
        drawCmdBuffers[currentFrameIndex],
        common.ownershipTransferBufferPtrs[currentFrameIndex],
        vk::AccessFlagBits::eVertexAttributeRead, vk::AccessFlags{},
        vk::PipelineStageFlagBits::eVertexInput,
        vk::PipelineStageFlagBits::eBottomOfPipe);
  }

  // end command buffer
  drawCmdBuffers[currentFrameIndex].end();
}

void VgeExample::buildComputeCommandBuffers() {
  compute.cmdBuffers[currentFrameIndex].begin({});

  // no matching release at first
  if (!compute.firstCompute[currentFrameIndex]) {
    // acquire barrier graphics -> compute
    vgeu::addQueueFamilyOwnershipTransferBarriers(
        graphics.queueFamilyIndex, compute.queueFamilyIndex,
        compute.cmdBuffers[currentFrameIndex],
        common.ownershipTransferBufferPtrs[currentFrameIndex],
        vk::AccessFlags{}, vk::AccessFlagBits::eShaderWrite,
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eComputeShader);
  }

  // pre compute animation
  {
    compute.cmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eCompute,
        *compute.pipelines.pipelineModelAnimate);
    for (auto instanceIdx = 0; instanceIdx < modelInstances.size();
         instanceIdx++) {
      const auto& modelInstance = modelInstances[instanceIdx];
      // animate only gltf models (modelMatrix)
      if (!modelInstance.model || modelInstance.clothModel) {
        continue;
      }
      // bind SSBO for animation input vertices
      modelInstance.model->bindSSBO(compute.cmdBuffers[currentFrameIndex],
                                    *compute.pipelineLayout, 1 /*set*/);

      compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
          vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 2 /*set*/,
          {*common.dynamicUboDescriptorSets[currentFrameIndex]},
          common.alignedSizeDynamicUboElt * instanceIdx);

      // bind SSBO for skin matrix and animated vertices
      compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
          vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 3 /*set*/,
          *compute.skinDescriptorSets[currentFrameIndex][instanceIdx], nullptr);
      compute.cmdBuffers[currentFrameIndex].dispatch(
          modelInstances[instanceIdx].model->getVertexCount() / sharedDataSize +
              1,
          1, 1);
    }
  }

  // TODO: check only animated buffer should be used
  // compute execution memory barrier
  vgeu::addComputeToComputeBarriers(
      compute.cmdBuffers[currentFrameIndex],
      common.ownershipTransferBufferPtrs[currentFrameIndex]);

  compute.cmdBuffers[currentFrameIndex].pushConstants<ComputePushConstantsData>(
      *compute.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
      compute.pc);
  // TODO: substeps need  or just submit with dt/numsubsteps
  for (auto substep = 0; substep < opts.numSubsteps; substep++) {
    //  integrate
    {
      // compute ubo
      compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
          vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 0 /*set*/,
          *compute.descriptorSets[currentFrameIndex], nullptr);
      compute.cmdBuffers[currentFrameIndex].bindPipeline(
          vk::PipelineBindPoint::eCompute,
          *compute.pipelines
               .pipelinesCloth[static_cast<uint32_t>(ComputeType::kIntegrate)]);
      for (auto instanceIdx = 0; instanceIdx < modelInstances.size();
           instanceIdx++) {
        const auto& modelInstance = modelInstances[instanceIdx];
        if (!modelInstance.clothModel) {
          continue;
        }
        // NOTE: not using but for validation error.
        compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 5 /*set*/,
            modelInstance.clothModel->getConstraintDescriptorSet(), nullptr);
        //
        compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 4 /*set*/,
            modelInstance.clothModel->getParticleDescriptorSet(
                currentFrameIndex),
            nullptr);
        compute.cmdBuffers[currentFrameIndex].dispatch(
            modelInstance.clothModel->getNumParticles() / sharedDataSize + 1, 1,
            1);
      }
    }
    // compute execution memory barrier
    vgeu::addComputeToComputeBarriers(
        compute.cmdBuffers[currentFrameIndex],
        compute.calculateBufferPtrs[currentFrameIndex]);
    // solve collision
    {
      // compute ubo
      compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
          vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 0 /*set*/,
          *compute.descriptorSets[currentFrameIndex], nullptr);

      uint32_t collisionInstanceIdx = findInstances("fox2")[0];
      // bind SSBO for skin matrix and animated vertices
      compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
          vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 3 /*set*/,
          *compute.skinDescriptorSets[currentFrameIndex][collisionInstanceIdx],
          nullptr);
      const auto& collisionModelInstance = modelInstances[collisionInstanceIdx];

      compute.cmdBuffers[currentFrameIndex].bindPipeline(
          vk::PipelineBindPoint::eCompute,
          *compute.pipelines.pipelinesCloth[static_cast<uint32_t>(
              ComputeType::kSolveCollision)]);
      for (auto instanceIdx = 0; instanceIdx < modelInstances.size();
           instanceIdx++) {
        const auto& modelInstance = modelInstances[instanceIdx];
        if (!modelInstance.clothModel) {
          continue;
        }
        compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 4 /*set*/,
            modelInstance.clothModel->getParticleDescriptorSet(
                currentFrameIndex),
            nullptr);
        compute.cmdBuffers[currentFrameIndex].dispatch(
            modelInstance.clothModel->getNumParticles() / sharedDataSize + 1,
            collisionModelInstance.model->getVertexCount() /
                    collisionWorkGroupSize +
                1,
            1);
      }
    }
    // compute execution memory barrier
    vgeu::addComputeToComputeBarriers(
        compute.cmdBuffers[currentFrameIndex],
        compute.calculateBufferPtrs[currentFrameIndex]);

    // solve distance constraints
    {
      // compute ubo
      compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
          vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 0 /*set*/,
          *compute.descriptorSets[currentFrameIndex], nullptr);
      for (auto instanceIdx = 0; instanceIdx < modelInstances.size();
           instanceIdx++) {
        const auto& modelInstance = modelInstances[instanceIdx];
        if (!modelInstance.clothModel) {
          continue;
        }
        compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 4 /*set*/,
            modelInstance.clothModel->getParticleDescriptorSet(
                currentFrameIndex),
            nullptr);
        compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 5 /*set*/,
            modelInstance.clothModel->getConstraintDescriptorSet(), nullptr);
        int firstConstraint = 0;
        for (auto passIndex = 0;
             passIndex < modelInstance.clothModel->getNumPasses();
             passIndex++) {
          compute.pc.constraintInfo.x = firstConstraint;
          compute.pc.constraintInfo.y =
              modelInstance.clothModel->getPassSize(passIndex);
          firstConstraint += modelInstance.clothModel->getPassSize(passIndex);
          compute.cmdBuffers[currentFrameIndex]
              .pushConstants<ComputePushConstantsData>(
                  *compute.pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0,
                  compute.pc);
          if (modelInstance.clothModel->isPassIndependent(passIndex)) {
            compute.cmdBuffers[currentFrameIndex].bindPipeline(
                vk::PipelineBindPoint::eCompute,
                *compute.pipelines.pipelinesCloth[static_cast<uint32_t>(
                    ComputeType::kSolveDistanceConstraintsGauss)]);
          } else {
            compute.cmdBuffers[currentFrameIndex].bindPipeline(
                vk::PipelineBindPoint::eCompute,
                *compute.pipelines.pipelinesCloth[static_cast<uint32_t>(
                    ComputeType::kSolveDistanceConstraintsJacobi)]);
          }
          compute.cmdBuffers[currentFrameIndex].dispatch(
              modelInstance.clothModel->getPassSize(passIndex) /
                      sharedDataSize +
                  1,
              1, 1);
          vgeu::addComputeToComputeBarriers(
              compute.cmdBuffers[currentFrameIndex],
              {modelInstance.clothModel->getCalculateSBPtr(currentFrameIndex)});
          if (!modelInstance.clothModel->isPassIndependent(passIndex)) {
            compute.cmdBuffers[currentFrameIndex].bindPipeline(
                vk::PipelineBindPoint::eCompute,
                *compute.pipelines.pipelinesCloth[static_cast<uint32_t>(
                    ComputeType::kAddCorrections)]);
            compute.cmdBuffers[currentFrameIndex].dispatch(
                modelInstance.clothModel->getNumParticles() / sharedDataSize +
                    1,
                1, 1);
            vgeu::addComputeToComputeBarriers(
                compute.cmdBuffers[currentFrameIndex],
                {modelInstance.clothModel->getCalculateSBPtr(
                    currentFrameIndex)});
          }
        }
      }
    }
    // update vel
    {
      compute.cmdBuffers[currentFrameIndex].bindPipeline(
          vk::PipelineBindPoint::eCompute,
          *compute.pipelines
               .pipelinesCloth[static_cast<uint32_t>(ComputeType::kUpdateVel)]);
      for (auto instanceIdx = 0; instanceIdx < modelInstances.size();
           instanceIdx++) {
        const auto& modelInstance = modelInstances[instanceIdx];
        if (!modelInstance.clothModel) {
          continue;
        }
        compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
            vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 4 /*set*/,
            modelInstance.clothModel->getParticleDescriptorSet(
                currentFrameIndex),
            nullptr);
        compute.cmdBuffers[currentFrameIndex].dispatch(
            modelInstance.clothModel->getNumParticles() / sharedDataSize + 1, 1,
            1);
      }
    }
    // compute execution memory barrier
    vgeu::addComputeToComputeBarriers(
        compute.cmdBuffers[currentFrameIndex],
        compute.calculateBufferPtrs[currentFrameIndex]);
  }
  // update mesh
  {
    compute.cmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eCompute,
        *compute.pipelines
             .pipelinesCloth[static_cast<uint32_t>(ComputeType::kUpdateMesh)]);
    for (auto instanceIdx = 0; instanceIdx < modelInstances.size();
         instanceIdx++) {
      const auto& modelInstance = modelInstances[instanceIdx];
      if (!modelInstance.clothModel) {
        continue;
      }
      modelInstance.model->bindSSBO(compute.cmdBuffers[currentFrameIndex],
                                    *compute.pipelineLayout, 1 /*set*/);
      compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
          vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 4 /*set*/,
          modelInstance.clothModel->getParticleDescriptorSet(currentFrameIndex),
          nullptr);
      compute.cmdBuffers[currentFrameIndex].dispatch(
          modelInstance.clothModel->getNumTriangles() * 3 / sharedDataSize + 1,
          1, 1);
    }
  }

  // compute execution memory barrier
  vgeu::addComputeToComputeBarriers(
      compute.cmdBuffers[currentFrameIndex],
      compute.calculateBufferPtrs[currentFrameIndex]);
  // normalize normals
  {
    compute.cmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eCompute,
        *compute.pipelines.pipelinesCloth[static_cast<uint32_t>(
            ComputeType::kUpdateNormals)]);
    for (auto instanceIdx = 0; instanceIdx < modelInstances.size();
         instanceIdx++) {
      const auto& modelInstance = modelInstances[instanceIdx];
      if (!modelInstance.clothModel) {
        continue;
      }
      modelInstance.model->bindSSBO(compute.cmdBuffers[currentFrameIndex],
                                    *compute.pipelineLayout, 1 /*set*/);
      compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
          vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 4 /*set*/,
          modelInstance.clothModel->getParticleDescriptorSet(currentFrameIndex),
          nullptr);
      compute.cmdBuffers[currentFrameIndex].dispatch(
          modelInstance.clothModel->getNumTriangles() / sharedDataSize + 1, 1,
          1);
    }
  }

  // TODO: other compute shader dispatch call
  // support [one animated model - multi clothes] interation

  // release barrier
  {
    vgeu::addQueueFamilyOwnershipTransferBarriers(
        compute.queueFamilyIndex, graphics.queueFamilyIndex,
        compute.cmdBuffers[currentFrameIndex],
        common.ownershipTransferBufferPtrs[currentFrameIndex],
        vk::AccessFlagBits::eShaderWrite, vk::AccessFlags{},
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eBottomOfPipe);
  }
  compute.cmdBuffers[currentFrameIndex].end();
  compute.firstCompute[currentFrameIndex] = false;
}

void VgeExample::viewChanged() {
  // std::cout << "Call: viewChanged()" << std::endl;
  camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
  // NOTE: moved updating ubo into render() to use frameindex.
}

// update UniformBuffers for currentFrameIndex
void VgeExample::updateGraphicsUbo() {
  // CHECK: ubo update frequency.
  graphics.globalUbo.view = camera.getView();
  graphics.globalUbo.projection = camera.getProjection();
  graphics.globalUbo.inverseView = camera.getInverseView();
  graphics.globalUbo.lightPos = glm::vec4(-5.f, -10.f, -10.f, 0.f);
  graphics.globalUbo.screenDim = glm::vec2{
      static_cast<float>(width),
      static_cast<float>(height),
  };
  graphics.globalUbo.pointSize.x =
      std::min(opts.pointSize[0], opts.pointSize[1]);
  graphics.globalUbo.pointSize.y =
      std::max(opts.pointSize[0], opts.pointSize[1]);
  std::memcpy(graphics.globalUniformBuffers[currentFrameIndex]->getMappedData(),
              &graphics.globalUbo, sizeof(GlobalUbo));
}

void VgeExample::updateComputeUbo() {
  compute.ubo.dt = paused ? 0.0f : frameTimer * opts.coefficientDeltaTime;

  {
    glm::vec2 normalizedMousePos{
        2.f * mouseData.mousePos.x / static_cast<float>(width) - 1.f,
        2.f * mouseData.mousePos.y / static_cast<float>(height) - 1.f};

    glm::vec4 rayStart = camera.getInverseView() *
                         glm::inverse(camera.getProjection()) *
                         glm::vec4(normalizedMousePos, 0.f, 1.f);
    rayStart /= rayStart.w;
    glm::vec4 rayEnd = camera.getInverseView() *
                       glm::inverse(camera.getProjection()) *
                       glm::vec4(normalizedMousePos, 0.1f, 1.f);
    rayEnd /= rayEnd.w;

    glm::vec3 rayDir(glm::normalize(rayEnd - rayStart));

    glm::vec3 planeNormal{camera.getView()[0][2], camera.getView()[1][2],
                          camera.getView()[2][2]};
    std::optional<glm::vec3> intersectionPt =
        ::rayPlaneIntersection(rayStart, rayDir, planeNormal, glm::vec3{0.f});

    glm::vec4 clickPos{0.f};
    if (intersectionPt.has_value()) {
      clickPos = glm::vec4(intersectionPt.value(), 0.f);
    }

    if (mouseData.left) {
      clickPos.w = 1.f;
    } else if (mouseData.right) {
      clickPos.w = 2.f;
    } else if (mouseData.middle) {
      clickPos.w = 3.f;
    } else {
      clickPos.w = 0.f;
    }

    compute.ubo.clickData = clickPos;
  }

  compute.ubo.thickness = opts.thickness;
  compute.ubo.friction = opts.friction;
  compute.ubo.radius = opts.collisionRadius;
  compute.ubo.stiffness = opts.stiffness;
  compute.ubo.alpha = opts.alpha;
  compute.ubo.jacobiScale = opts.jacobiScale;
  compute.ubo.numSubsteps = static_cast<uint32_t>(opts.numSubsteps);
  compute.ubo.gravity = glm::vec4(0.f, opts.gravity, 0.f, 0.f);
  if (atomicFloatFeatures.shaderBufferFloat32AtomicAdd) {
    compute.ubo.atomicAdd = 1u;
  } else if (atomicFloatFeatures.shaderBufferFloat32Atomics) {
    compute.ubo.atomicAdd = 2u;
  } else {
    compute.ubo.atomicAdd = 0u;
  }
  if (!opts.useSeparateNormal) {
    compute.ubo.atomicAdd = 0u;
  }

  std::memcpy(compute.uniformBuffers[currentFrameIndex]->getMappedData(),
              &compute.ubo, sizeof(compute.ubo));
}

void VgeExample::updateDynamicUbo() {
  float quadScale = 20.f;
  float animationTimer =
      (animationTime - animationLastTime) * opts.animationSpeed;
  // model move

  glm::vec3 up{0.f, -1.f, 0.f};
  // deg per sec;
  float rotationVelocity = 50.f;
  {
    size_t instanceIndex = findInstances("fox0")[0];
    glm::mat4 trt = glm::mat4{1.f};
    trt = glm::translate(trt, glm::vec3{quadScale * 1.5f, 0.f, 0.f});
    trt = glm::rotate(trt, glm::radians(rotationVelocity) * animationTimer, up);
    trt = glm::translate(trt, glm::vec3{quadScale * -1.5f, 0.f, 0.f});

    common.dynamicUbo[instanceIndex].modelMatrix =
        trt * common.dynamicUbo[instanceIndex].modelMatrix;
  }
  {
    size_t instanceIndex = findInstances("fox1")[0];
    glm::mat4 trt = glm::mat4{1.f};
    trt = glm::translate(trt, glm::vec3{quadScale * 1.5f, 0.f, 0.f});
    trt = glm::rotate(trt, glm::radians(rotationVelocity) * animationTimer, up);
    trt = glm::translate(trt, glm::vec3{quadScale * -1.5f, 0.f, 0.f});

    common.dynamicUbo[instanceIndex].modelMatrix =
        trt * common.dynamicUbo[instanceIndex].modelMatrix;
  }

  // update animation joint matrices for each shared model
  {
    std::unordered_set<const vgeu::glTF::Model*> updatedSharedModelSet;
    for (auto& modelInstance : modelInstances) {
      if (!modelInstance.model) {
        continue;
      }
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

    // update animation ssbo
    for (auto i = 0; i < modelInstances.size(); i++) {
      const auto& modelInstance = modelInstances[i];
      if (!modelInstance.model) {
        continue;
      }
      modelInstance.model->getSkinMatrices(compute.skinMatricesData[i]);
      std::memcpy(
          compute.skinMatricesBuffers[currentFrameIndex][i]->getMappedData(),
          compute.skinMatricesData[i].data(),
          compute.skinMatricesBuffers[currentFrameIndex][i]->getBufferSize());
    }
  }

  // update all dynamicUbo elements
  for (size_t j = 0; j < common.dynamicUbo.size(); j++) {
    std::memcpy(
        static_cast<char*>(
            common.dynamicUniformBuffers[currentFrameIndex]->getMappedData()) +
            j * common.alignedSizeDynamicUboElt,
        &common.dynamicUbo[j], common.alignedSizeDynamicUboElt);
  }
}

ModelInstance::ModelInstance(ModelInstance&& other) {
  model = other.model;
  simpleModel = other.simpleModel;
  clothModel = std::move(other.clothModel);
  name = other.name;
  isBone = other.isBone;
  animationIndex = other.animationIndex;
  animationTime = other.animationTime;
  transform = other.transform;
}

ModelInstance& ModelInstance::operator=(ModelInstance&& other) {
  model = other.model;
  simpleModel = other.simpleModel;
  clothModel = std::move(other.clothModel);
  name = other.name;
  isBone = other.isBone;
  animationIndex = other.animationIndex;
  animationTime = other.animationTime;
  transform = other.transform;
  return *this;
}

void VgeExample::addModelInstance(ModelInstance&& newInstance) {
  size_t instanceIdx;
  instanceIdx = modelInstances.size();
  modelInstances.push_back(std::move(newInstance));
  instanceMap[newInstance.name].push_back(instanceIdx);
}

const std::vector<size_t>& VgeExample::findInstances(const std::string& name) {
  assert(instanceMap.find(name) != instanceMap.end() &&
         "failed to find instance by name.");
  return instanceMap.at(name);
}

void VgeExample::setupCommandLineParser(CLI::App& app) {
  VgeBase::setupCommandLineParser(app);
  app.add_option("--numParticles, --np", numParticles, "number of particles")
      ->capture_default_str();
}

void VgeExample::onUpdateUIOverlay() {
  if (uiOverlay->header("Inputs")) {
    ImGui::Text("Mouse Left: %s", mouseData.left ? "true" : "false");
    ImGui::Text("Mouse Middle: %s", mouseData.middle ? "true" : "false");
    ImGui::Text("Mouse Right: %s", mouseData.right ? "true" : "false");
    ImGui::Text("Mouse Pos: (%f, %f)", mouseData.mousePos.x,
                mouseData.mousePos.y);
    ImGui::Text("Compute Click Data: (%f, %f, %f, %f)", compute.ubo.clickData.x,
                compute.ubo.clickData.y, compute.ubo.clickData.z,
                compute.ubo.clickData.w);
  }

  if (uiOverlay->header("Settings")) {
    if (ImGui::TreeNodeEx("Immediate", ImGuiTreeNodeFlags_DefaultOpen)) {
      uiOverlay->inputFloat("coefficientDeltaTime", &opts.coefficientDeltaTime,
                            0.001f, "%.3f");

      ImGui::DragFloat2("Drag pointSize min/max", opts.pointSize, 1.f, 1.f,
                        128.f, "%.0f");

      if (ImGui::RadioButton("renderWireMesh", opts.renderWireMesh)) {
        opts.renderWireMesh = !opts.renderWireMesh;
      }
      uiOverlay->inputFloat("animationSpeed", &opts.animationSpeed, 0.001f,
                            "%.3f");

      if (uiOverlay->inputFloat("keyboardMoveSpeed", &opts.moveSpeed, 0.01f,
                                "%.3f")) {
        cameraController.moveSpeed = this->opts.moveSpeed;
      }
      uiOverlay->inputFloat("lineWidth", &opts.lineWidth, 0.1f, "%.3f");

      ImGui::Spacing();
      ImGui::Spacing();
      uiOverlay->inputFloat("gravity", &opts.gravity, 0.001f, "%.3f");
      uiOverlay->inputFloat("collisionRadius", &opts.collisionRadius, 0.001f,
                            "%.3f");
      uiOverlay->inputFloat("stiffness", &opts.stiffness, 0.001f, "%.3f");
      uiOverlay->inputFloat("alpha", &opts.alpha, 0.001f, "%.3f");
      uiOverlay->inputFloat("jacobiScale", &opts.jacobiScale, 0.001f, "%.3f");
      uiOverlay->inputFloat("thickness", &opts.thickness, 0.001f, "%.3f");
      uiOverlay->inputFloat("friction", &opts.friction, 0.001f, "%.3f");
      if (ImGui::RadioButton("useSeparateNormal", opts.useSeparateNormal)) {
        opts.useSeparateNormal = !opts.useSeparateNormal;
      }

      uiOverlay->inputInt("numSubsteps", &opts.numSubsteps, 1);

      ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Initializers", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (uiOverlay->button("Restart")) {
        opts.cameraView = camera.getView();
        restart = true;
      }

      ImGui::Spacing();

      ImGui::Spacing();
      uiOverlay->inputInt("desiredSharedDataSize", &opts.desiredSharedDataSize,
                          16);
      ImGui::TreePop();
    }
  }
}

void VgeExample::setOptions(const std::optional<Options>& opts) {
  if (opts.has_value()) {
    this->opts = opts.value();
    // overwrite cli args for restart run
    cameraController.moveSpeed = this->opts.moveSpeed;
    desiredSharedDataSize =
        static_cast<uint32_t>(this->opts.desiredSharedDataSize);

  } else {
    // save cli args for initial run
  }
}

SimpleModel::SimpleModel(const vk::raii::Device& device, VmaAllocator allocator,
                         const vk::raii::Queue& transferQueue,
                         const vk::raii::CommandPool& commandPool)
    : device{device},
      allocator{allocator},
      transferQueue{transferQueue},
      commandPool{commandPool} {}

void SimpleModel::setNgon(uint32_t n, glm::vec4 color, bool useCenter) {
  isLines = false;
  std::vector<SimpleModel::Vertex> vertices;
  std::vector<uint32_t> indices;
  if (useCenter) {
    auto& vert = vertices.emplace_back();
    glm::vec4 pos{0.f, 0.f, 0.f, 1.f};
    glm::vec4 normal{0.f, 0.f, 1.f, 0.f};
    glm::vec2 uv{(pos.x + 1.f) / 2.f, (pos.y + 1.f) / 2.f};
    vert.pos = pos;
    vert.normal = normal;
    vert.color = color;
    vert.uv = uv;
    // n triangles
    for (auto i = 1; i <= n; i++) {
      indices.push_back(0);
      indices.push_back(i);
      indices.push_back(i % n + 1);
    }
  } else {
    // n-2 triangles
    for (auto i = 0; i < n - 2; i++) {
      indices.push_back(0);
      indices.push_back(i + 1);
      indices.push_back(i + 2);
    }
  }
  for (auto i = 0; i < n; i++) {
    auto& vert = vertices.emplace_back();
    glm::vec4 pos{0.f};
    pos.x = cos(glm::two_pi<float>() / static_cast<float>(n) *
                static_cast<float>(i));
    pos.y = sin(glm::two_pi<float>() / static_cast<float>(n) *
                static_cast<float>(i));
    pos.w = 1.f;
    glm::vec4 normal{0.f, 0.f, 1.f, 0.f};
    glm::vec2 uv{(pos.x + 1.f) / 2.f, (pos.y + 1.f) / 2.f};
    vert.pos = pos;
    vert.normal = normal;
    vert.color = color;
    vert.uv = uv;
  }
  this->vertices = vertices;
  this->indices = indices;
  createBuffers(vertices, indices);
}
void SimpleModel::setLineList(const std::vector<glm::vec4>& positions,
                              const std::vector<uint32_t>& indices,
                              glm::vec4 color) {
  isLines = true;
  std::vector<SimpleModel::Vertex> vertices;
  for (const auto& pos : positions) {
    auto& vert = vertices.emplace_back();
    glm::vec4 normal{0.f};
    glm::vec2 uv{0.f};
    vert.pos = pos;
    vert.normal = normal;
    vert.color = color;
    vert.uv = uv;
  }
  createBuffers(vertices, indices);
}

void SimpleModel::createBuffers(
    const std::vector<SimpleModel::Vertex>& vertices,
    const std::vector<uint32_t>& indices) {
  vertexBuffer = std::make_unique<vgeu::VgeuBuffer>(
      allocator, sizeof(SimpleModel::Vertex), vertices.size(),
      vk::BufferUsageFlagBits::eVertexBuffer |
          vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

  indexBuffer = std::make_unique<vgeu::VgeuBuffer>(
      allocator, sizeof(uint32_t), indices.size(),
      vk::BufferUsageFlagBits::eIndexBuffer |
          vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

  vgeu::VgeuBuffer vertexStagingBuffer(
      allocator, sizeof(SimpleModel::Vertex), vertices.size(),
      vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);
  std::memcpy(vertexStagingBuffer.getMappedData(), vertices.data(),
              vertexStagingBuffer.getBufferSize());

  vgeu::VgeuBuffer indexStagingBuffer(
      allocator, sizeof(uint32_t), indices.size(),
      vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);
  std::memcpy(indexStagingBuffer.getMappedData(), indices.data(),
              indexStagingBuffer.getBufferSize());

  vgeu::oneTimeSubmit(
      device, commandPool, transferQueue,
      [&](const vk::raii::CommandBuffer& cmdBuffer) {
        cmdBuffer.copyBuffer(
            vertexStagingBuffer.getBuffer(), vertexBuffer->getBuffer(),
            vk::BufferCopy(0, 0, vertexStagingBuffer.getBufferSize()));
        cmdBuffer.copyBuffer(
            indexStagingBuffer.getBuffer(), indexBuffer->getBuffer(),
            vk::BufferCopy(0, 0, indexStagingBuffer.getBufferSize()));
      });
}

uint32_t ModelInstance::getVertexCount() const {
  uint32_t vertexCount;
  if (model)
    vertexCount = model->getVertexCount();
  else if (simpleModel)
    vertexCount = simpleModel->vertexBuffer->getInstanceCount();
  else
    assert(false && "not defined ModelInstance type");
  return vertexCount;
}

SpatialHash::SpatialHash(const double spacing, const uint32_t maxNumObjects)
    : spacing(spacing),
      tableSize(2 * maxNumObjects),
      cellStart(tableSize + 1),
      cellEntries(maxNumObjects) {
  separator.resize(0);
  separator.push_back(0);
  objectIndex = 0;
}

void SpatialHash::resetTable() {
  separator.resize(0);
  separator.push_back(0);
  objectIndex = 0;
  for (auto i = 0; i < cellStart.size(); i++) {
    cellStart[i] = 0;
  }
}

void SpatialHash::addPos(const std::vector<glm::dvec3>& positions) {
  for (const auto& pos : positions) {
    uint32_t h = hashPos(pos);
    cellStart[h]++;
  }
  separator.push_back(separator.back() + positions.size());
}

void SpatialHash::createPartialSum() {
  uint32_t start = 0;
  for (auto i = 0; i < tableSize; i++) {
    start += cellStart[i];
    cellStart[i] = start;
  }
  cellStart[tableSize] = start;
}

void SpatialHash::addTableEntries(const std::vector<glm::dvec3>& positions) {
  for (auto i = 0; i < positions.size(); i++) {
    const auto& pos = positions[i];
    uint32_t h = hashPos(pos);
    cellStart[h]--;
    cellEntries[cellStart[h]] = std::make_pair(objectIndex, i);
  }
  objectIndex++;
}

void SpatialHash::queryTri(
    const glm::dvec4& aabb,
    std::vector<std::pair<uint32_t, uint32_t>>& queryIds) {
  queryIds.resize(0);
  // aabb-> cellCoords -> hash -> start, end
  // -> convert to object, index pair by binary search
  int x0 = discreteCoord(aabb.x);
  int x1 = discreteCoord(aabb.y);
  int y0 = discreteCoord(aabb.z);
  int y1 = discreteCoord(aabb.w);
  int z0 = 0;
  int z1 = 0;
  for (auto xi = x0; xi <= x1; xi++) {
    for (auto yi = y0; yi <= y1; yi++) {
      for (auto zi = z0; zi <= z1; zi++) {
        uint32_t h = hashDiscreteCoords(xi, yi, zi);
        uint32_t start = cellStart[h];
        uint32_t end = cellStart[h + 1];
        for (auto i = start; i < end; i++) {
          queryIds.push_back(cellEntries[i]);
        }
      }
    }
  }
}

uint32_t SpatialHash::hashDiscreteCoords(const int xi, const int yi,
                                         const int zi) {
  size_t seed = 0;
  vgeu::hashCombine(seed, xi, yi, zi);
  return static_cast<uint32_t>(seed) % tableSize;
}
int SpatialHash::discreteCoord(const double coord) {
  return static_cast<int>(std::floor(coord / spacing));
}
uint32_t SpatialHash::hashPos(const glm::dvec3& pos) {
  return hashDiscreteCoords(discreteCoord(pos.x), discreteCoord(pos.y),
                            discreteCoord(pos.z));
}

Cloth::Cloth(const vk::raii::Device& device, VmaAllocator allocator,
             const vk::raii::Queue& transferQueue,
             const uint32_t transferQueueFamilyIndex,
             const uint32_t computeQueueFamilyIndex,
             const vk::raii::CommandPool& commandPool,
             const vk::raii::DescriptorPool& descriptorPool,
             const vk::raii::DescriptorSetLayout& particleDescriptorSetLayout,
             const vk::raii::DescriptorSetLayout& constraintDescriptorSetLayout,
             const uint32_t framesInFlight)
    : device{device},
      allocator{allocator},
      transferQueue{transferQueue},
      transferQueueFamilyIndex{transferQueueFamilyIndex},
      computeQueueFamilyIndex{computeQueueFamilyIndex},
      commandPool{commandPool},
      descriptorPool{descriptorPool},
      particleDescriptorSetLayout{particleDescriptorSetLayout},
      constraintDescriptorSetLayout{constraintDescriptorSetLayout},
      framesInFlight{framesInFlight} {}

void Cloth::initParticlesData(const std::vector<vgeu::glTF::Vertex>& vertices,
                              const std::vector<uint32_t>& indices,
                              const glm::mat4& translate,
                              const glm::mat4& rotate, const glm::mat4& scale) {
  hasParticleBuffer = true;
  numParticles = vertices.size();
  numTris = indices.size() / 3;
  initialTransform = translate * rotate * scale;

  // TODO: initialize using compute shader cosidering performance.
  std::vector<ParticleCalculate> particlesCalculate;
  particlesCalculate.reserve(numParticles);
  for (auto i = 0; i < numParticles; i++) {
    glm::vec4 pos = vertices[i].pos;
    // NOTE: put skin index as w
    if (pos.w == -1.f) {
      pos.w = 1.f;
    }
    // else w as invMass
    ParticleCalculate newParticle{};
    newParticle.pos = pos;
    particlesCalculate.push_back(newParticle);
  }
  createParticleStorageBuffers(particlesCalculate);
  createParticleDescriptorSets();
}

void Cloth::initDistConstraintsData(const uint32_t numX, const uint32_t numY) {
  assert(numX * numY == numParticles);
  numPasses = 5;
  passSizes.resize(numPasses);
  // stretch x
  passSizes[0] = (numX / 2) * numY;
  passSizes[1] = ((numX - 1) / 2) * numY;
  // stretch y
  passSizes[2] = numX * (numY / 2);
  passSizes[3] = numX * ((numY - 1) / 2);
  // shear and bending constraints
  passSizes[4] = 2 * (numX - 1) * (numY - 1);
  passSizes[4] += (numX - 2) * numY + numX * (numY - 2);
  passIndependent.assign(5, true);
  passIndependent[4] = false;

  numConstraints = 0;
  for (auto n : passSizes) {
    numConstraints += n;
  }
  std::vector<DistConstraint> distConstraints;
  distConstraints.reserve(numConstraints);
  // stretch x
  for (auto isOdd = 0; isOdd < 2; isOdd++) {
    for (auto xi = 0; xi < (numX - isOdd) / 2; xi++) {
      for (auto yi = 0; yi < numY; yi++) {
        glm::ivec2 ids(yi * numX + xi * 2 + isOdd,
                       yi * numX + xi * 2 + isOdd + 1);
        distConstraints.push_back({ids, 0.f});
      }
    }
  }
  // stretch y
  for (auto isOdd = 0; isOdd < 2; isOdd++) {
    for (auto xi = 0; xi < numX; xi++) {
      for (auto yi = 0; yi < (numY - isOdd) / 2; yi++) {
        glm::ivec2 ids((2 * yi + isOdd) * numX + xi,
                       (2 * yi + isOdd + 1) * numX + xi);
        distConstraints.push_back({ids, 0.f});
      }
    }
  }
  // shearing
  for (auto xi = 0; xi < numX - 1; xi++) {
    for (auto yi = 0; yi < numY - 1; yi++) {
      {
        glm::ivec2 ids(yi * numX + xi, (yi + 1) * numX + (xi + 1));
        distConstraints.push_back({ids, 0.f});
      }
      {
        glm::ivec2 ids(yi * numX + (xi + 1), (yi + 1) * numX + xi);
        distConstraints.push_back({ids, 0.f});
      }
    }
  }
  // bending x
  for (auto xi = 0; xi < numX - 2; xi++) {
    for (auto yi = 0; yi < numY; yi++) {
      glm::ivec2 ids(yi * numX + xi, yi * numX + (xi + 2));
      distConstraints.push_back({ids, 0.f});
    }
  }
  // bending y
  for (auto xi = 0; xi < numX; xi++) {
    for (auto yi = 0; yi < numY - 2; yi++) {
      glm::ivec2 ids(yi * numX + xi, (yi + 2) * numX + xi);
      distConstraints.push_back({ids, 0.f});
    }
  }

  // TODO: pre compute rest length in compute shader

  createDistConstraintStorageBuffers(distConstraints);
  createDistConstraintDescriptorSets();
}

void Cloth::initDistConstraintsData(
    const std::vector<DistConstraint>& distConstraints) {
  // TODO: not implemented yet
}

void Cloth::integrate(const uint32_t frameIndex,
                      const vk::raii::CommandBuffer& cmdBuffer) {
  // TODO: not implemented yet
}

void Cloth::solveConstraints(const uint32_t frameIndex,
                             const vk::raii::CommandBuffer& cmdBuffer) {
  // TODO: not implemented yet
}

void Cloth::updateVel(const uint32_t frameIndex,
                      const vk::raii::CommandBuffer& cmdBuffer) {
  // TODO: not implemented yet
}

void Cloth::updateMesh(const uint32_t frameIndex,
                       const vk::raii::CommandBuffer& cmdBuffer) {
  // TODO: not implemented yet
}

void Cloth::createParticleStorageBuffers(
    const std::vector<ParticleCalculate>& particlesCalculate) {
  // buffer creation
  calculateSBs.resize(framesInFlight);
  for (auto i = 0; i < calculateSBs.size(); i++) {
    calculateSBs[i] = std::make_unique<vgeu::VgeuBuffer>(
        allocator, sizeof(ParticleCalculate), numParticles,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
  }

  // NOTE: numbers. particles with normal
  renderSBs.resize(framesInFlight);
  for (auto i = 0; i < renderSBs.size(); i++) {
    renderSBs[i] = std::make_unique<vgeu::VgeuBuffer>(
        allocator, sizeof(ParticleRender), numTris * 3,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eVertexBuffer,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
  }

  // copy particlesCalculate, only for invMass
  {
    vgeu::VgeuBuffer stagingBuffer(
        allocator, sizeof(ParticleCalculate), numParticles,
        vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    std::memcpy(stagingBuffer.getMappedData(), particlesCalculate.data(),
                stagingBuffer.getBufferSize());

    vgeu::oneTimeSubmit(
        device, commandPool, transferQueue,
        [&](const vk::raii::CommandBuffer& cmdBuffer) {
          std::vector<const vgeu::VgeuBuffer*> targetBufferPtrs;
          for (size_t i = 0; i < calculateSBs.size(); i++) {
            cmdBuffer.copyBuffer(
                stagingBuffer.getBuffer(), calculateSBs[i]->getBuffer(),
                vk::BufferCopy(0, 0, stagingBuffer.getBufferSize()));
            targetBufferPtrs.push_back(calculateSBs[i].get());
          }
          // release
          vgeu::addQueueFamilyOwnershipTransferBarriers(
              transferQueueFamilyIndex, computeQueueFamilyIndex, cmdBuffer,
              targetBufferPtrs, vk::AccessFlagBits::eTransferWrite,
              vk::AccessFlags{}, vk::PipelineStageFlagBits::eTransfer,
              vk::PipelineStageFlagBits::eBottomOfPipe);
        });
  }
}

void Cloth::createParticleDescriptorSets() {
  vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                          *particleDescriptorSetLayout);
  particleDescriptorSets.reserve(framesInFlight);
  for (int i = 0; i < framesInFlight; i++) {
    particleDescriptorSets.push_back(
        std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
  }
  std::vector<vk::DescriptorBufferInfo> calculateBufferInfos;
  calculateBufferInfos.reserve(particleDescriptorSets.size());
  std::vector<vk::DescriptorBufferInfo> renderBufferInfos;
  renderBufferInfos.reserve(particleDescriptorSets.size());
  for (int i = 0; i < particleDescriptorSets.size(); i++) {
    calculateBufferInfos.push_back(calculateSBs[i]->descriptorInfo());
    renderBufferInfos.push_back(renderSBs[i]->descriptorInfo());
  }

  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  writeDescriptorSets.reserve(particleDescriptorSets.size() * 3);
  for (int i = 0; i < particleDescriptorSets.size(); i++) {
    int prevFrameIdx = (i - 1 + framesInFlight) % framesInFlight;
    writeDescriptorSets.emplace_back(*particleDescriptorSets[i], 0, 0,
                                     vk::DescriptorType::eStorageBuffer,
                                     nullptr, calculateBufferInfos[i]);
    writeDescriptorSets.emplace_back(*particleDescriptorSets[i], 1, 0,
                                     vk::DescriptorType::eStorageBuffer,
                                     nullptr, renderBufferInfos[i]);
    writeDescriptorSets.emplace_back(
        *particleDescriptorSets[i], 2, 0, vk::DescriptorType::eStorageBuffer,
        nullptr, calculateBufferInfos[prevFrameIdx]);
  }
  device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

void Cloth::createDistConstraintStorageBuffers(
    const std::vector<DistConstraint>& distConstraints) {
  constraintSBs = std::make_unique<vgeu::VgeuBuffer>(
      allocator, sizeof(DistConstraint), numConstraints,
      vk::BufferUsageFlagBits::eStorageBuffer |
          vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);

  // copy
  {
    vgeu::VgeuBuffer stagingBuffer(
        allocator, sizeof(DistConstraint), numConstraints,
        vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    std::memcpy(stagingBuffer.getMappedData(), distConstraints.data(),
                stagingBuffer.getBufferSize());

    vgeu::oneTimeSubmit(
        device, commandPool, transferQueue,
        [&](const vk::raii::CommandBuffer& cmdBuffer) {
          cmdBuffer.copyBuffer(
              stagingBuffer.getBuffer(), constraintSBs->getBuffer(),
              vk::BufferCopy(0, 0, stagingBuffer.getBufferSize()));
        });
  }
}

void Cloth::createDistConstraintDescriptorSets() {
  vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                          *constraintDescriptorSetLayout);
  constraintDescriptorSet =
      std::move(vk::raii::DescriptorSets(device, allocInfo).front());

  std::vector<vk::DescriptorBufferInfo> bufferInfos;
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  bufferInfos.push_back(constraintSBs->descriptorInfo());
  writeDescriptorSets.emplace_back(*constraintDescriptorSet, 0, 0,
                                   vk::DescriptorType::eStorageBuffer, nullptr,
                                   bufferInfos.back());

  device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()