#include "pbd.hpp"

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
uint32_t mapToIntegrateSteps(uint32_t integrator) {
  uint32_t numSteps = 1u;
  switch (integrator) {
    case 1:
    case 2:
    case 4:
      numSteps = integrator;
      break;
    case 5:
    case 6:
    case 8:
      numSteps = integrator - 4u;
      break;
    default:
      assert(false && "failed: unsupported integrator type");
  }
  return numSteps;
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
}  // namespace
namespace vge {
VgeExample::VgeExample() : VgeBase() { title = "Particle Example"; }
VgeExample::~VgeExample() {}

void VgeExample::initVulkan() {
  cameraController.moveSpeed = opts.moveSpeed;
  // camera setup
  camera.setViewTarget(glm::vec3{0.f, -20.f, -20.f}, glm::vec3{0.f, 0.f, 0.f});
  camera.setPerspectiveProjection(
      glm::radians(60.f),
      static_cast<float>(width) / (static_cast<float>(height)), 0.1f, 256.f);
  // NOTE: coordinate space in world
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
  graphics.queueFamilyIndex = queueFamilyIndices.graphics;
  compute.queueFamilyIndex = queueFamilyIndices.compute;
  createStorageBuffers();
  createVertexSCI();
  prepareGraphics();
  prepared = true;
}

void VgeExample::prepareGraphics() {
  createUniformBuffers();
  createDescriptorSetLayout();
  createDescriptorSets();
  createPipelines();
  // create semaphore for compute-graphics sync
  {
    std::vector<vk::Semaphore> semaphoresToSignal;
    semaphoresToSignal.reserve(MAX_CONCURRENT_FRAMES);
    graphics.semaphores.reserve(MAX_CONCURRENT_FRAMES);
    for (size_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      vk::raii::Semaphore& semaphore =
          graphics.semaphores.emplace_back(device, vk::SemaphoreCreateInfo());
      semaphoresToSignal.push_back(*semaphore);
    }
    // initial signaled

    vk::SubmitInfo submitInfo({}, {}, {}, semaphoresToSignal);
    queue.submit(submitInfo);
    queue.waitIdle();
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
    addModelInstance(modelInstance);
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
    addModelInstance(modelInstance);
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
    modelInstance.animationIndex = -1;
    addModelInstance(modelInstance);
  }

  std::shared_ptr<vgeu::glTF::Model> fox3;
  fox3 = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  fox3->additionalBufferUsageFlags = vk::BufferUsageFlagBits::eStorageBuffer;
  fox3->loadFromFile(getAssetsPath() + "/models/fox-normal/fox-normal.gltf",
                     glTFLoadingFlags);
  {
    ModelInstance modelInstance{};
    modelInstance.model = fox3;
    modelInstance.name = "fox3";
    modelInstance.animationIndex = 1;
    addModelInstance(modelInstance);
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
    addModelInstance(modelInstance);
  }

  std::shared_ptr<vgeu::glTF::Model> bone1;
  bone1 = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  bone1->additionalBufferUsageFlags = vk::BufferUsageFlagBits::eStorageBuffer;
  bone1->loadFromFile(getAssetsPath() + "/models/bone.gltf", glTFLoadingFlags);
  {
    ModelInstance modelInstance{};
    modelInstance.model = bone1;
    modelInstance.name = "bone1";
    addModelInstance(modelInstance);
  }

  std::shared_ptr<vgeu::glTF::Model> bone4;
  bone4 = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  bone4->additionalBufferUsageFlags = vk::BufferUsageFlagBits::eStorageBuffer;
  bone4->loadFromFile(getAssetsPath() + "/models/bone4.gltf", glTFLoadingFlags);
  {
    ModelInstance modelInstance{};
    modelInstance.model = bone4;
    modelInstance.name = "bone4";
    addModelInstance(modelInstance);
  }

  std::shared_ptr<vgeu::glTF::Model> bone5;
  bone5 = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  bone5->additionalBufferUsageFlags = vk::BufferUsageFlagBits::eStorageBuffer;
  bone5->loadFromFile(getAssetsPath() + "/models/bone5.gltf", glTFLoadingFlags);
  {
    ModelInstance modelInstance{};
    modelInstance.model = bone5;
    modelInstance.name = "bone5";
    addModelInstance(modelInstance);
  }

  std::shared_ptr<vgeu::glTF::Model> dutchShipMedium;
  dutchShipMedium = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  dutchShipMedium->additionalBufferUsageFlags =
      vk::BufferUsageFlagBits::eStorageBuffer;
  dutchShipMedium->loadFromFile(
      getAssetsPath() +
          "/models/dutch_ship_medium_1k/dutch_ship_medium_1k.gltf",
      glTFLoadingFlags);
  {
    ModelInstance modelInstance{};
    modelInstance.model = dutchShipMedium;
    modelInstance.name = "dutchShipMedium";
    addModelInstance(modelInstance);
  }
}

void VgeExample::createStorageBuffers() {
  std::vector<Particle> particles;

  // tail
  {
    if (tailSize > 0) {
      tailData.resize(numParticles * tailSize);
      for (size_t i = 0; i < numParticles; i++) {
        for (size_t j = 0; j < tailSize; j++) {
          tailData[i * tailSize + j].pos = particles[i].pos;
          tailData[i * tailSize + j].pos.w = particles[i].vel.w;
        }
      }
    } else {
      // dummy data
      tailData.resize(1);
    }
    vgeu::VgeuBuffer tailStagingBuffer(
        globalAllocator->getAllocator(), sizeof(TailElt), tailData.size(),
        vk::BufferUsageFlagBits::eVertexBuffer |
            vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    std::memcpy(tailStagingBuffer.getMappedData(), tailData.data(),
                tailStagingBuffer.getBufferSize());

    tailBuffers.resize(MAX_CONCURRENT_FRAMES);
    for (size_t i = 0; i < tailBuffers.size(); i++) {
      tailBuffers[i] = std::make_unique<vgeu::VgeuBuffer>(
          globalAllocator->getAllocator(), sizeof(TailElt), tailData.size(),
          vk::BufferUsageFlagBits::eVertexBuffer |
              vk::BufferUsageFlagBits::eStorageBuffer |
              vk::BufferUsageFlagBits::eTransferDst,
          VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    }

    vgeu::oneTimeSubmit(
        device, commandPool, queue,
        [&](const vk::raii::CommandBuffer& cmdBuffer) {
          for (size_t i = 0; i < tailBuffers.size(); i++) {
            cmdBuffer.copyBuffer(
                tailStagingBuffer.getBuffer(), tailBuffers[i]->getBuffer(),
                vk::BufferCopy(0, 0, tailStagingBuffer.getBufferSize()));
            // TODO: pipeline barrier to the compute queue?
            // TODO: check spec and exs for ownership transfer
            // release
            if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
              vk::BufferMemoryBarrier bufferBarrier(
                  vk::AccessFlagBits::eTransferWrite, vk::AccessFlags{},
                  graphics.queueFamilyIndex, compute.queueFamilyIndex,
                  tailBuffers[i]->getBuffer(), 0ull,
                  tailBuffers[i]->getBufferSize());
              cmdBuffer.pipelineBarrier(
                  vk::PipelineStageFlagBits::eTransfer,
                  vk::PipelineStageFlagBits::eBottomOfPipe,
                  vk::DependencyFlags{}, nullptr, bufferBarrier, nullptr);
            }
          }
        });
  }

  // index buffer
  {
    if (tailSize > 0) {
      tailIndices.resize(numParticles * (tailSize + 1));
      for (size_t i = 0; i < numParticles; i++) {
        for (size_t j = 0; j < tailSize; j++) {
          tailIndices[i * (tailSize + 1) + j] = i * tailSize + j;
        }
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineInputAssemblyStateCreateInfo.html
        // 0xffffffff
        tailIndices[i * (tailSize + 1) + tailSize] = static_cast<uint32_t>(-1);
      }
    } else {
      tailIndices.resize(1);
    }
    vgeu::VgeuBuffer tailIndexStagingBuffer(
        globalAllocator->getAllocator(), sizeof(uint32_t), tailIndices.size(),
        vk::BufferUsageFlagBits::eIndexBuffer |
            vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    std::memcpy(tailIndexStagingBuffer.getMappedData(), tailIndices.data(),
                tailIndexStagingBuffer.getBufferSize());

    tailIndexBuffer = std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(uint32_t), tailIndices.size(),
        vk::BufferUsageFlagBits::eIndexBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
    vgeu::oneTimeSubmit(
        device, commandPool, queue,
        [&](const vk::raii::CommandBuffer& cmdBuffer) {
          cmdBuffer.copyBuffer(
              tailIndexStagingBuffer.getBuffer(), tailIndexBuffer->getBuffer(),
              vk::BufferCopy(0, 0, tailIndexStagingBuffer.getBufferSize()));
        });
  }
}

void VgeExample::createVertexSCI() {
  // vertex binding and attribute descriptions
  vertexInfos.bindingDescriptions.emplace_back(0 /*binding*/, sizeof(Particle),
                                               vk::VertexInputRate::eVertex);

  vertexInfos.attributeDescriptions.emplace_back(
      0 /*location*/, 0 /* binding */, vk::Format::eR32G32B32A32Sfloat,
      offsetof(Particle, pos));
  vertexInfos.attributeDescriptions.emplace_back(
      1 /*location*/, 0 /* binding */, vk::Format::eR32G32B32A32Sfloat,
      offsetof(Particle, vel));

  // for Runge-Kutta explicit method. RK4. but not used as vertex

  vertexInfos.vertexInputSCI = vk::PipelineVertexInputStateCreateInfo(
      vk::PipelineVertexInputStateCreateFlags{},
      vertexInfos.bindingDescriptions, vertexInfos.attributeDescriptions);

  // tail
  tailVertexInfos.bindingDescriptions.emplace_back(
      0 /*binding*/, sizeof(TailElt), vk::VertexInputRate::eVertex);

  tailVertexInfos.attributeDescriptions.emplace_back(
      0 /*location*/, 0 /* binding */, vk::Format::eR32G32B32A32Sfloat,
      offsetof(TailElt, pos));

  tailVertexInfos.vertexInputSCI = vk::PipelineVertexInputStateCreateInfo(
      vk::PipelineVertexInputStateCreateFlags{},
      tailVertexInfos.bindingDescriptions,
      tailVertexInfos.attributeDescriptions);
}

void VgeExample::setupDynamicUbo() {
  const float foxScale = 0.1f;
  glm::vec3 up{0.f, -1.f, 0.f};
  glm::vec3 right{1.f, 0.f, 0.f};
  glm::vec3 forward{0.f, 0.f, 1.f};
  dynamicUbo.resize(modelInstances.size());
  {
    size_t instanceIndex = findInstances("fox0")[0];
    dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, glm::vec3{-3.f, 0.f, 0.f});
    dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        dynamicUbo[instanceIndex].modelMatrix, glm::radians(180.f), up);
    // FlipY manually
    dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{foxScale, -foxScale, foxScale});
    dynamicUbo[instanceIndex].modelColor = glm::vec4{1.0f, 0.f, 0.f, 0.3f};
    dynamicUbo[instanceIndex].numVertices =
        modelInstances[instanceIndex].model->getVertexCount();
    dynamicUbo[instanceIndex].numIndices =
        modelInstances[instanceIndex].model->getIndexCount();
  }
  {
    size_t instanceIndex = findInstances("fox1")[0];
    dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, glm::vec3{3.f, 0.f, 0.f});
    dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        dynamicUbo[instanceIndex].modelMatrix, glm::radians(0.f), up);
    // FlipY manually
    dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{foxScale, -foxScale, foxScale});
    dynamicUbo[instanceIndex].modelColor = glm::vec4{0.0f, 0.f, 1.f, 0.3f};
    dynamicUbo[instanceIndex].numVertices =
        modelInstances[instanceIndex].model->getVertexCount();
    dynamicUbo[instanceIndex].numIndices =
        modelInstances[instanceIndex].model->getIndexCount();
  }
  {
    size_t instanceIndex = findInstances("fox2")[0];
    dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, glm::vec3{-2.f, 0.f, 0.f});
    dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        dynamicUbo[instanceIndex].modelMatrix, glm::radians(180.f), up);
    // FlipY manually
    dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{foxScale, -foxScale, foxScale});
    // default
    dynamicUbo[instanceIndex].modelColor = glm::vec4{0.f};
    dynamicUbo[instanceIndex].numVertices =
        modelInstances[instanceIndex].model->getVertexCount();
    dynamicUbo[instanceIndex].numIndices =
        modelInstances[instanceIndex].model->getIndexCount();
  }
  {
    size_t instanceIndex = findInstances("fox3")[0];
    dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, glm::vec3{+2.f, 0.f, 0.f});
    dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        dynamicUbo[instanceIndex].modelMatrix, glm::radians(180.f), up);
    // FlipY manually
    dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{foxScale, -foxScale, foxScale});
    dynamicUbo[instanceIndex].modelColor = glm::vec4{0.f, 1.f, 1.f, 0.3f};
    dynamicUbo[instanceIndex].numVertices =
        modelInstances[instanceIndex].model->getVertexCount();
    dynamicUbo[instanceIndex].numIndices =
        modelInstances[instanceIndex].model->getIndexCount();
  }
  {
    float appleScale = 100.f;
    size_t instanceIndex = findInstances("apple1")[0];
    // FlipY manually
    dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{appleScale, -appleScale, appleScale});
    dynamicUbo[instanceIndex].numVertices =
        modelInstances[instanceIndex].model->getVertexCount();
    dynamicUbo[instanceIndex].numIndices =
        modelInstances[instanceIndex].model->getIndexCount();
  }
  float boneScale = 5.f;
  {
    size_t instanceIndex = findInstances("bone1")[0];
    dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        dynamicUbo[instanceIndex].modelMatrix, glm::radians(90.f), forward);
    // FlipY manually
    dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{boneScale, -boneScale, boneScale});
    dynamicUbo[instanceIndex].numVertices =
        modelInstances[instanceIndex].model->getVertexCount();
    dynamicUbo[instanceIndex].numIndices =
        modelInstances[instanceIndex].model->getIndexCount();
  }
  {
    size_t instanceIndex = findInstances("bone4")[0];
    dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        dynamicUbo[instanceIndex].modelMatrix, glm::radians(90.f), right);
    // FlipY manually
    dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{boneScale, -boneScale, boneScale});
    dynamicUbo[instanceIndex].numVertices =
        modelInstances[instanceIndex].model->getVertexCount();
    dynamicUbo[instanceIndex].numIndices =
        modelInstances[instanceIndex].model->getIndexCount();
  }
  {
    size_t instanceIndex = findInstances("bone5")[0];
    // FlipY manually
    dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{boneScale, -boneScale, boneScale});
    dynamicUbo[instanceIndex].numVertices =
        modelInstances[instanceIndex].model->getVertexCount();
    dynamicUbo[instanceIndex].numIndices =
        modelInstances[instanceIndex].model->getIndexCount();
  }
  float shipScale = 1.f;
  {
    size_t instanceIndex = findInstances("dutchShipMedium")[0];
    dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, glm::vec3{0.f, 0.f, 0.f});
    dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        dynamicUbo[instanceIndex].modelMatrix, glm::radians(180.f), up);
    // FlipY manually
    dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{shipScale, -shipScale, shipScale});
    // default
    dynamicUbo[instanceIndex].modelColor = glm::vec4{0.f};
    dynamicUbo[instanceIndex].numVertices =
        modelInstances[instanceIndex].model->getVertexCount();
    dynamicUbo[instanceIndex].numIndices =
        modelInstances[instanceIndex].model->getIndexCount();
  }
}

void VgeExample::createUniformBuffers() {
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
                         MAX_CONCURRENT_FRAMES * 2 +
                             MAX_CONCURRENT_FRAMES * modelInstances.size() * 2 +
                             MAX_CONCURRENT_FRAMES * 2);
  // NOTE: need to check flag
  vk::DescriptorPoolCreateInfo descriptorPoolCI(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      MAX_CONCURRENT_FRAMES * 3 /*set globalUBO, dynamicUBO, computeUbo*/ +
          MAX_CONCURRENT_FRAMES *
              modelInstances.size() /*skin & animated vertex ssbo*/,
      poolSizes);
  descriptorPool = vk::raii::DescriptorPool(device, descriptorPoolCI);
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

  vk::PipelineLayoutCreateInfo pipelineLayoutCI({}, setLayouts);
  graphics.pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutCI);
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
}

void VgeExample::createPipelines() {
  vk::PipelineVertexInputStateCreateInfo vertexInputSCI =
      vertexInfos.vertexInputSCI;

  vk::PipelineInputAssemblyStateCreateInfo inputAssemblySCI(
      vk::PipelineInputAssemblyStateCreateFlags(),
      vk::PrimitiveTopology::ePointList);

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
      vk::PipelineDepthStencilStateCreateFlags(), false /*depthTestEnable*/,
      true, vk::CompareOp::eLessOrEqual, false, false, stencilOpState,
      stencilOpState);

  vk::PipelineColorBlendAttachmentState colorBlendAttachmentState(
      true, vk::BlendFactor::eOne, vk::BlendFactor::eOne, vk::BlendOp::eAdd,
      vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eDstAlpha, vk::BlendOp::eAdd,
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
      vgeu::readFile(getShadersPath() + "/particle/particle.vert.spv");
  auto fragCode =
      vgeu::readFile(getShadersPath() + "/particle/particle.frag.spv");
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
  {
    vertexInputSCI.setVertexBindingDescriptions(
        tailVertexInfos.bindingDescriptions);
    vertexInputSCI.setVertexAttributeDescriptions(
        tailVertexInfos.attributeDescriptions);
    // primitive restart enabled
    inputAssemblySCI.topology = vk::PrimitiveTopology::eLineStrip;
    inputAssemblySCI.primitiveRestartEnable = true;
    rasterizationSCI.polygonMode = vk::PolygonMode::eFill;
    vertCode = vgeu::readFile(getShadersPath() + "/particle/tail.vert.spv");
    fragCode = vgeu::readFile(getShadersPath() + "/particle/tail.frag.spv");
    // NOTE: after pipeline creation, shader modules can be destroyed.
    vertShaderModule = vgeu::createShaderModule(device, vertCode);
    fragShaderModule = vgeu::createShaderModule(device, fragCode);
    shaderStageCIs[0] = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eVertex,
        *vertShaderModule, "main", nullptr);
    shaderStageCIs[1] = vk::PipelineShaderStageCreateInfo(
        vk::PipelineShaderStageCreateFlags(),
        vk::ShaderStageFlagBits::eFragment, *fragShaderModule, "main", nullptr);
    tailPipeline =
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

  // calculate tail
  updateTailData();
  // update uniform buffers;
  updateDynamicUbo();
  updateComputeUbo();
  updateGraphicsUbo();

  // TODO: Fence and compute recording order

  {
    // draw cmds recording or command buffers should be built already.
    buildCommandBuffers();

    std::vector<vk::PipelineStageFlags> graphicsWaitDstStageMasks{
        vk::PipelineStageFlagBits::eVertexInput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
    };

    std::vector<vk::Semaphore> graphicsWaitSemaphores{
        *presentCompleteSemaphores[currentFrameIndex],
    };

    std::vector<vk::Semaphore> graphicsSignalSemaphore{
        *graphics.semaphores[currentFrameIndex],
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

  // acquire barrier compute -> graphics
  if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
    std::vector<vk::BufferMemoryBarrier> bufferBarriers;
    bufferBarriers.emplace_back(
        vk::AccessFlags{}, vk::AccessFlagBits::eVertexAttributeRead,
        compute.queueFamilyIndex, graphics.queueFamilyIndex,
        compute.storageBuffers[currentFrameIndex]->getBuffer(), 0ull,
        compute.storageBuffers[currentFrameIndex]->getBufferSize());
    bufferBarriers.emplace_back(
        vk::AccessFlags{}, vk::AccessFlagBits::eVertexAttributeRead,
        compute.queueFamilyIndex, graphics.queueFamilyIndex,
        tailBuffers[currentFrameIndex]->getBuffer(), 0ull,
        tailBuffers[currentFrameIndex]->getBufferSize());

    drawCmdBuffers[currentFrameIndex].pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eVertexInput, vk::DependencyFlags{}, nullptr,
        bufferBarriers, nullptr);
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

  // particles
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

    vk::DeviceSize offset(0);
    // TODO: bind and draw
  }

  // tail
  if (tailSize > 0) {
    drawCmdBuffers[currentFrameIndex].setLineWidth(opts.lineWidth);
    drawCmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eGraphics, *tailPipeline);
    vk::DeviceSize offset(0);
    drawCmdBuffers[currentFrameIndex].bindVertexBuffers(
        0, tailBuffers[currentFrameIndex]->getBuffer(), offset);
    drawCmdBuffers[currentFrameIndex].bindIndexBuffer(
        tailIndexBuffer->getBuffer(), 0, vk::IndexType::eUint32);
    drawCmdBuffers[currentFrameIndex].drawIndexed(numParticles * (tailSize + 1),
                                                  1, 0, 0, 0);
  }

  // UI overlay draw
  drawUI(drawCmdBuffers[currentFrameIndex]);

  // end renderpass
  drawCmdBuffers[currentFrameIndex].endRenderPass();

  // release graphics -> compute
  if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
    std::vector<vk::BufferMemoryBarrier> bufferBarriers;
    bufferBarriers.emplace_back(
        vk::AccessFlagBits::eVertexAttributeRead, vk::AccessFlags{},
        graphics.queueFamilyIndex, compute.queueFamilyIndex,
        compute.storageBuffers[currentFrameIndex]->getBuffer(), 0ull,
        compute.storageBuffers[currentFrameIndex]->getBufferSize());
    bufferBarriers.emplace_back(
        vk::AccessFlagBits::eVertexAttributeRead, vk::AccessFlags{},
        graphics.queueFamilyIndex, compute.queueFamilyIndex,
        tailBuffers[currentFrameIndex]->getBuffer(), 0ull,
        tailBuffers[currentFrameIndex]->getBufferSize());

    drawCmdBuffers[currentFrameIndex].pipelineBarrier(
        vk::PipelineStageFlagBits::eVertexInput,
        vk::PipelineStageFlagBits::eBottomOfPipe, vk::DependencyFlags{},
        nullptr, bufferBarriers, nullptr);
  }

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
void VgeExample::updateGraphicsUbo() {
  // CHECK: ubo update frequency.
  graphics.globalUbo.view = camera.getView();
  graphics.globalUbo.projection = camera.getProjection();
  graphics.globalUbo.inverseView = camera.getInverseView();
  graphics.globalUbo.screenDim = glm::vec2{
      static_cast<float>(width),
      static_cast<float>(height),
  };
  graphics.globalUbo.tailInfo.x = static_cast<float>(tailSize);
  graphics.globalUbo.tailInfo.y = static_cast<float>(opts.tailIntensity);
  graphics.globalUbo.tailInfo.z = static_cast<float>(opts.tailFadeOut);
  graphics.globalUbo.pointSize.x =
      std::min(opts.pointSize[0], opts.pointSize[1]);
  graphics.globalUbo.pointSize.y =
      std::max(opts.pointSize[0], opts.pointSize[1]);
  std::memcpy(graphics.globalUniformBuffers[currentFrameIndex]->getMappedData(),
              &graphics.globalUbo, sizeof(GlobalUbo));
}

void VgeExample::updateComputeUbo() {
  compute.ubo.dt = paused ? 0.0f : frameTimer * opts.coefficientDeltaTime;
  compute.ubo.tailTimer = tailTimer;
  compute.ubo.tailSize = tailSize;

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
}

void VgeExample::updateDynamicUbo() {
  float animationTimer =
      (animationTime - animationLastTime) * opts.animationSpeed;
  // model move

  glm::vec3 up{0.f, -1.f, 0.f};
  // deg per sec;
  float rotationVelocity = 50.f;
  {
    size_t instanceIndex = findInstances("fox0")[0];
    dynamicUbo[instanceIndex].modelMatrix =
        glm::rotate(glm::mat4{1.f},
                    glm::radians(rotationVelocity) * animationTimer, up) *
        dynamicUbo[instanceIndex].modelMatrix;
  }
  {
    size_t instanceIndex = findInstances("fox1")[0];
    dynamicUbo[instanceIndex].modelMatrix =
        glm::rotate(glm::mat4{1.f},
                    glm::radians(rotationVelocity) * animationTimer, up) *
        dynamicUbo[instanceIndex].modelMatrix;
  }
  {
    size_t instanceIndex = findInstances("dutchShipMedium")[0];
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

void VgeExample::setupCommandLineParser(CLI::App& app) {
  VgeBase::setupCommandLineParser(app);
  app.add_option("--integrator, -i", integrator,
                 "Integrator Type 1 euler, "
                 "2 midpoint, "
                 "4 rk-4, "
                 "5 symplectic euler, "
                 "6 verlet")
      ->capture_default_str();
  app.add_option("--numParticles, --np", numParticles, "number of particles")
      ->capture_default_str();
  app.add_option("--numAttractors, --na", numAttractors, "number of attractors")
      ->capture_default_str();
  app.add_option("--rotationVelocity, --rv", rotationVelocity,
                 "initial y-axis rotation velocity )")
      ->capture_default_str();
  app.add_option("--gravity, -g", gravity,
                 "gravity constants / 500.0 for model attraction")
      ->capture_default_str();
  app.add_option("--power, -p", power,
                 "power constants / 0.75 for model attraction")
      ->capture_default_str();
  app.add_option("--soften, -s", soften,
                 "soften constants / 2.0 for model attraction")
      ->capture_default_str();
  app.add_option("--tailSize, --ts", tailSize, "tail size")
      ->capture_default_str();
  app.add_option("--tailSampleTime, --tst", tailSampleTime, "tail sample time")
      ->capture_default_str();
}

void VgeExample::updateTailData() {
  if (!paused) {
    tailTimer += frameTimer;
  }
  // NOTE: timer 0.0 -> update tail's head position.
  // if tailTimer == 0.0 when paused, tail disappears by this impl.
  if (tailTimer > opts.tailSampleTime || tailTimer < 0.f) {
    tailTimer = 0.f;
  }
}

void VgeExample::onUpdateUIOverlay() {
  if (uiOverlay->header("Inputs")) {
    ImGui::Text("Mouse Left: %s", mouseData.left ? "true" : "false");
    ImGui::Text("Mouse Middle: %s", mouseData.middle ? "true" : "false");
    ImGui::Text("Mouse Right: %s", mouseData.right ? "true" : "false");
    ImGui::Text("Mouse Pos: (%f, %f)", mouseData.mousePos.x,
                mouseData.mousePos.y);
    ImGui::Text("Click Data: (%f, %f, %f, %f)", compute.ubo.clickData.x,
                compute.ubo.clickData.y, compute.ubo.clickData.z,
                compute.ubo.clickData.w);
  }

  if (uiOverlay->header("Settings")) {
    if (ImGui::TreeNodeEx("Immediate", ImGuiTreeNodeFlags_DefaultOpen)) {
      uiOverlay->inputFloat("coefficientDeltaTime", &opts.coefficientDeltaTime,
                            0.001f, "%.3f");
      uiOverlay->inputFloat("tailSampleTime", &opts.tailSampleTime, 0.001f,
                            "%.3f");
      uiOverlay->inputFloat("tailIntensity", &opts.tailIntensity, 0.01f,
                            "%.2f");
      uiOverlay->inputFloat("tailFadeOut", &opts.tailFadeOut, 0.1f, "%.1f");
      ImGui::DragFloat2("Drag pointSize min/max", opts.pointSize, 1.f, 1.f,
                        128.f, "%.0f");
      uiOverlay->inputFloat("gravity", &opts.gravity, 0.001f, "%.3f");
      uiOverlay->inputFloat("power", &opts.power, 0.01f, "%.3f");
      uiOverlay->inputFloat("soften", &opts.soften, 0.0001f, "%.4f");

      uiOverlay->inputFloat("animationSpeed", &opts.animationSpeed, 0.001f,
                            "%.3f");

      if (uiOverlay->inputFloat("keyboardMoveSpeed", &opts.moveSpeed, 0.01f,
                                "%.3f")) {
        cameraController.moveSpeed = this->opts.moveSpeed;
      }
      uiOverlay->inputFloat("lineWidth", &opts.lineWidth, 0.1f, "%.3f");
      // binding model for model attraction
      for (const auto& item : instanceMap) {
        std::string caption =
            item.first + " / verts: " +
            std::to_string(
                modelInstances[item.second[0]].model->getVertexCount());
        uiOverlay->radioButton(caption.c_str(), &opts.bindingModel,
                               item.second[0]);
      }
      ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Initializers", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (uiOverlay->button("Restart")) {
        restart = true;
      }
      // attraction type
      uiOverlay->radioButton("gravitational attraction", &opts.attractionType,
                             0);
      uiOverlay->radioButton("model attraction", &opts.attractionType, 1);

      uiOverlay->sliderInt("numAttractors", &opts.numAttractors, 2, 6);
      ImGui::DragInt("Drag numParticles", &opts.numParticles, 16.f,
                     opts.numAttractors, kMaxNumParticles);
      if (ImGui::TreeNodeEx("colors", ImGuiTreeNodeFlags_DefaultOpen)) {
        for (size_t i = 0; i < opts.numAttractors; i++) {
          std::string caption = "colors" + std::to_string(i);
          uiOverlay->colorPicker(caption.c_str(), opts.colors[i].data());
        }
        ImGui::TreePop();
      }
      uiOverlay->inputInt("desiredSharedDataSize", &opts.desiredSharedDataSize,
                          64);
      uiOverlay->inputFloat("rotationVelocity", &opts.rotationVelocity,
                            0.000001f, "%.6f");
      uiOverlay->inputInt("tailSize", &opts.tailSize, 1);
      if (ImGui::TreeNodeEx("integrator", ImGuiTreeNodeFlags_DefaultOpen)) {
        uiOverlay->radioButton("euler", &opts.integrator, 1);
        ImGui::SameLine(150.0);
        uiOverlay->radioButton("euler-symplectic", &opts.integrator, 5);

        uiOverlay->radioButton("midpoint", &opts.integrator, 2);
        ImGui::SameLine(150.0);
        uiOverlay->radioButton("verlet", &opts.integrator, 6);

        uiOverlay->radioButton("rk-4", &opts.integrator, 4);
        ImGui::SameLine(150.0);
        uiOverlay->radioButton("4th-symplectic", &opts.integrator, 8);
        ImGui::TreePop();
      }

      ImGui::TreePop();
    }
  }
}
void VgeExample::setOptions(const std::optional<Options>& opts) {
  if (opts.has_value()) {
    this->opts = opts.value();
    // overwrite cli args for restart run
    numAttractors = static_cast<uint32_t>(this->opts.numAttractors);
    numParticles = static_cast<uint32_t>(this->opts.numParticles);
    rotationVelocity = this->opts.rotationVelocity;
    tailSampleTime = this->opts.tailSampleTime;
    tailSize = static_cast<uint32_t>(this->opts.tailSize);
    integrator = static_cast<uint32_t>(this->opts.integrator);
    cameraController.moveSpeed = this->opts.moveSpeed;
    desiredSharedDataSize =
        static_cast<uint32_t>(this->opts.desiredSharedDataSize);
  } else {
    // save cli args for initial run
    this->opts.numAttractors = static_cast<int32_t>(numAttractors);
    this->opts.numParticles = static_cast<int32_t>(numParticles);
    this->opts.rotationVelocity = rotationVelocity;
    this->opts.gravity = gravity;
    this->opts.power = power;
    this->opts.soften = soften;
    this->opts.tailSampleTime = tailSampleTime;
    this->opts.tailSize = static_cast<int32_t>(tailSize);
    this->opts.integrator = static_cast<int32_t>(integrator);
  }
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()