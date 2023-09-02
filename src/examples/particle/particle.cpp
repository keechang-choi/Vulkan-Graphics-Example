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
}  // namespace
namespace vge {
VgeExample::VgeExample() : VgeBase() { title = "Particle Example"; }
VgeExample::~VgeExample() {}

void VgeExample::initVulkan() {
  cameraController.moveSpeed = 10.f;
  // camera setup
  camera.setViewTarget(glm::vec3{0.f, -20.f, -.1f}, glm::vec3{0.f, 0.f, 0.f});
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
  graphics.queueFamilyIndex = queueFamilyIndices.graphics;
  compute.queueFamilyIndex = queueFamilyIndices.compute;
  createStorageBuffers();
  createVertexSCI();
  prepareGraphics();
  prepareCompute();
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

void VgeExample::prepareCompute() {
  setupDynamicUbo();
  // create ubo
  {
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

    // dynamic UBO
    alignedSizeDynamicUboElt = padUniformBufferSize(sizeof(DynamicUboElt));
    dynamicUniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
    for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
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

  // create queue
  compute.queue = vk::raii::Queue(device, compute.queueFamilyIndex, 0);

  // create descriptorSetLayout
  {
    std::vector<vk::DescriptorSetLayout> setLayouts;

    // set 0
    {
      std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;

      layoutBindings.emplace_back(0 /* binding */,
                                  vk::DescriptorType::eStorageBuffer, 1,
                                  vk::ShaderStageFlagBits::eCompute);
      layoutBindings.emplace_back(1 /* binding */,
                                  vk::DescriptorType::eStorageBuffer, 1,
                                  vk::ShaderStageFlagBits::eCompute);
      layoutBindings.emplace_back(2 /* binding */,
                                  vk::DescriptorType::eUniformBuffer, 1,
                                  vk::ShaderStageFlagBits::eCompute);

      vk::DescriptorSetLayoutCreateInfo layoutCI(
          vk::DescriptorSetLayoutCreateFlags{}, layoutBindings);
      compute.descriptorSetLayout =
          vk::raii::DescriptorSetLayout(device, layoutCI);
      setLayouts.push_back(*compute.descriptorSetLayout);
    }

    // set 1 dynamic ubo
    {
      vk::DescriptorSetLayoutBinding layoutBinding(
          0, vk::DescriptorType::eUniformBufferDynamic, 1,
          vk::ShaderStageFlagBits::eVertex);
      vk::DescriptorSetLayoutCreateInfo layoutCI({}, 1, &layoutBinding);
      dynamicUboDescriptorSetLayout =
          vk::raii::DescriptorSetLayout(device, layoutCI);
      setLayouts.push_back(*dynamicUboDescriptorSetLayout);
    }

    // create pipelineLayout
    vk::PipelineLayoutCreateInfo pipelineLayoutCI({}, setLayouts);
    compute.pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutCI);
  }

  // create descriptorSets
  {
    vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                            *compute.descriptorSetLayout);
    compute.descriptorSets.reserve(MAX_CONCURRENT_FRAMES);
    for (size_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      compute.descriptorSets.push_back(
          std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
    }

    std::vector<vk::DescriptorBufferInfo> storageBufferInfos;
    storageBufferInfos.reserve(compute.descriptorSets.size());
    std::vector<vk::DescriptorBufferInfo> uniformBufferInfos;
    uniformBufferInfos.reserve(compute.descriptorSets.size());
    for (size_t i = 0; i < compute.descriptorSets.size(); i++) {
      storageBufferInfos.push_back(compute.storageBuffers[i]->descriptorInfo());
      uniformBufferInfos.push_back(compute.uniformBuffers[i]->descriptorInfo());
    }
    std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
    writeDescriptorSets.reserve(compute.descriptorSets.size());
    for (int i = 0; i < compute.descriptorSets.size(); i++) {
      int prevFrameIdx =
          (i - 1 + MAX_CONCURRENT_FRAMES) % MAX_CONCURRENT_FRAMES;
      writeDescriptorSets.emplace_back(
          *compute.descriptorSets[i], 0 /*binding*/, 0,
          vk::DescriptorType::eStorageBuffer, nullptr,
          storageBufferInfos[prevFrameIdx]);
      writeDescriptorSets.emplace_back(
          *compute.descriptorSets[i], 1 /*binding*/, 0,
          vk::DescriptorType::eStorageBuffer, nullptr, storageBufferInfos[i]);

      writeDescriptorSets.emplace_back(
          *compute.descriptorSets[i], 2 /*binding*/, 0,
          vk::DescriptorType::eUniformBuffer, nullptr, uniformBufferInfos[i]);
    }
    device.updateDescriptorSets(writeDescriptorSets, nullptr);

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

  // create pipelines
  {
    auto compCalculateCode = vgeu::readFile(
        getShadersPath() + "/particle/particle_calculate.comp.spv");
    vk::raii::ShaderModule compCacluateShaderModule =
        vgeu::createShaderModule(device, compCalculateCode);

    uint32_t maxComputeSharedMemorySize =
        physicalDevice.getProperties().limits.maxComputeSharedMemorySize;
    std::cout << "maxComputeSharedMemorySize: " << maxComputeSharedMemorySize
              << std::endl;
    SpecializationData specializationData{};
    specializationData.sharedDataSize = std::min(
        1024u,
        static_cast<uint32_t>(maxComputeSharedMemorySize / sizeof(glm::vec4)));
    specializationData.gravity = gravity;
    specializationData.power = power;
    specializationData.soften = soften;
    // TODO: for 1~4
    specializationData.rkStep = integrateStep;

    std::vector<vk::SpecializationMapEntry> specializationMapEntries;
    specializationMapEntries.emplace_back(
        0u, offsetof(SpecializationData, sharedDataSize), sizeof(uint32_t));
    specializationMapEntries.emplace_back(
        1u, offsetof(SpecializationData, gravity), sizeof(float));
    specializationMapEntries.emplace_back(
        2u, offsetof(SpecializationData, power), sizeof(float));
    specializationMapEntries.emplace_back(
        3u, offsetof(SpecializationData, soften), sizeof(float));
    specializationMapEntries.emplace_back(
        4u, offsetof(SpecializationData, rkStep), sizeof(uint32_t));

    for (uint32_t i = 1; i <= integrateStep; i++) {
      specializationData.rkStep = i;
      // NOTE: template argument deduction not work with implicit conversion
      vk::SpecializationInfo specializationInfo(
          specializationMapEntries,
          vk::ArrayProxyNoTemporaries<const SpecializationData>(
              specializationData));
      vk::PipelineShaderStageCreateInfo computeShaderStageCI(
          vk::PipelineShaderStageCreateFlags{},
          vk::ShaderStageFlagBits::eCompute, *compCacluateShaderModule, "main",
          &specializationInfo);
      vk::ComputePipelineCreateInfo computePipelineCI(vk::PipelineCreateFlags{},
                                                      computeShaderStageCI,
                                                      *compute.pipelineLayout);
      // 1st pass
      compute.pipelineCalculate.emplace_back(device, pipelineCache,
                                             computePipelineCI);
    }
    // 2nd pass
    auto compIntegrateCode = vgeu::readFile(
        getShadersPath() + "/particle/particle_integrate.comp.spv");
    vk::raii::ShaderModule compIntegrateShaderModule =
        vgeu::createShaderModule(device, compIntegrateCode);
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
    compute.pipelineIntegrate =
        vk::raii::Pipeline(device, pipelineCache, computePipelineCI);
  }
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
  }

  // create semaphore for compute-graphics sync
  {
    compute.semaphores.reserve(MAX_CONCURRENT_FRAMES);
    for (size_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      compute.semaphores.emplace_back(device, vk::SemaphoreCreateInfo());
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
  std::vector<glm::vec3> attractorsData = {
      glm::vec3(5.0f, 0.0f, 0.0f), glm::vec3(-5.0f, 0.0f, 0.0f),
      glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(0.0f, -5.0f, 0.0f),
      glm::vec3(0.0f, 0.0f, 5.0f), glm::vec3(0.0f, 0.0f, -5.0f),
  };
  std::vector<glm::vec3> attractors;
  for (size_t i = 0; i < std::min(numAttractors,
                                  static_cast<uint32_t>(attractorsData.size()));
       i++) {
    attractors.push_back(attractorsData[i]);
  }

  std::vector<Particle> particles;
  std::default_random_engine rndEngine;
  rndEngine.seed(1111);
  std::normal_distribution<float> normalDist(0.0f, 1.0f);

  std::vector<float> colors{
      //::packColor(2, 20, 200),
      ::packColor(5, 12, 129),  ::packColor(202, 42, 1),
      ::packColor(41, 86, 143), ::packColor(161, 40, 48),
      ::packColor(1, 75, 255),  ::packColor(246, 7, 9),
  };
  for (size_t i = 0; i < attractors.size(); i++) {
    uint32_t numParticlesPerAttractor = numParticles / attractors.size();
    if (i == attractors.size() - 1) {
      numParticlesPerAttractor =
          numParticles - numParticlesPerAttractor * (attractors.size() - 1);
    }
    for (size_t j = 0; j < numParticlesPerAttractor; j++) {
      Particle& particle = particles.emplace_back();
      glm::vec3 position;
      glm::vec3 velocity;
      float mass;
      float colorOffset = colors[i % colors.size()];

      if (j == 0) {
        position = attractors[i] * 1.5f;
        velocity = glm::vec3{0.f};
        mass = 90000.0f;
      } else {
        position = attractors[i] + glm::vec3{
                                       normalDist(rndEngine),
                                       normalDist(rndEngine),
                                       normalDist(rndEngine),
                                   } * 0.75f;
        float len = glm::length(glm::normalize(position - attractors[i]));
        position.y *= 2.0f - (len * len);

        // velocity = (glm::vec3{
        //                 normalDist(rndEngine),
        //                 normalDist(rndEngine),
        //                 normalDist(rndEngine),
        //             } * 0.2f -
        //             glm::normalize(position)) *
        //            10.0f;
        glm::vec3 angular =
            glm::vec3(0.5f, 1.5f, 0.5f) * (((i % 2) == 0) ? 1.0f : -1.0f);
        velocity = glm::cross((position - attractors[i]), angular) +
                   glm::vec3(normalDist(rndEngine), normalDist(rndEngine),
                             normalDist(rndEngine) * 0.025f);

        mass = (normalDist(rndEngine) * 0.5f + 0.5f) * 75.f;
      }
      glm::vec3 rot =
          glm::cross(glm::vec3{0.f, -1.0f, 0.f}, glm::normalize(attractors[i]));
      velocity += rot * rotationVelocity;
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
              vk::BufferUsageFlagBits::eTransferDst |
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
            // TODO: pipeline barrier to the compute queue?
            // TODO: check spec and exs for ownership transfer
            // release
            if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
              vk::BufferMemoryBarrier bufferBarrier(
                  vk::AccessFlagBits::eTransferWrite, vk::AccessFlags{},
                  graphics.queueFamilyIndex, compute.queueFamilyIndex,
                  compute.storageBuffers[i]->getBuffer(), 0ull,
                  compute.storageBuffers[i]->getBufferSize());
              cmdBuffer.pipelineBarrier(
                  vk::PipelineStageFlagBits::eTransfer,
                  vk::PipelineStageFlagBits::eBottomOfPipe,
                  vk::DependencyFlags{}, nullptr, bufferBarrier, nullptr);
            }
          }
        });
  }

  // tail
  {
    tails.resize(numParticles);
    // for (size_t i = 0; i < tails.size(); i++) {
    //   for (size_t j = 0; j < tailSize; j++) {
    //     glm::vec4 pos{1.f, 0.f, 2.f, 0.f};
    //     pos.z +=
    //         3.0 * (static_cast<float>(j + 1) / static_cast<float>(tailSize));
    //     pos.w = ::packColor(246, 7, 9);
    //     tails[i].push_back(pos);
    //   }
    // }
    tailsData.resize(tails.size() * tailSize);
    tailBuffers.resize(MAX_CONCURRENT_FRAMES);
    for (size_t i = 0; i < tailBuffers.size(); i++) {
      tailBuffers[i] = std::make_unique<vgeu::VgeuBuffer>(
          globalAllocator->getAllocator(), sizeof(TailElt), tailsData.size(),
          vk::BufferUsageFlagBits::eVertexBuffer |
              vk::BufferUsageFlagBits::eStorageBuffer,
          VMA_MEMORY_USAGE_AUTO,
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
              VMA_ALLOCATION_CREATE_MAPPED_BIT);
    }

    // index buffer
    std::vector<uint32_t> indices;
    indices.reserve(tails.size() * (tailSize + 1));
    for (size_t i = 0; i < tails.size(); i++) {
      for (size_t j = 0; j < tailSize; j++) {
        indices.push_back(i * tailSize + j);
      }
      // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPipelineInputAssemblyStateCreateInfo.html
      // 0xffffffff
      indices.push_back(static_cast<uint32_t>(-1));
    }

    vgeu::VgeuBuffer indexStagingBuffer(
        globalAllocator->getAllocator(), sizeof(uint32_t), indices.size(),
        vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    std::memcpy(indexStagingBuffer.getMappedData(), indices.data(),
                indexStagingBuffer.getBufferSize());

    tailIndexBuffer = std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(uint32_t), indices.size(),
        vk::BufferUsageFlagBits::eIndexBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
        VMA_MEMORY_USAGE_AUTO,
        /*VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT*/
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    // single Time command copy both buffers
    vgeu::oneTimeSubmit(
        device, commandPool, queue,
        [&](const vk::raii::CommandBuffer& cmdBuffer) {
          cmdBuffer.copyBuffer(
              indexStagingBuffer.getBuffer(), tailIndexBuffer->getBuffer(),
              vk::BufferCopy(0, 0, indexStagingBuffer.getBufferSize()));
        });
  }
  uint32_t* a = static_cast<uint32_t*>(tailIndexBuffer->getMappedData());
}

void VgeExample::createVertexSCI() {
  // TODO: vertex binding and attribute descriptions
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
  // tailVertexInfos.attributeDescriptions.emplace_back(
  //     1 /*location*/, 0 /* binding */, vk::Format::eR32G32B32A32Sfloat,
  //     offsetof(TailElt, vel));

  tailVertexInfos.vertexInputSCI = vk::PipelineVertexInputStateCreateInfo(
      vk::PipelineVertexInputStateCreateFlags{},
      tailVertexInfos.bindingDescriptions,
      tailVertexInfos.attributeDescriptions);
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
                         MAX_CONCURRENT_FRAMES * 2);
  // NOTE: need to check flag
  vk::DescriptorPoolCreateInfo descriptorPoolCI(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      MAX_CONCURRENT_FRAMES * 3 /*set globalUBO, dynamicUBO, computeUbo*/,
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

  // calculate tail
  updateTailSSBO();

  prepareFrame();

  // update uniform buffers;
  updateDynamicUbo();
  updateComputeUbo();
  updateGraphicsUbo();

  // TODO: Fence and compute recording order

  //  compute recording and submitting
  {
    buildComputeCommandBuffers();
    vk::PipelineStageFlags computeWaitDstStageMask(
        vk::PipelineStageFlagBits::eComputeShader);
    vk::SubmitInfo computeSubmitInfo(*graphics.semaphores[currentFrameIndex],
                                     computeWaitDstStageMask,
                                     *compute.cmdBuffers[currentFrameIndex],
                                     *compute.semaphores[currentFrameIndex]);
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
        *compute.semaphores[currentFrameIndex],
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

    // TODO: fence?
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
    vk::BufferMemoryBarrier bufferBarrier(
        vk::AccessFlags{}, vk::AccessFlagBits::eVertexAttributeRead,
        compute.queueFamilyIndex, graphics.queueFamilyIndex,
        compute.storageBuffers[currentFrameIndex]->getBuffer(), 0ull,
        compute.storageBuffers[currentFrameIndex]->getBufferSize());
    drawCmdBuffers[currentFrameIndex].pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eVertexInput, vk::DependencyFlags{}, nullptr,
        bufferBarrier, nullptr);
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
    drawCmdBuffers[currentFrameIndex].bindVertexBuffers(
        0, compute.storageBuffers[currentFrameIndex]->getBuffer(), offset);
    drawCmdBuffers[currentFrameIndex].draw(numParticles, 1, 0, 0);
  }

  // tail
  {
    drawCmdBuffers[currentFrameIndex].setLineWidth(1.f);
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
    vk::BufferMemoryBarrier bufferBarrier(
        vk::AccessFlagBits::eVertexAttributeRead, vk::AccessFlags{},
        graphics.queueFamilyIndex, compute.queueFamilyIndex,
        compute.storageBuffers[currentFrameIndex]->getBuffer(), 0ull,
        compute.storageBuffers[currentFrameIndex]->getBufferSize());
    drawCmdBuffers[currentFrameIndex].pipelineBarrier(
        vk::PipelineStageFlagBits::eVertexInput,
        vk::PipelineStageFlagBits::eBottomOfPipe, vk::DependencyFlags{},
        nullptr, bufferBarrier, nullptr);
  }

  // end command buffer
  drawCmdBuffers[currentFrameIndex].end();
}

void VgeExample::buildComputeCommandBuffers() {
  compute.cmdBuffers[currentFrameIndex].begin({});

  // acquire barrier graphics -> compute
  if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
    vk::BufferMemoryBarrier bufferBarrier(
        vk::AccessFlags{}, vk::AccessFlagBits::eShaderWrite,
        graphics.queueFamilyIndex, compute.queueFamilyIndex,
        compute.storageBuffers[currentFrameIndex]->getBuffer(), 0ull,
        compute.storageBuffers[currentFrameIndex]->getBufferSize());
    // NOTE: top of pipeline -> same as all commands,
    compute.cmdBuffers[currentFrameIndex].pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags{},
        nullptr, bufferBarrier, nullptr);
  }

  for (size_t i = 0; i < integrateStep; i++) {
    // 1st pass
    compute.cmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eCompute, *compute.pipelineCalculate[i]);
    compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 0,
        *compute.descriptorSets[currentFrameIndex], nullptr);
    // NOTE: number of local work group should cover all vertices
    // TODO: +1 makes program stop.
    compute.cmdBuffers[currentFrameIndex].dispatch(numParticles / 256 + 1, 1,
                                                   1);

    // memory barrier
    vk::BufferMemoryBarrier bufferBarrier(
        vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        compute.storageBuffers[currentFrameIndex]->getBuffer(), 0ull,
        compute.storageBuffers[currentFrameIndex]->getBufferSize());
    compute.cmdBuffers[currentFrameIndex].pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags{},
        nullptr, bufferBarrier, nullptr);
  }
  // 2nd pass
  compute.cmdBuffers[currentFrameIndex].bindPipeline(
      vk::PipelineBindPoint::eCompute, *compute.pipelineIntegrate);
  compute.cmdBuffers[currentFrameIndex].dispatch(numParticles / 256 + 1, 1, 1);

  // release barrier
  if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
    vk::BufferMemoryBarrier bufferBarrier(
        vk::AccessFlagBits::eShaderWrite, vk::AccessFlags{},
        compute.queueFamilyIndex, graphics.queueFamilyIndex,
        compute.storageBuffers[currentFrameIndex]->getBuffer(), 0ull,
        compute.storageBuffers[currentFrameIndex]->getBufferSize());
    compute.cmdBuffers[currentFrameIndex].pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eBottomOfPipe, vk::DependencyFlags{},
        nullptr, bufferBarrier, nullptr);
  }
  compute.cmdBuffers[currentFrameIndex].end();
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
  std::memcpy(graphics.globalUniformBuffers[currentFrameIndex]->getMappedData(),
              &graphics.globalUbo, sizeof(GlobalUbo));
}

void VgeExample::updateComputeUbo() {
  compute.ubo.dt = paused ? 0.0f : frameTimer * 0.05f;
  std::memcpy(compute.uniformBuffers[currentFrameIndex]->getMappedData(),
              &compute.ubo, sizeof(compute.ubo));
}

void VgeExample::updateDynamicUbo() {
  float animationTimer = animationTime - animationLastTime;
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

void VgeExample::setupCommandLineParser(CLI::App& app) {
  VgeBase::setupCommandLineParser(app);
  app.add_option("--intStep, --is", integrateStep, "Integrate Step 1, 2, 4")
      ->capture_default_str();
  app.add_option("--numParticles, --np", numParticles, "number of particles")
      ->capture_default_str();
  app.add_option("--numAttractors, --na", numAttractors, "number of attractors")
      ->capture_default_str();
  app.add_option("--rotationVelocity, --rv", rotationVelocity,
                 "initial y-axis rotation velocity )")
      ->capture_default_str();
  app.add_option("--gravity, -g", gravity, "gravity constants")
      ->capture_default_str();
  app.add_option("--power, -p", power, "power constants")
      ->capture_default_str();
  app.add_option("--soften, -s", soften, "soften constants")
      ->capture_default_str();
  app.add_option("--tailSize, --ts", tailSize, "tail size")
      ->capture_default_str();
  app.add_option("--tailSampleTime, --tst", tailSampleTime, "tail sample time")
      ->capture_default_str();
}
void VgeExample::updateTailSSBO() {
  // update
  {
    const Particle* particles = static_cast<Particle*>(
        compute.storageBuffers[currentFrameIndex]->getMappedData());
    if (tailTimer > tailSampleTime || tailTimer < 0.f) {
      for (size_t i = 0; i < tails.size(); i++) {
        bool isInit = false;
        // need to check not empty before pop.
        if (!tails[i].empty())
          tails[i].pop_front();
        else
          isInit = true;
        // back is head side, front is tail side.
        while (tails[i].size() < tailSize) {
          glm::vec4 packedTailElt = particles[i].pos;
          packedTailElt.w = isInit ? 0.f : particles[i].vel.w;
          tails[i].push_back(packedTailElt);
        }
      }
      tailTimer = 0.f;
    }
    if (!paused) {
      tailTimer += frameTimer;
    }
  }
  // copy
  {
    for (size_t i = 0; i < tails.size(); i++) {
      size_t j = 0;
      for (auto it = tails[i].begin(); it != tails[i].end(); it++) {
        glm::vec4 packedTailElt = *it;
        tailsData[i * tailSize + j].pos.x = packedTailElt.x;
        tailsData[i * tailSize + j].pos.y = packedTailElt.y;
        tailsData[i * tailSize + j].pos.z = packedTailElt.z;
        tailsData[i * tailSize + j].pos.w = packedTailElt.w;
        // color
        // glm::vec3 color = ::unpackColor(packedTailElt.w);
        // glm::vec3 fadedColor =
        //     color * (static_cast<float>((j + 1) * (j + 1)) /
        //              static_cast<float>(tails[i].size() * tails[i].size()));
        // tailsData[i * tailSize + j].pos.w =
        //     ::packColor(static_cast<uint8_t>(fadedColor.r),
        //                 static_cast<uint8_t>(fadedColor.g),
        //                 static_cast<uint8_t>(fadedColor.b));

        j++;
      }
    }
    std::memcpy(tailBuffers[currentFrameIndex]->getMappedData(),
                tailsData.data(), sizeof(TailElt) * tailsData.size());
  }
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()