#include "paint_splatter.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

// std
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace vge {

VgeExample::VgeExample() : VgeBase() { title = "Paint Splatter Example"; }
VgeExample::~VgeExample() {}

void VgeExample::setupCommandLineParser(CLI::App& app) {
  VgeBase::setupCommandLineParser(app);
}

void VgeExample::getEnabledFeatures() {
  // shaderClipDistance for gl_PointSize in vertex shader
  enabledFeatures.shaderClipDistance =
      physicalDevice.getFeatures().shaderClipDistance;
  // largePoints may be needed for point sizes > 1
  enabledFeatures.largePoints = physicalDevice.getFeatures().largePoints;
}

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

  // Set queue family indices (mirrors particle.cpp:92-93)
  graphics.queueFamilyIndex = queueFamilyIndices.graphics;
  compute.queueFamilyIndex = queueFamilyIndices.compute;

  createVertexBuffer();
  createIndexBuffer();
  createParticleBuffers();
  createUniformBuffers();
  createDescriptorSetLayout();
  createDescriptorPool();
  createDescriptorSets();
  createPipelines();
  createParticlePipeline();
  prepareCompute();

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

// ---------------------------------------------------------------------------
// Particle SSBO: device-local, one per frame in flight
// Seed: 16x16x16 lattice above the canvas floor (y < 0)
// ---------------------------------------------------------------------------
void VgeExample::createParticleBuffers() {
  // Build CPU seed data
  const int N = 16;
  const float spacing = 0.05f;
  // Seed the block high above the floor (~0.7 of domain height) so it falls
  // onto the canvas at y=0. Domain spans y in [-kDomainHeight, 0].
  const float centerY = -kDomainHeight * 0.7f;  // world -Y = up

  std::vector<Particle> cpuParticles;
  cpuParticles.reserve(static_cast<size_t>(N * N * N));

  for (int ix = 0; ix < N; ix++) {
    for (int iy = 0; iy < N; iy++) {
      for (int iz = 0; iz < N; iz++) {
        Particle p{};
        p.pos =
            glm::vec4((ix - N / 2) * spacing, centerY + (iy - N / 2) * spacing,
                      (iz - N / 2) * spacing, 1.f);
        p.vel = glm::vec4(0.f);
        p.predict = glm::vec4(0.f);
        // Color varies by position for visual distinction
        p.color =
            glm::vec4(static_cast<float>(ix) / N, static_cast<float>(iy) / N,
                      static_cast<float>(iz) / N, 1.f);
        cpuParticles.push_back(p);
      }
    }
  }

  numParticles = static_cast<uint32_t>(cpuParticles.size());
  assert(numParticles <= kMaxParticles);

  // Readback: print first particle + count
  std::cout << "[paint_splatter] numParticles=" << numParticles
            << "  particle[0].pos=(" << cpuParticles[0].pos.x << ","
            << cpuParticles[0].pos.y << "," << cpuParticles[0].pos.z << ")\n";

  // Staging buffer
  vgeu::VgeuBuffer stagingBuffer(
      globalAllocator->getAllocator(), sizeof(Particle), kMaxParticles,
      vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);
  std::memcpy(stagingBuffer.getMappedData(), cpuParticles.data(),
              sizeof(Particle) * numParticles);

  // Per-frame device-local SSBOs
  particleBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    particleBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(Particle), kMaxParticles,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eVertexBuffer |
            vk::BufferUsageFlagBits::eTransferDst |
            vk::BufferUsageFlagBits::eTransferSrc,  // debug readback copy
                                                    // source
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0));

    // Upload seed to each frame's SSBO
    vgeu::oneTimeSubmit(
        device, commandPool, queue,
        [&](const vk::raii::CommandBuffer& cmdBuffer) {
          cmdBuffer.copyBuffer(
              stagingBuffer.getBuffer(), particleBuffers[i]->getBuffer(),
              vk::BufferCopy(0, 0, sizeof(Particle) * numParticles));
        });
  }

  // Per-frame host-visible readback targets (debug y min/max).
  readbackBuffers.reserve(MAX_CONCURRENT_FRAMES);
  readbackPending.assign(MAX_CONCURRENT_FRAMES, 0);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    readbackBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(Particle), kMaxParticles,
        vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT));
  }
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
  // Canvas UBO: MAX_CONCURRENT_FRAMES
  // Compute UBO: MAX_CONCURRENT_FRAMES
  // Compute SSBO: MAX_CONCURRENT_FRAMES
  std::vector<vk::DescriptorPoolSize> poolSizes;
  poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer,
                         MAX_CONCURRENT_FRAMES * 2u);
  poolSizes.emplace_back(vk::DescriptorType::eStorageBuffer,
                         MAX_CONCURRENT_FRAMES);
  vk::DescriptorPoolCreateInfo descriptorPoolCI(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      MAX_CONCURRENT_FRAMES * 2u, poolSizes);
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
// Particle debug renderer: point list, same set=0 GlobalUbo layout
// Stride = sizeof(Particle) = 64; attr 0 = pos, attr 1 = vel (skip), attr 2 =
// predict (skip), attr 3 = color (location 1 in shader maps to inColor).
// ---------------------------------------------------------------------------
void VgeExample::createParticlePipeline() {
  auto vertCode =
      vgeu::readFile(getShadersPath() + "/paint_splatter/particle.vert.spv");
  auto fragCode =
      vgeu::readFile(getShadersPath() + "/paint_splatter/particle.frag.spv");

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

  // Binding 0: Particle SSBO used as vertex buffer, stride = sizeof(Particle)
  vk::VertexInputBindingDescription bindingDesc(0, sizeof(Particle));

  // location=0  inPos   = Particle::pos   at offset 0
  // location=1  inVel   = Particle::vel   at offset 16  (needed in vert for
  //                                                       attribute layout)
  // location=2  inPred  = Particle::predict at offset 32
  // location=3  inColor = Particle::color at offset 48
  // The shader only uses location=0 (inPos) and location=3 (inColor).
  // We declare all 4 so the binding stride is correct.
  std::vector<vk::VertexInputAttributeDescription> attrDescs;
  attrDescs.emplace_back(0, 0, vk::Format::eR32G32B32A32Sfloat,
                         static_cast<uint32_t>(offsetof(Particle, pos)));
  attrDescs.emplace_back(1, 0, vk::Format::eR32G32B32A32Sfloat,
                         static_cast<uint32_t>(offsetof(Particle, vel)));
  attrDescs.emplace_back(2, 0, vk::Format::eR32G32B32A32Sfloat,
                         static_cast<uint32_t>(offsetof(Particle, predict)));
  attrDescs.emplace_back(3, 0, vk::Format::eR32G32B32A32Sfloat,
                         static_cast<uint32_t>(offsetof(Particle, color)));

  vk::PipelineVertexInputStateCreateInfo vertexInputSCI(
      vk::PipelineVertexInputStateCreateFlags(), bindingDesc, attrDescs);

  // Point list topology
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

  // Reuse the same pipelineLayout (set=0 = GlobalUbo)
  vk::GraphicsPipelineCreateInfo graphicsPipelineCI(
      vk::PipelineCreateFlags(), shaderStageCIs, &vertexInputSCI,
      &inputAssemblySCI, nullptr, &viewportSCI, &rasterizationSCI,
      &multisampleSCI, &depthStencilSCI, &colorBlendSCI, &dynamicSCI,
      *pipelineLayout, *renderPass);

  particlePipeline =
      vk::raii::Pipeline(device, pipelineCache, graphicsPipelineCI);
}

// ---------------------------------------------------------------------------
// Compute setup: mirrors particle.cpp prepareGraphics / prepareCompute
// ---------------------------------------------------------------------------
void VgeExample::prepareCompute() {
  // --- 1. Create per-frame compute uniform buffers ---
  computeFirstUse.assign(MAX_CONCURRENT_FRAMES, 1);

  compute.ubo.dt = kFixedDt;
  compute.ubo.particleCount = numParticles;
  compute.ubo.gravity = gravity;
  compute.ubo._pad0 = 0.f;
  {
    const float half = kCanvasWorld * 0.5f;
    compute.ubo.canvasMin = glm::vec4(-half, -kDomainHeight, -half, 0.05f);
    compute.ubo.canvasMax = glm::vec4(half, 0.f, half, 0.f);
  }

  compute.uniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    compute.uniformBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(ComputeUbo), 1,
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT));
    std::memcpy(compute.uniformBuffers[i]->getMappedData(), &compute.ubo,
                sizeof(ComputeUbo));
  }

  // --- 2. Get compute queue (mirrors particle.cpp:161) ---
  compute.queue = vk::raii::Queue(device, compute.queueFamilyIndex, 0);

  // --- 3. Compute descriptor set layout: binding 0 = SSBO, binding 1 = UBO ---
  createComputeDescriptorSetLayout();

  // --- 4. Compute pipeline ---
  createComputePipeline();

  // --- 5. Compute descriptor sets ---
  createComputeDescriptorSets();

  // --- 6. Compute command pool + buffers ---
  vk::CommandPoolCreateInfo cmdPoolCI(
      vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      compute.queueFamilyIndex);
  compute.cmdPool = vk::raii::CommandPool(device, cmdPoolCI);

  vk::CommandBufferAllocateInfo cmdBufAllocInfo(
      *compute.cmdPool, vk::CommandBufferLevel::ePrimary,
      MAX_CONCURRENT_FRAMES);
  compute.cmdBuffers = vk::raii::CommandBuffers(device, cmdBufAllocInfo);

  // --- 7. Create semaphores ---
  // Compute semaphores: signalled by compute, waited by graphics
  compute.semaphores.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    compute.semaphores.emplace_back(device, vk::SemaphoreCreateInfo());
  }

  // Graphics semaphores: signalled by graphics, waited by compute
  // Mirrors particle.cpp prepareGraphics() lines 106-120
  {
    std::vector<vk::Semaphore> semaphoresToSignal;
    semaphoresToSignal.reserve(MAX_CONCURRENT_FRAMES);
    graphics.semaphores.reserve(MAX_CONCURRENT_FRAMES);
    for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      vk::raii::Semaphore& semaphore =
          graphics.semaphores.emplace_back(device, vk::SemaphoreCreateInfo());
      semaphoresToSignal.push_back(*semaphore);
    }
    // Initial signal so frame-0 compute submit does NOT deadlock waiting for
    // graphics (which has never run yet). Mirrors particle.cpp:117-119.
    vk::SubmitInfo submitInfo({}, {}, {}, semaphoresToSignal);
    queue.submit(submitInfo);
    queue.waitIdle();
  }
}

void VgeExample::createComputeDescriptorSetLayout() {
  std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
  // binding 0: particle SSBO
  layoutBindings.emplace_back(0, vk::DescriptorType::eStorageBuffer, 1,
                              vk::ShaderStageFlagBits::eCompute);
  // binding 1: ComputeUbo
  layoutBindings.emplace_back(1, vk::DescriptorType::eUniformBuffer, 1,
                              vk::ShaderStageFlagBits::eCompute);

  vk::DescriptorSetLayoutCreateInfo layoutCI({}, layoutBindings);
  compute.descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutCI);

  vk::PipelineLayoutCreateInfo pipelineLayoutCI({},
                                                *compute.descriptorSetLayout);
  compute.pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutCI);
}

void VgeExample::createComputePipeline() {
  auto compCode =
      vgeu::readFile(getShadersPath() + "/paint_splatter/pbf_predict.comp.spv");
  vk::raii::ShaderModule compShaderModule =
      vgeu::createShaderModule(device, compCode);

  vk::PipelineShaderStageCreateInfo shaderStageCI(
      vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute,
      *compShaderModule, "main", nullptr);

  vk::ComputePipelineCreateInfo pipelineCI(
      vk::PipelineCreateFlags(), shaderStageCI, *compute.pipelineLayout);
  compute.pipeline = vk::raii::Pipeline(device, pipelineCache, pipelineCI);
}

void VgeExample::createComputeDescriptorSets() {
  vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                          *compute.descriptorSetLayout);
  compute.descriptorSets.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    compute.descriptorSets.push_back(
        std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
  }

  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    vk::DescriptorBufferInfo ssboInfo(particleBuffers[i]->getBuffer(), 0,
                                      particleBuffers[i]->getBufferSize());
    vk::DescriptorBufferInfo uboInfo =
        compute.uniformBuffers[i]->descriptorInfo();

    std::array<vk::WriteDescriptorSet, 2> writes{
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 0, 0,
                               vk::DescriptorType::eStorageBuffer, nullptr,
                               ssboInfo),
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 1, 0,
                               vk::DescriptorType::eUniformBuffer, nullptr,
                               uboInfo),
    };
    device.updateDescriptorSets(writes, nullptr);
  }
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

void VgeExample::updateComputeUbo() {
  // dt: frameTimer by default, optional fixed dt for stable stepping.
  compute.ubo.dt = useFixedDt ? kFixedDt : frameTimer;
  compute.ubo.particleCount = numParticles;
  // gravity = +Y pulls particles down on screen (toward the floor at y=0).
  compute.ubo.gravity = gravity;
  compute.ubo._pad0 = 0.f;
  // Domain box: floor at cmax.y=0, extends upward (screen) to cmin.y=-height.
  const float half = kCanvasWorld * 0.5f;
  const float h = 0.05f;  // cell size / particle spacing (used in M4)
  compute.ubo.canvasMin = glm::vec4(-half, -kDomainHeight, -half, h);
  compute.ubo.canvasMax = glm::vec4(half, 0.f, half, 0.f);
  std::memcpy(compute.uniformBuffers[currentFrameIndex]->getMappedData(),
              &compute.ubo, sizeof(ComputeUbo));
}

// Debug READBACK (M3): record a copy of the particle SSBO into a host-visible
// buffer INSIDE the graphics command buffer (the buffer is owned by graphics
// there, after the compute->graphics acquire), so the queue-ownership ping-pong
// is left intact. The copy is read one frame later in consumeParticleReadback.
void VgeExample::recordParticleReadbackCopy() {
  if (!readbackRequest || numParticles == 0) return;
  const vk::raii::CommandBuffer& cmd = drawCmdBuffers[currentFrameIndex];
  // Make the compute-produced data visible to a transfer read (same queue).
  vk::BufferMemoryBarrier toTransfer(
      vk::AccessFlagBits::eVertexAttributeRead,
      vk::AccessFlagBits::eTransferRead, graphics.queueFamilyIndex,
      graphics.queueFamilyIndex,
      particleBuffers[currentFrameIndex]->getBuffer(), 0ull,
      sizeof(Particle) * numParticles);
  cmd.pipelineBarrier(vk::PipelineStageFlagBits::eVertexInput,
                      vk::PipelineStageFlagBits::eTransfer,
                      vk::DependencyFlags{}, nullptr, toTransfer, nullptr);
  cmd.copyBuffer(particleBuffers[currentFrameIndex]->getBuffer(),
                 readbackBuffers[currentFrameIndex]->getBuffer(),
                 vk::BufferCopy(0, 0, sizeof(Particle) * numParticles));
  readbackPending[currentFrameIndex] = 1;
  readbackRequest = false;
}

// Read the previously-recorded copy (its fence has been waited on) and print
// the particle y range to verify collision clamping numerically.
void VgeExample::consumeParticleReadback() {
  if (!readbackPending[currentFrameIndex]) return;
  readbackPending[currentFrameIndex] = 0;
  const Particle* data = static_cast<const Particle*>(
      readbackBuffers[currentFrameIndex]->getMappedData());
  float minY = data[0].pos.y, maxY = data[0].pos.y;
  for (uint32_t i = 1; i < numParticles; i++) {
    minY = std::min(minY, data[i].pos.y);
    maxY = std::max(maxY, data[i].pos.y);
  }
  std::cout << "[paint_splatter] particle y range: min=" << minY
            << " max=" << maxY << " (floor at y=0)" << std::endl;
}

// ---------------------------------------------------------------------------
// render / draw
// ---------------------------------------------------------------------------
void VgeExample::render() {
  if (!prepared) return;
  updateGlobalUbo();
  updateComputeUbo();

  // Request a particle readback ~once per second (recorded in this frame's
  // graphics command buffer, read back one frame later).
  readbackTimer += frameTimer;
  if (readbackTimer >= 1.0f) {
    readbackTimer = 0.f;
    readbackRequest = true;
  }

  draw();
}

// ---------------------------------------------------------------------------
// draw: two-submit handshake mirroring particle.cpp:1298-1395
// ---------------------------------------------------------------------------
void VgeExample::draw() {
  vk::Result result =
      device.waitForFences(*waitFences[currentFrameIndex], VK_TRUE,
                           std::numeric_limits<uint64_t>::max());
  assert(result != vk::Result::eTimeout && "Timed out: waitFence");

  device.resetFences(*waitFences[currentFrameIndex]);

  // This frame slot's previous submission has completed; if it recorded a
  // debug readback copy, its host-visible buffer is now valid to read.
  consumeParticleReadback();

  prepareFrame();

  // --- Compute submit: wait graphics semaphore, signal compute semaphore ---
  // Mirrors particle.cpp:1356-1366
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

  // --- Graphics submit: wait {compute semaphore, present semaphore},
  //     signal {graphics semaphore, render-complete semaphore} ---
  // Mirrors particle.cpp:1368-1393
  {
    buildCommandBuffers();

    std::vector<vk::PipelineStageFlags> graphicsWaitDstStageMasks{
        vk::PipelineStageFlagBits::eVertexInput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
    };

    std::vector<vk::Semaphore> graphicsWaitSemaphores{
        *compute.semaphores[currentFrameIndex],
        *presentCompleteSemaphores[currentFrameIndex],
    };

    std::vector<vk::Semaphore> graphicsSignalSemaphores{
        *graphics.semaphores[currentFrameIndex],
        *renderCompleteSemaphores[currentFrameIndex],
    };

    vk::SubmitInfo graphicsSubmitInfo(
        graphicsWaitSemaphores, graphicsWaitDstStageMasks,
        *drawCmdBuffers[currentFrameIndex], graphicsSignalSemaphores);

    queue.submit(graphicsSubmitInfo, *waitFences[currentFrameIndex]);
  }

  submitFrame();
}

// ---------------------------------------------------------------------------
// buildComputeCommandBuffers: acquire barrier, dispatch, release barrier
// Mirrors particle.cpp:1510-1641
// ---------------------------------------------------------------------------
void VgeExample::buildComputeCommandBuffers() {
  compute.cmdBuffers[currentFrameIndex].begin({});

  // Acquire barrier graphics -> compute (if different queue families).
  // Skip on the very first dispatch per buffer: no graphics release precedes
  // it, so acquiring would be an unmatched ownership transfer (validation).
  // Mirrors particle.cpp:1513-1532
  if (graphics.queueFamilyIndex != compute.queueFamilyIndex &&
      !computeFirstUse[currentFrameIndex]) {
    vk::BufferMemoryBarrier bufBarrier(
        vk::AccessFlags{}, vk::AccessFlagBits::eShaderWrite,
        graphics.queueFamilyIndex, compute.queueFamilyIndex,
        particleBuffers[currentFrameIndex]->getBuffer(), 0ull,
        particleBuffers[currentFrameIndex]->getBufferSize());

    compute.cmdBuffers[currentFrameIndex].pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags{},
        nullptr, bufBarrier, nullptr);
  }
  computeFirstUse[currentFrameIndex] = 0;

  // Bind predict pipeline + descriptor set for this frame
  compute.cmdBuffers[currentFrameIndex].bindPipeline(
      vk::PipelineBindPoint::eCompute, *compute.pipeline);
  compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
      vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 0,
      *compute.descriptorSets[currentFrameIndex], nullptr);

  // Dispatch: ceil(numParticles / 256)
  uint32_t groupCount = (numParticles + 255u) / 256u;
  compute.cmdBuffers[currentFrameIndex].dispatch(groupCount, 1, 1);

  // Release barrier compute -> graphics (if different queue families)
  // Mirrors particle.cpp:1621-1639
  if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
    vk::BufferMemoryBarrier bufBarrier(
        vk::AccessFlagBits::eShaderWrite, vk::AccessFlags{},
        compute.queueFamilyIndex, graphics.queueFamilyIndex,
        particleBuffers[currentFrameIndex]->getBuffer(), 0ull,
        particleBuffers[currentFrameIndex]->getBufferSize());

    compute.cmdBuffers[currentFrameIndex].pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eBottomOfPipe, vk::DependencyFlags{},
        nullptr, bufBarrier, nullptr);
  }

  compute.cmdBuffers[currentFrameIndex].end();
}

// ---------------------------------------------------------------------------
// buildCommandBuffers: acquire barrier, canvas draw, particle draw, UI
// Mirrors particle.cpp:1397-1508
// ---------------------------------------------------------------------------
void VgeExample::buildCommandBuffers() {
  drawCmdBuffers[currentFrameIndex].begin({});

  // Acquire barrier compute -> graphics (if different queue families)
  // Mirrors particle.cpp:1413-1431
  if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
    vk::BufferMemoryBarrier bufBarrier(
        vk::AccessFlags{}, vk::AccessFlagBits::eVertexAttributeRead,
        compute.queueFamilyIndex, graphics.queueFamilyIndex,
        particleBuffers[currentFrameIndex]->getBuffer(), 0ull,
        particleBuffers[currentFrameIndex]->getBufferSize());

    drawCmdBuffers[currentFrameIndex].pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eVertexInput, vk::DependencyFlags{}, nullptr,
        bufBarrier, nullptr);
  }

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

  // --- Canvas quad ---
  drawCmdBuffers[currentFrameIndex].bindPipeline(
      vk::PipelineBindPoint::eGraphics, *pipeline);
  drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
      vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0,
      {*descriptorSets[currentFrameIndex]}, nullptr);
  drawCmdBuffers[currentFrameIndex].bindVertexBuffers(
      0, {vertexBuffer->getBuffer()}, {0});
  drawCmdBuffers[currentFrameIndex].bindIndexBuffer(indexBuffer->getBuffer(), 0,
                                                    vk::IndexType::eUint32);
  drawCmdBuffers[currentFrameIndex].drawIndexed(indexBuffer->getInstanceCount(),
                                                1, 0, 0, 0);

  // --- Particle debug renderer ---
  if (showParticles && numParticles > 0) {
    drawCmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eGraphics, *particlePipeline);
    // reuse same set=0 descriptor (GlobalUbo)
    drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0,
        {*descriptorSets[currentFrameIndex]}, nullptr);
    vk::DeviceSize offset(0);
    drawCmdBuffers[currentFrameIndex].bindVertexBuffers(
        0, particleBuffers[currentFrameIndex]->getBuffer(), offset);
    drawCmdBuffers[currentFrameIndex].draw(numParticles, 1, 0, 0);
  }

  // ImGui overlay
  drawUI(drawCmdBuffers[currentFrameIndex]);

  drawCmdBuffers[currentFrameIndex].endRenderPass();

  // Debug readback copy (outside the render pass; buffer still owned by
  // graphics). The graphics submit's completion semaphore is waited on by the
  // next compute submit, so the copy finishes before compute overwrites.
  recordParticleReadbackCopy();

  // Release barrier graphics -> compute (if different queue families)
  // Mirrors particle.cpp:1487-1504
  if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
    vk::BufferMemoryBarrier bufBarrier(
        vk::AccessFlagBits::eVertexAttributeRead, vk::AccessFlags{},
        graphics.queueFamilyIndex, compute.queueFamilyIndex,
        particleBuffers[currentFrameIndex]->getBuffer(), 0ull,
        particleBuffers[currentFrameIndex]->getBufferSize());

    drawCmdBuffers[currentFrameIndex].pipelineBarrier(
        vk::PipelineStageFlagBits::eVertexInput,
        vk::PipelineStageFlagBits::eBottomOfPipe, vk::DependencyFlags{},
        nullptr, bufBarrier, nullptr);
  }

  drawCmdBuffers[currentFrameIndex].end();
}

void VgeExample::viewChanged() {
  camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
  // NOTE: UBO update deferred to render() to use correct frame index.
}

// ---------------------------------------------------------------------------
// ImGui panel
// ---------------------------------------------------------------------------
void VgeExample::onUpdateUIOverlay() {
  if (ImGui::CollapsingHeader("Paint Splatter",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    // Read-only info
    ImGui::Text("Frame time : %.3f ms", frameTimer * 1000.0f);

    glm::vec3 camPos = camera.getPosition();
    ImGui::Text("Camera pos : (%.2f, %.2f, %.2f)", camPos.x, camPos.y,
                camPos.z);

    ImGui::Text("Particles  : %u", numParticles);
    ImGui::Checkbox("Show particles", &showParticles);

    // --- sim params (M3) ---
    ImGui::SliderFloat("gravity (+Y)", &gravity, 0.f, 30.f);
    ImGui::Checkbox("fixed dt (1/120)", &useFixedDt);

    if (uiOverlay->button("Save (no-op)")) {
      std::cout << "[paint_splatter] Save button pressed (no-op)\n";
    }
  }
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()
