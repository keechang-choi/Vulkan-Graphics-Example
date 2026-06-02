#include "paint_splatter.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

// std
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace vge {

// Distinct default paint colours so multiple spoids are visually separable.
static glm::vec3 spoidPalette(int i) {
  static const glm::vec3 kColors[] = {
      {0.90f, 0.25f, 0.25f},  // red
      {0.25f, 0.55f, 0.90f},  // blue
      {0.30f, 0.75f, 0.35f},  // green
      {0.95f, 0.80f, 0.20f},  // yellow
      {0.85f, 0.35f, 0.80f},  // magenta
      {0.20f, 0.80f, 0.80f},  // cyan
      {0.95f, 0.55f, 0.20f},  // orange
      {0.60f, 0.40f, 0.85f},  // purple
  };
  constexpr int n = sizeof(kColors) / sizeof(kColors[0]);
  return kColors[((i % n) + n) % n];
}

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

  // One spoid above the canvas centre, driven by the keyboard (design §4).
  {
    Spoid s{};
    s.color = spoidPalette(0);
    spoids.push_back(s);
  }
  spoidController = std::make_unique<KeyboardSpoidController>();

  createVertexBuffer();
  createIndexBuffer();
  createParticleBuffers();
  createMarkerBuffers();
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
  // Per-frame device-local SSBOs (filled by seedParticles()).
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

  // Single pending-burst FIFO (one ping-pong chain).
  pendingEmits.clear();

  // M5: start empty — droplets are spawned by the emit pass, not a seed block.
  // (seedParticles() remains available as an M4 dam-break debug aid.)
  numParticles = 0;
  prevParticleCount = 0;
}

// Fill all per-frame particle SSBOs with the lattice block. Each particle gets
// a random horizontal (xz) initial velocity so the falling block spreads out
// instead of dropping as a rigid column (debug visualization aid).
void VgeExample::seedParticles() {
  const float spacing = kParticleSpacing;
  // Dam-break column: a tall, narrow slab of water resting on the floor (y=0)
  // against the -x wall of the fluid box. On release it collapses in +x and
  // sloshes, forming a dense pool (reaches rho0). Box is x,z in [-kFluidHalf,
  // kFluidHalf], floor at y=0, fluid at y<0.
  const float xlo = -kFluidHalf + spacing, xhi = -kFluidHalf + 0.7f;
  const float zlo = -kFluidHalf + spacing, zhi = kFluidHalf - spacing;
  const float yhi = -spacing;                  // top sits just below the floor
  const float ylo = -kDomainHeight + spacing;  // tall column up to the ceiling
  const int nx = std::max(1, static_cast<int>((xhi - xlo) / spacing));
  const int nz = std::max(1, static_cast<int>((zhi - zlo) / spacing));
  const int ny = std::max(1, static_cast<int>((yhi - ylo) / spacing));

  std::mt19937 rng(1337u);
  std::uniform_real_distribution<float> angle(0.f, 6.2831853f);
  std::uniform_real_distribution<float> speed(0.f, seedJitterXZ);

  std::vector<Particle> cpuParticles;
  cpuParticles.reserve(static_cast<size_t>(nx * ny * nz));

  for (int ix = 0; ix < nx; ix++) {
    for (int iy = 0; iy < ny; iy++) {
      for (int iz = 0; iz < nz; iz++) {
        Particle p{};
        p.pos = glm::vec4(xlo + ix * spacing, ylo + iy * spacing,
                          zlo + iz * spacing, 1.f);
        // Small random xz velocity (gravity provides the collapse).
        float a = angle(rng), s = speed(rng);
        p.vel = glm::vec4(std::cos(a) * s, 0.f, std::sin(a) * s, 0.f);
        p.predict = glm::vec4(0.f);
        // Color varies with height so the slosh/mixing is visible.
        float t = static_cast<float>(iy) / ny;
        p.color = glm::vec4(0.2f + 0.6f * t, 0.4f, 0.9f - 0.5f * t, 1.f);
        cpuParticles.push_back(p);
      }
    }
  }

  numParticles = static_cast<uint32_t>(cpuParticles.size());
  assert(numParticles <= kMaxParticles);

  std::cout << "[paint_splatter] seed numParticles=" << numParticles
            << "  jitterXZ=" << seedJitterXZ << "  particle[0].pos=("
            << cpuParticles[0].pos.x << "," << cpuParticles[0].pos.y << ","
            << cpuParticles[0].pos.z << ")" << std::endl;

  // Staging buffer -> upload to every per-frame SSBO.
  vgeu::VgeuBuffer stagingBuffer(
      globalAllocator->getAllocator(), sizeof(Particle), kMaxParticles,
      vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);
  std::memcpy(stagingBuffer.getMappedData(), cpuParticles.data(),
              sizeof(Particle) * numParticles);

  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    vgeu::oneTimeSubmit(
        device, commandPool, queue,
        [&](const vk::raii::CommandBuffer& cmdBuffer) {
          cmdBuffer.copyBuffer(
              stagingBuffer.getBuffer(), particleBuffers[i]->getBuffer(),
              vk::BufferCopy(0, 0, sizeof(Particle) * numParticles));
        });
  }
}

// Restart: stop the GPU, reseed the lattice, and reset the queue-ownership
// bootstrap (computeFirstUse) so the next compute dispatch skips its acquire
// barrier exactly like a fresh start (keeps validation clean).
void VgeExample::restartSimulation() {
  device.waitIdle();
  // M5: clear the canvas of live fluid (emit will repopulate). The SSBO slots
  // are simply abandoned by setting the live count to 0; no reseed needed.
  numParticles = 0;
  prevParticleCount = 0;
  pendingEmits.clear();
  poolFull = false;
  // Do NOT reset computeFirstUse here: unlike startup, the buffers are already
  // mid-ping-pong with a pending graphics->compute release. Skipping the next
  // compute acquire would leave that release unconsumed and the following
  // graphics release would duplicate it (VkBufferMemoryBarrier-buffer-00003).
  readbackPending.assign(MAX_CONCURRENT_FRAMES, 0);
  readbackTimer = 0.f;
  readbackRequest = false;
}

// Per-frame host-visible marker buffers (one Particle slot per spoid) drawn as
// points with the particle pipeline. Host-written each frame; not part of the
// compute<->graphics ping-pong.
void VgeExample::createMarkerBuffers() {
  markerBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    markerBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(Particle), kMaxSpoids,
        vk::BufferUsageFlagBits::eVertexBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
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
  // Compute set: 7 storage buffers per frame (particles, cellCount, cellStart,
  // cellOffset, sortedIds, deltaP, particlesPrev for the ping-pong predict).
  poolSizes.emplace_back(vk::DescriptorType::eStorageBuffer,
                         MAX_CONCURRENT_FRAMES * 7u);
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

  // Spoid marker pipeline: identical state (same layout + Particle vertex
  // input), but larger, bordered-disc shaders so the eyedroppers stand out.
  auto mVertCode =
      vgeu::readFile(getShadersPath() + "/paint_splatter/marker.vert.spv");
  auto mFragCode =
      vgeu::readFile(getShadersPath() + "/paint_splatter/marker.frag.spv");
  vk::raii::ShaderModule mVertModule =
      vgeu::createShaderModule(device, mVertCode);
  vk::raii::ShaderModule mFragModule =
      vgeu::createShaderModule(device, mFragCode);
  std::array<vk::PipelineShaderStageCreateInfo, 2> markerStageCIs{
      vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                        vk::ShaderStageFlagBits::eVertex,
                                        *mVertModule, "main", nullptr),
      vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                        vk::ShaderStageFlagBits::eFragment,
                                        *mFragModule, "main", nullptr),
  };
  vk::GraphicsPipelineCreateInfo markerPipelineCI(
      vk::PipelineCreateFlags(), markerStageCIs, &vertexInputSCI,
      &inputAssemblySCI, nullptr, &viewportSCI, &rasterizationSCI,
      &multisampleSCI, &depthStencilSCI, &colorBlendSCI, &dynamicSCI,
      *pipelineLayout, *renderPass);
  markerPipeline = vk::raii::Pipeline(device, pipelineCache, markerPipelineCI);
}

// ---------------------------------------------------------------------------
// Compute setup: mirrors particle.cpp prepareGraphics / prepareCompute
// ---------------------------------------------------------------------------
// Per-frame neighbor-grid buffers (device-local). Rebuilt every substep.
void VgeExample::createGridBuffers() {
  auto makeBuf = [&](uint32_t count, bool transferDst) {
    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer;
    if (transferDst) usage |= vk::BufferUsageFlagBits::eTransferDst;
    return std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(uint32_t), count, usage,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0);
  };
  cellCountBuffers.reserve(MAX_CONCURRENT_FRAMES);
  cellStartBuffers.reserve(MAX_CONCURRENT_FRAMES);
  cellOffsetBuffers.reserve(MAX_CONCURRENT_FRAMES);
  sortedIdBuffers.reserve(MAX_CONCURRENT_FRAMES);
  deltaPBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    cellCountBuffers.push_back(
        makeBuf(numCells, true));  // zeroed via fillBuffer
    cellStartBuffers.push_back(makeBuf(numCells + 1u, false));
    cellOffsetBuffers.push_back(makeBuf(numCells + 1u, false));
    sortedIdBuffers.push_back(makeBuf(kMaxParticles, false));
    // deltaP: one vec4 per particle
    deltaPBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(glm::vec4), kMaxParticles,
        vk::BufferUsageFlagBits::eStorageBuffer,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0));
  }
}

// Kernel normalization constants (3D), precomputed on host (avoid per-thread
// pow). poly6: 315/(64 pi h^9); spiky gradient magnitude: 45/(pi h^6).
static float kPoly6Const(float h) {
  return 315.f / (64.f * glm::pi<float>() * std::pow(h, 9.f));
}
static float kSpikyConst(float h) {
  return 45.f / (glm::pi<float>() * std::pow(h, 6.f));
}

void VgeExample::prepareCompute() {
  computeFirstUse.assign(MAX_CONCURRENT_FRAMES, 1);

  // --- Grid dimensions from the fluid domain + smoothing radius ---
  // M5: the domain spans the full canvas in x,z (droplets land anywhere).
  const float half = kDomainHalf;  // domain half-extent in x,z
  const float h = kSmoothingRadius;
  gridDim = glm::ivec3(static_cast<int>(std::ceil((2.f * half) / h)),
                       static_cast<int>(std::ceil(kDomainHeight / h)),
                       static_cast<int>(std::ceil((2.f * half) / h)));
  numCells = static_cast<uint32_t>(gridDim.x) * gridDim.y * gridDim.z;
  std::cout << "[paint_splatter] gridDim=(" << gridDim.x << "," << gridDim.y
            << "," << gridDim.z << ")  numCells=" << numCells << std::endl;

  // --- Rest density from the seed lattice (mass = 1, so rho0 = sum poly6) ---
  {
    const float s = kParticleSpacing;
    const float kp = kPoly6Const(h);
    float restRho = 0.f;
    int reach = static_cast<int>(std::ceil(h / s)) + 1;
    for (int dx = -reach; dx <= reach; dx++)
      for (int dy = -reach; dy <= reach; dy++)
        for (int dz = -reach; dz <= reach; dz++) {
          float r2 = (dx * dx + dy * dy + dz * dz) * s * s;
          if (r2 < h * h) {
            float t = h * h - r2;
            restRho += kp * t * t * t;
          }
        }
    rho0 = restRho;
    std::cout << "[paint_splatter] rho0 (rest density) = " << rho0 << std::endl;
  }

  createGridBuffers();

  // --- Initialize the compute UBO (per-frame copies filled below) ---
  compute.ubo = ComputeUbo{};
  compute.ubo.dt = kFixedDt;
  compute.ubo.particleCount = numParticles;
  compute.ubo.gravity = gravity;
  compute.ubo.h = h;
  compute.ubo.canvasMin = glm::vec4(-half, -kDomainHeight, -half, 0.f);
  compute.ubo.canvasMax = glm::vec4(half, 0.f, half, 0.f);
  compute.ubo.gridDim = glm::ivec4(gridDim, static_cast<int>(numCells));
  compute.ubo.rho0 = rho0;
  compute.ubo.epsCFM = epsCFM;
  compute.ubo.scorrK = scorrK;
  compute.ubo.scorrDq = scorrDqRatio * h;
  compute.ubo.scorrN = scorrN;
  compute.ubo.xsphC = xsphC;
  compute.ubo.kPoly6 = kPoly6Const(h);
  compute.ubo.kSpiky = kSpikyConst(h);
  {
    float dq2 = compute.ubo.scorrDq * compute.ubo.scorrDq;
    float t = h * h - dq2;
    float wq = compute.ubo.kPoly6 * t * t * t;
    compute.ubo.scorrDenom = (wq > 0.f) ? 1.f / wq : 0.f;
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
  createEmitPipeline();

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
  // 0 particle SSBO (cur), 1 ComputeUbo, 2 cellCount, 3 cellStart,
  // 4 cellOffset, 5 sortedIds, 6 deltaP, 7 particlesPrev (read-only, prev
  // buffer of the ping-pong chain; used by pbf_predict). All visible to every
  // PBF compute stage.
  layoutBindings.emplace_back(0, vk::DescriptorType::eStorageBuffer, 1,
                              vk::ShaderStageFlagBits::eCompute);
  layoutBindings.emplace_back(1, vk::DescriptorType::eUniformBuffer, 1,
                              vk::ShaderStageFlagBits::eCompute);
  for (uint32_t b = 2; b <= 6; b++) {
    layoutBindings.emplace_back(b, vk::DescriptorType::eStorageBuffer, 1,
                                vk::ShaderStageFlagBits::eCompute);
  }
  layoutBindings.emplace_back(7, vk::DescriptorType::eStorageBuffer, 1,
                              vk::ShaderStageFlagBits::eCompute);

  vk::DescriptorSetLayoutCreateInfo layoutCI({}, layoutBindings);
  compute.descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutCI);

  vk::PipelineLayoutCreateInfo pipelineLayoutCI({},
                                                *compute.descriptorSetLayout);
  compute.pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutCI);
}

void VgeExample::createComputePipeline() {
  auto makePipeline = [&](const std::string& name) {
    auto code = vgeu::readFile(getShadersPath() + "/paint_splatter/" + name +
                               ".comp.spv");
    vk::raii::ShaderModule module = vgeu::createShaderModule(device, code);
    vk::PipelineShaderStageCreateInfo stageCI(
        vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute,
        *module, "main", nullptr);
    vk::ComputePipelineCreateInfo pipelineCI(vk::PipelineCreateFlags(), stageCI,
                                             *compute.pipelineLayout);
    return vk::raii::Pipeline(device, pipelineCache, pipelineCI);
  };
  compute.pipeline = makePipeline("pbf_predict");
  compute.gridCount = makePipeline("grid_count");
  compute.gridScan = makePipeline("grid_scan");
  compute.gridScatter = makePipeline("grid_scatter");
  compute.lambda = makePipeline("pbf_lambda");
  compute.delta = makePipeline("pbf_delta");
  compute.apply = makePipeline("pbf_apply");
  compute.finalize = makePipeline("pbf_finalize");
}

// Emit pipeline (M5): its own layout = the compute descriptor-set layout (so it
// can write the particle SSBO at binding 0) plus an EmitPush push-constant
// range carrying the per-burst parameters.
void VgeExample::createEmitPipeline() {
  vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eCompute, 0,
                                  sizeof(EmitPush));
  vk::PipelineLayoutCreateInfo layoutCI({}, *compute.descriptorSetLayout,
                                        pushRange);
  compute.emitPipelineLayout = vk::raii::PipelineLayout(device, layoutCI);

  auto code =
      vgeu::readFile(getShadersPath() + "/paint_splatter/emit.comp.spv");
  vk::raii::ShaderModule module = vgeu::createShaderModule(device, code);
  vk::PipelineShaderStageCreateInfo stageCI(
      vk::PipelineShaderStageCreateFlags(), vk::ShaderStageFlagBits::eCompute,
      *module, "main", nullptr);
  vk::ComputePipelineCreateInfo pipelineCI(vk::PipelineCreateFlags(), stageCI,
                                           *compute.emitPipelineLayout);
  compute.emit = vk::raii::Pipeline(device, pipelineCache, pipelineCI);
}

// Reserve a contiguous slot range from the single global live count and enqueue
// one burst. With the ping-pong chain there is ONE evolving sim, so the burst
// is queued once (not per buffer) and drained into the current buffer; the
// predict pass then carries it forward. Append-only (no compaction until M6).
void VgeExample::enqueueDrop(const glm::vec3& origin, float holeRadius,
                             const glm::vec3& color, float emissionVel,
                             float concentration, int amount) {
  if (amount <= 0) return;
  uint32_t want = static_cast<uint32_t>(amount);
  if (numParticles >= kMaxParticles) {
    poolFull = true;
    return;
  }
  uint32_t avail = kMaxParticles - numParticles;
  uint32_t count = std::min(want, avail);
  if (count < want) poolFull = true;

  // Spawn on a JITTERED LATTICE at ~rest density (not random-in-a-ball). A
  // regular grid + small jitter guarantees a minimum particle separation, so no
  // two particles land on top of each other -- random sampling occasionally
  // clumps pairs, and those overlaps pop apart on the first solve (spawn
  // "explosion"). The shader lays `count` particles on a cube lattice of side
  // `ceil(cbrt(count))` at this spacing; holeRadius widens the spacing (a
  // bigger hole => a bigger, sparser drop) but never below the rest spacing.
  const int side = std::max(
      1, static_cast<int>(std::ceil(std::cbrt(static_cast<float>(count)))));
  const float restSpacing = kParticleSpacing;
  const float wantSpacing = (2.f * holeRadius) / static_cast<float>(side);
  const float spacing = std::max(restSpacing, wantSpacing);

  EmitPush push{};
  // originRadius.w carries the lattice spacing (see emit.comp lattice
  // sampling).
  push.originRadius = glm::vec4(origin, spacing);
  // Initial velocity is downward toward the floor (+Y in this engine's world).
  push.velConc = glm::vec4(0.f, emissionVel, 0.f, concentration);
  push.color = glm::vec4(color, 0.f);
  push.baseIndex = numParticles;
  push.count = count;
  push.seed = emitSeedCounter++;
  push._pad = 0u;

  numParticles += count;
  pendingEmits.push_back(push);
}

// Drain the pending bursts into the CURRENT buffer (ping-pong chain): one emit
// dispatch per burst, each writing its reserved [baseIndex, baseIndex+count)
// slot range. Runs before predict so the solver sees the new particles this
// frame; predict skips these slots (i >= prevCount) and carries the rest from
// the prev buffer. Drained once per frame -- the chain propagates them onward.
void VgeExample::recordEmit(const vk::raii::CommandBuffer& cmd,
                            uint32_t frame) {
  if (pendingEmits.empty()) return;
  // Bind the particle SSBO via the emit layout (its push-constant range makes
  // it a distinct, incompatible layout from the PBF passes' layout).
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                         *compute.emitPipelineLayout, 0,
                         *compute.descriptorSets[frame], nullptr);
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *compute.emit);
  for (const EmitPush& push : pendingEmits) {
    cmd.pushConstants<EmitPush>(*compute.emitPipelineLayout,
                                vk::ShaderStageFlagBits::eCompute, 0, push);
    cmd.dispatch((push.count + 255u) / 256u, 1, 1);
  }
  pendingEmits.clear();
  // emit (shader write) -> predict (shader read/write) barrier.
  vk::MemoryBarrier mb(
      vk::AccessFlagBits::eShaderWrite,
      vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
  cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                      vk::PipelineStageFlagBits::eComputeShader,
                      vk::DependencyFlags{}, mb, nullptr, nullptr);
}

// Read keyboard input, run the spoid controller, clamp spoids to the domain,
// and turn this frame's emit triggers into droplet bursts. IJKL move spoids in
// the canvas plane, U/O change height, Space releases a drop from each selected
// spoid (WASD/arrows belong to the camera, so spoids use a separate key set).
void VgeExample::updateSpoids() {
  if (!spoidController) return;
  GLFWwindow* w = vgeuWindow->getGLFWwindow();
  InputState in;
  glm::vec3 mv(0.f);
  if (glfwGetKey(w, GLFW_KEY_L) == GLFW_PRESS) mv.x += 1.f;
  if (glfwGetKey(w, GLFW_KEY_J) == GLFW_PRESS) mv.x -= 1.f;
  if (glfwGetKey(w, GLFW_KEY_I) == GLFW_PRESS) mv.z += 1.f;
  if (glfwGetKey(w, GLFW_KEY_K) == GLFW_PRESS) mv.z -= 1.f;
  if (glfwGetKey(w, GLFW_KEY_O) == GLFW_PRESS)
    mv.y += 1.f;  // lower toward floor
  if (glfwGetKey(w, GLFW_KEY_U) == GLFW_PRESS) mv.y -= 1.f;  // raise
  in.move = mv;
  bool space = glfwGetKey(w, GLFW_KEY_SPACE) == GLFW_PRESS;
  in.emit = space && !spaceWasDown;  // edge-triggered: one burst per press
  spaceWasDown = space;

  std::vector<int> emitDrops;
  spoidController->update(frameTimer, spoids, in, emitDrops);

  // Keep spoids inside the domain and above the floor (y < 0).
  const float m = 0.05f;
  for (auto& s : spoids) {
    s.pos.x = glm::clamp(s.pos.x, -kDomainHalf + m, kDomainHalf - m);
    s.pos.z = glm::clamp(s.pos.z, -kDomainHalf + m, kDomainHalf - m);
    s.pos.y = glm::clamp(s.pos.y, -kDomainHeight + m, -0.1f);
  }

  for (int idx : emitDrops) {
    const Spoid& s = spoids[idx];
    enqueueDrop(s.pos, s.holeRadius, s.color, s.emissionVelocity,
                s.concentration, s.amount);
  }
}

// Arrange the spoids evenly on a circle in the X-Z plane (n-way angular split).
// A single spoid sits at the centre; >=2 spread around a circle whose radius
// fits inside the domain. Heights (pos.y) are preserved so vertical keyboard
// control still works.
void VgeExample::arrangeSpoidsCircle() {
  const int n = static_cast<int>(spoids.size());
  if (n == 0) return;
  if (n == 1) {
    spoids[0].pos.x = 0.f;
    spoids[0].pos.z = 0.f;
    return;
  }
  const float radius = kDomainHalf * 0.6f;  // inside the domain walls
  for (int i = 0; i < n; i++) {
    const float ang =
        glm::two_pi<float>() * static_cast<float>(i) / static_cast<float>(n);
    spoids[i].pos.x = radius * std::cos(ang);
    spoids[i].pos.z = radius * std::sin(ang);
  }
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
    // Ping-pong: binding 0 = this frame's buffer (cur, written), binding 7 =
    // the previous frame's buffer (prev, read by pbf_predict). Mirrors
    // particle.cpp:252-261 (prevFrameIdx -> in, i -> out).
    const uint32_t prevIdx =
        (i + MAX_CONCURRENT_FRAMES - 1u) % MAX_CONCURRENT_FRAMES;
    vk::DescriptorBufferInfo ssboInfo(particleBuffers[i]->getBuffer(), 0,
                                      particleBuffers[i]->getBufferSize());
    vk::DescriptorBufferInfo prevInfo(
        particleBuffers[prevIdx]->getBuffer(), 0,
        particleBuffers[prevIdx]->getBufferSize());
    vk::DescriptorBufferInfo uboInfo =
        compute.uniformBuffers[i]->descriptorInfo();
    vk::DescriptorBufferInfo cellCountInfo(
        cellCountBuffers[i]->getBuffer(), 0,
        cellCountBuffers[i]->getBufferSize());
    vk::DescriptorBufferInfo cellStartInfo(
        cellStartBuffers[i]->getBuffer(), 0,
        cellStartBuffers[i]->getBufferSize());
    vk::DescriptorBufferInfo cellOffsetInfo(
        cellOffsetBuffers[i]->getBuffer(), 0,
        cellOffsetBuffers[i]->getBufferSize());
    vk::DescriptorBufferInfo sortedInfo(sortedIdBuffers[i]->getBuffer(), 0,
                                        sortedIdBuffers[i]->getBufferSize());
    vk::DescriptorBufferInfo deltaInfo(deltaPBuffers[i]->getBuffer(), 0,
                                       deltaPBuffers[i]->getBufferSize());

    std::array<vk::WriteDescriptorSet, 8> writes{
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 0, 0,
                               vk::DescriptorType::eStorageBuffer, nullptr,
                               ssboInfo),
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 1, 0,
                               vk::DescriptorType::eUniformBuffer, nullptr,
                               uboInfo),
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 2, 0,
                               vk::DescriptorType::eStorageBuffer, nullptr,
                               cellCountInfo),
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 3, 0,
                               vk::DescriptorType::eStorageBuffer, nullptr,
                               cellStartInfo),
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 4, 0,
                               vk::DescriptorType::eStorageBuffer, nullptr,
                               cellOffsetInfo),
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 5, 0,
                               vk::DescriptorType::eStorageBuffer, nullptr,
                               sortedInfo),
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 6, 0,
                               vk::DescriptorType::eStorageBuffer, nullptr,
                               deltaInfo),
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 7, 0,
                               vk::DescriptorType::eStorageBuffer, nullptr,
                               prevInfo),
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
  // canvasInfo.w doubles as the density-debug-color flag for particle.vert.
  globalUbo.canvasInfo = glm::vec4(kCanvasWorld * 0.5f, kCanvasWorld * 0.5f,
                                   kCanvasWorld, colorByDensity ? 1.f : 0.f);
  std::memcpy(uniformBuffers[currentFrameIndex]->getMappedData(), &globalUbo,
              sizeof(GlobalUbo));
}

void VgeExample::updateComputeUbo() {
  // Substep dt: split the frame into `substeps` smaller PBF steps for
  // stability.
  float frameDt = useFixedDt ? kFixedDt : frameTimer;
  int sub = std::max(1, substeps);
  compute.ubo.dt = frameDt / static_cast<float>(sub);
  compute.ubo.particleCount = numParticles;
  // Ping-pong: predict carries [0, prevCount) from the prev buffer; emit wrote
  // the freshly-spawned [prevCount, numParticles) into cur this frame.
  compute.ubo.prevCount = prevParticleCount;
  compute.ubo.gravity = gravity;
  // Fluid domain box (collision walls): the full canvas footprint in x,z.
  compute.ubo.canvasMin =
      glm::vec4(-kDomainHalf, -kDomainHeight, -kDomainHalf, 0.f);
  compute.ubo.canvasMax = glm::vec4(kDomainHalf, 0.f, kDomainHalf, 0.f);
  // Live-tunable PBF params (grid dims / kernel constants fixed in prepare).
  compute.ubo.rho0 = rho0;
  compute.ubo.epsCFM = epsCFM;
  compute.ubo.scorrK = scorrK;
  compute.ubo.scorrDq = scorrDqRatio * compute.ubo.h;
  compute.ubo.scorrN = scorrN;
  compute.ubo.xsphC = xsphC;
  compute.ubo.velDamp = velDamp;
  compute.ubo.velClampFactor = velClampFactor;
  compute.ubo.solverRelax = solverRelax;
  {
    float dq2 = compute.ubo.scorrDq * compute.ubo.scorrDq;
    float t = compute.ubo.h * compute.ubo.h - dq2;
    float wq = compute.ubo.kPoly6 * t * t * t;
    compute.ubo.scorrDenom = (wq > 0.f) ? 1.f / wq : 0.f;
  }
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
  if (numParticles == 0) return;
  const Particle* data = static_cast<const Particle*>(
      readbackBuffers[currentFrameIndex]->getMappedData());
  float minY = data[0].pos.y, maxY = data[0].pos.y;
  double sumRho = 0.0, maxRho = 0.0;
  double sumSpeed = 0.0, maxSpeed = 0.0;  // vel.xyz magnitude (vel.w = density)
  uint32_t nearFloor = 0;                 // within 0.3 of the floor (y >= -0.3)
  double pileSpeedSum = 0.0, pileSpeedMax = 0.0;  // jiggle of the settled pile
  for (uint32_t i = 0; i < numParticles; i++) {
    minY = std::min(minY, data[i].pos.y);
    maxY = std::max(maxY, data[i].pos.y);
    sumRho += data[i].vel.w;  // finalize stored rho/rho0 here
    maxRho = std::max(maxRho, static_cast<double>(data[i].vel.w));
    double sp = std::sqrt(data[i].vel.x * data[i].vel.x +
                          data[i].vel.y * data[i].vel.y +
                          data[i].vel.z * data[i].vel.z);
    sumSpeed += sp;
    maxSpeed = std::max(maxSpeed, sp);
    if (data[i].pos.y >= -0.3f) {
      nearFloor++;
      pileSpeedSum += sp;
      pileSpeedMax = std::max(pileSpeedMax, sp);
    }
  }
  std::cout << "[paint_splatter] y[" << minY << "," << maxY
            << "] | rho/rho0 mean=" << (sumRho / numParticles)
            << " max=" << maxRho
            << " | speed mean=" << (sumSpeed / numParticles)
            << " max=" << maxSpeed
            << " | nearFloor%=" << (100.0 * nearFloor / numParticles)
            << " | PILE speed mean="
            << (nearFloor ? pileSpeedSum / nearFloor : 0.0)
            << " max=" << pileSpeedMax << std::endl;
}

// ---------------------------------------------------------------------------
// render / draw
// ---------------------------------------------------------------------------
void VgeExample::render() {
  if (!prepared) return;

  if (restartRequested) {
    restartRequested = false;
    restartSimulation();
  }

  // Spoid keyboard control + emit triggers (Task 10).
  updateSpoids();

  // Task 9 (M5): optional hardcoded auto-drop from the first spoid's position
  // to exercise the emit pass hands-free (off by default; spoids drive
  // emission).
  if (autoEmit && !spoids.empty()) {
    autoEmitTimer += frameTimer;
    if (autoEmitTimer >= autoEmitInterval) {
      autoEmitTimer = 0.f;
      // Drop from every selected spoid (matches the Space behaviour); if none
      // are selected, fall back to all spoids so auto-emit always does
      // something visible.
      bool anySelected = false;
      for (const Spoid& s : spoids) anySelected |= s.selected;
      for (const Spoid& s : spoids) {
        if (anySelected && !s.selected) continue;
        enqueueDrop(s.pos, s.holeRadius, s.color, s.emissionVelocity,
                    s.concentration, s.amount);
      }
    }
  }

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

  // Emit pending droplet bursts first so the new particles are part of this
  // frame's solve (mirrors the per-frame order in the design spec).
  recordEmit(compute.cmdBuffers[currentFrameIndex], currentFrameIndex);

  compute.cmdBuffers[currentFrameIndex].bindDescriptorSets(
      vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 0,
      *compute.descriptorSets[currentFrameIndex], nullptr);

  // Run the PBF solver, substepping the frame for stability.
  for (int s = 0; s < std::max(1, substeps); s++) {
    recordPbfSubstep(compute.cmdBuffers[currentFrameIndex], currentFrameIndex);
  }

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

  // The current buffer now holds the full live state [0, numParticles). Next
  // frame it becomes the prev buffer, so its live count is what predict must
  // carry forward (ping-pong chain).
  prevParticleCount = numParticles;
}

// One PBF substep: predict -> build neighbor grid -> {lambda, delta, apply} x
// solverIters -> finalize. A compute->compute buffer barrier separates every
// dispatch so each pass sees the previous one's writes (mirrors the multi-pass
// barrier structure of particle.cpp:1568-1619).
void VgeExample::recordPbfSubstep(const vk::raii::CommandBuffer& cmd,
                                  uint32_t frame) {
  const uint32_t groupCount = (numParticles + 255u) / 256u;
  const uint32_t cellGroups = (numCells + 255u) / 256u;

  // Generic compute->compute SSBO barrier (shader write -> shader read/write).
  auto barrier = [&]() {
    vk::MemoryBarrier mb(
        vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::DependencyFlags{}, mb, nullptr, nullptr);
  };
  auto dispatchParticles = [&](const vk::raii::Pipeline& pipe) {
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipe);
    cmd.dispatch(groupCount, 1, 1);
  };

  // 1. predict
  dispatchParticles(compute.pipeline);
  barrier();

  // 2. build neighbor grid: clear counts -> count -> scan -> scatter
  cmd.fillBuffer(cellCountBuffers[frame]->getBuffer(), 0,
                 cellCountBuffers[frame]->getBufferSize(), 0u);
  // transfer-write (fill) -> shader-read barrier
  {
    vk::MemoryBarrier mb(
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                        vk::PipelineStageFlagBits::eComputeShader,
                        vk::DependencyFlags{}, mb, nullptr, nullptr);
  }
  dispatchParticles(compute.gridCount);
  barrier();
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *compute.gridScan);
  cmd.dispatch(1, 1, 1);  // single-invocation serial scan
  barrier();
  dispatchParticles(compute.gridScatter);
  barrier();

  // 3. solve: Jacobi iterations of {lambda -> delta -> apply}
  for (int it = 0; it < std::max(1, solverIters); it++) {
    dispatchParticles(compute.lambda);
    barrier();
    dispatchParticles(compute.delta);
    barrier();
    dispatchParticles(compute.apply);
    barrier();
  }

  // 4. finalize: velocity update + XSPH + commit position
  dispatchParticles(compute.finalize);
  barrier();
  (void)cellGroups;  // grid clear uses fillBuffer, not a dispatch
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

  // --- Spoid markers (M5): one point per spoid via the particle pipeline ---
  if (showSpoids && !spoids.empty()) {
    uint32_t n =
        std::min<uint32_t>(static_cast<uint32_t>(spoids.size()), kMaxSpoids);
    Particle* m = static_cast<Particle*>(
        markerBuffers[currentFrameIndex]->getMappedData());
    for (uint32_t i = 0; i < n; i++) {
      m[i].pos = glm::vec4(spoids[i].pos, 1.f);
      m[i].vel = glm::vec4(0.f);      // density-debug tint reads vel.w (=0)
      m[i].predict = glm::vec4(0.f);  // unused by the renderer
      // Marker shows the spoid's paint colour (selection is shown in the UI).
      m[i].color = glm::vec4(spoids[i].color, 1.f);
    }
    drawCmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eGraphics, *markerPipeline);
    drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0,
        {*descriptorSets[currentFrameIndex]}, nullptr);
    vk::DeviceSize offset(0);
    drawCmdBuffers[currentFrameIndex].bindVertexBuffers(
        0, markerBuffers[currentFrameIndex]->getBuffer(), offset);
    drawCmdBuffers[currentFrameIndex].draw(n, 1, 0, 0);
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

    ImGui::Text("Particles  : %u / %u", numParticles, kMaxParticles);
    ImGui::Checkbox("Show particles", &showParticles);
    if (poolFull) {
      ImGui::TextColored(ImVec4(1.f, 0.4f, 0.3f, 1.f),
                         "particle pool full - drop rejected");
    }

    // --- emit (M5 Task 9: auto-drop) ---
    ImGui::Checkbox("auto emit", &autoEmit);
    ImGui::SliderFloat("auto interval (s)", &autoEmitInterval, 0.1f, 2.f);

    // --- Spoids (M5 Task 10) ---
    if (ImGui::CollapsingHeader("Spoids", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::TextWrapped(
          "Keys: IJKL move, U/O height, Space = drop from selected");
      ImGui::Checkbox("show spoids", &showSpoids);

      for (int i = 0; i < static_cast<int>(spoids.size()); i++) {
        ImGui::PushID(i);
        ImGui::Checkbox("##sel", &spoids[i].selected);
        ImGui::SameLine();
        if (ImGui::RadioButton("edit", selectedSpoidUi == i)) {
          selectedSpoidUi = i;
        }
        ImGui::SameLine();
        ImGui::Text("#%d (%.2f, %.2f, %.2f)", i, spoids[i].pos.x,
                    spoids[i].pos.y, spoids[i].pos.z);
        ImGui::PopID();
      }

      if (uiOverlay->button("+ Add spoid") && spoids.size() < kMaxSpoids) {
        Spoid s{};
        s.color = spoidPalette(static_cast<int>(spoids.size()));
        spoids.push_back(s);
        // Re-spread all spoids evenly on a circle in the X-Z plane.
        arrangeSpoidsCircle();
      }
      if (uiOverlay->button("- Remove spoid") && spoids.size() > 1) {
        spoids.pop_back();
        if (selectedSpoidUi >= static_cast<int>(spoids.size())) {
          selectedSpoidUi = static_cast<int>(spoids.size()) - 1;
        }
        arrangeSpoidsCircle();
      }

      if (selectedSpoidUi >= 0 &&
          selectedSpoidUi < static_cast<int>(spoids.size())) {
        Spoid& s = spoids[selectedSpoidUi];
        ImGui::Separator();
        ImGui::Text("Editing spoid #%d", selectedSpoidUi);
        ImGui::SliderFloat("hole radius", &s.holeRadius, 0.02f, 0.5f);
        ImGui::SliderFloat("emission vel", &s.emissionVelocity, 0.f, 8.f);
        ImGui::SliderInt("amount", &s.amount, 10, 2000);
        ImGui::SliderFloat("concentration", &s.concentration, 0.f, 1.f);
        ImGui::ColorEdit3("color", &s.color.x);
      }
    }

    // --- sim params (M3) ---
    ImGui::SliderFloat("gravity (+Y)", &gravity, 0.f, 30.f);
    ImGui::Checkbox("fixed dt (1/120)", &useFixedDt);
    ImGui::SliderFloat("seed jitter xz", &seedJitterXZ, 0.f, 5.f);
    if (uiOverlay->button("Restart")) {
      restartRequested = true;
    }

    // --- PBF solver (M4) ---
    if (ImGui::CollapsingHeader("PBF solver", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Text("rho0 (rest) : %.1f", rho0);
      ImGui::SliderInt("substeps", &substeps, 1, 16);
      ImGui::SliderInt("solverIters", &solverIters, 1, 6);
      ImGui::SliderFloat("solver relax", &solverRelax, 0.05f, 1.f);
      ImGui::SliderFloat("epsCFM", &epsCFM, 1.f, 1000.f);
      ImGui::SliderFloat("scorrK", &scorrK, 0.f, 0.5f);
      ImGui::SliderFloat("scorrDq/h", &scorrDqRatio, 0.05f, 0.5f);
      ImGui::SliderFloat("xsphC", &xsphC, 0.f, 1.f);
      ImGui::SliderFloat("vel damping", &velDamp, 0.f, 20.f);
      ImGui::SliderFloat("vel clamp (CFL, 0=off)", &velClampFactor, 0.f, 0.5f);
      ImGui::Checkbox("color by density", &colorByDensity);
    }

    if (uiOverlay->button("Save (no-op)")) {
      std::cout << "[paint_splatter] Save button pressed (no-op)\n";
    }
  }
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()
