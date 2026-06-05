#include "paint_splatter.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
// stb_image_write: the implementation is compiled once in vgeu_gltf.cpp
// (STB_IMAGE_WRITE_IMPLEMENTATION there); here we only need the declarations.
#include "stb_image_write.h"

// std
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <random>
#include <string>
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
  // Only START-TIME-ONLY settings get CLI flags (everything else is a live
  // ImGui slider). These two are allocated/computed once in prepare() with no
  // runtime path, so they can only be chosen at launch.
  app.add_option("--canvas-res", kCanvasTexRes,
                 "canvas accumulation texture resolution (px, default 2048)");
  app.add_option("--spacing", kParticleSpacing,
                 "PBF rest particle spacing (default 0.03 = 0.3h); sets rho0 + "
                 "emit density");
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

  // One spoid above the canvas centre. Phase 2: the default control mode is
  // pendulum (see spoidControlMode default), so build the controller for it.
  {
    Spoid s{};
    s.color = spoidPalette(0);
    spoids.push_back(s);
  }
  arrangeSpoidsCircle();  // also seeds per-spoid offsetAngle0
  setSpoidControlMode(spoidControlMode);

  createVertexBuffer();
  createIndexBuffer();
  createParticleBuffers();
  createMarkerBuffers();
  createChainBuffers();
  createUniformBuffers();
  createCanvasImage();
  createDescriptorSetLayout();
  createDescriptorPool();
  createDescriptorSets();
  createPipelines();
  createParticlePipeline();
  createLinePipeline();
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
  // Phase 2: apply a pending world-scale change. Grid buffers are pre-sized to
  // maxNumCells, so we only recompute the extents + grid dims and rebuild the
  // canvas quad (device-local, safe after waitIdle). canvasMin/Max are
  // rederived from kDomainHalf/kDomainHeight each frame in updateComputeUbo();
  // gridDim is set once here and carried by the per-frame whole-struct copy.
  if (worldScale != appliedWorldScale) {
    computeScaledDims();
    createVertexBuffer();  // rebuild the canvas quad at the new kCanvasWorld
    compute.ubo.gridDim = glm::ivec4(gridDim, static_cast<int>(numCells));
    for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      std::memcpy(compute.uniformBuffers[i]->getMappedData(), &compute.ubo,
                  sizeof(ComputeUbo));
    }
    std::cout << "[paint_splatter] worldScale=" << appliedWorldScale
              << " gridDim=(" << gridDim.x << "," << gridDim.y << ","
              << gridDim.z << ") numCells=" << numCells << std::endl;
  }
  // M5: clear the canvas of live fluid (emit will repopulate). The SSBO slots
  // are simply abandoned by setting the live count to 0; no reseed needed.
  numParticles = 0;
  prevParticleCount = 0;
  pendingEmits.clear();
  poolFull = false;
  // M6-C-2: reset the compaction live-count state and zero the GPU counters so
  // the next frame's prev-slot read sees an empty buffer (not stale survivors).
  cumEmitted = 0;
  emittedThisFrame = 0;
  liveCountDisplay = 0;
  liveUpperFromReadback = kMaxParticles;
  liveReadbackValid.assign(MAX_CONCURRENT_FRAMES, 0);
  liveCopyCumEmitted.assign(MAX_CONCURRENT_FRAMES, 0);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    vgeu::oneTimeSubmit(
        device, commandPool, queue, [&](const vk::raii::CommandBuffer& cmd) {
          cmd.fillBuffer(liveCountBuffers[i]->getBuffer(), 0,
                         liveCountBuffers[i]->getBufferSize(), 0u);
        });
  }
  // Clear the painting back to blank white paper (canvas stays in GENERAL).
  {
    vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0,
                                    1);
    vgeu::oneTimeSubmit(
        device, commandPool, queue, [&](const vk::raii::CommandBuffer& cmd) {
          vk::ClearColorValue white(std::array<float, 4>{1.f, 1.f, 1.f, 1.f});
          cmd.clearColorImage(canvasImage->getImage(),
                              vk::ImageLayout::eGeneral, white, range);
        });
  }
  // Do NOT reset computeFirstUse here: unlike startup, the buffers are already
  // mid-ping-pong with a pending graphics->compute release. Skipping the next
  // compute acquire would leave that release unconsumed and the following
  // graphics release would duplicate it (VkBufferMemoryBarrier-buffer-00003).
  readbackPending.assign(MAX_CONCURRENT_FRAMES, 0);
  readbackTimer = 0.f;
  readbackRequest = false;
  // User-approved: restart also resets each spoid's colour (palette by index)
  // and position (re-spread on the X-Z circle + reset y). Other params (amount,
  // holeRadius, concentration, ...) and the spoid count are preserved.
  for (int i = 0; i < static_cast<int>(spoids.size()); i++) {
    spoids[i].color = spoidPalette(i);
  }
  arrangeSpoidsCircle();
  // Phase 2: also return the pendulum to its initial state (rebuild the chain
  // from config + reset each spoid's rotary offset phase) so Restart gives a
  // fresh swing, mirroring the keyboard-mode spoid reposition above.
  if (spoidControlMode == SpoidControlMode::Pendulum) {
    resetPendulum();
    for (Spoid& s : spoids) s.offsetPhase = s.offsetAngle0;
  }
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

// Per-frame host-visible buffers for chain visualization. jointMarkerBuffers:
// one Particle slot per node (drawn as marker points). lineBuffers: two slots
// per link (drawn as an eLineList).
void VgeExample::createChainBuffers() {
  jointMarkerBuffers.reserve(MAX_CONCURRENT_FRAMES);
  lineBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    jointMarkerBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(Particle), kMaxChainNodes,
        vk::BufferUsageFlagBits::eVertexBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT));
    lineBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(Particle), kMaxChainNodes * 2,
        vk::BufferUsageFlagBits::eVertexBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT));
  }
}

// Canvas accumulation texture (M6): a single RGBA8 image, the permanent
// painting. Created in GENERAL layout (valid for both the compute imageStore in
// M6-B and the fragment sample) and cleared to opaque white (blank paper).
void VgeExample::createCanvasImage() {
  vk::Extent2D extent(kCanvasTexRes, kCanvasTexRes);
  canvasImage = std::make_unique<vgeu::VgeuImage>(
      device, globalAllocator->getAllocator(), vk::Format::eR8G8B8A8Unorm,
      extent, vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
          vk::ImageUsageFlagBits::eTransferDst |
          vk::ImageUsageFlagBits::eTransferSrc,
      vk::ImageLayout::eUndefined, VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
      vk::ImageAspectFlagBits::eColor, 1);

  canvasSampler = vk::raii::Sampler(
      device,
      vk::SamplerCreateInfo({}, vk::Filter::eLinear, vk::Filter::eLinear,
                            vk::SamplerMipmapMode::eNearest,
                            vk::SamplerAddressMode::eClampToEdge,
                            vk::SamplerAddressMode::eClampToEdge,
                            vk::SamplerAddressMode::eClampToEdge, 0.f, false,
                            1.f, false, vk::CompareOp::eAlways, 0.f, 0.f,
                            vk::BorderColor::eFloatOpaqueWhite, false));

  // Undefined -> General, then clear to opaque white (blank canvas).
  vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
  vgeu::oneTimeSubmit(
      device, commandPool, queue, [&](const vk::raii::CommandBuffer& cmd) {
        vk::ImageMemoryBarrier toGeneral(
            vk::AccessFlags{}, vk::AccessFlagBits::eTransferWrite,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            canvasImage->getImage(), range);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eTransfer,
                            vk::DependencyFlags{}, nullptr, nullptr, toGeneral);
        vk::ClearColorValue white(std::array<float, 4>{1.f, 1.f, 1.f, 1.f});
        cmd.clearColorImage(canvasImage->getImage(), vk::ImageLayout::eGeneral,
                            white, range);
      });
}

// M7 Task 14: export the accumulation canvas to a PNG. device.waitIdle (the
// simplest correct sync), copy the GENERAL-layout RGBA8 image into a
// host-visible buffer (copyImageToBuffer; GENERAL is a valid transfer-src
// layout), then write it with stbi_write_png. Filename is
// build/paint_<timestamp>.png (the app runs from build/). On failure the
// status string reports it and the sim continues.
void VgeExample::saveCanvasPng() {
  const uint32_t w = kCanvasTexRes;
  const uint32_t h = kCanvasTexRes;

  // Stop the GPU so the canvas is settled and safe to read (no in-flight
  // deposit/sample). Heavy but correct, and Save is a rare manual action.
  device.waitIdle();

  vgeu::VgeuBuffer staging(globalAllocator->getAllocator(), 4u, w * h,
                           vk::BufferUsageFlagBits::eTransferDst,
                           VMA_MEMORY_USAGE_AUTO,
                           VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                               VMA_ALLOCATION_CREATE_MAPPED_BIT);

  vgeu::oneTimeSubmit(
      device, commandPool, queue, [&](const vk::raii::CommandBuffer& cmd) {
        vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1,
                                        0, 1);
        // Transition GENERAL -> TRANSFER_SRC_OPTIMAL for the copy (also makes
        // prior deposit/sample writes visible), then restore GENERAL so the
        // descriptor / deposit / sample keep working. Validation flags GENERAL
        // as non-optimal for vkCmdCopyImageToBuffer.
        vk::ImageMemoryBarrier toTransfer(
            vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
            vk::AccessFlagBits::eTransferRead, vk::ImageLayout::eGeneral,
            vk::ImageLayout::eTransferSrcOptimal, VK_QUEUE_FAMILY_IGNORED,
            VK_QUEUE_FAMILY_IGNORED, canvasImage->getImage(), range);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader |
                                vk::PipelineStageFlagBits::eFragmentShader,
                            vk::PipelineStageFlagBits::eTransfer,
                            vk::DependencyFlags{}, nullptr, nullptr,
                            toTransfer);
        vk::BufferImageCopy region(
            0, 0, 0,
            vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0,
                                       1),
            vk::Offset3D(0, 0, 0), vk::Extent3D(w, h, 1));
        cmd.copyImageToBuffer(canvasImage->getImage(),
                              vk::ImageLayout::eTransferSrcOptimal,
                              staging.getBuffer(), region);
        vk::ImageMemoryBarrier backToGeneral(
            vk::AccessFlagBits::eTransferRead,
            vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
            vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            canvasImage->getImage(), range);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                            vk::PipelineStageFlagBits::eComputeShader |
                                vk::PipelineStageFlagBits::eFragmentShader,
                            vk::DependencyFlags{}, nullptr, nullptr,
                            backToGeneral);
      });

  // Timestamped filename (relative to cwd = build/).
  std::time_t t = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &t);
#else
  localtime_r(&t, &tm);
#endif
  char name[64];
  std::strftime(name, sizeof(name), "paint_%Y%m%d_%H%M%S.png", &tm);

  int ok = stbi_write_png(name, static_cast<int>(w), static_cast<int>(h), 4,
                          staging.getMappedData(), static_cast<int>(w * 4u));
  if (ok) {
    saveStatus = std::string("saved ") + name;
    std::cout << "[paint_splatter] " << saveStatus << std::endl;
  } else {
    saveStatus = std::string("SAVE FAILED: ") + name;
    std::cerr << "[paint_splatter] " << saveStatus << std::endl;
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
  // set=0, binding=1: canvas accumulation texture (fragment stage, sampled by
  // canvas.frag). The particle/marker pipelines share this layout but don't
  // read binding 1 (a bound-but-unused sampler is valid).
  std::array<vk::DescriptorSetLayoutBinding, 2> bindings{
      vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                     vk::ShaderStageFlagBits::eVertex),
      vk::DescriptorSetLayoutBinding(1,
                                     vk::DescriptorType::eCombinedImageSampler,
                                     1, vk::ShaderStageFlagBits::eFragment),
  };
  vk::DescriptorSetLayoutCreateInfo layoutCI({}, bindings);
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
  // Compute set: 9 storage buffers per frame (particles, cellCount, cellStart,
  // cellOffset, sortedIds, deltaP, particlesPrev for the ping-pong predict,
  // plus liveCount + prevLiveCount for M6-C-2 compaction).
  poolSizes.emplace_back(vk::DescriptorType::eStorageBuffer,
                         MAX_CONCURRENT_FRAMES * 9u);
  // Canvas texture sampler (one per graphics descriptor set).
  poolSizes.emplace_back(vk::DescriptorType::eCombinedImageSampler,
                         MAX_CONCURRENT_FRAMES);
  // Canvas storage image (one per compute descriptor set, deposit pass).
  poolSizes.emplace_back(vk::DescriptorType::eStorageImage,
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
  // The canvas image is sampled in GENERAL layout (it's also a compute storage
  // image in M6-B); descriptorImageInfo is stable so build it once per set.
  std::vector<vk::DescriptorImageInfo> imageInfos;
  imageInfos.reserve(uniformBuffers.size());
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
  writeDescriptorSets.reserve(uniformBuffers.size() * 2u);

  for (uint32_t i = 0; i < static_cast<uint32_t>(uniformBuffers.size()); i++) {
    bufferInfos.push_back(uniformBuffers[i]->descriptorInfo());
    writeDescriptorSets.emplace_back(*descriptorSets[i], 0, 0,
                                     vk::DescriptorType::eUniformBuffer,
                                     nullptr, bufferInfos.back());
    imageInfos.push_back(canvasImage->descriptorImageInfo(
        *canvasSampler, vk::ImageLayout::eGeneral));
    writeDescriptorSets.emplace_back(*descriptorSets[i], 1, 0,
                                     vk::DescriptorType::eCombinedImageSampler,
                                     imageInfos.back());
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

void VgeExample::createLinePipeline() {
  // Vertex input: same Particle layout as the particle/marker pipelines.
  vk::VertexInputBindingDescription bindingDesc(0, sizeof(Particle),
                                                vk::VertexInputRate::eVertex);
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

  vk::PipelineInputAssemblyStateCreateInfo inputAssemblySCI(
      vk::PipelineInputAssemblyStateCreateFlags(),
      vk::PrimitiveTopology::eLineList);
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

  auto vCode =
      vgeu::readFile(getShadersPath() + "/paint_splatter/chain_line.vert.spv");
  auto fCode =
      vgeu::readFile(getShadersPath() + "/paint_splatter/chain_line.frag.spv");
  vk::raii::ShaderModule vModule = vgeu::createShaderModule(device, vCode);
  vk::raii::ShaderModule fModule = vgeu::createShaderModule(device, fCode);
  std::array<vk::PipelineShaderStageCreateInfo, 2> stageCIs{
      vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                        vk::ShaderStageFlagBits::eVertex,
                                        *vModule, "main", nullptr),
      vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                        vk::ShaderStageFlagBits::eFragment,
                                        *fModule, "main", nullptr),
  };
  vk::GraphicsPipelineCreateInfo lineCI(
      vk::PipelineCreateFlags(), stageCIs, &vertexInputSCI, &inputAssemblySCI,
      nullptr, &viewportSCI, &rasterizationSCI, &multisampleSCI,
      &depthStencilSCI, &colorBlendSCI, &dynamicSCI, *pipelineLayout,
      *renderPass);
  linePipeline = vk::raii::Pipeline(device, pipelineCache, lineCI);
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
  // Phase 2: size the per-cell buffers to maxNumCells (capacity at the largest
  // worldScale) so changing the scale on Restart only updates gridDim/numCells,
  // never reallocates these buffers or rewrites their descriptors.
  const uint32_t cellCap = std::max(numCells, maxNumCells);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    cellCountBuffers.push_back(
        makeBuf(cellCap, true));  // zeroed via fillBuffer
    cellStartBuffers.push_back(makeBuf(cellCap + 1u, false));
    cellOffsetBuffers.push_back(makeBuf(cellCap + 1u, false));
    sortedIdBuffers.push_back(makeBuf(kMaxParticles, false));
    // deltaP: one vec4 per particle
    deltaPBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(glm::vec4), kMaxParticles,
        vk::BufferUsageFlagBits::eStorageBuffer,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0));
  }
}

// M6-C-2: per-slot GPU live-count buffers (uint[1]) + host-visible readback
// copies. The device-local counter is reset to 0 each frame (binding 9) and
// atomic-incremented by pbf_predict (compaction) and emit; binding 10 reads the
// prev slot's counter. Initialized to 0 so the very first prev-slot read is
// valid (not garbage).
void VgeExample::createLiveCountBuffers() {
  liveCountBuffers.clear();
  liveCountReadbackBuffers.clear();
  liveCountBuffers.reserve(MAX_CONCURRENT_FRAMES);
  liveCountReadbackBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    liveCountBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(uint32_t), 1,
        vk::BufferUsageFlagBits::eStorageBuffer |
            vk::BufferUsageFlagBits::eTransferDst |  // fillBuffer(0)
            vk::BufferUsageFlagBits::eTransferSrc,   // copy -> readback
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0));
    liveCountReadbackBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(uint32_t), 1,
        vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT));
  }
  liveReadbackValid.assign(MAX_CONCURRENT_FRAMES, 0);
  liveCopyCumEmitted.assign(MAX_CONCURRENT_FRAMES, 0);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    vgeu::oneTimeSubmit(
        device, commandPool, queue, [&](const vk::raii::CommandBuffer& cmd) {
          cmd.fillBuffer(liveCountBuffers[i]->getBuffer(), 0,
                         liveCountBuffers[i]->getBufferSize(), 0u);
        });
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

// Phase 2 world scale: recompute the canvas + simulation domain + neighbor grid
// from base extents * worldScale. The smoothing radius h (and thus the particle
// size) is NOT scaled, so a bigger world simply holds more, smaller-looking
// particles and the M8 fluid tuning (h / rho0 / spacing) is preserved. Pure
// CPU: sets kCanvasWorld/kDomainHalf/kDomainHeight, gridDim/numCells (current
// scale), and maxNumCells (capacity at kMaxWorldScale, used to pre-size grid
// buffers so no realloc/descriptor rewrite is needed when the scale changes on
// Restart).
void VgeExample::computeScaledDims() {
  // Base (unscaled) extents -- the original compile-time literals.
  constexpr float kBaseCanvasWorld = 4.0f;
  constexpr float kBaseDomainHalf = 2.0f;  // == kBaseCanvasWorld * 0.5
  constexpr float kBaseDomainHeight = 3.0f;
  const float s = glm::clamp(worldScale, 1.0f, kMaxWorldScale);
  worldScale = s;
  appliedWorldScale = s;
  kCanvasWorld = kBaseCanvasWorld * s;
  kDomainHalf = kBaseDomainHalf * s;
  kDomainHeight = kBaseDomainHeight * s;
  const float h = kSmoothingRadius;  // grid cell size = h (unscaled)
  gridDim = glm::ivec3(static_cast<int>(std::ceil((2.f * kDomainHalf) / h)),
                       static_cast<int>(std::ceil(kDomainHeight / h)),
                       static_cast<int>(std::ceil((2.f * kDomainHalf) / h)));
  numCells = static_cast<uint32_t>(gridDim.x) * gridDim.y * gridDim.z;
  // Capacity at the largest allowed scale (grid buffers are sized to this
  // once).
  const glm::ivec3 maxDim(
      static_cast<int>(std::ceil((2.f * kBaseDomainHalf * kMaxWorldScale) / h)),
      static_cast<int>(std::ceil((kBaseDomainHeight * kMaxWorldScale) / h)),
      static_cast<int>(
          std::ceil((2.f * kBaseDomainHalf * kMaxWorldScale) / h)));
  maxNumCells = static_cast<uint32_t>(maxDim.x) * maxDim.y * maxDim.z;
}

void VgeExample::prepareCompute() {
  computeFirstUse.assign(MAX_CONCURRENT_FRAMES, 1);

  // --- Grid dimensions from the fluid domain + smoothing radius ---
  // M5: the domain spans the full canvas in x,z (droplets land anywhere).
  computeScaledDims();             // sets kDomainHalf/Height/kCanvasWorld +
                                   // gridDim/numCells
  const float half = kDomainHalf;  // domain half-extent in x,z
  const float h = kSmoothingRadius;
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
  createLiveCountBuffers();

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
  compute.ubo.cohesionFloor = cohesionFloor;  // M8 bounded cohesion
  compute.ubo.dpClampFactor = dpClampFactor;  // M8 dp clamp
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
  // 8 = canvas accumulation image (storage), written by deposit.comp (M6).
  // Bound in every compute set; only the deposit pass reads it.
  layoutBindings.emplace_back(8, vk::DescriptorType::eStorageImage, 1,
                              vk::ShaderStageFlagBits::eCompute);
  // 9 = liveCount (cur, M6-C-2 compaction atomic counter), 10 = prevLiveCount
  // (prev slot's counter, read-only, for pbf_predict's exact compaction range).
  layoutBindings.emplace_back(9, vk::DescriptorType::eStorageBuffer, 1,
                              vk::ShaderStageFlagBits::eCompute);
  layoutBindings.emplace_back(10, vk::DescriptorType::eStorageBuffer, 1,
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
  compute.integrate = makePipeline("integrate");  // M8: per-substep integration
  compute.gridCount = makePipeline("grid_count");
  compute.gridScan = makePipeline("grid_scan");
  compute.gridScatter = makePipeline("grid_scatter");
  compute.lambda = makePipeline("pbf_lambda");
  compute.delta = makePipeline("pbf_delta");
  compute.apply = makePipeline("pbf_apply");
  compute.finalize = makePipeline("pbf_finalize");
  // Deposit shares the PBF pipeline layout (binding 8 = canvas image). Built
  // here but dispatched on the GRAPHICS queue (see buildCommandBuffers).
  compute.deposit = makePipeline("deposit");
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
                             float concentration, int amount,
                             const glm::vec3& segPrev) {
  if (amount <= 0) return;
  uint32_t count = static_cast<uint32_t>(amount);
  // M6-C-2: slots are now assigned by a GPU atomic (no host baseIndex). Clamp
  // the burst against a conservative estimate of slots already in use this
  // frame (prevParticleCount is an upper bound on the surviving particles +
  // emittedThisFrame) so we never overflow the pool; the emit shader also drops
  // any overflow (idx >= kMax) as a hard backstop.
  uint32_t usedEst =
      std::min(kMaxParticles, prevParticleCount + emittedThisFrame);
  if (usedEst >= kMaxParticles) {
    poolFull = true;
    return;
  }
  uint32_t avail = kMaxParticles - usedEst;
  if (count > avail) {
    count = avail;
    poolFull = true;
  }

  // Spawn shape (M8): two modes (toggled by sphericalSpawn).
  //  - BALL (mode 1): uniform in a ball of radius `holeRadius` -> holeRadius
  //    drives droplet size. Viable now that the soft epsCFM resolves the
  //    close-pair overlaps that random placement makes (the lattice existed to
  //    avoid those pops). The droplet's density = amount / ball-volume; if that
  //    differs from rest density the cohesion/incompressibility constraints
  //    just relax it toward rho0 on the first frames.
  //  - LATTICE (mode 0): jittered cube grid at the REST spacing -> a fresh
  //    droplet is exactly at rest density; size comes from `amount`
  //    (diameter ~= cbrt(amount)*restSpacing). Guaranteed min separation.
  const float spacing = kParticleSpacing;
  const float wRadius = sphericalSpawn ? holeRadius : spacing;

  EmitPush push{};
  // originRadius.w = ball radius (mode 1) or lattice spacing (mode 0).
  push.originRadius = glm::vec4(origin, wRadius);
  // Initial velocity is downward toward the floor (+Y in this engine's world).
  push.velConc = glm::vec4(0.f, emissionVel, 0.f, concentration);
  push.color = glm::vec4(color, 0.f);
  push.mode =
      sphericalSpawn ? 1u : 0u;  // spawn shape (slot from GPU atomicAdd)
  push.count = count;
  push.seed = emitSeedCounter++;
  push._pad = 0u;
  // Ball sweep start: ignored by the lattice mode; for the ball mode it sweeps
  // the spawn along [segPrev -> origin] (a moving stream). A static burst
  // passes segPrev == origin so the sweep collapses to a point.
  push.segPrev = glm::vec4(segPrev, 0.f);

  emittedThisFrame += count;
  cumEmitted += count;
  pendingEmits.push_back(push);
}

// Drain the pending bursts into the CURRENT buffer: one emit dispatch per
// burst. Since M6-C-2 each spawned particle reserves its slot via an atomicAdd
// on liveCount (binding 9), so emit must run AFTER pbf_predict has compacted
// the survivors -- the new particles then append right after them. Drained once
// per frame; the ping-pong chain carries them forward next frame.
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
  // emit (shader write) -> grid/solve (shader read/write) barrier.
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
  // Drive the controller with the same fixed dt as the fluid when useFixedDt is
  // on. The pendulum predict is symplectic Euler, whose bounded-energy property
  // only holds at a CONSTANT step -- variable frameTimer makes it drift (and be
  // fps-dependent). A fixed step also keeps the swing in lock-step with the
  // PBF.
  const float controllerDt = useFixedDt ? kFixedDt : frameTimer;
  spoidController->update(controllerDt, spoids, in, emitDrops);

  // Spoids may move OUTSIDE the canvas footprint (so a stroke can enter/leave
  // the painting) -- only a generous outer bound stops them being lost. y stays
  // between the floor and the ceiling.
  if (spoidControlMode == SpoidControlMode::Keyboard) {
    const float m = 0.05f;
    const float xzBound = 2.f * kDomainHalf;  // one canvas-width past each edge
    for (auto& s : spoids) {
      s.pos.x = glm::clamp(s.pos.x, -xzBound, xzBound);
      s.pos.z = glm::clamp(s.pos.z, -xzBound, xzBound);
      s.pos.y = glm::clamp(s.pos.y, -kDomainHeight + m, -0.1f);
    }
  }

  for (int idx : emitDrops) {
    const Spoid& s = spoids[idx];
    // Space / keyboard drop = a static burst (segPrev == origin).
    enqueueDrop(s.pos, s.holeRadius, s.color, s.emissionVelocity,
                s.concentration, s.amount, s.pos);
  }
}

// Arrange the spoids evenly on a circle in the X-Z plane (n-way angular split).
// A single spoid sits at the centre; >=2 spread around a circle whose radius
// fits inside the domain. Heights (pos.y) are preserved so vertical keyboard
// control still works.
void VgeExample::arrangeSpoidsCircle() {
  const int n = static_cast<int>(spoids.size());
  if (n == 0) return;
  // Re-level every spoid to the default spawn height (y) too, so
  // adding/removing a spoid resets the whole set to a clean starting layout.
  const float y = -1.25f;  // matches Spoid::pos default (M8: start height /2)
  // Phase 2: also seed each spoid's rotary-offset start angle evenly around the
  // ring so that with offsetR>0 the spoids paint an evenly-spaced rosette.
  auto seedAngle = [&](int i) {
    spoids[i].offsetAngle0 =
        glm::two_pi<float>() * static_cast<float>(i) / static_cast<float>(n);
    spoids[i].offsetPhase = spoids[i].offsetAngle0;
  };
  if (n == 1) {
    spoids[0].pos = glm::vec3(0.f, y, 0.f);
    seedAngle(0);
    return;
  }
  const float radius = kDomainHalf * 0.3f;  // compact circle near the centre
  for (int i = 0; i < n; i++) {
    const float ang =
        glm::two_pi<float>() * static_cast<float>(i) / static_cast<float>(n);
    spoids[i].pos =
        glm::vec3(radius * std::cos(ang), y, radius * std::sin(ang));
    seedAngle(i);
  }
}

// Build pendulumChains[0] as a straight line hung from the pivot, displaced
// from the +Y (down) vertical by (initTheta, initPhi). node[0] = pivot (fixed).
void VgeExample::resetPendulum() {
  if (pendulumChains.empty()) pendulumChains.resize(1);
  PendulumChain& c = pendulumChains[0];
  const int n =
      std::max(1, std::min(c.numLinks, static_cast<int>(kMaxChainNodes) - 1));
  c.numLinks = n;
  // uniform link length + uniform default mass if not sized to n.
  c.linkLength.assign(n, c.totalLength / static_cast<float>(n));
  if (static_cast<int>(c.bobMass.size()) != n) c.bobMass.assign(n, 1.0f);
  // direction from vertical +Y by (theta, phi): theta from +Y, phi about Y.
  const float st = std::sin(c.initTheta), ct = std::cos(c.initTheta);
  const glm::vec3 dir(st * std::cos(c.initPhi), ct, st * std::sin(c.initPhi));
  c.nodes.assign(n + 1, PendulumNode{});
  c.nodes[0].pos = c.pivot;
  c.nodes[0].prevPos = c.pivot;
  c.nodes[0].invMass = 0.f;  // pivot pinned
  glm::vec3 p = c.pivot;
  for (int i = 1; i <= n; i++) {
    p += dir * c.linkLength[i - 1];
    c.nodes[i].pos = p;
    c.nodes[i].prevPos = p;
    c.nodes[i].vel = glm::vec3(0.f);
    float m = c.bobMass[i - 1];
    c.nodes[i].invMass = (m > 0.f) ? 1.0f / m : 0.f;  // m<=0 => pinned
  }
  // initial tip velocity, decomposed in the bob's sphere tangent plane at the
  // start pose: meridional e_theta (radial/中心방향, swings in the vertical
  // plane) + azimuthal e_phi (tangential/접선방향, circles about +Y). Both are
  // unit, perpendicular to the string, so neither fights the length constraint.
  if (n >= 1 && (c.initSpeedRadial != 0.f || c.initSpeedTangential != 0.f)) {
    const float sp = std::sin(c.initPhi), cp = std::cos(c.initPhi);
    const glm::vec3 eTheta(ct * cp, -st, ct * sp);  // d(dir)/dtheta
    const glm::vec3 ePhi(-sp, 0.f, cp);             // d(dir)/dphi (normalized)
    c.nodes[n].vel = eTheta * c.initSpeedRadial + ePhi * c.initSpeedTangential;
  }
}

glm::vec3 PendulumSpoidController::emissionPoint(const PendulumChain& c,
                                                 const Spoid& s) const {
  if (c.nodes.size() < 2) return c.pivot;
  int ni =
      (s.nodeIndex < 0) ? static_cast<int>(c.nodes.size()) - 1 : s.nodeIndex;
  ni = std::max(1, std::min(ni, static_cast<int>(c.nodes.size()) - 1));
  glm::vec3 node = c.nodes[ni].pos;
  if (s.offsetR == 0.f) return node;
  // link direction at this node (node - parent).
  glm::vec3 d = node - c.nodes[ni - 1].pos;
  float dl = glm::length(d);
  if (dl < 1e-6f) return node;
  d /= dl;
  // orthonormal basis of the plane perpendicular to the string. Pick the world
  // axis LEAST aligned with d as the reference, so the basis stays well
  // conditioned across the whole swing -- in particular through theta~90 where
  // the old abs(d.x)<0.9 test flipped the basis (the spoid's offset circle
  // glitched there). cross(d, ref) is never near-zero with this choice.
  glm::vec3 ad(std::abs(d.x), std::abs(d.y), std::abs(d.z));
  glm::vec3 ref = (ad.x <= ad.y && ad.x <= ad.z) ? glm::vec3(1, 0, 0)
                  : (ad.y <= ad.z)               ? glm::vec3(0, 1, 0)
                                                 : glm::vec3(0, 0, 1);
  glm::vec3 e1 = glm::normalize(glm::cross(d, ref));
  glm::vec3 e2 = glm::cross(d, e1);
  float ph = s.offsetPhase;  // angle0 + omega*t, advanced in update()
  return node + s.offsetR * (std::cos(ph) * e1 + std::sin(ph) * e2);
}

void PendulumSpoidController::update(float dt, std::vector<Spoid>& spoids,
                                     const InputState& in,
                                     std::vector<int>& emitDrops) {
  (void)in;
  (void)emitDrops;  // pendulum mode uses stream emission (render loop)
  for (PendulumChain& c : chains) stepChain(c, dt);  // no-op until later
  if (chains.empty()) return;
  for (Spoid& s : spoids) {
    s.offsetPhase += s.offsetOmega * dt;
    s.pos = emissionPoint(chains[0], s);
  }
}

void PendulumSpoidController::stepChain(PendulumChain& c, float dt) {
  if (c.nodes.size() < 2 || dt <= 0.f) return;
  const int sub = std::max(1, c.substeps);
  const float sdt = dt / static_cast<float>(sub);
  const float g = gravity;      // +Y (down on screen)
  const float maxSpeed = 50.f;  // coarse CFL safety net (world u/s)
  for (int s = 0; s < sub; s++) {
    // 1. predict (skip the fixed pivot, node 0)
    for (size_t i = 1; i < c.nodes.size(); i++) {
      PendulumNode& nd = c.nodes[i];
      if (nd.invMass == 0.f) continue;  // pinned
      nd.vel.y += g * sdt;
      nd.prevPos = nd.pos;
      nd.pos += nd.vel * sdt;
    }
    // 2. distance constraints (Gauss-Seidel)
    for (int it = 0; it < std::max(1, c.iters); it++) {
      for (size_t i = 1; i < c.nodes.size(); i++) {
        PendulumNode& a = c.nodes[i - 1];
        PendulumNode& b = c.nodes[i];
        float w = a.invMass + b.invMass;
        if (w == 0.f) continue;  // both pinned
        glm::vec3 d = b.pos - a.pos;
        float len = glm::length(d);
        if (len < 1e-6f) continue;
        float L = c.linkLength[i - 1];
        glm::vec3 corr = (len - L) / (w * len) * d;
        a.pos += a.invMass * corr;
        b.pos -= b.invMass * corr;
      }
    }
    // 3. velocity update + 4. damping
    const float airK = std::max(0.f, 1.f - c.airDamping * sdt);
    for (size_t i = 1; i < c.nodes.size(); i++) {
      PendulumNode& nd = c.nodes[i];
      if (nd.invMass == 0.f) {
        nd.vel = glm::vec3(0.f);
        continue;
      }
      nd.vel = (nd.pos - nd.prevPos) / sdt;
      nd.vel *= airK;  // air resistance (global)
    }
    // joint/string friction: damp the relative velocity component ALONG each
    // link (energy lost at the string end).
    for (size_t i = 1; i < c.nodes.size(); i++) {
      PendulumNode& a = c.nodes[i - 1];
      PendulumNode& b = c.nodes[i];
      glm::vec3 d = b.pos - a.pos;
      float len = glm::length(d);
      if (len < 1e-6f) continue;
      glm::vec3 axis = d / len;
      glm::vec3 rel = b.vel - a.vel;
      float along = glm::dot(rel, axis);
      glm::vec3 damp = axis * (along * c.jointDamping);
      if (b.invMass > 0.f) b.vel -= damp;
      if (a.invMass > 0.f) a.vel += damp;
    }
    // stability: CFL speed cap + soft-clamp y above the floor (pivot exempt).
    for (size_t i = 1; i < c.nodes.size(); i++) {
      PendulumNode& nd = c.nodes[i];
      float sp = glm::length(nd.vel);
      if (sp > maxSpeed) nd.vel *= maxSpeed / sp;
      // floor y=0, ceiling y=-ceilingHeight (= kDomainHeight, scaled). keep
      // bobs above the floor; the bound tracks the world scale.
      nd.pos.y = glm::clamp(nd.pos.y, -ceilingHeight + 0.05f, -0.1f);
    }
  }
}

void VgeExample::setSpoidControlMode(SpoidControlMode mode) {
  spoidControlMode = mode;
  if (mode == SpoidControlMode::Pendulum) {
    resetPendulum();
    for (Spoid& s : spoids) s.offsetPhase = s.offsetAngle0;
    spoidController = std::make_unique<PendulumSpoidController>(
        pendulumChains, gravity, kDomainHeight);
  } else {
    spoidController = std::make_unique<KeyboardSpoidController>();
  }
}

void VgeExample::startTopViewAnim() {
  if (cameraAnim.active || cameraAnim.locked) {
    // toggle off: release back to the orbit controller.
    cameraAnim.active = false;
    cameraAnim.locked = false;
    return;
  }
  cameraAnim.fromEye = camera.getPosition();
  // aim at the canvas centre (0,0,0), the orbit target.
  cameraAnim.fromTarget = glm::vec3(0.f);
  cameraAnim.toEye = glm::vec3(0.f, -6.f, 0.f);  // overhead (world -Y is up)
  cameraAnim.toTarget = glm::vec3(0.f);
  cameraAnim.up =
      glm::vec3(0.f, 0.f, -1.f);  // non-degenerate for a +Y view dir
  cameraAnim.t = 0.f;
  cameraAnim.active = true;
}

void VgeExample::updateCameraAnim() {
  if (!cameraAnim.active && !cameraAnim.locked) return;
  glm::vec3 eye = cameraAnim.toEye, target = cameraAnim.toTarget;
  if (cameraAnim.active) {
    cameraAnim.t += frameTimer;
    float u = glm::clamp(cameraAnim.t / cameraAnim.duration, 0.f, 1.f);
    float e = u * u * (3.f - 2.f * u);  // smoothstep ease in/out
    eye = glm::mix(cameraAnim.fromEye, cameraAnim.toEye, e);
    target = glm::mix(cameraAnim.fromTarget, cameraAnim.toTarget, e);
    if (u >= 1.f) {
      cameraAnim.active = false;
      cameraAnim.locked = true;
    }
  }
  camera.setViewTarget(eye, target, cameraAnim.up);
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
    // Canvas as a storage image (no sampler) in GENERAL layout for deposit.
    vk::DescriptorImageInfo canvasStoreInfo(
        nullptr, *canvasImage->getImageView(), vk::ImageLayout::eGeneral);
    // M6-C-2 ping-pong: binding 9 = this slot's live counter (cur, written),
    // binding 10 = the prev slot's counter (read by pbf_predict for the exact
    // compaction range). Same prevIdx as the particle 0/7 ping-pong above.
    vk::DescriptorBufferInfo liveInfo(liveCountBuffers[i]->getBuffer(), 0,
                                      liveCountBuffers[i]->getBufferSize());
    vk::DescriptorBufferInfo prevLiveInfo(
        liveCountBuffers[prevIdx]->getBuffer(), 0,
        liveCountBuffers[prevIdx]->getBufferSize());

    std::array<vk::WriteDescriptorSet, 11> writes{
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
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 8, 0,
                               vk::DescriptorType::eStorageImage,
                               canvasStoreInfo),
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 9, 0,
                               vk::DescriptorType::eStorageBuffer, nullptr,
                               liveInfo),
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 10, 0,
                               vk::DescriptorType::eStorageBuffer, nullptr,
                               prevLiveInfo),
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
  globalUbo.renderParams = glm::vec4(pointScale, 0.f, 0.f, 0.f);
  std::memcpy(uniformBuffers[currentFrameIndex]->getMappedData(), &globalUbo,
              sizeof(GlobalUbo));
}

void VgeExample::updateComputeUbo() {
  // Substep dt: split the frame into `substeps` smaller PBF steps for
  // stability.
  float frameDt = useFixedDt ? kFixedDt : frameTimer;
  int sub = std::max(1, substeps);
  compute.ubo.dt = frameDt / static_cast<float>(sub);
  // M6-C-2 dispatch/draw upper bounds. prevParticleCount carries last frame's
  // bound; shrink it with the readback-derived bound (both are valid upper
  // bounds on the prev buffer's live count, so min stays valid). numParticles
  // then adds this frame's fresh emits. Correctness is enforced by the
  // in-shader liveCount/prevLiveCount guards -- these host counts only need to
  // be >= the true counts, which they are by construction.
  prevParticleCount = std::min(prevParticleCount, liveUpperFromReadback);
  numParticles = std::min(kMaxParticles, prevParticleCount + emittedThisFrame);
  compute.ubo.particleCount = numParticles;
  compute.ubo.prevCount =
      prevParticleCount;  // reference only (GPU guards rule)
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
  compute.ubo.dryRate = dryRate;  // used by deposit (M6-C drying)
  compute.ubo.depositStrength = depositStrength;
  compute.ubo.depositHeight = depositHeight;
  compute.ubo.depositRadius = depositRadius;  // canvas stamp size (world units)
  compute.ubo.drySettle = drySettle;  // drying freezes near-floor motion
  compute.ubo.cohesionFloor = cohesionFloor;  // M8: bounded cohesion pull
  compute.ubo.dpClampFactor = dpClampFactor;  // M8: UBO-driven dp clamp
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
  float minY = 1e30f, maxY = -1e30f;
  // Horizontal bounding-box extent (M8 cohesion check): a cohesive droplet
  // keeps a small x/z extent while falling; a dispersing cloud grows it.
  float minX = 1e30f, maxX = -1e30f, minZ = 1e30f, maxZ = -1e30f;
  double sumRho = 0.0, maxRho = 0.0;
  double sumSpeed = 0.0, maxSpeed = 0.0;  // vel.xyz magnitude (vel.w = density)
  uint32_t nearFloor = 0;                 // within 0.3 of the floor (y >= -0.3)
  double pileSpeedSum = 0.0, pileSpeedMax = 0.0;  // jiggle of the settled pile
  uint32_t nValid = 0;  // M6-C-2: skip the offscreen-parked compaction tail
  for (uint32_t i = 0; i < numParticles; i++) {
    if (data[i].pos.y > 1e6f) continue;  // pbf_finalize parked stale slots here
    nValid++;
    minY = std::min(minY, data[i].pos.y);
    maxY = std::max(maxY, data[i].pos.y);
    minX = std::min(minX, data[i].pos.x);
    maxX = std::max(maxX, data[i].pos.x);
    minZ = std::min(minZ, data[i].pos.z);
    maxZ = std::max(maxZ, data[i].pos.z);
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
  if (nValid == 0) return;
  std::cout
      << "[paint_splatter] live=" << nValid << " (disp=" << liveCountDisplay
      << ") y[" << minY << "," << maxY
      << "] | rho/rho0 mean=" << (sumRho / nValid) << " max=" << maxRho
      << " | speed mean=" << (sumSpeed / nValid) << " max=" << maxSpeed
      << " | nearFloor%=" << (100.0 * nearFloor / nValid)
      << " | ext x=" << (maxX - minX) << " z=" << (maxZ - minZ)
      << " | PILE speed mean=" << (nearFloor ? pileSpeedSum / nearFloor : 0.0)
      << " max=" << pileSpeedMax
      << (spoidControlMode == SpoidControlMode::Pendulum &&
                  !pendulumChains.empty() && pendulumChains[0].nodes.size() >= 2
              ? " | tip=(" +
                    std::to_string(pendulumChains[0].nodes.back().pos.x) + "," +
                    std::to_string(pendulumChains[0].nodes.back().pos.y) + "," +
                    std::to_string(pendulumChains[0].nodes.back().pos.z) +
                    ") tipv=" +
                    std::to_string(
                        glm::length(pendulumChains[0].nodes.back().vel))
              : std::string())
      << std::endl;
}

// M6-C-2: read this slot's live-count copy (recorded when the slot was last
// submitted, its fence now waited on). Used for ImGui display and to derive a
// shrinking upper bound on the live count. live_now <= R + (emits since the
// copy), a valid bound that tracks particles drying out so the dispatch range
// does not peg at kMax. NOT used for correctness (the in-shader guards are).
void VgeExample::consumeLiveCountReadback() {
  uint32_t slot = currentFrameIndex;
  if (!liveReadbackValid[slot]) return;
  uint32_t r = *static_cast<const uint32_t*>(
      liveCountReadbackBuffers[slot]->getMappedData());
  liveCountDisplay = std::min(r, kMaxParticles);
  uint64_t since = cumEmitted - liveCopyCumEmitted[slot];
  uint64_t bound = static_cast<uint64_t>(r) + since;
  liveUpperFromReadback =
      static_cast<uint32_t>(std::min<uint64_t>(bound, kMaxParticles));
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

  // M7: handle a pending canvas->PNG save (does device.waitIdle internally).
  if (saveRequested) {
    saveRequested = false;
    saveCanvasPng();
  }

  // M6-C-2: count this frame's fresh emits (feeds the dispatch upper bound).
  emittedThisFrame = 0;

  // Spoid keyboard control + emit triggers (Task 10).
  updateSpoids();

  // Emission: continuous STREAM (M8) takes precedence over the discrete
  // auto-burst. Both drop from the selected spoids (or all, if none selected).
  bool anySelected = false;
  for (const Spoid& s : spoids) anySelected |= s.selected;
  auto isEmitter = [&](const Spoid& s) { return !anySelected || s.selected; };

  if (streamMode && !spoids.empty()) {
    // Continuous stream: each frame emit streamRate*dt particles from every
    // emitting spoid, swept along its motion this frame ([prevPos -> pos]) so a
    // fast stroke stays connected. A fractional accumulator carries the
    // sub-particle remainder so low rates still emit evenly.
    const float dt = frameTimer;
    for (Spoid& s : spoids) {
      if (!isEmitter(s)) continue;
      // massDrainRate==0 => the reservoir never depletes, so don't gate on it.
      if (massDrainRate > 0.f && s.paintMass <= 0.f) continue;  // empty
      s.emitAccum += streamRate * dt;
      int n = static_cast<int>(s.emitAccum);
      if (n > 0) {
        s.emitAccum -= static_cast<float>(n);
        enqueueDrop(s.pos, s.holeRadius, s.color, s.emissionVelocity,
                    s.concentration, n, s.prevPos);
        s.paintMass -= massDrainRate * static_cast<float>(n);
      }
    }
  } else if (autoEmit && !spoids.empty()) {
    // Discrete auto-burst on a timer (M5): a static ball/lattice per interval.
    autoEmitTimer += frameTimer;
    if (autoEmitTimer >= autoEmitInterval) {
      autoEmitTimer = 0.f;
      for (const Spoid& s : spoids) {
        if (!isEmitter(s)) continue;
        enqueueDrop(s.pos, s.holeRadius, s.color, s.emissionVelocity,
                    s.concentration, s.amount, s.pos);
      }
    }
  }

  // Remember each spoid's position for next frame's stream sweep (do this AFTER
  // emitting so the stream spans the motion just taken).
  for (Spoid& s : spoids) s.prevPos = s.pos;

  updateCameraAnim();  // overrides the orbit view while animating/locked
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
  consumeLiveCountReadback();  // M6-C-2: live count for display + dispatch
                               // bound

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

  const vk::raii::CommandBuffer& ccmd = compute.cmdBuffers[currentFrameIndex];
  const uint32_t groupCount = (numParticles + 255u) / 256u;
  auto computeBarrier = [&]() {
    vk::MemoryBarrier mb(
        vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    ccmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eComputeShader,
                         vk::DependencyFlags{}, mb, nullptr, nullptr);
  };

  // M6-C-2: reset THIS slot's live counter to 0 before compaction/emit
  // atomic-append into it. The prev slot's counter (binding 10) is left intact
  // so pbf_predict can read the exact valid range of the prev buffer.
  ccmd.fillBuffer(liveCountBuffers[currentFrameIndex]->getBuffer(), 0,
                  liveCountBuffers[currentFrameIndex]->getBufferSize(), 0u);
  {
    vk::MemoryBarrier mb(
        vk::AccessFlagBits::eTransferWrite,
        vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite);
    ccmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                         vk::PipelineStageFlagBits::eComputeShader,
                         vk::DependencyFlags{}, mb, nullptr, nullptr);
  }

  ccmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                          *compute.pipelineLayout, 0,
                          *compute.descriptorSets[currentFrameIndex], nullptr);

  // Compaction-predict, ONCE per frame (re-running per substep would
  // double-count via the atomic). Packs the prev buffer's survivors to the
  // front of cur and sets liveCount; dispatched on the upper bound, the
  // prevLiveCount[0] guard trims to the exact survivor range.
  if (groupCount > 0) {
    ccmd.bindPipeline(vk::PipelineBindPoint::eCompute, *compute.pipeline);
    ccmd.dispatch(groupCount, 1, 1);
    computeBarrier();  // predict writes liveCount -> emit must see it
  }

  // Emit appends fresh particles AFTER the survivors via the same atomic.
  // recordEmit binds the emit pipeline layout, so rebind the PBF sets after.
  recordEmit(ccmd, currentFrameIndex);
  ccmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                          *compute.pipelineLayout, 0,
                          *compute.descriptorSets[currentFrameIndex], nullptr);

  // Grid build + solve + finalize, substepped for stability.
  for (int s = 0; s < std::max(1, substeps); s++) {
    recordPbfSubstep(ccmd, currentFrameIndex);
  }

  // M6-C-2: copy the final live count into the host-visible readback buffer
  // (ImGui display + dispatch-bound shrink). Same compute queue -> no QFOT.
  {
    vk::MemoryBarrier mb(vk::AccessFlagBits::eShaderWrite,
                         vk::AccessFlagBits::eTransferRead);
    ccmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eTransfer,
                         vk::DependencyFlags{}, mb, nullptr, nullptr);
    ccmd.copyBuffer(liveCountBuffers[currentFrameIndex]->getBuffer(),
                    liveCountReadbackBuffers[currentFrameIndex]->getBuffer(),
                    vk::BufferCopy(0, 0, sizeof(uint32_t)));
    liveReadbackValid[currentFrameIndex] = 1;
    liveCopyCumEmitted[currentFrameIndex] = cumEmitted;
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

  // Carry this frame's dispatch upper bound as next frame's prev bound (an
  // upper bound on the prev buffer's live count). updateComputeUbo shrinks it
  // with the readback-derived bound before use. The exact prev count is read on
  // the GPU (binding 10) by pbf_predict, so this host value need only be >=.
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

  // NOTE (M6-C-2/M8): predict (the compaction step) runs ONCE per frame in
  // buildComputeCommandBuffers, before emit -- NOT here -- because its atomic
  // append would double-count if repeated per substep. Force INTEGRATION,
  // however, must repeat per substep, so it runs here (integrate.comp) as the
  // first pass of each substep: apply gravity + predict x* = pos + v*dt over
  // the compacted cur buffer, then rebuild the grid over those predicted
  // positions.

  // 0. integrate external forces (gravity) -> predict x* (M8, per substep)
  dispatchParticles(compute.integrate);
  barrier();

  // 1. build neighbor grid: clear counts -> count -> scan -> scatter
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
  // Mirrors particle.cpp:1413-1431. The particle buffer is read both by the
  // deposit compute dispatch (shader read) below and the vertex stage, so the
  // acquire covers both stages/accesses.
  if (graphics.queueFamilyIndex != compute.queueFamilyIndex) {
    vk::BufferMemoryBarrier bufBarrier(
        vk::AccessFlags{},
        vk::AccessFlagBits::eVertexAttributeRead |
            vk::AccessFlagBits::eShaderRead,
        compute.queueFamilyIndex, graphics.queueFamilyIndex,
        particleBuffers[currentFrameIndex]->getBuffer(), 0ull,
        particleBuffers[currentFrameIndex]->getBufferSize());

    drawCmdBuffers[currentFrameIndex].pipelineBarrier(
        vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eVertexInput |
            vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlags{}, nullptr, bufBarrier, nullptr);
  }

  // --- Deposit (M6): stamp near-floor particles into the canvas. Run as a
  //     compute dispatch ON THE GRAPHICS QUEUE (canvas image is graphics-owned,
  //     so no cross-queue ownership transfer), before the render pass; an image
  //     barrier then makes the writes visible to the fragment sample. ---
  if (numParticles > 0) {
    drawCmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eCompute, *compute.deposit);
    drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 0,
        *compute.descriptorSets[currentFrameIndex], nullptr);
    drawCmdBuffers[currentFrameIndex].dispatch((numParticles + 255u) / 256u, 1,
                                               1);
    vk::ImageMemoryBarrier canvasBarrier(
        vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        canvasImage->getImage(),
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    drawCmdBuffers[currentFrameIndex].pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eFragmentShader, vk::DependencyFlags{},
        nullptr, nullptr, canvasBarrier);
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

  // --- Phase 2: pendulum chain (links + joints) ---
  if (showChain && spoidControlMode == SpoidControlMode::Pendulum &&
      !pendulumChains.empty()) {
    const PendulumChain& c = pendulumChains[0];
    const uint32_t nodeCount = std::min<uint32_t>(
        static_cast<uint32_t>(c.nodes.size()), kMaxChainNodes);
    if (nodeCount >= 2) {
      // joints
      Particle* jm = static_cast<Particle*>(
          jointMarkerBuffers[currentFrameIndex]->getMappedData());
      for (uint32_t i = 0; i < nodeCount; i++) {
        jm[i].pos = glm::vec4(c.nodes[i].pos, 1.f);
        jm[i].vel = glm::vec4(0.f);
        jm[i].predict = glm::vec4(0.f);
        jm[i].color = glm::vec4(0.9f, 0.9f, 0.2f, 1.f);  // joint = yellow
      }
      // links: two verts per segment (node i-1 -> node i)
      Particle* lv = static_cast<Particle*>(
          lineBuffers[currentFrameIndex]->getMappedData());
      const uint32_t links = nodeCount - 1;
      for (uint32_t i = 0; i < links; i++) {
        lv[2 * i].pos = glm::vec4(c.nodes[i].pos, 1.f);
        lv[2 * i].color = glm::vec4(0.7f, 0.7f, 0.7f, 1.f);  // string = grey
        lv[2 * i + 1].pos = glm::vec4(c.nodes[i + 1].pos, 1.f);
        lv[2 * i + 1].color = glm::vec4(0.7f, 0.7f, 0.7f, 1.f);
      }
      vk::DeviceSize off(0);
      // lines
      drawCmdBuffers[currentFrameIndex].bindPipeline(
          vk::PipelineBindPoint::eGraphics, *linePipeline);
      drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0,
          {*descriptorSets[currentFrameIndex]}, nullptr);
      drawCmdBuffers[currentFrameIndex].bindVertexBuffers(
          0, lineBuffers[currentFrameIndex]->getBuffer(), off);
      drawCmdBuffers[currentFrameIndex].draw(links * 2, 1, 0, 0);
      // joints (marker pipeline)
      drawCmdBuffers[currentFrameIndex].bindPipeline(
          vk::PipelineBindPoint::eGraphics, *markerPipeline);
      drawCmdBuffers[currentFrameIndex].bindVertexBuffers(
          0, jointMarkerBuffers[currentFrameIndex]->getBuffer(), off);
      drawCmdBuffers[currentFrameIndex].draw(nodeCount, 1, 0, 0);
    }
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

    // M6-C-2: liveCountDisplay is the read-back compacted count (bounded over a
    // sustained session); numParticles is the conservative dispatch/draw bound.
    ImGui::Text("Particles  : %u live / %u max  (bound %u)", liveCountDisplay,
                kMaxParticles, numParticles);
    ImGui::Checkbox("Show particles", &showParticles);
    if (poolFull) {
      ImGui::TextColored(ImVec4(1.f, 0.4f, 0.3f, 1.f),
                         "particle pool full - drop rejected");
    }

    // --- emit (M5 Task 9: auto-drop) ---
    ImGui::Checkbox("auto emit", &autoEmit);
    ImGui::DragFloat("auto interval (s)", &autoEmitInterval, 0.002f, 0.f, 5.f,
                     "%.3f");
    // M8: continuous stream (overrides auto burst). streamRate particles/s per
    // spoid, swept along motion so fast strokes stay connected (no dots).
    ImGui::Checkbox("stream mode (continuous)", &streamMode);
    ImGui::DragFloat("stream rate (/s)", &streamRate, 20.f, 0.f, 20000.f,
                     "%.0f");
    // Phase 2: paint reservoir drain per emitted particle. 0 = paint never runs
    // out; raise it so each spoid's paintMass (1.0) depletes over time.
    ImGui::DragFloat("mass drain rate", &massDrainRate, 1.0e-5f, 0.f, 1.0e-2f,
                     "%.5f");
    // M8: spawn shape. Ball -> holeRadius drives droplet size; lattice ->
    // amount drives size (fresh droplet exactly at rest density).
    ImGui::Checkbox("spherical spawn (ball)", &sphericalSpawn);

    // --- Phase 2: spoid control mode ---
    int modeI = static_cast<int>(spoidControlMode);
    if (ImGui::RadioButton("keyboard", &modeI, 0)) {
      setSpoidControlMode(SpoidControlMode::Keyboard);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("pendulum", &modeI, 1)) {
      setSpoidControlMode(SpoidControlMode::Pendulum);
    }

    // --- Phase 2: pendulum config (only in pendulum mode) ---
    if (spoidControlMode == SpoidControlMode::Pendulum &&
        !pendulumChains.empty()) {
      PendulumChain& c = pendulumChains[0];
      if (ImGui::CollapsingHeader("Pendulum", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("show chain", &showChain);
        bool rebuild = false;
        rebuild |= ImGui::DragInt("links (n)", &c.numLinks, 0.1f, 1,
                                  static_cast<int>(kMaxChainNodes) - 1);
        rebuild |= ImGui::DragFloat("total length", &c.totalLength, 0.01f, 0.1f,
                                    2.8f, "%.2f");
        rebuild |= ImGui::DragFloat3("pivot", &c.pivot.x, 0.01f);
        rebuild |= ImGui::DragFloat("init theta (rad)", &c.initTheta, 0.01f,
                                    0.f, 3.14f, "%.2f");
        rebuild |= ImGui::DragFloat("init phi (rad)", &c.initPhi, 0.01f, 0.f,
                                    6.28f, "%.2f");
        rebuild |= ImGui::DragFloat("init speed radial", &c.initSpeedRadial,
                                    0.01f, -10.f, 10.f, "%.2f");
        rebuild |=
            ImGui::DragFloat("init speed tangential", &c.initSpeedTangential,
                             0.01f, -10.f, 10.f, "%.2f");
        // finer steps + more decimals: small damping changes are visible.
        ImGui::DragFloat("air damping (/s)", &c.airDamping, 0.0002f, 0.f, 5.f,
                         "%.4f");
        ImGui::DragFloat("joint damping", &c.jointDamping, 0.0002f, 0.f, 1.f,
                         "%.4f");
        ImGui::DragInt("substeps", &c.substeps, 0.2f, 1, 32);
        ImGui::DragInt("constraint iters", &c.iters, 0.1f, 1, 16);
        // per-node mass (mass<=0 => pinned). Resize to numLinks lazily.
        if (static_cast<int>(c.bobMass.size()) != c.numLinks) {
          c.bobMass.assign(std::max(1, c.numLinks), 1.0f);
        }
        for (int i = 0; i < static_cast<int>(c.bobMass.size()); i++) {
          char lbl[32];
          std::snprintf(lbl, sizeof(lbl), "bob %d mass (<=0 pin)", i + 1);
          rebuild |=
              ImGui::DragFloat(lbl, &c.bobMass[i], 0.02f, 0.f, 10.f, "%.2f");
        }
        if (uiOverlay->button("reset pendulum") || rebuild) {
          resetPendulum();
          for (Spoid& s : spoids) s.offsetPhase = s.offsetAngle0;
        }
      }
    }

    // --- Spoids (M5 Task 10) ---
    if (ImGui::CollapsingHeader("Spoids", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::TextWrapped(
          "Keys: IJKL move, U/O height, Space = drop from selected");
      ImGui::Checkbox("show spoids", &showSpoids);

      // Edit mode: a single radio toggles "edit ALL spoids" -- param edits
      // below then apply to every spoid at once (per-spoid radios edit just
      // one).
      if (ImGui::RadioButton("edit ALL spoids", editAllSpoids)) {
        editAllSpoids = true;
      }

      for (int i = 0; i < static_cast<int>(spoids.size()); i++) {
        ImGui::PushID(i);
        ImGui::Checkbox("##sel", &spoids[i].selected);
        ImGui::SameLine();
        if (ImGui::RadioButton("edit",
                               !editAllSpoids && selectedSpoidUi == i)) {
          editAllSpoids = false;
          selectedSpoidUi = i;
        }
        ImGui::SameLine();
        ImGui::Text("#%d (%.2f, %.2f, %.2f)", i, spoids[i].pos.x,
                    spoids[i].pos.y, spoids[i].pos.z);
        ImGui::PopID();
      }

      if (uiOverlay->button("+ Add spoid") && spoids.size() < kMaxSpoids) {
        // Inherit the currently-edited spoid's params (amount, holeRadius, ...)
        // so a freshly added spoid matches your latest tuning; only the colour
        // is assigned fresh from the palette.
        Spoid s = (selectedSpoidUi >= 0 &&
                   selectedSpoidUi < static_cast<int>(spoids.size()))
                      ? spoids[selectedSpoidUi]
                      : Spoid{};
        s.color = spoidPalette(static_cast<int>(spoids.size()));
        s.selected = true;
        spoids.push_back(s);
        // Re-spread all spoids evenly on a circle in the X-Z plane (+ reset y).
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
        // In "edit ALL" mode the widgets show this spoid's values but any
        // change is propagated to every spoid (so you can retune all of them
        // together).
        ImGui::Text(editAllSpoids ? "Editing ALL spoids (via #%d)"
                                  : "Editing spoid #%d",
                    selectedSpoidUi);
        // applyAll(field): copy the just-edited field to all spoids when in
        // edit-ALL mode. Per-field so it doesn't clobber the other params.
        if (ImGui::DragFloat("hole radius (ball)", &s.holeRadius, 0.002f, 0.02f,
                             0.5f, "%.3f") &&
            editAllSpoids)
          for (auto& o : spoids) o.holeRadius = s.holeRadius;
        if (ImGui::DragFloat("emission vel", &s.emissionVelocity, 0.02f, 0.f,
                             8.f, "%.2f") &&
            editAllSpoids)
          for (auto& o : spoids) o.emissionVelocity = s.emissionVelocity;
        if (ImGui::DragInt("amount (burst)", &s.amount, 1.f, 0, 2000) &&
            editAllSpoids)
          for (auto& o : spoids) o.amount = s.amount;
        if (ImGui::DragFloat("concentration", &s.concentration, 0.005f, 0.f,
                             1.f, "%.3f") &&
            editAllSpoids)
          for (auto& o : spoids) o.concentration = s.concentration;
        if (ImGui::ColorEdit3("color", &s.color.x) && editAllSpoids)
          for (auto& o : spoids) o.color = s.color;
        // Phase 2: per-spoid rotary offset (r, angle0, omega) + paint
        // reservoir.
        if (spoidControlMode == SpoidControlMode::Pendulum) {
          if (ImGui::DragFloat("offset r", &s.offsetR, 0.005f, 0.f, 1.f,
                               "%.3f") &&
              editAllSpoids)
            for (auto& o : spoids) o.offsetR = s.offsetR;
          if (ImGui::DragFloat("offset angle0 (rad)", &s.offsetAngle0, 0.01f,
                               0.f, 6.28f, "%.2f") &&
              editAllSpoids)
            for (auto& o : spoids) o.offsetAngle0 = s.offsetAngle0;
          if (ImGui::DragFloat("offset omega (rad/s)", &s.offsetOmega, 0.02f,
                               -20.f, 20.f, "%.2f") &&
              editAllSpoids)
            for (auto& o : spoids) o.offsetOmega = s.offsetOmega;
          ImGui::Text("paint mass : %.3f", s.paintMass);
          if (uiOverlay->button("refill paint")) {
            if (editAllSpoids)
              for (auto& o : spoids) o.paintMass = 1.f;
            else
              s.paintMass = 1.f;
          }
        }
      }
    }

    // --- sim params (M3) ---
    ImGui::DragFloat("gravity (+Y)", &gravity, 0.05f, 0.f, 30.f, "%.2f");
    ImGui::Checkbox("fixed dt (1/120)", &useFixedDt);
    ImGui::DragFloat("seed jitter xz", &seedJitterXZ, 0.01f, 0.f, 5.f, "%.2f");
    // Phase 2: world scale. Multiplies the canvas + sim domain + grid (NOT the
    // particle size); applied on Restart. The label flags a pending change.
    ImGui::DragFloat("world scale", &worldScale, 0.05f, 1.0f, kMaxWorldScale,
                     "%.2f");
    if (worldScale != appliedWorldScale) {
      ImGui::SameLine();
      ImGui::TextDisabled("(Restart to apply)");
    }
    if (uiOverlay->button("Restart")) {
      restartRequested = true;
    }

    // --- PBF solver (M4) ---
    if (ImGui::CollapsingHeader("PBF solver", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Text("rho0 (rest) : %.1f", rho0);
      ImGui::DragInt("substeps", &substeps, 1.f, 1, 16);
      ImGui::DragInt("solverIters", &solverIters, 1.f, 1, 6);
      ImGui::DragFloat("solver relax", &solverRelax, 0.005f, 0.05f, 1.f,
                       "%.3f");
      // epsCFM is the SOFTNESS knob: high (~1e5) = soft/stable, low (~1e3) =
      // stiff/crisp-but-poppy. Logarithmic so 1..1e6 is reachable.
      ImGui::DragFloat("epsCFM (soft<-)", &epsCFM, 100.f, 1.f, 1.0e6f, "%.0f",
                       ImGuiSliderFlags_Logarithmic);
      // M8 cohesion/crown dials
      ImGui::DragFloat("cohesion floor", &cohesionFloor, 0.01f, 0.f, 1.f,
                       "%.3f");
      ImGui::DragFloat("dp clamp (xh, 0=off)", &dpClampFactor, 0.005f, 0.f,
                       0.5f, "%.3f");
      ImGui::DragFloat("scorrK", &scorrK, 1.0e-6f, 0.f, 0.1f, "%.6f",
                       ImGuiSliderFlags_Logarithmic);
      ImGui::DragFloat("scorrDq/h", &scorrDqRatio, 0.002f, 0.05f, 0.5f, "%.3f");
      ImGui::DragFloat("xsphC", &xsphC, 0.005f, 0.f, 1.f, "%.3f");
      ImGui::DragFloat("vel damping (/s)", &velDamp, 0.02f, 0.f, 20.f, "%.2f");
      ImGui::DragFloat("vel clamp (CFL, 0=off)", &velClampFactor, 0.002f, 0.f,
                       0.5f, "%.3f");
      ImGui::Checkbox("color by density", &colorByDensity);
    }

    // --- Canvas deposit (M6) ---
    if (ImGui::CollapsingHeader("Canvas deposit",
                                ImGuiTreeNodeFlags_DefaultOpen)) {
      // Per-frame stamp alpha = concentration * depositStrength. Keep SMALL so
      // overlapping colours mix smoothly instead of flickering (compaction
      // reorders particles -> order-dependent alpha-over oscillates at high
      // alpha) and so concentration/blending stays visible (no saturation).
      ImGui::DragFloat("deposit strength", &depositStrength, 0.001f, 0.f, 0.5f,
                       "%.3f");
      ImGui::DragFloat("deposit radius (world)", &depositRadius, 0.001f, 0.005f,
                       0.06f, "%.3f");
      ImGui::DragFloat("dry rate (/s)", &dryRate, 0.05f, 0.f, 10.f, "%.2f");
      ImGui::DragFloat("deposit height", &depositHeight, 0.002f, 0.01f, 0.5f,
                       "%.3f");
      // Drying freezes near-floor motion (paint setting); 0 = off, 1 = full.
      ImGui::DragFloat("dry settle (freeze)", &drySettle, 0.01f, 0.f, 1.f,
                       "%.2f");
      // Live-particle sprite size scale.
      ImGui::DragFloat("particle size", &pointScale, 0.01f, 0.1f, 2.f, "%.2f");

      // Phase 2: smoothly animate to/from an overhead top-down view.
      if (uiOverlay->button(cameraAnim.locked || cameraAnim.active
                                ? "Free camera"
                                : "Top view")) {
        startTopViewAnim();
      }

      // M7 Task 14: export the canvas to build/paint_<timestamp>.png.
      if (uiOverlay->button("Save PNG")) {
        saveRequested = true;  // handled next render() (does device.waitIdle)
      }
      if (!saveStatus.empty()) {
        ImGui::TextWrapped("%s", saveStatus.c_str());
      }
    }

    if (uiOverlay->button("Save (no-op)")) {
      std::cout << "[paint_splatter] Save button pressed (no-op)\n";
    }
  }
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()
