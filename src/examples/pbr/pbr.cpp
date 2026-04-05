#include "pbr.hpp"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtx/matrix_query.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

#include <array>
#include <cstring>
#include <limits>
#include <memory>

namespace vge {

VgeExample::VgeExample() : VgeBase() { title = "PBR Example"; }
VgeExample::~VgeExample() {}

void VgeExample::setupCommandLineParser(CLI::App& app) {}

void VgeExample::setOptions(const std::optional<Options>& opts) {
  if (opts.has_value()) {
    this->opts = opts.value();
    cameraController.moveSpeed = this->opts.moveSpeed;
  }
}

void VgeExample::initVulkan() {
  cameraController.moveSpeed = opts.moveSpeed;
  if (glm::isIdentity(opts.cameraView, 1e-6f)) {
    camera.setViewTarget(glm::vec3{0.f, -10.f, -20.f}, glm::vec3{0.f, 0.f, 0.f});
  } else {
    camera.setViewMatrix(opts.cameraView);
  }
  camera.setPerspectiveProjection(
      glm::radians(60.f),
      static_cast<float>(width) / static_cast<float>(height), 0.1f, 256.f);
  VgeBase::initVulkan();
}

void VgeExample::getEnabledExtensions() {}

void VgeExample::getEnabledFeatures() {
  enabledFeatures.samplerAnisotropy = physicalDevice.getFeatures().samplerAnisotropy;
  enabledFeatures.fillModeNonSolid  = physicalDevice.getFeatures().fillModeNonSolid;
}

void VgeExample::prepare() {
  VgeBase::prepare();
  loadAssets();
  setupDynamicUbo();
  prepareOffScreenFrameBuffer();
  prepareUniformBuffers();
  setupDescriptors();
  preparePipelines();
  prepared = true;
}

void VgeExample::render() {
  if (!prepared) return;
  draw();
}

void VgeExample::viewChanged() {}

void VgeExample::onUpdateUIOverlay() {
  if (uiOverlay->header("Settings")) {
    if (ImGui::TreeNodeEx("Immediate", ImGuiTreeNodeFlags_DefaultOpen)) {
      for (int i = 0; i < opts.numTargets; i++) {
        std::string caption = "debugDisplayTarget: " + std::to_string(i);
        uiOverlay->radioButton(caption.c_str(), &opts.debugDisplayTarget, i);
      }
      ImGui::DragFloat("Far Clamping", &opts.farClamp, 0.1f,
                       camera.getNearPlane(), camera.getFarPlane(), "%.1f");
      ImGui::Separator();
      ImGui::Checkbox("Show Debug Views", &opts.showDebugViews);
      ImGui::Checkbox("Use Spheres", &opts.useSpheres);
      if (opts.useSpheres) {
        uiOverlay->colorPicker("Sphere Albedo", opts.sphereAlbedo.data());
      }
      ImGui::Separator();
      ImGui::Checkbox("Directional Light", &opts.useDirectionalLight);
      if (opts.useDirectionalLight) {
        ImGui::DragFloat3("Dir Light Dir", opts.dirLightDir.data(), 0.01f, -1.f, 1.f, "%.2f");
      } else {
        ImGui::Checkbox("Animate Lights", &opts.animateLights);
        ImGui::DragFloat("Rotation Speed", &opts.rotationSpeed, 0.05f, 0.0f, 10.f, "%.2f");
        ImGui::DragFloat("Orbit Radius",   &opts.orbitRadius,   0.1f,  0.5f, 30.f, "%.1f");
        ImGui::DragFloat("Orbit Height",   &opts.orbitHeight,   0.1f, -20.f, 0.f,  "%.1f");
        ImGui::DragInt("Num Lights", &opts.numLights, 1, 1, MAX_LIGHTS);
        ImGui::DragFloat("Sprite Size", &opts.spriteSize, 0.01f, 0.05f, 2.f, "%.2f");
      }
      ImGui::DragFloat("Light Intensity", &opts.lightIntensity, 0.5f, 0.0f, 100.f, "%.1f");
      ImGui::TreePop();
    }
  }
}

std::unique_ptr<vgeu::VgeuImage> VgeExample::createDummyTexture(
    std::array<uint8_t, 4> rgba) {
  // Upload a 1x1 RGBA pixel into a shader-readable image.
  vgeu::VgeuBuffer staging(
      globalAllocator->getAllocator(), 4, 1,
      vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);
  std::memcpy(staging.getMappedData(), rgba.data(), 4);

  auto img = std::make_unique<vgeu::VgeuImage>(
      device, globalAllocator->getAllocator(),
      vk::Format::eR8G8B8A8Unorm, vk::Extent2D{1, 1},
      vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
      vk::ImageLayout::eUndefined, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
      vk::ImageAspectFlagBits::eColor, 1);

  vk::BufferImageCopy region(
      0, 0, 0,
      vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      vk::Offset3D{0, 0, 0}, vk::Extent3D{1, 1, 1});
  vgeu::oneTimeSubmit(device, commandPool, queue,
      [&](const vk::raii::CommandBuffer& cmd) {
        vgeu::setImageLayout(cmd, img->getImage(), vk::Format::eR8G8B8A8Unorm,
            0, 1, vk::ImageLayout::eUndefined,
            vk::ImageLayout::eTransferDstOptimal);
        cmd.copyBufferToImage(staging.getBuffer(), img->getImage(),
            vk::ImageLayout::eTransferDstOptimal, region);
        vgeu::setImageLayout(cmd, img->getImage(), vk::Format::eR8G8B8A8Unorm,
            0, 1, vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eShaderReadOnlyOptimal);
      });
  return img;
}

void VgeExample::loadAssets() {
  vgeu::FileLoadingFlags glTFLoadingFlags =
      vgeu::FileLoadingFlagBits::kPreMultiplyVertexColors |
      vgeu::FileLoadingFlagBits::kPreTransformVertices |
      vgeu::FileLoadingFlagBits::kFlipY;

  std::shared_ptr<vgeu::glTF::Model> floor = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool, MAX_CONCURRENT_FRAMES);
  floor->descriptorBindingFlags =
      vgeu::DescriptorBindingFlagBits::kImageBaseColor |
      vgeu::DescriptorBindingFlagBits::kImageNormalMap |
      vgeu::DescriptorBindingFlagBits::kImageMetallicRoughness |
      vgeu::DescriptorBindingFlagBits::kImageEmissive;
  floor->loadFromFile(getAssetsPath() + "/models/metal_plate/metal_plate_1k.gltf", glTFLoadingFlags);
  {
    ModelInstance inst{};
    inst.model = floor;
    inst.name  = "floor";
    addModelInstance(std::move(inst));
  }

  std::shared_ptr<vgeu::glTF::Model> damagedHelmet = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool, MAX_CONCURRENT_FRAMES);
  damagedHelmet->descriptorBindingFlags =
      vgeu::DescriptorBindingFlagBits::kImageBaseColor |
      vgeu::DescriptorBindingFlagBits::kImageNormalMap |
      vgeu::DescriptorBindingFlagBits::kImageMetallicRoughness |
      vgeu::DescriptorBindingFlagBits::kImageEmissive;
  damagedHelmet->loadFromFile(
      getAssetsPath() + "/models/DamagedHelmet/glTF/DamagedHelmet.gltf", glTFLoadingFlags);
  for (int i = 0; i < opts.modelNumZ; i++) {
    for (int j = 0; j < opts.modelNumX; j++) {
      ModelInstance inst{};
      inst.model = damagedHelmet;
      inst.name  = "damagedHelmet_" + std::to_string(i) + "-" + std::to_string(j);
      addModelInstance(std::move(inst));
    }
  }

  // Sphere model (geometry only; textures are provided via dummy descriptor set)
  std::shared_ptr<vgeu::glTF::Model> sphere = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool, MAX_CONCURRENT_FRAMES);
  sphere->descriptorBindingFlags =
      vgeu::DescriptorBindingFlagBits::kImageBaseColor |
      vgeu::DescriptorBindingFlagBits::kImageNormalMap |
      vgeu::DescriptorBindingFlagBits::kImageMetallicRoughness |
      vgeu::DescriptorBindingFlagBits::kImageEmissive;
  sphere->loadFromFile(getAssetsPath() + "/models/sphere/untitled.gltf", glTFLoadingFlags);
  for (int i = 0; i < opts.modelNumZ; i++) {
    for (int j = 0; j < opts.modelNumX; j++) {
      ModelInstance inst{};
      inst.model = sphere;
      inst.name  = "sphere_" + std::to_string(i) + "-" + std::to_string(j);
      inst.sceneMode = ModelInstance::SceneMode::kSphereOnly;
      addModelInstance(std::move(inst));
    }
  }

  // 1x1 dummy textures for sphere draw calls:
  //   albedo  = white     (overridden by modelColor via modelColor.a=1.0)
  //   normal  = flat +Z   (128,128,255 -> tangent-space (0,0,1) -> passes geometric normal through)
  //   metrough = neutral  (g=128->roughness~0.5; overridden by pbrOverride)
  //   emissive = black
  sphereDummyAlbedo   = createDummyTexture({255, 255, 255, 255});
  sphereDummyNormal   = createDummyTexture({128, 128, 255, 255});
  sphereDummyMetRough = createDummyTexture({0,   128,   0, 255});
  sphereDummyEmissive = createDummyTexture({0,     0,   0, 255});
}

void VgeExample::setupDynamicUbo() {
  glm::vec3 up{0.f, -1.f, 0.f};
  glm::vec3 right{1.f, 0.f, 0.f};
  dynamicUbo.resize(modelInstances.size());
  {
    size_t idx = findInstances("floor")[0];
    dynamicUbo[idx].modelMatrix = glm::scale(glm::mat4{1.f}, glm::vec3{10.f, 10.f, 10.f});
    dynamicUbo[idx].modelColor  = glm::vec4{0.f, 0.f, 0.f, 0.f};
  }
  const float helmetScale = 1.0f;
  for (int i = 0; i < opts.modelNumZ; i++) {
    for (int j = 0; j < opts.modelNumX; j++) {
      size_t idx = findInstances(
          "damagedHelmet_" + std::to_string(i) + "-" + std::to_string(j))[0];
      const float x = -((opts.modelNumX - 1) * opts.spacingX * 0.5f) + j * opts.spacingX;
      const float z = -((opts.modelNumZ - 1) * opts.spacingZ * 0.5f) + i * opts.spacingZ;
      const float y = -4.f;
      dynamicUbo[idx].modelMatrix = glm::translate(glm::mat4{1.f}, glm::vec3{x, y, z});
      dynamicUbo[idx].modelMatrix = glm::rotate(dynamicUbo[idx].modelMatrix, glm::radians(90.f), up);
      dynamicUbo[idx].modelMatrix = glm::rotate(dynamicUbo[idx].modelMatrix, glm::radians(-90.f), right);
      dynamicUbo[idx].modelMatrix = glm::scale(dynamicUbo[idx].modelMatrix,
                                               glm::vec3{helmetScale, helmetScale, helmetScale});
      dynamicUbo[idx].modelColor  = glm::vec4{0.f, 0.f, 0.f, 0.f};
    }
  }
  // Sphere instances: row i (Z axis) = metallic 0→1, col j (X axis) = roughness 0→1
  const float sphereScale = 1.5f;
  for (int i = 0; i < opts.modelNumZ; i++) {
    for (int j = 0; j < opts.modelNumX; j++) {
      size_t idx = findInstances(
          "sphere_" + std::to_string(i) + "-" + std::to_string(j))[0];
      const float x = -((opts.modelNumX - 1) * opts.spacingX * 0.5f) + j * opts.spacingX;
      const float z = -((opts.modelNumZ - 1) * opts.spacingZ * 0.5f) + i * opts.spacingZ;
      const float y = -4.f;
      dynamicUbo[idx].modelMatrix = glm::translate(glm::mat4{1.f}, glm::vec3{x, y, z});
      dynamicUbo[idx].modelMatrix = glm::scale(dynamicUbo[idx].modelMatrix,
                                               glm::vec3{sphereScale, sphereScale, sphereScale});
      float metallic  = (opts.modelNumZ <= 1) ? 0.0f
                      : static_cast<float>(i) / static_cast<float>(opts.modelNumZ - 1);
      float roughness = (opts.modelNumX <= 1) ? 0.0f
                      : static_cast<float>(j) / static_cast<float>(opts.modelNumX - 1);
      dynamicUbo[idx].pbrOverride = glm::vec4(metallic, roughness, 1.0f, 0.0f);
      dynamicUbo[idx].modelColor  = glm::vec4(opts.sphereAlbedo[0], opts.sphereAlbedo[1],
                                               opts.sphereAlbedo[2], 1.0f);
    }
  }
}

std::unique_ptr<vgeu::VgeuImage> VgeExample::createAttachment(
    vk::Format format, vk::ImageUsageFlags usage) {
  vk::ImageAspectFlags aspectMask{};
  if (usage & vk::ImageUsageFlagBits::eColorAttachment)
    aspectMask = vk::ImageAspectFlagBits::eColor;
  else if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment)
    aspectMask = vk::ImageAspectFlagBits::eDepth;
  assert(aspectMask != vk::ImageAspectFlagBits::eNone);
  usage = usage | vk::ImageUsageFlagBits::eSampled;
  return std::make_unique<vgeu::VgeuImage>(
      device, globalAllocator->getAllocator(), format,
      vk::Extent2D{offScreenFrameBuf.width, offScreenFrameBuf.height},
      vk::ImageTiling::eOptimal, usage, vk::ImageLayout::eUndefined,
      VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
      aspectMask, 1);
}

void VgeExample::prepareOffScreenFrameBuffer() {
  offScreenFrameBuf.width  = 2048;
  offScreenFrameBuf.height = 2048;
  offScreenFrameBuf.isFirstFrame.resize(MAX_CONCURRENT_FRAMES, true);

  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    offScreenFrameBuf.position.push_back(createAttachment(
        vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment));
    offScreenFrameBuf.normal.push_back(createAttachment(
        vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment));
    offScreenFrameBuf.albedo.push_back(createAttachment(
        vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment));
    offScreenFrameBuf.arm.push_back(createAttachment(
        vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment));
    offScreenFrameBuf.emissive.push_back(createAttachment(
        vk::Format::eR16G16B16A16Sfloat, vk::ImageUsageFlagBits::eColorAttachment));
    offScreenFrameBuf.depth.push_back(createAttachment(
        depthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment));
  }

  // Render pass with 5 color attachments + depth
  std::vector<vk::AttachmentDescription> attachmentDescs;
  for (uint32_t i = 0; i < offScreenFrameBuf.numAttachments; i++) {
    attachmentDescs.emplace_back(
        vk::AttachmentDescriptionFlags(), vk::Format::eUndefined,
        vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);
  }
  attachmentDescs[0].format = offScreenFrameBuf.position[0]->getFormat();
  attachmentDescs[1].format = offScreenFrameBuf.normal[0]->getFormat();
  attachmentDescs[2].format = offScreenFrameBuf.albedo[0]->getFormat();
  attachmentDescs[3].format = offScreenFrameBuf.arm[0]->getFormat();
  attachmentDescs[4].format = offScreenFrameBuf.emissive[0]->getFormat();
  attachmentDescs[5].format = offScreenFrameBuf.depth[0]->getFormat();

  std::vector<vk::AttachmentReference> colorRefs;
  for (uint32_t i = 0; i < 5; i++)
    colorRefs.emplace_back(i, vk::ImageLayout::eColorAttachmentOptimal);
  vk::AttachmentReference depthRef(5, vk::ImageLayout::eDepthStencilAttachmentOptimal);
  vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, colorRefs, {}, &depthRef);

  std::vector<vk::SubpassDependency> deps;
  deps.emplace_back(VK_SUBPASS_EXTERNAL, 0u,
      vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests,
      vk::PipelineStageFlagBits::eEarlyFragmentTests | vk::PipelineStageFlagBits::eLateFragmentTests,
      vk::AccessFlagBits::eDepthStencilAttachmentWrite,
      vk::AccessFlagBits::eDepthStencilAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentRead,
      vk::DependencyFlags());
  deps.emplace_back(VK_SUBPASS_EXTERNAL, 0u,
      vk::PipelineStageFlagBits::eBottomOfPipe,
      vk::PipelineStageFlagBits::eColorAttachmentOutput,
      vk::AccessFlagBits::eMemoryRead,
      vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead,
      vk::DependencyFlags());
  deps.emplace_back(0u, VK_SUBPASS_EXTERNAL,
      vk::PipelineStageFlagBits::eColorAttachmentOutput,
      vk::PipelineStageFlagBits::eBottomOfPipe,
      vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eColorAttachmentRead,
      vk::AccessFlagBits::eMemoryRead,
      vk::DependencyFlags());

  offScreenFrameBuf.renderPass = vk::raii::RenderPass(
      device, vk::RenderPassCreateInfo({}, attachmentDescs, subpass, deps));

  offScreenFrameBuf.frameBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    std::array<vk::ImageView, 6> attachments{
        *offScreenFrameBuf.position[i]->getImageView(),
        *offScreenFrameBuf.normal[i]->getImageView(),
        *offScreenFrameBuf.albedo[i]->getImageView(),
        *offScreenFrameBuf.arm[i]->getImageView(),
        *offScreenFrameBuf.emissive[i]->getImageView(),
        *offScreenFrameBuf.depth[i]->getImageView()};
    offScreenFrameBuf.frameBuffers.push_back(vk::raii::Framebuffer(device,
        vk::FramebufferCreateInfo({}, *offScreenFrameBuf.renderPass, attachments,
                                   offScreenFrameBuf.width, offScreenFrameBuf.height, 1)));
  }

  vk::SamplerCreateInfo samplerCI(
      {}, vk::Filter::eNearest, vk::Filter::eNearest,
      vk::SamplerMipmapMode::eLinear,
      vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge,
      0.f, true, 1.f, false, vk::CompareOp::eNever, 0.f, 1.f,
      vk::BorderColor::eFloatOpaqueWhite);
  colorSampler = vk::raii::Sampler(device, samplerCI);
}

void VgeExample::prepareUniformBuffers() {
  alignedSizeDynamicUboElt = vgeu::padBufferSize(physicalDevice, sizeof(DynamicUboElt), true);
  uniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    auto dynamic = std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), alignedSizeDynamicUboElt, dynamicUbo.size(),
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT);
    for (size_t j = 0; j < dynamicUbo.size(); j++) {
      std::memcpy(static_cast<char*>(dynamic->getMappedData()) + j * alignedSizeDynamicUboElt,
                  &dynamicUbo[j], alignedSizeDynamicUboElt);
    }

    auto offScreen = std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(UniformDataOffscreen), 1,
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT);
    std::memcpy(offScreen->getMappedData(), &uniformDataOffscreen, sizeof(UniformDataOffscreen));

    auto composition = std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(UniformDataComposition), 1,
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT);
    std::memcpy(composition->getMappedData(), &uniformDataComposition, sizeof(UniformDataComposition));

    uniformBuffers.push_back({std::move(dynamic), std::move(offScreen), std::move(composition)});
  }
}

void VgeExample::setupDescriptors() {
  // Descriptor pool
  std::vector<vk::DescriptorPoolSize> poolSizes;
  poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer,
      MAX_CONCURRENT_FRAMES /*offscreen*/ +
      MAX_CONCURRENT_FRAMES /*composition*/ +
      MAX_CONCURRENT_FRAMES /*sprite->compositionUBO*/);
  poolSizes.emplace_back(vk::DescriptorType::eUniformBufferDynamic,
      MAX_CONCURRENT_FRAMES /*dynamic*/);
  poolSizes.emplace_back(vk::DescriptorType::eCombinedImageSampler,
      static_cast<uint32_t>(MAX_CONCURRENT_FRAMES * offScreenFrameBuf.numAttachments)
          + 4u /*sphere dummy: 4 combined image samplers*/);

  vk::DescriptorPoolCreateInfo poolCI(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      MAX_CONCURRENT_FRAMES /*composition*/ +
      MAX_CONCURRENT_FRAMES * 2 /*offscreen + dynamic*/ +
      MAX_CONCURRENT_FRAMES /*sprite*/ +
      1u /*sphere dummy*/,
      poolSizes);
  descriptorPool = vk::raii::DescriptorPool(device, poolCI);

  // Composition descriptor set layout: bindings 0-5 = G-buffer samplers, binding 6 = UBO
  {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (uint32_t b = 0; b < 6; b++)
      bindings.emplace_back(b, vk::DescriptorType::eCombinedImageSampler, 1,
                            vk::ShaderStageFlagBits::eFragment);
    bindings.emplace_back(6, vk::DescriptorType::eUniformBuffer, 1,
                          vk::ShaderStageFlagBits::eFragment);
    compositionDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo({}, bindings));

    pipelineLayoutComposition = vk::raii::PipelineLayout(
        device, vk::PipelineLayoutCreateInfo({}, *compositionDescriptorSetLayout));
  }

  // Offscreen UBO descriptor set layout: binding 0 = UniformDataOffscreen
  {
    vk::DescriptorSetLayoutBinding binding(0, vk::DescriptorType::eUniformBuffer, 1,
                                           vk::ShaderStageFlagBits::eVertex);
    offScreenUboDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo({}, binding));
  }

  // Dynamic UBO descriptor set layout: binding 0 = dynamic
  {
    vk::DescriptorSetLayoutBinding binding(0, vk::DescriptorType::eUniformBufferDynamic, 1,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
    dynamicUboDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo({}, binding));
  }

  // Sphere image descriptor set layout: 4 CombinedImageSampler bindings (albedo, normal, metRough, emissive)
  // Used for both the pipeline layout set 2 and the sphere dummy descriptor set allocation.
  // Must be "identically defined" to floor/helmet models' descriptorSetLayoutImage (same 4 eFragment CIS).
  {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (uint32_t b = 0; b < 4; b++)
      bindings.emplace_back(b, vk::DescriptorType::eCombinedImageSampler, 1,
                            vk::ShaderStageFlagBits::eFragment);
    sphereImageSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo({}, bindings));
  }

  // Offscreen pipeline layout: set0=offscreenUBO, set1=dynamicUBO, set2=modelImage, set3=modelUBO
  {
    std::vector<vk::DescriptorSetLayout> setLayouts = {
        *offScreenUboDescriptorSetLayout,
        *dynamicUboDescriptorSetLayout,
        *sphereImageSetLayout,
        *modelInstances[0].model->descriptorSetLayoutUbo};
    pipelineLayoutOffScreen = vk::raii::PipelineLayout(
        device, vk::PipelineLayoutCreateInfo({}, setLayouts));
  }

  // Sprite light descriptor set layout: binding 0 = composition UBO (lights)
  {
    vk::DescriptorSetLayoutBinding binding(0, vk::DescriptorType::eUniformBuffer, 1,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
    spriteLightDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo({}, binding));
  }

  // Sprite pipeline layout: set0=offscreenUBO, set1=spriteLightLayout, push constant
  {
    std::vector<vk::DescriptorSetLayout> setLayouts = {
        *offScreenUboDescriptorSetLayout,
        *spriteLightDescriptorSetLayout};
    vk::PushConstantRange pcRange(
        vk::ShaderStageFlagBits::eVertex, 0, sizeof(SpritePushConstants));
    pipelineLayoutSprite = vk::raii::PipelineLayout(
        device, vk::PipelineLayoutCreateInfo({}, setLayouts, pcRange));
  }

  // Allocate and write composition descriptor sets
  {
    vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool, *compositionDescriptorSetLayout);
    descriptorSets.composition.reserve(MAX_CONCURRENT_FRAMES);
    for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++)
      descriptorSets.composition.push_back(
          std::move(vk::raii::DescriptorSets(device, allocInfo).front()));

    for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      auto bufInfo   = uniformBuffers[i].composition->descriptorInfo();
      auto posInfo   = offScreenFrameBuf.position[i]->descriptorImageInfo(
          *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);
      auto normInfo  = offScreenFrameBuf.normal[i]->descriptorImageInfo(
          *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);
      auto albInfo   = offScreenFrameBuf.albedo[i]->descriptorImageInfo(
          *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);
      auto armInfo   = offScreenFrameBuf.arm[i]->descriptorImageInfo(
          *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);
      auto emissInfo = offScreenFrameBuf.emissive[i]->descriptorImageInfo(
          *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);
      auto depthInfo = offScreenFrameBuf.depth[i]->descriptorImageInfo(
          *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);

      std::vector<vk::WriteDescriptorSet> writes;
      writes.emplace_back(*descriptorSets.composition[i], 0, 0,
          vk::DescriptorType::eCombinedImageSampler, posInfo,   nullptr);
      writes.emplace_back(*descriptorSets.composition[i], 1, 0,
          vk::DescriptorType::eCombinedImageSampler, normInfo,  nullptr);
      writes.emplace_back(*descriptorSets.composition[i], 2, 0,
          vk::DescriptorType::eCombinedImageSampler, albInfo,   nullptr);
      writes.emplace_back(*descriptorSets.composition[i], 3, 0,
          vk::DescriptorType::eCombinedImageSampler, armInfo,   nullptr);
      writes.emplace_back(*descriptorSets.composition[i], 4, 0,
          vk::DescriptorType::eCombinedImageSampler, emissInfo, nullptr);
      writes.emplace_back(*descriptorSets.composition[i], 5, 0,
          vk::DescriptorType::eCombinedImageSampler, depthInfo, nullptr);
      writes.emplace_back(*descriptorSets.composition[i], 6, 0,
          vk::DescriptorType::eUniformBuffer, nullptr, bufInfo);
      device.updateDescriptorSets(writes, nullptr);
    }
  }

  // Allocate and write offscreen UBO descriptor sets
  {
    vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool, *offScreenUboDescriptorSetLayout);
    descriptorSets.offScreenUboDescriptorSets.reserve(MAX_CONCURRENT_FRAMES);
    std::vector<vk::DescriptorBufferInfo> bufInfos;
    std::vector<vk::WriteDescriptorSet> writes;
    for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      descriptorSets.offScreenUboDescriptorSets.push_back(
          std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
      bufInfos.push_back(uniformBuffers[i].offScreen->descriptorInfo());
      writes.emplace_back(*descriptorSets.offScreenUboDescriptorSets[i], 0, 0,
          vk::DescriptorType::eUniformBuffer, nullptr, bufInfos.back());
    }
    device.updateDescriptorSets(writes, nullptr);
  }

  // Allocate and write dynamic UBO descriptor sets
  {
    vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool, *dynamicUboDescriptorSetLayout);
    descriptorSets.dynamicUboDescriptorSets.reserve(MAX_CONCURRENT_FRAMES);
    std::vector<vk::DescriptorBufferInfo> bufInfos;
    std::vector<vk::WriteDescriptorSet> writes;
    for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      descriptorSets.dynamicUboDescriptorSets.push_back(
          std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
      bufInfos.push_back(uniformBuffers[i].dynamic->descriptorInfo(alignedSizeDynamicUboElt, 0));
      writes.emplace_back(*descriptorSets.dynamicUboDescriptorSets[i], 0, 0,
          vk::DescriptorType::eUniformBufferDynamic, nullptr, bufInfos.back());
    }
    device.updateDescriptorSets(writes, nullptr);
  }

  // Allocate and write sprite descriptor sets (binding to composition UBO buffer)
  {
    vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool, *spriteLightDescriptorSetLayout);
    descriptorSets.sprite.reserve(MAX_CONCURRENT_FRAMES);
    std::vector<vk::DescriptorBufferInfo> bufInfos;
    std::vector<vk::WriteDescriptorSet> writes;
    for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      descriptorSets.sprite.push_back(
          std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
      bufInfos.push_back(uniformBuffers[i].composition->descriptorInfo());
      writes.emplace_back(*descriptorSets.sprite[i], 0, 0,
          vk::DescriptorType::eUniformBuffer, nullptr, bufInfos.back());
    }
    device.updateDescriptorSets(writes, nullptr);
  }

  // Sphere dummy descriptor set (set 2): 4 combined image samplers bound to 1x1 dummy textures.
  // Uses same layout as glTF model image descriptors (4 bindings matching descriptorBindingFlags).
  // Bound manually before sphere draws; kBindImages NOT set so model::draw() doesn't override it.
  {
    vk::DescriptorSetAllocateInfo allocInfo(
        *descriptorPool, *sphereImageSetLayout);
    sphereDummyDescriptorSet = std::move(
        vk::raii::DescriptorSets(device, allocInfo).front());

    auto albInfo   = sphereDummyAlbedo->descriptorImageInfo(
        *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);
    auto normInfo  = sphereDummyNormal->descriptorImageInfo(
        *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);
    auto mrInfo    = sphereDummyMetRough->descriptorImageInfo(
        *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);
    auto emissInfo = sphereDummyEmissive->descriptorImageInfo(
        *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);

    std::vector<vk::WriteDescriptorSet> writes;
    writes.emplace_back(*sphereDummyDescriptorSet, 0, 0,
        vk::DescriptorType::eCombinedImageSampler, albInfo,   nullptr);
    writes.emplace_back(*sphereDummyDescriptorSet, 1, 0,
        vk::DescriptorType::eCombinedImageSampler, normInfo,  nullptr);
    writes.emplace_back(*sphereDummyDescriptorSet, 2, 0,
        vk::DescriptorType::eCombinedImageSampler, mrInfo,    nullptr);
    writes.emplace_back(*sphereDummyDescriptorSet, 3, 0,
        vk::DescriptorType::eCombinedImageSampler, emissInfo, nullptr);
    device.updateDescriptorSets(writes, nullptr);
  }
}

void VgeExample::preparePipelines() {
  vk::PipelineInputAssemblyStateCreateInfo inputAssemblySCI(
      {}, vk::PrimitiveTopology::eTriangleList);
  vk::PipelineRasterizationStateCreateInfo rasterizationSCI(
      {}, false, false, vk::PolygonMode::eFill,
      vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise,
      false, 0.f, 0.f, 0.f, 1.f);
  vk::PipelineColorBlendAttachmentState blendAttachment(
      false, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
      vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
  vk::PipelineColorBlendStateCreateInfo colorBlendSCI(
      {}, false, vk::LogicOp::eNoOp, blendAttachment, {{1.f, 1.f, 1.f, 1.f}});
  vk::StencilOpState stencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep,
                                    vk::StencilOp::eKeep, vk::CompareOp::eAlways);
  vk::PipelineDepthStencilStateCreateInfo depthStencilSCI(
      {}, true, true, vk::CompareOp::eLessOrEqual, false, false,
      stencilOpState, stencilOpState);
  vk::PipelineViewportStateCreateInfo viewportSCI({}, 1, nullptr, 1, nullptr);
  vk::PipelineMultisampleStateCreateInfo multisampleSCI({}, vk::SampleCountFlagBits::e1);
  std::array<vk::DynamicState, 3> dynStates = {
      vk::DynamicState::eViewport, vk::DynamicState::eScissor, vk::DynamicState::eLineWidth};
  vk::PipelineDynamicStateCreateInfo dynamicSCI({}, dynStates);

  // --- Composition pipeline (PBR deferred lighting) ---
  {
    auto vertCode = vgeu::readFile(getShadersPath() + "/pbr/pbr.vert.spv");
    auto fragCode = vgeu::readFile(getShadersPath() + "/pbr/pbr.frag.spv");
    auto vertModule = vgeu::createShaderModule(device, vertCode);
    auto fragModule = vgeu::createShaderModule(device, fragCode);

    SpecializationData specData{0};
    std::vector<vk::SpecializationMapEntry> specEntries;
    specEntries.emplace_back(0u, offsetof(SpecializationData, displayTargetIndex), sizeof(uint32_t));
    vk::SpecializationInfo specInfo(specEntries,
        vk::ArrayProxyNoTemporaries<const SpecializationData>(specData));

    std::array<vk::PipelineShaderStageCreateInfo, 2> stages{
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                                          *vertModule, "main", nullptr),
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment,
                                          *fragModule, "main", &specInfo)};

    vk::PipelineVertexInputStateCreateInfo emptyVertexSCI{};
    rasterizationSCI.cullMode = vk::CullModeFlagBits::eFront;  // big triangle CW

    vk::GraphicsPipelineCreateInfo pipelineCI(
        vk::PipelineCreateFlagBits::eAllowDerivatives, stages,
        &emptyVertexSCI, &inputAssemblySCI, nullptr, &viewportSCI,
        &rasterizationSCI, &multisampleSCI, &depthStencilSCI, &colorBlendSCI,
        &dynamicSCI, *pipelineLayoutComposition, *renderPass);
    pipelines.composition = vk::raii::Pipeline(device, pipelineCache, pipelineCI);

    // Derivative pipelines for debug display targets
    pipelineCI.flags = vk::PipelineCreateFlagBits::eDerivative;
    pipelineCI.basePipelineHandle = *pipelines.composition;
    pipelineCI.basePipelineIndex  = -1;
    for (uint32_t i = 0; i < static_cast<uint32_t>(opts.numTargets); i++) {
      specData.displayTargetIndex = i;
      vk::SpecializationInfo si(specEntries,
          vk::ArrayProxyNoTemporaries<const SpecializationData>(specData));
      stages[1] = vk::PipelineShaderStageCreateInfo(
          {}, vk::ShaderStageFlagBits::eFragment, *fragModule, "main", &si);
      pipelines.displayTargets.emplace_back(device, pipelineCache, pipelineCI);
    }
  }

  // --- G-Buffer (offscreen MRT) pipeline ---
  {
    auto vertCode = vgeu::readFile(getShadersPath() + "/pbr/mrt.vert.spv");
    auto fragCode = vgeu::readFile(getShadersPath() + "/pbr/mrt.frag.spv");
    auto vertModule = vgeu::createShaderModule(device, vertCode);
    auto fragModule = vgeu::createShaderModule(device, fragCode);

    std::array<vk::PipelineShaderStageCreateInfo, 2> stages{
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                                          *vertModule, "main", nullptr),
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment,
                                          *fragModule, "main", nullptr)};

    auto vertexInputSCI = vgeu::glTF::Vertex::getPipelineVertexInputState({
        vgeu::glTF::VertexComponent::kPosition,
        vgeu::glTF::VertexComponent::kUV,
        vgeu::glTF::VertexComponent::kColor,
        vgeu::glTF::VertexComponent::kNormal,
        vgeu::glTF::VertexComponent::kTangent});
    rasterizationSCI.cullMode = vk::CullModeFlagBits::eNone;

    std::array<vk::PipelineColorBlendAttachmentState, 5> blendAttachments;
    blendAttachments.fill(blendAttachment);
    colorBlendSCI.setAttachments(blendAttachments);

    vk::GraphicsPipelineCreateInfo pipelineCI(
        vk::PipelineCreateFlagBits::eAllowDerivatives, stages,
        &vertexInputSCI, &inputAssemblySCI, nullptr, &viewportSCI,
        &rasterizationSCI, &multisampleSCI, &depthStencilSCI, &colorBlendSCI,
        &dynamicSCI, *pipelineLayoutOffScreen, *offScreenFrameBuf.renderPass);
    pipelines.offScreen = vk::raii::Pipeline(device, pipelineCache, pipelineCI);
  }

  // --- Sprite pipeline (forward, depth test on, depth write off) ---
  {
    auto vertCode = vgeu::readFile(getShadersPath() + "/pbr/sprite.vert.spv");
    auto fragCode = vgeu::readFile(getShadersPath() + "/pbr/sprite.frag.spv");
    auto vertModule = vgeu::createShaderModule(device, vertCode);
    auto fragModule = vgeu::createShaderModule(device, fragCode);

    std::array<vk::PipelineShaderStageCreateInfo, 2> stages{
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                                          *vertModule, "main", nullptr),
        vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment,
                                          *fragModule, "main", nullptr)};

    vk::PipelineVertexInputStateCreateInfo emptyVertexSCI{};
    rasterizationSCI.cullMode = vk::CullModeFlagBits::eNone;

    // Depth test ON, depth write OFF
    vk::PipelineDepthStencilStateCreateInfo spriteDepthSCI(
        {}, true /*depthTestEnable*/, false /*depthWriteEnable*/,
        vk::CompareOp::eLessOrEqual, false, false, stencilOpState, stencilOpState);

    vk::PipelineColorBlendAttachmentState spriteBlendAttachment(
        true,
        vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd,
        vk::BlendFactor::eOne,      vk::BlendFactor::eOneMinusSrcAlpha, vk::BlendOp::eAdd,
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
    colorBlendSCI.setAttachments(spriteBlendAttachment);

    vk::GraphicsPipelineCreateInfo pipelineCI(
        {}, stages, &emptyVertexSCI, &inputAssemblySCI, nullptr, &viewportSCI,
        &rasterizationSCI, &multisampleSCI, &spriteDepthSCI, &colorBlendSCI,
        &dynamicSCI, *pipelineLayoutSprite, *renderPass);
    pipelines.sprite = vk::raii::Pipeline(device, pipelineCache, pipelineCI);
  }
}

void VgeExample::updateUboComposition() {
  // Advance animation time using VgeBase::frameTimer (delta time in seconds)
  if (opts.animateLights && !paused) {
    lightAnimTime += frameTimer * opts.rotationSpeed;
  }

  // Light palette: White, Red, Blue, Yellow, Green, Orange, Purple, Cyan, etc.
  static const glm::vec3 kColors[MAX_LIGHTS] = {
      {1.5f, 1.5f, 1.5f}, {1.5f, 0.0f, 0.0f}, {0.0f, 0.0f, 2.5f},
      {1.5f, 1.5f, 0.0f}, {0.0f, 1.5f, 0.2f}, {1.5f, 0.7f, 0.3f},
      {1.0f, 0.3f, 1.0f}, {0.3f, 1.0f, 1.0f}, {1.0f, 0.5f, 0.0f},
      {0.5f, 0.0f, 1.0f}};

  uniformDataComposition.numLights = opts.numLights;
  for (int i = 0; i < opts.numLights; i++) {
    float phase = (2.0f * glm::pi<float>() * i) / opts.numLights;
    float angle = lightAnimTime + phase;
    uniformDataComposition.lights[i].position = glm::vec4(
        opts.orbitRadius * std::cos(angle),
        opts.orbitHeight,
        opts.orbitRadius * std::sin(angle),
        1.0f);
    uniformDataComposition.lights[i].color  = kColors[i % MAX_LIGHTS] * opts.lightIntensity;
    uniformDataComposition.lights[i].radius = 15.0f;
  }

  uniformDataComposition.viewPos =
      glm::vec4(camera.getPosition(), 0.f) * glm::vec4(-1.f, 1.f, -1.f, 1.f);
  uniformDataComposition.debugDisplayTarget = opts.debugDisplayTarget;
  uniformDataComposition.nearPlane = camera.getNearPlane();
  uniformDataComposition.farPlane  = camera.getFarPlane();
  uniformDataComposition.farClamp  = opts.farClamp;

  uniformDataComposition.useDirectionalLight = opts.useDirectionalLight ? 1 : 0;
  glm::vec3 dir = glm::normalize(glm::vec3(opts.dirLightDir[0], opts.dirLightDir[1], opts.dirLightDir[2]));
  uniformDataComposition.dirLightDir   = glm::vec4(dir, 0.f);
  uniformDataComposition.dirLightColor = glm::vec3(1.f) * opts.lightIntensity;

  std::memcpy(uniformBuffers[currentFrameIndex].composition->getMappedData(),
              &uniformDataComposition, sizeof(UniformDataComposition));
}

void VgeExample::updateUboOffScreen() {
  uniformDataOffscreen.projection = camera.getProjection();
  uniformDataOffscreen.view       = camera.getView();
  std::memcpy(uniformBuffers[currentFrameIndex].offScreen->getMappedData(),
              &uniformDataOffscreen, sizeof(UniformDataOffscreen));
}

void VgeExample::buildCommandBuffers() {
  const vk::raii::CommandBuffer& cmd = drawCmdBuffers[currentFrameIndex];
  cmd.begin({});

  // --- Pass 1: G-Buffer offscreen ---
  {
    std::array<vk::ClearValue, 6> clearValues;
    for (int i = 0; i < 5; i++) clearValues[i].color = vk::ClearColorValue(0.f, 0.f, 0.f, 0.f);
    clearValues[5].depthStencil = vk::ClearDepthStencilValue(1.f, 0);

    cmd.beginRenderPass(
        vk::RenderPassBeginInfo(
            *offScreenFrameBuf.renderPass,
            *offScreenFrameBuf.frameBuffers[currentFrameIndex],
            vk::Rect2D({}, vk::Extent2D(offScreenFrameBuf.width, offScreenFrameBuf.height)),
            clearValues),
        vk::SubpassContents::eInline);
    cmd.setViewport(0, vk::Viewport(0.f, 0.f,
        static_cast<float>(offScreenFrameBuf.width),
        static_cast<float>(offScreenFrameBuf.height), 0.f, 1.f));
    cmd.setScissor(0, vk::Rect2D({}, vk::Extent2D(offScreenFrameBuf.width, offScreenFrameBuf.height)));
    cmd.setLineWidth(1.f);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.offScreen);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayoutOffScreen, 0,
        {*descriptorSets.offScreenUboDescriptorSets[currentFrameIndex]}, nullptr);

    for (size_t instIdx = 0; instIdx < modelInstances.size(); instIdx++) {
      const auto& inst = modelInstances[instIdx];
      if (!inst.model) continue;
      // Skip instances that belong to the inactive mode
      if (inst.sceneMode == ModelInstance::SceneMode::kModelOnly && opts.useSpheres) continue;
      if (inst.sceneMode == ModelInstance::SceneMode::kSphereOnly && !opts.useSpheres) continue;

      cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayoutOffScreen, 1,
          {*descriptorSets.dynamicUboDescriptorSets[currentFrameIndex]},
          static_cast<uint32_t>(alignedSizeDynamicUboElt * instIdx));

      if (inst.sceneMode == ModelInstance::SceneMode::kSphereOnly) {
        // Bind dummy textures at set 2; skip kBindImages so model::draw() doesn't override them.
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayoutOffScreen, 2,
            {*sphereDummyDescriptorSet}, nullptr);
        inst.model->draw(currentFrameIndex, cmd, vgeu::RenderFlags{} /*no kBindImages*/,
                         *pipelineLayoutOffScreen, 2);
      } else {
        inst.model->draw(currentFrameIndex, cmd, vgeu::RenderFlagBits::kBindImages,
                         *pipelineLayoutOffScreen, 2);
      }
    }
    cmd.endRenderPass();
  }

  // --- Pass 2: Composition (PBR) + Pass 3: Sprite — same swapchain renderpass ---
  {
    std::array<vk::ClearValue, 2> clearValues;
    clearValues[0].color        = vk::ClearColorValue(0.5f, 0.5f, 0.5f, 0.5f);
    clearValues[1].depthStencil = vk::ClearDepthStencilValue(1.f, 0);

    cmd.beginRenderPass(
        vk::RenderPassBeginInfo(
            *renderPass, *frameBuffers[currentImageIndex],
            vk::Rect2D({}, swapChainData->swapChainExtent),
            clearValues),
        vk::SubpassContents::eInline);

    const float w = static_cast<float>(swapChainData->swapChainExtent.width);
    const float h = static_cast<float>(swapChainData->swapChainExtent.height);
    cmd.setViewport(0, vk::Viewport(0.f, 0.f, w, h, 0.f, 1.f));
    cmd.setScissor(0, vk::Rect2D({}, swapChainData->swapChainExtent));
    cmd.setLineWidth(1.f);

    // Composition (PBR fullscreen)
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.composition);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayoutComposition, 0,
        {*descriptorSets.composition[currentFrameIndex]}, nullptr);
    cmd.draw(3, 1, 0, 0);  // big triangle

    // Sprite forward pass (billboard quads for each point light; skip in directional mode)
    if (!opts.useDirectionalLight) {
      cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.sprite);
      std::array<vk::DescriptorSet, 2> spriteDescSets = {
          *descriptorSets.offScreenUboDescriptorSets[currentFrameIndex],
          *descriptorSets.sprite[currentFrameIndex]};
      cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayoutSprite, 0,
          spriteDescSets, nullptr);
      SpritePushConstants pc{opts.spriteSize};
      cmd.pushConstants<SpritePushConstants>(*pipelineLayoutSprite,
                        vk::ShaderStageFlagBits::eVertex, 0, pc);
      // 6 vertices per quad, opts.numLights instances
      cmd.draw(6, static_cast<uint32_t>(opts.numLights), 0, 0);
    }

    // Rebind composition descriptor set before displayTargets loop.
    // After the sprite pass, set 0 is bound with pipelineLayoutSprite.
    // displayTargets pipelines use pipelineLayoutComposition, which is incompatible for set 0.
    // Rebinding here prevents VUID-vkCmdDraw-None-02697.
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayoutComposition, 0,
        {*descriptorSets.composition[currentFrameIndex]}, nullptr);

    // Display target sub-viewports (debug G-buffer views, top-right corner) - drawn last to stay on top
    if (opts.showDebugViews) {
      const int kRows = 5;
      const int kCols = (opts.numTargets - 1) / kRows + 1;
      const float scale = 1.f / static_cast<float>(kRows);
      const float vw = w * scale, vh = h * scale;
      for (int i = 1; i < opts.numTargets; i++) {
        float vx = (w - vw * kCols) + vw * (i / kRows);
        float vy = vh * (i % kRows);
        cmd.setViewport(0, vk::Viewport(vx, vy, vw, vh, 0.f, 1.f));
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipelines.displayTargets[i]);
        cmd.draw(3, 1, 0, 0);
      }
    }

    // Reset viewport to full screen for UI
    cmd.setViewport(0, vk::Viewport(0.f, 0.f, w, h, 0.f, 1.f));

    drawUI(cmd);
    cmd.endRenderPass();
  }
  cmd.end();
  offScreenFrameBuf.isFirstFrame[currentFrameIndex] = false;
}

void VgeExample::updateDynamicUbo() {
  // Sync sphere albedo color from opts to GPU for the current frame.
  // Called every frame so that UI color picker changes take effect immediately.
  for (size_t instIdx = 0; instIdx < modelInstances.size(); instIdx++) {
    if (modelInstances[instIdx].sceneMode != ModelInstance::SceneMode::kSphereOnly) continue;
    dynamicUbo[instIdx].modelColor = glm::vec4(
        opts.sphereAlbedo[0], opts.sphereAlbedo[1], opts.sphereAlbedo[2], 1.0f);
    std::memcpy(
        static_cast<char*>(uniformBuffers[currentFrameIndex].dynamic->getMappedData())
            + instIdx * alignedSizeDynamicUboElt,
        &dynamicUbo[instIdx], sizeof(DynamicUboElt));
  }
}

void VgeExample::draw() {
  {
    vk::Result result = device.waitForFences(*waitFences[currentFrameIndex],
        VK_TRUE, std::numeric_limits<uint64_t>::max());
    assert(result != vk::Result::eTimeout);
    device.resetFences(*waitFences[currentFrameIndex]);
  }
  prepareFrame();
  updateUboOffScreen();
  updateUboComposition();
  updateDynamicUbo();
  buildCommandBuffers();
  {
    vk::PipelineStageFlags waitStage(vk::PipelineStageFlagBits::eColorAttachmentOutput);
    vk::SubmitInfo submitInfo(
        *presentCompleteSemaphores[currentFrameIndex], waitStage,
        *drawCmdBuffers[currentFrameIndex],
        *renderCompleteSemaphores[currentFrameIndex]);
    queue.submit(submitInfo, *waitFences[currentFrameIndex]);
  }
  submitFrame();
}

void VgeExample::addModelInstance(ModelInstance&& newInstance) {
  size_t idx = modelInstances.size();
  instanceMap[newInstance.name].push_back(idx);
  modelInstances.push_back(std::move(newInstance));
}

const std::vector<size_t>& VgeExample::findInstances(const std::string& name) {
  assert(instanceMap.find(name) != instanceMap.end());
  return instanceMap.at(name);
}

ModelInstance::ModelInstance(ModelInstance&& other) {
  model          = other.model;
  name           = other.name;
  isBone         = other.isBone;
  animationIndex = other.animationIndex;
  animationTime  = other.animationTime;
  transform      = other.transform;
  sceneMode      = other.sceneMode;
}

ModelInstance& ModelInstance::operator=(ModelInstance&& other) {
  model          = other.model;
  name           = other.name;
  isBone         = other.isBone;
  animationIndex = other.animationIndex;
  animationTime  = other.animationTime;
  transform      = other.transform;
  sceneMode      = other.sceneMode;
  return *this;
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()
