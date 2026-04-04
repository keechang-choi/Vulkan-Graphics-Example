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
      ImGui::Checkbox("Animate Lights", &opts.animateLights);
      ImGui::DragFloat("Rotation Speed", &opts.rotationSpeed, 0.05f, 0.0f, 10.f, "%.2f");
      ImGui::DragFloat("Orbit Radius",   &opts.orbitRadius,   0.1f,  0.5f, 30.f, "%.1f");
      ImGui::DragFloat("Orbit Height",   &opts.orbitHeight,   0.1f, -20.f, 0.f,  "%.1f");
      ImGui::DragInt("Num Lights", &opts.numLights, 1, 1, MAX_LIGHTS);
      ImGui::DragFloat("Sprite Size", &opts.spriteSize, 0.01f, 0.05f, 2.f, "%.2f");
      ImGui::TreePop();
    }
  }
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
}

void VgeExample::setupDynamicUbo() {
  glm::vec3 up{0.f, -1.f, 0.f};
  glm::vec3 right{1.f, 0.f, 0.f};
  dynamicUbo.resize(modelInstances.size());
  {
    size_t idx = findInstances("floor")[0];
    dynamicUbo[idx].modelMatrix = glm::scale(glm::mat4{1.f}, glm::vec3{10.f, 10.f, 10.f});
    dynamicUbo[idx].modelColor  = glm::vec4{1.f, 0.f, 0.f, 0.3f};
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
      dynamicUbo[idx].modelColor  = glm::vec4{1.f, 0.f, 0.f, 0.3f};
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
      static_cast<uint32_t>(MAX_CONCURRENT_FRAMES * offScreenFrameBuf.numAttachments));

  vk::DescriptorPoolCreateInfo poolCI(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      MAX_CONCURRENT_FRAMES /*composition*/ +
      MAX_CONCURRENT_FRAMES * 2 /*offscreen + dynamic*/ +
      MAX_CONCURRENT_FRAMES /*sprite*/,
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
                                           vk::ShaderStageFlagBits::eVertex);
    dynamicUboDescriptorSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo({}, binding));
  }

  // Offscreen pipeline layout: set0=offscreenUBO, set1=dynamicUBO, set2=modelImage, set3=modelUBO
  {
    std::vector<vk::DescriptorSetLayout> setLayouts = {
        *offScreenUboDescriptorSetLayout,
        *dynamicUboDescriptorSetLayout,
        *modelInstances[0].model->descriptorSetLayoutImage,
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
}

}  // namespace vge
