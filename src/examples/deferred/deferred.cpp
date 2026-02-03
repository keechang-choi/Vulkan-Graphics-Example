#include "deferred.hpp"

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

namespace vge {
VgeExample::VgeExample() : VgeBase() { title = "Deferred Shading Example"; }
VgeExample::~VgeExample() {}
void VgeExample::setupCommandLineParser(CLI::App& app) {}
void VgeExample::setOptions(const std::optional<Options>& opts) {
  if (opts.has_value()) {
    this->opts = opts.value();
    // overwrite cli args for restart run
    cameraController.moveSpeed = this->opts.moveSpeed;
  } else {
    // save cli args for initial run
  }
}

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

void VgeExample::getEnabledExtensions() {}
void VgeExample::getEnabledFeatures() {
  enabledFeatures.samplerAnisotropy =
      physicalDevice.getFeatures().samplerAnisotropy;
  enabledFeatures.fillModeNonSolid =
      physicalDevice.getFeatures().fillModeNonSolid;
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
  if (!prepared) {
    return;
  }
  // draw
  draw();
}

void VgeExample::viewChanged() {}
void VgeExample::onUpdateUIOverlay() {
  if (uiOverlay->header("Settings")) {
    if (ImGui::TreeNodeEx("Immediate", ImGuiTreeNodeFlags_DefaultOpen)) {
      for (auto i = 0; i < opts.numTargets; i++) {
        std::string caption = "debugDisplayTarget: " + std::to_string(i);
        uiOverlay->radioButton(caption.c_str(), &opts.debugDisplayTarget, i);
      }
      ImGui::TreePop();
    }
  }
}

void VgeExample::loadAssets() {
  // NOTE: no flip or preTransform for animation and skinning
  vgeu::FileLoadingFlags glTFLoadingFlags =
      vgeu::FileLoadingFlagBits::kPreMultiplyVertexColors |
      vgeu::FileLoadingFlagBits::kPreTransformVertices |
      vgeu::FileLoadingFlagBits::kFlipY;

  std::shared_ptr<vgeu::glTF::Model> floor;
  floor = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  floor->descriptorBindingFlags =
      vgeu::DescriptorBindingFlagBits::kImageBaseColor |
      vgeu::DescriptorBindingFlagBits::kImageNormalMap;
  floor->loadFromFile(
      getAssetsPath() + "/models/metal_plate/metal_plate_1k.gltf",
      glTFLoadingFlags);
  {
    ModelInstance modelInstance{};
    modelInstance.model = floor;
    modelInstance.name = "floor";
    addModelInstance(std::move(modelInstance));
  }

  std::shared_ptr<vgeu::glTF::Model> damagedHelmet;
  damagedHelmet = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  damagedHelmet->descriptorBindingFlags =
      vgeu::DescriptorBindingFlagBits::kImageBaseColor |
      vgeu::DescriptorBindingFlagBits::kImageNormalMap;
  damagedHelmet->loadFromFile(
      getAssetsPath() + "/models/DamagedHelmet/glTF/DamagedHelmet.gltf",
      glTFLoadingFlags);
  {
    ModelInstance modelInstance{};
    modelInstance.model = damagedHelmet;
    modelInstance.name = "damagedHelmet1";
    addModelInstance(std::move(modelInstance));
  }
  {
    ModelInstance modelInstance{};
    modelInstance.model = damagedHelmet;
    modelInstance.name = "damagedHelmet2";
    addModelInstance(std::move(modelInstance));
  }
}

std::unique_ptr<vgeu::VgeuImage> VgeExample::createAttachment(
    vk::Format format, vk::ImageUsageFlags usage) {
  vk::ImageAspectFlags aspectMask{};
  vk::ImageLayout imageLayout;
  if (usage & vk::ImageUsageFlagBits::eColorAttachment) {
    aspectMask = vk::ImageAspectFlagBits::eColor;
    imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
  } else if (usage & vk::ImageUsageFlagBits::eDepthStencilAttachment) {
    aspectMask = vk::ImageAspectFlagBits::eDepth;
    /* NOTE(kcchoi): image view validation error for layout transition
    if (format >= vk::Format::eD16UnormS8Uint)
      aspectMask |= vk::ImageAspectFlagBits::eStencil;
    */
    imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
  }
  assert(aspectMask != vk::ImageAspectFlagBits::eNone);
  // NOTE(kcchoi): undefined or preinitialized validation error
  // VUID-VkImageCreateInfo-initialLayout-00993
  imageLayout = vk::ImageLayout::eUndefined;
  usage = usage | vk::ImageUsageFlagBits::eSampled;
  return std::make_unique<vgeu::VgeuImage>(
      device, globalAllocator->getAllocator(), format,
      vk::Extent2D{offScreenFrameBuf.width, offScreenFrameBuf.height},
      vk::ImageTiling::eOptimal, usage, imageLayout,
      VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
      aspectMask, 1);
}

void VgeExample::setupDynamicUbo() {
  glm::vec3 up{0.f, -1.f, 0.f};
  glm::vec3 right{1.f, 0.f, 0.f};
  glm::vec3 forward{0.f, 0.f, -1.f};
  dynamicUbo.resize(modelInstances.size());
  {
    size_t instanceIndex = findInstances("floor")[0];
    dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, glm::vec3{0.f, 0.f, 0.f});
    dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        dynamicUbo[instanceIndex].modelMatrix, glm::radians(0.f), up);
    dynamicUbo[instanceIndex].modelMatrix = glm::scale(
        dynamicUbo[instanceIndex].modelMatrix, glm::vec3{10.0, 10.0, 10.0});
    dynamicUbo[instanceIndex].modelColor = glm::vec4{1.0f, 0.f, 0.f, 0.3f};
  }
  const float HelmetScale = 1.00f;
  {
    size_t instanceIndex = findInstances("damagedHelmet1")[0];
    dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, glm::vec3{-4.f, -4.f, 0.f});
    dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        dynamicUbo[instanceIndex].modelMatrix, glm::radians(90.f), up);
    dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        dynamicUbo[instanceIndex].modelMatrix, glm::radians(-90.f), right);
    dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{HelmetScale, HelmetScale, HelmetScale});
    dynamicUbo[instanceIndex].modelColor = glm::vec4{1.0f, 0.f, 0.f, 0.3f};
  }
  {
    size_t instanceIndex = findInstances("damagedHelmet2")[0];
    dynamicUbo[instanceIndex].modelMatrix =
        glm::translate(glm::mat4{1.f}, glm::vec3{4.f, -4.f, 0.f});
    // {0,-1,0} is up vector, rotate second
    dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        dynamicUbo[instanceIndex].modelMatrix, glm::radians(90.f), up);
    // {1,0,0} is right vector, rotate first
    dynamicUbo[instanceIndex].modelMatrix = glm::rotate(
        dynamicUbo[instanceIndex].modelMatrix, glm::radians(-90.f), right);
    dynamicUbo[instanceIndex].modelMatrix =
        glm::scale(dynamicUbo[instanceIndex].modelMatrix,
                   glm::vec3{HelmetScale, HelmetScale, HelmetScale});
    dynamicUbo[instanceIndex].modelColor = glm::vec4{1.0f, 0.f, 0.f, 0.3f};
  }
}

void VgeExample::prepareOffScreenFrameBuffer() {
  offScreenFrameBuf.width = 2048;
  offScreenFrameBuf.height = 2048;

  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    offScreenFrameBuf.position.push_back(
        std::move(createAttachment(vk::Format::eR16G16B16A16Sfloat,
                                   vk::ImageUsageFlagBits::eColorAttachment)));
    offScreenFrameBuf.normal.push_back(
        std::move(createAttachment(vk::Format::eR16G16B16A16Sfloat,
                                   vk::ImageUsageFlagBits::eColorAttachment)));
    offScreenFrameBuf.albedo.push_back(
        std::move(createAttachment(vk::Format::eR16G16B16A16Sfloat,
                                   vk::ImageUsageFlagBits::eColorAttachment)));
    offScreenFrameBuf.depth.push_back(std::move(createAttachment(
        depthFormat, vk::ImageUsageFlagBits::eDepthStencilAttachment)));
  }
  // render pass creation, subpass dependency
  std::vector<vk::AttachmentDescription> attachmentDescriptions;
  // position, normal, albedo, depth
  for (uint32_t i = 0; i < 4; i++) {
    vk::ImageLayout initialLayout;
    vk::ImageLayout finalLayout;
    if (i == 3) {
      initialLayout = vk::ImageLayout::eUndefined;
      // validation error after adding offscreen depth image
      // when using vk::ImageLayout::eDepthAttachmentOptimal
      // finalLayout = vk::ImageLayout::eDepthAttachmentOptimal;
      finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    } else {
      initialLayout = vk::ImageLayout::eUndefined;
      finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    }
    attachmentDescriptions.emplace_back(
        vk::AttachmentDescriptionFlags(), vk::Format::eUndefined,
        vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
        vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eDontCare, initialLayout, finalLayout);
  }
  attachmentDescriptions[0].format = offScreenFrameBuf.position[0]->getFormat();
  attachmentDescriptions[1].format = offScreenFrameBuf.normal[0]->getFormat();
  attachmentDescriptions[2].format = offScreenFrameBuf.albedo[0]->getFormat();
  attachmentDescriptions[3].format = offScreenFrameBuf.depth[0]->getFormat();
  std::vector<vk::AttachmentReference> colorReferences;
  colorReferences.emplace_back(0, vk::ImageLayout::eColorAttachmentOptimal);
  colorReferences.emplace_back(1, vk::ImageLayout::eColorAttachmentOptimal);
  colorReferences.emplace_back(2, vk::ImageLayout::eColorAttachmentOptimal);
  vk::AttachmentReference depthReference(
      3, vk::ImageLayout::eDepthStencilAttachmentOptimal);
  vk::SubpassDescription subpassDescription(
      vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, {},
      colorReferences, {}, &depthReference);

  // subpass dependencies for layout transition
  std::vector<vk::SubpassDependency> dependencies;
  dependencies.emplace_back(VK_SUBPASS_EXTERNAL, 0u,
                            vk::PipelineStageFlagBits::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits::eLateFragmentTests,
                            vk::PipelineStageFlagBits::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits::eLateFragmentTests,
                            vk::AccessFlagBits::eDepthStencilAttachmentWrite,
                            vk::AccessFlagBits::eDepthStencilAttachmentWrite |
                                vk::AccessFlagBits::eDepthStencilAttachmentRead,
                            vk::DependencyFlags());
  dependencies.emplace_back(VK_SUBPASS_EXTERNAL, 0u,
                            vk::PipelineStageFlagBits::eBottomOfPipe,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::AccessFlagBits::eMemoryRead,
                            vk::AccessFlagBits::eColorAttachmentWrite |
                                vk::AccessFlagBits::eColorAttachmentRead,
                            vk::DependencyFlags());
  // transition for lighting render pass after geometry pass
  /*dependencies.emplace_back(0u, VK_SUBPASS_EXTERNAL,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits::eBottomOfPipe,
                            vk::AccessFlagBits::eColorAttachmentWrite |
                                vk::AccessFlagBits::eColorAttachmentRead,
                            vk::AccessFlagBits::eMemoryRead,
                            vk::DependencyFlags());*/

  vk::RenderPassCreateInfo renderPassCreateInfo(
      vk::RenderPassCreateFlags(), attachmentDescriptions, subpassDescription,
      dependencies);
  offScreenFrameBuf.renderPass =
      vk::raii::RenderPass(device, renderPassCreateInfo);
  // frame buffer creation

  offScreenFrameBuf.frameBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    std::array<vk::ImageView, 4> attachments{};
    attachments[0] = *offScreenFrameBuf.position[i]->getImageView();
    attachments[1] = *offScreenFrameBuf.normal[i]->getImageView();
    attachments[2] = *offScreenFrameBuf.albedo[i]->getImageView();
    attachments[3] = *offScreenFrameBuf.depth[i]->getImageView();

    vk::FramebufferCreateInfo framebufferCreateInfo(
        vk::FramebufferCreateFlags(), *offScreenFrameBuf.renderPass,
        attachments, offScreenFrameBuf.width, offScreenFrameBuf.height, 1);
    offScreenFrameBuf.frameBuffers.push_back(
        vk::raii::Framebuffer(device, framebufferCreateInfo));
  }
  // sampler creation
  vk::SamplerCreateInfo samplerCI(
      vk::SamplerCreateFlags{}, vk::Filter::eNearest, vk::Filter::eNearest,
      vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge, 0.f, true, 1.0f, false,
      vk::CompareOp::eNever, 0.f, static_cast<float>(1.f),
      vk::BorderColor::eFloatOpaqueWhite);
  colorSampler = vk::raii::Sampler(device, samplerCI);
  assert(static_cast<VkSampler>(*colorSampler) != VK_NULL_HANDLE);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    assert(static_cast<VkImageView>(
               *offScreenFrameBuf.position[i]->getImageView()) !=
           VK_NULL_HANDLE);
    assert(static_cast<VkImageView>(
               *offScreenFrameBuf.normal[i]->getImageView()) != VK_NULL_HANDLE);
    assert(static_cast<VkImageView>(
               *offScreenFrameBuf.albedo[i]->getImageView()) != VK_NULL_HANDLE);
    assert(static_cast<VkImageView>(
               *offScreenFrameBuf.depth[i]->getImageView()) != VK_NULL_HANDLE);
  }
}
void VgeExample::prepareUniformBuffers() {
  alignedSizeDynamicUboElt =
      vgeu::padBufferSize(physicalDevice, sizeof(DynamicUboElt), true);
  // NOTE(kcchoi): move buffer unique_ptr after buffer allocation and copy.
  uniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    std::unique_ptr<vgeu::VgeuBuffer> dynamic =
        std::make_unique<vgeu::VgeuBuffer>(
            globalAllocator->getAllocator(), alignedSizeDynamicUboElt,
            dynamicUbo.size(), vk::BufferUsageFlagBits::eUniformBuffer,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT |
                VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT);
    for (size_t j = 0; j < dynamicUbo.size(); j++) {
      std::memcpy(static_cast<char*>(dynamic->getMappedData()) +
                      j * alignedSizeDynamicUboElt,
                  &dynamicUbo[j], alignedSizeDynamicUboElt);
    }

    std::unique_ptr<vgeu::VgeuBuffer> offScreen =
        std::make_unique<vgeu::VgeuBuffer>(
            globalAllocator->getAllocator(), sizeof(UniformDataOffscreen), 1,
            vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT |
                VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT);
    std::memcpy(offScreen->getMappedData(), &uniformDataOffscreen,
                sizeof(UniformDataOffscreen));

    std::unique_ptr<vgeu::VgeuBuffer> composition =
        std::make_unique<vgeu::VgeuBuffer>(
            globalAllocator->getAllocator(), sizeof(UniformDataComposition), 1,
            vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT |
                VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT);
    std::memcpy(composition->getMappedData(), &uniformDataComposition,
                sizeof(UniformDataComposition));

    uniformBuffers.push_back(
        {std::move(dynamic), std::move(offScreen), std::move(composition)});
  }
  std::cout << "sizeof(UniformDataComposition): "
            << sizeof(UniformDataComposition) << std::endl;
}
void VgeExample::setupDescriptors() {
  // model desciptors in gltf class.
  // pool. for each descriptor type
  std::vector<vk::DescriptorPoolSize> poolSizes;

  poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer,
                         /* uniform offscreen, uniform composition*/
                         MAX_CONCURRENT_FRAMES + MAX_CONCURRENT_FRAMES);
  poolSizes.emplace_back(vk::DescriptorType::eUniformBufferDynamic,
                         /* uniform offscreen */
                         MAX_CONCURRENT_FRAMES);
  poolSizes.emplace_back(vk::DescriptorType::eCombinedImageSampler,
                         /* position, normal, albedo */
                         MAX_CONCURRENT_FRAMES * 4);
  // max sets
  vk::DescriptorPoolCreateInfo descriptorPoolCI(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      /* composition set */
      MAX_CONCURRENT_FRAMES +
          /* offscree set (ubo, dynamic)*/
          MAX_CONCURRENT_FRAMES * 2,
      poolSizes);
  descriptorPool = vk::raii::DescriptorPool(device, descriptorPoolCI);
  // layout
  // can use generalized descriptor set layout for offscreen and composition,
  // but we dont.
  {
    // deferred compisition
    std::vector<vk::DescriptorSetLayout> setLayouts;
    {
      // set 0
      std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
      // position
      layoutBindings.emplace_back(0 /* binding */,
                                  vk::DescriptorType::eCombinedImageSampler, 1,
                                  vk::ShaderStageFlagBits::eFragment);
      // normal
      layoutBindings.emplace_back(1 /* binding */,
                                  vk::DescriptorType::eCombinedImageSampler, 1,
                                  vk::ShaderStageFlagBits::eFragment);
      // albedo
      layoutBindings.emplace_back(2 /* binding */,
                                  vk::DescriptorType::eCombinedImageSampler, 1,
                                  vk::ShaderStageFlagBits::eFragment);
      // depth
      layoutBindings.emplace_back(3 /* binding */,
                                  vk::DescriptorType::eCombinedImageSampler, 1,
                                  vk::ShaderStageFlagBits::eFragment);
      // fragment uniform
      layoutBindings.emplace_back(4 /* binding */,
                                  vk::DescriptorType::eUniformBuffer, 1,
                                  vk::ShaderStageFlagBits::eFragment);
      vk::DescriptorSetLayoutCreateInfo layoutCI(
          vk::DescriptorSetLayoutCreateFlags{}, layoutBindings);
      compositionDescriptorSetLayout =
          vk::raii::DescriptorSetLayout(device, layoutCI);
      setLayouts.push_back(*compositionDescriptorSetLayout);
    }
    // create pipelineLayout
    vk::PipelineLayoutCreateInfo pipelineLayoutCI({}, setLayouts);
    pipelineLayoutCompoisition =
        vk::raii::PipelineLayout(device, pipelineLayoutCI);
  }
  {
    // off screen
    std::vector<vk::DescriptorSetLayout> setLayouts;
    {
      // set 0
      std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
      // vertex uniform
      layoutBindings.emplace_back(0 /* binding */,
                                  vk::DescriptorType::eUniformBuffer, 1,
                                  vk::ShaderStageFlagBits::eVertex);
      vk::DescriptorSetLayoutCreateInfo layoutCI(
          vk::DescriptorSetLayoutCreateFlags{}, layoutBindings);
      offScreenUboDescriptorSetLayout =
          vk::raii::DescriptorSetLayout(device, layoutCI);
      setLayouts.push_back(*offScreenUboDescriptorSetLayout);
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

    // create pipelineLayout
    vk::PipelineLayoutCreateInfo pipelineLayoutCI({}, setLayouts);
    pipelineLayoutOffScreen =
        vk::raii::PipelineLayout(device, pipelineLayoutCI);
  }
  // descriptor sets
  {
    // deferred composition
    vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                            *compositionDescriptorSetLayout);
    descriptorSets.composition.reserve(MAX_CONCURRENT_FRAMES);
    for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      // NOTE: move descriptor set
      descriptorSets.composition.push_back(
          std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
    }
    for (int i = 0; i < descriptorSets.composition.size(); i++) {
      vk::DescriptorBufferInfo bufferInfo =
          uniformBuffers[i].composition->descriptorInfo();
      vk::DescriptorImageInfo posImageInfo =
          offScreenFrameBuf.position[i]->descriptorImageInfo(
              *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);
      vk::DescriptorImageInfo normImageInfo =
          offScreenFrameBuf.normal[i]->descriptorImageInfo(
              *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);
      vk::DescriptorImageInfo albedoImageInfo =
          offScreenFrameBuf.albedo[i]->descriptorImageInfo(
              *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);
      vk::DescriptorImageInfo depthImageInfo =
          offScreenFrameBuf.depth[i]->descriptorImageInfo(
              *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);

      std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
      writeDescriptorSets.reserve(5);

      // NOTE: dstBinding, dstArrayElement
      writeDescriptorSets.emplace_back(
          *descriptorSets.composition[i], 0, 0,
          vk::DescriptorType::eCombinedImageSampler, posImageInfo, nullptr);
      writeDescriptorSets.emplace_back(
          *descriptorSets.composition[i], 1, 0,
          vk::DescriptorType::eCombinedImageSampler, normImageInfo, nullptr);
      writeDescriptorSets.emplace_back(
          *descriptorSets.composition[i], 2, 0,
          vk::DescriptorType::eCombinedImageSampler, albedoImageInfo, nullptr);
      writeDescriptorSets.emplace_back(
          *descriptorSets.composition[i], 3, 0,
          vk::DescriptorType::eCombinedImageSampler, depthImageInfo, nullptr);
      writeDescriptorSets.emplace_back(*descriptorSets.composition[i], 4, 0,
                                       vk::DescriptorType::eUniformBuffer,
                                       nullptr, bufferInfo);
      device.updateDescriptorSets(writeDescriptorSets, nullptr);
    }
  }
  // offscreen UBO
  {
    vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                            *offScreenUboDescriptorSetLayout);
    descriptorSets.offScreenUboDescriptorSets.reserve(MAX_CONCURRENT_FRAMES);
    for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      // NOTE: move descriptor set
      descriptorSets.offScreenUboDescriptorSets.push_back(
          std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
    }
    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(descriptorSets.offScreenUboDescriptorSets.size());
    std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
    writeDescriptorSets.reserve(
        descriptorSets.offScreenUboDescriptorSets.size());
    for (int i = 0; i < descriptorSets.offScreenUboDescriptorSets.size(); i++) {
      // copy
      bufferInfos.push_back(uniformBuffers[i].offScreen->descriptorInfo());
      // NOTE: ArrayProxyNoTemporaries has no T rvalue constructor.
      writeDescriptorSets.emplace_back(
          *descriptorSets.offScreenUboDescriptorSets[i], 0, 0,
          vk::DescriptorType::eUniformBuffer, nullptr, bufferInfos.back());
    }
    device.updateDescriptorSets(writeDescriptorSets, nullptr);
  }
  // dynamic UBO
  {
    vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                            *dynamicUboDescriptorSetLayout);
    descriptorSets.dynamicUboDescriptorSets.reserve(MAX_CONCURRENT_FRAMES);
    for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
      descriptorSets.dynamicUboDescriptorSets.push_back(
          std::move(vk::raii::DescriptorSets(device, allocInfo).front()));
    }
    std::vector<vk::DescriptorBufferInfo> bufferInfos;
    bufferInfos.reserve(descriptorSets.dynamicUboDescriptorSets.size());
    std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
    writeDescriptorSets.reserve(descriptorSets.dynamicUboDescriptorSets.size());
    for (int i = 0; i < descriptorSets.dynamicUboDescriptorSets.size(); i++) {
      // NOTE: descriptorBufferInfo range be alignedSizeDynamicUboElt
      bufferInfos.push_back(uniformBuffers[i].dynamic->descriptorInfo(
          alignedSizeDynamicUboElt, 0));
      writeDescriptorSets.emplace_back(
          *descriptorSets.dynamicUboDescriptorSets[i], 0, 0,
          vk::DescriptorType::eUniformBufferDynamic, nullptr,
          bufferInfos.back());
    }
    device.updateDescriptorSets(writeDescriptorSets, nullptr);
  }
}
void VgeExample::preparePipelines() {
  // pipeline layout already created for offscreen and composition.

  vk::PipelineInputAssemblyStateCreateInfo inputAssemblySCI(
      vk::PipelineInputAssemblyStateCreateFlags(),
      vk::PrimitiveTopology::eTriangleList);

  vk::PipelineRasterizationStateCreateInfo rasterizationSCI(
      vk::PipelineRasterizationStateCreateFlags(), false, false,
      vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack,
      vk::FrontFace::eCounterClockwise, false, 0.0f, 0.0f, 0.0f, 1.0f);
  vk::PipelineColorBlendAttachmentState colorBlendAttachmentState(
      false, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
      vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
  vk::PipelineColorBlendStateCreateInfo colorBlendSCI(
      vk::PipelineColorBlendStateCreateFlags(), false, vk::LogicOp::eNoOp,
      colorBlendAttachmentState, {{1.0f, 1.0f, 1.0f, 1.0f}});
  vk::StencilOpState stencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep,
                                    vk::StencilOp::eKeep,
                                    vk::CompareOp::eAlways);
  vk::PipelineDepthStencilStateCreateInfo depthStencilSCI(
      vk::PipelineDepthStencilStateCreateFlags(), true /*depthTestEnable*/,
      true, vk::CompareOp::eLessOrEqual, false, false, stencilOpState,
      stencilOpState);
  vk::PipelineViewportStateCreateInfo viewportSCI(
      vk::PipelineViewportStateCreateFlags(), 1, nullptr, 1, nullptr);
  vk::PipelineMultisampleStateCreateInfo multisampleSCI(
      vk::PipelineMultisampleStateCreateFlags(), vk::SampleCountFlagBits::e1);
  std::array<vk::DynamicState, 3> dynamicStates = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor,
      vk::DynamicState::eLineWidth,
  };
  vk::PipelineDynamicStateCreateInfo dynamicSCI(
      vk::PipelineDynamicStateCreateFlags(), dynamicStates);

  {
    auto vertCode =
        vgeu::readFile(getShadersPath() + "/deferred/deferred.vert.spv");
    auto fragCode =
        vgeu::readFile(getShadersPath() + "/deferred/deferred.frag.spv");
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

    // deferred vert : one big triangle generated by vertex index.
    vk::PipelineVertexInputStateCreateInfo emptyVertexInputSCI(
        vk::PipelineVertexInputStateCreateFlags{});
    // triangle cw in vert shader
    rasterizationSCI.cullMode = vk::CullModeFlagBits::eFront;
    // base and derivatives
    vk::GraphicsPipelineCreateInfo pipelineCI(
        vk::PipelineCreateFlagBits::eAllowDerivatives, shaderStageCIs,
        &emptyVertexInputSCI, &inputAssemblySCI, nullptr, &viewportSCI,
        &rasterizationSCI, &multisampleSCI, &depthStencilSCI, &colorBlendSCI,
        &dynamicSCI, *pipelineLayoutCompoisition, *renderPass);

    pipelines.composition =
        vk::raii::Pipeline(device, pipelineCache, pipelineCI);
  }
  {
    auto vertCode = vgeu::readFile(getShadersPath() + "/deferred/mrt.vert.spv");
    auto fragCode = vgeu::readFile(getShadersPath() + "/deferred/mrt.frag.spv");
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
    vk::PipelineVertexInputStateCreateInfo vertexInputSCI =
        vgeu::glTF::Vertex::getPipelineVertexInputState({
            vgeu::glTF::VertexComponent::kPosition,
            vgeu::glTF::VertexComponent::kUV,
            vgeu::glTF::VertexComponent::kColor,
            vgeu::glTF::VertexComponent::kNormal,
            vgeu::glTF::VertexComponent::kTangent,
        });
    // TODO(kcchoi): check face winding order in model
    rasterizationSCI.cullMode = vk::CullModeFlagBits::eNone;
    // TODO(kcchoi): check mask color for 0x0
    // position, normal, albedo
    std::array<vk::PipelineColorBlendAttachmentState, 3> blendAttachmentStates{
        vk::PipelineColorBlendAttachmentState(
            false, vk::BlendFactor::eZero, vk::BlendFactor::eZero,
            vk::BlendOp::eAdd, vk::BlendFactor::eZero, vk::BlendFactor::eZero,
            vk::BlendOp::eAdd,
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB |
                vk::ColorComponentFlagBits::eA),
        vk::PipelineColorBlendAttachmentState(
            false, vk::BlendFactor::eZero, vk::BlendFactor::eZero,
            vk::BlendOp::eAdd, vk::BlendFactor::eZero, vk::BlendFactor::eZero,
            vk::BlendOp::eAdd,
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB |
                vk::ColorComponentFlagBits::eA),
        vk::PipelineColorBlendAttachmentState(
            false, vk::BlendFactor::eZero, vk::BlendFactor::eZero,
            vk::BlendOp::eAdd, vk::BlendFactor::eZero, vk::BlendFactor::eZero,
            vk::BlendOp::eAdd,
            vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB |
                vk::ColorComponentFlagBits::eA),
    };
    colorBlendSCI.setAttachments(blendAttachmentStates);
    vk::GraphicsPipelineCreateInfo pipelineCI(
        vk::PipelineCreateFlagBits::eAllowDerivatives, shaderStageCIs,
        &vertexInputSCI, &inputAssemblySCI, nullptr, &viewportSCI,
        &rasterizationSCI, &multisampleSCI, &depthStencilSCI, &colorBlendSCI,
        &dynamicSCI, *pipelineLayoutOffScreen, *offScreenFrameBuf.renderPass);
    pipelines.offScreen = vk::raii::Pipeline(device, pipelineCache, pipelineCI);
  }
}
void VgeExample::updateUboComposition() {
  uniformDataComposition.numLights = 6;
  // White
  uniformDataComposition.lights[0].position =
      glm::vec4(0.0f, -5.0f, 1.0f, 0.0f);
  uniformDataComposition.lights[0].color = glm::vec3(1.5f);
  uniformDataComposition.lights[0].radius = 15.0f * 0.25f;
  // Red
  uniformDataComposition.lights[1].position =
      glm::vec4(-2.0f, -5.0f, 0.0f, 0.0f);
  uniformDataComposition.lights[1].color = glm::vec3(1.0f, 0.0f, 0.0f);
  uniformDataComposition.lights[1].radius = 15.0f;
  // Blue
  uniformDataComposition.lights[2].position =
      glm::vec4(2.0f, -5.0f, 0.0f, 0.0f);
  uniformDataComposition.lights[2].color = glm::vec3(0.0f, 0.0f, 2.5f);
  uniformDataComposition.lights[2].radius = 5.0f;
  // Yellow
  uniformDataComposition.lights[3].position = glm::vec4(0.0f, -1.f, 0.f, 0.0f);
  uniformDataComposition.lights[3].color = glm::vec3(1.0f, 1.0f, 0.0f);
  uniformDataComposition.lights[3].radius = 2.0f;
  // Green
  uniformDataComposition.lights[4].position = glm::vec4(0.0f, -5.f, 0.0f, 0.0f);
  uniformDataComposition.lights[4].color = glm::vec3(0.0f, 1.0f, 0.2f);
  uniformDataComposition.lights[4].radius = 5.0f;
  // Yellow
  uniformDataComposition.lights[5].position =
      glm::vec4(0.0f, -5.0f, 0.0f, 0.0f);
  uniformDataComposition.lights[5].color = glm::vec3(1.0f, 0.7f, 0.3f);
  uniformDataComposition.lights[5].radius = 25.0f;

  // light animation
  if (!paused) {
    uniformDataComposition.lights[0].position.x =
        sin(glm::radians(360.0f * timer)) * 5.0f;
    uniformDataComposition.lights[0].position.z =
        cos(glm::radians(360.0f * timer)) * 5.0f;

    uniformDataComposition.lights[1].position.x =
        -4.0f + sin(glm::radians(360.0f * timer) + 45.0f) * 2.0f;
    uniformDataComposition.lights[1].position.z =
        0.0f + cos(glm::radians(360.0f * timer) + 45.0f) * 2.0f;

    uniformDataComposition.lights[2].position.x =
        4.0f + sin(glm::radians(360.0f * timer)) * 2.0f;
    uniformDataComposition.lights[2].position.z =
        0.0f + cos(glm::radians(360.0f * timer)) * 2.0f;

    uniformDataComposition.lights[4].position.x =
        0.0f + sin(glm::radians(360.0f * timer + 90.0f)) * 5.0f;
    uniformDataComposition.lights[4].position.z =
        0.0f - cos(glm::radians(360.0f * timer + 45.0f)) * 5.0f;

    uniformDataComposition.lights[5].position.x =
        0.0f + sin(glm::radians(-360.0f * timer + 135.0f)) * 10.0f;
    uniformDataComposition.lights[5].position.z =
        0.0f - cos(glm::radians(-360.0f * timer - 45.0f)) * 10.0f;
  }
  uniformDataComposition.viewPos = glm::vec4(camera.getPosition(), 0.0f) *
                                   glm::vec4(-1.0f, 1.0f, -1.0f, 1.0f);

  uniformDataComposition.debugDisplayTarget = opts.debugDisplayTarget;
  memcpy(uniformBuffers[currentFrameIndex].composition->getMappedData(),
         &uniformDataComposition, sizeof(UniformDataComposition));
}

void VgeExample::updateUboOffScreen() {
  // dynamic ubo fixed.
  uniformDataOffscreen.projection = camera.getProjection();
  uniformDataOffscreen.view = camera.getView();
  std::memcpy(uniformBuffers[currentFrameIndex].offScreen->getMappedData(),
              &uniformDataOffscreen, sizeof(UniformDataOffscreen));
}
void VgeExample::buildCommandBuffers() {
  const vk::raii::CommandBuffer& cmdBuffer = drawCmdBuffers[currentFrameIndex];
  cmdBuffer.begin({});
  // first render pass for offscreen pass to fill g buffers of attachments.
  {
    std::array<vk::ClearValue, 4> clearValues;
    clearValues[0].color = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f);
    clearValues[1].color = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f);
    clearValues[2].color = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 0.0f);
    clearValues[3].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);
    // NOTE(kcchoi): offscreen frame buffer index as currentFrameIndex
    vk::RenderPassBeginInfo renderPassBeginInfo(
        *offScreenFrameBuf.renderPass,
        *offScreenFrameBuf.frameBuffers[currentFrameIndex],
        vk::Rect2D(vk::Offset2D(0, 0), vk::Extent2D(offScreenFrameBuf.width,
                                                    offScreenFrameBuf.height)),
        clearValues);

    cmdBuffer.beginRenderPass(renderPassBeginInfo,
                              vk::SubpassContents::eInline);
    cmdBuffer.setViewport(
        0,
        vk::Viewport(0.0f, 0.0f, static_cast<float>(offScreenFrameBuf.width),
                     static_cast<float>(offScreenFrameBuf.height), 0.0f, 1.0f));
    cmdBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0),
                                       vk::Extent2D(offScreenFrameBuf.width,
                                                    offScreenFrameBuf.height)));
    cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                           *pipelines.offScreen);
    // offscreen ubo
    cmdBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, *pipelineLayoutOffScreen, 0 /*set 0*/,
        {*descriptorSets.offScreenUboDescriptorSets[currentFrameIndex]},
        nullptr);

    // models
    for (size_t instanceIdx = 0; instanceIdx < modelInstances.size();
         instanceIdx++) {
      const auto& modelInstance = modelInstances[instanceIdx];
      if (!modelInstance.model) {
        continue;
      }
      // dynamic
      cmdBuffer.bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics, *pipelineLayoutOffScreen,
          1 /*set 1*/,
          {*descriptorSets.dynamicUboDescriptorSets[currentFrameIndex]},
          alignedSizeDynamicUboElt * instanceIdx);
      // modelInstance.model->bindBuffers(cmdBuffer);
      modelInstance.model->draw(currentFrameIndex, cmdBuffer,
                                vgeu::RenderFlagBits::kBindImages,
                                *pipelineLayoutOffScreen, 2);
    }

    cmdBuffer.endRenderPass();
  }

  // Image layout transition already done by final layout on attachment.
  // mem availabilty and visilbility.
  {
    // NOTE(kcchoi): attachment final layout -> shader read only optimal
    vk::ImageLayout oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    std::vector<vk::ImageMemoryBarrier> imageMemoryBarriers;
    // Position image
    imageMemoryBarriers.emplace_back(
        vk::AccessFlagBits::eColorAttachmentWrite,
        vk::AccessFlagBits::eShaderRead, oldLayout,
        vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        offScreenFrameBuf.position[currentFrameIndex]->getImage(),
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    // Normal image
    imageMemoryBarriers.emplace_back(
        vk::AccessFlagBits::eColorAttachmentWrite,
        vk::AccessFlagBits::eShaderRead, oldLayout,
        vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        offScreenFrameBuf.normal[currentFrameIndex]->getImage(),
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    // Albedo image
    imageMemoryBarriers.emplace_back(
        vk::AccessFlagBits::eColorAttachmentWrite,
        vk::AccessFlagBits::eShaderRead, oldLayout,
        vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        offScreenFrameBuf.albedo[currentFrameIndex]->getImage(),
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                              vk::PipelineStageFlagBits::eFragmentShader,
                              vk::DependencyFlags{}, nullptr, nullptr,
                              imageMemoryBarriers);
  }
  {
    // eDepthStencilAttachmentOptimal -> validation error in subresurce range
    vk::ImageLayout oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    std::vector<vk::ImageMemoryBarrier> imageMemoryBarriers;
    // depth image
    imageMemoryBarriers.emplace_back(
        vk::AccessFlagBits::eDepthStencilAttachmentWrite,
        vk::AccessFlagBits::eShaderRead, oldLayout,
        vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        offScreenFrameBuf.depth[currentFrameIndex]->getImage(),
        vk::ImageSubresourceRange(
            vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil,
            0, 1, 0, 1));
    cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eLateFragmentTests,
                              vk::PipelineStageFlagBits::eFragmentShader,
                              vk::DependencyFlags{}, nullptr, nullptr,
                              imageMemoryBarriers);
  }

  // second render pass for composition
  // NOTE(kcchoi): no semaphores for explcit synchronizaion.
  {
    std::array<vk::ClearValue, 2> clearValues;
    clearValues[0].color = vk::ClearColorValue(0.5f, 0.5f, 0.5f, 0.5f);
    clearValues[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

    vk::RenderPassBeginInfo renderPassBeginInfo(
        *renderPass, *frameBuffers[currentImageIndex],
        vk::Rect2D(vk::Offset2D(0, 0), swapChainData->swapChainExtent),
        clearValues);

    cmdBuffer.beginRenderPass(renderPassBeginInfo,
                              vk::SubpassContents::eInline);
    cmdBuffer.setViewport(
        0,
        vk::Viewport(0.0f, 0.0f,
                     static_cast<float>(swapChainData->swapChainExtent.width),
                     static_cast<float>(swapChainData->swapChainExtent.height),
                     0.0f, 1.0f));
    cmdBuffer.setScissor(
        0, vk::Rect2D(vk::Offset2D(0, 0), swapChainData->swapChainExtent));
    cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                           *pipelines.composition);
    // composition ubo
    cmdBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, *pipelineLayoutCompoisition,
        0 /*set 0*/, {*descriptorSets.composition[currentFrameIndex]}, nullptr);
    // big triangle covers full screen quad.
    cmdBuffer.draw(3, 1, 0, 0);
    // UI overlay draw
    drawUI(cmdBuffer);
    cmdBuffer.endRenderPass();
  }
  cmdBuffer.end();
}

void VgeExample::draw() {
  {
    // TODO(kcchoi): update base synch primitives
    vk::Result result =
        device.waitForFences(*waitFences[currentFrameIndex], VK_TRUE,
                             std::numeric_limits<uint64_t>::max());
    assert(result != vk::Result::eTimeout && "Timed out: waitFence");
    device.resetFences(*waitFences[currentFrameIndex]);
  }
  prepareFrame();
  // update ubo
  updateUboOffScreen();
  updateUboComposition();
  // cmd buffs
  buildCommandBuffers();
  // offscreen rendering
  // scene rendering
  {
    // TODO(kcchoi): present, render sema
    vk::PipelineStageFlags waitDstStageMask(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    vk::SubmitInfo submitInfo(*presentCompleteSemaphores[currentFrameIndex],
                              waitDstStageMask,
                              *drawCmdBuffers[currentFrameIndex],
                              *renderCompleteSemaphores[currentFrameIndex]);
    queue.submit(submitInfo, *waitFences[currentFrameIndex]);
  }
  submitFrame();
  // queue.waitIdle();  // for synch test only
}

void VgeExample::addModelInstance(ModelInstance&& newInstance) {
  size_t instanceIdx = modelInstances.size();
  modelInstances.push_back(std::move(newInstance));
  instanceMap[newInstance.name].push_back(instanceIdx);
}

const std::vector<size_t>& VgeExample::findInstances(const std::string& name) {
  assert(instanceMap.find(name) != instanceMap.end() &&
         "failed to find instance by name.");
  return instanceMap.at(name);
}

ModelInstance::ModelInstance(ModelInstance&& other) {
  model = other.model;
  simpleModel = other.simpleModel;
  name = other.name;
  isBone = other.isBone;
  animationIndex = other.animationIndex;
  animationTime = other.animationTime;
  transform = other.transform;
}

ModelInstance& ModelInstance::operator=(ModelInstance&& other) {
  model = other.model;
  simpleModel = other.simpleModel;
  name = other.name;
  isBone = other.isBone;
  animationIndex = other.animationIndex;
  animationTime = other.animationTime;
  transform = other.transform;
  return *this;
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()