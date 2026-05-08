#include "soap_bubble.hpp"

#include "vgeu_utils.hpp"

namespace vge {

VgeExample::VgeExample() : VgeBase() { title = "soap_bubble"; }
VgeExample::~VgeExample() {}

void VgeExample::setupCommandLineParser(CLI::App& app) {
  VgeBase::setupCommandLineParser(app);
  app.add_option("--thicknessMin", opts.thicknessMin,
                 "film min thickness (nm)");
  app.add_option("--thicknessMax", opts.thicknessMax,
                 "film max thickness (nm)");
  app.add_option("--n1", opts.n1, "outside refractive index");
  app.add_option("--n2", opts.n2, "film refractive index");
  app.add_option("--n3", opts.n3, "inside refractive index");
  app.add_option("--spectralSamples", opts.spectralSamples,
                 "samples in [380,780]nm");
  app.add_option("--thicknessMode", opts.thicknessMode,
                 "0=Texture, 1=Procedural");
  app.add_option("--gravityStrength", opts.gravityStrength);
  app.add_option("--noiseScale", opts.noiseScale);
  app.add_option("--useAnimation", opts.useAnimation);
  app.add_option("--driftSpeed", opts.driftSpeed);
  app.add_option("--roughness", opts.roughness);
  app.add_option("--rtMode", opts.rtMode, "0=both, 1=R-only, 2=T-only");
  app.add_option("--useSSR", opts.useSSR);
  app.add_option("--refractDepth", opts.refractDepth,
                 "empirical depth along T_dir for SSR sample");
  app.add_option("--iblExposure", opts.iblExposure);
  app.add_option("--iblGamma", opts.iblGamma);
  app.add_option("--useJitter", opts.useJitter);
  app.add_option("--skyboxLod", opts.skyboxLod);
  app.add_option("--model", opts.model, "model: helmet | sphere")
      ->check(CLI::IsMember({"helmet", "sphere"}))
      ->capture_default_str();

  app.add_option_function<std::vector<std::string>>(
         "--bgEnable",
         [this](const std::vector<std::string>& names) {
           // First occurrence resets all to disabled, then enables only named.
           for (auto& bg : opts.backgrounds) bg.enabled = false;
           for (const auto& name : names) {
             for (auto& bg : opts.backgrounds) {
               auto slash = bg.path.find_last_of('/');
               std::string leaf = (slash == std::string::npos)
                                      ? bg.path
                                      : bg.path.substr(slash + 1);
               auto dot = leaf.find_last_of('.');
               std::string stem =
                   (dot == std::string::npos) ? leaf : leaf.substr(0, dot);
               if (leaf == name || stem == name) bg.enabled = true;
             }
           }
         },
         "enable only the named background instances; "
         "names match leaf filename or stem (e.g. apple, Fox, "
         "smooth_sphere, dutch_ship_medium_1k)")
      ->take_all();
}

void VgeExample::setOptions(const std::optional<Options>& o) {
  if (o) opts = *o;
}

void VgeExample::initVulkan() {
  camera.setViewTarget(glm::vec3{0.f, 0.f, -3.f}, glm::vec3{0.f, 0.f, 0.f});
  camera.setPerspectiveProjection(
      glm::radians(60.f),
      static_cast<float>(width) / static_cast<float>(height), 0.1f, 256.f);
  VgeBase::initVulkan();
}
void VgeExample::getEnabledExtensions() {}
void VgeExample::getEnabledFeatures() {
  enabledFeatures.samplerAnisotropy = VK_TRUE;
}

void VgeExample::prepare() {
  VgeBase::prepare();
  loadAssets();
  prepareIBL();
  prepareOffscreen();
  prepareUniformBuffers();
  setupDescriptors();
  preparePipelines();
  prepared = true;
}

void VgeExample::loadAssets() {
  bubbleModel = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  vgeu::FileLoadingFlags loadFlags =
      vgeu::FileLoadingFlagBits::kPreMultiplyVertexColors |
      vgeu::FileLoadingFlagBits::kPreTransformVertices |
      vgeu::FileLoadingFlagBits::kFlipY;
  std::string modelPath =
      (opts.model == "sphere")
          ? "/models/sphere/pirate-gold/pirate-gold-pbr.gltf"
          : "/models/DamagedHelmet/glTF/DamagedHelmet.gltf";
  bubbleModel->loadFromFile(getAssetsPath() + modelPath, loadFlags);

  heightTexture = std::make_unique<vgeu::Texture2D>(
      getAssetsPath() + "/models/sphere/pirate-gold/pirate-gold_height.png",
      device, globalAllocator->getAllocator(), queue, commandPool, true);

  // Background scene models
  vgeu::FileLoadingFlags bgLoadFlags =
      vgeu::FileLoadingFlagBits::kPreMultiplyVertexColors |
      vgeu::FileLoadingFlagBits::kPreTransformVertices |
      vgeu::FileLoadingFlagBits::kFlipY;
  bgModels.reserve(opts.backgrounds.size());
  for (const auto& inst : opts.backgrounds) {
    auto m = std::make_shared<vgeu::glTF::Model>(
        device, globalAllocator->getAllocator(), queue, commandPool,
        MAX_CONCURRENT_FRAMES);
    m->loadFromFile(getAssetsPath() + inst.path, bgLoadFlags);
    bgModels.push_back(std::move(m));
  }
}

void VgeExample::destroyOffscreen() {
  // Tear down in reverse order; clear vectors so emplace later starts clean.
  offscreenFramebuffers.clear();
  offscreenColors.clear();
  offscreenDepths.clear();
}

void VgeExample::prepareOffscreen() {
  destroyOffscreen();

  vk::Extent2D extent = swapChainData->swapChainExtent;
  vk::Format colorFmt = swapChainData->colorFormat;

  // Create per-frame color + depth images.
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
    offscreenColors.push_back(std::make_unique<vgeu::VgeuImage>(
        device, globalAllocator->getAllocator(), colorFmt, extent,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment |
            vk::ImageUsageFlagBits::eSampled,
        vk::ImageLayout::eUndefined, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        vk::ImageAspectFlagBits::eColor, 1));

    offscreenDepths.push_back(std::make_unique<vgeu::VgeuImage>(
        device, globalAllocator->getAllocator(), depthFormat, extent,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::ImageLayout::eUndefined, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        vk::ImageAspectFlagBits::eDepth, 1));
  }

  // Render pass: color (eShaderReadOnly <-> eColorAttachment cycle) + depth.
  {
    std::array<vk::AttachmentDescription, 2> attachments;
    attachments[0] = vk::AttachmentDescription(
        {}, colorFmt, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear,
        vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare,
        vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal);
    attachments[1] = vk::AttachmentDescription(
        {}, depthFormat, vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare,
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::AttachmentReference colorRef(0,
                                     vk::ImageLayout::eColorAttachmentOptimal);
    vk::AttachmentReference depthRef(
        1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {},
                                   colorRef, {}, &depthRef);

    std::array<vk::SubpassDependency, 2> deps;
    deps[0] = vk::SubpassDependency(
        VK_SUBPASS_EXTERNAL, 0, vk::PipelineStageFlagBits::eFragmentShader,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::AccessFlagBits::eShaderRead,
        vk::AccessFlagBits::eColorAttachmentWrite);
    deps[1] =
        vk::SubpassDependency(0, VK_SUBPASS_EXTERNAL,
                              vk::PipelineStageFlagBits::eColorAttachmentOutput,
                              vk::PipelineStageFlagBits::eFragmentShader,
                              vk::AccessFlagBits::eColorAttachmentWrite,
                              vk::AccessFlagBits::eShaderRead);

    offscreenRenderPass = vk::raii::RenderPass(
        device, vk::RenderPassCreateInfo({}, attachments, subpass, deps));
  }

  // Per-frame framebuffer (one color + one depth view).
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
    std::array<vk::ImageView, 2> views{*offscreenColors[i]->getImageView(),
                                       *offscreenDepths[i]->getImageView()};
    offscreenFramebuffers.push_back(vk::raii::Framebuffer(
        device, vk::FramebufferCreateInfo({}, *offscreenRenderPass, views,
                                          extent.width, extent.height, 1)));
  }

  // Sampler for the bubble pass to read offscreen color.
  sceneColorSampler = vk::raii::Sampler(
      device,
      vk::SamplerCreateInfo({}, vk::Filter::eLinear, vk::Filter::eLinear,
                            vk::SamplerMipmapMode::eNearest,
                            vk::SamplerAddressMode::eClampToEdge,
                            vk::SamplerAddressMode::eClampToEdge,
                            vk::SamplerAddressMode::eClampToEdge, 0.f, false,
                            1.f, false, vk::CompareOp::eAlways, 0.f, 0.f,
                            vk::BorderColor::eFloatOpaqueBlack, false));

  // Init barrier: bring color images to ShaderReadOnly so the very first
  // offscreen render pass's initialLayout assumption holds. Depth attachments
  // are transitioned by the render pass itself (initialLayout = Undefined).
  vgeu::oneTimeSubmit(
      device, commandPool, queue, [&](const vk::raii::CommandBuffer& cmd) {
        for (auto& img : offscreenColors) {
          vk::ImageMemoryBarrier b(
              vk::AccessFlags{}, vk::AccessFlagBits::eShaderRead,
              vk::ImageLayout::eUndefined,
              vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED,
              VK_QUEUE_FAMILY_IGNORED, img->getImage(),
              vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1,
                                        0, 1));
          cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                              vk::PipelineStageFlagBits::eFragmentShader, {},
                              nullptr, nullptr, b);
        }
      });
}

void VgeExample::prepareIBL() {
  iblConfig.hdrPath =
      getAssetsPath() + "/textures/hdr/tree_lined_driveway_4k.hdr";
  iblConfig.commonShadersPath = getShadersPath() + "/common";
  iblConfig.useJitter = opts.useJitter;
  iblBaker = std::make_unique<vgeu::IBLBaker>(
      device, globalAllocator->getAllocator(), queue, commandPool);
  iblBaker->bake(iblConfig);
}

void VgeExample::prepareUniformBuffers() {
  uniformBuffers.resize(MAX_CONCURRENT_FRAMES);
  for (auto& ub : uniformBuffers) {
    ub.globals = std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(GlobalsUbo), 1,
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    ub.bubbleParams = std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(BubbleParamsUbo), 1,
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
  }
}

void VgeExample::setupDescriptors() {
  // Pool sizes:
  //   UBOs: 2 (globals, bubbleParams) * MAX_CONCURRENT_FRAMES
  //         + skybox UBO (1 * MAX_CONCURRENT_FRAMES)
  //   CIS:  height texture (1) + env (MAX_CONCURRENT_FRAMES)
  //         + skybox cubemap (MAX_CONCURRENT_FRAMES)
  std::vector<vk::DescriptorPoolSize> poolSizes{
      {vk::DescriptorType::eUniformBuffer,
       3u * MAX_CONCURRENT_FRAMES /*globals + params + skybox*/},
      {vk::DescriptorType::eCombinedImageSampler, 1u +
                                                      4u *
                                                          MAX_CONCURRENT_FRAMES /*height + env + skybox + bgIrr + scene*/}};
  // Set count: globals + params + env + bgIrr + scene (per-frame) +
  // height(1) + skybox(per-frame)
  uint32_t maxSets = 5u * MAX_CONCURRENT_FRAMES + 1u + MAX_CONCURRENT_FRAMES;
  descriptorPool = vk::raii::DescriptorPool(
      device, vk::DescriptorPoolCreateInfo(
                  vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, maxSets,
                  poolSizes));

  // set=0 Globals UBO (vert + frag)
  {
    vk::DescriptorSetLayoutBinding b(
        0, vk::DescriptorType::eUniformBuffer, 1,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
    globalsSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo({}, b));
  }
  // set=1 BubbleParams UBO (frag)
  {
    vk::DescriptorSetLayoutBinding b(0, vk::DescriptorType::eUniformBuffer, 1,
                                     vk::ShaderStageFlagBits::eFragment);
    bubbleParamsSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo({}, b));
  }
  // set=2 Height texture (frag)
  {
    vk::DescriptorSetLayoutBinding b(0,
                                     vk::DescriptorType::eCombinedImageSampler,
                                     1, vk::ShaderStageFlagBits::eFragment);
    heightTexSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo({}, b));
  }
  // set=3 prefiltered cubemap (frag)
  {
    vk::DescriptorSetLayoutBinding b(0,
                                     vk::DescriptorType::eCombinedImageSampler,
                                     1, vk::ShaderStageFlagBits::eFragment);
    envSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo({}, b));
  }

  globalsDescSets.reserve(MAX_CONCURRENT_FRAMES);
  bubbleParamsDescSets.reserve(MAX_CONCURRENT_FRAMES);
  envDescSets.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
    // globals
    globalsDescSets.push_back(
        std::move(vk::raii::DescriptorSets(
                      device, vk::DescriptorSetAllocateInfo(*descriptorPool,
                                                            *globalsSetLayout))
                      .front()));
    vk::DescriptorBufferInfo globalsBI(uniformBuffers[i].globals->getBuffer(),
                                       0, sizeof(GlobalsUbo));
    device.updateDescriptorSets(
        vk::WriteDescriptorSet(*globalsDescSets[i], 0, 0,
                               vk::DescriptorType::eUniformBuffer, {},
                               globalsBI),
        nullptr);

    // bubbleParams
    bubbleParamsDescSets.push_back(
        std::move(vk::raii::DescriptorSets(
                      device, vk::DescriptorSetAllocateInfo(
                                  *descriptorPool, *bubbleParamsSetLayout))
                      .front()));
    vk::DescriptorBufferInfo paramsBI(
        uniformBuffers[i].bubbleParams->getBuffer(), 0,
        sizeof(BubbleParamsUbo));
    device.updateDescriptorSets(
        vk::WriteDescriptorSet(*bubbleParamsDescSets[i], 0, 0,
                               vk::DescriptorType::eUniformBuffer, {},
                               paramsBI),
        nullptr);

    // env (prefiltered cubemap)
    envDescSets.push_back(std::move(
        vk::raii::DescriptorSets(device, vk::DescriptorSetAllocateInfo(
                                             *descriptorPool, *envSetLayout))
            .front()));
    vk::DescriptorImageInfo envInfo(*iblBaker->iblSampler(),
                                    *iblBaker->prefilteredMap().getImageView(),
                                    vk::ImageLayout::eShaderReadOnlyOptimal);
    device.updateDescriptorSets(
        vk::WriteDescriptorSet(*envDescSets[i], 0, 0,
                               vk::DescriptorType::eCombinedImageSampler,
                               envInfo),
        nullptr);
  }

  // Height texture (single set; texture is per-application, not per-frame)
  heightTexDescSet =
      std::move(vk::raii::DescriptorSets(
                    device, vk::DescriptorSetAllocateInfo(*descriptorPool,
                                                          *heightTexSetLayout))
                    .front());
  {
    vk::DescriptorImageInfo heightInfo(*heightTexture->sampler,
                                       heightTexture->descriptorInfo.imageView,
                                       vk::ImageLayout::eShaderReadOnlyOptimal);
    device.updateDescriptorSets(
        vk::WriteDescriptorSet(*heightTexDescSet, 0, 0,
                               vk::DescriptorType::eCombinedImageSampler,
                               heightInfo),
        nullptr);
  }

  // bg pass set=1: irradiance map (frag)
  {
    vk::DescriptorSetLayoutBinding b(0,
                                     vk::DescriptorType::eCombinedImageSampler,
                                     1, vk::ShaderStageFlagBits::eFragment);
    bgIrradianceSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo({}, b));
  }

  bgIrradianceDescSets.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
    bgIrradianceDescSets.push_back(
        std::move(vk::raii::DescriptorSets(
                      device, vk::DescriptorSetAllocateInfo(
                                  *descriptorPool, *bgIrradianceSetLayout))
                      .front()));
    vk::DescriptorImageInfo info(*iblBaker->iblSampler(),
                                 *iblBaker->irradianceMap().getImageView(),
                                 vk::ImageLayout::eShaderReadOnlyOptimal);
    device.updateDescriptorSets(
        vk::WriteDescriptorSet(*bgIrradianceDescSets[i], 0, 0,
                               vk::DescriptorType::eCombinedImageSampler, info),
        nullptr);
  }

  // bubble pass set=4: scene color (frag)
  {
    vk::DescriptorSetLayoutBinding b(0,
                                     vk::DescriptorType::eCombinedImageSampler,
                                     1, vk::ShaderStageFlagBits::eFragment);
    sceneColorSetLayout = vk::raii::DescriptorSetLayout(
        device, vk::DescriptorSetLayoutCreateInfo({}, b));
  }

  sceneColorDescSets.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
    sceneColorDescSets.push_back(
        std::move(vk::raii::DescriptorSets(
                      device, vk::DescriptorSetAllocateInfo(
                                  *descriptorPool, *sceneColorSetLayout))
                      .front()));
    vk::DescriptorImageInfo info(*sceneColorSampler,
                                 *offscreenColors[i]->getImageView(),
                                 vk::ImageLayout::eShaderReadOnlyOptimal);
    device.updateDescriptorSets(
        vk::WriteDescriptorSet(*sceneColorDescSets[i], 0, 0,
                               vk::DescriptorType::eCombinedImageSampler, info),
        nullptr);
  }

  skybox = std::make_unique<vgeu::Skybox>(
      device, pipelineCache, descriptorPool, renderPass,
      iblConfig.commonShadersPath, *iblBaker, MAX_CONCURRENT_FRAMES);
}

void VgeExample::preparePipelines() {
  std::array<vk::DescriptorSetLayout, 5> setLayouts{
      *globalsSetLayout, *bubbleParamsSetLayout, *heightTexSetLayout,
      *envSetLayout, *sceneColorSetLayout};
  bubblePipelineLayout = vk::raii::PipelineLayout(
      device, vk::PipelineLayoutCreateInfo({}, setLayouts));

  auto vertCode =
      vgeu::readFile(getShadersPath() + "/soap_bubble/bubble.vert.spv");
  auto fragCode =
      vgeu::readFile(getShadersPath() + "/soap_bubble/bubble.frag.spv");
  auto vertSM = vgeu::createShaderModule(device, vertCode);
  auto fragSM = vgeu::createShaderModule(device, fragCode);

  std::array<vk::PipelineShaderStageCreateInfo, 2> stages{
      vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                                        *vertSM, "main"),
      vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment,
                                        *fragSM, "main"),
  };

  auto vertexInputSCI = vgeu::glTF::Vertex::getPipelineVertexInputState(
      {vgeu::glTF::VertexComponent::kPosition, vgeu::glTF::VertexComponent::kUV,
       vgeu::glTF::VertexComponent::kColor,
       vgeu::glTF::VertexComponent::kNormal,
       vgeu::glTF::VertexComponent::kTangent});

  vk::PipelineInputAssemblyStateCreateInfo iaCI(
      {}, vk::PrimitiveTopology::eTriangleList);

  vk::PipelineViewportStateCreateInfo vpCI({}, 1, nullptr, 1, nullptr);

  vk::PipelineRasterizationStateCreateInfo rsCI(
      {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eFront,
      vk::FrontFace::eCounterClockwise, false, 0.f, 0.f, 0.f, 1.f);

  vk::PipelineMultisampleStateCreateInfo msCI({}, vk::SampleCountFlagBits::e1);

  // Depth test on, depth write OFF (transparent surfaces in Task 21+)
  vk::PipelineDepthStencilStateCreateInfo dsCI(
      {}, true /*depthTest*/, true /*depthWrite*/, vk::CompareOp::eLessOrEqual);

  // Blend OFF: bubble outputs alpha=1 with R/T compositing already done in
  // shader; no alphablend fudge needed.
  vk::PipelineColorBlendAttachmentState cbAtt(
      false, vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha,
      vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero,
      vk::BlendOp::eAdd,
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
  vk::PipelineColorBlendStateCreateInfo cbCI({}, false, vk::LogicOp::eClear,
                                             cbAtt);

  std::array<vk::DynamicState, 2> dynStates{vk::DynamicState::eViewport,
                                            vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dynCI({}, dynStates);

  vk::GraphicsPipelineCreateInfo pipelineCI(
      {}, stages, &vertexInputSCI, &iaCI, nullptr, &vpCI, &rsCI, &msCI, &dsCI,
      &cbCI, &dynCI, *bubblePipelineLayout, *renderPass);
  bubblePipeline = vk::raii::Pipeline(device, pipelineCache, pipelineCI);

  // Background pipeline
  std::array<vk::DescriptorSetLayout, 2> bgSetLayouts{*globalsSetLayout,
                                                      *bgIrradianceSetLayout};
  vk::PushConstantRange bgPcRange(
      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
      sizeof(BgPushConstant));
  bgPipelineLayout = vk::raii::PipelineLayout(
      device, vk::PipelineLayoutCreateInfo({}, bgSetLayouts, bgPcRange));

  auto bgVertCode =
      vgeu::readFile(getShadersPath() + "/soap_bubble/bg.vert.spv");
  auto bgFragCode =
      vgeu::readFile(getShadersPath() + "/soap_bubble/bg.frag.spv");
  auto bgVertSM = vgeu::createShaderModule(device, bgVertCode);
  auto bgFragSM = vgeu::createShaderModule(device, bgFragCode);

  std::array<vk::PipelineShaderStageCreateInfo, 2> bgStages{
      vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                                        *bgVertSM, "main"),
      vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment,
                                        *bgFragSM, "main"),
  };

  vk::PipelineRasterizationStateCreateInfo bgRsCI(
      {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eFront,
      vk::FrontFace::eCounterClockwise, false, 0.f, 0.f, 0.f, 1.f);

  vk::PipelineDepthStencilStateCreateInfo bgDsCI(
      {}, true /*depthTest*/, true /*depthWrite*/, vk::CompareOp::eLessOrEqual);

  vk::PipelineColorBlendAttachmentState bgCbAtt(
      false, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
      vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
  vk::PipelineColorBlendStateCreateInfo bgCbCI({}, false, vk::LogicOp::eClear,
                                               bgCbAtt);

  vk::GraphicsPipelineCreateInfo bgPipelineCI(
      {}, bgStages, &vertexInputSCI, &iaCI, nullptr, &vpCI, &bgRsCI, &msCI,
      &bgDsCI, &bgCbCI, &dynCI, *bgPipelineLayout, *renderPass);
  bgPipeline = vk::raii::Pipeline(device, pipelineCache, bgPipelineCI);
}

static glm::mat4 buildModelMatrix(const BgInstance& inst) {
  glm::mat4 m{1.f};
  m = glm::translate(m, inst.translate);
  m = glm::rotate(m, glm::radians(inst.eulerDeg.y), glm::vec3(0, 1, 0));
  m = glm::rotate(m, glm::radians(inst.eulerDeg.x), glm::vec3(1, 0, 0));
  m = glm::rotate(m, glm::radians(inst.eulerDeg.z), glm::vec3(0, 0, 1));
  m = glm::scale(m, glm::vec3(inst.scale));
  return m;
}

void VgeExample::buildCommandBuffers() {
  const auto& cmd = drawCmdBuffers[currentFrameIndex];
  cmd.begin({});

  std::array<vk::ClearValue, 2> clearValues;
  clearValues[0].color = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
  clearValues[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

  // Helper to record skybox + background draws into the currently-bound
  // render pass. Reused for both the offscreen pass and the swapchain pass
  // when SSR is on.
  auto recordSkyboxAndBg = [&]() {
    skybox->draw(cmd, currentFrameIndex, camera.getView(),
                 camera.getProjection(), opts.skyboxLod);

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *bgPipeline);
    std::array<vk::DescriptorSet, 2> bgDescSets{
        *globalsDescSets[currentFrameIndex],
        *bgIrradianceDescSets[currentFrameIndex]};
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *bgPipelineLayout,
                           0, bgDescSets, nullptr);
    for (size_t i = 0; i < opts.backgrounds.size(); ++i) {
      if (!opts.backgrounds[i].enabled) continue;
      BgPushConstant pc{
          buildModelMatrix(opts.backgrounds[i]),
          glm::vec4(opts.backgrounds[i].baseColor, 1.0f),
      };
      cmd.pushConstants<BgPushConstant>(
          *bgPipelineLayout,
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
          0, pc);
      bgModels[i]->draw(currentFrameIndex, cmd);
    }
  };

  // Offscreen pass (only when SSR is on).
  if (opts.useSSR) {
    vk::RenderPassBeginInfo offBegin(*offscreenRenderPass,
                                     *offscreenFramebuffers[currentFrameIndex],
                                     {{0, 0}, {width, height}}, clearValues);
    cmd.beginRenderPass(offBegin, vk::SubpassContents::eInline);
    cmd.setViewport(
        0, vk::Viewport(0.f, 0.f, (float)width, (float)height, 0.f, 1.f));
    cmd.setScissor(0, vk::Rect2D({0, 0}, {width, height}));
    recordSkyboxAndBg();
    cmd.endRenderPass();
  }

  // Swapchain pass.
  vk::RenderPassBeginInfo rpBegin(*renderPass, *frameBuffers[currentImageIndex],
                                  {{0, 0}, {width, height}}, clearValues);
  cmd.beginRenderPass(rpBegin, vk::SubpassContents::eInline);

  cmd.setViewport(
      0, vk::Viewport(0.f, 0.f, (float)width, (float)height, 0.f, 1.f));
  cmd.setScissor(0, vk::Rect2D({0, 0}, {width, height}));

  recordSkyboxAndBg();

  // bubble pass
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *bubblePipeline);
  std::array<vk::DescriptorSet, 5> descSets{
      *globalsDescSets[currentFrameIndex],
      *bubbleParamsDescSets[currentFrameIndex], *heightTexDescSet,
      *envDescSets[currentFrameIndex], *sceneColorDescSets[currentFrameIndex]};
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                         *bubblePipelineLayout, 0, descSets, nullptr);
  bubbleModel->draw(currentFrameIndex, cmd);

  drawUI(cmd);
  cmd.endRenderPass();
  cmd.end();
}

void VgeExample::draw() {
  {
    vk::Result result = device.waitForFences(*waitFences[currentFrameIndex],
                                             VK_TRUE, UINT64_MAX);
    assert(result != vk::Result::eTimeout);
    device.resetFences(*waitFences[currentFrameIndex]);
  }
  prepareFrame();
  buildCommandBuffers();
  {
    vk::PipelineStageFlags waitStage(
        vk::PipelineStageFlagBits::eColorAttachmentOutput);
    vk::SubmitInfo submitInfo(*presentCompleteSemaphores[currentFrameIndex],
                              waitStage, *drawCmdBuffers[currentFrameIndex],
                              *renderCompleteSemaphores[currentFrameIndex]);
    queue.submit(submitInfo, *waitFences[currentFrameIndex]);
  }
  submitFrame();
}

void VgeExample::render() {
  if (!prepared) return;
  updateGlobalsUbo();
  updateBubbleParamsUbo();
  draw();
}

void VgeExample::viewChanged() {}

void VgeExample::onUpdateUIOverlay() {
  if (ImGui::CollapsingHeader("Model", ImGuiTreeNodeFlags_DefaultOpen)) {
    const char* items[] = {"helmet", "sphere"};
    int idx = (opts.model == "sphere") ? 1 : 0;
    if (ImGui::Combo("model", &idx, items, IM_ARRAYSIZE(items))) {
      opts.model = items[idx];
      restart = true;
    }
  }
  if (ImGui::CollapsingHeader("Thin Film", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::SliderFloat("thicknessMin (nm)", &opts.thicknessMin, 0.f, 2000.f);
    ImGui::SliderFloat("thicknessMax (nm)", &opts.thicknessMax, 0.f, 2000.f);
    ImGui::SliderFloat("n1 (outside)", &opts.n1, 1.0f, 2.5f);
    ImGui::SliderFloat("n2 (film)", &opts.n2, 1.0f, 2.5f);
    ImGui::SliderFloat("n3 (inside)", &opts.n3, 1.0f, 2.5f);

    static int32_t sampleIdx = 1;
    const char* sampleOpts[] = {"8", "16", "32", "64"};
    if (ImGui::Combo("spectralSamples", &sampleIdx, sampleOpts,
                     IM_ARRAYSIZE(sampleOpts))) {
      const int32_t values[] = {8, 16, 32, 64};
      opts.spectralSamples = values[sampleIdx];
    }
  }

  if (ImGui::CollapsingHeader("Thickness Source",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::RadioButton("Texture", &opts.thicknessMode, 0);
    ImGui::SameLine();
    ImGui::RadioButton("Procedural", &opts.thicknessMode, 1);
    if (opts.thicknessMode == 1) {
      ImGui::SliderFloat("gravityStrength", &opts.gravityStrength, 0.f, 5.f);
      ImGui::SliderFloat("noiseScale", &opts.noiseScale, 0.1f, 10.f);
    }
  }

  if (ImGui::CollapsingHeader("Animation")) {
    ImGui::Checkbox("useAnimation", &opts.useAnimation);
    if (opts.useAnimation) {
      ImGui::SliderFloat("driftSpeed", &opts.driftSpeed, 0.f, 2.f);
    }
  }

  if (ImGui::CollapsingHeader("Surface", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::SliderFloat("roughness", &opts.roughness, 0.f, 1.f);
  }

  if (ImGui::CollapsingHeader("R/T Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::RadioButton("Both", &opts.rtMode, 0);
    ImGui::SameLine();
    ImGui::RadioButton("R-only", &opts.rtMode, 1);
    ImGui::SameLine();
    ImGui::RadioButton("T-only", &opts.rtMode, 2);
  }

  if (ImGui::CollapsingHeader("Refraction", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Use Screen-Space Refraction (T)", &opts.useSSR);
    if (opts.useSSR) {
      ImGui::SliderFloat("refractDepth", &opts.refractDepth, 0.0f, 2.0f);
    }
  }

  if (ImGui::CollapsingHeader("IBL / Env")) {
    ImGui::SliderFloat("iblExposure", &opts.iblExposure, 0.f, 10.f);
    ImGui::SliderFloat("iblGamma", &opts.iblGamma, 1.f, 3.f);
    ImGui::SliderFloat("skyboxLod", &opts.skyboxLod, 0.f, 9.f);
    if (ImGui::Checkbox("useJitter (rebakes)", &opts.useJitter)) {
      device.waitIdle();
      iblConfig.useJitter = opts.useJitter;
      iblBaker->rebakeFiltering(iblConfig);
      // Re-bind both prefilteredMap (bubble pass) and irradianceMap (bg pass)
      // since rebakeFiltering rebuilds both image views.
      for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
        vk::DescriptorImageInfo envInfo(
            *iblBaker->iblSampler(), *iblBaker->prefilteredMap().getImageView(),
            vk::ImageLayout::eShaderReadOnlyOptimal);
        device.updateDescriptorSets(
            vk::WriteDescriptorSet(*envDescSets[i], 0, 0,
                                   vk::DescriptorType::eCombinedImageSampler,
                                   envInfo),
            nullptr);

        vk::DescriptorImageInfo irrInfo(
            *iblBaker->iblSampler(), *iblBaker->irradianceMap().getImageView(),
            vk::ImageLayout::eShaderReadOnlyOptimal);
        device.updateDescriptorSets(
            vk::WriteDescriptorSet(*bgIrradianceDescSets[i], 0, 0,
                                   vk::DescriptorType::eCombinedImageSampler,
                                   irrInfo),
            nullptr);
      }
    }
  }

  if (ImGui::CollapsingHeader("Background", ImGuiTreeNodeFlags_DefaultOpen)) {
    for (size_t i = 0; i < opts.backgrounds.size(); ++i) {
      ImGui::PushID(static_cast<int>(i));
      const auto& path = opts.backgrounds[i].path;
      auto slash = path.find_last_of('/');
      std::string label =
          (slash == std::string::npos) ? path : path.substr(slash + 1);
      ImGui::Checkbox(label.c_str(), &opts.backgrounds[i].enabled);
      ImGui::PopID();
    }
  }

  if (ImGui::CollapsingHeader("Debug")) {
    glm::vec3 cp = camera.getPosition();
    ImGui::Text("camera: %.2f, %.2f, %.2f", cp.x, cp.y, cp.z);
    ImGui::Checkbox("showThicknessHeatmap", &opts.showThicknessHeatmap);
    ImGui::Checkbox("showFresnelOnly", &opts.showFresnelOnly);
    ImGui::Checkbox("showNormal", &opts.showNormal);
  }
}

void VgeExample::updateGlobalsUbo() {
  globalsUbo.view = camera.getView();
  globalsUbo.projection = camera.getProjection();
  if (opts.model == "helmet") {
    // DamagedHelmet ships in Y-up convention; same rotations as the pbr
    // example.
    glm::vec3 up{0.f, -1.f, 0.f};
    glm::vec3 right{1.f, 0.f, 0.f};
    globalsUbo.model = glm::rotate(glm::mat4{1.f}, glm::radians(90.f), up);
    globalsUbo.model =
        glm::rotate(globalsUbo.model, glm::radians(-90.f), right);
  } else {
    globalsUbo.model = glm::mat4{1.f};
  }
  globalsUbo.viewPos = glm::vec4(camera.getPosition(), 1.0f);
  std::memcpy(uniformBuffers[currentFrameIndex].globals->getMappedData(),
              &globalsUbo, sizeof(GlobalsUbo));
}

void VgeExample::updateBubbleParamsUbo() {
  bubbleParamsUbo.thicknessMin = opts.thicknessMin;
  bubbleParamsUbo.thicknessMax = opts.thicknessMax;
  bubbleParamsUbo.n1 = opts.n1;
  bubbleParamsUbo.n2 = opts.n2;
  bubbleParamsUbo.n3 = opts.n3;
  bubbleParamsUbo.spectralSamples = opts.spectralSamples;
  bubbleParamsUbo.thicknessMode = opts.thicknessMode;
  bubbleParamsUbo.gravityStrength = opts.gravityStrength;
  bubbleParamsUbo.noiseScale = opts.noiseScale;
  bubbleParamsUbo.useAnimation = opts.useAnimation ? 1 : 0;
  bubbleParamsUbo.driftSpeed = opts.driftSpeed;
  bubbleParamsUbo.roughness = opts.roughness;
  bubbleParamsUbo.iblExposure = opts.iblExposure;
  bubbleParamsUbo.iblGamma = opts.iblGamma;
  bubbleParamsUbo.time = static_cast<float>(timer);
  bubbleParamsUbo.rtMode = opts.rtMode;
  bubbleParamsUbo.showThicknessHeatmap = opts.showThicknessHeatmap ? 1 : 0;
  bubbleParamsUbo.showFresnelOnly = opts.showFresnelOnly ? 1 : 0;
  bubbleParamsUbo.showNormal = opts.showNormal ? 1 : 0;
  bubbleParamsUbo.useSSR = opts.useSSR ? 1 : 0;
  bubbleParamsUbo.refractDepth = opts.refractDepth;
  std::memcpy(uniformBuffers[currentFrameIndex].bubbleParams->getMappedData(),
              &bubbleParamsUbo, sizeof(BubbleParamsUbo));
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()
