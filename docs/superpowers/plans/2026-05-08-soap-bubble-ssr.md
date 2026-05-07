# Soap Bubble — Screen-Space Refraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional screen-space refraction (SSR) mode to `soap_bubble`: when enabled, the bubble shader's T-direction sample comes from the rendered scene color (refracted along an empirical offset), letting the apple/fox/sphere/dutch_ship background instances appear refracted through the bubble. ImGui toggles instantly between SSR and cubemap-T.

**Architecture:** Add a per-frame offscreen color+depth attachment that holds skybox+bg rendered once before the swapchain pass. Bubble pipeline grows from 4 to 5 descriptor sets (set=4 = sceneColor sampler2D bound to current frame's offscreen view). The bubble fragment shader branches on `params.useSSR`: ON → project `worldPos + T_dir * refractDepth` to NDC and sample sceneColor at that UV (cubemap fallback when UV out of frame); OFF → existing cubemap-T behavior. `buildCommandBuffers` runs the offscreen pass only when `opts.useSSR` is true; both render-pass setups are alive at all times so the toggle is per-frame and instant.

**Tech Stack:** Vulkan-HPP RAII, VMA via `vgeu::VgeuImage`, GLSL 450, ImGui, CLI11, MinGW + Ninja, clang-format.

**Reference Spec:** `docs/superpowers/specs/2026-05-08-soap-bubble-ssr-design.md`

---

## Working environment

- All bash commands MUST use the `rtk` prefix (e.g. `rtk git add`, `rtk clang-format`, `rtk ./mingwBuild.bat Debug`).
- All `.cpp/.hpp` edits MUST be passed through `rtk clang-format -i <file>` before commit.
- Build: `rtk ./mingwBuild.bat Debug` (cd to repo root). Produces `build/soap_bubble.exe` and `*.spv` next to each shader source. `.spv` files are gitignored — never `git add` them.
- Run: `rtk ./build/soap_bubble.exe --model sphere` (validation layer on for Debug). Zero new VUIDs is the bar.
- Working directory throughout: repo root `C:\Users\rlckd\Desktop\kc\Vulkan-Graphics-Example`.

---

## File map

| Status | Path | Role |
|---|---|---|
| **Modify** | `shaders/soap_bubble/bubble.frag` | Add `set=4 sceneColor` binding, `useSSR`/`refractDepth` UBO fields, SSR branch in T composite |
| **Modify** | `src/examples/soap_bubble/soap_bubble.hpp` | Options `useSSR`/`refractDepth`, BubbleParamsUbo same fields, offscreen resource handles, bg defaults flipped to true |
| **Modify** | `src/examples/soap_bubble/soap_bubble.cpp` | `prepareOffscreen()`, descriptor / pipeline-layout growth, ImGui Refraction group, CLI options, frame branching, `windowResized` override |

No new shader files; no `vgeu_*` modifications.

---

# Stage 1 — Offscreen color + depth + sampler + render pass

Goal: per-frame offscreen attachments and a render pass that draws skybox+bg into them. Built and ready to use, but no shader/pipeline yet samples them. Visual output unchanged after this stage.

---

## Task 1: Offscreen infrastructure

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.hpp`
- Modify: `src/examples/soap_bubble/soap_bubble.cpp`

- [ ] **Step 1: Add offscreen handles to `soap_bubble.hpp`**

In `class VgeExample`, just before the existing `// Background pipeline / descriptor handles` block (which contains `bgIrradianceSetLayout` etc.), insert:

```cpp
  // Offscreen attachments for screen-space refraction.
  // Per-frame so frame N+1 doesn't race frame N's sample.
  std::vector<std::unique_ptr<vgeu::VgeuImage>> offscreenColors;
  std::vector<std::unique_ptr<vgeu::VgeuImage>> offscreenDepths;
  std::vector<vk::raii::Framebuffer> offscreenFramebuffers;
  vk::raii::RenderPass offscreenRenderPass = nullptr;
  vk::raii::Sampler sceneColorSampler = nullptr;
```

Also add the method declaration in the public section, near `prepareIBL`:

```cpp
  void prepareOffscreen();
  void destroyOffscreen();
```

- [ ] **Step 2: Implement `prepareOffscreen()` and `destroyOffscreen()` in `soap_bubble.cpp`**

In `src/examples/soap_bubble/soap_bubble.cpp`, just before `void VgeExample::prepareIBL() {`, insert these two methods:

```cpp
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
        {}, colorFmt, vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::ImageLayout::eShaderReadOnlyOptimal);
    attachments[1] = vk::AttachmentDescription(
        {}, depthFormat, vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare,
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::AttachmentReference colorRef(
        0, vk::ImageLayout::eColorAttachmentOptimal);
    vk::AttachmentReference depthRef(
        1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {},
                                   colorRef, {}, &depthRef);

    std::array<vk::SubpassDependency, 2> deps;
    deps[0] = vk::SubpassDependency(
        VK_SUBPASS_EXTERNAL, 0,
        vk::PipelineStageFlagBits::eFragmentShader,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::AccessFlagBits::eShaderRead,
        vk::AccessFlagBits::eColorAttachmentWrite);
    deps[1] = vk::SubpassDependency(
        0, VK_SUBPASS_EXTERNAL,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eFragmentShader,
        vk::AccessFlagBits::eColorAttachmentWrite,
        vk::AccessFlagBits::eShaderRead);

    offscreenRenderPass = vk::raii::RenderPass(
        device, vk::RenderPassCreateInfo({}, attachments, subpass, deps));
  }

  // Per-frame framebuffer (one color + one depth view).
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
    std::array<vk::ImageView, 2> views{
        *offscreenColors[i]->getImageView(),
        *offscreenDepths[i]->getImageView()};
    offscreenFramebuffers.push_back(vk::raii::Framebuffer(
        device,
        vk::FramebufferCreateInfo({}, *offscreenRenderPass, views,
                                  extent.width, extent.height, 1)));
  }

  // Sampler for the bubble pass to read offscreen color.
  sceneColorSampler = vk::raii::Sampler(
      device,
      vk::SamplerCreateInfo({}, vk::Filter::eLinear, vk::Filter::eLinear,
                            vk::SamplerMipmapMode::eNearest,
                            vk::SamplerAddressMode::eClampToEdge,
                            vk::SamplerAddressMode::eClampToEdge,
                            vk::SamplerAddressMode::eClampToEdge,
                            0.f, false, 1.f, false, vk::CompareOp::eAlways,
                            0.f, 0.f, vk::BorderColor::eFloatOpaqueBlack,
                            false));

  // Init barrier: bring color images to ShaderReadOnly so the very first
  // offscreen render pass's initialLayout assumption holds. Depth attachments
  // are transitioned by the render pass itself (initialLayout = Undefined).
  vgeu::oneTimeSubmit(device, commandPool, queue,
                      [&](const vk::raii::CommandBuffer& cmd) {
                        for (auto& img : offscreenColors) {
                          vk::ImageMemoryBarrier b(
                              vk::AccessFlags{},
                              vk::AccessFlagBits::eShaderRead,
                              vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eShaderReadOnlyOptimal,
                              VK_QUEUE_FAMILY_IGNORED,
                              VK_QUEUE_FAMILY_IGNORED, img->getImage(),
                              vk::ImageSubresourceRange(
                                  vk::ImageAspectFlagBits::eColor, 0, 1,
                                  0, 1));
                          cmd.pipelineBarrier(
                              vk::PipelineStageFlagBits::eTopOfPipe,
                              vk::PipelineStageFlagBits::eFragmentShader, {},
                              nullptr, nullptr, b);
                        }
                      });
}
```

- [ ] **Step 3: Call `prepareOffscreen()` from `prepare()`**

In `src/examples/soap_bubble/soap_bubble.cpp`, modify the existing `prepare()`:

```cpp
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
```

- [ ] **Step 4: Format and build**

```
rtk clang-format -i src/examples/soap_bubble/soap_bubble.hpp src/examples/soap_bubble/soap_bubble.cpp
rtk ./mingwBuild.bat Debug
```

Expected: build succeeds. Validation layer is not exercised yet by SSR (no offscreen pass in command buffer); existing example continues to render unchanged.

- [ ] **Step 5: Smoke run**

```
rtk ./build/soap_bubble.exe --model sphere
```

Expected: visual identical to commit before this task. Console: zero new VUIDs.

- [ ] **Step 6: Commit**

```
rtk git add src/examples/soap_bubble/soap_bubble.hpp src/examples/soap_bubble/soap_bubble.cpp
rtk git commit -m "feat(soap_bubble): add offscreen color/depth + render pass for SSR

Per-frame offscreen color and depth attachments (size = swapchain),
offscreen render pass with ShaderReadOnly<->ColorAttachment layout
cycle handled by subpass dependencies, per-frame framebuffer, scene
color sampler, and a one-time init barrier to ShaderReadOnly so the
first frame's initialLayout assumption holds. No descriptor binding
yet; bubble shader does not sample these.

Spec: docs/superpowers/specs/2026-05-08-soap-bubble-ssr-design.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

# Stage 2 — sceneColor descriptor + bubble pipeline 5 sets

Goal: bubble pipeline can receive the offscreen image as a sampler. Shader does not yet declare set=4 (declaring more layout sets than the shader uses is allowed by Vulkan).

---

## Task 2: sceneColor descriptor set + bubble layout grow

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.hpp`
- Modify: `src/examples/soap_bubble/soap_bubble.cpp`

- [ ] **Step 1: Add descriptor handles to `soap_bubble.hpp`**

In `class VgeExample` member section, near the existing `bgIrradianceDescSets` declaration, append:

```cpp
  // Scene-color descriptor (set=4) for SSR; per-frame, points at the matching
  // offscreen color image view.
  vk::raii::DescriptorSetLayout sceneColorSetLayout = nullptr;
  std::vector<vk::raii::DescriptorSet> sceneColorDescSets;
```

- [ ] **Step 2: Grow the descriptor pool and create the set layout / sets in `setupDescriptors()`**

In `src/examples/soap_bubble/soap_bubble.cpp`, replace the existing pool size block:

```cpp
  std::vector<vk::DescriptorPoolSize> poolSizes{
      {vk::DescriptorType::eUniformBuffer,
       3u * MAX_CONCURRENT_FRAMES /*globals + params + skybox*/},
      {vk::DescriptorType::eCombinedImageSampler,
       1u + 3u * MAX_CONCURRENT_FRAMES /*height + env + skybox + bgIrr*/}};
  // Set count: globals + params + env + bgIrr (per-frame) + height(1) +
  // skybox(per-frame)
  uint32_t maxSets = 4u * MAX_CONCURRENT_FRAMES + 1u + MAX_CONCURRENT_FRAMES;
```

with:

```cpp
  std::vector<vk::DescriptorPoolSize> poolSizes{
      {vk::DescriptorType::eUniformBuffer,
       3u * MAX_CONCURRENT_FRAMES /*globals + params + skybox*/},
      {vk::DescriptorType::eCombinedImageSampler,
       1u +
           4u * MAX_CONCURRENT_FRAMES /*height + env + skybox + bgIrr + scene*/}};
  // Set count: globals + params + env + bgIrr + scene (per-frame) +
  // height(1) + skybox(per-frame)
  uint32_t maxSets =
      5u * MAX_CONCURRENT_FRAMES + 1u + MAX_CONCURRENT_FRAMES;
```

Then, just before the existing `skybox = std::make_unique<vgeu::Skybox>(...)` call near the end of `setupDescriptors()`, insert:

```cpp
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
                      device,
                      vk::DescriptorSetAllocateInfo(*descriptorPool,
                                                    *sceneColorSetLayout))
                      .front()));
    vk::DescriptorImageInfo info(
        *sceneColorSampler, *offscreenColors[i]->getImageView(),
        vk::ImageLayout::eShaderReadOnlyOptimal);
    device.updateDescriptorSets(
        vk::WriteDescriptorSet(*sceneColorDescSets[i], 0, 0,
                               vk::DescriptorType::eCombinedImageSampler,
                               info),
        nullptr);
  }
```

- [ ] **Step 3: Grow bubble pipeline layout to 5 sets in `preparePipelines()`**

Replace the existing bubble pipeline layout setup:

```cpp
  std::array<vk::DescriptorSetLayout, 4> setLayouts{
      *globalsSetLayout, *bubbleParamsSetLayout, *heightTexSetLayout,
      *envSetLayout};
  bubblePipelineLayout = vk::raii::PipelineLayout(
      device, vk::PipelineLayoutCreateInfo({}, setLayouts));
```

with:

```cpp
  std::array<vk::DescriptorSetLayout, 5> setLayouts{
      *globalsSetLayout, *bubbleParamsSetLayout, *heightTexSetLayout,
      *envSetLayout, *sceneColorSetLayout};
  bubblePipelineLayout = vk::raii::PipelineLayout(
      device, vk::PipelineLayoutCreateInfo({}, setLayouts));
```

Bubble draw call's bind also needs to bind 5 sets. Replace:

```cpp
  std::array<vk::DescriptorSet, 4> descSets{
      *globalsDescSets[currentFrameIndex],
      *bubbleParamsDescSets[currentFrameIndex], *heightTexDescSet,
      *envDescSets[currentFrameIndex]};
```

with:

```cpp
  std::array<vk::DescriptorSet, 5> descSets{
      *globalsDescSets[currentFrameIndex],
      *bubbleParamsDescSets[currentFrameIndex], *heightTexDescSet,
      *envDescSets[currentFrameIndex],
      *sceneColorDescSets[currentFrameIndex]};
```

- [ ] **Step 4: Format and build**

```
rtk clang-format -i src/examples/soap_bubble/soap_bubble.hpp src/examples/soap_bubble/soap_bubble.cpp
rtk ./mingwBuild.bat Debug
```

Expected: build succeeds.

- [ ] **Step 5: Smoke run**

```
rtk ./build/soap_bubble.exe --model sphere
```

Expected: visual identical to before. Validation: zero new VUIDs. (The bubble pipeline now declares 5 sets; the shader still uses only 4. Vulkan permits this — the unused set is silently ignored.)

- [ ] **Step 6: Commit**

```
rtk git add src/examples/soap_bubble/soap_bubble.hpp src/examples/soap_bubble/soap_bubble.cpp
rtk git commit -m "feat(soap_bubble): add sceneColor descriptor (set=4) and 5-set bubble layout

Per-frame sceneColor descriptor sets bound to the corresponding offscreen
color image view; bubble pipeline layout grows from 4 to 5 sets; pool
sizes increased to cover the new per-frame CIS + set. Shader does not
yet declare set=4, which is permitted (extra layout sets are ignored).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

# Stage 3 — Shader SSR + frame logic + UI/CLI lockstep

Goal: SSR fully wired. ImGui toggle works; default ON shows refracted backgrounds.

---

## Task 3: Shader SSR branch + UBO/Options/UI/CLI/buildCommandBuffers (LOCKSTEP)

**Files:**
- Modify: `shaders/soap_bubble/bubble.frag`
- Modify: `src/examples/soap_bubble/soap_bubble.hpp`
- Modify: `src/examples/soap_bubble/soap_bubble.cpp`

This task is one logical commit because the host UBO, shader UBO block, and `buildCommandBuffers` branching must all agree on the new state for the binary to render correctly.

- [ ] **Step 1: Update `bubble.frag` UBO + add sceneColor binding + SSR branch**

In `shaders/soap_bubble/bubble.frag`, replace the entire `BubbleParams` UBO block:

```glsl
layout(set = 1, binding = 0) uniform BubbleParams {
  float thicknessMin;
  float thicknessMax;
  float n1;
  float n2;
  float n3;
  int spectralSamples;
  int thicknessMode;
  float gravityStrength;
  float noiseScale;
  int useAnimation;
  float driftSpeed;
  float roughness;
  float iblExposure;
  float iblGamma;
  float time;
  int rtMode;                // 0=both, 1=R-only, 2=T-only
  int showThicknessHeatmap;
  int showFresnelOnly;
  int showNormal;
  int _pad0;
} params;
```

with:

```glsl
layout(set = 1, binding = 0) uniform BubbleParams {
  float thicknessMin;
  float thicknessMax;
  float n1;
  float n2;
  float n3;
  int spectralSamples;
  int thicknessMode;
  float gravityStrength;
  float noiseScale;
  int useAnimation;
  float driftSpeed;
  float roughness;
  float iblExposure;
  float iblGamma;
  float time;
  int rtMode;                // 0=both, 1=R-only, 2=T-only
  int showThicknessHeatmap;
  int showFresnelOnly;
  int showNormal;
  int _pad0;
  int useSSR;                // 0=cubemap T, 1=screen-space refraction
  float refractDepth;        // empirical offset along T_dir
  int _pad1;
  int _pad2;
} params;
```

Add the sceneColor binding right after the existing `prefilteredCubemap` declaration:

```glsl
layout(set = 3, binding = 0) uniform samplerCube prefilteredCubemap;
layout(set = 4, binding = 0) uniform sampler2D sceneColor;
```

Replace the existing single-line T sample (`vec3 envT = textureLod(prefilteredCubemap, T_dir, lod).rgb;`) with the SSR-branched block:

```glsl
  vec3 envT;
  if (params.useSSR != 0) {
    vec3 P_behind = inWorldPos + T_dir * params.refractDepth;
    vec4 ndc = globals.projection * globals.view * vec4(P_behind, 1.0);
    vec2 uv = (ndc.xy / ndc.w) * 0.5 + 0.5;
    if (all(greaterThanEqual(uv, vec2(0.0))) &&
        all(lessThanEqual(uv, vec2(1.0)))) {
      envT = texture(sceneColor, uv).rgb;
    } else {
      envT = textureLod(prefilteredCubemap, T_dir, lod).rgb;
    }
  } else {
    envT = textureLod(prefilteredCubemap, T_dir, lod).rgb;
  }
```

- [ ] **Step 2: Update `BubbleParamsUbo` and `Options` in `soap_bubble.hpp`**

In `BubbleParamsUbo`, append four fields after `int32_t _pad0;`:

```cpp
struct BubbleParamsUbo {
  // ... existing fields up to and including _pad0 ...
  int32_t _pad0;
  // -- 16 --
  int32_t useSSR;
  float refractDepth;
  int32_t _pad1;
  int32_t _pad2;
};
```

In `struct Options`, append after `bool showNormal = false;` (before `// Background scene`):

```cpp
  // Refraction
  bool useSSR = true;
  float refractDepth = 0.5f;
```

Flip all four background instances `enabled` flag from `false` to `true`. The four updated entries should read:

```cpp
      {"/models/apple/food_apple_01_4k.gltf", glm::vec3(-1.5f, 0.3f, 1.5f),
       glm::vec3(0.f, 25.f, 0.f), 7.5f, glm::vec3(0.85f, 0.18f, 0.18f), true},
      {"/models/fox/Fox.gltf", glm::vec3(1.6f, -0.2f, 1.2f),
       glm::vec3(0.f, 180.f, 0.f), 0.015f, glm::vec3(0.95f, 0.62f, 0.20f),
       true},
      {"/models/sphere/smooth_sphere.gltf", glm::vec3(0.0f, 1.4f, 2.5f),
       glm::vec3(0.f, 0.f, 0.f), 0.6f, glm::vec3(0.30f, 0.55f, 0.85f), true},
      {"/models/dutch_ship_medium_1k/dutch_ship_medium_1k.gltf",
       glm::vec3(0.0f, -1.5f, 2.0f), glm::vec3(0.f, 90.f, 15.f), 0.1f,
       glm::vec3(0.55f, 0.42f, 0.30f), true},
```

- [ ] **Step 3: Wire UBO write, CLI, ImGui in `soap_bubble.cpp`**

In `updateBubbleParamsUbo()`, append after the existing `bubbleParamsUbo.showNormal = opts.showNormal ? 1 : 0;` line:

```cpp
  bubbleParamsUbo.useSSR = opts.useSSR ? 1 : 0;
  bubbleParamsUbo.refractDepth = opts.refractDepth;
```

In `setupCommandLineParser()`, append after the `--rtMode` line:

```cpp
  app.add_option("--useSSR", opts.useSSR);
  app.add_option("--refractDepth", opts.refractDepth,
                 "empirical depth along T_dir for SSR sample");
```

In `onUpdateUIOverlay()`, insert a new collapsible header between the existing `R/T Debug` block and the `IBL / Env` block:

```cpp
  if (ImGui::CollapsingHeader("Refraction", ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::Checkbox("Use Screen-Space Refraction (T)", &opts.useSSR);
    if (opts.useSSR) {
      ImGui::SliderFloat("refractDepth", &opts.refractDepth, 0.0f, 2.0f);
    }
  }
```

- [ ] **Step 4: Branch `buildCommandBuffers` on `opts.useSSR`**

In `buildCommandBuffers()`, replace the entire body between `cmd.begin({});` and the final `cmd.end();` (i.e., the render-pass content) with:

```cpp
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
          vk::ShaderStageFlagBits::eVertex |
              vk::ShaderStageFlagBits::eFragment,
          0, pc);
      bgModels[i]->draw(currentFrameIndex, cmd);
    }
  };

  // Offscreen pass (only when SSR is on).
  if (opts.useSSR) {
    vk::RenderPassBeginInfo offBegin(
        *offscreenRenderPass, *offscreenFramebuffers[currentFrameIndex],
        {{0, 0}, {width, height}}, clearValues);
    cmd.beginRenderPass(offBegin, vk::SubpassContents::eInline);
    cmd.setViewport(0, vk::Viewport(0.f, 0.f, (float)width, (float)height,
                                    0.f, 1.f));
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
      *envDescSets[currentFrameIndex],
      *sceneColorDescSets[currentFrameIndex]};
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                         *bubblePipelineLayout, 0, descSets, nullptr);
  bubbleModel->draw(currentFrameIndex, cmd);

  drawUI(cmd);
  cmd.endRenderPass();
```

(The skybox+bg code now lives in `recordSkyboxAndBg`. The previous loose-form skybox call and bg pass are replaced by two calls to the lambda — one in the offscreen pass when SSR is on, one in the swap pass always.)

- [ ] **Step 5: Format and build**

```
rtk clang-format -i src/examples/soap_bubble/soap_bubble.hpp src/examples/soap_bubble/soap_bubble.cpp
rtk ./mingwBuild.bat Debug
```

Expected: build succeeds, `bubble.frag.spv` regenerates.

- [ ] **Step 6: Smoke run + spot-check SSR**

```
rtk ./build/soap_bubble.exe --model sphere
```

Expected on first frame:
- Default state (`useSSR=on`, all 4 bg enabled, `refractDepth=0.5`): bubble shows refracted apple/fox/sphere/dutch_ship through it. Visually distinct from the cubemap-T era.
- ImGui Refraction group present with checkbox and slider.
- Toggle SSR off → bubble's T region reverts to cubemap appearance (no scene refraction).
- Validation: zero new VUIDs.

If `useSSR=on` shows a *black* T region instead of the scene, the most likely cause is the offscreen pass not actually executing — confirm `if (opts.useSSR)` branch wraps the `beginRenderPass` for offscreen. If you see a UV mismatch (e.g., refracted scene appears flipped vertically), it's the Y-axis convention; the spec §3 noted the existing example uses kFlipY at vertex load + standard NDC mapping, which should be correct. If it isn't, change `(ndc.xy / ndc.w) * 0.5 + 0.5` to `vec2(...) * vec2(0.5, -0.5) + vec2(0.5)`.

- [ ] **Step 7: Commit**

```
rtk git add shaders/soap_bubble/bubble.frag src/examples/soap_bubble/soap_bubble.hpp src/examples/soap_bubble/soap_bubble.cpp
rtk git commit -m "feat(soap_bubble): screen-space refraction (T) + ImGui/CLI/branching

Bubble shader gains set=4 sceneColor binding and useSSR/refractDepth UBO
fields; the T composite branches on params.useSSR. ON path projects
worldPos + T_dir * refractDepth to NDC, samples sceneColor at the UV
(cubemap fallback when UV out of frame). OFF path keeps cubemap T.
buildCommandBuffers runs an offscreen pass (skybox + bg) before the swap
pass when opts.useSSR; otherwise just the swap pass. Defaults: useSSR=true,
all 4 background instances enabled.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

# Stage 4 — Resize handling + sanity sign-off

---

## Task 4: windowResized override

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.hpp`
- Modify: `src/examples/soap_bubble/soap_bubble.cpp`

- [ ] **Step 1: Declare the override in `soap_bubble.hpp`**

In `class VgeExample`, near the existing `void viewChanged() override;`, add:

```cpp
  void windowResized() override;
```

- [ ] **Step 2: Implement the override in `soap_bubble.cpp`**

Insert the new method just before `void VgeExample::onUpdateUIOverlay() {`:

```cpp
void VgeExample::windowResized() {
  // VgeBase::windowResized is called after swapchain is recreated and
  // the new extent is in width/height + swapChainData->swapChainExtent.
  // Rebuild offscreen attachments to match the new extent and rebind the
  // sceneColor descriptors to the new image views.
  prepareOffscreen();
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
    vk::DescriptorImageInfo info(
        *sceneColorSampler, *offscreenColors[i]->getImageView(),
        vk::ImageLayout::eShaderReadOnlyOptimal);
    device.updateDescriptorSets(
        vk::WriteDescriptorSet(*sceneColorDescSets[i], 0, 0,
                               vk::DescriptorType::eCombinedImageSampler,
                               info),
        nullptr);
  }
}
```

- [ ] **Step 3: Format and build**

```
rtk clang-format -i src/examples/soap_bubble/soap_bubble.hpp src/examples/soap_bubble/soap_bubble.cpp
rtk ./mingwBuild.bat Debug
```

Expected: build succeeds.

- [ ] **Step 4: Smoke run + resize test**

```
rtk ./build/soap_bubble.exe --model sphere
```

Drag the window edge to resize. Expected:
- SSR continues to work at every intermediate resize (no flicker, no validation errors).
- Refraction effect tracks the new resolution (edges are sharp, no scaling artifacts).
- Validation: zero new VUIDs across the resize event.

- [ ] **Step 5: Commit**

```
rtk git add src/examples/soap_bubble/soap_bubble.hpp src/examples/soap_bubble/soap_bubble.cpp
rtk git commit -m "feat(soap_bubble): rebuild offscreen on window resize

Override VgeBase::windowResized to re-run prepareOffscreen() and
re-bind the sceneColor descriptor sets to the new image views.
Without this, post-resize SSR samples a stale offscreen image at
the old extent.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Sanity sign-off

**Files:** none modified (verification only).

- [ ] **Step 1: Walk §5.1 SSR visual sanity**

Run `rtk ./build/soap_bubble.exe --model sphere` and verify each row of spec §5.1:

| input | expected | pass? |
|---|---|---|
| default state | bubble shows refracted apple/fox/sphere/dutch_ship | |
| `useSSR=off` (toggle in ImGui) | identical to commit `62864c9` | |
| `refractDepth=0` | bubble shows scene-at-this-pixel without bubble (because offscreen has no bubble) | |
| `refractDepth=2.0` | strong lens distortion | |
| Bubble at frame edge | UV-out-of-frame triggers cubemap fallback (smooth, no black) | |
| `R/T Debug = T-only` + SSR on | only refracted background visible | |
| `R/T Debug = R-only` + SSR on | reflection only; SSR has no effect | |
| Disable all 4 bg, SSR on | only skybox in offscreen; bubble refracts skybox subtly | |

If any row fails, debug. Most likely culprits: NDC→UV Y flip (try `vec2(0.5, -0.5) + vec2(0.5)`), or descriptor binding (verify `sceneColorDescSets[i]` actually points at `offscreenColors[i]`).

- [ ] **Step 2: §5.2 Toggle responsiveness**

Toggle `useSSR` checkbox repeatedly. Drag `refractDepth` slider mid-frame. Expected: no flicker, no validation errors, smooth visual transitions.

- [ ] **Step 3: §5.3 Window resize**

Drag-resize the window through several sizes (small, large, very small, restore). Expected: SSR continues to work cleanly at every resize event.

- [ ] **Step 4: §5.4 useJitter rebake**

Toggle `useJitter` (in IBL / Env group) while SSR is on. Expected: rebake completes without breaking SSR (cubemap rebuilds, but offscreen is unaffected).

- [ ] **Step 5: §5.5 Validation cleanliness**

Final console scan: zero new VUIDs introduced by this branch's commits. Pre-existing `VUID-02697` from PBR remains out of scope.

If all rows pass, no further commit is needed.

---

## End-of-plan check

```
rtk git log --oneline -7
rtk git status
```

Expected log (top-down):
- `feat(soap_bubble): rebuild offscreen on window resize`
- `feat(soap_bubble): screen-space refraction (T) + ImGui/CLI/branching`
- `feat(soap_bubble): add sceneColor descriptor (set=4) and 5-set bubble layout`
- `feat(soap_bubble): add offscreen color/depth + render pass for SSR`
- `docs(soap_bubble): add screen-space refraction spec`
- (preceding commits…)

`git status` should be clean.
