# PBR Sphere Mode & Debug View Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add runtime-toggleable debug view visibility and a metallic/roughness sphere grid mode to the PBR deferred rendering example.

**Architecture:** Extend `DynamicUboElt` with a `pbrOverride` vec4 so each instance can override metallic/roughness in `mrt.frag`. Load both model sets at startup; skip draws based on `opts.useSpheres`. Bind 1×1 dummy textures at set 2 before sphere draws (no `kBindImages`).

**Tech Stack:** Vulkan-Hpp RAII, GLSL 450, glslangValidator, CMake, ImGui

---

## File Map

| File | Change |
|---|---|
| `src/examples/pbr/pbr.hpp` | Options fields, `DynamicUboElt.pbrOverride`, `ModelInstance.sceneMode`, dummy texture members, new method declarations |
| `src/examples/pbr/pbr.cpp` | `loadAssets`, `setupDynamicUbo`, `setupDescriptors`, `buildCommandBuffers`, `onUpdateUIOverlay`, `draw`, new `createDummyTexture`/`updateDynamicUbo` |
| `shaders/pbr/mrt.vert` | Add `pbrOverride` to `ModelUbo` |
| `shaders/pbr/mrt.frag` | Add `ModelUbo` with `pbrOverride`, ARM override logic, tangent guard |

---

## Task 1: Extend pbr.hpp data structures

**Files:**
- Modify: `src/examples/pbr/pbr.hpp`

- [ ] **Step 1: Add options fields to `struct Options`**

In `pbr.hpp`, after the existing `float lightIntensity = 1.0f;` line, add:

```cpp
bool showDebugViews = true;
bool useSpheres = false;
std::array<float, 4> sphereAlbedo = {1.0f, 1.0f, 1.0f, 1.0f};
```

- [ ] **Step 2: Extend `DynamicUboElt`**

Replace the existing struct:
```cpp
struct DynamicUboElt {
  glm::mat4 modelMatrix{1.f};
  glm::vec4 modelColor{0.f};
};
```
With:
```cpp
struct DynamicUboElt {
  glm::mat4 modelMatrix{1.f};
  glm::vec4 modelColor{0.f};
  glm::vec4 pbrOverride{0.f};  // x=metallic, y=roughness, z=useOverride(0/1), w=unused
};
```

- [ ] **Step 3: Add `SceneMode` enum and field to `ModelInstance`**

In `struct ModelInstance`, add before the constructors:
```cpp
enum class SceneMode { kModelOnly, kSphereOnly };
SceneMode sceneMode = SceneMode::kModelOnly;
```

- [ ] **Step 4: Add dummy texture members and new method declarations to `class VgeExample`**

Add in the public methods section:
```cpp
std::unique_ptr<vgeu::VgeuImage> createDummyTexture(std::array<uint8_t, 4> rgba);
void updateDynamicUbo();
```

Add in the member variables section (after `vk::raii::Sampler colorSampler`):
```cpp
// Dummy textures for sphere pass (1x1 pixels)
std::unique_ptr<vgeu::VgeuImage> sphereDummyAlbedo;
std::unique_ptr<vgeu::VgeuImage> sphereDummyNormal;
std::unique_ptr<vgeu::VgeuImage> sphereDummyMetRough;
std::unique_ptr<vgeu::VgeuImage> sphereDummyEmissive;
vk::raii::DescriptorSet sphereDummyDescriptorSet = nullptr;
```

- [ ] **Step 5: Commit**

```bash
cd C:/Users/rlckd/Desktop/kc/Vulkan-Graphics-Example
rtk git add src/examples/pbr/pbr.hpp
rtk git commit -m "feat(pbr): extend data structures for sphere mode and debug view toggle"
```

---

## Task 2: Update mrt.vert — add pbrOverride to ModelUbo

**Files:**
- Modify: `shaders/pbr/mrt.vert`

- [ ] **Step 1: Add `pbrOverride` to the `ModelUbo` struct**

In `shaders/pbr/mrt.vert`, replace:
```glsl
layout (set = 1, binding = 0) uniform ModelUbo 
{
	mat4 modelMatrix;
	vec4 modelColor;
} modelUbo;
```
With:
```glsl
layout (set = 1, binding = 0) uniform ModelUbo 
{
	mat4 modelMatrix;
	vec4 modelColor;
	vec4 pbrOverride;  // x=metallic, y=roughness, z=useOverride(0/1), w=unused
} modelUbo;
```

(No other changes to mrt.vert — `pbrOverride` is consumed in the fragment shader.)

- [ ] **Step 2: Compile mrt.vert to verify no errors**

```bash
cd C:/Users/rlckd/Desktop/kc/Vulkan-Graphics-Example
glslangValidator -V shaders/pbr/mrt.vert -o shaders/pbr/mrt.vert.spv
```

Expected: no errors, `mrt.vert.spv` updated.

- [ ] **Step 3: Commit**

```bash
rtk git add shaders/pbr/mrt.vert shaders/pbr/mrt.vert.spv
rtk git commit -m "feat(pbr): add pbrOverride to mrt.vert ModelUbo"
```

---

## Task 3: Update mrt.frag — ARM override + tangent guard

**Files:**
- Modify: `shaders/pbr/mrt.frag`

- [ ] **Step 1: Add `ModelUbo` to mrt.frag and update descriptor set layout for `dynamicUboDescriptorSetLayout`**

First, note that `dynamicUboDescriptorSetLayout` in `pbr.cpp` currently uses `vk::ShaderStageFlagBits::eVertex` only. We need to add `eFragment` so the fragment shader can access `ModelUbo`. We'll do this in Task 5.

- [ ] **Step 2: Replace the full `shaders/pbr/mrt.frag` content**

```glsl
#version 450

layout (set = 2, binding = 0) uniform sampler2D samplerColorMap;
layout (set = 2, binding = 1) uniform sampler2D samplerNormalMap;
layout (set = 2, binding = 2) uniform sampler2D samplerMetallicRoughnessMap;
layout (set = 2, binding = 3) uniform sampler2D samplerEmissionMap;

layout (set = 1, binding = 0) uniform ModelUbo {
	mat4 modelMatrix;
	vec4 modelColor;
	vec4 pbrOverride;  // x=metallic, y=roughness, z=useOverride(0/1), w=unused
} modelUbo;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec4 inColor;
layout (location = 3) in vec4 inWorldPos;
layout (location = 4) in vec3 inTangent;

layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outAlbedo;
layout (location = 3) out vec4 outArm;
layout (location = 4) out vec4 outEmissive;

void main() 
{
	vec4 albedo = texture(samplerColorMap, inUV);
	vec3 color = mix(albedo.rgb, inColor.rgb, inColor.a);
	outAlbedo = albedo;

	outPosition = inWorldPos;

	// Normal: use tangent-space mapping only when tangent is valid.
	// The sphere model has no TANGENT attribute (zero tangent), so we fall
	// back to the geometric normal to avoid NaN from normalize(vec3(0)).
	vec3 N = normalize(inNormal);
	if (length(inTangent) > 0.001) {
		vec3 T = normalize(inTangent);
		vec3 B = normalize(cross(N, T));
		mat3 TBN = mat3(T, B, N);
		vec3 normalMapSample = texture(samplerNormalMap, inUV).xyz * 2.0 - vec3(1.0);
		N = normalize(TBN * normalMapSample);
	}
	outNormal = vec4(N, 1.0);

	// ARM (AO/Roughness/Metallic): override per-instance when useSpheres is active.
	// pbrOverride: x=metallic, y=roughness, z=useOverride
	vec3 arm = vec3(0.0);
	if (modelUbo.pbrOverride.z > 0.5) {
		// glTF metallicRoughness convention: g=roughness, b=metallic
		arm = vec3(0.0, modelUbo.pbrOverride.y, modelUbo.pbrOverride.x);
	} else {
		arm.rgb = texture(samplerMetallicRoughnessMap, inUV).rgb;
		// arm.r = texture(samplerOcclusionMap, inUV).r; // occlusion
	}
	outArm = vec4(arm, 1.0);

	vec3 emissive = vec3(0.0);
	emissive.rgb = texture(samplerEmissionMap, inUV).rgb;
	outEmissive = vec4(emissive, 1.0);
}
```

- [ ] **Step 3: Compile mrt.frag to verify no errors**

```bash
glslangValidator -V shaders/pbr/mrt.frag -o shaders/pbr/mrt.frag.spv
```

Expected: no errors, `mrt.frag.spv` updated.

- [ ] **Step 4: Commit**

```bash
rtk git add shaders/pbr/mrt.frag shaders/pbr/mrt.frag.spv
rtk git commit -m "feat(pbr): add pbrOverride ARM logic and tangent guard to mrt.frag"
```

---

## Task 4: Add createDummyTexture helper and load sphere in loadAssets()

**Files:**
- Modify: `src/examples/pbr/pbr.cpp`

- [ ] **Step 1: Add `createDummyTexture()` implementation before `loadAssets()`**

In `pbr.cpp`, add this function before `VgeExample::loadAssets()`:

```cpp
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
```

- [ ] **Step 2: Extend `loadAssets()` to load sphere model and create dummy textures**

At the end of `VgeExample::loadAssets()`, after the existing helmet instances loop, add:

```cpp
  // Sphere model (geometry only; textures are provided via dummy descriptor set)
  std::shared_ptr<vgeu::glTF::Model> sphere = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool, MAX_CONCURRENT_FRAMES);
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

  // 1×1 dummy textures for sphere draw calls:
  //   albedo  = white     (overridden by modelColor via modelColor.a=1.0)
  //   normal  = flat +Z   (128,128,255 → tangent-space (0,0,1) → passes geometric normal through)
  //   metrough = neutral  (g=128→roughness≈0.5; overridden by pbrOverride)
  //   emissive = black
  sphereDummyAlbedo   = createDummyTexture({255, 255, 255, 255});
  sphereDummyNormal   = createDummyTexture({128, 128, 255, 255});
  sphereDummyMetRough = createDummyTexture({0,   128,   0, 255});
  sphereDummyEmissive = createDummyTexture({0,     0,   0, 255});
```

- [ ] **Step 3: Commit**

```bash
rtk git add src/examples/pbr/pbr.cpp
rtk git commit -m "feat(pbr): add createDummyTexture helper and sphere model loading"
```

---

## Task 5: Update setupDynamicUbo() for sphere grid

**Files:**
- Modify: `src/examples/pbr/pbr.cpp`

- [ ] **Step 1: Add sphere instance setup at the end of `setupDynamicUbo()`**

After the existing helmet loop in `VgeExample::setupDynamicUbo()`, add:

```cpp
  // Sphere instances: rows = metallic 0→1, columns = roughness 0→1
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
```

- [ ] **Step 2: Commit**

```bash
rtk git add src/examples/pbr/pbr.cpp
rtk git commit -m "feat(pbr): set sphere grid metallic/roughness in setupDynamicUbo"
```

---

## Task 6: Update setupDescriptors() — extend pool and add sphere dummy set

**Files:**
- Modify: `src/examples/pbr/pbr.cpp`

- [ ] **Step 1: Extend descriptor pool for sphere dummy set**

In `VgeExample::setupDescriptors()`, update the pool sizes:

Change the `eCombinedImageSampler` pool size line from:
```cpp
  poolSizes.emplace_back(vk::DescriptorType::eCombinedImageSampler,
      static_cast<uint32_t>(MAX_CONCURRENT_FRAMES * offScreenFrameBuf.numAttachments));
```
To:
```cpp
  poolSizes.emplace_back(vk::DescriptorType::eCombinedImageSampler,
      static_cast<uint32_t>(MAX_CONCURRENT_FRAMES * offScreenFrameBuf.numAttachments)
          + 4u /*sphere dummy: 4 combined image samplers*/);
```

Change the `maxSets` count in `vk::DescriptorPoolCreateInfo` from:
```cpp
      MAX_CONCURRENT_FRAMES /*composition*/ +
      MAX_CONCURRENT_FRAMES * 2 /*offscreen + dynamic*/ +
      MAX_CONCURRENT_FRAMES /*sprite*/,
```
To:
```cpp
      MAX_CONCURRENT_FRAMES /*composition*/ +
      MAX_CONCURRENT_FRAMES * 2 /*offscreen + dynamic*/ +
      MAX_CONCURRENT_FRAMES /*sprite*/ +
      1u /*sphere dummy*/,
```

- [ ] **Step 2: Add `eFragment` to `dynamicUboDescriptorSetLayout` stage flags**

Change:
```cpp
    vk::DescriptorSetLayoutBinding binding(0, vk::DescriptorType::eUniformBufferDynamic, 1,
                                           vk::ShaderStageFlagBits::eVertex);
```
To:
```cpp
    vk::DescriptorSetLayoutBinding binding(0, vk::DescriptorType::eUniformBufferDynamic, 1,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment);
```

This allows `mrt.frag` to read `modelUbo.pbrOverride` from the dynamic UBO at set 1.

- [ ] **Step 3: Allocate and write sphere dummy descriptor set**

At the end of `VgeExample::setupDescriptors()`, after the sprite descriptor set allocation block, add:

```cpp
  // Sphere dummy descriptor set (set 2): 4 combined image samplers → 1×1 dummy textures.
  // Uses the same layout as glTF model image descriptors (4 bindings: base, normal, metRough, emissive).
  // Bound manually before sphere draws; kBindImages is NOT set so model::draw() doesn't override it.
  {
    vk::DescriptorSetAllocateInfo allocInfo(
        *descriptorPool, *modelInstances[0].model->descriptorSetLayoutImage);
    sphereDummyDescriptorSet = std::move(
        vk::raii::DescriptorSets(device, allocInfo).front());

    auto albInfo  = sphereDummyAlbedo->descriptorImageInfo(
        *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);
    auto normInfo = sphereDummyNormal->descriptorImageInfo(
        *colorSampler, vk::ImageLayout::eShaderReadOnlyOptimal);
    auto mrInfo   = sphereDummyMetRough->descriptorImageInfo(
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
```

- [ ] **Step 4: Commit**

```bash
rtk git add src/examples/pbr/pbr.cpp
rtk git commit -m "feat(pbr): extend descriptor pool and add sphere dummy descriptor set"
```

---

## Task 7: Update buildCommandBuffers() — mode skip and debug view toggle

**Files:**
- Modify: `src/examples/pbr/pbr.cpp`

- [ ] **Step 1: Update the G-buffer pass draw loop**

In `VgeExample::buildCommandBuffers()`, inside the G-buffer offscreen render pass, replace the existing model draw loop:

```cpp
    for (size_t instIdx = 0; instIdx < modelInstances.size(); instIdx++) {
      const auto& inst = modelInstances[instIdx];
      if (!inst.model) continue;
      cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayoutOffScreen, 1,
          {*descriptorSets.dynamicUboDescriptorSets[currentFrameIndex]},
          static_cast<uint32_t>(alignedSizeDynamicUboElt * instIdx));
      inst.model->draw(currentFrameIndex, cmd, vgeu::RenderFlagBits::kBindImages,
                       *pipelineLayoutOffScreen, 2);
    }
```

With:

```cpp
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
        inst.model->draw(currentFrameIndex, cmd, 0 /*no kBindImages*/,
                         *pipelineLayoutOffScreen, 2);
      } else {
        inst.model->draw(currentFrameIndex, cmd, vgeu::RenderFlagBits::kBindImages,
                         *pipelineLayoutOffScreen, 2);
      }
    }
```

- [ ] **Step 2: Wrap displayTargets sub-viewport loop with showDebugViews toggle**

In the swapchain render pass section, find:

```cpp
    // Display target sub-viewports (debug G-buffer views, top-right corner) - drawn last to stay on top
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
```

Replace with:

```cpp
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
```

- [ ] **Step 3: Commit**

```bash
rtk git add src/examples/pbr/pbr.cpp
rtk git commit -m "feat(pbr): add mode-based draw skip and showDebugViews toggle in buildCommandBuffers"
```

---

## Task 8: Add updateDynamicUbo() and call from draw()

**Files:**
- Modify: `src/examples/pbr/pbr.cpp`

- [ ] **Step 1: Add `updateDynamicUbo()` implementation**

Add this function before `VgeExample::draw()`:

```cpp
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
```

- [ ] **Step 2: Call `updateDynamicUbo()` from `draw()`**

In `VgeExample::draw()`, after `updateUboComposition()`, add:
```cpp
  updateDynamicUbo();
```

- [ ] **Step 3: Commit**

```bash
rtk git add src/examples/pbr/pbr.cpp
rtk git commit -m "feat(pbr): add updateDynamicUbo for per-frame sphere albedo sync"
```

---

## Task 9: Add UI controls in onUpdateUIOverlay()

**Files:**
- Modify: `src/examples/pbr/pbr.cpp`

- [ ] **Step 1: Add checkboxes and color picker**

In `VgeExample::onUpdateUIOverlay()`, inside the `ImGui::TreeNodeEx("Immediate", ...)` block, after the existing `ImGui::Separator()` (before or after the light controls), add:

```cpp
      ImGui::Separator();
      ImGui::Checkbox("Show Debug Views", &opts.showDebugViews);
      ImGui::Checkbox("Use Spheres", &opts.useSpheres);
      if (opts.useSpheres) {
        uiOverlay->colorPicker("Sphere Albedo", opts.sphereAlbedo.data());
      }
```

- [ ] **Step 2: Commit**

```bash
rtk git add src/examples/pbr/pbr.cpp
rtk git commit -m "feat(pbr): add showDebugViews and useSpheres UI controls"
```

---

## Task 10: Build and verify

**Files:** none (build only)

- [ ] **Step 1: Build Shaders target**

```bash
cd C:/Users/rlckd/Desktop/kc/Vulkan-Graphics-Example/build
cmake --build . --target Shaders
```

Expected: no compile errors. All `*.spv` files in `shaders/pbr/` updated.

- [ ] **Step 2: Build pbr target**

```bash
cmake --build . --target pbr
```

Expected: zero errors, zero warnings about new code.

- [ ] **Step 3: Run and verify debug view toggle**

Launch the pbr example. In the UI, uncheck "Show Debug Views".  
Expected: the 9 sub-viewport tiles in the top-right disappear. Re-check → they reappear.

- [ ] **Step 4: Run and verify sphere mode**

Check "Use Spheres".  
Expected:
- Helmets and floor disappear; 4×4 spheres appear
- Bottom-left sphere: metallic=0, roughness=0 (shiny, low metallic → dielectric specular)
- Top-right sphere: metallic=1, roughness=1 (rough metallic)
- Color picker changes sphere albedo in real time
- No Vulkan validation errors

- [ ] **Step 5: Commit SPV artifacts and final build output**

```bash
cd C:/Users/rlckd/Desktop/kc/Vulkan-Graphics-Example
rtk git add shaders/pbr/mrt.vert.spv shaders/pbr/mrt.frag.spv
rtk git commit -m "build: update mrt shader spv for pbr sphere mode"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** showDebugViews toggle ✓, sphere mode ✓, metallic/roughness grid ✓, albedo color picker ✓, floor hidden in sphere mode ✓ (floor is `kModelOnly`), runtime toggle ✓
- [x] **Placeholders:** none
- [x] **Type consistency:** `ModelInstance::SceneMode::kModelOnly/kSphereOnly` used consistently across Tasks 1, 5, 7, 8. `DynamicUboElt.pbrOverride` defined in Task 1, written in Task 5, read in Task 3. `sphereDummyDescriptorSet` declared in Task 1, allocated in Task 6, bound in Task 7.
- [x] **Descriptor pool:** `eCombinedImageSampler` count +4 for dummy set (Task 6 Step 1). maxSets +1 (Task 6 Step 1). Dynamic UBO layout gets `eFragment` stage (Task 6 Step 2).
- [x] **Tangent guard:** sphere model has no TANGENT → normalize(0,0,0) → NaN in mrt.vert → guarded in mrt.frag with `length(inTangent) > 0.001` (Task 3).
- [x] **Per-frame update:** `updateDynamicUbo()` syncs sphereAlbedo to GPU per frame (Task 8).
