# Paint Splatter — Subtractive Color Mixing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the canvas's order-dependent alpha-over deposit with an order-independent subtractive (optical-density / Beer-Lambert glazing) mixing model, so overlaps are flicker-free, repeated coverage darkens, and pigments mix subtractively.

**Architecture:** Deposit accumulates per-RGB-channel optical density into three new `R32_UINT` images via `imageAtomicAdd` (addition is commutative → order-independent). A new `resolve.comp` pass converts density → `reflectance = exp(-D)` and writes the existing RGBA8 `canvasImage` each frame; `canvas.frag` and the PNG save path are unchanged. All passes run on the graphics queue where the canvas already lives.

**Tech Stack:** C++17, Vulkan-Hpp RAII, GLSL compute shaders, ImGui (`vgeu_ui_overlay`), the existing `paint_splatter` example.

**Design spec:** `docs/superpowers/specs/2026-06-07-paint-splatter-subtractive-color-mixing-design.md`

---

## Verification Model (read first)

This repo has **no unit-test framework**; the example is verified by running it. Verification primitives (same as prior milestones):

- **VL-CLEAN**: run the debug build (validation layers on) and confirm **zero validation errors/warnings** in stdout/stderr.
- **GATE (user)**: a STOP. The user visually confirms on-screen behavior. Do not pass a GATE without user approval.

## Conventions

- **Build:** `rtk cmd /c mingwBuild.bat` from repo root (shaders compile as part of the build). Run the binary separately: `rtk ./build/paint_splatter.exe`. **Run build and run as SEPARATE commands** (chaining build+run in one PowerShell call has broken before).
- **New shader files need a CMake reconfigure** the first time (GLOB_RECURSE at `CMakeLists.txt:97` only re-scans on configure). After adding `resolve.comp`, if `resolve.comp.spv` is not produced under `shaders/paint_splatter/`, re-run the configure step of `mingwBuild.bat` (or delete `build/CMakeCache.txt` and rebuild) and confirm `resolve.comp.spv` appears.
- **clang-format:** run `clang-format -i` on every edited `.cpp/.hpp` before each commit (user rule). Shaders are NOT clang-formatted.
- **rtk:** prefix shell commands with `rtk`.
- **World convention (locked):** screen-up = world −Y; gravity +Y (down on screen); canvas floor at y=0; fluid/spoids at y<0.
- All paths are relative to repo root `C:\Users\rlckd\Desktop\kc\Vulkan-Graphics-Example`.
- **Density model constants:** fixed-point scale `densityScale` (default 1e5) and resolve clamp `maxDensity` (default 8.0) live in `ComputeUbo`. `EPS = 1/255` is a shader constant capping per-channel density of a fully saturated paint channel.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/examples/paint_splatter/paint_splatter.hpp` | Modify | `ComputeUbo` density fields; host knobs `densityScale`/`maxDensity`; `densityImages[3]` + `compute.resolve` members |
| `src/examples/paint_splatter/paint_splatter.cpp` | Modify | image creation/clear, descriptor pool/layout/writes, UBO update, resolve pipeline, deposit→resolve dispatch + barriers, restart clear, ImGui |
| `shaders/paint_splatter/deposit.comp` | Modify | accumulate per-channel optical density via `imageAtomicAdd` (replaces rgba8 alpha-over) |
| `shaders/paint_splatter/resolve.comp` | Create | density → `exp(-D)` → RGBA8 canvas |
| `docs/paint_splatter_debug_log.md` | Modify | record the milestone + result |

Bindings (compute descriptor set): existing 0–10 unchanged; **new 11 = densityR, 12 = densityG, 13 = densityB** (`R32_UINT` storage images). Binding 8 (canvas storage image) is now written only by `resolve.comp`.

---

## Task 1 — ComputeUbo density fields + host knobs + UI (no behavior change)

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.hpp` (`ComputeUbo` at lines 57–91; host knobs near 389)
- Modify: `src/examples/paint_splatter/paint_splatter.cpp` (`updateComputeUbo` near 1841; ImGui near 2836)

- [ ] **Step 1: Extend `ComputeUbo`.** In `paint_splatter.hpp`, replace the tail of `struct ComputeUbo` (the `dpClampFactor` field block and the `// -- 144 --` marker + `static_assert`, lines 86–91) with:

```cpp
  float dpClampFactor;  // -- 140 -- per-iteration |dp| cap = factor*h; <=0
                        // disables. Low/off lets the incompressible rebound
                        // (crown) grow; default 0.2 = old behavior (M8)
  float densityScale;   // -- 144 -- fixed-point scale for atomic density accum
  float maxDensity;     // -- 148 -- resolve clamp on accumulated optical density
  float padDensity0;    // -- 152 --
  float padDensity1;    // -- 156 --
  // -- 160 --
};
static_assert(sizeof(ComputeUbo) == 160, "ComputeUbo std140 size");
```

- [ ] **Step 2: Add host knobs.** In `paint_splatter.hpp`, immediately AFTER `float depositRadius = 0.01f;` (line 391) add:

```cpp
  // Subtractive color-mixing (optical density) knobs. densityScale = fixed-point
  // scale for the atomic accumulators; maxDensity = resolve-time clamp so a
  // texel can't go past ~black under pathological build-up.
  float densityScale = 1.0e5f;
  float maxDensity = 8.0f;
```

- [ ] **Step 3: Feed them to the UBO.** In `paint_splatter.cpp` `updateComputeUbo`, after `compute.ubo.dpClampFactor = dpClampFactor;` (line 1841) add:

```cpp
  compute.ubo.densityScale = densityScale;  // subtractive-mix fixed-point scale
  compute.ubo.maxDensity = maxDensity;      // resolve clamp on optical density
```

- [ ] **Step 4: Add a UI slider.** In `paint_splatter.cpp`, immediately AFTER the `deposit strength` slider (line 2836–2837, the `ImGui::DragFloat("deposit strength", ...)` call) add:

```cpp
      ImGui::DragFloat("max density", &maxDensity, 0.05f, 0.5f, 32.f, "%.2f");
```

- [ ] **Step 5: clang-format the edited files.**

Run: `rtk clang-format -i src/examples/paint_splatter/paint_splatter.cpp src/examples/paint_splatter/paint_splatter.hpp`
Expected: no output (success).

- [ ] **Step 6: Build.**

Run: `rtk cmd /c mingwBuild.bat`
Expected: build succeeds; the new `static_assert(sizeof(ComputeUbo) == 160)` compiles (proves std140 size is right).

- [ ] **Step 7: Run, confirm VL-CLEAN.**

Run: `rtk ./build/paint_splatter.exe`
Expected: app runs as before (no visual change yet), zero validation errors/warnings in stdout/stderr. Close the window.

- [ ] **Step 8: Commit.**

```bash
rtk git add src/examples/paint_splatter/paint_splatter.hpp src/examples/paint_splatter/paint_splatter.cpp
rtk git commit -m "feat(paint_splatter): add density-mixing UBO fields + maxDensity UI"
```

---

## Task 2 — Density images: create, clear, bind (bound but unused)

Create the three `R32_UINT` accumulator images, clear them to 0 at creation and on restart, grow the pool, add layout bindings 11/12/13, and write the descriptors. No shader uses them yet, so this stays VL-CLEAN.

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.hpp` (member near `canvasSampler`, line 378)
- Modify: `src/examples/paint_splatter/paint_splatter.cpp` (`createCanvasImage` 422–460; `createDescriptorPool` 591–593; `createComputeDescriptorSetLayout` 1162–1169; descriptor writes 1726–1773; `restartSimulation` 349–359)

- [ ] **Step 1: Add the member.** In `paint_splatter.hpp`, immediately AFTER `vk::raii::Sampler canvasSampler = nullptr;` (line 378) add:

```cpp
  // Subtractive mixing: per-RGB-channel optical-density accumulators (M-color).
  // R32_UINT so deposit can imageAtomicAdd fixed-point density; resolve.comp
  // reads them and writes canvasImage. Single shared images (like canvasImage),
  // GENERAL layout, persistent (cleared only on restart).
  std::array<std::unique_ptr<vgeu::VgeuImage>, 3> densityImages;
```

- [ ] **Step 2: Create + clear the images.** In `paint_splatter.cpp` `createCanvasImage`, immediately BEFORE the closing `}` of the function (after the `vgeu::oneTimeSubmit(...)` block that clears the canvas white, i.e. after line 459) add:

```cpp
  // Density accumulator images (R32_UINT, one per RGB channel). Created in
  // GENERAL and cleared to 0 (= zero optical density = white paper after
  // resolve). Mirror canvasImage's single-shared, persistent lifetime.
  vk::ImageSubresourceRange dRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
  for (int k = 0; k < 3; k++) {
    densityImages[k] = std::make_unique<vgeu::VgeuImage>(
        device, globalAllocator->getAllocator(), vk::Format::eR32Uint, extent,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferDst,
        vk::ImageLayout::eUndefined, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        vk::ImageAspectFlagBits::eColor, 1);
    vgeu::oneTimeSubmit(
        device, commandPool, queue, [&](const vk::raii::CommandBuffer& cmd) {
          vk::ImageMemoryBarrier toGeneral(
              vk::AccessFlags{}, vk::AccessFlagBits::eTransferWrite,
              vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
              VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
              densityImages[k]->getImage(), dRange);
          cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                              vk::PipelineStageFlagBits::eTransfer,
                              vk::DependencyFlags{}, nullptr, nullptr, toGeneral);
          vk::ClearColorValue zero(std::array<uint32_t, 4>{0u, 0u, 0u, 0u});
          cmd.clearColorImage(densityImages[k]->getImage(),
                              vk::ImageLayout::eGeneral, zero, dRange);
        });
  }
```

- [ ] **Step 3: Grow the descriptor pool.** In `paint_splatter.cpp` `createDescriptorPool`, replace the canvas storage-image pool size (lines 591–593) with:

```cpp
  // Storage images per compute set: 1 canvas (binding 8, written by resolve) +
  // 3 density accumulators (bindings 11-13, atomic-added by deposit, read by
  // resolve).
  poolSizes.emplace_back(vk::DescriptorType::eStorageImage,
                         MAX_CONCURRENT_FRAMES * 4u);
```

- [ ] **Step 4: Add layout bindings 11/12/13.** In `paint_splatter.cpp` `createComputeDescriptorSetLayout`, immediately AFTER the binding-10 `layoutBindings.emplace_back(10, ...)` block (line 1168–1169) and BEFORE `vk::DescriptorSetLayoutCreateInfo layoutCI(...)` (line 1171) add:

```cpp
  // 11/12/13 = per-RGB-channel optical-density accumulators (R32_UINT storage).
  // deposit.comp imageAtomicAdds into them; resolve.comp reads them.
  for (uint32_t b = 11; b <= 13; b++) {
    layoutBindings.emplace_back(b, vk::DescriptorType::eStorageImage, 1,
                                vk::ShaderStageFlagBits::eCompute);
  }
```

- [ ] **Step 5: Write the density descriptors.** In `paint_splatter.cpp`, in the compute descriptor-write loop, immediately AFTER the `prevLiveInfo` declaration (line 1734–1736) and BEFORE `std::array<vk::WriteDescriptorSet, 11> writes{` (line 1738) add:

```cpp
    vk::DescriptorImageInfo densityInfo0(
        nullptr, *densityImages[0]->getImageView(), vk::ImageLayout::eGeneral);
    vk::DescriptorImageInfo densityInfo1(
        nullptr, *densityImages[1]->getImageView(), vk::ImageLayout::eGeneral);
    vk::DescriptorImageInfo densityInfo2(
        nullptr, *densityImages[2]->getImageView(), vk::ImageLayout::eGeneral);
```

- [ ] **Step 6: Add them to the write array.** In the same loop, change `std::array<vk::WriteDescriptorSet, 11> writes{` (line 1738) to `std::array<vk::WriteDescriptorSet, 14> writes{`, and immediately AFTER the binding-10 `prevLiveInfo` write (the entry ending `prevLiveInfo),` at line 1769–1771) add (still inside the `{ ... }` initializer, before its closing `};`):

```cpp
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 11, 0,
                               vk::DescriptorType::eStorageImage, densityInfo0),
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 12, 0,
                               vk::DescriptorType::eStorageImage, densityInfo1),
        vk::WriteDescriptorSet(*compute.descriptorSets[i], 13, 0,
                               vk::DescriptorType::eStorageImage, densityInfo2),
```

- [ ] **Step 7: Clear density images on restart.** In `paint_splatter.cpp` `restartSimulation`, replace the canvas white-clear block (lines 349–359) with:

```cpp
  // Clear the painting back to blank white paper: zero the density accumulators
  // (resolve then yields white) and also clear the RGBA8 canvas white directly
  // so the very first frame before the next resolve is already blank. All stay
  // in GENERAL.
  {
    vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0,
                                    1);
    vgeu::oneTimeSubmit(
        device, commandPool, queue, [&](const vk::raii::CommandBuffer& cmd) {
          vk::ClearColorValue white(std::array<float, 4>{1.f, 1.f, 1.f, 1.f});
          cmd.clearColorImage(canvasImage->getImage(),
                              vk::ImageLayout::eGeneral, white, range);
          vk::ClearColorValue zero(std::array<uint32_t, 4>{0u, 0u, 0u, 0u});
          for (int k = 0; k < 3; k++) {
            cmd.clearColorImage(densityImages[k]->getImage(),
                                vk::ImageLayout::eGeneral, zero, range);
          }
        });
  }
```

- [ ] **Step 8: clang-format.**

Run: `rtk clang-format -i src/examples/paint_splatter/paint_splatter.cpp src/examples/paint_splatter/paint_splatter.hpp`
Expected: success.

- [ ] **Step 9: Build, run, VL-CLEAN.**

Run: `rtk cmd /c mingwBuild.bat`
Then: `rtk ./build/paint_splatter.exe`
Expected: builds; runs identically to today (deposit still writes the canvas via the OLD alpha-over path — unchanged in this task; density images are bound but unused). Zero validation errors. Close the window.

- [ ] **Step 10: Commit.**

```bash
rtk git add src/examples/paint_splatter/paint_splatter.hpp src/examples/paint_splatter/paint_splatter.cpp
rtk git commit -m "feat(paint_splatter): allocate + bind density accumulator images"
```

---

## Task 3 — `resolve.comp` shader + pipeline (built, not yet dispatched)

Add the resolve shader and build its pipeline. Not dispatched yet, so no visual change.

**Files:**
- Create: `shaders/paint_splatter/resolve.comp`
- Modify: `src/examples/paint_splatter/paint_splatter.hpp` (`compute` struct, after `deposit` member at line 594)
- Modify: `src/examples/paint_splatter/paint_splatter.cpp` (`createComputePipeline`, after line 1202)

- [ ] **Step 1: Write `resolve.comp`.** Create `shaders/paint_splatter/resolve.comp` with:

```glsl
#version 450
layout(local_size_x = 16, local_size_y = 16) in;

// Resolve pass (subtractive color mixing): convert the per-RGB-channel optical
// density accumulators into a displayable reflectance and write the RGBA8
// canvas. reflectance = exp(-D); D = 0 => white paper, larger D => darker and
// more saturated (Beer-Lambert glazing). Runs on the GRAPHICS queue right after
// deposit, before the render pass samples canvasImg.

layout(std140, set = 0, binding = 1) uniform Ubo {
  float dt;
  uint count;
  float gravity;
  float h;
  vec4 cmin;
  vec4 cmax;
  ivec4 gridDim;
  float rho0;
  float epsCFM;
  float scorrK;
  float scorrDq;
  float scorrN;
  float xsphC;
  float kPoly6;
  float kSpiky;
  float scorrDenom;
  float velDamp;
  float velClampFactor;
  float solverRelax;
  uint prevCount;
  float dryRate;
  float depositStrength;
  float depositHeight;
  float depositRadius;
  float drySettle;
  float pad0;
  float pad1;
  float densityScale;
  float maxDensity;
  float pad2;
  float pad3;
}
u;

layout(set = 0, binding = 8, rgba8) uniform image2D canvasImg;
layout(set = 0, binding = 11, r32ui) uniform uimage2D densityR;
layout(set = 0, binding = 12, r32ui) uniform uimage2D densityG;
layout(set = 0, binding = 13, r32ui) uniform uimage2D densityB;

void main() {
  ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
  ivec2 size = imageSize(canvasImg);
  if (texel.x >= size.x || texel.y >= size.y) return;

  // Fixed-point uint accumulators -> float optical density per channel.
  vec3 d = vec3(float(imageLoad(densityR, texel).r),
                float(imageLoad(densityG, texel).r),
                float(imageLoad(densityB, texel).r)) /
           max(u.densityScale, 1.0);
  d = min(d, vec3(u.maxDensity));
  vec3 rgb = exp(-d);  // d=0 -> white; subtractive glazing as d grows
  imageStore(canvasImg, texel, vec4(clamp(rgb, 0.0, 1.0), 1.0));
}
```

- [ ] **Step 2: Add the pipeline member.** In `paint_splatter.hpp`, immediately AFTER `vk::raii::Pipeline deposit = nullptr;` (line 594) add:

```cpp
    // Resolve (color-mixing): density accumulators -> RGBA8 canvas. Shares the
    // compute pipelineLayout (binding 8 canvas + 11-13 density). Dispatched on
    // the GRAPHICS queue right after deposit (see buildCommandBuffers).
    vk::raii::Pipeline resolve = nullptr;
```

- [ ] **Step 3: Build the pipeline.** In `paint_splatter.cpp` `createComputePipeline`, immediately AFTER `compute.deposit = makePipeline("deposit");` (line 1202) add:

```cpp
  compute.resolve = makePipeline("resolve");
```

- [ ] **Step 4: Reconfigure + build (new shader).**

Run: `rtk cmd /c mingwBuild.bat`
Expected: build succeeds AND `shaders/paint_splatter/resolve.comp.spv` is produced. If the `.spv` is missing, re-run the configure step (or delete `build/CMakeCache.txt` then rebuild) so GLOB_RECURSE re-scans, and rebuild. Confirm `resolve.comp.spv` exists before continuing.

- [ ] **Step 5: Run, VL-CLEAN.**

Run: `rtk ./build/paint_splatter.exe`
Expected: runs identically (resolve pipeline is built but never dispatched yet). Zero validation errors. Close the window.

- [ ] **Step 6: Commit.**

```bash
rtk git add shaders/paint_splatter/resolve.comp src/examples/paint_splatter/paint_splatter.hpp src/examples/paint_splatter/paint_splatter.cpp
rtk git commit -m "feat(paint_splatter): add resolve.comp (density -> reflectance) pipeline"
```

---

## Task 4 — Switchover: density-accumulating deposit + resolve dispatch + barriers

Rewrite `deposit.comp` to accumulate optical density (atomic, order-independent) instead of alpha-over, and dispatch `resolve.comp` between deposit and the render pass. This is the behavior change.

**Files:**
- Modify: `shaders/paint_splatter/deposit.comp` (whole body + Ubo block)
- Modify: `src/examples/paint_splatter/paint_splatter.cpp` (`buildCommandBuffers` deposit block, lines 2337–2355)

- [ ] **Step 1: Extend the deposit Ubo block.** In `shaders/paint_splatter/deposit.comp`, replace the tail of the `Ubo` block — the `float drySettle; float pad0; float pad1;` lines (the last three fields before `} u;`) — with:

```glsl
  float drySettle;
  float pad0;
  float pad1;
  float densityScale;  // fixed-point scale for atomic density accumulation
  float maxDensity;    // (unused here; read by resolve.comp)
  float pad2;
  float pad3;
```

- [ ] **Step 2: Add the density image bindings.** In `deposit.comp`, replace the canvas storage-image binding line `layout(set = 0, binding = 8, rgba8) uniform image2D canvasImg;` with:

```glsl
// Canvas (binding 8) is no longer written here -- resolve.comp owns it now.
// Deposit accumulates per-RGB-channel optical density into r32ui images.
layout(set = 0, binding = 11, r32ui) uniform uimage2D densityR;
layout(set = 0, binding = 12, r32ui) uniform uimage2D densityG;
layout(set = 0, binding = 13, r32ui) uniform uimage2D densityB;
```

Then, in the unchanged body below, change the canvas-size line `ivec2 size = imageSize(canvasImg);` (originally line ~66) to read from a density image instead (all four images share `kCanvasTexRes`):

```glsl
  ivec2 size = imageSize(densityR);
```

- [ ] **Step 3: Replace the stamp loop.** In `deposit.comp`, replace the body from `float a = clamp(p[i].color.a * u.depositStrength, 0.0, 1.0);` through the end of the nested `for` loops (the block that does the rgba8 `imageLoad`/alpha-over/`imageStore`, ending at the `}` closing the `dy` loop) with:

```glsl
  // Subtractive mixing in optical-density space. A paint colour `c` (reflectance
  // on white, in (0,1]) has per-channel density D = -ln(clamp(c, EPS, 1)); EPS
  // caps a fully-saturated channel at a finite density. Contributions ADD, and
  // addition is order-independent -> no flicker, and accumulation darkens
  // (glazing). Fixed-point: imageAtomicAdd(uint(w * D * densityScale)).
  const float EPS = 1.0 / 255.0;
  vec3 dPaint = -log(clamp(p[i].color.rgb, vec3(EPS), vec3(1.0)));
  float wBase = max(p[i].color.a * u.depositStrength, 0.0);
  int r2max = rad * rad;
  for (int dy = -rad; dy <= rad; dy++) {
    for (int dx = -rad; dx <= rad; dx++) {
      int r2 = dx * dx + dy * dy;
      if (r2 > r2max) continue;
      ivec2 texel = clamp(center + ivec2(dx, dy), ivec2(0), size - 1);
      // Smooth center->edge falloff so the disk has a soft edge.
      float fr = 1.0 - sqrt(float(r2)) / float(rad);
      fr = clamp(fr, 0.0, 1.0);
      float falloff = fr * fr * (3.0 - 2.0 * fr);  // smoothstep
      vec3 contrib = dPaint * (wBase * falloff * u.densityScale);
      imageAtomicAdd(densityR, texel, uint(contrib.r));
      imageAtomicAdd(densityG, texel, uint(contrib.g));
      imageAtomicAdd(densityB, texel, uint(contrib.b));
    }
  }
```

Note: the lines ABOVE this block (the `i >= u.count` / floor-height / `pos.w <= 0` / uv-range guards, the `center` computation, `texelWorld`, and `rad`) are unchanged — only the colour/stamp portion is replaced.

- [ ] **Step 4: Dispatch resolve + barriers.** In `paint_splatter.cpp` `buildCommandBuffers`, replace the deposit block (the `if (numParticles > 0) { ... }` at lines 2337–2355, from `drawCmdBuffers[...].bindPipeline(... *compute.deposit)` through the closing `}` after the `canvasBarrier` pipelineBarrier) with:

```cpp
  if (numParticles > 0) {
    // Deposit: atomic-add per-channel optical density into the 3 density images.
    drawCmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eCompute, *compute.deposit);
    drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 0,
        *compute.descriptorSets[currentFrameIndex], nullptr);
    drawCmdBuffers[currentFrameIndex].dispatch((numParticles + 255u) / 256u, 1,
                                               1);
    // Barrier: deposit writes (atomic) -> resolve reads, on the 3 density imgs.
    std::array<vk::ImageMemoryBarrier, 3> densityBarriers;
    for (int k = 0; k < 3; k++) {
      densityBarriers[k] = vk::ImageMemoryBarrier(
          vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
          vk::ImageLayout::eGeneral, vk::ImageLayout::eGeneral,
          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
          densityImages[k]->getImage(),
          vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0,
                                    1));
    }
    drawCmdBuffers[currentFrameIndex].pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags{},
        nullptr, nullptr, densityBarriers);

    // Resolve: density -> reflectance -> RGBA8 canvas. One thread per texel.
    drawCmdBuffers[currentFrameIndex].bindPipeline(
        vk::PipelineBindPoint::eCompute, *compute.resolve);
    drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, *compute.pipelineLayout, 0,
        *compute.descriptorSets[currentFrameIndex], nullptr);
    const uint32_t groups = (kCanvasTexRes + 15u) / 16u;
    drawCmdBuffers[currentFrameIndex].dispatch(groups, groups, 1);

    // Barrier: resolve writes canvas -> fragment sample in the render pass.
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
```

- [ ] **Step 5: clang-format (cpp only; shaders are not formatted).**

Run: `rtk clang-format -i src/examples/paint_splatter/paint_splatter.cpp`
Expected: success.

- [ ] **Step 6: Build.**

Run: `rtk cmd /c mingwBuild.bat`
Expected: builds; `deposit.comp.spv` and `resolve.comp.spv` recompiled.

- [ ] **Step 7: Run, VL-CLEAN + behavior smoke check.**

Run: `rtk ./build/paint_splatter.exe`
Expected: zero validation errors/warnings. Paint now appears on the canvas via the resolve path (white paper where untouched). Drop paint and confirm: (a) marks appear, (b) repeated coverage on one spot gets DARKER (glazing), (c) overlapping two colors does NOT flicker as particles reorder. Press Restart → canvas returns to white. Close the window.

- [ ] **Step 8: Commit.**

```bash
rtk git add shaders/paint_splatter/deposit.comp src/examples/paint_splatter/paint_splatter.cpp
rtk git commit -m "feat(paint_splatter): subtractive density deposit + resolve switchover"
```

---

## Task 5 — User visual GATE + debug-log entry

**Files:**
- Modify: `docs/paint_splatter_debug_log.md`

- [ ] **Step 1: GATE (user).** Ask the user to run `rtk ./build/paint_splatter.exe` and visually confirm:
  - White paper where no paint; marks appear correctly.
  - Same-spot repeated stamps darken (glazing), no white-dilution paleness.
  - Two overlapping colors: no flicker (order-independent); **cyan + yellow → green** reads correctly. (Note honestly: RGB-blue + yellow reads muddy, not vivid green — that needs spectral/KM, out of scope.)
  - Restart returns the canvas to white.
  - Saved PNG (Save button) matches the on-screen canvas.

  **STOP. Do not proceed without the user's approval.** If the user wants tuning, expose/adjust `densityScale`, `maxDensity`, `depositStrength`, `depositRadius` live and iterate before continuing.

- [ ] **Step 2: Record the milestone.** In `docs/paint_splatter_debug_log.md`, append a new section at the end:

```markdown
## M-color — Subtractive (optical-density) color mixing (2026-06-07)

Replaced the canvas alpha-over deposit with order-independent subtractive mixing
in optical-density space (design spec
`docs/superpowers/specs/2026-06-07-paint-splatter-subtractive-color-mixing-design.md`).

- **Model:** per-RGB-channel optical density `D = -ln(clamp(c, 1/255, 1))`,
  weighted by `concentration * depositStrength * smoothstep-falloff`, accumulated
  via `imageAtomicAdd` (fixed-point, scale `densityScale`) into 3 `R32_UINT`
  images. Addition is commutative -> order-independent (kills the compaction
  reorder flicker) and subtractive (glazing: more coverage darkens). A new
  `resolve.comp` writes `reflectance = exp(-min(D, maxDensity))` into the
  existing RGBA8 canvas each frame; `canvas.frag` + PNG path unchanged.
- **Bindings:** compute set 11/12/13 = densityR/G/B; binding 8 (canvas) now
  written only by resolve. Pool storage-images MAX*1 -> MAX*4. ComputeUbo
  144 -> 160 (+densityScale, maxDensity, 2 pad).
- **Known limitation (accepted):** RGB-primary subtractive gives cyan+yellow->
  green well, but artist "blue+yellow->green" reads muddy (RGB blue lacks a cyan
  component); a vivid result needs spectral / Kubelka-Munk (Mixbox). Deferred.
- **Verification:** VL-CLEAN; glazing + flicker-free overlaps confirmed on
  screen (user gate); restart clears to white; PNG matches.
```

- [ ] **Step 3: Commit.**

```bash
rtk git add docs/paint_splatter_debug_log.md
rtk git commit -m "docs(paint_splatter): record subtractive color-mixing milestone"
```

---

## Self-Review notes (addressed)

- **Spec coverage:** density model + atomic accumulation (Task 4 deposit + Task 2 images), resolve pass (Task 3/4), 3× R32_UINT storage + pool/layout/writes (Task 2), ComputeUbo fields + UI (Task 1), restart clear (Task 2), PNG/`canvas.frag` unchanged (verified, untouched), honest limitation recorded (Task 5). All spec sections map to a task.
- **Type/name consistency:** `densityImages` (`std::array<unique_ptr<VgeuImage>,3>`), `compute.resolve`, UBO `densityScale`/`maxDensity`, bindings 11/12/13, write-array size 14, pool `MAX*4`, `static_assert(... == 160)` are used identically across tasks.
- **Ordering:** images created (Task 2) before layout/pool/writes reference them (same task); resolve pipeline built (Task 3) before dispatched (Task 4); deposit shader rewrite and resolve dispatch land together (Task 4) so the canvas never goes dark mid-plan.
```
