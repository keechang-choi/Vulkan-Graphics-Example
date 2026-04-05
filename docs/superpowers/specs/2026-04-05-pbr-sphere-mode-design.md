# PBR Example: Debug View Toggle & Sphere Mode

**Date:** 2026-04-05  
**Branch:** claude-test

## Overview

Add two runtime-toggleable options to the PBR deferred rendering example:
1. **Debug View Toggle** — show/hide the G-buffer sub-viewport debug views in the top-right
2. **Sphere Mode** — replace helmets+floor with a 4×4 sphere grid where rows vary metallic (0→1) and columns vary roughness (0→1), with a configurable albedo color

---

## Section 1: Options & Data Structures

### `pbr.hpp` — Options additions

```cpp
bool showDebugViews = true;
bool useSpheres = false;
std::array<float, 4> sphereAlbedo = {1.0f, 1.0f, 1.0f, 1.0f};
```

### `pbr.hpp` — DynamicUboElt extended (80 → 96 bytes)

```cpp
struct DynamicUboElt {
  glm::mat4 modelMatrix{1.f};  // 64 bytes
  glm::vec4 modelColor{0.f};   // 16 bytes
  glm::vec4 pbrOverride{0.f};  // 16 bytes: x=metallic, y=roughness, z=useOverride(0/1), w=unused
};
```

### `pbr.hpp` — ModelInstance sceneMode

```cpp
enum class SceneMode { kModelOnly, kSphereOnly };
// added as member of ModelInstance:
SceneMode sceneMode = SceneMode::kModelOnly;
```

- `floor` → `kModelOnly`
- `damagedHelmet_i-j` → `kModelOnly`
- `sphere_i-j` → `kSphereOnly`

---

## Section 2: Textures & Shaders

### Dummy textures (C++, 1×1 pixel each)

Created in `loadAssets()` after sphere model is loaded. Uploaded via staging buffer.

| Member | Format | Pixel value | Purpose |
|---|---|---|---|
| `sphereDummyAlbedo` | R8G8B8A8_UNORM | (255,255,255,255) | white, overridden by modelColor |
| `sphereDummyNormal` | R8G8B8A8_UNORM | (128,128,255,255) | flat +Z normal |
| `sphereDummyMetRough` | R8G8B8A8_UNORM | (0,128,0,255) | mid roughness, overridden |
| `sphereDummyEmissive` | R8G8B8A8_UNORM | (0,0,0,255) | black |

Sampled via a dedicated `sphereDummyDescriptorSet` at set 2, bound before sphere draw calls.

### `mrt.vert` — ModelUbo struct extended

```glsl
layout (set = 1, binding = 0) uniform ModelUbo {
    mat4 modelMatrix;
    vec4 modelColor;
    vec4 pbrOverride;  // x=metallic, y=roughness, z=useOverride(0/1)
} modelUbo;
```

### `mrt.frag` — ARM override logic

```glsl
vec3 arm;
if (modelUbo.pbrOverride.z > 0.5) {
    arm = vec3(0.0, modelUbo.pbrOverride.y, modelUbo.pbrOverride.x);  // g=roughness, b=metallic
} else {
    arm.rgb = texture(samplerMetallicRoughnessMap, inUV).rgb;
}
```

Albedo is handled by existing `mix(albedo.rgb, inColor.rgb, inColor.a)`. For spheres, `modelColor = vec4(sphereAlbedo.rgb, 1.0)` fully overrides texture albedo.

---

## Section 3: Model Loading & Instance Setup

### `loadAssets()` — sphere model always loaded

- Load `assets/models/sphere/untitled.gltf` with same glTFLoadingFlags as helmet
- Create 4×4 `sphere_i-j` ModelInstances, all with `sceneMode = kSphereOnly`
- Create 1×1 dummy textures and their descriptor set

### `setupDynamicUbo()` — sphere grid values

```
row i (세로, Z axis) → metallic:  i / (modelNumZ - 1)  [0→1]
col j (가로, X axis) → roughness: j / (modelNumX - 1)  [0→1]
```

```cpp
dynamicUbo[idx].pbrOverride = glm::vec4(metallic, roughness, 1.f, 0.f);
dynamicUbo[idx].modelColor  = glm::vec4(opts.sphereAlbedo[0], opts.sphereAlbedo[1],
                                         opts.sphereAlbedo[2], 1.0f);
```

Sphere positions are identical to helmet grid positions (same x/y/z spacing). Floor is hidden (`kModelOnly`) when sphere mode is on.

### `buildCommandBuffers()` — mode-based draw skip

```cpp
for (size_t instIdx = 0; instIdx < modelInstances.size(); instIdx++) {
    const auto& inst = modelInstances[instIdx];
    if (inst.sceneMode == SceneMode::kModelOnly && opts.useSpheres) continue;
    if (inst.sceneMode == SceneMode::kSphereOnly && !opts.useSpheres) continue;
    // bind set 2 dummy textures for sphere, skip kBindImages flag
    if (inst.sceneMode == SceneMode::kSphereOnly) {
        cmd.bindDescriptorSets(..., pipelineLayoutOffScreen, 2,
            {*sphereDummyDescriptorSets[currentFrameIndex]}, nullptr);
        inst.model->draw(currentFrameIndex, cmd, 0 /*no kBindImages*/,
                         *pipelineLayoutOffScreen, 2);
    } else {
        inst.model->draw(currentFrameIndex, cmd, vgeu::RenderFlagBits::kBindImages,
                         *pipelineLayoutOffScreen, 2);
    }
}
```

Debug view toggle:
```cpp
if (opts.showDebugViews) {
    for (int i = 1; i < opts.numTargets; i++) { /* sub-viewport draws */ }
}
```

---

## Section 4: UI & Descriptor Binding

### `onUpdateUIOverlay()` additions

```cpp
ImGui::Checkbox("Show Debug Views", &opts.showDebugViews);
ImGui::Checkbox("Use Spheres", &opts.useSpheres);
if (opts.useSpheres) {
    uiOverlay->colorPicker("Sphere Albedo", opts.sphereAlbedo.data());
}
```

### `updateDynamicUbo()` (called every frame)

When sphere mode is active, sync modelColor from `opts.sphereAlbedo` each frame so color picker changes are reflected immediately.

### Descriptor pool sizing

The pool count for dynamic UBO sets must account for all instances (helmets + spheres + floor). Already computed from `modelInstances.size()` — no manual change needed since instances are added to `modelInstances` during `loadAssets()`.

Sphere dummy descriptor set layout = same as set 2 layout used for glTF materials (4 combined image samplers). Allocated separately from the pool.

---

## Files Changed

| File | Change |
|---|---|
| `src/examples/pbr/pbr.hpp` | Options fields, DynamicUboElt pbrOverride, ModelInstance sceneMode, dummy texture/descriptor members |
| `src/examples/pbr/pbr.cpp` | loadAssets, setupDynamicUbo, setupDescriptors, buildCommandBuffers, onUpdateUIOverlay, updateUboComposition |
| `shaders/pbr/mrt.vert` | ModelUbo pbrOverride field |
| `shaders/pbr/mrt.frag` | ARM override logic |

---

## Open Questions / Risks

- **Sphere model vertex attributes**: `assets/models/sphere/untitled.gltf` must export position, UV, normal, tangent. If tangent is missing, the TBN matrix in mrt.frag will produce artifacts (mitigated by flat normal map — flat normal map output means TBN accuracy doesn't matter much visually).
- **DynamicUboElt size change**: Existing code computes `alignedSizeDynamicUboElt` at runtime via `padUniformBufferSize(sizeof(DynamicUboElt))`, so changing the struct size automatically updates alignment. No manual fix needed.
- **updateDynamicUbo timing**: sphereAlbedo must be flushed to GPU each frame when sphere mode is active. Current code already calls update functions per frame in `draw()`.
