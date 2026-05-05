# Soap Bubble alphaBase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `alphaBase` opacity-floor parameter to the soap bubble shader, exposed as an ImGui slider [0,1] and a CLI flag, to lift head-on (low-fresnel) regions of the bubble surface (and serve as a diagnostic for back-face show-through by allowing full opacity).

**Architecture:** One float traveling through the existing UBO chain: `Options` (host) → `updateBubbleParamsUbo()` → `BubbleParamsUbo` (std140) → `bubble.frag` `BubbleParams` block → final `alpha` calculation. CLI exposure mirrors the existing `--alphaScale` pattern. ImGui slider added next to the existing `alphaScale` slider.

**Tech Stack:** C++17, Vulkan 1.x via Vulkan-Hpp/raii, GLSL 450, ImGui, glslangValidator (CMake-driven), CLI11.

**Spec:** `docs/superpowers/specs/2026-05-05-soap-bubble-alpha-base-design.md`

---

## File Structure

Files modified (no new files):

| File | Responsibility for this change |
|---|---|
| `src/examples/soap_bubble/soap_bubble.hpp` | Add `alphaBase` to `Options` struct and `BubbleParamsUbo` struct (matching std140 layout) |
| `src/examples/soap_bubble/soap_bubble.cpp` | CLI option, UBO field write-through, ImGui slider |
| `shaders/soap_bubble/bubble.frag` | Add `alphaBase` to `BubbleParams` UBO block; use it in the alpha line |

The existing `BubbleParamsUbo` struct ends with two `float _pad0;` `float _pad1;` slots (`soap_bubble.hpp:73-74`). We insert `alphaBase` next to `alphaScale` to keep semantically related fields grouped, then consume one pad to keep the struct size constant. The frag UBO block must be reordered identically.

**Existing std140 layout:**
```
| thicknessMin | thicknessMax | n1            | n2     |  block 1
| n3           | spectralS    | thicknessMode | gravS  |  block 2
| noiseScale   | useAnim      | driftSpeed    | rough  |  block 3
| alphaScale   | iblExposure  | iblGamma      | time   |  block 4
| showHeat     | showFresnel  | _pad0         | _pad1  |  block 5
```

**New std140 layout:**
```
| thicknessMin | thicknessMax | n1            | n2     |  block 1
| n3           | spectralS    | thicknessMode | gravS  |  block 2
| noiseScale   | useAnim      | driftSpeed    | rough  |  block 3
| alphaScale   | alphaBase    | iblExposure   | iblGamma| block 4   ← changed
| time         | showHeat     | showFresnel   | _pad0  |  block 5   ← changed
```

Net size unchanged (one `_pad1` slot replaced by `alphaBase`).

---

## Task 1: Add `alphaBase` to host structs

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.hpp:30` (`Options::alphaScale` block)
- Modify: `src/examples/soap_bubble/soap_bubble.hpp:50-75` (`BubbleParamsUbo`)

- [ ] **Step 1: Add `alphaBase` to `Options`**

In `soap_bubble.hpp`, locate:
```cpp
  // Surface & Blending
  float roughness = 0.0f;
  float alphaScale = 1.0f;
```

Change to:
```cpp
  // Surface & Blending
  float roughness = 0.0f;
  float alphaScale = 1.0f;
  float alphaBase = 0.0f;
```

- [ ] **Step 2: Reorder `BubbleParamsUbo` to insert `alphaBase`**

Replace the existing struct body (`soap_bubble.hpp:50-75`) with:

```cpp
struct BubbleParamsUbo {
  float thicknessMin;
  float thicknessMax;
  float n1;
  float n2;
  // -- 16 byte boundary --
  float n3;
  int32_t spectralSamples;
  int32_t thicknessMode;
  float gravityStrength;
  // -- 16 --
  float noiseScale;
  int32_t useAnimation;
  float driftSpeed;
  float roughness;
  // -- 16 --
  float alphaScale;
  float alphaBase;
  float iblExposure;
  float iblGamma;
  // -- 16 --
  float time;
  int32_t showThicknessHeatmap;
  int32_t showFresnelOnly;
  float _pad0;
};
```

Note: `_pad1` is gone (consumed by `alphaBase`). Total size unchanged.

- [ ] **Step 3: Build to confirm header compiles**

Run: `rtk cmake --build build --target soap_bubble`
Expected: build succeeds. (Compile fails if a UBO write referencing `_pad1` still exists, but `updateBubbleParamsUbo()` does not write `_pad0/_pad1`, so this should be clean.)

- [ ] **Step 4: Commit**

```bash
rtk git add src/examples/soap_bubble/soap_bubble.hpp
rtk git commit -m "feat(soap_bubble): add alphaBase field to host UBO struct"
```

---

## Task 2: Wire `alphaBase` through `updateBubbleParamsUbo` and CLI

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.cpp:28` (CLI option list)
- Modify: `src/examples/soap_bubble/soap_bubble.cpp:467` (UBO write-through)

- [ ] **Step 1: Add CLI option**

In `soap_bubble.cpp`, locate:
```cpp
  app.add_option("--alphaScale", opts.alphaScale);
```

Add a new line directly below:
```cpp
  app.add_option("--alphaScale", opts.alphaScale);
  app.add_option("--alphaBase", opts.alphaBase);
```

- [ ] **Step 2: Add UBO write-through**

In `updateBubbleParamsUbo()`, locate:
```cpp
  bubbleParamsUbo.alphaScale = opts.alphaScale;
```

Add a new line directly below:
```cpp
  bubbleParamsUbo.alphaScale = opts.alphaScale;
  bubbleParamsUbo.alphaBase = opts.alphaBase;
```

- [ ] **Step 3: Build to confirm**

Run: `rtk cmake --build build --target soap_bubble`
Expected: build succeeds.

- [ ] **Step 4: Commit**

```bash
rtk git add src/examples/soap_bubble/soap_bubble.cpp
rtk git commit -m "feat(soap_bubble): wire alphaBase through CLI and UBO update"
```

---

## Task 3: Add `alphaBase` to fragment shader UBO and use it

**Files:**
- Modify: `shaders/soap_bubble/bubble.frag:9-30` (`BubbleParams` block)
- Modify: `shaders/soap_bubble/bubble.frag:194` (alpha calc)

- [ ] **Step 1: Reorder shader UBO block to match host**

Replace the existing `layout(set = 1, binding = 0) uniform BubbleParams { ... } params;` block (`bubble.frag:9-30`) with:

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
  float alphaScale;
  float alphaBase;
  float iblExposure;
  float iblGamma;
  float time;
  int showThicknessHeatmap;
  int showFresnelOnly;
  float _pad0;
} params;
```

(Order must mirror `BubbleParamsUbo` exactly; `_pad1` is gone.)

- [ ] **Step 2: Use `alphaBase` in alpha calculation**

In `bubble.frag`, locate:
```glsl
  float fresnel = fresnelSchlick(cosTheta1, params.n1, params.n2);
  float alpha = clamp(fresnel * params.alphaScale, 0.0, 1.0);
```

Change the second line to:
```glsl
  float fresnel = fresnelSchlick(cosTheta1, params.n1, params.n2);
  float alpha =
      clamp(params.alphaBase + fresnel * params.alphaScale, 0.0, 1.0);
```

- [ ] **Step 3: Build (recompiles `bubble.frag.spv` via CMake glslangValidator rule)**

Run: `rtk cmake --build build --target soap_bubble`
Expected: build succeeds; `shaders/soap_bubble/bubble.frag.spv` regenerates.

- [ ] **Step 4: Commit**

```bash
rtk git add shaders/soap_bubble/bubble.frag shaders/soap_bubble/bubble.frag.spv
rtk git commit -m "feat(soap_bubble): use alphaBase floor in fragment alpha"
```

(Note: `.spv` is checked in alongside source per the existing repo convention — verify with `rtk git ls-files shaders/soap_bubble/`.)

---

## Task 4: Add ImGui slider

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.cpp:401` (Surface & Blending header)

- [ ] **Step 1: Add the slider**

Locate:
```cpp
  if (ImGui::CollapsingHeader("Surface & Blending",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::SliderFloat("roughness", &opts.roughness, 0.f, 1.f);
    ImGui::SliderFloat("alphaScale", &opts.alphaScale, 0.f, 3.f);
  }
```

Change to:
```cpp
  if (ImGui::CollapsingHeader("Surface & Blending",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::SliderFloat("roughness", &opts.roughness, 0.f, 1.f);
    ImGui::SliderFloat("alphaScale", &opts.alphaScale, 0.f, 3.f);
    ImGui::SliderFloat("alphaBase", &opts.alphaBase, 0.f, 1.f);
  }
```

- [ ] **Step 2: Build**

Run: `rtk cmake --build build --target soap_bubble`
Expected: build succeeds.

- [ ] **Step 3: Visual smoke test**

Run: `./build/src/examples/soap_bubble/soap_bubble.exe` (or wherever the binary lands — confirm with `rtk ls build/src/examples/soap_bubble`).

Expected visual checks (from spec test plan):
1. With `alphaBase = 0`, the scene looks identical to the prior build.
2. Slide `alphaBase` to `1.0` → bubble is fully opaque, no see-through.
3. At `alphaBase ≈ 0.3`, head-on regions visibly lift while rim still brightens via `alphaScale * fresnel`.

If any check fails, do not commit — debug first.

- [ ] **Step 4: clang-format the modified .cpp**

Run: `rtk clang-format -i src/examples/soap_bubble/soap_bubble.cpp src/examples/soap_bubble/soap_bubble.hpp`
Expected: no errors. Diff may be empty if formatting was already clean.

- [ ] **Step 5: Commit**

```bash
rtk git add src/examples/soap_bubble/soap_bubble.cpp src/examples/soap_bubble/soap_bubble.hpp
rtk git commit -m "feat(soap_bubble): expose alphaBase slider in ImGui"
```

(The .hpp is included only in case clang-format produced a diff there; if `rtk git status` shows no .hpp change, omit it from `git add`.)

---

## Self-Review Notes

- **Spec coverage:** Spec sections "Touch points 1-6" all map to Tasks 1-4. CLI flag (touch point 4 implicit via `--alphaScale` pattern) is covered in Task 2 step 1. Test plan steps map to Task 4 step 3.
- **Placeholder scan:** No TBDs/TODOs. All code shown.
- **Type consistency:** `alphaBase` is `float` everywhere (host struct, UBO, GLSL). Field order in `BubbleParamsUbo` matches frag UBO block 1:1. Slider range matches default value (0.0) and forced-opaque endpoint (1.0).
- **Out of scope (per spec):** Diagnosing the seam itself; fixing back-face show-through if that's the root cause. Those are follow-up changes after this lands.
