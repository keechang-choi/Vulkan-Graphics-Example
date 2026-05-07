# Soap Bubble — Thin-film R/T + Background Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite bubble fragment shader with the corrected amplitude-form thin-film interference formula, switch from Fresnel-as-alpha alpha-blending to energy-balanced R/T compositing (alpha = 1) sampled from the prefiltered cubemap, and add an N-extensible background scene system (4 hardcoded glTF instances, push-constant transform + Lambert/IBL-irradiance lighting) as scaffolding for a future SSR spec.

**Architecture:** Three logical stages.
**Stage 1** rewrites only the bubble pass: shader physics + UBO/Options/UI/CLI lockstep change — all modifications happen together because any partial change leaves the SPIR-V/host structs out of sync.
**Stage 2** adds the background scene: data structs, shaders, pipeline, descriptor set — all wired but not yet drawn.
**Stage 3** activates background draws + ImGui/CLI controls + visual/validation sign-off.

**Tech Stack:** Vulkan-HPP RAII, VMA, glm, GLSL 450, ImGui, CLI11, MinGW + Ninja, clang-format.

**Reference Spec:** `docs/superpowers/specs/2026-05-08-soap-bubble-thin-film-rt-design.md`

---

## Working environment

- All bash commands MUST use the `rtk` prefix (e.g. `rtk git add`, `rtk clang-format`, `rtk ./mingwBuild.bat Debug`).
- All `.cpp/.hpp` edits MUST be passed through `rtk clang-format -i <file>` before commit.
- Build: `rtk ./mingwBuild.bat Debug` (cd to repo root). It runs `ninja all && ninja Shaders`, producing `build/soap_bubble.exe` and updated `*.spv` next to each `.glsl`.
- Run: `rtk ./build/soap_bubble.exe` (validation layer on for Debug build). New VUID errors must be 0. Pre-existing `VUID-02697` from PBR is unrelated and remains out of scope.
- Working directory throughout: repo root `C:\Users\rlckd\Desktop\kc\Vulkan-Graphics-Example`.

---

## File map

| Status   | Path | Role |
|---|---|---|
| **Modify** | `shaders/soap_bubble/bubble.frag` | Amplitude-form thin-film, R/T composite, refract, `rtMode` debug, alpha=1 (alphaScale/Base removed) |
| **Create** | `shaders/soap_bubble/bg.vert` | Background vertex shader (push-constant model) |
| **Create** | `shaders/soap_bubble/bg.frag` | Background fragment shader (Lambert + irradiance, push-constant baseColor) |
| **Modify** | `src/examples/soap_bubble/soap_bubble.hpp` | Trim Options (drop alpha knobs), add `rtMode`, add `BgInstance`, `BgPushConstant`, `backgrounds` defaults, bg pipeline/desc handles |
| **Modify** | `src/examples/soap_bubble/soap_bubble.cpp` | Bubble pipeline blend off + cull eBack, bg pass infrastructure (load/desc/pipeline/draw), ImGui R/T Debug + Background groups, CLI `--rtMode` + `--bgEnable` |

`bubble.vert` is unchanged. `vgeu_ibl` requires no edits (`IBLBaker::irradianceMap()` already exists).

CMake auto-discovers new `.cpp/.glsl` files via glob, so no CMake edits are needed.

---

# Stage 1 — Bubble physics & R/T compositing

Goal: bubble shader produces saturated thin-film color and a complementary transmission tint with no alpha-blending fudge. This is one logical commit because the shader UBO block, host `BubbleParamsUbo`, `Options`, `updateBubbleParamsUbo`, ImGui, CLI, and pipeline state all change together — partial changes wedge the binding mismatch.

---

## Task 1: Bubble shader + UBO/Options/UI/CLI/pipeline lockstep

**Files:**
- Modify: `shaders/soap_bubble/bubble.frag`
- Modify: `src/examples/soap_bubble/soap_bubble.hpp`
- Modify: `src/examples/soap_bubble/soap_bubble.cpp`

- [ ] **Step 1: Rewrite the bubble fragment shader**

Open `shaders/soap_bubble/bubble.frag` and replace its contents with:

```glsl
#version 450

layout(set = 0, binding = 0) uniform Globals {
  mat4 view;
  mat4 projection;
  mat4 model;
  vec4 viewPos;
} globals;

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

layout(set = 2, binding = 0) uniform sampler2D heightTex;
layout(set = 3, binding = 0) uniform samplerCube prefilteredCubemap;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846;

vec3 wymanCMF(float lambda) {
  float x = 0.398 * exp(-1250.0 * pow(log((lambda + 570.1) / 1014.0), 2.0)) +
            1.132 * exp(-234.0 * pow(log((1338.0 - lambda) / 743.5), 2.0));
  float y = 1.011 * exp(-0.5 * pow((lambda - 556.1) / 46.14, 2.0));
  float z = 2.060 * exp(-32.0 * pow(log((lambda - 265.8) / 180.4), 2.0));
  return vec3(x, y, z);
}

float fresnelSchlick(float cosTheta, float nFrom, float nTo) {
  float f0 = (nFrom - nTo) / (nFrom + nTo);
  f0 = f0 * f0;
  return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

// XYZ -> linear sRGB (D65). NOTE: no max(0) clamp here; deferred to after
// composite so out-of-gamut channels can cancel against env contributions.
vec3 xyzToSrgb(vec3 xyz) {
  mat3 M = mat3(
       3.2406, -0.9689,  0.0557,
      -1.5372,  1.8758, -0.2040,
      -0.4986,  0.0415,  1.0570);
  return M * xyz;
}

float hash3(vec3 p) {
  p = fract(p * vec3(443.8975, 397.2973, 491.1871));
  p += dot(p, p.yzx + 19.19);
  return fract((p.x + p.y) * p.z);
}

float valueNoise3D(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);
  float n000 = hash3(i + vec3(0, 0, 0));
  float n100 = hash3(i + vec3(1, 0, 0));
  float n010 = hash3(i + vec3(0, 1, 0));
  float n110 = hash3(i + vec3(1, 1, 0));
  float n001 = hash3(i + vec3(0, 0, 1));
  float n101 = hash3(i + vec3(1, 0, 1));
  float n011 = hash3(i + vec3(0, 1, 1));
  float n111 = hash3(i + vec3(1, 1, 1));
  float nx00 = mix(n000, n100, u.x);
  float nx10 = mix(n010, n110, u.x);
  float nx01 = mix(n001, n101, u.x);
  float nx11 = mix(n011, n111, u.x);
  float nxy0 = mix(nx00, nx10, u.y);
  float nxy1 = mix(nx01, nx11, u.y);
  return mix(nxy0, nxy1, u.z);
}

float thicknessAt(vec2 uv, vec3 worldPos, vec3 normal) {
  float h;
  if (params.thicknessMode == 0) {
    vec2 sampleUV = uv;
    if (params.useAnimation != 0) {
      sampleUV += vec2(params.driftSpeed * params.time, 0.0);
    }
    h = texture(heightTex, sampleUV).r;
  } else {
    float gravity = clamp(0.5 + worldPos.y * params.gravityStrength, 0.0, 1.0);
    vec3 noisePos = worldPos * params.noiseScale;
    if (params.useAnimation != 0) {
      noisePos += vec3(0.0, 0.0, params.time * params.driftSpeed);
    }
    float n = valueNoise3D(noisePos);
    h = mix(gravity, n, 0.5);
  }
  return mix(params.thicknessMin, params.thicknessMax, h);
}

// Per-wavelength reflectance R(lambda) for a single thin film bounded by n1/n3.
float thinFilmR(float d, float cosTheta1, float cosTheta2, float deltaPhi,
                float lambda, float R1, float R2) {
  float opdPhase = (4.0 * PI * params.n2 * d * cosTheta2) / lambda;
  // Amplitude form: R = R1 + R2 + 2 * sqrt(R1*R2) * cos(opdPhase + deltaPhi).
  return R1 + R2 + 2.0 * sqrt(max(R1 * R2, 0.0)) * cos(opdPhase + deltaPhi);
}

void main() {
  vec3 N = normalize(inWorldNormal);
  vec3 V = normalize(globals.viewPos.xyz - inWorldPos);

  if (params.showNormal != 0) {
    outColor = vec4(N * 0.5 + 0.5, 1.0);
    return;
  }

  float cosTheta1 = max(dot(N, V), 0.001);
  float sinTheta1Sq = 1.0 - cosTheta1 * cosTheta1;
  float sinTheta2 = (params.n1 / params.n2) * sqrt(max(sinTheta1Sq, 0.0));
  float cosTheta2 = (sinTheta2 >= 1.0) ? 0.0
                                       : sqrt(1.0 - sinTheta2 * sinTheta2);

  float R1 = fresnelSchlick(cosTheta1, params.n1, params.n2);
  float R2 = (sinTheta2 >= 1.0)
                 ? 1.0
                 : fresnelSchlick(cosTheta2, params.n2, params.n3);
  float phi1 = (params.n1 < params.n2) ? PI : 0.0;
  float phi2 = (params.n2 < params.n3) ? PI : 0.0;
  float deltaPhi = phi1 - phi2;

  float d = thicknessAt(inUV, inWorldPos, N);

  if (params.showThicknessHeatmap != 0) {
    float t = clamp((d - params.thicknessMin) /
                        max(params.thicknessMax - params.thicknessMin, 1.0),
                    0.0, 1.0);
    vec3 c = vec3(0.267 + 0.5 * t, 0.005 + 0.9 * t,
                  0.329 + 0.4 * sin(t * PI));
    outColor = vec4(c, 1.0);
    return;
  }
  if (params.showFresnelOnly != 0) {
    outColor = vec4(R1, R1, R1, 1.0);
    return;
  }

  // Spectral integration: accumulate XYZ for both R(lambda) and T(lambda)=1-R.
  vec3 XYZ_R = vec3(0.0);
  vec3 XYZ_T = vec3(0.0);
  float yWeight = 0.0;
  int N_samples = clamp(params.spectralSamples, 4, 64);
  for (int i = 0; i < N_samples; ++i) {
    float lambda = mix(380.0, 780.0,
                       (float(i) + 0.5) / float(N_samples));
    float R = (sinTheta2 >= 1.0)
                  ? 1.0
                  : thinFilmR(d, cosTheta1, cosTheta2, deltaPhi, lambda, R1, R2);
    R = clamp(R, 0.0, 1.0);
    vec3 cmf = wymanCMF(lambda);
    XYZ_R += R * cmf;
    XYZ_T += (1.0 - R) * cmf;
    yWeight += cmf.y;
  }
  float invY = 1.0 / max(yWeight, 1e-6);
  vec3 rgbR = xyzToSrgb(XYZ_R * invY);
  vec3 rgbT = xyzToSrgb(XYZ_T * invY);

  vec3 R_dir = reflect(-V, N);
  vec3 T_dir = refract(-V, N, params.n1 / params.n2);
  if (length(T_dir) < 1e-3) T_dir = -V;

  float maxLod = float(textureQueryLevels(prefilteredCubemap) - 1);
  float lod = clamp(params.roughness, 0.0, 1.0) * maxLod;
  vec3 envR = textureLod(prefilteredCubemap, R_dir, lod).rgb;
  vec3 envT = textureLod(prefilteredCubemap, T_dir, lod).rgb;

  vec3 colorR = rgbR * envR;
  vec3 colorT = rgbT * envT;
  vec3 color;
  if (params.rtMode == 1) {
    color = colorR;
  } else if (params.rtMode == 2) {
    color = colorT;
  } else {
    color = colorR + colorT;
  }
  color = max(color, vec3(0.0)) * params.iblExposure;
  color = pow(color, vec3(1.0 / params.iblGamma));

  outColor = vec4(color, 1.0);
}
```

Notes:
- `R1`/`R2` are reused inside the spectral loop because they are wavelength-independent under Schlick (no dispersion). The amplitude correction is the `sqrt(R1*R2)` term replacing the buggy `R1*R2`.
- `xyzToSrgb` no longer clamps to ≥0 internally; the clamp is applied once after `colorR + colorT`.
- TIR (sinTheta2 ≥ 1) shortcuts to R=1 (full reflection) so `T_dir = -V` fallback matters even less.
- `rtMode` debug branches happen *after* spectral integration so R-only / T-only are exact subsets of the both case.

- [ ] **Step 2: Update host `BubbleParamsUbo` and `Options` in `soap_bubble.hpp`**

In `src/examples/soap_bubble/soap_bubble.hpp`:

(a) In `struct Options`, remove these two fields:

```cpp
  float alphaScale = 1.0f;
  float alphaBase = 0.0f;
```

and replace with:

```cpp
  // R/T Debug
  int32_t rtMode = 0;  // 0=both, 1=R-only, 2=T-only
```

The new `Options` struct should now read (the comments mark the changes; place `rtMode` next to surface/blending or as its own region):

```cpp
struct Options {
  // Thin Film
  float thicknessMin = 200.f;  // nm
  float thicknessMax = 800.f;
  float n1 = 1.0f;
  float n2 = 1.33f;
  float n3 = 1.0f;
  int32_t spectralSamples = 16;
  // Thickness Source
  int32_t thicknessMode = 0;  // 0=Texture, 1=Procedural
  float gravityStrength = 1.0f;
  float noiseScale = 2.0f;
  // Animation
  bool useAnimation = false;
  float driftSpeed = 0.2f;
  // Surface
  float roughness = 0.0f;
  // R/T Debug
  int32_t rtMode = 0;  // 0=both, 1=R-only, 2=T-only
  // IBL / Env
  float iblExposure = 4.5f;
  float iblGamma = 2.2f;
  bool useJitter = true;
  float skyboxLod = 0.0f;
  // Debug
  bool showThicknessHeatmap = false;
  bool showFresnelOnly = false;
  bool showNormal = false;
  // Model
  std::string model = "helmet";
};
```

(b) In `struct BubbleParamsUbo`, replace the existing struct with:

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
  float iblExposure;
  float iblGamma;
  float time;
  int32_t rtMode;
  // -- 16 --
  int32_t showThicknessHeatmap;
  int32_t showFresnelOnly;
  int32_t showNormal;
  int32_t _pad0;
};
```

This matches the shader UBO layout exactly (alphaScale/Base removed, `rtMode` added).

- [ ] **Step 3: Update `updateBubbleParamsUbo` and `setupCommandLineParser` in `soap_bubble.cpp`**

In `src/examples/soap_bubble/soap_bubble.cpp`:

(a) `updateBubbleParamsUbo()` — remove the two lines:

```cpp
  bubbleParamsUbo.alphaScale = opts.alphaScale;
  bubbleParamsUbo.alphaBase = opts.alphaBase;
```

and add right after `bubbleParamsUbo.time = static_cast<float>(timer);`:

```cpp
  bubbleParamsUbo.rtMode = opts.rtMode;
```

(b) `setupCommandLineParser(CLI::App& app)` — remove these lines:

```cpp
  app.add_option("--alphaScale", opts.alphaScale);
  app.add_option("--alphaBase", opts.alphaBase);
```

and add `--rtMode` next to `--roughness`:

```cpp
  app.add_option("--rtMode", opts.rtMode,
                 "0=both, 1=R-only, 2=T-only");
```

- [ ] **Step 4: Update ImGui in `onUpdateUIOverlay`**

In the existing "Surface & Blending" collapsing header in `onUpdateUIOverlay`, replace:

```cpp
  if (ImGui::CollapsingHeader("Surface & Blending",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
    ImGui::SliderFloat("roughness", &opts.roughness, 0.f, 1.f);
    ImGui::SliderFloat("alphaScale", &opts.alphaScale, 0.f, 3.f);
    ImGui::SliderFloat("alphaBase", &opts.alphaBase, 0.f, 1.f);
  }
```

with:

```cpp
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
```

- [ ] **Step 5: Update bubble pipeline state in `preparePipelines`**

In `preparePipelines()`, locate the rasterization and blend create-infos and modify:

```cpp
  vk::PipelineRasterizationStateCreateInfo rsCI(
      {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
      vk::FrontFace::eCounterClockwise, false, 0.f, 0.f, 0.f, 1.f);
```

→ change `eNone` to `eBack`:

```cpp
  vk::PipelineRasterizationStateCreateInfo rsCI(
      {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack,
      vk::FrontFace::eCounterClockwise, false, 0.f, 0.f, 0.f, 1.f);
```

And change the blend attachment from enabled to disabled:

```cpp
  vk::PipelineColorBlendAttachmentState cbAtt(
      true, vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha,
      vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero,
      vk::BlendOp::eAdd,
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
```

→ flip the first arg `true` to `false`:

```cpp
  vk::PipelineColorBlendAttachmentState cbAtt(
      false, vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha,
      vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero,
      vk::BlendOp::eAdd,
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
```

`depthWrite` already became `true` in commit `91e6e4b`, no change here.

- [ ] **Step 6: Format and build**

```
rtk clang-format -i src/examples/soap_bubble/soap_bubble.hpp src/examples/soap_bubble/soap_bubble.cpp
rtk ./mingwBuild.bat Debug
```

Expected: build succeeds with no errors. `bubble.frag.spv` is regenerated.

If the SPIR-V compiler emits warnings about unused `_pad0` it is harmless — leave it.

- [ ] **Step 7: Smoke run + spot-check sanity rows**

```
rtk ./build/soap_bubble.exe --model sphere
```

Quickly walk a few rows of the §5.1 sanity table from the spec. You don't need to formally check every row at this stage — just confirm the new code paths don't crash and produce different colors than before:

1. Default soap (n=1/1.33/1, 200–800 nm): saturated rainbow visible, not flat gray.
2. Slide `thicknessMin = thicknessMax = 0` (animation off): bubble area should look like the environment behind it (R=0, T=1).
3. Toggle `R/T Debug` to `R-only`: rim-bright reflection. Toggle to `T-only`: complementary tint, brighter mid-bubble.
4. Validation layer console: 0 *new* VUID errors. (The pre-existing PBR-era VUID-02697 is unrelated.)

If any of (1)–(3) fails, the most likely cause is a UBO field-order mismatch between host `BubbleParamsUbo` and shader UBO block; fix the order to match exactly before continuing.

- [ ] **Step 8: Commit**

```
rtk git add shaders/soap_bubble/bubble.frag shaders/soap_bubble/bubble.frag.spv src/examples/soap_bubble/soap_bubble.cpp src/examples/soap_bubble/soap_bubble.hpp
rtk git commit -m "feat(soap_bubble): amplitude-form thin-film + R/T compositing

Replace the buggy reflectance-as-amplitude two-beam formula with the
correct sqrt(R1*R2) form, drop the alpha=Fresnel*scale+base fudge, and
composite R(lambda) and T(lambda) inside the shader at alpha=1 with the
prefiltered cubemap sampled in both reflect and refract directions. Add
an rtMode debug toggle (Both / R-only / T-only). Bubble pipeline now
runs blend OFF and cull eBack to match the alpha=1 model.

Spec: docs/superpowers/specs/2026-05-08-soap-bubble-thin-film-rt-design.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

# Stage 2 — Background scaffolding (no draws yet)

Goal: data structs, shaders, descriptor set, and pipeline are constructed without any visible change. Build cleanly so the next stage can simply add `cmd.pushConstants` + `bgModels[i]->draw(...)`.

---

## Task 2: Background structs, shaders, descriptor + pipeline

**Files:**
- Create: `shaders/soap_bubble/bg.vert`
- Create: `shaders/soap_bubble/bg.frag`
- Modify: `src/examples/soap_bubble/soap_bubble.hpp`
- Modify: `src/examples/soap_bubble/soap_bubble.cpp`

- [ ] **Step 1: Create `bg.vert`**

Write `shaders/soap_bubble/bg.vert`:

```glsl
#version 450

layout(set = 0, binding = 0) uniform Globals {
  mat4 view;
  mat4 projection;
  mat4 model;       // unused in bg pass; bubble pass owns this slot
  vec4 viewPos;
} globals;

layout(push_constant) uniform PC {
  mat4 model;
  vec4 baseColor;
} pc;

layout(location = 0) in vec4 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec3 inNormal;
layout(location = 4) in vec4 inTangent;

layout(location = 0) out vec3 outWorldNormal;
layout(location = 1) out vec3 outBaseColor;

void main() {
  vec4 wp = pc.model * vec4(inPos.xyz, 1.0);
  outWorldNormal = mat3(pc.model) * inNormal;
  outBaseColor = pc.baseColor.rgb;
  gl_Position = globals.projection * globals.view * wp;
}
```

The `inColor` and `inTangent` attributes are declared because the shared `vgeu::glTF::Vertex` input layout includes them; they're unused here and will be DCE'd by the compiler.

- [ ] **Step 2: Create `bg.frag`**

Write `shaders/soap_bubble/bg.frag`:

```glsl
#version 450

layout(set = 1, binding = 0) uniform samplerCube irradianceMap;

layout(location = 0) in vec3 inWorldNormal;
layout(location = 1) in vec3 inBaseColor;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846;

void main() {
  vec3 N = normalize(inWorldNormal);
  vec3 irradiance = texture(irradianceMap, N).rgb;
  vec3 diffuse = irradiance * inBaseColor / PI;
  vec3 color = pow(max(diffuse, vec3(0.0)), vec3(1.0 / 2.2));
  outColor = vec4(color, 1.0);
}
```

Fixed exposure 1.0 / gamma 2.2 by design — see spec §3.

- [ ] **Step 3: Add `BgInstance`/`BgPushConstant` and `backgrounds` defaults to `soap_bubble.hpp`**

Add the following to `src/examples/soap_bubble/soap_bubble.hpp` near the top of `namespace vge`, after the existing `Options` struct definition. (Order: keep `Options` containing `std::vector<BgInstance> backgrounds` initialization, so put `BgInstance` *above* `Options`.)

Move the existing `struct Options` definition *down* one slot and add above it:

```cpp
struct BgInstance {
  std::string path;       // assetsPath-relative
  glm::vec3 translate;
  glm::vec3 eulerDeg;     // degrees
  float scale;            // uniform
  glm::vec3 baseColor;    // linear RGB
  bool enabled;
};

struct BgPushConstant {
  glm::mat4 model;
  glm::vec4 baseColor;
};
static_assert(sizeof(BgPushConstant) == 80,
              "BgPushConstant size must be 80 bytes");
```

Then in `struct Options`, append the field (just before `// Model:` line):

```cpp
  // Background scene (hardcoded N-extensible; per-instance enable togglable)
  std::vector<BgInstance> backgrounds = {
      {"/models/apple/food_apple_01_4k.gltf",
       glm::vec3(-1.5f, 0.3f, 1.5f), glm::vec3(0.f, 25.f, 0.f), 1.5f,
       glm::vec3(0.85f, 0.18f, 0.18f), true},
      {"/models/fox/Fox.gltf",
       glm::vec3(1.6f, -0.2f, 1.2f), glm::vec3(0.f, -20.f, 0.f), 0.015f,
       glm::vec3(0.95f, 0.62f, 0.20f), true},
      {"/models/sphere/smooth_sphere.gltf",
       glm::vec3(0.0f, -1.5f, 2.0f), glm::vec3(0.f, 0.f, 0.f), 0.6f,
       glm::vec3(0.30f, 0.55f, 0.85f), true},
      {"/models/dutch_ship_medium_1k/dutch_ship_medium_1k.gltf",
       glm::vec3(0.0f, 1.4f, 2.5f), glm::vec3(0.f, 180.f, 15.f), 0.5f,
       glm::vec3(0.55f, 0.42f, 0.30f), true},
  };
```

Add to `class VgeExample` in the public members section, near the existing `bubbleModel` declaration:

```cpp
  // Background scene
  std::vector<std::shared_ptr<vgeu::glTF::Model>> bgModels;

  // Background pipeline / descriptor handles
  vk::raii::DescriptorSetLayout bgIrradianceSetLayout = nullptr;
  vk::raii::PipelineLayout bgPipelineLayout = nullptr;
  vk::raii::Pipeline bgPipeline = nullptr;
  std::vector<vk::raii::DescriptorSet> bgIrradianceDescSets;
```

- [ ] **Step 4: Load background models in `loadAssets()`**

In `src/examples/soap_bubble/soap_bubble.cpp`, append to the end of `loadAssets()`:

```cpp
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
```

- [ ] **Step 5: Grow descriptor pool and create irradiance descriptor set in `setupDescriptors`**

In `setupDescriptors()`, replace the existing pool creation:

```cpp
  std::vector<vk::DescriptorPoolSize> poolSizes{
      {vk::DescriptorType::eUniformBuffer,
       3u * MAX_CONCURRENT_FRAMES /*globals + params + skybox*/},
      {vk::DescriptorType::eCombinedImageSampler,
       1u + 2u * MAX_CONCURRENT_FRAMES /*height + env + skybox*/}};
  uint32_t maxSets = 3u * MAX_CONCURRENT_FRAMES + 1u + MAX_CONCURRENT_FRAMES;
```

with:

```cpp
  std::vector<vk::DescriptorPoolSize> poolSizes{
      {vk::DescriptorType::eUniformBuffer,
       3u * MAX_CONCURRENT_FRAMES /*globals + params + skybox*/},
      {vk::DescriptorType::eCombinedImageSampler,
       1u + 3u * MAX_CONCURRENT_FRAMES /*height + env + skybox + bgIrr*/}};
  uint32_t maxSets =
      4u * MAX_CONCURRENT_FRAMES + 1u + MAX_CONCURRENT_FRAMES;
```

Then, before the existing `skybox = std::make_unique<vgeu::Skybox>(...)` call near the end of `setupDescriptors()`, insert the irradiance set layout and per-frame allocations:

```cpp
  // set=1 for bg pass (irradiance map)
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
                      device,
                      vk::DescriptorSetAllocateInfo(*descriptorPool,
                                                    *bgIrradianceSetLayout))
                      .front()));
    vk::DescriptorImageInfo info(*iblBaker->iblSampler(),
                                 *iblBaker->irradianceMap().getImageView(),
                                 vk::ImageLayout::eShaderReadOnlyOptimal);
    device.updateDescriptorSets(
        vk::WriteDescriptorSet(*bgIrradianceDescSets[i], 0, 0,
                               vk::DescriptorType::eCombinedImageSampler,
                               info),
        nullptr);
  }
```

- [ ] **Step 6: Build the bg pipeline in `preparePipelines`**

At the end of `preparePipelines()` (after `bubblePipeline` is created), append:

```cpp
  // Background pipeline
  std::array<vk::DescriptorSetLayout, 2> bgSetLayouts{*globalsSetLayout,
                                                      *bgIrradianceSetLayout};
  vk::PushConstantRange bgPcRange(
      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
      0, sizeof(BgPushConstant));
  bgPipelineLayout = vk::raii::PipelineLayout(
      device,
      vk::PipelineLayoutCreateInfo({}, bgSetLayouts, bgPcRange));

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
      {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack,
      vk::FrontFace::eCounterClockwise, false, 0.f, 0.f, 0.f, 1.f);

  vk::PipelineDepthStencilStateCreateInfo bgDsCI({}, true /*depthTest*/,
                                                 true /*depthWrite*/,
                                                 vk::CompareOp::eLessOrEqual);

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
```

The bg pipeline reuses `vertexInputSCI`, `iaCI`, `vpCI`, `msCI`, `dynCI` from earlier in the function. (They are local variables already in scope.)

- [ ] **Step 7: Format and build**

```
rtk clang-format -i src/examples/soap_bubble/soap_bubble.hpp src/examples/soap_bubble/soap_bubble.cpp
rtk ./mingwBuild.bat Debug
```

Expected: build succeeds, `bg.vert.spv` and `bg.frag.spv` are produced, the binary still runs identically to Stage 1 (no draws added yet).

- [ ] **Step 8: Smoke run**

```
rtk ./build/soap_bubble.exe --model sphere
```

Expected: starts and renders the same as after Task 1. No new VUID errors. (The bg pipeline was constructed but is never bound this stage.)

- [ ] **Step 9: Commit**

```
rtk git add shaders/soap_bubble/bg.vert shaders/soap_bubble/bg.vert.spv shaders/soap_bubble/bg.frag shaders/soap_bubble/bg.frag.spv src/examples/soap_bubble/soap_bubble.cpp src/examples/soap_bubble/soap_bubble.hpp
rtk git commit -m "feat(soap_bubble): add background scene infrastructure (no draws yet)

Add BgInstance/BgPushConstant types, hardcoded 4-instance default list
(apple, fox, sphere, dutch_ship), per-frame irradiance descriptor sets,
and a bg pipeline that reuses the glTF vertex layout. The pipeline is
constructed but not yet bound; activated in the next commit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

# Stage 3 — Activate background pass + ImGui/CLI + sign-off

---

## Task 3: Background draw integration

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.cpp`

- [ ] **Step 1: Add a model-matrix helper**

In `soap_bubble.cpp`, just above `buildCommandBuffers()`, add a small helper:

```cpp
static glm::mat4 buildModelMatrix(const BgInstance& inst) {
  glm::mat4 m{1.f};
  m = glm::translate(m, inst.translate);
  m = glm::rotate(m, glm::radians(inst.eulerDeg.y), glm::vec3(0, 1, 0));
  m = glm::rotate(m, glm::radians(inst.eulerDeg.x), glm::vec3(1, 0, 0));
  m = glm::rotate(m, glm::radians(inst.eulerDeg.z), glm::vec3(0, 0, 1));
  m = glm::scale(m, glm::vec3(inst.scale));
  return m;
}
```

Y-X-Z order matches the typical "yaw, pitch, roll" intuition for visual placement.

- [ ] **Step 2: Insert the bg pass into `buildCommandBuffers()`**

In `buildCommandBuffers()`, between the `skybox->draw(...)` call and the `// bubble pass` comment, insert:

```cpp
  // background pass
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
```

`vk::raii::CommandBuffer` exposes a templated `pushConstants<T>` overload that takes the value by reference; using it avoids the `sizeof` + `&pc` ceremony.

- [ ] **Step 3: Format and build**

```
rtk clang-format -i src/examples/soap_bubble/soap_bubble.cpp
rtk ./mingwBuild.bat Debug
```

Expected: build succeeds.

- [ ] **Step 4: Run + visual check**

```
rtk ./build/soap_bubble.exe --model sphere
```

Expected:
- All 4 background objects (apple, fox, sphere, dutch_ship) visible with their configured baseColors, lit by IBL irradiance (clear shading variation across normals).
- Bubble in front of background — moving the camera should show consistent depth occlusion (no z-fighting, no holes).
- Validation layer: 0 new VUID errors.
- Toggle `R/T Debug` to T-only — backgrounds remain unchanged, only the bubble's color shifts (sanity).

If any background object appears mis-scaled or off-position, that's expected first-pass — record adjustments visually and bake them into a follow-up tweak commit at the end of Task 5. Don't block this commit on aesthetic tuning.

- [ ] **Step 5: Commit**

```
rtk git add src/examples/soap_bubble/soap_bubble.cpp
rtk git commit -m "feat(soap_bubble): draw background scene before bubble pass

Insert per-instance push-constant draws between skybox and bubble passes,
gated by per-instance enabled flag. Camera occlusion behaves normally;
bubble at alpha=1 still hides backgrounds inside its silhouette.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Background ImGui group + CLI `--bgEnable`

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.cpp`

- [ ] **Step 1: Add the Background ImGui group**

In `onUpdateUIOverlay()`, just before the existing `if (ImGui::CollapsingHeader("Debug"))` block, insert:

```cpp
  if (ImGui::CollapsingHeader("Background", ImGuiTreeNodeFlags_DefaultOpen)) {
    for (size_t i = 0; i < opts.backgrounds.size(); ++i) {
      ImGui::PushID(static_cast<int>(i));
      // Show the leaf path component as the label.
      const auto& path = opts.backgrounds[i].path;
      auto slash = path.find_last_of('/');
      std::string label =
          (slash == std::string::npos) ? path : path.substr(slash + 1);
      ImGui::Checkbox(label.c_str(), &opts.backgrounds[i].enabled);
      ImGui::PopID();
    }
  }
```

`ImGui::PushID(i)` is required so each checkbox has a unique widget ID; without it, ImGui collapses identical labels under one shared state.

- [ ] **Step 2: Add `--bgEnable` to the CLI parser**

In `setupCommandLineParser()`, after the existing `app.add_option("--model", ...)` block (kept at the end), append:

```cpp
  app.add_option_function<std::vector<std::string>>(
         "--bgEnable",
         [this](const std::vector<std::string>& names) {
           // First occurrence resets all to disabled.
           for (auto& bg : opts.backgrounds) bg.enabled = false;
           for (const auto& name : names) {
             for (auto& bg : opts.backgrounds) {
               // Match by leaf filename (stem before extension also OK).
               auto slash = bg.path.find_last_of('/');
               std::string leaf = (slash == std::string::npos)
                                      ? bg.path
                                      : bg.path.substr(slash + 1);
               // strip extension
               auto dot = leaf.find_last_of('.');
               std::string stem = (dot == std::string::npos)
                                       ? leaf
                                       : leaf.substr(0, dot);
               if (leaf == name || stem == name) bg.enabled = true;
             }
           }
         },
         "enable only the named background instances; "
         "names match leaf filename or stem (e.g. apple, Fox, "
         "smooth_sphere, dutch_ship_medium_1k)")
      ->take_all();
```

`take_all()` allows multiple `--bgEnable foo bar baz`.

- [ ] **Step 3: Format and build**

```
rtk clang-format -i src/examples/soap_bubble/soap_bubble.cpp
rtk ./mingwBuild.bat Debug
```

- [ ] **Step 4: Run + verify ImGui & CLI**

```
rtk ./build/soap_bubble.exe --model sphere
```

In the running app:
1. Open the "Background" group → toggle `Fox.gltf` off → fox disappears next frame.
2. Toggle it back on → fox reappears.
3. Quit.

Then run with CLI override:

```
rtk ./build/soap_bubble.exe --model sphere --bgEnable apple smooth_sphere
```

Expected: only apple and the small blue sphere visible (fox + dutch_ship hidden). The Background ImGui group reflects the override (their checkboxes start unchecked).

- [ ] **Step 5: Commit**

```
rtk git add src/examples/soap_bubble/soap_bubble.cpp
rtk git commit -m "feat(soap_bubble): ImGui Background group + --bgEnable CLI

Per-instance checkboxes in a new Background collapsible header
(label = leaf filename), and a CLI override that, when invoked, resets
all instances to disabled and enables only the named ones. Names match
either the leaf filename or its stem.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Sanity sign-off + visual tweak commit (if needed)

**Files:**
- Modify (optional): `src/examples/soap_bubble/soap_bubble.hpp` — only if visual placement adjustments are warranted.

- [ ] **Step 1: Walk the §5.1 sanity table**

Run `rtk ./build/soap_bubble.exe --model sphere` and verify each row of spec §5.1. For each row, record pass/fail in your notes (do NOT add the table to the plan or spec — the spec is the durable artifact, the run is the validation).

| Input | Expected | Pass? |
|---|---|---|
| `thicknessMin = thicknessMax = 0` | bubble area shows pure environment | |
| `n1=n2=n3=1.0` | bubble area shows pure environment | |
| `n1=1, n2=2.4, n3=1` | very saturated, dark silhouette | |
| `thicknessMin = thicknessMax = 550nm` | single hue, R/T complementary on toggle | |
| camera rotation | hue shift with cosθ | |
| `rtMode = R-only` | bright rim, hue dominant | |
| `rtMode = T-only` | brighter mid, complementary hue | |
| `roughness = 0` vs `1` | sharp ↔ blurry on both R and T | |

If any row fails, debug the formula (most likely a typo in deltaPhi sign or a `sqrt` argument), fix in `bubble.frag`, recompile, recommit with a `fix(soap_bubble): ...` message before continuing.

- [ ] **Step 2: §5.3 Background sanity**

- All 4 instances enabled, each visible with its baseColor.
- Toggle each off/on in ImGui — immediate.
- Camera motion shows consistent depth occlusion (bubble in front of any object that's between bubble and camera ↔ object hides bubble silhouette where it's in front).
- Lambert shading varies with orientation on each model (irradiance is functional).

- [ ] **Step 3: §5.4 Validation cleanliness**

Console scan:
- 0 *new* VUID messages.
- Pre-existing VUID-02697 (PBR-era) is still allowed.

If any new VUID appears, fix and recommit. Common offenders:
- Push constant range stage flags don't match the layout's declared stages (here both should be `VERT|FRAG`).
- Descriptor pool sizing too small (the pool needs `1 + 3 * MAX_CONCURRENT_FRAMES` CIS now).

- [ ] **Step 4 (optional): visual placement tweak**

If during §5.3 any background object is visibly mis-scaled or hidden behind the bubble in a way that hurts the demo, edit the `opts.backgrounds` initializer in `soap_bubble.hpp` and re-run. When happy, format + commit:

```
rtk clang-format -i src/examples/soap_bubble/soap_bubble.hpp
rtk git add src/examples/soap_bubble/soap_bubble.hpp
rtk git commit -m "tweak(soap_bubble): adjust background placement for visibility

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

If no tweak needed, skip this step.

- [ ] **Step 5: §5.2 visual regression note**

Compare with `git stash` + checkout of the pre-Stage-1 commit (`91e6e4b`):
- Old: grayish bubble with Fresnel rim halo only.
- New: saturated rainbow with complementary T tint inside, no halo, energy-balanced.

Take a mental note (or screenshot if useful — but don't commit screenshots into the repo). If the contrast is *not* obviously different, the most likely cause is the spectral integration loop never running (check `spectralSamples` value and shader UBO layout match).

- [ ] **Step 6: Final summary commit (only if outstanding adjustments)**

If Steps 1–5 turned up no further code changes, no commit. Otherwise commit fixes individually with `fix(soap_bubble): ...` messages.

---

## End-of-plan check

When all checkboxes above are ticked:

```
rtk git log --oneline -8
rtk git status
```

Expected log (top-down):
- `tweak(soap_bubble): adjust background placement…` (optional — only if Step 4 ran)
- `feat(soap_bubble): ImGui Background group + --bgEnable CLI`
- `feat(soap_bubble): draw background scene before bubble pass`
- `feat(soap_bubble): add background scene infrastructure (no draws yet)`
- `feat(soap_bubble): amplitude-form thin-film + R/T compositing`
- `docs(soap_bubble): add thin-film R/T compositing + background scene spec`
- (preceding commits…)

`git status` should be clean.
