# Soap Bubble — Thin-film Physics Correction + R/T Compositing + Background Scene

**Date:** 2026-05-08
**Branch:** `claude-test`
**Supersedes direction of:**
- `docs/superpowers/specs/2026-04-27-soap-bubble-shader-design.md` (initial example)
- `docs/superpowers/specs/2026-05-05-soap-bubble-alpha-base-design.md` (alpha floor follow-up — its `alphaBase`/`alphaScale` knobs are removed by this spec)

**References:**
- [Wikipedia — Thin-film interference](https://en.wikipedia.org/wiki/Thin-film_interference)
- Wyman 2013 — analytical fit to CIE 1931 XYZ color matching functions

---

## Motivation

Current `soap_bubble` example renders the bubble with two structural problems
that prevent the result from looking like a soap bubble:

1. **Thin-film formula has a unit error.** `bubble.frag` writes
   `R = r₁² + r₂² + 2·r₁·r₂·cos(φ + Δφ)` where `r₁`, `r₂` are *Schlick reflectances*
   (≈ amplitude²). The standard two-beam formula expects amplitudes there.
   For soap (n₁=n₃=1, n₂=1.33) at normal incidence the cosine modulation term
   is therefore ~25× weaker than physical, so the spectral color signal is
   barely visible and ends up looking like a faint gray tint.
2. **Alpha = Fresnel · scale + base** is a fudge to compensate for (1).
   It produces a vignette-shaped opacity, not the energy-balanced reflection +
   transmission an actual soap film exhibits. Background showing through the
   bubble is just attenuated framebuffer, not color-modulated by `T(λ)`.

Additionally, IBL only contributes reflection. `T(λ) = 1 − R(λ)` is unused, so
the colorful "complementary" tinting that real bubbles show on the transmitted
side is missing.

This spec rewrites the bubble shader to:

- Use the correct **amplitude-form** spectral interference formula.
- Composite both **R(λ) · environment_reflected** and **T(λ) · environment_transmitted**
  inside the shader at `alpha = 1`, with `alphablend OFF`.
- Sample transmission from the same prefiltered cubemap in the `refract` direction.
- Add an `rtMode` debug toggle (Both / R-only / T-only) for verification.

It also adds a **background scene system** as scaffolding for future
screen-space refraction (SSR), and to give the example more visual interest now:

- N background instances rendered with a single shared pipeline
  (push-constant transform + base color, Lambert + IBL irradiance lighting).
- Initial 4 instances hardcoded (apple, fox, sphere, dutch_ship), placed at
  ~+z slightly behind bubble so the silhouette partially occludes them now and
  future SSR will reveal them through the bubble.

**Out of scope:** screen-space refraction itself (separate later spec).

---

## Decision Summary (brainstorming outcome)

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Spec scope | Physics correction + R/T compositing via IBL. **No SSR.** | SSR requires offscreen-pass infrastructure orthogonal to the physics rework |
| 2 | Background lighting | Lambert + IBL irradiance map (`IBLBaker.irradianceMap()` reused) | Visually consistent with bubble's environment, zero new light source |
| 3 | Background data model | Hardcoded `std::vector<BgInstance>` in `Options` initializer; per-instance `enabled` toggle in ImGui | "n-extensible" architecturally; transform/color sliders would clutter UI |
| 4 | R/T composite | `alpha = 1`, `outColor = R·env(R_dir) + T·env(refract_dir)`, alphablend OFF, `rtMode` debug enum | Drops the Fresnel-as-alpha fudge entirely; energy-balanced; debug toggles separate the two terms |
| 5 | Background placement | Hybrid: at +z slightly behind bubble, partially occluded by silhouette | Visible now (rim peek), and primed for SSR's visual payoff later |

---

## §1. Architecture & File Changes

### Frame pipeline order

```
1. Skybox       pass — depth write OFF
2. Background   pass — depth write ON, blend OFF, opaque, N draws (new)
3. Bubble       pass — depth write ON, blend OFF, alpha=1, single draw
4. UI overlay
```

`alpha = 1` means the bubble silhouette currently covers what's behind. Future
SSR will instead sample a captured scene-color image at refracted UVs.

### File changes

```
shaders/soap_bubble/
  bubble.vert        — unchanged
  bubble.frag        — REWRITE (amplitude form, R+T composite, refract, alphablend dropped)
  bg.vert            — NEW (Globals view/proj + push-constant model)
  bg.frag            — NEW (Lambert + irradiance, push-constant baseColor)

src/examples/soap_bubble/
  soap_bubble.hpp    — Options trimmed (alphaScale/alphaBase removed),
                       BgInstance struct, BgPushConstant struct,
                       std::vector<BgInstance> backgrounds added
  soap_bubble.cpp    — preparePipelines: bubble blend OFF, cull eBack,
                       new bg pipeline,
                       buildCommandBuffers: bg pass added,
                       descriptor pool resized (irradiance per-frame),
                       ImGui: Surface&Blending shrunk, R/T Debug + Background groups added
```

`vgeu_ibl` requires no changes — `IBLBaker::irradianceMap()` already exists.

### Bubble pass descriptor sets (unchanged layout)

| set | binding | resource | note |
|---|---|---|---|
| 0 | 0 | Globals UBO | unchanged |
| 1 | 0 | BubbleParams UBO | trimmed (alphaScale/Base out, rtMode in) |
| 2 | 0 | heightTex (sampler2D) | unchanged |
| 3 | 0 | prefilteredCubemap | unchanged (sampled twice: R_dir + T_dir) |

### Background pass descriptor sets

| set | binding | resource |
|---|---|---|
| 0 | 0 | Globals UBO (shared with bubble) |
| 1 | 0 | irradianceMap (samplerCube) |

Push constant range (bg pass): `mat4 model + vec4 baseColor` = 80 bytes,
stages `VERT | FRAG`.

---

## §2. Bubble Shader — Physics & Compositing

### Per-wavelength reflectance/transmittance (amplitude form)

```
sinθ₂ = (n₁/n₂) · sinθ₁                          // Snell
cosθ₂ = √(1 − sin²θ₂)                            // TIR: sinθ₂ ≥ 1 ⇒ R=1, T=0

R₁    = Schlick(cosθ₁, n₁→n₂)                    // outer-boundary reflectance
R₂    = Schlick(cosθ₂, n₂→n₃)                    // inner-boundary reflectance

φ_OPD = 4π · n₂ · d · cosθ₂ / λ
Δφ    = (n₁ < n₂ ? π : 0) − (n₂ < n₃ ? π : 0)    // boundary phase jumps

R(λ)  = R₁ + R₂ + 2·√(R₁·R₂) · cos(φ_OPD + Δφ)   // ★ corrected: √-form
T(λ)  = 1 − R(λ)                                  // non-absorbing thin film
```

Why this differs from the current code: the current `r₁² + r₂² + 2·r₁·r₂·cos(...)`
treats Schlick reflectance as if it were amplitude. The standard two-beam
intensity formula `|r₁ + r₂·e^iφ|² = r₁² + r₂² + 2·r₁·r₂·cos(φ)` uses
*amplitudes*; converted to reflectances `R = r²` it becomes
`R₁ + R₂ + 2·√(R₁·R₂)·cos(φ)`. For soap at normal incidence this makes the
modulation amplitude jump from ~8 × 10⁻⁴ to ~4 × 10⁻², roughly 50× stronger.

Sanity: soap at d=0 has Δφ=π → cos(0+π)=−1 → `R = R₁ + R₂ − 2√(R₁R₂) = (√R₁ − √R₂)²`.
Since R₁ = R₂ for n₁=n₃=1, n₂=1.33, this gives R=0 (the classic black film).

### Spectral integration

Single loop accumulates both R-XYZ and T-XYZ:

```glsl
vec3 XYZ_R = vec3(0);
vec3 XYZ_T = vec3(0);
float yWeight = 0.0;        // CIE Y normalization, source SPD = 1
int N = clamp(params.spectralSamples, 4, 64);
for (int i = 0; i < N; ++i) {
  float lambda = mix(380.0, 780.0, (float(i) + 0.5) / float(N));
  float R = thinFilmR(lambda);
  vec3 cmf = wymanCMF(lambda);
  XYZ_R += R         * cmf;
  XYZ_T += (1.0 - R) * cmf;
  yWeight += cmf.y;
}
float invY = 1.0 / max(yWeight, 1e-6);
vec3 rgbR = xyzToSrgb(XYZ_R * invY);
vec3 rgbT = xyzToSrgb(XYZ_T * invY);
// note: max(0) clamp deferred to after composite (see below)
```

`xyzToSrgb` no longer clamps to ≥0 internally — that previously zeroed
out-of-gamut channels and distorted thin-film color. The clamp moves to after
multiplication with environment radiance, where most negative components have
already been cancelled by positive env contributions.

### Reflection / transmission directions

```glsl
vec3 R_dir = reflect(-V, N);
vec3 T_dir = refract(-V, N, params.n1 / params.n2);
if (length(T_dir) < 1e-3) T_dir = -V;            // TIR fallback

float maxLod = float(textureQueryLevels(prefilteredCubemap) - 1);
float lod    = clamp(params.roughness, 0.0, 1.0) * maxLod;

vec3 envR = textureLod(prefilteredCubemap, R_dir, lod).rgb;
vec3 envT = textureLod(prefilteredCubemap, T_dir, lod).rgb;
```

`refract` with `eta = n₁/n₂` is the "outer-to-inner" direction — physically
the light from behind that arrives at the camera through the bubble.
For a thin film the in/out displacement is sub-pixel so we treat it as a
single direction lookup against the env (no walking-through-thickness).

### Composite + debug

```glsl
vec3 colorR = rgbR * envR;
vec3 colorT = rgbT * envT;

vec3 color;
if (params.rtMode == 1)      color = colorR;          // R-only
else if (params.rtMode == 2) color = colorT;          // T-only
else                         color = colorR + colorT; // both

color = max(color, vec3(0.0)) * params.iblExposure;
color = pow(color, vec3(1.0 / params.iblGamma));
outColor = vec4(color, 1.0);                          // alpha = 1
```

Existing debug paths (`showThicknessHeatmap`, `showFresnelOnly`, `showNormal`)
keep their early-return structure and run before the composite block.

### `BubbleParamsUbo` changes

Remove: `float alphaScale`, `float alphaBase`.
Add: `int32_t rtMode` (0=both, 1=R, 2=T).
Re-pack to keep 16-byte alignment; total stays under one cache line region.

### Bubble pipeline state

- `cbAtt.blendEnable = false` (was true)
- `dsCI.depthWrite = true` (already so since the latest commit)
- `rsCI.cullMode = eBack` (was eNone — alpha=1 means back-face is irrelevant
  and back culling is the conventional default)

---

## §3. Background Scene System

### Data model (`soap_bubble.hpp`)

```cpp
struct BgInstance {
  std::string path;       // assetsPath-relative
  glm::vec3   translate;
  glm::vec3   eulerDeg;   // X-Y-Z degrees (UI-friendly; Quat sliders are noisier)
  float       scale;      // uniform
  glm::vec3   baseColor;  // linear RGB
  bool        enabled;
};

struct BgPushConstant {
  glm::mat4 model;         // 64
  glm::vec4 baseColor;     // 16  (.a unused)
};                         // 80 bytes total
```

### Default instances (initial values)

| name | path | translate | eulerDeg | scale | baseColor (linear) |
|---|---|---|---|---|---|
| apple | `models/apple/food_apple_01_4k.gltf` | (−1.5, 0.3, 1.5) | (0, 25, 0) | 1.5 | (0.85, 0.18, 0.18) |
| fox | `models/fox/Fox.gltf` | (1.6, −0.2, 1.2) | (0, −20, 0) | 0.015 | (0.95, 0.62, 0.20) |
| sphere | `models/sphere/smooth_sphere.gltf` | (0.0, −1.5, 2.0) | (0, 0, 0) | 0.6 | (0.30, 0.55, 0.85) |
| dutch_ship | `models/dutch_ship_medium_1k/dutch_ship_medium_1k.gltf` | (0.0, 1.4, 2.5) | (0, 180, 15) | 0.5 | (0.55, 0.42, 0.30) |

Values are first-pass — adjust visually during §5.3 sanity. Adjustments land
as code commits, not spec edits.

### Shaders

**`bg.vert`** (excerpt):

```glsl
#version 450
layout(set = 0, binding = 0) uniform Globals {
  mat4 view; mat4 projection; mat4 model_unused; vec4 viewPos;
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
  outWorldNormal = mat3(pc.model) * inNormal;   // uniform scale; transpose-inverse skipped
  outBaseColor   = pc.baseColor.rgb;
  gl_Position    = globals.projection * globals.view * wp;
}
```

The `model_unused` field exists in `Globals` because the bubble pass uses it.
Splitting structs would be churn — leave bg pass to ignore it.

**`bg.frag`** (excerpt):

```glsl
#version 450
const float PI = 3.14159265358979323846;

layout(set = 1, binding = 0) uniform samplerCube irradianceMap;

layout(location = 0) in vec3 inWorldNormal;
layout(location = 1) in vec3 inBaseColor;
layout(location = 0) out vec4 outColor;

void main() {
  vec3 N = normalize(inWorldNormal);
  vec3 irradiance = texture(irradianceMap, N).rgb;
  vec3 diffuse = irradiance * inBaseColor / PI;
  // Fixed exposure/gamma to keep bg pass UI-free; revisit if bubble exposure
  // makes the bg look mismatched in practice.
  vec3 color = pow(max(diffuse, vec3(0.0)), vec3(1.0 / 2.2));
  outColor = vec4(color, 1.0);
}
```

If the visual mismatch with bubble's exposure becomes objectionable during
§5.3, expose `iblExposure/iblGamma` to bg pass through a small UBO. Don't
solve it preemptively.

### BG pipeline

- topology: `eTriangleList`
- vertex input: same `vgeu::glTF::Vertex` layout as bubble (Position, UV, Color, Normal, Tangent)
- raster: `cullBack`, `frontCCW`, `fillSolid`
- depth: test ON, write ON, `eLessOrEqual`
- blend: OFF
- push constant range: `{ VERT|FRAG, 0, 80 }`
- descriptor sets: `{ globalsSetLayout, bgIrradianceSetLayout }`

### Draw loop (excerpt)

```cpp
// after skybox.draw(...)
cmd.bindPipeline(eGraphics, *bgPipeline);
cmd.bindDescriptorSets(eGraphics, *bgPipelineLayout, 0,
                       { *globalsDescSets[currentFrameIndex],
                         *bgIrradianceDescSets[currentFrameIndex] }, {});
for (size_t i = 0; i < opts.backgrounds.size(); ++i) {
  if (!opts.backgrounds[i].enabled) continue;
  BgPushConstant pc{ buildModelMatrix(opts.backgrounds[i]),
                     glm::vec4(opts.backgrounds[i].baseColor, 1.0f) };
  cmd.pushConstants(*bgPipelineLayout,
                    vk::ShaderStageFlagBits::eVertex |
                        vk::ShaderStageFlagBits::eFragment,
                    0, sizeof(pc), &pc);
  bgModels[i]->draw(currentFrameIndex, cmd);
}
// then bubble pass
```

### Asset loading

`loadAssets()` loads all 4 background models regardless of `enabled` (so
runtime toggles are immediate, no reload). 4 models at this size is no
memory concern.

---

## §4. ImGui / CLI / Validation

### ImGui changes

- **Surface & Blending** group: keep `roughness`, drop `alphaScale` and `alphaBase`.
- **R/T Debug** group (NEW): radio buttons `[•] Both [ ] R-only [ ] T-only`.
- **Background** group (NEW): one checkbox per instance (`apple`, `fox`,
  `sphere`, `dutch_ship`); no transform/color sliders — by design (B).
- **Debug** group: unchanged (`showThicknessHeatmap`, `showFresnelOnly`,
  `showNormal`, camera readout).

### CLI changes

- Remove: `--alphaScale`, `--alphaBase`.
- Add: `--rtMode {0,1,2}` (default 0=both).
- Add: `--bgEnable <name>` (multi-occurrence). If invoked at least once, all
  instances start disabled and only the named ones are enabled. If never
  invoked, all default to enabled.

### Validation plan

**§5.1 Thin-film amplitude-form sanity**

| input | expected | what it checks |
|---|---|---|
| `thicknessMin = thicknessMax = 0` | bubble area shows pure environment (R=0 ⇒ T=1 across all λ) | Δφ=π ⇒ cos=−1 ⇒ R=0 |
| `n1=n2=n3=1.0` | bubble area = pure environment (R=0 by Fresnel) | refractive-index branches |
| `n1=1, n2=2.4, n3=1` (diamond film) | very saturated reflected colors + dark silhouette (high R) | √(R₁R₂)·cos amplitude is large |
| `thicknessMin = thicknessMax = 550nm` | single hue, R/T are complementary | OPD computation |
| camera rotation | same point shifts hue subtly with cosθ | view-angle dependence |
| `rtMode = R-only` | bright rim, hue dominant from spectral R | R term isolated |
| `rtMode = T-only` | dark rim, complementary hue, brighter at head-on | T term isolated |
| `roughness = 0` vs `1` | sharp R + sharp T vs both blurry | LOD sweep on prefiltered |

**§5.2 Visual-direction regression**

Compare a build immediately before this spec lands vs after, same camera/inputs:
- Default soap params (n=1/1.33/1, 200–800 nm) — old build looked grayish with a
  Fresnel rim only; new build must show saturated rainbow + a complementary
  transmission tint.
- The original spec's sanity table from `2026-04-27` should still pass for the
  rows that didn't depend on the alpha fudge.

**§5.3 Background sanity**

- All 4 instances enabled: each visible with its baseColor, Lambert shading
  visible from irradiance (clear light/dark sides).
- Disabling any instance via ImGui: removed immediately next frame.
- Bubble in front of background: depth occlusion is consistent under camera
  motion (no z-fighting, no holes).
- Lighting consistency: same baseColor on different orientations shows
  natural shading variation (irradiance is being sampled, not a constant).

**§5.4 Validation layer cleanliness**

- Zero new VUID errors from the bg pipeline / push-constant range.
- Descriptor pool sizes account for the added irradiance set per frame.
- Existing PBR-era VUID-02697 is unrelated and remains out of scope.

**§5.5 Build / format / platform**

- Windows + MinGW, validation on debug.
- `clang-format -i` on all touched `.cpp/.hpp` before each commit.
- SPIR-V outputs regenerated for `bubble.frag`, `bg.vert`, `bg.frag`.

---

## Out of Scope

- **Screen-space refraction.** Separate later spec; this spec's background
  placement (hybrid +z behind bubble) is its scaffolding.
- **`alpha < 1` + framebuffer compositing.** Requires SSR infrastructure.
- **Belcour-Barla 2017 closed-form spectral integration.** Direct Wikipedia
  formula is the educational target.
- **Per-instance background transform/color in ImGui.** Decision (B): code
  edit only. CLI also limited to `--bgEnable`.
- **Multiple lights / shadows for background.** Lambert + irradiance only.
- **Per-pass exposure/gamma matching.** Bg pass uses fixed (1.0 / 2.2);
  revisit only if §5.3 shows objectionable mismatch.
- **Linux/macOS verification.**

---

## Work Order (to feed into writing-plans)

1. `bubble.frag` — amplitude-form rewrite + R/T composite + `rtMode` branch +
   alphablend dropped + `alphaScale/Base` removed; validate via §5.1 table.
2. `BubbleParamsUbo` / `Options` / ImGui R/T Debug group / CLI `--rtMode`.
3. `bg.vert`, `bg.frag`; `BgInstance`, `BgPushConstant` structs.
4. Background model loading (4 assets), irradiance descriptor, bg pipeline,
   bg pass added to `buildCommandBuffers`.
5. ImGui Background group, CLI `--bgEnable`.
6. §5.3 background sanity, §5.4 validation cleanliness.
7. §5.2 visual regression vs pre-spec build.
