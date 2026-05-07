# Soap Bubble — Screen-Space Refraction (Optional T Source)

**Date:** 2026-05-08
**Branch:** `claude-test`
**Builds on:** `docs/superpowers/specs/2026-05-08-soap-bubble-thin-film-rt-design.md`
  (background scene infrastructure was placed there as scaffolding for this spec)

---

## Motivation

The current `soap_bubble` example samples the prefiltered cubemap in both the
reflect and refract directions to produce R(λ) and T(λ) compositing. Because
the T direction always hits the cubemap (never the actual scene), the
background instances (apple, fox, sphere, dutch_ship) are not visible through
the bubble. The bubble looks like it sits in front of a scene rather than
*through* it.

Screen-space refraction (SSR) replaces the T-direction cubemap sample with a
sample of the rendered scene color, projected from the world-space refracted
ray onto screen UVs. Toggling SSR on shows the background instances refracted
through the bubble; toggling off recovers the current cubemap-only behavior.
Both modes coexist so users can A/B compare in real time.

---

## Decision Summary

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Affected term | T only; R stays cubemap | bubble's R direction usually faces the environment (away from scene); cubemap is the right source. SSR's payoff is the T direction |
| 2 | Toggle UX | ImGui checkbox + `refractDepth` slider, instant per-frame switch | Both render-pass setups always alive, no waitIdle on toggle. A/B compare is the core value |
| 3 | Scene-color filling | Re-render skybox + bg twice (offscreen + swapchain) | bg/skybox cost is trivial; no new fullscreen-blit pipeline needed |
| 4 | Refraction sampling | Empirical depth offset along T_dir, projected to NDC | Thin-film bubble's refraction offset is small; depth-aware ray march is overkill for the demo |
| 5 | Out-of-frame UV fallback | Sample `prefilteredCubemap` at T_dir | Semantic match (T = environment behind bubble); cubemap descriptor already bound; smoother than clamp-to-edge stretch |
| 6 | Pipeline strategy | Single bubble pipeline + descriptor with sceneColor + shader-branch on `useSSR` UBO flag | Less code than two pipelines; toggle is descriptor-free; per-pixel branch cost negligible |
| 7 | Default state | SSR ON, background 4 instances all enabled | Demo's value is the SSR effect; defaults need to show it on first run |

---

## §1. Architecture & Frame Flow

Two paths in `buildCommandBuffers`, switched per-frame on `opts.useSSR`:

**SSR ON path:**
```
1. Offscreen pass (own render pass + framebuffer)
   skybox.draw    → offscreenColor
   bg loop        → offscreenColor
   end pass       (image returns to ShaderReadOnlyOptimal via render-pass dependency)
2. Swapchain pass
   skybox.draw    → swapchain
   bg loop        → swapchain  (re-rendered; cheap)
   bubble pipeline binds sceneColor (offscreenColor view) at set=4
   bubble.draw    → swapchain  (samples sceneColor for T direction in shader)
3. UI overlay
```

**SSR OFF path:**
```
1. Swapchain pass
   skybox.draw    → swapchain
   bg loop        → swapchain
   bubble.draw    → swapchain  (shader takes T from cubemap)
2. UI overlay
```

The offscreen pass is *skipped entirely* when `useSSR=false`. The bubble
shader's per-pixel branch on `params.useSSR` decides between scene-color
and cubemap sampling for T. The sceneColor descriptor is always bound to
a valid offscreen image view (initialized once at startup), so a stale
SSR-OFF frame still has a coherent image layout for descriptor validity
even though the shader never samples it.

### Files

```
shaders/soap_bubble/
  bubble.frag         — add set=4 sceneColor sampler;
                        useSSR / refractDepth UBO fields;
                        SSR branch in T compositing

src/examples/soap_bubble/
  soap_bubble.hpp     — Options { useSSR, refractDepth };
                        BubbleParamsUbo: same fields, 16-byte aligned;
                        offscreen resource handles;
                        bg defaults flipped back to enabled=true
  soap_bubble.cpp     — prepareOffscreen() new function;
                        setupDescriptors: sceneColor set/layout, pool grown;
                        preparePipelines: bubble layout has 5 sets;
                        buildCommandBuffers: SSR-on/off branching;
                        windowResize: rebuild offscreen + rebind sceneColor;
                        ImGui Refraction group;
                        CLI --useSSR, --refractDepth
```

No new shader files; no `vgeu_*` modifications.

### Bubble pass descriptor sets

| set | binding | resource | note |
|---|---|---|---|
| 0 | 0 | Globals UBO | unchanged |
| 1 | 0 | BubbleParams UBO | + `useSSR`, `refractDepth` |
| 2 | 0 | heightTex | unchanged |
| 3 | 0 | prefilteredCubemap | unchanged (used for R; also T fallback when SSR off / off-frame) |
| 4 | 0 | sceneColor (sampler2D) | NEW — per-frame, points at offscreen image view |

---

## §2. Offscreen Infrastructure

### Color image (per frame)

`MAX_CONCURRENT_FRAMES` independent images so the bubble pass of frame N+1
can sample frame N+1's own offscreen contents without racing the previous
frame's sample. Cost: 2 × swapchain_resolution × 4 bytes (≈16 MB at 1920×1080,
double-buffered). Trivial on any GPU running this demo.

| property | value |
|---|---|
| extent | swapchain extent |
| format | swapchain color format (matches for visual consistency) |
| usage | `eColorAttachment | eSampled` |
| sample count | 1 |
| init layout | `eUndefined` → `eShaderReadOnlyOptimal` once at startup |

### Depth image (per frame, transient)

| property | value |
|---|---|
| extent | swapchain extent |
| format | same depth format as swapchain pass |
| usage | `eDepthStencilAttachment` (no `eSampled` — never read by SSR) |
| init layout | `eUndefined`, transitioned by render pass |

### Sampler (`sceneColorSampler`)

Distinct from `iblSampler` — the latter has trilinear mip filtering for
prefiltered cubemap LOD. SSR sceneColor only has 1 mip level.

| property | value |
|---|---|
| min/mag filter | `eLinear` |
| mipmap mode | `eNearest` |
| address U/V | `eClampToEdge` (defensive; shader branch handles out-of-frame primarily) |
| anisotropy | off |

### Offscreen render pass

Two attachments, one subpass.

```
attachment 0 (color)
  format       = swapchainColorFormat
  loadOp       = eClear           clearValue = (0,0,0,1)
  storeOp      = eStore
  initialLayout= eShaderReadOnlyOptimal
  finalLayout  = eShaderReadOnlyOptimal

attachment 1 (depth)
  format       = depthFormat
  loadOp       = eClear           clearValue = depth=1.0
  storeOp      = eDontCare
  initialLayout= eUndefined
  finalLayout  = eDepthStencilAttachmentOptimal

subpass 0
  colorAttachment   = ref(0, eColorAttachmentOptimal)
  depthAttachment   = ref(1, eDepthStencilAttachmentOptimal)

dependencies
  External -> 0
    srcStage  = eFragmentShader   (previous frame's bubble pass sampled this image)
    dstStage  = eColorAttachmentOutput
    srcAccess = eShaderRead
    dstAccess = eColorAttachmentWrite
  0 -> External
    srcStage  = eColorAttachmentOutput
    dstStage  = eFragmentShader   (this frame's bubble pass will sample)
    srcAccess = eColorAttachmentWrite
    dstAccess = eShaderRead
```

The render pass's implicit layout transition handles the `eShaderReadOnly →
eColorAttachment → eShaderReadOnly` cycle every frame; no
`vkCmdPipelineBarrier` calls in the per-frame command buffer.

### One-shot init barrier

After image creation, before the first render pass:

```cpp
auto cmd = beginOneShotCommand(...);
for (auto& img : offscreenColors) {
  vk::ImageMemoryBarrier b(
      vk::AccessFlags{}, vk::AccessFlagBits::eShaderRead,
      vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *img.image,
      vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
  cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                      vk::PipelineStageFlagBits::eFragmentShader,
                      {}, nullptr, nullptr, b);
}
endOneShotCommand(...);
```

This guarantees the offscreen render pass's `initialLayout = eShaderReadOnly`
assumption holds on the very first frame (and on every SSR-OFF frame, since
those leave the layout untouched).

### Framebuffer

One framebuffer per frame: `{ offscreenColorViews[i], offscreenDepthViews[i] }`.

### Resize

On window resize, `prepareOffscreen()` re-runs (image, view, framebuffer
recreated; sampler reused). All `sceneColorDescSets` are re-written to point
at the new views.

---

## §3. Bubble Shader Changes

UBO add:

```glsl
layout(set = 1, binding = 0) uniform BubbleParams {
  // ... existing fields ...
  int rtMode;
  // ... existing debug fields ...
  int _pad0;
  int useSSR;
  float refractDepth;
  int _pad1;
  int _pad2;
} params;

layout(set = 4, binding = 0) uniform sampler2D sceneColor;
```

Composite block (replaces existing single-source T sample):

```glsl
vec3 envR = textureLod(prefilteredCubemap, R_dir, lod).rgb;

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
// ... existing R/T composite, rtMode branching, exposure/gamma ...
```

`refractDepth` controls how far along T_dir the "behind point" is projected.
At 0, the sample point is the bubble surface itself (which is not yet drawn
into the offscreen image, so the result is "what the scene looks like at this
exact pixel without the bubble"). At higher values, the lens effect grows.

Y-axis convention: the existing pipeline uses negative-Y-up framebuffers
(`kFlipY` load flag affects vertex Y, projection follows). The standard
NDC-to-UV mapping `(ndc.xy / ndc.w) * 0.5 + 0.5` is correct given the
example's existing convention; verify in §5.1.

---

## §4. ImGui / CLI / Validation

### Options additions

```cpp
struct Options {
  // ... existing fields ...
  // Refraction
  bool useSSR = true;          // default ON
  float refractDepth = 0.5f;   // empirical, sphere radius 1 reference
};
```

Background `enabled` for all 4 instances flips back to `true` (was `false`
in the latest commit).

### `BubbleParamsUbo`

Append after `int32_t showNormal; int32_t _pad0;`:

```cpp
  int32_t useSSR;
  float refractDepth;
  int32_t _pad1;
  int32_t _pad2;
```

This keeps 16-byte alignment.

### ImGui — new Refraction group

Inserted between `R/T Debug` and `IBL / Env`:

```cpp
if (ImGui::CollapsingHeader("Refraction", ImGuiTreeNodeFlags_DefaultOpen)) {
  ImGui::Checkbox("Use Screen-Space Refraction (T)", &opts.useSSR);
  if (opts.useSSR) {
    ImGui::SliderFloat("refractDepth", &opts.refractDepth, 0.0f, 2.0f);
  }
}
```

### CLI

Append after `--rtMode`:

```cpp
app.add_option("--useSSR", opts.useSSR);
app.add_option("--refractDepth", opts.refractDepth,
               "empirical depth along T_dir for SSR sample");
```

### Validation plan

**§5.1 SSR visual sanity**

| input | expected |
|---|---|
| `useSSR=on`, all 4 bg enabled, default `refractDepth=0.5` | bubble shows refracted apple/fox/sphere/dutch_ship through it; obviously different from cubemap-T era |
| `useSSR=off` | identical to commit immediately before this spec lands |
| `refractDepth=0` | bubble shows "scene at this pixel without the bubble" since offscreen pass doesn't draw the bubble |
| `refractDepth=2.0` | strong lens distortion; background appears further/inverted depending on view |
| Bubble at frame edge (camera angled to push bubble silhouette near screen edge) | UV-out-of-frame triggers cubemap fallback; visible smooth transition, no black artifacts |
| `R/T Debug = T-only` + SSR on | refracted background only (no R component); validates T isolation under SSR |
| `R/T Debug = R-only` + SSR on | reflection only; SSR has no effect (R uses cubemap regardless) |
| Disable all 4 bg, SSR on | offscreen contains only skybox; bubble still refracts the skybox (subtle lens warp) |

**§5.2 Toggle responsiveness**

ImGui `useSSR` checkbox toggles take effect on the next frame with no
flicker, no validation errors, no descriptor invalidation. Same for
`refractDepth` slider mid-drag.

**§5.3 Window resize**

Drag-resize the window. After resize: SSR continues to work, no
sampling artifacts (e.g., scaled-stale offscreen content), no
validation errors about framebuffer extent mismatches.

**§5.4 useJitter rebake**

Toggling `useJitter` should not affect SSR — it rebakes IBL filtering
(prefilter + irradiance), independent of offscreen images. Verify
`Use Screen-Space Refraction` keeps working across the rebake.

**§5.5 Validation layer cleanliness**

Zero new VUIDs. Common suspects:
- `VUID-VkRenderPassBeginInfo-framebuffer-04627` — framebuffer/renderPass attachment count mismatch
- `VUID-vkCmdDrawIndexed-None-02699` — descriptor sampling invalid image (init barrier missed)
- `VUID-vkCmdPipelineBarrier-...` — incorrect access masks on init transition

**§5.6 Build / format / platform**

- Windows + MinGW + Ninja, validation on debug
- `clang-format -i` on touched `.cpp/.hpp`
- `bubble.frag.spv` regenerated (no other shaders touched)

---

## Out of Scope

- **Depth-aware refraction** (Q2 options B/C). Empirical offset only.
- **R term via SSR.** Reflection samples cubemap permanently; bubble's R
  direction usually faces away from on-screen geometry.
- **Roughness blur on sceneColor.** sceneColor is single-mip / sharp; the
  blurry-reflection feel comes from the prefiltered cubemap on R only.
- **MSAA on offscreen pass.** samples=1.
- **Soft frame-edge blend** between SSR and cubemap fallback. Hard branch only;
  frame-edge seam is acceptable for the demo.
- **A second-tier example** that generalizes SSR to non-bubble materials.
  Stay focused on the soap-bubble use case.

---

## Work Order (writing-plans guidance)

1. `prepareOffscreen()`: per-frame color image / view, depth image / view,
   `sceneColorSampler`, offscreen render pass, framebuffers, init barrier.
2. Pool grow + `sceneColorSetLayout` + per-frame `sceneColorDescSets` in
   `setupDescriptors()`.
3. Bubble pipeline layout: 5 sets (add `sceneColorSetLayout`).
4. `bubble.frag`: UBO `useSSR`/`refractDepth`, `set=4` binding,
   SSR-vs-cubemap branch in T composite block.
5. `Options` / `BubbleParamsUbo` / `updateBubbleParamsUbo` / CLI / ImGui
   Refraction group; flip `backgrounds[*].enabled` defaults to true.
6. `buildCommandBuffers`: branch on `opts.useSSR`. SSR ON → offscreen pass
   first, then swap pass. SSR OFF → swap pass only.
7. `windowResize` override: rebuild offscreen + rebind sceneColor descs.
8. §5.1–§5.5 sanity walk.
