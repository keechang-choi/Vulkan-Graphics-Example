# Soap Bubble — alphaBase opacity floor

Date: 2026-05-05
Branch: `claude-test`
Follow-up to: `docs/superpowers/specs/2026-04-27-soap-bubble-shader-design.md`

## Motivation

Visual review on the sphere model showed an unexpected seam-like artifact on
the bubble surface. Skybox is clean, debug toggles (`showThicknessHeatmap`,
`showFresnelOnly`) are clean, model swap doesn't change it. Working hypothesis:
because `depthWrite=false` and the surface is alpha-blended, the apparent seam
comes from depth/transparency interaction (e.g. back-side fragments showing
through). Forcing the surface to be opaque should make the seam disappear if
this hypothesis is correct.

The current alpha formula is:

```glsl
// bubble.frag:193-194
float fresnel = fresnelSchlick(cosTheta1, params.n1, params.n2);
float alpha   = clamp(fresnel * params.alphaScale, 0.0, 1.0);
```

`alphaScale` already widens the fresnel-driven alpha but cannot lift the
head-on (low-fresnel) regions. We need a floor.

## Design

Add a single `alphaBase` parameter:

```glsl
float alpha = clamp(params.alphaBase + fresnel * params.alphaScale, 0.0, 1.0);
```

- Range: `[0.0, 1.0]`
- Default: `0.0` (preserves current behavior)
- At `alphaBase = 1.0`: surface becomes fully opaque regardless of fresnel —
  serves both as a diagnostic ("is this seam back-face show-through?") and as
  a physically reasonable knob (thicker films have a baseline opacity).

## Touch points

1. `BubbleParamsUbo` (struct in `soap_bubble.hpp` or `soap_bubble.cpp`) — add
   `float alphaBase`.
2. `bubble.frag` — add `alphaBase` to the matching `params` UBO block, use it
   in line 194.
3. `opts` struct + `updateBubbleParamsUbo()` — wire through.
4. `onUpdateUIOverlay()` — add `ImGui::SliderFloat("alphaBase", ..., 0.0f, 1.0f)`
   next to the existing `alphaScale` slider in the "Surface & Blending" group.
5. Recompile shader (`bubble.frag.spv`).
6. Run clang-format on touched `.cpp/.hpp`.

## Out of scope

- The seam diagnosis itself. This change is the diagnostic *tool*; once landed
  the user will set `alphaBase=1` and confirm the hypothesis. Any follow-up fix
  (separate front/back depth strategy, OIT, or pre-multiplied alpha cleanup)
  is a different change.

## Test plan

Build, run `./build/soap_bubble.exe`. Verify:
1. `alphaBase=0` looks identical to current build.
2. Sliding `alphaBase` to 1.0 makes the bubble fully opaque (no see-through).
3. Mid values (~0.3) raise head-on regions while still showing rim brightening
   from `alphaScale * fresnel`.
