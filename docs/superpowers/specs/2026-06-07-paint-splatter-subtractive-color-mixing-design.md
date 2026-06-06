# Paint Splatter — Subtractive Color Mixing (canvas deposit) — Design

Date: 2026-06-07
Status: Approved (awaiting spec review → plan)
Companion: `docs/paint_splatter_debug_log.md` (M6-C-2 "DEFERRED — order-independent
deposit accumulation", line ~506; user note line ~521 "물감 섞이는 것도 alpha
blending이 아니라 색이 혼합되는 방식을 모사").

## Problem

The canvas accumulation (`deposit.comp` → `canvasImage` RGBA8) mixes paint with
**alpha-over** (`paint·a + dst·(1−a)`). Three coupled defects:

1. **Order dependence → flicker.** Non-atomic `imageLoad`/`imageStore`, and the
   per-frame particle REORDER from compaction, make the alpha-over result depend
   on particle processing order → colors "z-fight" at overlaps.
2. **Paleness.** Low `depositStrength` blends toward the WHITE canvas base, so
   marks stay pale until many stamps accumulate (which fast drying prevents).
3. **Not real pigment mixing.** Alpha-over is light/optics-agnostic averaging
   toward white, not how paint actually mixes.

## Goal

Replace alpha-over with an **order-independent subtractive (CMY/density) mixing**
model so that: overlaps are flicker-free, repeated coverage gets richer/darker
(glazing), and mixing behaves like absorbing pigments (e.g. cyan+yellow→green).

Chosen by the user during brainstorming:
- Mixing model: **subtractive (CMY / optical density)** — not weighted-average
  RGB (additive, muddy) and not spectral/Kubelka–Munk (out of scope).
- Overlap behavior: **glazing** — layering darkens (Beer–Lambert additive
  density), not normalized wet-in-wet average.
- Storage/resolve architecture: **Approach A** — 3× `R32_UINT` density images +
  a dedicated resolve compute pass that writes the existing RGBA8 `canvasImage`;
  `canvas.frag` and the PNG save path are unchanged.

## Core model — optical density (the key idea)

Accumulate, per texel, **per-RGB-channel optical density** instead of premultiplied
color. Density contributions ADD, and addition is commutative → the accumulation
is inherently order-independent (solving flicker) AND subtractive (solving the
mixing realism), in one model.

- A paint color `c` (its RGB as reflectance on white paper, ∈ (0,1]) has
  per-channel density `D_paint = -ln(clamp(c, EPS, 1))`. `EPS ≈ 1/255` caps the
  density of a fully-saturated channel so it is finite.
- One stamp deposits weight `w = concentration · depositStrength · falloff`,
  contributing `w · D_paint` to the texel, where `falloff` is a smooth
  center→edge attenuation over the stamp disk (`smoothstep`), softening the mark
  vs today's hard disk.
- Accumulation: `D_total = Σ wᵢ · D_paintᵢ` (sum over all stamps/particles/frames).
  Order-independent; larger D ⇒ darker + more saturated (**glazing**).
- Resolve: `reflectance = exp(-min(D_total, maxDensity))`, clamp [0,1] → rgb.
  Density 0 ⇒ white paper. Output alpha = 1 (opaque canvas).

**Bounding.** Per-texel density is bounded the same way alpha-over is today:
drying (`pos.w → 0`) stops a particle from stamping. `maxDensity` (UI slider) is a
resolve-time safety clamp against pathological build-up.

### Fixed-point storage

Image atomics are core only for `r32ui`, so each RGB channel needs its own image:
`densityImages[3]` = `densityR/G/B`, format `R32_UINT`. The deposit accumulates a
fixed-point integer:

```
imageAtomicAdd(densityChannel, texel, uint(w * D_paint_channel * S))
```

with `S = kDensityScale` (default `1e5`). Precision = `1/S` density units;
overflow headroom = `2^32 / S ≈ 4.3e4` density units, far beyond any
drying-bounded reality. Resolve clamps to `maxDensity` regardless.

### Honest limitation (accepted)

RGB-channel subtractive density is a large improvement over alpha-over (no
white-dilution, order-independent, glazing, **cyan+yellow→green works**), but the
artist expectation **RGB-blue + yellow → green** will read as a dark muddy mix:
RGB primaries carry almost no cyan component, so a true vivid green needs a
spectral / Kubelka–Munk (Mixbox-style) model. That is explicitly out of scope for
this milestone and recorded as a future upgrade path.

## Architecture (Approach A)

```
particles ──▶ deposit.comp (graphics queue)
                  │  imageAtomicAdd per RGB channel  (order-independent)
                  ▼
        densityImages[3]  (R32_UINT, persistent, GENERAL)
                  │  barrier: shaderWrite → shaderRead
                  ▼
            resolve.comp (graphics queue, full-texel dispatch)
                  │  rgb = exp(-min(D/S, maxDensity));  imageStore
                  ▼
            canvasImage (RGBA8, GENERAL)  ── unchanged downstream ──▶
                  │  barrier: shaderWrite → fragment sample
                  ▼
            canvas.frag (bilinear sample, UNCHANGED)
            PNG save path (copyImageToBuffer → stbi, UNCHANGED)
```

### Shader changes

- **`deposit.comp` (modify).** Same dispatch site (graphics queue, after the
  compute→graphics acquire, before render). Replace the rgba8 alpha-over
  load/store with `imageAtomicAdd` into the 3 density images, weighting each
  texel by a `smoothstep` falloff over the disk. Existing guards unchanged
  (floor height, `pos.w > 0` drying gate, uv-in-range, world→uv mapping,
  `depositRadius` → texel radius).
- **`resolve.comp` (new).** 16×16 local size, dispatched `ceil(res/16)²`. Per
  texel: read the 3 density images, `D = min(sum/S, maxDensity)`,
  `rgb = exp(-D)`, `imageStore(canvasImage, texel, vec4(rgb, 1.0))`. No push
  constants → reuses the existing compute pipeline layout. `S` and `maxDensity`
  come from `ComputeUbo`.
- **`canvas.frag`: unchanged.**

### Barriers (all on the graphics queue, one command buffer)

1. deposit (density images, shader write) → resolve (density images, shader
   read): an `ImageMemoryBarrier` / memory barrier shaderWrite→shaderRead on the
   3 density images.
2. resolve (canvasImage, shader write) → render pass fragment sample: reuse/adjust
   the existing `canvasBarrier` so the sample sees the resolved image.

### Host changes (`paint_splatter.cpp` / `.hpp`)

- **`createCanvasImage`**: also create `densityImages[3]` (`R32_UINT`,
  `kCanvasTexRes²`, Storage | TransferDst, GENERAL).
- **Descriptors**: add 3 storage-image bindings (e.g. 11/12/13) to the compute
  descriptor set layout used by deposit + resolve; bump the descriptor pool's
  storage-image count. resolve also binds `canvasImage` (binding 8, already
  present).
- **`ComputeUbo`**: add `densityScale`, `maxDensity` (and, if used,
  `depositFalloff`); update the size `static_assert` (e.g. 144 → 160) + padding.
- **Pipelines / command buffers**: build the resolve pipeline; record resolve
  right after deposit with barrier (1), and barrier (2) before the render pass.
- **restart**: `clearColorImage(densityImages[i], uint 0)` for all 3; keep the
  existing white clear of `canvasImage` (covers the pre-first-resolve frame).
- **ImGui**: add a `maxDensity` slider (safety clamp). `depositStrength`,
  `depositRadius`, `depositHeight` keep their meaning (now weight the density
  contribution).
- **PNG save: unchanged** — resolve keeps `canvasImage` current every frame.

## Out of scope (YAGNI)

- Spectral / Kubelka–Munk (Mixbox) pigment mixing.
- Wet-in-wet normalized-average mode (a density/weight ratio); only glazing.
- Paper absorption / capillary diffusion / bleeding.

These are recorded as candidate future milestones, not part of this work.

## Testing / verification (visual-gate centric, per debug-log convention)

- Build + run **VL-CLEAN** (zero validation / STDERR), no device-loss.
- Single color, repeated stamps on one spot → progressively **darker** (glazing).
- Two overlapping colors → **no flicker** (order-independent) and **cyan+yellow→
  green**.
- White paper (density 0) stays white; **restart** returns to white.
- Saved **PNG matches** the on-screen canvas.
- Final on-screen confirmation is the user's visual gate.

## Files touched (anticipated)

- `shaders/paint_splatter/deposit.comp` (modify)
- `shaders/paint_splatter/resolve.comp` (new)
- `src/examples/paint_splatter/paint_splatter.hpp` (ComputeUbo fields,
  densityImages, resolve pipeline members)
- `src/examples/paint_splatter/paint_splatter.cpp` (image creation, descriptors,
  pool, UBO, resolve pipeline + recording, barriers, restart clear, ImGui)
- `docs/paint_splatter_debug_log.md` (record the milestone + result)
