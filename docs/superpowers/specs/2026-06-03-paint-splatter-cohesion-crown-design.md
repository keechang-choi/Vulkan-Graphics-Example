# Paint Splatter — M8: Cohesive Droplet + Crown Splash (Design)

**Date:** 2026-06-03
**Branch:** `claude-test`
**Status:** Phase 1 tuning milestone (the "crown splash deferred to a tuning
milestone after M6" item from the Phase 1 design spec §6)
**Companions:** `docs/superpowers/specs/2026-05-31-paint-splatter-pbf-design.md`
(Phase 1), `docs/paint_splatter_debug_log.md` (bug history),
`docs/paint_splatter_pbf_reference_analysis.md` (reference-validated theory).

---

## Motivation

A released droplet should **fall as a cohesive blob** and, on hitting the canvas,
**spread in a crown-like splash** (radial sheet + satellite droplets). Today it
does neither: particles disperse from the moment they fall, and impacts produce no
crown. This is the last Phase-1 quality gap before Phase 2 (pendulum), and the
user has chosen the **emergent-physics** approach (A) over an art-directed impact
heuristic (B).

## Root cause (confirmed by the reference analysis)

Cohesion in PBF comes from the **incompressibility constraint** holding the blob
together (surface tension `scorr` only sharpens the surface). That constraint is
*off* in our build:

- `rho0` is computed from a rest lattice at `kParticleSpacing = 0.005`
  (`paint_splatter.cpp:854`), giving `rho0 ≈ 8e6`.
- Droplets actually emit at lattice spacing `max(kParticleSpacing, holeRadius/side)
  ≈ 0.017` (`enqueueDrop`, `paint_splatter.cpp:1078`), so a droplet's density is
  `≈ (0.005/0.017)³ ≈ 0.025 · rho0`.
- `pbf_lambda` then sees `C = max(rho/rho0 − 1, 0) = 0` always → **no density
  pressure**, and in `pbf_delta` the `scorr` term is divided by `rho0 ≈ 8e6` →
  **no surface tension**. The droplet is a cloud of non-interacting points;
  gravity + jitter disperse it irreversibly, and impacts have no incompressible
  rebound to launch a crown.

The Yuki Koyama reference (CPU PBF, Macklin 2013) confirms our kernels and Δp
formula are **term-for-term correct**; the defect is purely parameter scale.

## Goal / Acceptance

A droplet released from a spoid:
1. stays visibly cohesive while falling (a recognizable blob, not a spreading
   cloud), and
2. on impact produces a visible radial spread / crown (satellite droplets are a
   bonus, not a hard gate — see Known Limitation),

with the simulation **stable** (no explosion / NaN / device-loss) and
**VL-CLEAN**, at the cost-conservative budget (substeps ≈ 3–4). All knobs remain
live ImGui sliders so the final look is dialed in at the user GATE.

Non-goals (out of scope for M8): order-independent / pigment-mixing deposit, the
Phase-2 pendulum/stream/top-down work, and any kernel-math change (the reference
proved the math correct).

---

## Approach (A: emergent physics)

One load-bearing change plus a stabilization-recipe correction. Nothing here adds
a new compute pass or buffer — it is parameter scale, one emit-spacing change, and
defaults, all in existing shaders/host code.

### §1. Re-match `rho0` to the actual droplet packing (the fix)

- Set the rest spacing that feeds `rho0` to the PBF-standard **`0.5h = 0.05`**
  (`kParticleSpacing 0.005 → 0.05`). `rho0` recomputes in `prepare()` to a sane
  value (sum of poly6 over a 0.05 lattice within `h=0.1`), so a droplet packed at
  that spacing sits at `rho ≈ rho0`. **`h = 0.1` is unchanged** (it is the grid
  cell size; shrinking it explodes `numCells` — see debug log).
- **Emit droplets at the rest spacing**, so a fresh droplet is at rest density.
  Concretely, `enqueueDrop` must use `spacing = restSpacing` (drop the
  `max(restSpacing, holeRadius/side)` that currently lets sparse — sub-rest —
  drops through). A droplet's diameter then emerges as `cbrt(amount)·restSpacing`.
- **`holeRadius` ↔ `amount` reconciliation** (decide in the plan; recommended
  default): keep `amount` as the particle-count control and let droplet size
  emerge from it; treat `holeRadius` as the *minimum* emission-disk radius (used
  later for stream mode) rather than a size override. Rationale: a cohesive
  droplet must be at rest density, which fixes spacing, so size and count cannot
  both be free. Alternative (also acceptable): make `holeRadius` the droplet
  radius and derive `amount` from the ball volume; the plan picks one and shows
  the derived value in ImGui.

### §2. Stabilization recipe (corrected by the reference)

The revived density constraint reintroduces the old close-encounter pop unless
stabilized the *modern* way — **small steps, not band-aids**:

- **Substeps over iterations:** default toward `substeps ≈ 3–4`, `solverIters ≈ 2`
  (was 1×4). With the droplet near the constraint manifold (now that `rho0`
  matches), corrections shrink with dt and substeps converge instead of
  amplifying. Both stay live sliders; cost-conservative budget allows ≈3–4.
- **Gentle global damping:** `velDamp` default down to **~0.3–1.0 /s** (was 8).
  The form (`vi *= 1 − velDamp·dt`, per substep, `pbf_finalize`) already matches
  the reference's `v *= 0.999`; only the default is wrong (8/s ≈ kills all bulk
  velocity → kills the crown). Settling of *deposited* paint stays drying's job
  (`drySettle`).
- **Clamps become off-by-default safety nets:** the reference uses **neither** a
  Δp clamp nor a CFL velocity clamp. Both cap the incompressible rebound velocity
  that *is* the crown. So: keep `velClampFactor = 0` (off) by default, and make
  the `pbf_delta` `maxDp = 0.2h` clamp **toggleable/slack** (e.g. raise the cap or
  gate it) rather than always-on. They remain available as sliders to tame a
  specific misbehaving high-velocity drop, but are not the primary stabilizer.
- **Re-tune `epsCFM`:** because `Σ‖∇C‖² ∝ 1/rho0²`, dropping `rho0` ~10⁶× rescales
  the λ denominator; the current `epsCFM = 100` will be mis-scaled. After the
  `rho0` change, READBACK a typical `Σ‖∇C‖²` and set `epsCFM` to a small fraction
  (~1e-3…1e-2) of it so λ stays bounded. Expose it as a slider.

### §3. Crown expressiveness dials (tuning, live at the GATE)

With incompressibility back, the crown is controlled by, in priority order:
`emissionVelocity ↑` (impact momentum) × `velDamp ↓` (don't dissipate it) ×
clamps-off (don't cap the rebound) × `solverIters/substeps` (enough to resolve the
incompressible push). `scorrK` is a *secondary* surface-sharpening knob (the
reference itself flags scorr as weak/heuristic), not the cohesion source.

### §4. Spawn-pop contingency (only if needed)

A freshly seeded jittered lattice is a slightly "bad initial state" (the reference
runs a 20-step relaxation pre-pass to fix this offline). Our mitigations, in order:
(a) the **jittered cube lattice at exactly the rest spacing** (already
implemented — keep it; minimizes initial density error), (b) substeps relax the
first frames. **Only if** spawn pops persist on screen, add a brief per-droplet
"soft" window (reduced stiffness / extra damping for the first N frames of life via
a per-particle age in `pos.w`/`predict.w` bookkeeping). This is a contingency,
not baseline scope — do not implement preemptively.

---

## Files touched (no new passes/buffers)

| File | Change |
|---|---|
| `paint_splatter.hpp` | `kParticleSpacing 0.005→0.05`; `velDamp` default; `epsCFM` default; substeps/solverIters defaults; (maybe) Δp-clamp toggle field |
| `paint_splatter.cpp` | `enqueueDrop` spacing = restSpacing; `holeRadius`/`amount` reconciliation + ImGui; `epsCFM` READBACK helper for re-tune; ImGui slider ranges/defaults |
| `shaders/pbf_delta.comp` | Δp clamp → toggleable/slack (UBO-driven), not hard-coded 0.2h |
| `shaders/pbf_finalize.comp` | no form change (velDamp already per-substep); only if the Δp-clamp toggle needs a UBO field |
| `ComputeUbo` (hpp + all PBF shaders) | only if a Δp-clamp toggle/cap field is added (size-assert update + pad) — otherwise unchanged |

Most of M8 is parameter defaults + one emit-spacing line + an `epsCFM` re-tune;
the only potential struct change is an optional Δp-clamp control.

---

## Verification

Per the project model (no unit tests): **VL-CLEAN**, struct **ASSERT**s,
numeric **READBACK** (rho/rho0 mean should now sit near ~1.0 inside a droplet, not
~0.001; peak speed bounded; live count bounded), and the **user GATE** (cohesive
fall + visible crown on screen). Self-verify stability over a sustained auto-emit
run before the gate, then tune the crown dials live with the user.

### Known limitation (carried from Phase 1 §6)
Satellite droplets / a sharp crown may remain *modest* at the cost-conservative
budget (substeps 3–4, ~300 particles/drop). The acceptance gate is **cohesive fall
+ a visible radial spread**; a dramatic satellite-throwing crown may need higher
resolution / more substeps (the "표현력 우선" budget) or, later, the art-directed
impact-momentum heuristic (approach B) — both explicitly deferred.

---

## References
- Macklin, Müller. *Position Based Fluids.* ACM TOG 32(4), 2013.
- yuki-koyama/position-based-fluids (CPU reference) — analyzed in
  `docs/paint_splatter_pbf_reference_analysis.md`; validated kernels + the
  substeps/damping recipe used here.
