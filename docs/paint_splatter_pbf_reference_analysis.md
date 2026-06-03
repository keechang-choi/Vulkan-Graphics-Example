# Paint Splatter — PBF Reference Analysis (yuki-koyama/position-based-fluids)

Analysis of the reference CPU PBF implementation at
`C:\Users\rlckd\Desktop\kc\position-based-fluids` (Yuki Koyama, MIT), a faithful
single-threaded C++ implementation of Macklin & Müller 2013. Read against our GPU
`paint_splatter` solver to validate the theory behind the **M8 cohesion / crown
fix** (design: re-match `rho0` to the actual droplet packing so incompressibility
+ surface tension revive — see the M8 design spec and `paint_splatter_debug_log.md`).

The reference is CPU/offline (Alembic export, no rendering, no GPU), so its *value
is purely theoretical*: it shows the canonical parameter relationships and the
stabilization recipe that a correct PBF uses. The files that matter are
`src/main.cpp` (`step()` = one solver iteration + the `main()` driver) and
`src/kernels.cpp`.

---

## Headline conclusion

**Our kernel math is correct; the bug is parameter scale, not formulas.** Every
kernel and the core constraint match the reference exactly. What broke cohesion is
that our parameters (`kParticleSpacing=0.005` → `rho0≈8e6` vs. an actual droplet
packing of `rho≈0.001·rho0`) put the solver in a *dead regime* where both the
density constraint and surface tension evaluate to ~0. This confirms the M8 fix
direction and adds five concrete refinements to the plan (see "Refinements").

---

## Side-by-side: reference vs. our solver

| Aspect | Reference (`main.cpp`/`kernels.cpp`) | Our solver | Verdict |
|---|---|---|---|
| Poly6 | `315/(64π h⁹)(h²−r²)³` | `kPoly6=315/(64π h⁹)`, same form | ✅ identical |
| Spiky grad | `−45/(π h⁶)(h−r)²·r̂` | `−kSpiky(h−r)²(rij/r)`, `kSpiky=45/(π h⁶)` | ✅ identical |
| Constraint | `C = ρ/ρ0 − 1` (signed) | `C = max(ρ/ρ0 − 1, 0)` (compression-only) | ⚠️ ours clamps; OK for free surface (see note) |
| λ | `−C / (Σ\|∇C\|² + εCFM)` | same | ✅ |
| Δp | `(1/ρ0) Σ_j (λi+λj+scorr)∇W` | same (`dp/=rho0`) | ✅ form identical |
| Density uses mass | `ρ = Σ mⱼ W` (mass-weighted) | unit mass (`ρ=Σ W`), `rho0` from rest lattice | ✅ equivalent if rho0 consistent |
| **rest_density** | **1000, with mass tuned so initial packing ≈ rest_density** | computed from rest lattice — **but lattice spacing ≠ emit spacing** | ❌ **THE bug** |
| Substeps / iters | **5 substeps × 2 iters** | 1 substep × 4 iters | ⚠️ refine (Finding 2) |
| Global damping | `v *= 0.999` per substep (~0.74/s) | `velDamp=8/s` (≈ e⁻⁸/s, ~0.0003) | ❌ ours ~3000× too strong (Finding 3) |
| Δp clamp | **none** | `‖Δp‖ ≤ 0.2h` | ⚠️ band-aid; off-by-default (Finding 4) |
| CFL velocity clamp | **none** | `velClampFactor` (currently 0=off) | ⚠️ keep off (Finding 4) |
| εCFM | `1e5` (at ρ0=1000) | `100` (at ρ0=8e6) | ⚠️ rescale after fix (Finding 6) |
| scorr (surface tension) | `corr_k = m·1e-4`, `corr_h=0.30`, n=4, "no ground, may not work" | `scorrK=0.1`, `scorrDqRatio=0.2`, n=4 | ⚠️ weak/heuristic both sides (Finding 7) |
| Collision | naive box clamp (`cwiseMax/Min`) | naive box clamp | ✅ identical approach |
| Init handling | **20-step relaxation pre-pass** at tiny dt | none (jittered lattice only) | ℹ️ spawn-pop relevance (Finding 5) |

---

## Findings that change our plan

### Finding 1 — rest density must equal the *actual* particle packing (confirms THE bug)
The reference tunes `particle.m = 3000/N` so that the initial packing density
equals `rest_density=1000`, then **relaxes the block for 20 steps** to settle any
residual error *before* simulating. Density is meaningful only relative to a
`rho0` that matches how tightly particles are actually packed. Our solver computes
`rho0` from a rest lattice at `kParticleSpacing=0.005`, but droplets emit at
spacing `~0.034`, so `rho/rho0≈0.001` → `C=max(…,0)=0` forever → no pressure, and
`scorr/rho0≈0` → no surface tension. **Fix (unchanged from design): make the rest
spacing that feeds `rho0` equal the spacing droplets actually emit at** (`0.5h=0.05`).
This is the load-bearing change; everything else is tuning around it.

### Finding 2 — XPBD "small steps": prefer MANY substeps over many iterations
Reference: **5 substeps × 2 solver iters**, not our 1×4. Modern PBF/XPBD stability
comes from shrinking dt (substeps), not from grinding more Jacobi iterations at a
big dt. Our debug log said "substeps made it worse" — but that was measured in the
*dead sparse regime* where corrections stay ~one spacing and `v=Δp/dt` grows as dt
shrinks. **Once rho0 matches (Finding 1), the droplet sits near the constraint
manifold, corrections shrink with dt, and substeps behave correctly.**
→ **Plan change:** default toward `substeps≈3–4, solverIters≈2` (was 1/4). Keep
both as live sliders; the cost-conservative budget (user choice "가") tolerates
~3–4 substeps.

### Finding 3 — global damping should be GENTLE, not a bulk-velocity killer
Reference applies `v *= 0.999` *per substep* (~0.74/s effective) — a whisper of
drag. Our `velDamp=8/s` is ~3000× stronger and was added to force the gas-like
dead-regime fluid to settle. With incompressibility revived, that much drag will
**kill the crown** (it dissipates exactly the impact energy that launches the
radial sheet). → **Plan change:** drop `velDamp` default to ~`0.3–1.0/s` and apply
it as a per-substep multiplicative factor like the reference. Settling of
*deposited* paint is drying's job (M6 `drySettle`), not bulk drag's.

### Finding 4 — the Δp clamp and CFL velocity clamp are band-aids; keep them OFF by default
The reference has **neither** a per-iteration Δp clamp nor a CFL velocity clamp.
It stays stable on substeps + rest-density-matching + gentle damping alone. Our
`maxDp=0.2h` clamp and `velClampFactor` were introduced to mask the dead-regime
explosion (now fixed by Finding 1). Both *cap the incompressible rebound velocity*,
which is precisely the mechanism that makes a crown. → **Plan change (revises the
M8 design):** do **not** re-enable `velClamp` as a default. Lead with substeps;
keep `velClamp` and the Δp clamp as *off-by-default, slider-exposed safety nets*
only, to be dialed in if a specific high-velocity drop misbehaves. (This reverses
the design's "velClamp 안전망 재활성" item — substeps replace it.)

### Finding 5 — clean initial state matters (spawn-pop mitigation)
The reference's 20-step relaxation pre-pass exists to "resolve bad initial states"
— a freshly seeded block that isn't quite at rest density will otherwise pop. We
can't relax each droplet for 20 steps in real time, but the same risk applies at
emit. Mitigations, in order of preference: (a) emit on a **jittered cube lattice at
exactly the rest spacing** (already implemented — keep it; it minimizes initial
density error by construction), (b) rely on substeps to relax the first few frames,
(c) *if* spawn pops persist, emit with a brief reduced-stiffness / extra-damping
window for the first N frames of a droplet's life. Treat (c) as a contingency, not
baseline scope.

### Finding 6 — εCFM must be rescaled after rho0 changes
`εCFM` is added to `Σ‖∇C‖²` in λ's denominator; `∇C ∝ ∇W/rho0`, so `Σ‖∇C‖² ∝
1/rho0²`. Changing `rho0` from `8e6` to `~(rest lattice @0.05)` rescales that sum
by ~10⁶, so our `εCFM=100` will be wildly mis-scaled relative to the new gradient
magnitudes (the reference uses `1e5` at `rho0=1000`). → **Plan addition:** after
the rho0 fix, **recompute/re-tune `εCFM`** so it is a small fraction of a typical
`Σ‖∇C‖²` (READBACK the denominator's scale; pick εCFM ≈ 1e-3…1e-2 of it). Expose
it and verify λ is bounded. This was missing from the design.

### Finding 7 — surface tension (scorr) is weak and heuristic on BOTH sides; don't lean on it for cohesion
The reference literally comments that its tensile-pressure coefficient "has no
ground and may not work well" and sets it tiny (`m·1e-4`). Lesson: **the density
(incompressibility) constraint, not scorr, is what holds a droplet together.**
scorr only prevents clustering / sharpens the surface. → **Plan emphasis:** rho0
matching (Finding 1) does the cohesion heavy-lifting; treat `scorrK` as a
secondary surface-sharpening knob, not the primary cohesion source. (Refines the
design's "scorrK up for cohesion" to "scorrK is secondary".)

### Finding 8 — kernels and the Δp formula are verified correct
Poly6, Spiky, Spiky-gradient, λ, and Δp all match the reference term-for-term.
**No shader-math changes are needed** for M8 — the work is parameter scale
(`kParticleSpacing`/`rho0`), the substep/iters/damping recipe, and clamp defaults.
This narrows M8 to tuning + a few host-side constant changes, with **no risk of a
kernel-derivation bug** to chase.

---

## Net effect on the M8 plan

The reference *validates the core fix* (Finding 1, 8) and *sharpens the
stabilization recipe* away from the design's first draft:

1. rho0 ↔ emit-spacing match — **keep** (load-bearing).
2. Stability via **substeps (≈3–4) + low iters (≈2)** — **changed** (was 2–3 substeps / 4 iters).
3. `velDamp` → **~0.3–1.0/s, gentle, per-substep** — **changed** (was just "lowered").
4. Δp clamp + CFL clamp → **off by default, safety-net sliders only** — **reversed** (design wanted velClamp re-enabled).
5. **Re-tune εCFM** to the new gradient scale — **new** (was missing).
6. scorrK = secondary surface knob, not cohesion source — **clarified**.
7. Spawn-pop contingency = brief soft-emit window *if needed* — **new, contingency only**.

These refinements are folded into the M8 design spec before the implementation plan.
