# Paint Splatter M8 — Cohesive Droplet + Crown Splash Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a released paint droplet fall as a cohesive blob and spread in a crown-like splash on impact, by re-matching `rho0` to the actual droplet packing, restoring a bounded cohesion pull, and stabilizing via substeps instead of clamps.

**Architecture:** Pure parameter-scale + small shader/host edits to the existing GPU PBF solver — no new compute pass or buffer. The load-bearing fix re-couples `rho0` (rest density) to the spacing droplets actually emit at, which revives the incompressibility (crown) and lets a **bounded signed density constraint** `C = max(ρ/ρ0−1, −cohesionFloor)` provide in-flight cohesion. Stability comes from XPBD-style substeps + gentle damping, not from the old Δp/velocity clamps (which cap the rebound that makes a crown). Two trailing `ComputeUbo` pads are repurposed (`cohesionFloor`, `dpClampFactor`) so the struct size is unchanged.

**Tech Stack:** C++17, Vulkan-Hpp RAII, GLSL compute (SPIR-V), ImGui, the existing `paint_splatter` example.

---

## Verification Model (read first)

This repo has **no unit-test framework**; the example is verified by running it. This plan uses the same primitives as the Phase-1 plan:

- **VL-CLEAN**: run the debug build (validation layers on) and confirm **zero validation errors/warnings** in stdout.
- **ASSERT**: compile-time `static_assert` on GPU structs (already present; M8 must keep `sizeof(ComputeUbo)==144`).
- **READBACK**: the existing `consumeParticleReadback()` prints numeric sanity (live count, y-range, `rho/rho0` mean/max, speed, `nearFloor%`). M8 adds a horizontal-extent (bbox) print to measure cohesion objectively.
- **GATE (user)**: a STOP. The user visually confirms cohesive fall + a visible crown. Do not mark M8 done until the user approves.

**Single-droplet test protocol** (used by Tasks 1, 3, 4, 6 readbacks): run the binary, in ImGui set **autoEmit OFF**, press **Space once** over an empty domain to release one droplet, and watch the readback line. A cohesive droplet keeps a small horizontal extent (`xExt`,`zExt` ≈ droplet diameter `cbrt(amount)·0.05`) while falling; a dispersing cloud grows it. On impact, the extent widens at the floor (`nearFloor%` rises) = the crown.

## Conventions

- **Build:** `rtk cmd /c mingwBuild.bat` from repo root (shaders compile as part of the build). Run the binary from `build/`: `./build/paint_splatter.exe`.
- **clang-format:** run `clang-format -i` on every edited `.cpp/.hpp` before each commit (user rule). Shaders (`.comp`) are not clang-formatted.
- **rtk:** prefix shell commands with `rtk`.
- **World convention (locked):** screen-up = world −Y; gravity +Y; floor at y=0; fluid/spoids at y<0. Do not change any signs.
- All file paths below are relative to the repo root `C:\Users\rlckd\Desktop\kc\Vulkan-Graphics-Example`.

## ComputeUbo field offsets (for the GLSL UBO blocks)

The std140 `ComputeUbo` (size 144) layout, after M8 repurposes the two trailing pads:

```
0 dt | 4 count | 8 gravity | 12 h
16 cmin(vec4) | 32 cmax(vec4) | 48 gridDim(ivec4)
64 rho0 | 68 epsCFM | 72 scorrK | 76 scorrDq
80 scorrN | 84 xsphC | 88 kPoly6 | 92 kSpiky
96 scorrDenom | 100 velDamp | 104 velClampFactor | 108 solverRelax
112 prevCount(uint) | 116 dryRate | 120 depositStrength | 124 depositHeight
128 depositRadius | 132 drySettle | 136 cohesionFloor | 140 dpClampFactor
```

A shader that needs a field at offset N must declare every field up to N in order (std140). `pbf_lambda` needs through `cohesionFloor` (136); `pbf_delta` needs through `dpClampFactor` (140).

---

## Task 1 — rho0 ↔ emit-spacing match (scale fix)

**Goal:** a fresh droplet sits at rest density (`rho/rho0 ≈ 1`), reviving incompressibility. (Cohesion is added in Task 3; here the droplet may still disperse, but `rho/rho0` must jump from ~0 to ~1 and the sim must stay VL-CLEAN.)

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.hpp:436`
- Modify: `src/examples/paint_splatter/paint_splatter.cpp:1076-1080` (`enqueueDrop`)

- [ ] **Step 1: Raise the rest spacing so `rho0` matches a dense droplet.** In `paint_splatter.hpp`, change line 436:

```cpp
  static constexpr float kParticleSpacing = 0.05f;  // 0.5h: rest spacing == emit
                                                    // spacing so droplets sit at
                                                    // rest density (rho0~8078).
```

(Was `0.005f`. `rho0` is recomputed from this lattice in `prepare()` at line 854–870; no other change is needed there.)

- [ ] **Step 2: Emit droplets at the rest spacing (amount drives size).** In `paint_splatter.cpp`, replace the spacing computation in `enqueueDrop` (lines 1069–1080) with:

```cpp
  // Spawn on a JITTERED LATTICE at the REST spacing so a fresh droplet is at
  // rest density (rho ~= rho0): only then are the incompressibility + bounded
  // cohesion constraints active (M8). `count` particles fill a cube of side
  // ceil(cbrt(count)) -> droplet diameter ~= side * restSpacing, so `amount`
  // (count) drives droplet size. `holeRadius` no longer sets droplet size (a
  // cohesive droplet must be at rest density, which fixes spacing); it is kept
  // as the spoid's emission-disk radius for the future Phase-2 stream mode.
  const int side = std::max(
      1, static_cast<int>(std::ceil(std::cbrt(static_cast<float>(count)))));
  const float spacing = kParticleSpacing;
```

(`holeRadius` stays a function parameter — do not change the signature — it is just no longer used for spacing here. The `wantSpacing`/`restSpacing`/`max` lines are removed.)

- [ ] **Step 3: Build.** Run: `rtk cmd /c mingwBuild.bat`
Expected: `paint_splatter` builds; stdout prints `rho0 (rest density) = 8078...` (≈8078, NOT ~8e6).

- [ ] **Step 4: Run the single-droplet test.** Run: `./build/paint_splatter.exe`, set autoEmit OFF, press Space once.
Expected: **VL-CLEAN**; the readback `rho/rho0 mean` for the live droplet rises to **~0.6–1.0** (was ~0.001). Speed stays bounded (no NaN/explosion; a brief impact spike is fine). The droplet may still spread horizontally while falling — that is expected (no cohesion yet) and is fixed in Task 3.

- [ ] **Step 5: Commit.**

```bash
clang-format -i src/examples/paint_splatter/paint_splatter.cpp src/examples/paint_splatter/paint_splatter.hpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): match rho0 to emit spacing (M8 scale fix)"
```

---

## Task 2 — Stabilization defaults (substeps over clamps)

**Goal:** switch the solver to the XPBD "small steps" recipe (more substeps, fewer iters) and gentle damping, so the revived constraint stays stable without the heavy `velDamp` that kills the crown. Still compression-only here, so the droplet won't be cohesive yet — the gate is **stability**.

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.hpp:443,455,456`

- [ ] **Step 1: Change the stabilization defaults.** In `paint_splatter.hpp`:

Line 443 — gentle global damping (was `8.0f`; reference uses ~0.74/s):
```cpp
  float velDamp = 0.5f;  // gentle per-substep drag (reference v*=0.999); high
                         // values dissipate impact energy and kill the crown
```

Line 455 — substeps (was `1`):
```cpp
  int substeps = 3;  // XPBD small-steps: stability from many small dt, not from
                     // grinding iters at a big dt (reference uses 5)
```

Line 456 — solver iters (was `4`):
```cpp
  int solverIters = 2;  // fewer iters per substep (reference uses 2)
```

- [ ] **Step 2: Build.** Run: `rtk cmd /c mingwBuild.bat`
Expected: builds clean.

- [ ] **Step 3: Run with autoEmit ON for ~15 s.** Run: `./build/paint_splatter.exe`
Expected: **VL-CLEAN** over a sustained run; readback `speed max` stays bounded (occasional falling/impact spikes only, no runaway), `live` count stays bounded, no NaN/device-loss. The fluid is still loose (no cohesion) — fine.

- [ ] **Step 4: Commit.**

```bash
clang-format -i src/examples/paint_splatter/paint_splatter.cpp src/examples/paint_splatter/paint_splatter.hpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): XPBD substep recipe + gentle velDamp defaults (M8)"
```

---

## Task 3 — Bounded signed constraint (the cohesion knob) + bbox readback

**Goal:** restore in-flight cohesion. Add `cohesionFloor` (reusing `ComputeUbo.pad0`) and change `pbf_lambda`'s constraint to `C = max(ρ/ρ0−1, −cohesionFloor)`, so under-dense (stretched) droplet particles get a *bounded* attractive correction. Add a horizontal-extent print to measure cohesion.

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.hpp:82` (pad0 → cohesionFloor), add host field near line 439
- Modify: `src/examples/paint_splatter/paint_splatter.cpp` (`prepare()` ubo init ~line 889, `updateComputeUbo()` ~line 1327, `consumeParticleReadback()` ~line 1370–1402)
- Modify: `shaders/paint_splatter/pbf_lambda.comp`

- [ ] **Step 1: Rename the `ComputeUbo` pad and keep the size assert.** In `paint_splatter.hpp`, change line 82 (inside `struct ComputeUbo`):

```cpp
  float cohesionFloor;  // -- 136 -- bounded signed constraint: C = max(rho/rho0
                        // -1, -cohesionFloor). 0 = compression-only (no
                        // cohesion); ~0.3-0.5 = cohesive droplet (M8)
```

(Was `float pad0;`. Leave `float pad1;` at 140 for now — Task 4 renames it. `static_assert(sizeof(ComputeUbo) == 144, ...)` stays unchanged and must still hold.)

- [ ] **Step 2: Add the host knob.** In `paint_splatter.hpp`, just after `float epsCFM = 100.f;` (line 438), add:

```cpp
  // Bounded signed density constraint (M8): under-dense particles get a bounded
  // attractive pull (cohesion) toward rest density; the floor caps it so a
  // sub-monolayer cannot run away (the M4 "THE big one" explosion). 0 =
  // compression-only. This is the PRIMARY in-flight cohesion knob.
  float cohesionFloor = 0.4f;
```

- [ ] **Step 3: Feed it into the UBO (both init sites).** In `paint_splatter.cpp`:

In `prepare()`'s ubo init block, after `compute.ubo.epsCFM = epsCFM;` (line 886), add:
```cpp
  compute.ubo.cohesionFloor = cohesionFloor;
```

In `updateComputeUbo()`, after `compute.ubo.drySettle = drySettle;` (line 1327), add:
```cpp
  compute.ubo.cohesionFloor = cohesionFloor;
```

- [ ] **Step 4: Change the constraint in `pbf_lambda.comp`.** Extend the UBO block to reach offset 136 and clamp the constraint with the floor. Replace the UBO declaration (lines 12–30, ending at `scorrDenom`) so it continues:

```glsl
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
  float cohesionFloor;  // -- 136 --
}
u;
```

Then replace the constraint line (was `float C = max(rho * invRho0 - 1.0, 0.0);` at line 96):

```glsl
  // Bounded signed constraint (M8): allow a bounded NEGATIVE C so under-dense
  // (stretched) particles are pulled back toward rest density = cohesion, while
  // the -cohesionFloor clamp caps the attractive pull so a perpetually
  // under-dense sub-monolayer cannot run away (compression-only is the
  // cohesionFloor==0 case).
  float C = max(rho * invRho0 - 1.0, -u.cohesionFloor);
```

- [ ] **Step 5: Add a horizontal-extent (cohesion) print to the readback.** In `consumeParticleReadback()` (`paint_splatter.cpp`), add bbox tracking. After `float minY = 1e30f, maxY = -1e30f;` (line 1370) add:
```cpp
  float minX = 1e30f, maxX = -1e30f, minZ = 1e30f, maxZ = -1e30f;
```
Inside the loop, after the `maxY = std::max(maxY, data[i].pos.y);` line (1380), add:
```cpp
    minX = std::min(minX, data[i].pos.x);
    maxX = std::max(maxX, data[i].pos.x);
    minZ = std::min(minZ, data[i].pos.z);
    maxZ = std::max(maxZ, data[i].pos.z);
```
And extend the `std::cout` (before `<< std::endl;` at line 1402) with:
```cpp
            << " | ext x=" << (maxX - minX) << " z=" << (maxZ - minZ)
```

- [ ] **Step 6: Build.** Run: `rtk cmd /c mingwBuild.bat`
Expected: builds clean; `static_assert(sizeof(ComputeUbo)==144)` still compiles.

- [ ] **Step 7: Single-droplet cohesion test.** Run: `./build/paint_splatter.exe`, autoEmit OFF, Space once.
Expected: **VL-CLEAN**; while the droplet falls, `ext x` / `ext z` stay **small and roughly constant** (≈ `cbrt(amount)·0.05`, e.g. ~0.33 for amount=300) instead of growing — that is cohesion. `rho/rho0 mean` ~0.8–1.0. Stable, no explosion. (Compare against Task 1, where `ext` grew during the fall.)

- [ ] **Step 8: Commit.**

```bash
clang-format -i src/examples/paint_splatter/paint_splatter.cpp src/examples/paint_splatter/paint_splatter.hpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): bounded signed constraint for droplet cohesion (M8)"
```

---

## Task 4 — Δp clamp → UBO-driven (free the rebound)

**Goal:** move the hard-coded `maxDp = 0.2h` Δp clamp in `pbf_delta` to a UBO field (`dpClampFactor`, reusing `pad1`) so it can be lowered toward 0 at the gate to let the incompressible rebound (crown) grow. Default keeps the current `0.2` (behavior-preserving) — this task is a safe refactor; Task 6 tunes it down live.

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.hpp:83` (pad1 → dpClampFactor), add host field
- Modify: `src/examples/paint_splatter/paint_splatter.cpp` (`prepare()` ubo init, `updateComputeUbo()`)
- Modify: `shaders/paint_splatter/pbf_delta.comp`

- [ ] **Step 1: Rename the second pad.** In `paint_splatter.hpp`, change line 83 (inside `struct ComputeUbo`):

```cpp
  float dpClampFactor;  // -- 140 -- per-iteration |dp| cap = factor*h; <=0
                        // disables. Low/off lets the incompressible rebound
                        // (crown) grow; default 0.2 = old behavior (M8)
```

(Was `float pad1;`. `static_assert(sizeof(ComputeUbo)==144)` stays.)

- [ ] **Step 2: Add the host knob.** In `paint_splatter.hpp`, just after the `cohesionFloor` field added in Task 3 Step 2, add:

```cpp
  // Per-iteration Δp clamp (pbf_delta), as a multiple of h. <=0 disables it.
  // Default 0.2 preserves the old clamp; lower it toward 0 for a stronger crown
  // once substeps keep the sim stable (M8).
  float dpClampFactor = 0.2f;
```

- [ ] **Step 3: Feed it into the UBO (both init sites).** In `paint_splatter.cpp`:

In `prepare()`'s ubo init, after the `compute.ubo.cohesionFloor = cohesionFloor;` line added in Task 3, add:
```cpp
  compute.ubo.dpClampFactor = dpClampFactor;
```

In `updateComputeUbo()`, after the `compute.ubo.cohesionFloor = cohesionFloor;` line added in Task 3, add:
```cpp
  compute.ubo.dpClampFactor = dpClampFactor;
```

- [ ] **Step 4: Use it in `pbf_delta.comp`.** Extend the UBO block (currently ends at `solverRelax`, offset 108) to reach offset 140, then drive the clamp from it. Replace the tail of the UBO declaration so it reads:

```glsl
  float velDamp;
  float velClampFactor;
  float solverRelax;
  uint prevCount;
  float dryRate;
  float depositStrength;
  float depositHeight;
  float depositRadius;
  float drySettle;
  float cohesionFloor;
  float dpClampFactor;  // -- 140 --
}
u;
```

Then replace the hard-coded clamp (lines 95–99: `float maxDp = 0.2 * u.h; ... if (dl > maxDp) dp *= maxDp / dl;`) with:

```glsl
  // Stability clamp (Macklin 2013 §5), now UBO-driven: bound the per-iteration
  // position correction so stiff constraints cannot overshoot. dpClampFactor<=0
  // disables it (substeps then carry stability), which lets the incompressible
  // impact rebound launch a stronger crown.
  if (u.dpClampFactor > 0.0) {
    float maxDp = u.dpClampFactor * u.h;
    float dl = length(dp);
    if (dl > maxDp) dp *= maxDp / dl;
  }
```

- [ ] **Step 5: Build + run (behavior-preserving check).** Run: `rtk cmd /c mingwBuild.bat && ./build/paint_splatter.exe`
Expected: **VL-CLEAN**; single-droplet readback is essentially identical to Task 3 Step 7 (default `dpClampFactor=0.2` reproduces the old clamp). `static_assert` holds.

- [ ] **Step 6: Commit.**

```bash
clang-format -i src/examples/paint_splatter/paint_splatter.cpp src/examples/paint_splatter/paint_splatter.hpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): UBO-driven Δp clamp (dpClampFactor) (M8)"
```

---

## Task 5 — ImGui exposure + holeRadius label cleanup

**Goal:** surface the two new knobs (`cohesionFloor`, `dpClampFactor`) as live sliders so the crown/cohesion balance is dialed in at the gate, and relabel `holeRadius` so its new meaning is clear.

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.cpp` (`onUpdateUIOverlay`, ~lines 1968, 2002–2007)

- [ ] **Step 1: Add the cohesion + Δp-clamp sliders.** In `onUpdateUIOverlay()`, after the `epsCFM` slider (line 2002), add:

```cpp
      ImGui::DragFloat("cohesion floor", &cohesionFloor, 0.005f, 0.f, 1.f,
                       "%.3f");
      ImGui::DragFloat("dp clamp (xh, 0=off)", &dpClampFactor, 0.005f, 0.f, 0.5f,
                       "%.3f");
```

- [ ] **Step 2: Widen the velDamp slider range** so the new gentle default is easy to fine-tune. Replace line 2006:

```cpp
      ImGui::DragFloat("vel damping (/s)", &velDamp, 0.02f, 0.f, 20.f, "%.2f");
```

- [ ] **Step 3: Relabel holeRadius** to reflect that it no longer drives droplet size. Replace the `hole radius` slider (line 1968):

```cpp
        if (ImGui::DragFloat("hole radius (disk)", &s.holeRadius, 0.002f, 0.02f,
                             0.5f, "%.3f") &&
            editAllSpoids)
```

(Keep the following two lines that propagate to all spoids unchanged.)

- [ ] **Step 4: Build + run.** Run: `rtk cmd /c mingwBuild.bat && ./build/paint_splatter.exe`
Expected: **VL-CLEAN**; the three sliders appear and move; dragging `cohesion floor` to 0 makes the droplet disperse (no cohesion), back to ~0.4 makes it cohesive — confirms the knob is wired.

- [ ] **Step 5: Commit.**

```bash
clang-format -i src/examples/paint_splatter/paint_splatter.cpp src/examples/paint_splatter/paint_splatter.hpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): ImGui sliders for cohesion/dp-clamp + holeRadius relabel (M8)"
```

---

## Task 6 — Crown tuning pass + final self-verify

**Goal:** find a default that gives **cohesive fall + a visible crown on impact**, set those defaults, and self-verify the numeric signature before the user gate.

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.hpp` (defaults found below)

- [ ] **Step 1: Tune live to find the crown.** Run `./build/paint_splatter.exe`, autoEmit OFF, and iterate with single drops (Space), adjusting in this priority order (per spec §3):
  1. raise the spoid `emission vel` (impact momentum) — try ~4–8;
  2. lower `dp clamp` toward 0 (free the rebound) — try 0.05–0.1;
  3. keep `vel damping` low (~0.3–0.7);
  4. balance `cohesion floor` (~0.3–0.5): enough to stay a blob in flight, not so much the impact can't open it;
  5. if the crown needs more particles, raise `amount` (cost-conservative: keep ≤ ~600) and/or `substeps` (≤4).
  Watch the readback: during fall `ext x/z` small (cohesion); at impact `ext x/z` widens + `nearFloor%` rises (crown).

- [ ] **Step 2: Bake the chosen values as defaults.** Edit the corresponding fields in `paint_splatter.hpp` to the values found in Step 1. The expected ballpark (adjust to what actually looked right):
  - `float velDamp = 0.5f;` (Task 2) — keep or nudge within 0.3–0.7.
  - `float dpClampFactor = 0.08f;` (was 0.2 — lower for crown).
  - `float cohesionFloor = 0.4f;` — keep or nudge.
  - `Spoid::emissionVelocity` default (line 110): raise from `2.f` to the tuned value (e.g. `5.f`).
  Record the final numbers in a comment so the gate is reproducible.

- [ ] **Step 3: Build + final self-verify (single drop + sustained).** Run: `rtk cmd /c mingwBuild.bat && ./build/paint_splatter.exe`
Expected, **VL-CLEAN**:
  - single drop (autoEmit OFF): `ext x/z` stays ~droplet-sized during the fall, then widens at the floor; `nearFloor%` climbs after impact; no explosion/NaN.
  - sustained (autoEmit ON, ~20 s): live count bounded, `speed max` bounded, canvas accumulates marks, no device-loss.

- [ ] **Step 4: Commit.**

```bash
clang-format -i src/examples/paint_splatter/paint_splatter.cpp src/examples/paint_splatter/paint_splatter.hpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): tune cohesion/crown defaults (M8)"
```

- [ ] **GATE (user) — M8:** Build and run. **User confirms on screen:** a released droplet (a) falls as a cohesive blob (not a spreading cloud), and (b) spreads in a visible crown/radial splash on impact, with no flicker/validation-error/device-loss. Satellite droplets are a bonus, not required (see spec Known Limitation). Do not consider M8 done until the user approves. After approval, the deferred **Phase-1 closeout** (docs/09 + CLI flags) and then **Phase 2** can begin.

---

## Self-Review Notes (coverage vs spec)

- Spec §1 rho0 ↔ emit-spacing match → Task 1 (kParticleSpacing 0.05 + enqueueDrop restSpacing). ✔
- Spec §1 holeRadius/amount reconciliation (amount=control) → Task 1 Step 2 + Task 5 Step 3 relabel. ✔
- Spec §1.5 bounded signed constraint (cohesion knob) → Task 3 (cohesionFloor + pbf_lambda). ✔
- Spec §2 substeps over iters + gentle velDamp → Task 2. ✔
- Spec §2 clamps off-by-default safety nets → Task 4 (dpClampFactor, UBO-driven) + Task 6 (tuned low); velClampFactor already 0. ✔
- Spec §2 epsCFM auto-realigns (rho0→8078) → Task 1 (no manual rescale; verified in Step 4 readback λ bounded). ✔
- Spec §3 crown dials + cohesion/crown balance → Task 6 live tuning. ✔
- Spec §4 spawn-pop contingency → NOT implemented (contingency only; jittered lattice at rest spacing kept in Task 1). ✔ (out of baseline scope by design)
- Spec "no new pass/buffer; struct size 144" → only pad0/pad1 repurposed (Tasks 3,4); static_assert unchanged. ✔
- Spec verification (VL-CLEAN/ASSERT/READBACK/GATE) → every task ends in build+run+readback; Task 6 ends in user GATE. ✔

**Known v1 simplifications (decided here, not placeholders):** droplet size is driven by `amount` (holeRadius repurposed) — Task 1; `dpClampFactor` default stays 0.2 until Task 6 tunes it down (safe-by-default) — Task 4; crown may stay modest at the cost-conservative budget — spec Known Limitation, Task 6 gate accepts "visible radial spread".
