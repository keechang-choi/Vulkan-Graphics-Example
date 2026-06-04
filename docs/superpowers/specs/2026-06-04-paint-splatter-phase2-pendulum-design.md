# Paint Splatter Phase 2 — PBD Pendulum-Driven Spoids + Top View Animation (Design)

**Status:** Approved design, ready for implementation planning.
**Date:** 2026-06-04
**Predecessor:** Phase 1 complete (PBF solver M1–M7 + M8 cohesion/crown, all committed on `claude-test`). See `docs/superpowers/specs/2026-05-31-paint-splatter-pbf-design.md` (§4 SpoidController abstraction, §6 phasing) and `docs/09_paint_splatter.md`.

---

## Goal

Add a `PendulumSpoidController` that drives spoids along the motion of an **n-link PBD pendulum** (double pendulum and beyond), so paint is dripped onto the canvas as a harmonograph/spirograph pattern. Each spoid hangs off a chosen pendulum node with a **rotating perpendicular offset** `(r, angle₀, ω)`. Add a button that **smoothly animates the camera to a top-down view** for framing before saving.

This is the Phase 2 work deferred at the end of Phase 1 ("continuous stream + pendulum + top-down view — a separate spec").

## Architecture (the load-bearing insight)

**Phase 2 is almost entirely host-side C++.** The pendulum is a CPU controller that, each frame, writes only `spoid.pos` — exactly the value the existing render / emit / stream / deposit / PNG-save paths already consume via `spoidController->update(...)` at `paint_splatter.cpp:1166`. Consequences:

- **No shader, no compute pass, no UBO, no GPU-struct change.** Every `static_assert(sizeof(...))` is untouched. The PBF solver, emitter, canvas deposition, and PNG export are all reused as-is.
- The only GPU-facing additions are **visualization**: reuse the existing `markerPipeline` (point sprites) for joint markers, and add one small `eLineList` pipeline for the chain links.
- Risk is low and concentrated in tuning the PBD integrator for stability (double pendulum is chaotic).

`Spoid` is a host-only POD (it is *not* uploaded as a GPU struct; markers are built per-frame from spoid positions into `markerBuffers`, and emission goes through `enqueueDrop`). Adding fields to `Spoid` is therefore free.

**Tech Stack:** C++17, Vulkan-Hpp RAII, GLSL (only for the new line shaders + reuse of `marker.*`), ImGui (`vgeu_ui_overlay`), the existing `paint_splatter` example.

## World convention (locked — from Phase 1)

Screen-up = world −Y. Gravity pulls **+Y** (down on screen). Floor = X-Z plane at **y=0** (domain max-Y). Fluid/spoids live at **y<0**; "higher up" = more negative y. Domain: `kDomainHeight = 3.0` (ceiling at y=−3), `kDomainHalf = 2.0` (canvas 4×4), `kCanvasWorld = 4.0`. The pendulum hangs from a high pivot (y≈−2.8) toward the floor (+Y); a chain of `totalLength≈1.5` puts the resting tip near y≈−1.3 (matching the Phase-1 spoid default `pos.y = −1.25`), keeping the whole swing envelope safely above the floor.

---

## Data Model (all host-side, in `paint_splatter.hpp`)

```cpp
// One point mass in a pendulum chain. node[0] is the fixed pivot (invMass=0).
struct PendulumNode {
  glm::vec3 pos{0.f};
  glm::vec3 prevPos{0.f};   // PBD: position at the start of the substep
  glm::vec3 vel{0.f};
  float invMass = 1.f;      // 1/m; pivot = 0 (immovable)
};

// One suspended pendulum assembly (n-link chain). v1 uses a single instance;
// the controller holds a vector so multiple independent chains generalize later.
struct PendulumChain {
  glm::vec3 pivot{0.f, -2.8f, 0.f};   // fixed suspension point (near ceiling)
  std::vector<PendulumNode> nodes;    // [0]=pivot, [1..numLinks]=bobs
  std::vector<float> linkLength;      // L_i = rest |node_i - node_{i-1}|

  // Configuration (UI-editable):
  int   numLinks    = 2;       // n: 1 = simple, 2 = double pendulum (default)
  float totalLength = 1.5f;    // sum of links (v1: split uniformly into n)
  std::vector<float> bobMass;  // per bob (node 1..n); built on resetChains,
                               // default uniform 1.0. CONVENTION: mass <= 0 =>
                               // infinite mass => invMass=0 => node is PINNED
                               // (immovable), so an intermediate node can be
                               // fixed to make a constrained chain.
  float airDamping  = 0.1f;    // global velocity damping /s (air resistance)
  float jointDamping= 0.05f;   // damping of along-link relative velocity
                               // (string-end / pivot friction)
  int   substeps    = 8;       // XPBD small-steps (stability)
  int   iters       = 4;       // distance-constraint Gauss-Seidel iters/substep

  // Initial state: chain starts as a straight line displaced from vertical by
  // (initTheta from +Y, initPhi azimuth); tip optionally gets a tangential push.
  float initTheta   = 0.6f;    // rad from the +Y (down) axis
  float initPhi     = 0.f;     // rad azimuth in X-Z
  float initSpeed   = 0.f;     // initial tangential speed at the tip (world u/s)
};
```

`Spoid` (Phase 1 fields kept; Phase 2 fields appended — host-only, no assert impact):

```cpp
  // --- Phase 2: pendulum attachment (ignored unless PendulumSpoidController) ---
  int   nodeIndex   = -1;    // attach node in the chain; -1 = tip (last node).
                             // Generalizes to any link/chain later.
  float offsetR     = 0.f;   // offset distance in the plane perpendicular to the
                             // link direction at the attach node (0 = on node).
  float offsetAngle0= 0.f;   // start angle in that perpendicular plane (rad)
  float offsetOmega = 0.f;   // constant spin rate of the offset (rad/s) -> spiro
  float offsetPhase = 0.f;   // runtime: angle0 + omega*t, carried across frames
  float paintMass   = 1.f;   // paint reservoir; decreases on emit, 0 => stop
```

## Controller

```cpp
class PendulumSpoidController : public SpoidController {
public:
  std::vector<PendulumChain> chains;   // v1: size 1
  void update(float dt, std::vector<Spoid>& spoids,
              const InputState& in, std::vector<int>& emitDrops) override;
  void resetChains();                  // (re)build nodes from config + init state;
                                       //   invMass_i = bobMass_i>0 ? 1/bobMass_i : 0
                                       //   (mass<=0 => pinned); pivot invMass = 0
};
```

`update()` per frame:
1. **PBD-step each chain** (algorithm below), advancing the bobs under gravity + constraints + damping.
2. For each spoid: `offsetPhase += offsetOmega * dt`.
3. Set `spoid.pos = emissionPoint(chain, nodeIndex, offsetR, offsetPhase)` (geometry below). This is the only output the rest of the pipeline needs.

Emission gating and the actual emit stay in `render()`'s stream loop (`paint_splatter.cpp:1481`): the controller positions, `render()` emits and drains the reservoir — concerns stay separated. Pendulum mode pairs with **stream mode** (already implemented), whose `[prevPos → pos]` sweep turns the swinging motion into connected strokes. `emitDrops` (Space burst) is left available but unused by the pendulum.

Construction: `prepare()` swaps the controller based on a mode (keyboard vs pendulum). Default at launch stays **keyboard** (Phase 1 behavior); switching to pendulum is a UI choice that calls `resetChains()`.

---

## PBD Pendulum Step (pure PBD, n-link chain)

Each frame is divided into `substeps` (substep dt `sdt = frameDt / substeps`). Per substep, for every chain:

1. **Predict** (all nodes except the fixed pivot):
   `vel += vec3(0, gravity, 0) * sdt`  (gravity is +Y), then `prevPos = pos; pos += vel * sdt`.
2. **Solve distance constraints** — `iters` Gauss-Seidel sweeps. For each link `i` (between node `i−1` and node `i`) with rest length `L_i`:
   - `d = pos_i − pos_{i-1}; len = |d|; C = len − L_i`
   - distribute the correction by inverse mass: `w = invMass_{i-1} + invMass_i`; if `w==0` skip; `corr = (C / (w·len)) · d`
   - `pos_{i-1} += invMass_{i-1} · corr; pos_i −= invMass_i · corr`
   The pivot (`invMass=0`) never moves, so its child absorbs 100% of the correction (mass cancels in the single-link case, as expected; in multi-link cases nodes share by `invMass` ratio). Any node with `bobMass<=0` is likewise `invMass=0` (pinned) — a fully-pinned link (`w==0`) is just skipped.
3. **Velocity update:** `vel = (pos − prevPos) / sdt`.
4. **Damping:**
   - air resistance (global): `vel *= max(0, 1 − airDamping·sdt)`.
   - joint/string friction: damp the component of relative velocity *along* each link by `jointDamping` (removes energy "from the string end").

**Stability safety nets** (chosen because the double pendulum is chaotic and there is no analytic fallback): a generous per-substep speed cap (CFL-style) on node velocity, and a soft clamp of node `pos.y` into `[−kDomainHeight+margin, −0.1]` after the solve (pivot exempt). If the swing still diverges, the remedy is more `substeps` / more `iters` / higher damping — exposed as live sliders. The Phase-1 spoid position clamp at `paint_splatter.cpp:1171-1177` is **skipped for pendulum-driven spoids** (the chain physics owns their positions); only a generous x/z safety bound is retained.

---

## Spoid Rotary Offset Geometry `(r, angle₀, ω)`

For an attach node with link direction `d = normalize(node − parent)` (for the tip, `parent` is the previous node; for node 1, the pivot), build an orthonormal basis of the plane perpendicular to the string:
- `e1 = normalize(cross(d, ref))`, `e2 = cross(d, e1)`, where `ref` is a world axis (e.g. +X) swapped to +Z only when nearly parallel to `d` (singularity guard).

Then:
```
phase     = offsetAngle0 + offsetOmega * t   (carried in offsetPhase)
spoid.pos = node + offsetR * (cos(phase)*e1 + sin(phase)*e2)
```

`r=0` pins the spoid to the node. A nonzero `r` with `ω≠0` makes the emission point orbit the swinging bob → the swing (bob) × rotation (offset) compose into rosette/spirograph paint trails. Each spoid has independent `(r, angle₀, ω)`, so several spoids on the same tip draw different superimposed figures.

---

## Paint Reservoir (`paintMass`) — simple first

In `render()`'s stream loop, gate emission per spoid: `if (paintMass <= 0) continue;` After emitting `n` particles this frame: `paintMass -= kDrain * n` (start `kDrain ≈ 1e-4`, so a full `paintMass=1` reservoir lasts ~10k emitted particles; tune live). Refill via a UI button / restart. This is the deliberately simple v1 model.

**Refinement path (deferred):** drain proportional to emitted *volume × velocity* (true outflow), and couple `paintMass` into the chain (lighter bob as paint drains → damping/period shift). The data model already separates the chain's `bobMass` (dynamics) from the spoid's `paintMass` (reservoir), so this is additive.

---

## Chain Visualization (in scope for v1)

- **Joints (nodes) → sprites:** reuse the existing `markerPipeline` (point-list, `marker.vert/frag`, set up at `paint_splatter.cpp:748-771`, drawn at `1877-1897`). Build a per-frame marker buffer with one point per chain node (including the pivot), sized like the existing spoid markers.
- **Links → lines:** add one `eLineList` graphics pipeline (mirroring the marker pipeline's state and the shared set=0 layout) plus a small per-frame line vertex buffer holding the node-pair endpoints of each link. New shaders `chain_line.vert/frag` (trivial: transform by `projection*view`, flat color).

Both are drawn after the canvas and particles in `buildCommandBuffers()`. An ImGui toggle hides them.

---

## Top-View Camera Animation

Camera control stays as Phase 1 (free orbit via the base `cameraController`). Add an ImGui button **"Top view"** that starts a one-shot animation:

```cpp
struct CameraAnim {
  bool  active = false;
  float t = 0.f, duration = 1.0f;
  glm::vec3 fromEye, fromTarget, toEye, toTarget, up;
};
```

On button press: capture the current eye/target (from `camera.getPosition()` / inverseView), set `toEye = (0, −topHeight, 0)` (start `topHeight ≈ 6`, framing the 4×4 canvas with margin), `toTarget = (0,0,0)`, `up = (0,0,−1)` (non-degenerate, since the view direction becomes +Y). In `render()`, while `active`, advance `t`, ease-in-out interpolate eye/target, and call `camera.setViewTarget(eye, target, up)` directly **after** the base controller runs (so the controller is effectively overridden for the duration). When `t≥duration`, hold the final top view; subsequent orbit input resumes from there.

**Save PNG is unchanged** (`paint_splatter.cpp:2107`, handled at `:1464`). It exports the canvas *texture* directly, independent of the camera, so the animation is purely on-screen framing. (A future convenience: chain "animate → save"; out of v1 scope.)

---

## Scope / YAGNI

**v1 (this spec):**
- Single `PendulumChain`, configurable `numLinks` (default 2 = double pendulum), **per-node bob mass** (mass≤0 ⇒ infinite/pinned), uniform link length, initial `(θ, φ, v)`.
- Per-spoid rotary offset `(r, angle₀, ω)`, attached to the tip node.
- Pure PBD integrator with substeps/iters/air+joint damping and stability safety nets.
- Reservoir `paintMass` with simple linear drain gating emission.
- Chain visualization: link lines + joint sprites (toggle).
- Top-view camera animation button.
- Controller mode switch (keyboard ↔ pendulum) in UI; keyboard remains default.

**Deferred (structure supports, not built now):**
- Per-link length; multiple independent chains; attaching spoids to arbitrary links/chains.
- `paintMass` coupling into dynamics; drain ∝ outflow volume·velocity.
- Emitted particles inheriting the spoid's tangential velocity (sling).
- Analytic harmonograph mode (explicitly dropped by the user — pure PBD only).

---

## Verification Model

No GPU-struct/UBO/shader-physics change → every existing `static_assert` is unaffected (the only new shaders are the trivial `chain_line.*`).

- **VL-CLEAN:** debug build runs with zero validation errors/warnings (host-side changes are nearly automatic; the new line pipeline is the only thing to validate).
- **READBACK:** the existing count/y-range/speed/`rho/rho0`/`ext` readback line stays. Add a tip-position/tip-speed print to numerically confirm the swing is bounded (not diverging).
- **GATE (user):** on screen — a double pendulum swings smoothly (no divergence/NaN/device-loss), the spoid(s) drip a visible harmonograph/spirograph pattern onto the canvas, the chain lines + joint sprites render, and the "Top view" button smoothly animates to an overhead framing. Do not consider Phase 2 done until the user approves.

## Decisions Made (resolved during brainstorming)

1. **Mass:** two distinct masses — chain per-node `bobMass` (dynamics; `invMass = mass>0 ? 1/mass : 0`, so **mass≤0 means infinite mass / pinned node**; mostly cancels in a single-link swing, shares by inverse-mass in multi-link) and spoid `paintMass` (reservoir gating emission, decreases over time). Trajectory is governed mainly by length/angles/damping in v1.
2. **Damping in PBD:** yes — post-solve velocity damping. Air resistance = global; string-end/pivot friction = along-link relative-velocity damping. Both are sliders.
3. **Single assembly, generalized to n-link:** one hanging pendulum that is a chain of `n` links (double pendulum default), structured (`vector<PendulumChain>`, `nodeIndex`) so multiple chains / arbitrary attachment generalize later.
4. **Integrator:** pure PBD chain. No analytic fallback (user chose "A only").
5. **Camera:** orbit control unchanged; add a button-triggered smooth animation to a top-down view for framing. Save PNG stays separate and camera-independent.
6. **Visualization:** link lines + joint sprites, in scope for v1.
7. **Ignored:** moment of inertia, inter-node vibrational coupling beyond the distance constraints (point masses only).
