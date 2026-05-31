# Paint Splatter — GPU Position Based Fluids (PBF)

**Date:** 2026-05-31
**Branch:** `claude-test`
**Status:** Phase 1 design (Phase 2 scoped but deferred to a follow-up spec)

---

## Motivation

A new example: a large quad **canvas** that gets painted by **paint droplets**
splashing onto it, like flicked/dripped paint. Droplets are released from
**eyedroppers (spoids)** positioned above the canvas. Each droplet falls in 3D,
hits the canvas, and splashes naturally — including secondary (satellite)
droplets that bounce back up and land elsewhere. Where fluid touches the canvas,
color is deposited permanently into an accumulation texture, which can be saved
to PNG.

The user wants:
- A physically natural droplet/splash simulation, GPU-compute based.
- Per-spoid parametrization: count, hole size, mass, emission velocity, pressure,
  amount of water, color, concentration (viscosity/opacity).
- Individual spoid control — selected via ImGui toggles, moved together via keyboard.
- Manual keyboard control first; later the spoids are driven by a **pendulum** so
  the swinging motion "paints" the canvas.

This builds on the project's existing GPU-compute (`particle`) and
position-based / spatial-hash (`pbd`) infrastructure, and the offscreen-capture
pattern from `soap_bubble`.

---

## Decision Summary

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Splash fidelity | Full 3D, individual particles (fall + collision + bounce + satellite droplets) | The "튀는" splash and bounce-back are the core visual; user chose the most natural option |
| 2 | Fluid solver | **PBF** (Position Based Fluids, Macklin & Müller 2013) — NOT FLIP, NOT SPH | Built-in surface tension (cohesive droplets), no 3D MAC grid / pressure projection, GPU-native, reuses `pbd` spatial-hash pattern |
| 3 | Domain | Full 3D over the volume above the canvas | Needed for radial crown splash + satellite droplets landing elsewhere |
| 4 | Layering | Two layers: (L1) live PBF particles, (L2) permanent canvas accumulation texture | Cleanly separates transient water from the saved painting |
| 5 | Deposition | Contact stamp + drying absorption (particle is consumed) — "방식 B" | Keeps particle count bounded/predictable; paint doesn't pool; maps "concentration" → absorption rate; canvas accumulates over long pendulum runs |
| 6 | Blend mode | alpha-over (wet-on-wet), colors mix | Simple, intuitive; subtractive pigment mixing deferred to Phase 2 |
| 7 | Emission mode | Architecture supports both; **Phase 1 = A (discrete droplet burst)**, Phase 2 = B (continuous stream) | Droplet unit matches "물방울 떨어뜨리기" and pendulum phase-per-drop; same code path (per-frame spawn count) extends to stream |
| 8 | Spoid control | ImGui checkbox multi-select + keyboard moves the selected set; ImGui edits params | Matches user's requested UX |
| 9 | Spoid → pendulum | Abstract `SpoidController` interface; Phase 1 `KeyboardSpoidController`, Phase 2 `PendulumSpoidController` | Swap controller without touching solver/emitter; `mass`/`emissionVelocity`/`pressure` left independent now, coupled by pendulum later |
| 10 | Camera | 3D free orbit camera (`vgeu_camera`), canvas quad textured with the accumulation texture | Shows splash in 3D + the painting at once; top-down preview deferred to Phase 2 |
| 11 | Save | ImGui "Save" button → GPU→CPU readback → PNG via `stb_image_write` | Reuses soap_bubble capture pattern; button (not key) per user request |

---

## §1. Architecture & Components

Location: `src/examples/paint_splatter/` (`.cpp`/`.hpp`) and
`shaders/paint_splatter/`, following the `particle` / `soap_bubble` example
structure and registered in the examples `CMakeLists.txt`.

Single-responsibility units:

| Component | Responsibility | Depends on |
|---|---|---|
| `PbfSolver` | Drives the PBF loop (predict → hash → solve → finalize) via compute dispatches | particle SSBO, `NeighborGrid` |
| `NeighborGrid` | Uniform-grid neighbor search (count → partial-sum → scatter) | `pbd` SpatialHash pattern (ported to GPU compute) |
| `CanvasDeposit` | Stamp near-floor (y≈0) particle color into canvas texture (alpha-over) + decrement absorption alpha + mark consumed | canvas storage image, particle SSBO |
| `Spoid` (POD) | Emitter state: pos(x,y,z), holeRadius, mass, emissionVelocity, pressure, amount, color, concentration | — |
| `SpoidController` (interface) | Per-frame: update spoid positions + emit triggers | — |
| `KeyboardSpoidController` | ImGui multi-select + keyboard move + Space to release a drop | `Spoid[]` |
| `Emitter` | Produces "particles to spawn this frame" per spoid and appends to particle SSBO (A: burst N, B: small per-frame count) | `Spoid[]`, particle SSBO |
| `CanvasRenderer` | Renders the canvas quad with the accumulation texture in 3D | canvas texture, `vgeu_camera` |
| `ParticleRenderer` | Debug-toggle rendering of live particles (point/sphere), optional density color | particle SSBO |
| `TextureSaver` | Canvas texture GPU→CPU readback → PNG | `stb_image_write`, soap_bubble capture pattern |
| `UiOverlay` | ImGui: per-spoid toggle/params, Save button, global sim settings | `vgeu_ui_overlay` |

---

## §2. PBF Solver (the simulation core)

Implements Algorithm 1 of Macklin & Müller 2013, decomposed into compute
dispatches. Compute↔graphics synchronization (semaphores, buffer ownership,
per-frame ordering) must follow the **`cloth` and `particle`** examples closely —
this is a known sharp edge and gets dedicated attention in the plan (§9, M2):

```
per frame, for numSubSteps:
  1. predict   : v += dt·g ; x* = x + dt·v                         [comp]
  2. hashGrid  : NeighborGrid.build over x*  (partial-sum)         [comp]
  3. solve × iters (2–4):
       - computeLambda : density constraint λ_i  (Eq. 9/11, ε CFM) [comp]
       - computeDeltaP : Δp_i (Eq. 14, scorr surface tension)
                         + canvas/wall collision response          [comp]
       - applyDeltaP   : x* += Δp                                  [comp]
  4. finalize  : v = (x* − x)/dt ; vorticity confinement ;
                 XSPH viscosity ; x = x*                           [comp]
post:
  5. deposit   : CanvasDeposit — y≈0 particles stamp canvas + dry  [comp, imageStore]
  6. compact   : remove consumed particles (free-list / stream compaction)
```

PBF detail mapping:
- **Kernels**: Poly6 for density (Eq. 2), Spiky for gradient (Eq. 7).
- **Constraint regularization**: CFM relaxation ε (Eq. 11) to avoid the
  vanishing-gradient instability.
- **Surface tension / anti-clustering**: artificial pressure `scorr`
  (Eq. 13–14), `Δq = 0.1–0.3h`, `k = 0.1`, `n = 4`.
- **Energy**: optional vorticity confinement (Eq. 16) + XSPH viscosity
  (Eq. 17, `c ≈ 0.01`).
- **Collision**: canvas is a solid half-space (z ≥ 0); walls clamp the domain.
  Collision is resolved inside the constraint loop (paper §6).
- **Solver style**: Jacobi (parallel), neighbors recomputed once per substep,
  fixed iteration count (2–4) for predictable cost.

### Spoid parameter → PBF mapping

| Spoid param | Meaning | PBF / particle mapping |
|---|---|---|
| holeRadius | hole size | emission disk radius (droplet size) |
| mass | pendulum bob mass | input to emissionVelocity/pressure (independent in Phase 1) |
| emissionVelocity | initial droplet speed | particle initial v |
| pressure | emission pressure | weights flow rate / initial speed / droplet size |
| amount | water amount | particles per drop |
| color | paint color | particle color (with FLIP-style color diffusion) |
| concentration | viscosity/opacity | XSPH `c` + stamp alpha + surface tension `k` + drying rate |

---

## §3. Painting Layer (deposition + canvas texture)

- **L2 canvas texture**: an RGBA8 (sRGB) storage image, default 2048²
  (resolution exposed as a setting), sampled by `CanvasRenderer` and saved by
  `TextureSaver`.
- **Deposition (method B)**: in the `deposit` compute pass, a particle within ε
  of the canvas floor (the X-Z plane at **y = 0**) writes `alpha-over` into the
  canvas texel at its **(x,z)→UV**, with stamp alpha derived from
  `concentration`. (World convention: screen-up = −Y, gravity = +Y, floor at
  y=0, fluid/spoids at y<0 — see the design note below.) The particle's remaining "wetness" alpha
  then decays by a per-particle **drying rate**; when exhausted, the particle is
  marked consumed and removed in `compact`.
- **Blend**: alpha-over wet-on-wet; colors mix as paint accumulates.
- **Concurrency**: deposition writes from many particles to the same texel use
  atomics or a fixed-point accumulation scheme (exact mechanism decided in the
  plan; correctness over a single frame's overlapping stamps is required).

---

## §4. Spoids, Control, and the Pendulum Hook

- `Spoid[]` is a CPU-side array (POD), uploaded to the emitter as needed.
- `SpoidController` interface: `update(dt, inputs) → writes spoid positions and
  per-spoid emit requests`.
  - **Phase 1 `KeyboardSpoidController`**: ImGui checkboxes select a subset;
    arrow keys / WASD move the *selected* spoids together (+Z height);
    Space releases one droplet from every selected spoid; ImGui edits the
    selected spoid's params.
  - **Phase 2 `PendulumSpoidController`** (deferred): drives spoid position from
    pendulum state and couples `emissionVelocity = f(mass, angularSpeed)`,
    `pressure = g(mass)`.
- `Emitter` consumes emit requests and produces this frame's spawn particles
  (A = burst of `amount`; B = small per-frame count). Same SSBO-append path for
  both, so Phase 2 stream reuses Phase 1 code.

---

## §5. Rendering, Save, and Frame Flow

- **Camera**: 3D free orbit (`vgeu_camera`). **World convention (locked):** this
  engine renders with screen-up = world **−Y** (`setViewTarget`→`lookAtLH` with
  negated up, positive viewport height, `perspectiveLH_ZO`). So gravity = **+Y**,
  the canvas floor is the X-Z plane at **y=0** (domain max-Y), fluid/spoids live
  at **y<0**, and the camera sits at y<0 (e.g. `eye=(0,-4,-4)`) looking down at
  the floor. All later milestones keep signs consistent with this.
- **Graphics**: `CanvasRenderer` (textured quad) + optional `ParticleRenderer`
  (debug particle/density view) + `UiOverlay`.
- **Compute ↔ graphics** synchronized with semaphores (reuse `particle` pattern).
- **Save**: ImGui "Save" button → `TextureSaver` does a GPU→CPU readback of the
  canvas texture and writes a PNG with `stb_image_write` (reuse soap_bubble
  offscreen-capture pattern).

Per-frame order:
```
SpoidController → Spoid[] (positions, emit triggers)
  → Emitter (append spawn particles to SSBO)
  → PbfSolver (predict/hash/solve/finalize, per substep)
  → CanvasDeposit (stamp + dry) → compact (remove consumed)
  → Graphics: CanvasRenderer + ParticleRenderer(opt) + UiOverlay
  → [ImGui Save pressed] → TextureSaver → PNG
```

---

## §6. Phasing

- **Phase 1 (this spec)**: 3D PBF solver + discrete droplet emission (A) +
  canvas deposition (B) + keyboard/ImGui control + 3D camera + PNG save (button).
  **Acceptance**: a released droplet falls, splashes with visible satellite
  droplets, leaves a mark on the canvas, and the saved PNG matches the on-screen
  canvas.
- **Phase 2 (follow-up spec)**: continuous stream emission (B),
  `PendulumSpoidController`, top-down view toggle, optional watercolor bleed
  (deposition method C / subtractive mixing).

---

## §7. Verification & Testing

No automated unit-test infra exists (consistent with other examples), so verify
by visual + numerical sanity:

- **PBF correctness**: a mini dam-break / collapsing water column settles near
  rest density without particle explosion or disappearance (visualize density
  via debug color, as in the FLIP reference and PBF Fig. 2).
- **Surface tension**: a single released droplet stays cohesive in flight and
  produces satellite droplets on impact (PBF artificial-pressure behavior).
- **Deposition**: one droplet → a mark of the expected radius; saved PNG matches
  the on-screen canvas pixels.
- **Control**: only ImGui-selected spoids move under keyboard input.
- **Tunability**: CLI flags / ImGui toggles for particle count, solver iters,
  gravity, etc. (project convention).

---

## §8. Error Handling & Edge Cases

- Particle pool (`maxParticles`) exhausted → reject emission + ImGui warning.
- Out-of-domain particles / wall contact → clamped by PBF collision constraints.
- UBO/SSBO structs → compile-time `static_assert` on size + zero-init padding
  (lesson from `BubbleParamsUbo` in soap_bubble).
- Save (file IO) failure → ImGui error message; simulation continues.

---

## §9. Implementation Milestones (user-gated, screen-verified)

The plan MUST be split into small vertical slices where **each milestone renders
something on screen and the user visually verifies + approves before the next
one starts**. Rationale: a big-bang implementation that renders nothing is very
slow to debug; early screen checkpoints localize failures. Start from the most
minimal on-screen skeleton, add compute/graphics sync with trivial data next,
then physics incrementally, then features.

| M | Slice | On-screen check (user gate) |
|---|---|---|
| M1 | Render skeleton: canvas quad (UV-checker) + 3D camera + ImGui overlay, no sim | Quad visible, camera orbits, ImGui responds |
| M2 | Particle SSBO + `ParticleRenderer` + trivial compute pass (e.g. gravity only), **compute↔graphics sync wired per `cloth`/`particle`** | A static/falling particle block renders correctly; no sync stalls/flicker/validation errors |
| M3 | PBF integrate + canvas/wall collision (no density solve) | Particles fall, hit the canvas plane, pile up without leaking |
| M4 | `NeighborGrid` + full PBF density solve + surface tension | A dropped block behaves as a cohesive incompressible fluid (mini dam-break), no explosion |
| M5 | `Emitter` + `KeyboardSpoidController` (discrete droplet) | Releasing a droplet → falls, splashes, visible satellite droplets |
| M6 | `CanvasDeposit` (stamp + dry) + compaction | Droplet leaves a permanent mark; particle count stays bounded |
| M7 | ImGui spoid multi-select + params + Save (PNG) | Select/move/edit spoids; saved PNG matches on-screen canvas |

Each milestone is a stop-and-verify gate, not just an internal checkpoint.

---

## References

- Macklin, Müller. *Position Based Fluids.* ACM TOG 32(4), 2013.
  (`pbf_sig_preprint.pdf`) — core solver.
- Müller et al. *Position Based Dynamics.* 2007 — PBD framework underneath PBF.
- Ten Minute Physics #18, FLIP fluid (`18-flip.html`) — conceptual background and
  the uniform-grid partial-sum neighbor-search pattern (shared with `pbd`).
- Project precedents: `src/examples/particle` (GPU compute multi-pipeline +
  compute/graphics sync), `src/examples/pbd` (SpatialHash partial-sum),
  `src/examples/soap_bubble` (offscreen capture, UBO size asserts).
