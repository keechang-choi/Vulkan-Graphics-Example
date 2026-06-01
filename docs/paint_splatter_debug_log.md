# Paint Splatter — Debugging Log

An ongoing record of bugs found and fixed while implementing the `paint_splatter`
GPU PBF example. Newest milestone last. Companion to the design spec
(`docs/superpowers/specs/2026-05-31-paint-splatter-pbf-design.md`) and the plan
(`docs/superpowers/plans/2026-05-31-paint-splatter-pbf.md`).

World convention (locked): screen-up = world **−Y**; gravity pulls **+Y** (down on
screen); canvas floor is the X-Z plane at **y=0** (domain max-Y); fluid/spoids live
at y<0; camera at y<0 looking at the floor.

---

## M2 — Compute/graphics sync skeleton

### Bug: unmatched queue-ownership acquire at startup (split queues)
- **Symptom:** `VUID/UNASSIGNED-VkBufferMemoryBarrier-buffer-00004` — compute
  command buffer's *acquire* barrier (graphics→compute) "has no matching release
  barrier queued for execution." Only on GPUs where graphics and compute are
  **different queue families** (here graphics=family 0, compute=family 2). Two
  errors at startup, one per in-flight frame buffer.
- **Root cause:** the very first compute dispatch on each particle buffer acquires
  queue ownership from graphics, but no graphics *release* has happened yet (the
  graphics command buffer's release runs only after the first render).
- **Fix:** skip the compute acquire barrier on the first dispatch per buffer via a
  per-frame `computeFirstUse` flag; keep the compute *release* so the graphics
  acquire still pairs. After the first frame the ping-pong is fully paired and
  validation-clean. (`buildComputeCommandBuffers`.)

### Bug: out-of-band readback broke the QFOT ping-pong
- **Symptom:** copying the particle SSBO via an out-of-band `oneTimeSubmit` (for a
  debug y-range readback) produced `VkBufferMemoryBarrier-buffer-00004` and
  missing `TRANSFER_SRC` errors.
- **Root cause:** the particle buffer participates in the compute↔graphics
  queue-ownership ping-pong; an extra graphics-queue copy inserts an unpaired use
  that breaks the release/acquire chain.
- **Fix:** (1) add `eTransferSrc` usage to the particle buffers; (2) record the
  readback copy **inside the graphics command buffer** (where the buffer is already
  owned by graphics, after the compute→graphics acquire) and read it back one frame
  later. (`recordParticleReadbackCopy` / `consumeParticleReadback`.)

---

## M4 — PBF density solve + neighbor grid

The block initially **exploded** (density rho/rho0 → hundreds, particles flung to
the domain walls, intermittent NaN). Found via systematic debugging; three real
bugs, each isolated with a minimal test.

### Bug 1: Spiky kernel gradient sign
- **Symptom:** with the density solve enabled, the block collapsed to a point then
  exploded; density grew unbounded.
- **Root cause:** the Spiky gradient was returned with a **positive** constant. The
  true gradient is negative: `∇W_spiky(r) = -45/(π h⁶)(h−r)² · (r⃗/|r⃗|)` with
  `r⃗ = pᵢ − pⱼ`. A flipped sign makes Δp point the wrong way → over-dense particles
  **attract** instead of repel → collapse.
- **Fix:** return `-kSpiky·(h−r)²·(rij/r)` in `spikyGrad` (both `pbf_lambda` and
  `pbf_delta`). λ uses only the squared magnitude so it is unaffected; only Δp's
  direction matters.

### Bug 2: XSPH viscosity not normalized (≈8078× too strong)
- **Symptom:** strong local clustering (max density rho/rho0 ≈ 7+) and instability.
- **Root cause:** XSPH summed `Σ_j (vⱼ−vᵢ)·W_poly6`. The sum of poly6 over neighbors
  is ≈ `rho0` (here 8078), so the viscosity term was ~8078× too large.
- **Fix:** divide the XSPH sum by `rho0` (density-weighted average):
  `vᵢ += xsphC · Σ(vⱼ−vᵢ)W / rho0`. (`pbf_finalize`.) This removed the clustering.

### Bug 3 (root cause of the explosion): velocity feedback runaway
- **Symptom:** even a **completely static** rest block (gravity 0, no seed jitter)
  exploded: density started near rho0 (≈1) then grew exponentially
  (1.4 → 2 → 5 → 100 → …) within ~10 frames.
- **Isolation test (decisive):** force `vel = 0` in `pbf_finalize` (pure position
  projection, no momentum). The position solver was then **perfectly stable** —
  the block stayed put, density stayed sane (≤ ~1.3), no NaN. This proved the
  position solver is correct and the **velocity feedback** drives the explosion.
- **Root cause:** PBF recovers velocity as `v = (x* − x)/dt`. With a small substep
  `dt`, a modest position correction Δp becomes a large velocity (Δp/dt); that
  velocity moves particles further next step → larger correction → positive
  feedback → exponential blow-up.
- **Fixes (standard PBF stabilization, Macklin 2013 §5):**
  - **Δp clamp** in `pbf_delta`: bound the per-iteration position correction to
    `0.2·h`.
  - **CFL velocity clamp** in `pbf_finalize`: cap speed to `0.5·h/dt` (a particle
    may not travel more than ~½ cell per substep).
  - Density-only-compression option was tried (`C = max(ρ/ρ0−1, 0)`) but removed
    because it deletes cohesion; cohesion is wanted for droplets.

### Result
Stable, VL-CLEAN, no NaN/device-loss. Incompressibility works (max density reaches
~rho0). Density-debug coloring added: `pbf_finalize` writes `rho/rho0` into
`vel.w`; `particle.vert` maps it to a blue→cyan→red sci-color when
`GlobalUbo.canvasInfo.w` (the "color by density" toggle) is set.

### Open issue (under investigation): gas-like dispersal
The fluid is stable but **disperses to fill the domain** (including the ceiling at
y=−3, against gravity) at low density (mean rho/rho0 ≈ 0.37) instead of pooling at
the floor — it behaves more like a gas than a liquid. User confirmed on screen
("기체처럼 흩어져서 돌아다님").

Deeper analysis (recorded before fixing):
- **PBF's density constraint has no restoring force against *uniform* expansion.**
  If the whole fluid expands uniformly to ρ=0.37ρ0, every particle sees the same
  low density, the neighbor ∇W terms are symmetric and cancel (Σ∇W ≈ 0), so Δp ≈ 0
  in the bulk. The constraint resists *compression gradients*, not uniform
  rarefaction. Real PBF fluids stay together via **gravity confinement** (piling
  at a floor compresses the bottom) plus **surface tension (scorr)** — not the
  density constraint alone.
- **The CFL velocity clamp is too permissive.** With substeps=1, dt=1/120, the cap
  `0.5·h/dt ≈ 6 u/s` lets a particle rise ballistically `v²/2g = 6²/19.6 ≈ 1.8`
  world units against gravity — enough to reach the ceiling. So the fluid fills the
  3D domain instead of settling.
- **XSPH only damps *relative* velocity** `(vⱼ−vᵢ)`, not bulk motion. The kinetic
  energy from falling becomes bulk motion that XSPH does not dissipate, so the
  fluid stays "hot" and agitated.

Candidate fix being tested: add a **global velocity damping (drag)** in
`pbf_finalize` to dissipate bulk kinetic energy so the fluid can settle, exposed as
an ImGui slider (`velDamp`, per-second drag rate; kept low enough to preserve
splash dynamics for M5). Secondary: lower the velocity-clamp factor.

**Results (readback `nearFloor%` = fraction of particles within 0.3 of floor):**
- `velDamp=2`: still fills domain (min y reaches −3, mean rho/rho0 ≈ 0.38). Too weak.
- `velDamp=6`: no particles at the ceiling (min y ≈ −2.3), ~56% near the floor —
  the fluid now pools downward instead of filling the domain. But it stays a
  **loose, fluffy low pile** (mean rho/rho0 ≈ 0.40, max ≈ 1.0): under the weak
  hydrostatic pressure of a thin layer the fluid does not compress all the way to
  rho0, and the Δp clamp (0.2h) limits how fast cohesion can pull it together.
- **Conclusion:** damping fixes the gas-like *dispersal/wandering* (energy now
  dissipates, fluid falls and stays near the floor), but a denser puddle needs
  stiffer cohesion (relax the Δp clamp now that velocity is damped) and/or stronger
  gravity. This is now a tuning target best dialed in live via the ImGui sliders
  (`velDamp`, `epsCFM`, `scorrK`, Δp clamp, gravity) with the user watching, rather
  than blind build-run-readback loops.
- **Relaxing the Δp clamp to 0.5h made it WORSE** (nearFloor 56% → 20%, fills domain
  again): larger position corrections fling particles apart faster than damping
  removes the energy. Reverted to 0.2h. So the best blind config is **Δp 0.2h +
  velDamp≈6** (≈56% near floor, nothing at the ceiling). Default `velDamp=6`.

### Why the density debug view is all blue (not a bug)
`particle.vert` maps `vel.w = rho/rho0` to blue(≤0.5)→cyan(≈1.0)→red(≥1.5). With
the fluid spread thin over the full 4×4 canvas it never reaches rho0, so every
particle is ≤0.5 → all blue. This faithfully shows the real issue: the dropped
block, spread over the large canvas footprint, is a sub-monolayer (mostly surface
particles) and cannot reach rest density. Each particle is a *fluid parcel* (not a
molecule); ~13824 of them approximate the water.

### M4 verification scene → confined dam-break
A block dropped onto the full 4×4 canvas spreads into a thin, all-blue film that is
hard to read as "water". To verify the solver convincingly, switch the **M4 test
scene** to a **confined dam-break**: a smaller fluid box (collision walls +
neighbor grid over `x,z ∈ [−kFluidHalf, kFluidHalf]`, `kFluidHalf=1.0`) with a tall
narrow water *column* seeded against one wall. On release it collapses, sloshes,
and forms a **deep pool that reaches rho0** (green/cyan in the density view) — the
classic incompressible-water sanity test. The full-canvas domain returns for M5
(droplets), where dense pooling is not needed. (Recording before implementing.)

### Bug: restart re-triggered the M2 startup QFOT issue
- **Symptom:** clicking **Restart** produced
  `UNASSIGNED-VkBufferMemoryBarrier-buffer-00003` warnings — a graphics→compute
  *release* "duplicates existing barrier queued for execution, without intervening
  acquire operation."
- **Root cause:** `restartSimulation` reset `computeFirstUse = 1`, making the next
  compute dispatch **skip its acquire**. But unlike at startup, the buffers were
  mid-ping-pong with a graphics→compute release already pending. Skipping the
  acquire left that release unconsumed, so the next graphics release duplicated it.
- **Fix:** do **not** reset `computeFirstUse` on restart — the buffers are already
  bootstrapped, so the compute should keep acquiring normally and consume the
  pending release. (Startup still sets the flag once in `prepareCompute`.)

---

## M5 — Emitter + keyboard spoid control

No bugs hit during implementation; recording the key design decisions that kept
it correct on the first build.

### Append without a GPU counter (deferred to M6)
The plan calls for a GPU `liveCount` atomic in M5, but M5 has **no compaction**
(particles are never removed until M6), so the slot range is strictly
append-only and the **host** already knows the next free index. `emit.comp`
writes `particle[baseIndex + i]` directly — no `atomicAdd`, no indirect dispatch.
The GPU counter arrives in M6 where concurrent removal actually needs it.

### Per-frame emit queues keep the two sims in lockstep
The two per-frame particle buffers are **independent-but-identical** sims (no
ping-pong; kept identical by determinism — see M4 note). Emission would break
that: a burst enqueued on frame F only touches the buffer rendered at F. Fix:
`enqueueDrop` reserves the slot range **once** from the shared host `numParticles`
(computing `baseIndex`) and pushes the *same* `EmitPush` onto **every** per-frame
buffer's FIFO. Each buffer drains its queue (one emit dispatch per burst) at the
**start** of its compute command buffer, before predict, so the new slots are
written before any pass reads them. Because `baseIndex` is fixed at enqueue time
and `emit.comp` jitters deterministically per slot, both buffers write identical
particles into identical slots → no divergence/flicker.

### Emit pipeline needs its own layout (push-constant incompatibility)
`EmitPush` is a push constant, so the emit pipeline layout = compute descriptor
set layout **+ a push-constant range**. That differs from the PBF passes' layout
(no push constants), and the two are *not* compatible for descriptor binding, so
`recordEmit` re-binds the particle SSBO via `emitPipelineLayout` before
dispatching; the substep loop then re-binds via the PBF `pipelineLayout`.

### Key bindings: spoids avoid the camera's keys
The engine's free camera already owns **WASD/QE + arrows** (`vge_base.cpp`
`cameraController.moveInPlaneXZ`). To avoid fighting it, the
`KeyboardSpoidController` uses a separate set: **IJKL** (canvas plane), **U/O**
(height), **Space** (edge-triggered drop from each selected spoid). GLFW reading
lives in the example (`updateSpoids`); the controller takes a source-agnostic
`InputState` so the Phase-2 pendulum controller can swap in.

### M5 state / verification
Domain widened from the M4 confined box (`kFluidHalf=1`) to the full canvas
(`kDomainHalf=2`); sim starts empty. Task 9 auto-emit run: VL-CLEAN, droplets
spawn near the ceiling, fall, splash on the floor (`nearFloor%` 0→55%, transient
impact density bounded ~3, no explosion). Task 10 idle: VL-CLEAN, spoid marker
renders. Keyboard/Space interaction is the user's M5 gate.

### Bug: droplet over-packing → explosive emit (scatter)
- **Symptom:** dropped droplets exploded into spray on emit.
- **Root cause:** `amount` (300) particles spawned into a thin hole disk packed
  ~7× rest density; the density solve blasted the over-compressed blob apart.
- **Fix:** size the spawn volume to `amount` at rest density — emit uniformly in
  a 3D ball of host-computed radius `max(holeRadius, cbrt(3·amount·spacing³/4π))`.
  Impact density peaked ~3.0 → ~0.9. (`enqueueDrop` + `emit.comp` ball sampling.)

### THE big one: velocity-feedback explosion was the UNCLAMPED density constraint
- **Symptom (user):** even after the over-packing fix, particles "boil" — too
  much repulsion, bouncing around the whole domain. Measured: max particle speed
  pinned at *exactly* the CFL clamp ceiling (`velClampFactor·h/dt`) every frame,
  independent of gravity; lowering the ceiling just re-pinned there; **removing**
  the clamp and adding substeps made it far WORSE (speed ~140).
- **Why substeps backfired:** the per-substep position correction stays ~one
  particle spacing (driven by the constraint residual, not motion), so the
  velocity recovery `v=(x*−x)/dt` *grows* as dt shrinks. Small-steps only helps
  when the fluid is near its constraint manifold (corrections → 0); ours never is.
- **Root cause:** `pbf_lambda` used `C = rho/rho0 − 1` **unclamped**. In the
  sub-monolayer regime (sparse fluid over the 4×4 canvas, mean rho ≈ 0.4·rho0)
  almost every particle is under-dense → `C<0` → `λ>0` → the density constraint
  perpetually **pulls under-dense particles together**. Those corrections ÷ dt
  became the velocity that flung particles into neighbours → chain reaction. The
  CFL clamp only masked the symptom.
- **Fix:** compression-only constraint `C = max(rho/rho0 − 1, 0)` (standard
  free-surface PBF). Under-dense → no density correction; only real overlaps are
  resolved; droplet cohesion is left to **scorr (surface tension)**, its proper
  role. (M4 had tried this and reverted it for "lost cohesion" — but that reveals
  the cohesion was coming from the *unstable* under-density pull; scorr is the
  correct mechanism.)
- **Result:** stable with **NO velocity clamp** (`velClampFactor=0`), substeps=1,
  iters=4. speed mean 3.0→0.25, max 6(pinned)→~0.7 (brief falling-droplet spikes
  only), nearFloor 20%→78% — the fluid now pools calmly at the floor. The CFL
  clamp is kept only as an optional safety net (slider, 0 disables).
