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

### Spawn distribution + Jacobi under-relaxation (partial mitigations)
- Spawn "explosion" partly from random-in-ball placing accidental close pairs →
  switched emit to a **jittered cube lattice** (regular rest spacing + jitter <
  spacing/2) so a minimum separation is guaranteed (`emit.comp`, `enqueueDrop`).
- Pile "flicker"/oscillation partly Jacobi (parallel) solver overshoot →
  **under-relaxation** `dp *= solverRelax` (0.3) in `pbf_delta` (Macklin 2013).
  Helped but did NOT remove the floor jiggle (see below — the real cause).

### THE flicker root cause: per-frame particle buffers DIVERGE (open / unfixed)
- **Symptom (user):** piled particles (and to a lesser extent everything)
  "flicker"/jitter, worst where many particles pile densely on the floor. None
  of the solver tuning (relax, damp, iters, clamp, substeps) removed it.
- **Decisive measurement:** added a one-shot debug that copies BOTH per-frame
  particle SSBOs to the host and diffs them. After ~5 s with ~2400 particles:
  **`DIVERGENCE buf0 vs buf1: max=1.30  mean=0.19` world units.** The two
  per-frame buffers hold *completely different* fluid states (mean 0.19, max 1.3
  ≈ a third of the canvas).
- **Root cause (architecture, NOT the solver):** this example steps each
  per-frame particle buffer **in place and independently** — `particleBuffers[cur]`
  is read+written every frame with no cross-buffer coupling — so the two
  `MAX_CONCURRENT_FRAMES` buffers are **two independent simulations**. They were
  assumed identical ("deterministic, same seed"), but they diverge because the
  neighbour build is **non-deterministic**: `grid_scatter` uses
  `atomicAdd(cellOffset)` to place particle ids, so the intra-cell order in
  `sortedIds` differs between the two buffers' builds. Float SPH sums are
  non-associative → different rounding → exponential (chaotic) divergence over
  steps. The renderer **alternates** which buffer it draws each frame, so the
  screen flips between two different sims → flicker. Densest at the floor pile
  where positions are most sensitive.
- **Why the Jacobi solve itself needs no atomics (answer to a related Q):** in
  Jacobi each thread writes **only its own** particle (`deltaP[i]`, `predict.w`,
  `predict.xyz`), reading neighbours' positions from the same iteration's
  snapshot, so there are no write conflicts and no atomics in lambda/delta/apply.
  Atomics are used **only** in the grid build (count + scatter) where they are
  needed — and the scatter's atomic *ordering* is exactly what makes the two
  independent sims diverge.
- **CORRECT FIX (matches `particle`/`cloth`): ping-pong, not single-buffer.**
  `particle.cpp`/`cloth.cpp` run multi-buffer simulations correctly by **reading
  the previous frame's buffer and writing the current** (`prevFrameIndex` →
  `currentFrameIndex`) — ONE evolving chain that still pipelines across frames.
  This example instead steps in place, which is the bug. Fix plan:
  1. `predict` reads `particleBuffers[prevFrameIndex]`, writes
     `particleBuffers[currentFrameIndex]` (the rest of the PBF passes stay
     in-place on `cur`); emit appends new particles to `cur`.
  2. `prevFrameIndex = (currentFrameIndex + MAX-1) % MAX`. The existing
     compute↔graphics semaphore handshake already supports this (it's why
     `particle.cpp` pairs `graphics.semaphores[cur]` the way it does).
  3. Per-frame emit queues collapse to a single pending list (one chain, emit
     once), and `numParticles` is a single global live count for the chain.
  A single-buffer + serialize approach also works but needlessly gives up the
  cross-frame pipelining that `particle`/`cloth` keep — so ping-pong is the
  right design here.
- **Status:** root cause confirmed and documented; fix (ping-pong) now
  IMPLEMENTED (see below). Solver-side mitigations (compression-only, relax) are
  committed and correct on their own; the lattice spawn was later reverted to
  ball sampling (see "Spawn distribution" update below).

### FIX IMPLEMENTED: ping-pong chain (read prev, write cur)
Replaced the in-place per-buffer stepping with the `particle.cpp`/`cloth.cpp`
ping-pong so the two `MAX_CONCURRENT_FRAMES` buffers form ONE evolving sim.
- **Descriptor layout:** added compute SSBO binding **7 = particlesPrev**
  (read-only). Descriptor set `i` binds binding 0 -> `particleBuffers[i]` (cur,
  written) and binding 7 -> `particleBuffers[(i-1+N)%N]` (prev, read) — mirrors
  `particle.cpp:252-261` (`prevFrameIdx` -> in, `i` -> out). Pool bumped to
  `MAX*7` storage buffers.
- **`pbf_predict.comp`:** reads `pprev[i]` (binding 7), writes the full `p[i]`
  (binding 0): carries `pos`/`color`/wetness forward and integrates
  `vel`/`predict`. The remaining PBF passes (grid, lambda, delta, apply,
  finalize) stay in place on `cur`. finalize commits `pos = predict` on `cur`,
  which next frame is read as `prev`.
- **Emit as one chain:** per-frame `emitQueues` collapsed to a single
  `pendingEmits` list, drained+cleared once per frame into `cur`'s
  `[prevCount, numParticles)` slots. `numParticles` is the single global live
  count; `prevParticleCount` tracks the count at the end of the previous frame
  (== what the prev buffer holds) and is fed to the shader as `ComputeUbo.prevCount`.
- **predict skips freshly-spawned slots:** `if (i >= prevCount) return;` — those
  `[prevCount, numParticles)` particles were just written into `cur` by
  `emit.comp` this frame, so predict must not clobber them with stale `prev` data.
- **Barriers unchanged:** QFOT acquire/release still only on `cur`
  (`particleBuffers[currentFrameIndex]`); prev is read without an extra acquire,
  exactly as `particle.cpp` does (validated reference).
- **ComputeUbo:** `prevCount` (uint) added at offset 112 + 3 pad uints; size
  assert bumped 112 -> 128.
- **Verification (autoEmit on, 10 s):** VL-CLEAN (zero validation/STDERR),
  `rho/rho0` mean 0.19->0.37 (no explosion, max ~0.57), speed mean 0.13-0.7
  (occasional ~6 = falling-droplet spikes), `nearFloor%` 0->88% (calm floor
  pile), no NaN / device-loss. On-screen flicker disappearance is the user's
  visual gate. autoEmit defaulted ON so the scene is alive on launch.

### Spawn distribution: lattice -> ball -> back to lattice
Briefly switched the jittered cube lattice to **uniform-ball** sampling (option
B), with the host (`enqueueDrop`) passing a rest-density-sized ball radius
`max(holeRadius, cbrt(3*count*spacing^3 / 4pi))` so the ball volume is not
over-packed. (The earlier broken state had the ball shader reading a *lattice
spacing* as the radius, over-packing ~100x; fixed by passing a real radius.)
**Then reverted to the jittered lattice again:** even a rest-density-sized ball
uses *random* placement, which still produces occasional close pairs whose local
over-density pops on the first solve (a likely cause of the "emit explosion" the
user suspected). The lattice guarantees a minimum separation by construction, so
no spawn pop — at the cost of a grid-shaped (vs round) initial droplet. Spacing
(`kParticleSpacing = 0.05 = 0.5h`) is the PBF-standard rest spacing (~33
neighbours within h); not the over-packing culprit.

### Close-encounter explosion fixed by shrinking rest spacing (h kept fixed)
User observed particles still "pop" not only on spawn but **whenever they get
close** during the sim, and found that dropping `kParticleSpacing` to 0.005 made
it go away. Confirmed and reasoned out:
- **`kParticleSpacing` is the rest separation** — the distance below which
  particles read as over-packed and repel. Explosions happen when particles get
  much closer than spacing (collision/accumulation) → local density spikes → λ
  spikes → Δp/dt becomes a large velocity. Smaller spacing means a given close
  distance is no longer "over-packed", so no repulsion → no pop.
- **`kSmoothingRadius` (h) is the GRID CELL size; do NOT shrink it.** The user's
  attempted `h = 0.01` blew `numCells` from 40³=48k to 400³=48M → multi-GB
  cell buffers + a single-invocation serial prefix scan over 48M cells → stall /
  error. h is independent of the explosion; only spacing matters there.
- **Set h=0.1 (grid unchanged, 48k cells), spacing=0.005 independently.**
  Measured (auto-emit, 12 s, VL-CLEAN, no perf drop): speed max 6.3→**1.9**
  (pop gone), nearFloor ~87%. rho0 8078→**8e6**, so rho/rho0 ~**0.001**: the
  compression-only density constraint is now effectively OFF — that is *why*
  there is no pop (no density repulsion), with cohesion left to scorr +
  collision. Trade-off: ~no incompressible-pool behaviour; fine for a paint
  splatter demo, tune spacing up (0.01–0.02) if droplets look too dispersed.
  Note emit lattice spacing = max(restSpacing, 2*holeRadius/side), so droplet
  size still follows holeRadius; restSpacing only sets rho0 / the repel floor.

### OPEN ISSUE: crown splash / satellite droplets not happening (future tuning)
User expectation: a droplet hitting the floor (or a wall) should throw up a
radial crown + satellite droplets, which the current build does NOT do. This is
a direct consequence of the close-encounter fix above: `kParticleSpacing=0.005`
makes `rho/rho0≈0` so the compression-only density constraint is effectively
off, and a crown needs the opposite — strong incompressible rebound at impact to
launch a radial sheet. So stability (no pop) and crown expressiveness are in
tension here. No new theory is strictly required (PBF + scorr + collision can
produce it in principle), but a dedicated splash-tuning pass is needed: restore
partial incompressibility (mid spacing ~0.02 + velClamp/Δp-clamp to bound the
pop) + higher emissionVelocity + lower velDamp + more particle resolution, and/or
an impact-time momentum-redistribution heuristic (convert floor-normal velocity
into radial v.xz on contact). Recorded in design spec §6. Deferred to a tuning
milestone after M6 (deposition).

그리고 이런 파이프라인으로 렌더까지 이어지게 할 수 있는지도 검토 필요.
네가 처음 이야기한

종이에 떨어진 물방울

은 사실 일반 물 렌더링과 다르다.

중요한 건

얇은 액체막
+
젖은 종이

이다.

그래서 나는 오히려

PBF↓
Surface Thickness Texture↓
Wetness Map↓
Paper Shader

를 추천한다.

물방울이 살아있는 동안만 PBF를 쓰고

충돌 후에는

wetness
absorption
capillary diffusion

로 넘어가는 구조.

실제로 네가 언리얼이나 Vulkan에서 직접 만든다면 현재 그래픽스 업계에서 가장 흔한 구조는:

PBF / FLIP Particles↓
Particle Depth Rendering↓
Gaussian Blur↓
Normal Reconstruction↓
Thickness Buffer↓
Fresnel↓
Screen Space Refraction↓
Final Composite

이다.

오프라인 VFX는 "메쉬 생성 후 렌더", 실시간은 "스크린 스페이스 유체 렌더링"이라고 생각하면 거의 맞다. 특히 작은 물방울 수천~수만 개를 실시간으로 다룰 때는 메쉬 생성 비용 때문에 스크린 스페이스 접근이 훨씬 많이 사용된다.

## M6-C-2 — Compaction (IMPLEMENTED 2026-06-03; design blueprint below)

**DONE & self-verified (VL-CLEAN + bounded live count).** Implemented the
blueprint below with one robustness deviation. autoEmit (~300 particles / 0.6s)
ran 11s: live count held a steady ~900-1200 band (was unbounded -> ~5-6k before
compaction), with frame-to-frame drops (1200->1041, 1200->944) proving
compaction removes dried particles; STDERR validation-clean; physics stable
(rho bounded, speed ~0.7, particles reach floor, no explosion). Awaits the user
visual GATE (marks accumulate / paint dries / count bounded on screen).

Deviations from / refinements to the blueprint:
- **predict reads the EXACT prev count from the GPU (new binding 10 =
  prevLiveCount), not a host `prevCount`.** The blueprint's host prevCount lags
  the prev buffer by one ping-pong cycle (N=2: a slot's liveCount readback is
  only available 2 frames later), so it would mismatch the prev buffer's true
  range -> over-count re-introduces stale slots as ghosts, under-count drops
  live particles. binding 10 = liveCountBuffers[prevIdx] (the prev slot's own
  counter, written when it was last `cur`, NOT reset this frame) gives predict
  the exact range: `if (i >= prevLiveCount[0]) return;`. Mirrors the particle
  0/7 ping-pong. So correctness no longer depends on readback accuracy.
- **liveCount never crosses queues.** Only compute reads/writes it (predict +
  emit atomicAdd into binding 9; grid/lambda/delta/apply/finalize guard on it).
  deposit (graphics queue) does NOT read liveCount -- it keeps its host-`count`
  + geometric guards and rejects the parked tail by uv-out-of-range. So no QFOT
  on the counter; the per-frame readback copy is a same-queue transfer.
- **finalize parks the stale tail offscreen.** The buffer is packed
  [0, liveCount); the host dispatches/draws an UPPER BOUND. finalize sets
  `p[i].pos = vec4(1e18)` for `i >= liveCount[0]` so the point-draw clips them
  and deposit rejects them, without needing draw-indirect or an exact host draw
  count. Next frame's compaction reads only [0, prevLiveCount) so never picks
  them up.
- **Host counts are conservative UPPER BOUNDS, not the live count.**
  numParticles = min(kMax, prevBound + emittedThisFrame); prevBound is shrunk by
  a readback-derived bound (`live_now <= R + emittedSince`, provably >= true) so
  dispatch stays tight instead of pegging at kMax. ImGui shows the readback
  live count. Readback is display + dispatch-tightening only; GPU guards own
  correctness (matches plan Task 12's "readback not for control flow" spirit).
- predict-compaction + emit run ONCE per frame (re-running the atomic per
  substep would double-count); recordPbfSubstep now starts at the grid rebuild.

Files touched: shaders pbf_predict/emit/grid_count/grid_scatter/pbf_lambda/
pbf_delta/pbf_apply/pbf_finalize.comp; paint_splatter.hpp/.cpp (liveCount
buffers + bindings 9/10, pool 7->9 SSBO, enqueueDrop/updateComputeUbo/
consumeLiveCountReadback/buildComputeCommandBuffers/restartSimulation/ImGui).
grid_scan.comp and deposit.comp unchanged.

### Post-M6-C-2 visual tuning (2026-06-03, user iteration)
- dry settle (pbf_finalize): near-floor velocity scaled by wetness so drying
  particles freeze (paint setting) instead of sliding. `drySettle` slider.
- deposit/sprite size: `depositRadius` (world units) replaces the fixed h*0.5
  stamp; `pointScale` (GlobalUbo.renderParams.x) scales gl_PointSize. Both
  sliders. ComputeUbo 128->144 (+depositRadius,drySettle,pad0,pad1); GlobalUbo
  208->224 (+renderParams). VL-clean confirmed with the new sizes.
- deposit flicker mitigation: depositStrength default lowered (0.5->0.03) so the
  per-frame alpha is small -> compaction's per-frame particle REORDER makes the
  order-dependent alpha-over oscillate less at color overlaps (the "z-fighting
  between colors" the user saw is this reorder x order-dependence, not 3D depth).
- dryRate default 0.5->2.0 (paint absorbs faster -> lower steady live count).
- restart now also resets spoid colour (palette) + position (arrangeSpoidsCircle);
  count + other params kept (user-approved).
- ImGui: most float params -> DragFloat, amount/substeps/solverIters -> DragInt
  (fine control); "edit ALL spoids" radio propagates a param edit to every spoid.

### DEFERRED (future work) — order-independent deposit accumulation
The low-depositStrength PALENESS and the color-overlap FLICKER are two sides of
the SAME alpha-over limitation: rich-immediate color needs high per-stamp alpha
(-> order-dependent -> flickers when compaction reorders particles), smooth
mixing needs low alpha (-> blends toward the WHITE canvas base -> pale until
enough stamps accumulate, which fast drying prevents). The proper fix is an
ORDER-INDEPENDENT weighted accumulation: per texel store sum(paint_i * w_i) and
sum(w_i) via imageAtomicAdd (addition is commutative -> order-independent), and
canvas.frag outputs sum_color/sum_w. Gives saturated color + correct color
mixing + zero flicker, all independent of particle order. Cost: canvas RGBA8 ->
atomic-capable storage (4x R32_UINT fixed-point images, or packed), rewrite
deposit.comp (atomicAdd), canvas.frag (normalize on read), the PNG save path
(readback + normalize), and restart clear. Sizable -- its own milestone. User
deferred it (2026-06-03); current alpha-over model kept for now.

---

### Original blueprint (design, pre-implementation)
Goal: remove dried-out particles (pos.w<=0) so the live count stays bounded; the
"droplet absorbed into the canvas and gone" behaviour. The hard part: the live
count becomes a GPU-dynamic value, but the solver currently dispatches/guards on
the host's fixed `numParticles`. Chosen design (ping-pong-compatible):

- **`liveCountBuffers[frame]`**: a `uint[1]` storage buffer, compute set binding
  9. Reset to 0 (vkCmdFillBuffer) at the top of each frame's compute cmd buffer.
- **predict becomes the compaction step.** Dispatch `prevCount` threads
  (`ComputeUbo.prevCount` = host's previous live count, readback-delayed one
  frame). Each: `if (i>=prevCount) return; Particle s=pprev[i]; if (s.pos.w<=0)
  return;` (drop dead) `uint d=atomicAdd(liveCount,1); ...; p[d]=integrated(s);`.
  So cur is packed [0,liveCount) with only survivors, carried from prev.
- **emit appends after predict via the same atomic.** Each spawned particle:
  `uint d=atomicAdd(liveCount,1); if (d>=kMax) return; p[d]=spawn;`. No host
  baseIndex anymore (the host counter can't know the post-compaction index).
- **All other passes guard on `liveCount[0]`, not `u.count`.** grid_count,
  grid_scatter, pbf_lambda/delta/apply, pbf_finalize, deposit read binding 9 and
  `if (gid >= liveCount[0]) return;`. They are dispatched on a host UPPER BOUND
  (prevCount + emittedThisFrame, <= kMax); the guard trims the slack. grid_scan
  is unchanged (works over numCells).
- **Host**: `numParticles` stops being the exact count and becomes the dispatch
  upper bound; a one-frame-delayed readback of `liveCount` gives the real count
  for ImGui, pool-full checks, and next frame's `prevCount`. Reuse the existing
  in-graphics-cmd readback pattern (a tiny uint copy) to avoid breaking the QFOT
  ping-pong.
- Barriers: fill(0)->predict (transfer->compute), predict->emit, emit->grid, as
  today; liveCount is written by predict+emit (atomic) and read by every later
  pass, so a compute->compute barrier already covers it.

Risk notes: predict's atomicAdd reorders particles (intra-frame nondeterminism)
but that's fine now — ping-pong made it ONE chain, so reordering doesn't fork two
sims. dispatch upper bound must never under-count (would drop live particles), so
host tracks prevCount conservatively (use last readback; on the very first frames
liveCount=0).

### Reference for a future hybrid direction (not implemented)
Chentanez, Müller, Kim, *Coupling 3D Eulerian, Heightfield and Particle Methods*
(SCA 2014) — couples PBF/SPH particles + a 3D Eulerian grid + an SWE height
field via a shared density field (grid `g` + particle `rho_p` = combined `c`).
Directly addresses our M4 "sub-monolayer / gas-like dispersal" limitation:
*particles alone are a poor way to represent bulk liquid*. A natural fit for a
later phase of paint_splatter — thin canvas paint film as a height field,
splash droplets as PBF particles — but a large architecture change, out of scope
for Phase 1. Recorded as a Phase 3 candidate in the design spec.
