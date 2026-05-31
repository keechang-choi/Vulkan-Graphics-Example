# Paint Splatter (GPU PBF) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `paint_splatter` Vulkan example where parametrized eyedroppers (spoids) release paint droplets that fall, splash, and bounce on a quad canvas via a GPU Position-Based-Fluids solver, depositing color into a permanent canvas texture saveable to PNG.

**Architecture:** GPU-compute 3D PBF (Macklin & Müller 2013) drives live fluid particles; a separate accumulation texture is the permanent painting. Compute and graphics run on separate queues synchronized with per-frame semaphores + queue-ownership barriers, following `src/examples/particle` and `src/examples/cloth`. The plan is split into 7 user-gated milestones (M1–M7); each ends on screen and MUST be visually approved by the user before the next begins.

**Tech Stack:** C++17, Vulkan-Hpp RAII, GLSL compute/vertex/fragment (compiled to SPIR-V), VMA, ImGui (`vgeu_ui_overlay`), GLFW, `stb_image_write`, CLI11.

---

## Verification Model (read first)

This repo has **no unit-test framework**; examples are verified by running the app. This plan therefore uses these verification primitives instead of pytest:

- **VL-CLEAN**: run with validation layers enabled (default debug build) and confirm **zero validation errors/warnings** in stdout.
- **ASSERT**: compile-time `static_assert` on every GPU-facing struct's `sizeof` (std140/std430), plus runtime `assert` on buffer sizes/counts.
- **READBACK**: a debug readback of a mapped SSBO/texture to print a numeric sanity value (count, min/max, mean) to stdout.
- **GATE (user)**: a STOP. Build, run, and the **user visually confirms** the milestone's on-screen check. Do not start the next milestone until the user says it passes.

Each milestone ends with a `GATE (user)` step. The executing agent must pause there.

---

## Conventions & Shared Facts

- **Example dir:** `src/examples/paint_splatter/{paint_splatter.cpp,paint_splatter.hpp}`. Shaders: `shaders/paint_splatter/*.{comp,vert,frag}`. CMake auto-discovers the example dir (`src/examples/CMakeLists.txt` globs subdirectories) and shaders (verify the shader build glob in the root `CMakeLists.txt` during Task 0).
- **Base class:** subclass `VgeBase` (see `src/examples/particle/particle.hpp:93`). Frames-in-flight = `MAX_CONCURRENT_FRAMES` (use the same constant other examples read; confirm in `vge_base.hpp`). All per-frame GPU resources are vectors indexed by `currentFrameIndex`.
- **Buffers:** use `vgeu::VgeuBuffer` (`src/base/vgeu_buffer.hpp`) and `vgeu::VgeuImage` for allocations (VMA-backed).
- **Sync handshake (the sharp edge — copy exactly):** the compute↔graphics ping-pong from `particle.cpp:1356-1394` (submit) + `particle.cpp:1413-1431` (graphics acquire barrier) + `particle.cpp:1510-1640` (compute acquire/release barriers). Reproduced concretely in M2.
- **GPU struct rule:** every struct shared with a shader gets explicit `// -- 16 --` boundary comments, zero-init `_padN` members, and a `static_assert(sizeof(...) == N, ...)` — lesson from `BubbleParamsUbo` (`soap_bubble.hpp:85-118`).
- **clang-format:** run `clang-format -i` on every edited `.cpp/.hpp` before each commit (user rule).
- **Build:** `mingwBuild.bat` (Windows/MinGW) from repo root. Shaders compile as part of the build (confirm in Task 0). Run the example binary from `build/` (`set_target_properties ... RUNTIME_OUTPUT_DIRECTORY .../build`).
- **rtk:** prefix shell commands with `rtk` per user rule (e.g. `rtk git commit ...`).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/examples/paint_splatter/paint_splatter.hpp` | `VgeExample` class, all GPU struct layouts, `Spoid`, `Options` |
| `src/examples/paint_splatter/paint_splatter.cpp` | setup, per-frame `render()/draw()`, command-buffer builders, UI |
| `shaders/paint_splatter/canvas.vert` `canvas.frag` | textured canvas quad (M1) |
| `shaders/paint_splatter/particle.vert` `particle.frag` | point/disk particle debug render (M2) |
| `shaders/paint_splatter/pbf_predict.comp` | integrate ext. forces, predict x* (M2/M3) |
| `shaders/paint_splatter/grid_count.comp` `grid_scan.comp` `grid_scatter.comp` | uniform-grid neighbor build (M4) |
| `shaders/paint_splatter/pbf_lambda.comp` | density constraint λ (M4) |
| `shaders/paint_splatter/pbf_delta.comp` | Δp + scorr + collision (M4) |
| `shaders/paint_splatter/pbf_apply.comp` | x* += Δp (M4) |
| `shaders/paint_splatter/pbf_finalize.comp` | velocity update + vorticity + XSPH (M4) |
| `shaders/paint_splatter/emit.comp` | spawn particles from emit requests (M5) |
| `shaders/paint_splatter/deposit.comp` | stamp canvas + dry + mark consumed (M6) |
| `shaders/paint_splatter/compact.comp` | remove consumed particles (M6) |

Helper logic (`Spoid`, `SpoidController`, deposition params) lives inside `paint_splatter.{hpp,cpp}` to start (single example owner). Split into separate translation units only if `paint_splatter.cpp` grows past ~1500 lines (matching project norms where `cloth`/`pbd` are large single files).

---

## Milestone 1 — Render Skeleton (canvas quad + camera + ImGui)

**Goal of M1:** the example compiles, opens a window, shows a textured quad (UV checker) you can orbit, with a working ImGui panel. No simulation, no compute.

### Task 0: Scaffold the example and confirm the build wires it up

**Files:**
- Create: `src/examples/paint_splatter/paint_splatter.hpp`
- Create: `src/examples/paint_splatter/paint_splatter.cpp`
- Read-only: `src/examples/CMakeLists.txt`, root `CMakeLists.txt` (shader glob)

- [ ] **Step 1: Copy the smallest example as a skeleton.** Start from `src/examples/triangle/triangle.{cpp,hpp}` (the minimal `VgeBase` subclass). Rename the class members/namespace usage to `paint_splatter`, strip triangle-specific geometry. Keep: `initVulkan`, `getEnabledFeatures`, `prepare`, `render`, `viewChanged`, `setupCommandLineParser`, `onUpdateUIOverlay`, `buildCommandBuffers`, `draw`. The file must define `namespace vge { class VgeExample : public VgeBase {...} }`.

- [ ] **Step 2: Add a `main` entry.** Confirm how triangle's binary entry is produced — `src/examples/CMakeLists.txt:4` sets `MAIN_CPP = <example>/<example>.cpp`, so `paint_splatter.cpp` must contain the `main()` (or the macro other examples use, e.g. `VULKAN_EXAMPLE_MAIN()` — copy whatever `triangle.cpp` uses verbatim).

- [ ] **Step 3: Build.** Run: `rtk cmd /c mingwBuild.bat` (from repo root). Expected: a `paint_splatter` target builds and links; binary appears in `build/`.

- [ ] **Step 4: Run.** Run: `./build/paint_splatter.exe`. Expected: window opens with the triangle-derived placeholder, VL-CLEAN.

- [ ] **Step 5: Commit.**
```bash
rtk git add src/examples/paint_splatter shaders/paint_splatter
rtk git commit -m "feat(paint_splatter): scaffold example from triangle skeleton"
```

### Task 1: Canvas quad geometry + shaders + 3D camera

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.{hpp,cpp}`
- Create: `shaders/paint_splatter/canvas.vert`, `shaders/paint_splatter/canvas.frag`

- [ ] **Step 1: Define the GlobalUbo struct** (camera matrices) in the hpp, with the size assert:
```cpp
struct GlobalUbo {
  glm::mat4 projection{1.f};
  glm::mat4 view{1.f};
  glm::mat4 inverseView{1.f};
  glm::vec4 canvasInfo{0.f};  // xy = canvas world half-extent, z = world size, w unused
};
static_assert(sizeof(GlobalUbo) == 208, "GlobalUbo std140 size");
```

- [ ] **Step 2: Create the canvas quad.** A single X-Z plane quad at y=0, side length `kCanvasWorld` (e.g. 4.0), with positions + UVs (0..1). Upload to a `vgeu::VgeuBuffer` vertex buffer (4 verts) + index buffer (6 indices). Follow the vertex-buffer creation idiom in `triangle.cpp`.

- [ ] **Step 3: Write `canvas.vert`** — transform vert by `projection*view*model`, pass UV through:
```glsl
#version 450
layout(location=0) in vec3 inPos;
layout(location=1) in vec2 inUv;
layout(set=0,binding=0) uniform GlobalUbo { mat4 projection; mat4 view; mat4 inverseView; vec4 canvasInfo; } ubo;
layout(location=0) out vec2 vUv;
void main(){ vUv = inUv; gl_Position = ubo.projection * ubo.view * vec4(inPos,1.0); }
```

- [ ] **Step 4: Write `canvas.frag`** — sample the canvas texture; for M1 the texture is a procedural UV checker so we can verify orientation before any real texture exists:
```glsl
#version 450
layout(location=0) in vec2 vUv;
layout(location=0) out vec4 outColor;
void main(){
  vec2 g = step(0.5, fract(vUv * 8.0));
  float c = abs(g.x - g.y);          // checker
  outColor = vec4(mix(vec3(0.85), vec3(0.25), c), 1.0);
}
```

- [ ] **Step 5: Pipeline + descriptor for set=0 (GlobalUbo).** Create per-frame uniform buffers for `GlobalUbo`, a descriptor set layout (one UBO at binding 0), the pipeline layout, and a graphics pipeline (depth test on, back-face cull off so the quad is visible from both sides). Reuse the pipeline-creation idiom from `triangle.cpp`/`soap_bubble.cpp:preparePipelines`.

- [ ] **Step 6: `updateGlobalUbo()`** sets projection/view/inverseView from `camera` (see `particle.cpp:1660-1668`). Wire the orbit camera in `viewChanged()` (`camera.setAspectRatio(...)`, `particle.cpp:1643`).

- [ ] **Step 7: `buildCommandBuffers()`** — begin render pass with a mid-gray clear, bind pipeline + set=0, bind canvas vertex/index buffers, `drawIndexed(6,...)`, render ImGui overlay, end pass. Model the structure on `particle.cpp:1397-1446`.

- [ ] **Step 8: Build + run.** Run: `rtk cmd /c mingwBuild.bat && ./build/paint_splatter.exe`. Expected: VL-CLEAN; a checkered quad visible.

- [ ] **Step 9: Commit.**
```bash
clang-format -i src/examples/paint_splatter/paint_splatter.cpp src/examples/paint_splatter/paint_splatter.hpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): canvas quad + 3D camera + checker shader (M1)"
```

### Task 2: ImGui panel skeleton

- [ ] **Step 1: `onUpdateUIOverlay()`** — add a collapsing header "Paint Splatter" with a couple of read-only fields (frame time, camera pos) and a placeholder "Save" button (no-op for now). Follow `soap_bubble.cpp:onUpdateUIOverlay` for the `vgeu_ui_overlay` API.

- [ ] **Step 2: Build + run.** Confirm the panel renders and the button is clickable (logs to stdout).

- [ ] **Step 3: Commit.**
```bash
clang-format -i src/examples/paint_splatter/*.cpp src/examples/paint_splatter/*.hpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): ImGui panel skeleton (M1)"
```

- [ ] **GATE (user) — M1:** Build and run. **User confirms:** the checkered canvas quad is visible, the orbit camera rotates/zooms it, and the ImGui panel renders and responds. Do not proceed until approved.

---

## Milestone 2 — Particle buffer + render + compute/graphics sync skeleton

**Goal of M2:** a static (or gravity-only) block of particles renders as disks, driven through the **full compute→graphics semaphore handshake** on separate queues. This proves the sharp-edge sync path with trivial data before any real physics. This is the highest-risk milestone for sync bugs.

### Task 3: Particle SSBO + struct layout

**Files:** Modify `paint_splatter.{hpp,cpp}`

- [ ] **Step 1: Define the `Particle` std430 struct** with explicit padding + assert:
```cpp
struct Particle {
  glm::vec4 pos;     // xyz position, w = packed color OR unused (decide M5); start: w=1
  glm::vec4 vel;     // xyz velocity, w = wetness/alpha (used M6); start: w=0
  glm::vec4 predict; // xyz predicted x*, w = lambda (used M4)
  glm::vec4 color;   // rgba paint color (a = concentration), start opaque white
};
static_assert(sizeof(Particle) == 64, "Particle std430 size");
```

- [ ] **Step 2: Allocate per-frame particle SSBOs** sized `kMaxParticles` (start 1<<16 = 65536). Use `eStorageBuffer | eVertexBuffer | eTransferDst` usage. Add a `numParticles` counter (host-side for M2; becomes a GPU counter buffer in M5).

- [ ] **Step 3: Seed a static block.** On `prepare()`, fill a CPU vector with an NxNxN lattice of particles centered above the canvas (y = 1.5), upload to all per-frame SSBOs. READBACK: print `numParticles` and particle[0].pos.

- [ ] **Step 4: Commit.**
```bash
clang-format -i src/examples/paint_splatter/*
rtk git add -A && rtk git commit -m "feat(paint_splatter): particle SSBO + static seed block (M2)"
```

### Task 4: Particle debug renderer

**Files:** Modify `paint_splatter.{hpp,cpp}`; Create `shaders/paint_splatter/particle.vert`, `particle.frag`

- [ ] **Step 1: `particle.vert`** — read `pos.xyz` as a point, set `gl_PointSize` from distance, pass color:
```glsl
#version 450
layout(location=0) in vec4 inPos;     // from Particle.pos
layout(location=1) in vec4 inColor;   // from Particle.color
layout(set=0,binding=0) uniform GlobalUbo { mat4 projection; mat4 view; mat4 inverseView; vec4 canvasInfo; } ubo;
layout(location=0) out vec4 vColor;
void main(){
  vec4 clip = ubo.projection * ubo.view * vec4(inPos.xyz,1.0);
  gl_Position = clip;
  gl_PointSize = clamp(16.0 / max(clip.w,0.001), 2.0, 32.0);
  vColor = inColor;
}
```

- [ ] **Step 2: `particle.frag`** — round disk (discard outside radius), output color (copy the disk discard from `18-flip.html` pointFragmentShader logic).

- [ ] **Step 3: Vertex input for the particle pipeline.** Binding stride = `sizeof(Particle)` (64), attribute 0 = `pos` at `offsetof(Particle,pos)`, attribute 1 = `color` at `offsetof(Particle,color)`. Topology = `ePointList`. Enable `VkPhysicalDeviceFeatures::largePoints`/`shaderClipDistance` only if needed; confirm in `getEnabledFeatures()`.

- [ ] **Step 4: Draw particles** in `buildCommandBuffers()` after the canvas: bind particle pipeline, bind the current-frame SSBO as vertex buffer, `draw(numParticles,1,0,0)`. Add an ImGui checkbox `showParticles`.

- [ ] **Step 5: Build + run.** Expected: VL-CLEAN; a static block of colored disks floats above the canvas.

- [ ] **Step 6: Commit.**
```bash
clang-format -i src/examples/paint_splatter/*
rtk git add -A && rtk git commit -m "feat(paint_splatter): particle disk debug renderer (M2)"
```

### Task 5: Compute queue, compute UBO, and the trivial predict pass

**Files:** Modify `paint_splatter.{hpp,cpp}`; Create `shaders/paint_splatter/pbf_predict.comp`

- [ ] **Step 1: Add the `compute` struct block** (queue, cmdPool, cmdBuffers, semaphores, uniformBuffers, storageBuffers refs, descriptorSetLayout, descriptorSets, pipelineLayout, pipelines) mirroring `particle.hpp:160-206`. Add a `graphics.semaphores` vector too (`particle.hpp:157`).

- [ ] **Step 2: Define `ComputeUbo` std140** with assert:
```cpp
struct ComputeUbo {
  float dt; uint32_t particleCount; float gravity; float _pad0;
  glm::vec4 canvasMin;  // xyz world-min of fluid domain, w = cell size h
  glm::vec4 canvasMax;  // xyz world-max, w unused
};
static_assert(sizeof(ComputeUbo) == 48, "ComputeUbo std140 size");
```

- [ ] **Step 3: `prepareCompute()`** — get compute queue + family index, create command pool/buffers (one per frame), create the compute semaphores, and **signal the graphics semaphores once at init** so frame 0's compute submit doesn't deadlock waiting on a never-signaled semaphore. Copy this init-signal idiom from `particle.cpp:prepareCompute` (around `particle.cpp:117-118` shows the one-shot signal submit pattern).

- [ ] **Step 4: `pbf_predict.comp`** — trivial for M2: apply gravity to vel, write `predict = pos + dt*vel` (no integration of pos yet, so the block stays put unless `gravity>0`):
```glsl
#version 450
layout(local_size_x = 256) in;
struct Particle { vec4 pos; vec4 vel; vec4 predict; vec4 color; };
layout(std430, set=0, binding=0) buffer Particles { Particle p[]; };
layout(std140, set=0, binding=1) uniform Ubo { float dt; uint count; float gravity; float _p0; vec4 cmin; vec4 cmax; } u;
void main(){
  uint i = gl_GlobalInvocationID.x;
  if (i >= u.count) return;
  p[i].vel.xyz += vec3(0.0, -u.gravity, 0.0) * u.dt;
  p[i].predict.xyz = p[i].pos.xyz + p[i].vel.xyz * u.dt;
}
```

- [ ] **Step 5: Create the compute descriptor set** (binding 0 = particle SSBO, binding 1 = ComputeUbo) and the `pbf_predict` pipeline.

- [ ] **Step 6: `buildComputeCommandBuffers()`** — begin; **graphics→compute acquire barrier** if `compute.queueFamilyIndex != graphics.queueFamilyIndex` (copy `particle.cpp:1513-1532`); bind predict pipeline + descriptor set; `dispatch(ceil(numParticles/256),1,1)`; **compute→graphics release barrier** (copy `particle.cpp:1621-1639`); end. For M2 we do NOT yet copy predict→pos (block holds), so set `gravity=0` default to keep it static; a `gravity` ImGui slider lets you confirm the pass runs by watching it fall once M3 integrates pos.

- [ ] **Step 7: Wire the submit handshake in `draw()`** EXACTLY as `particle.cpp:1356-1394`:
  - compute submit: wait `graphics.semaphores[frame]` @ `eComputeShader`, signal `compute.semaphores[frame]`.
  - graphics submit: wait `{compute.semaphores[frame], presentCompleteSemaphores[frame]}` @ `{eVertexInput, eColorAttachmentOutput}`, signal `{graphics.semaphores[frame], renderCompleteSemaphores[frame]}`.
  - add the **graphics acquire barrier** for the SSBO in `buildCommandBuffers()` (copy `particle.cpp:1413-1431`).

- [ ] **Step 8: Build + run.** Expected: VL-CLEAN with separate compute/graphics queues; block still renders (static at gravity=0). Toggle nothing yet. The critical pass criterion is **zero validation/sync errors** across many frames.

- [ ] **Step 9: Commit.**
```bash
clang-format -i src/examples/paint_splatter/*
rtk git add -A && rtk git commit -m "feat(paint_splatter): compute queue + predict pass + cross-queue sync handshake (M2)"
```

- [ ] **GATE (user) — M2:** Build and run. **User confirms:** particle block renders stably for many seconds with **no flicker, no validation errors, no device-lost**, on separate compute/graphics queues. (Sync correctness is the whole point of this gate.) Do not proceed until approved.

---

## Milestone 3 — Integrate + canvas/wall collision (no density solve)

**Goal of M3:** particles fall under gravity, collide with the canvas floor (y=0) and domain walls, and pile up without leaking. Still no incompressibility, so they'll stack into a thin pancake — that's expected and correct for this stage.

### Task 6: Position integration + collision

**Files:** Modify `pbf_predict.comp`; Create `shaders/paint_splatter/pbf_finalize.comp` (minimal); Modify `paint_splatter.cpp`

- [ ] **Step 1: Promote predict→pos with collision.** For M3 (no constraint solve), use a single pass: predict, clamp `predict` to the domain box `[cmin.xyz, cmax.xyz]` with floor at y=0, then write back. Replace the body of `pbf_predict.comp`:
```glsl
  uint i = gl_GlobalInvocationID.x;
  if (i >= u.count) return;
  vec3 v = p[i].vel.xyz + vec3(0.0,-u.gravity,0.0)*u.dt;
  vec3 x = p[i].pos.xyz + v*u.dt;
  // domain + canvas-floor collision (solid half-space y>=0)
  vec3 lo = u.cmin.xyz, hi = u.cmax.xyz;
  if (x.y < lo.y) { x.y = lo.y; v.y = 0.0; v.xz *= 0.98; } // floor friction
  if (x.x < lo.x){x.x=lo.x; v.x=0.0;} if (x.x>hi.x){x.x=hi.x; v.x=0.0;}
  if (x.z < lo.z){x.z=lo.z; v.z=0.0;} if (x.z>hi.z){x.z=hi.z; v.z=0.0;}
  p[i].vel.xyz = v; p[i].pos.xyz = x;
```

- [ ] **Step 2: Set domain.** In `updateComputeUbo()` set `cmin = (-half, 0, -half)`, `cmax = (half, worldHeight, half)`, `h = particleSpacing` (used in M4). Default `gravity` to e.g. 9.8 (world units/s²); expose `dt` (e.g. frameTimer or fixed 1/120 with substeps) and `gravity` as ImGui sliders.

- [ ] **Step 3: Build + run.** Expected: the block falls, hits y=0, spreads into a flat layer; nothing escapes the box. READBACK: print min/max particle y each second; min should clamp at 0.

- [ ] **Step 4: Commit.**
```bash
clang-format -i src/examples/paint_splatter/*
rtk git add -A && rtk git commit -m "feat(paint_splatter): gravity integration + box/floor collision (M3)"
```

- [ ] **GATE (user) — M3:** **User confirms:** particles fall, collide with the canvas floor and walls, and stay inside the domain (no leaking/explosion). Do not proceed until approved.

---

## Milestone 4 — Neighbor grid + PBF density solve + surface tension

**Goal of M4:** a dropped block behaves like a cohesive, ~incompressible fluid (mini dam-break): it splashes, stays cohesive (surface tension), and does not explode or vanish. This is the scientific core.

### Task 7: Uniform-grid neighbor search (GPU counting sort)

**Files:** Create `shaders/paint_splatter/grid_count.comp`, `grid_scan.comp`, `grid_scatter.comp`; Modify `paint_splatter.{hpp,cpp}`

Background: this is the GPU version of the partial-sum spatial hash in `src/examples/pbd` (`SpatialHash`: `addPos`→count, `createPartialSum`→prefix sum, fill→scatter) and FLIP's `pushParticlesApart` (`18-flip.html:152-191`). Cell size `h` = PBF smoothing radius. Grid covers the domain box.

- [ ] **Step 1: Allocate grid buffers** (per-frame or shared, double-buffer not required since rebuilt each substep): `cellCount[numCells]` (uint), `cellStart[numCells+1]` (uint, prefix sums), `sortedParticleIds[kMaxParticles]` (uint). `numCells = gridDimX*gridDimY*gridDimZ` from `ceil(domain/h)`.

- [ ] **Step 2: `grid_count.comp`** — clear then, per particle, compute cell index from `predict.xyz`, `atomicAdd(cellCount[cell],1)`. (Two dispatches: a clear pass over cells, then the count pass over particles. Or `vkCmdFillBuffer` to zero `cellCount`.)
```glsl
// per particle:
uint i = gid; if (i>=count) return;
ivec3 c = ivec3(floor((p[i].predict.xyz - cmin.xyz)/h));
c = clamp(c, ivec3(0), gridDim-1);
uint cell = (c.z*gridDim.y + c.y)*gridDim.x + c.x;
atomicAdd(cellCount[cell], 1u);
```

- [ ] **Step 3: `grid_scan.comp`** — exclusive prefix sum of `cellCount` → `cellStart`. For the first implementation use a **single-workgroup serial scan** if `numCells` is small enough, otherwise a standard Blelloch scan. SIMPLEST correct option for v1: do the prefix sum on the CPU after a readback is NOT allowed (breaks async); instead implement a naive multi-pass scan or a single-invocation loop guarded by `numCells` size. Document the chosen scan and its `numCells` ceiling in a comment.

- [ ] **Step 4: `grid_scatter.comp`** — per particle, recompute cell, `idx = atomicAdd(cellOffset[cell],1)` (a copy of `cellStart`), write `sortedParticleIds[idx] = i`. Mirrors `18-flip.html:182-191`.

- [ ] **Step 5: Dispatch order in compute cmd buffer:** fill(cellCount=0) → grid_count → grid_scan → grid_scatter, with a `eComputeShader→eComputeShader` buffer barrier between each (copy the in-compute barrier from `particle.cpp:1599-1608`).

- [ ] **Step 6: READBACK sanity:** sum of `cellCount` == `numParticles`; `cellStart[numCells]` == numParticles. Print and assert.

- [ ] **Step 7: Commit.**
```bash
clang-format -i src/examples/paint_splatter/*
rtk git add -A && rtk git commit -m "feat(paint_splatter): GPU uniform-grid neighbor search (M4)"
```

### Task 8: PBF density constraint (λ, Δp, scorr) + collision in solver

**Files:** Create `shaders/paint_splatter/pbf_lambda.comp`, `pbf_delta.comp`, `pbf_apply.comp`, `pbf_finalize.comp`; Modify `paint_splatter.cpp`

Constants (paper §3–5): `h` (smoothing radius), `rho0` (rest density), `epsCFM` (Eq. 11 relaxation), `scorrK=0.1`, `scorrDq=0.2*h`, `scorrN=4`, XSPH `c=0.01`, optional vorticity `eps`. Poly6 & Spiky kernels (3D). Add these to `ComputeUbo` (extend struct + bump the static_assert size; keep 16-byte alignment + pads).

- [ ] **Step 1: Kernel helpers (shared GLSL include or pasted per shader).** 3D Poly6 and Spiky-gradient:
```glsl
// W_poly6(r) = 315/(64 pi h^9) (h^2-r^2)^3, r<=h
// grad W_spiky(r) = -45/(pi h^6) (h-r)^2 * (rvec/r), r<=h
float poly6(float r2, float h){ if(r2>=h*h) return 0.0; float t=h*h-r2; return KPOLY6 * t*t*t; }
vec3 spikyGrad(vec3 rij, float r, float h){ if(r>=h||r<=1e-6) return vec3(0); float t=h-r; return KSPIKY*t*t*(rij/r); }
```
Precompute `KPOLY6 = 315/(64*pi*h^9)`, `KSPIKY = -45/(pi*h^6)` on host and pass in UBO (avoids per-thread pow).

- [ ] **Step 2: `pbf_lambda.comp`** — for each particle, loop over the 27 neighbor cells using `cellStart/sortedParticleIds`, accumulate density `rho` (Poly6) and `sumGradC2`; compute `C = rho/rho0 - 1`; `lambda = -C / (sumGradC2 + epsCFM)` (Eq. 11). Store in `p[i].predict.w`.

- [ ] **Step 3: `pbf_delta.comp`** — for each particle, loop neighbors, accumulate `dp = (1/rho0) * sum_j (lambda_i + lambda_j + scorr) * spikyGrad`, where `scorr = -scorrK * (poly6(|rij|^2)/poly6(scorrDq^2))^scorrN` (Eq. 13–14). Then apply collision response against the domain box + canvas floor to the candidate `predict+dp` (clamp into box, as in M3). Write the resulting delta to a scratch buffer (or directly to a separate `deltaP` field — add a 5th vec4 to Particle OR use a parallel SSBO `vec4 deltaP[]`; choose the parallel SSBO to keep Particle at 64B). 

- [ ] **Step 4: `pbf_apply.comp`** — `p[i].predict.xyz += deltaP[i].xyz`. Separate pass so all reads in `pbf_delta` see the same iteration's positions (Jacobi, per paper §6).

- [ ] **Step 5: `pbf_finalize.comp`** — `v = (predict - pos)/dt`; apply XSPH viscosity (Eq. 17, neighbor loop) and optional vorticity confinement (Eq. 15–16); `pos = predict`. Color diffusion among neighbors (copy the averaging from `18-flip.html:239-245`) so mixed droplets blend.

- [ ] **Step 6: Solver loop in compute cmd buffer.** Per substep: predict → build grid (Task 7) → repeat `solverIters` (2–4) of {lambda → delta → apply} with compute barriers between → finalize. Expose `substeps` and `solverIters` as ImGui sliders. Mirror the multi-pass barrier structure of `particle.cpp:1568-1619`.

- [ ] **Step 7: Tune defaults** so a dropped block settles to ~rest density: pick `h`, `rho0`, particle spacing consistent with each other (spacing ≈ 0.5h..0.6h; seed the block at that spacing). READBACK: print mean density vs rho0; should be ≈1.0 after settling, not diverging.

- [ ] **Step 8: Density debug color (optional toggle).** Add an ImGui toggle that colors particles by `rho/rho0` (blue→red), reusing `setSciColor` logic (`18-flip.html:601-621`) to visually confirm incompressibility (like PBF Fig. 2 / FLIP grid view).

- [ ] **Step 9: Build + run.** Expected: dropped block splashes and settles cohesively; no explosion (particles flying to infinity) or implosion (NaN/vanish). VL-CLEAN.

- [ ] **Step 10: Commit.**
```bash
clang-format -i src/examples/paint_splatter/*
rtk git add -A && rtk git commit -m "feat(paint_splatter): PBF density solve + surface tension + viscosity (M4)"
```

- [ ] **GATE (user) — M4:** **User confirms:** a dropped block of fluid behaves like cohesive incompressible water (mini dam-break) — splashes, stays cohesive, settles, no explosion/vanish. Density debug color stays near rest. Do not proceed until approved.

---

## Milestone 5 — Emitter + keyboard spoid control (discrete droplets)

**Goal of M5:** pressing a key releases a droplet from each selected spoid; it falls, splashes, and produces satellite droplets. Spoids are selectable (ImGui) and movable (keyboard).

### Task 9: GPU particle counter + emit pass

**Files:** Create `shaders/paint_splatter/emit.comp`; Modify `paint_splatter.{hpp,cpp}`

- [ ] **Step 1: Convert `numParticles` to a GPU counter buffer** (`uint liveCount`) so compute can append. Add an `EmitRequest` SSBO/UBO array: per request `{vec4 originRadius (xyz=origin, w=holeRadius); vec4 velColor0; vec4 color; uint countToSpawn; ...}` with size assert.

- [ ] **Step 2: `emit.comp`** — one invocation per (request, slot): sample a position inside the hole disk (use a hash/RNG on gid for jitter), set initial downward velocity from `emissionVelocity`, set color/concentration, `atomicAdd(liveCount,1)` to get the destination index (clamp to kMaxParticles; reject if full). Append into the particle SSBO.

- [ ] **Step 3: Dispatch emit first** in the compute cmd buffer (before predict), guarded by a host-set `numEmitThisFrame`. Barrier emit→predict.

- [ ] **Step 4: Build + run** with a hardcoded single-spoid auto-emit every N frames. Expected: droplets spawn, fall, splash with satellites (PBF cohesion). READBACK: liveCount grows then (after M6) bounds.

- [ ] **Step 5: Commit.**
```bash
clang-format -i src/examples/paint_splatter/*
rtk git add -A && rtk git commit -m "feat(paint_splatter): GPU emit pass + particle counter (M5)"
```

### Task 10: Spoid model + SpoidController + keyboard control

**Files:** Modify `paint_splatter.{hpp,cpp}`

- [ ] **Step 1: Define `Spoid` (host POD)** and the `SpoidController` interface:
```cpp
struct Spoid {
  glm::vec3 pos{0,2,0};
  float holeRadius=0.1f, mass=1.f, emissionVelocity=2.f, pressure=1.f;
  int amount=400;                 // particles per drop
  glm::vec3 color{0.2,0.4,0.9};
  float concentration=1.f;        // -> stamp alpha + XSPH/scorr weight + dry rate
  bool selected=false;
};
struct SpoidController {
  virtual ~SpoidController()=default;
  virtual void update(float dt, std::vector<Spoid>& spoids,
                      const struct InputState& in, std::vector<int>& emitDrops)=0;
};
```

- [ ] **Step 2: `KeyboardSpoidController`** — arrow/WASD move all `selected` spoids in canvas XY (+QE for height); Space pushes each selected spoid's index into `emitDrops` (one burst of `amount`). Read input via the project's input path (check how `vgeu_keyboard_movement_controller` / GLFW keys are read in `particle.cpp`/`cloth.cpp`).

- [ ] **Step 3: Build EmitRequests** from `emitDrops` each frame; set `numEmitThisFrame`. Derive initial velocity `= emissionVelocity` downward (leave `mass`/`pressure` as independent inputs for now; comment the Phase-2 coupling point `emissionVelocity=f(mass,ω)`).

- [ ] **Step 4: ImGui spoid panel** — list spoids with a `selected` checkbox each; sliders for the selected spoid's params; a "+ Add spoid" / "− Remove" control; render a small marker (reuse particle pipeline or a debug line) at each spoid position so you can see them.

- [ ] **Step 5: Build + run.** Expected: select spoids via checkboxes, move them with keys, Space drops a droplet from each selected one; droplets splash. VL-CLEAN.

- [ ] **Step 6: Commit.**
```bash
clang-format -i src/examples/paint_splatter/*
rtk git add -A && rtk git commit -m "feat(paint_splatter): Spoid model + KeyboardSpoidController + ImGui select (M5)"
```

- [ ] **GATE (user) — M5:** **User confirms:** ImGui checkboxes select spoids; keyboard moves only the selected ones; Space releases a droplet from each selected spoid that falls and splashes with visible satellite droplets. Do not proceed until approved.

---

## Milestone 6 — Canvas deposition (stamp + dry) + compaction

**Goal of M6:** droplets touching the canvas deposit permanent color into the accumulation texture (alpha-over), then dry out and are removed, keeping particle count bounded.

### Task 11: Canvas accumulation texture + deposit pass

**Files:** Create `shaders/paint_splatter/deposit.comp`; Modify `canvas.frag`, `paint_splatter.{hpp,cpp}`

- [ ] **Step 1: Create the canvas storage image** — RGBA8 (or R32_UINT packed for atomic blending — see Step 3), default 2048², usage `eStorage | eSampled | eTransferSrc`, initial layout cleared to transparent/white. Make it a `vgeu::VgeuImage`. Bind as `sampler2D` to `canvas.frag` (replace the checker) and as `image2D`/`uimage2D` to `deposit.comp`.

- [ ] **Step 2: World→UV mapping.** A particle at world `(x,y,z)` maps to canvas UV `((x-cmin.x)/size, (z-cmin.z)/size)`. Deposit only when `pos.y < depositHeight` (≈ particleRadius) and moving downward/at rest.

- [ ] **Step 3: `deposit.comp`** — for each particle within deposit height: compute texel, **alpha-over blend** its `color` with stamp alpha `= concentration * depositStrength`. Concurrency: multiple particles may hit one texel in a frame. Use **`R32_UINT` + `imageAtomicMax`/packed accumulation OR a per-texel spinlock-free additive scheme**. SIMPLEST correct v1: pack premultiplied RGBA into a `uint`, use `imageAtomicAdd` on four 8-bit-ish fixed-point channels with saturation handled on read, OR accept minor races with a plain `imageStore` (visually fine for a demo) and document the tradeoff. Pick `imageStore` alpha-over for v1; note atomic upgrade as a follow-up.

- [ ] **Step 4: Drying.** After stamping, decrement `p[i].vel.w` (wetness, init 1.0 at emit) by `dryRate * dt * (1/concentration)`. When `vel.w <= 0`, mark the particle consumed (e.g. set `pos.w = -1` sentinel).

- [ ] **Step 5: Dispatch deposit** after finalize, barrier finalize→deposit (SSBO) and an image barrier so the graphics sample sees the writes (`eComputeShader→eFragmentShader`, image layout `General→ShaderReadOnlyOptimal` or keep `General` and sample). Confirm layout handling is VL-CLEAN.

- [ ] **Step 6: Build + run.** Expected: a dropped droplet leaves a permanent colored mark of roughly its splash radius; the canvas accumulates marks. READBACK: confirm consumed particles get sentinel.

- [ ] **Step 7: Commit.**
```bash
clang-format -i src/examples/paint_splatter/*
rtk git add -A && rtk git commit -m "feat(paint_splatter): canvas accumulation texture + deposit/dry pass (M6)"
```

### Task 12: Particle compaction (remove consumed)

**Files:** Create `shaders/paint_splatter/compact.comp`; Modify `paint_splatter.cpp`

- [ ] **Step 1: `compact.comp`** — stream-compaction: copy live particles (`pos.w >= 0`) into a second SSBO using an `atomicAdd` on a new `liveCount`, then swap buffers (ping-pong the particle SSBO). Simpler v1: a single-pass scatter with atomic compaction into a back buffer; swap pointers each frame. Document buffer ping-pong and ensure the renderer + all compute passes read the current front buffer.

- [ ] **Step 2: Reset `liveCount`** appropriately each frame (the compaction pass writes the new count). Update `numParticles`-equivalent host mirror via a small readback for ImGui display only (not for control flow).

- [ ] **Step 3: Build + run** with continuous auto-drop. Expected: particle count rises during a splash and falls as paint dries — **count stays bounded** over a long run; canvas keeps accumulating. VL-CLEAN.

- [ ] **Step 4: Commit.**
```bash
clang-format -i src/examples/paint_splatter/*
rtk git add -A && rtk git commit -m "feat(paint_splatter): GPU particle compaction (M6)"
```

- [ ] **GATE (user) — M6:** **User confirms:** droplets leave permanent marks on the canvas (alpha-over accumulation), paint dries (particles get absorbed), and live particle count stays bounded over a sustained painting session. Do not proceed until approved.

---

## Milestone 7 — ImGui params polish + PNG save

**Goal of M7:** full per-spoid parameter editing through ImGui and a working "Save" button that writes the canvas to PNG matching the on-screen image.

### Task 13: Full ImGui parameter surface + CLI flags

**Files:** Modify `paint_splatter.{hpp,cpp}`

- [ ] **Step 1: Expose all `Spoid` params** (holeRadius, mass, emissionVelocity, pressure, amount, color picker, concentration) for the selected spoid(s), plus global sim params (gravity, substeps, solverIters, h, rho0, dryRate, depositStrength). Add `numSpoids` add/remove.

- [ ] **Step 2: CLI flags** in `setupCommandLineParser` for the common knobs (kMaxParticles, gravity, solverIters, canvas resolution) following `particle.cpp:setupCommandLineParser`.

- [ ] **Step 3: Build + run.** Expected: editing a selected spoid's params changes its next droplets (bigger holeRadius → bigger drop, higher concentration → more opaque/slower-drying mark).

- [ ] **Step 4: Commit.**
```bash
clang-format -i src/examples/paint_splatter/*
rtk git add -A && rtk git commit -m "feat(paint_splatter): full ImGui spoid params + CLI flags (M7)"
```

### Task 14: PNG save (GPU→CPU readback)

**Files:** Modify `paint_splatter.{hpp,cpp}`

- [ ] **Step 1: `TextureSaver` logic** — on the ImGui "Save" button: `device.waitIdle()` (simplest correct), copy the canvas image to a host-visible staging buffer (`vkCmdCopyImageToBuffer` with the right layout transition), map it, and write PNG via `stb_image_write` (`stbi_write_png`, already vendored at `external/tinygltf/stb_image_write.h`). Reuse the offscreen-capture/readback pattern from `soap_bubble.cpp` (search its capture path).

- [ ] **Step 2: Filename + error handling** — write to `build/paint_<timestamp>.png`; on IO failure show an ImGui error string; simulation continues (no crash).

- [ ] **Step 3: Build + run.** Expected: clicking Save writes a PNG; open it and confirm it matches the on-screen canvas (same marks, orientation, colors). Handle the sRGB/format conversion so saved colors match.

- [ ] **Step 4: Commit.**
```bash
clang-format -i src/examples/paint_splatter/*
rtk git add -A && rtk git commit -m "feat(paint_splatter): ImGui Save button -> canvas PNG export (M7)"
```

### Task 15: Docs

**Files:** Create `docs/09_paint_splatter.md`

- [ ] **Step 1: Write the example doc** in the style of `docs/04_particle.md` / `docs/05_pbd.md`: overview, PBF concept, compute pipeline table, shader table, sync structure, options, references (PBF paper, FLIP). Add it to `docs/00_overview.md` if that file lists examples.

- [ ] **Step 2: Commit.**
```bash
rtk git add docs/ && rtk git commit -m "docs: add paint_splatter example doc"
```

- [ ] **GATE (user) — M7:** **User confirms:** per-spoid params edit live; the Save button exports a PNG that matches the on-screen canvas. This completes Phase 1. Do not proceed to Phase 2 (continuous stream, pendulum, top-down view) — that is a separate spec.

---

## Self-Review Notes (coverage vs spec)

- Spec §2 PBF loop → M2 (predict+sync), M4 (grid, lambda, delta, apply, finalize). ✔
- Spec §2 spoid→PBF param mapping → M5 Task 10 Step 1 + M7 Task 13. ✔
- Spec §3 deposition (method B, alpha-over, drying, consume) → M6 Tasks 11–12. ✔
- Spec §4 SpoidController abstraction (Keyboard now, Pendulum later) → M5 Task 10. ✔
- Spec §5 camera + Save button → M1 Task 1 + M7 Task 14. ✔
- Spec §6 phasing (Phase 1 only) → M7 final gate stops before Phase 2. ✔
- Spec §7 verification → Verification Model section + per-milestone GATEs/READBACKs. ✔
- Spec §8 error handling (pool exhausted, asserts, save failure) → M5 Task 9 Step 2, struct asserts throughout, M7 Task 14 Step 2. ✔
- Spec §9 user-gated milestones M1–M7 → one milestone section each, each ending in GATE (user). ✔

**Known v1 simplifications flagged for the executor (decide-in-task, not placeholders):** prefix-scan strategy (Task 7 Step 3), deposit atomic-vs-imageStore blend (Task 11 Step 3), compaction ping-pong (Task 12 Step 1). Each task states a concrete v1 choice and the upgrade path.
