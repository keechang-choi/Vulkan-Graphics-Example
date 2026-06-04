# Paint Splatter Phase 2 — PBD Pendulum-Driven Spoids + Top View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `PendulumSpoidController` that drives spoids along the motion of an n-link PBD pendulum (double pendulum default), with per-spoid rotating offset, a paint reservoir, chain visualization, and a button that smoothly animates the camera to a top-down view.

**Architecture:** Entirely host-side except two trivial line shaders. The pendulum is a CPU controller that each frame writes only `spoid.pos`; the existing PBF/emit/stream/deposit/PNG paths reuse that value unchanged. No compute pass, UBO, or GPU-struct change. The example owns `std::vector<PendulumChain>`; the controller mutates it and positions spoids. Visualization reuses the marker pipeline for joints and adds one `eLineList` pipeline for links.

**Tech Stack:** C++17, Vulkan-Hpp RAII, GLSL (two trivial new shaders), ImGui (`vgeu_ui_overlay`), the existing `paint_splatter` example.

**Design spec:** `docs/superpowers/specs/2026-06-04-paint-splatter-phase2-pendulum-design.md`

---

## Verification Model (read first)

This repo has **no unit-test framework**; the example is verified by running it. Verification primitives (same as Phase 1):

- **VL-CLEAN**: run the debug build (validation layers on) and confirm **zero validation errors/warnings** in stdout.
- **READBACK**: the existing `consumeParticleReadback()` numeric line, extended here with a pendulum tip print to confirm the swing is bounded.
- **GATE (user)**: a STOP. The user visually confirms the on-screen behavior. Do not pass a GATE without user approval.

## Conventions

- **Build:** `rtk cmd /c mingwBuild.bat` from repo root (shaders compile as part of the build). Run the binary from `build/`: `./build/paint_splatter.exe`. **Run build and run as SEPARATE commands** (chaining build+run in one PowerShell call has broken before).
- **New shader files need a CMake reconfigure** the first time (GLOB_RECURSE at `CMakeLists.txt:97` only re-scans on configure). If `chain_line.vert.spv`/`chain_line.frag.spv` are not produced, delete `build/CMakeCache.txt`-adjacent stamp or re-run the configure step of `mingwBuild.bat`; confirm the two `.spv` files appear under `shaders/paint_splatter/`.
- **clang-format:** run `clang-format -i` on every edited `.cpp/.hpp` before each commit (user rule). Shaders are not clang-formatted.
- **rtk:** prefix shell commands with `rtk`.
- **World convention (locked):** screen-up = world −Y; gravity +Y (down on screen); floor at y=0; fluid/spoids at y<0; "higher up" = more negative y. Pivot near ceiling (y≈−2.8); chain hangs toward +Y.
- All paths are relative to repo root `C:\Users\rlckd\Desktop\kc\Vulkan-Graphics-Example`.

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/examples/paint_splatter/paint_splatter.hpp` | Modify | `PendulumNode`/`PendulumChain` structs, `Spoid` Phase-2 fields, `PendulumSpoidController`, new example members |
| `src/examples/paint_splatter/paint_splatter.cpp` | Modify | controller wiring, PBD step, rotary geometry, reservoir gating, line/joint draw, camera animation, UI |
| `shaders/paint_splatter/chain_line.vert` | Create | transform a node position by `projection*view`, pass color |
| `shaders/paint_splatter/chain_line.frag` | Create | output flat color |
| `docs/09_paint_splatter.md` | Modify | add a Phase 2 section |

---

## Task 1 — Data model: pendulum structs + Spoid Phase-2 fields

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.hpp` (Spoid at lines 114-128; SpoidController/KeyboardSpoidController at 137-163; example members near 374-376)

- [ ] **Step 1: Append Phase-2 fields to `Spoid`.** In `paint_splatter.hpp`, after `float emitAccum = 0.f;` (line 127, the last field before the closing `};` of `struct Spoid`), add:

```cpp
  // --- Phase 2: pendulum attachment (used only by PendulumSpoidController) ---
  int nodeIndex = -1;       // attach node in the chain; -1 = tip (last node)
  float offsetR = 0.f;      // offset distance in the plane perpendicular to the
                            // link direction at the attach node (0 = on node)
  float offsetAngle0 = 0.f;  // start angle in that perpendicular plane (rad)
  float offsetOmega = 0.f;   // constant spin rate of the offset (rad/s)
  float offsetPhase = 0.f;   // runtime: angle0 + omega*t, carried across frames
  float paintMass = 1.f;     // paint reservoir; decreases on emit, 0 => stop
```

- [ ] **Step 2: Add the pendulum structs.** In `paint_splatter.hpp`, immediately AFTER the `KeyboardSpoidController` class (after its closing `};` at line 163) and BEFORE `struct Options` (line 167), add:

```cpp
// --- Phase 2: PBD pendulum -------------------------------------------------
// One point mass in a pendulum chain. node[0] is the fixed pivot (invMass 0).
struct PendulumNode {
  glm::vec3 pos{0.f};
  glm::vec3 prevPos{0.f};  // PBD: position at the start of the substep
  glm::vec3 vel{0.f};
  float invMass = 1.f;     // 1/m; pivot = 0 (immovable)
};

// One suspended pendulum assembly (n-link chain). The example owns a vector of
// these; v1 uses a single chain. CONVENTION: bobMass[i] <= 0 => infinite mass
// => invMass 0 => node i+1 is PINNED (immovable).
struct PendulumChain {
  glm::vec3 pivot{0.f, -2.8f, 0.f};  // fixed suspension point (near ceiling)
  std::vector<PendulumNode> nodes;   // [0]=pivot, [1..numLinks]=bobs
  std::vector<float> linkLength;     // rest |node_i - node_{i-1}|, i=1..numLinks
  // config (UI-editable):
  int numLinks = 2;          // n: 1 = simple, 2 = double pendulum (default)
  float totalLength = 1.5f;  // sum of links (split uniformly into n)
  std::vector<float> bobMass;  // per bob; built on reset, default uniform 1.0
  float airDamping = 0.1f;     // global velocity damping /s (air resistance)
  float jointDamping = 0.05f;  // damping of along-link relative velocity
  int substeps = 8;            // XPBD small-steps
  int iters = 4;               // distance-constraint Gauss-Seidel iters/substep
  // initial state (chain starts as a straight line displaced from +Y vertical):
  float initTheta = 0.6f;  // rad from the +Y (down) axis
  float initPhi = 0.f;     // rad azimuth in X-Z
  float initSpeed = 0.f;   // initial tangential speed at the tip (world u/s)
};
```

- [ ] **Step 3: Forward-declare the controller.** In `paint_splatter.hpp`, change the `SpoidController` comment block (lines 137-138) is fine as-is; the new controller is defined in the .cpp is NOT possible (it has state used by the example). Instead define `PendulumSpoidController` in the .hpp in Task 2. For now, no further hpp change in this step.

- [ ] **Step 4: Add example members.** In `paint_splatter.hpp`, in the private section after `std::unique_ptr<SpoidController> spoidController;` (line 375), add:

```cpp
  // --- Phase 2: pendulum -----------------------------------------------------
  enum class SpoidControlMode { Keyboard, Pendulum };
  SpoidControlMode spoidControlMode = SpoidControlMode::Keyboard;
  std::vector<PendulumChain> pendulumChains;  // owned here; controller mutates
  static constexpr uint32_t kMaxChainNodes = 32;  // pivot + up to 31 bobs
  bool showChain = true;  // draw link lines + joint sprites
  // Per-frame host-visible buffers for chain visualization (Particle stride, so
  // they reuse the marker/line pipelines' vertex input). jointMarkerBuffers: one
  // point per node. lineBuffers: two points per link (eLineList).
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> jointMarkerBuffers;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> lineBuffers;
  vk::raii::Pipeline linePipeline = nullptr;  // eLineList, chain_line shaders
  void resetPendulum();      // (re)build pendulumChains[0] from config + init
  void createChainBuffers();
  void createLinePipeline();
  // --- Phase 2: top-view camera animation ---
  struct CameraAnim {
    bool active = false;
    bool locked = false;  // hold the top view after the animation completes
    float t = 0.f, duration = 1.0f;
    glm::vec3 fromEye{0.f}, fromTarget{0.f};
    glm::vec3 toEye{0.f, -6.f, 0.f}, toTarget{0.f}, up{0.f, 0.f, -1.f};
  } cameraAnim;
  void startTopViewAnim();
  void updateCameraAnim();
```

- [ ] **Step 5: Build.** Run: `rtk cmd /c mingwBuild.bat`
Expected: compiles clean (no new behavior yet; the structs/members are unused). `static_assert(sizeof(...))` lines are untouched (Spoid is host-only).

- [ ] **Step 6: Commit.**

```bash
clang-format -i src/examples/paint_splatter/paint_splatter.hpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): Phase 2 pendulum data model + members"
```

---

## Task 2 — PendulumSpoidController + mode switch (static chain, no PBD yet)

**Goal:** add the controller (positioning each pendulum spoid at the static tip), a `resetPendulum()` that builds a straight static chain from config, and a UI mode toggle. No motion yet — this proves the wiring and that a spoid follows the tip.

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.hpp` (define `PendulumSpoidController`)
- Modify: `src/examples/paint_splatter/paint_splatter.cpp` (`resetPendulum`, `updateSpoids` exemption, UI toggle)

- [ ] **Step 1: Define `PendulumSpoidController`.** In `paint_splatter.hpp`, after the `PendulumChain` struct (added in Task 1 Step 2), add:

```cpp
// Drives spoids along an n-link PBD pendulum. The example owns the chains and a
// gravity value; this controller holds references to them. update() steps the
// PBD chains, advances each spoid's offset phase, and writes spoid.pos = the
// emission point (attach node + rotating perpendicular offset).
class PendulumSpoidController : public SpoidController {
 public:
  PendulumSpoidController(std::vector<PendulumChain>& chains,
                          const float& gravity)
      : chains(chains), gravity(gravity) {}
  std::vector<PendulumChain>& chains;
  const float& gravity;
  void update(float dt, std::vector<Spoid>& spoids, const InputState& in,
              std::vector<int>& emitDrops) override;

 private:
  void stepChain(PendulumChain& c, float dt);  // PBD (filled in Task 4)
  // emission point for a spoid attached to chain c (Task 5 fills the offset).
  glm::vec3 emissionPoint(const PendulumChain& c, const Spoid& s) const;
};
```

- [ ] **Step 2: Implement `resetPendulum()`** in `paint_splatter.cpp`. Add this function (place it next to `arrangeSpoidsCircle`, near line 1191):

```cpp
// Build pendulumChains[0] as a straight line hung from the pivot, displaced
// from the +Y (down) vertical by (initTheta, initPhi). node[0] = pivot (fixed).
void VgeExample::resetPendulum() {
  if (pendulumChains.empty()) pendulumChains.resize(1);
  PendulumChain& c = pendulumChains[0];
  const int n = std::max(1, std::min(c.numLinks,
                                     static_cast<int>(kMaxChainNodes) - 1));
  c.numLinks = n;
  // uniform link length + uniform default mass if not sized to n.
  c.linkLength.assign(n, c.totalLength / static_cast<float>(n));
  if (static_cast<int>(c.bobMass.size()) != n) c.bobMass.assign(n, 1.0f);
  // direction from vertical +Y by (theta, phi): theta from +Y, phi about Y.
  const float st = std::sin(c.initTheta), ct = std::cos(c.initTheta);
  const glm::vec3 dir(st * std::cos(c.initPhi), ct, st * std::sin(c.initPhi));
  c.nodes.assign(n + 1, PendulumNode{});
  c.nodes[0].pos = c.pivot;
  c.nodes[0].prevPos = c.pivot;
  c.nodes[0].invMass = 0.f;  // pivot pinned
  glm::vec3 p = c.pivot;
  for (int i = 1; i <= n; i++) {
    p += dir * c.linkLength[i - 1];
    c.nodes[i].pos = p;
    c.nodes[i].prevPos = p;
    c.nodes[i].vel = glm::vec3(0.f);
    float m = c.bobMass[i - 1];
    c.nodes[i].invMass = (m > 0.f) ? 1.0f / m : 0.f;  // m<=0 => pinned
  }
  // initial tangential push at the tip (perpendicular to the last link).
  if (c.initSpeed != 0.f && n >= 1) {
    glm::vec3 ref = (std::abs(dir.x) < 0.9f) ? glm::vec3(1, 0, 0)
                                             : glm::vec3(0, 0, 1);
    glm::vec3 tangent = glm::normalize(glm::cross(dir, ref));
    c.nodes[n].vel = tangent * c.initSpeed;
  }
}
```

- [ ] **Step 3: Implement the controller `update()` + `emissionPoint()` (no-offset version).** In `paint_splatter.cpp`, add (near `resetPendulum`):

```cpp
glm::vec3 PendulumSpoidController::emissionPoint(const PendulumChain& c,
                                                 const Spoid& s) const {
  if (c.nodes.size() < 2) return c.pivot;
  int ni = (s.nodeIndex < 0) ? static_cast<int>(c.nodes.size()) - 1 : s.nodeIndex;
  ni = std::max(1, std::min(ni, static_cast<int>(c.nodes.size()) - 1));
  return c.nodes[ni].pos;  // offset added in Task 5
}

void PendulumSpoidController::update(float dt, std::vector<Spoid>& spoids,
                                     const InputState& in,
                                     std::vector<int>& emitDrops) {
  (void)in;
  (void)emitDrops;  // pendulum mode uses stream emission (render loop)
  for (PendulumChain& c : chains) stepChain(c, dt);  // no-op until Task 4
  if (chains.empty()) return;
  for (Spoid& s : spoids) {
    s.offsetPhase += s.offsetOmega * dt;
    s.pos = emissionPoint(chains[0], s);
  }
}

// Placeholder until Task 4 adds the PBD step.
void PendulumSpoidController::stepChain(PendulumChain& c, float dt) {
  (void)c;
  (void)dt;
}
```

- [ ] **Step 4: Exempt pendulum spoids from the position clamp.** In `paint_splatter.cpp`, the spoid clamp loop is at lines 1171-1177 inside `updateSpoids`. Wrap it so it only runs in keyboard mode:

```cpp
  if (spoidControlMode == SpoidControlMode::Keyboard) {
    const float m = 0.05f;
    const float xzBound = 2.f * kDomainHalf;  // one canvas-width past each edge
    for (auto& s : spoids) {
      s.pos.x = glm::clamp(s.pos.x, -xzBound, xzBound);
      s.pos.z = glm::clamp(s.pos.z, -xzBound, xzBound);
      s.pos.y = glm::clamp(s.pos.y, -kDomainHeight + m, -0.1f);
    }
  }
```

(The pendulum's own physics + a soft clamp added in Task 4 own the bob positions.)

- [ ] **Step 5: Add a mode switch helper + UI.** In `paint_splatter.cpp`, add a helper near `resetPendulum`:

```cpp
void VgeExample::setSpoidControlMode(SpoidControlMode mode) {
  spoidControlMode = mode;
  if (mode == SpoidControlMode::Pendulum) {
    resetPendulum();
    spoidController =
        std::make_unique<PendulumSpoidController>(pendulumChains, gravity);
  } else {
    spoidController = std::make_unique<KeyboardSpoidController>();
  }
}
```

Declare it in the hpp private section (near `resetPendulum();`):
```cpp
  void setSpoidControlMode(SpoidControlMode mode);
```

In `onUpdateUIOverlay()` (`paint_splatter.cpp`), just after the `ImGui::Checkbox("spherical spawn (ball)", &sphericalSpawn);` line (1968), add a mode radio:

```cpp
    // --- Phase 2: spoid control mode ---
    int modeI = static_cast<int>(spoidControlMode);
    if (ImGui::RadioButton("keyboard", &modeI, 0)) {
      setSpoidControlMode(SpoidControlMode::Keyboard);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("pendulum", &modeI, 1)) {
      setSpoidControlMode(SpoidControlMode::Pendulum);
    }
```

- [ ] **Step 6: Build + run.** Run: `rtk cmd /c mingwBuild.bat` then `./build/paint_splatter.exe`
Expected: **VL-CLEAN**. Default keyboard mode behaves as before. Click "pendulum": the spoid jumps to the static tip of a (not-yet-drawn) double chain hanging from (0,−2.8,0); with stream mode on it drips a static line of paint there. No motion yet.

- [ ] **Step 7: Commit.**

```bash
clang-format -i src/examples/paint_splatter/paint_splatter.cpp src/examples/paint_splatter/paint_splatter.hpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): PendulumSpoidController + mode switch (static chain)"
```

---

## Task 3 — Chain visualization: link lines + joint sprites

**Goal:** see the chain — joints as marker sprites (reusing `markerPipeline`) and links as lines (new `eLineList` pipeline + trivial shaders).

**Files:**
- Create: `shaders/paint_splatter/chain_line.vert`, `shaders/paint_splatter/chain_line.frag`
- Modify: `src/examples/paint_splatter/paint_splatter.cpp` (`createChainBuffers`, `createLinePipeline`, prepare() wiring, draw)

- [ ] **Step 1: Write `chain_line.vert`.** Create `shaders/paint_splatter/chain_line.vert` (matches the Particle vertex input so it reuses the marker pipeline's `vertexInputSCI`):

```glsl
#version 450
layout(location = 0) in vec4 inPos;
layout(location = 1) in vec4 inVel;
layout(location = 2) in vec4 inPredict;
layout(location = 3) in vec4 inColor;
layout(set = 0, binding = 0) uniform GlobalUbo {
  mat4 projection;
  mat4 view;
  mat4 inverseView;
  vec4 canvasInfo;
}
ubo;
layout(location = 0) out vec4 vColor;
void main() {
  gl_Position = ubo.projection * ubo.view * vec4(inPos.xyz, 1.0);
  vColor = inColor;
}
```

- [ ] **Step 2: Write `chain_line.frag`.** Create `shaders/paint_splatter/chain_line.frag`:

```glsl
#version 450
layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;
void main() { outColor = vColor; }
```

- [ ] **Step 3: Implement `createChainBuffers()`.** In `paint_splatter.cpp`, add (next to `createMarkerBuffers`, near line 343):

```cpp
// Per-frame host-visible buffers for chain visualization. jointMarkerBuffers:
// one Particle slot per node (drawn as marker points). lineBuffers: two slots
// per link (drawn as an eLineList).
void VgeExample::createChainBuffers() {
  jointMarkerBuffers.reserve(MAX_CONCURRENT_FRAMES);
  lineBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    jointMarkerBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(Particle), kMaxChainNodes,
        vk::BufferUsageFlagBits::eVertexBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT));
    lineBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(Particle), kMaxChainNodes * 2,
        vk::BufferUsageFlagBits::eVertexBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT));
  }
}
```

- [ ] **Step 4: Implement `createLinePipeline()`.** In `paint_splatter.cpp`, add this function. It mirrors the marker pipeline block (`createParticlePipeline`, lines 748-771) but with `eLineList` topology and the `chain_line` shaders. It rebuilds the shared vertex-input + fixed-function state locally (copying the same values used at lines 680-743) so it is self-contained:

```cpp
void VgeExample::createLinePipeline() {
  // Vertex input: same Particle layout as the particle/marker pipelines.
  vk::VertexInputBindingDescription bindingDesc(0, sizeof(Particle),
                                                vk::VertexInputRate::eVertex);
  std::vector<vk::VertexInputAttributeDescription> attrDescs;
  attrDescs.emplace_back(0, 0, vk::Format::eR32G32B32A32Sfloat,
                         static_cast<uint32_t>(offsetof(Particle, pos)));
  attrDescs.emplace_back(1, 0, vk::Format::eR32G32B32A32Sfloat,
                         static_cast<uint32_t>(offsetof(Particle, vel)));
  attrDescs.emplace_back(2, 0, vk::Format::eR32G32B32A32Sfloat,
                         static_cast<uint32_t>(offsetof(Particle, predict)));
  attrDescs.emplace_back(3, 0, vk::Format::eR32G32B32A32Sfloat,
                         static_cast<uint32_t>(offsetof(Particle, color)));
  vk::PipelineVertexInputStateCreateInfo vertexInputSCI(
      vk::PipelineVertexInputStateCreateFlags(), bindingDesc, attrDescs);

  vk::PipelineInputAssemblyStateCreateInfo inputAssemblySCI(
      vk::PipelineInputAssemblyStateCreateFlags(),
      vk::PrimitiveTopology::eLineList);
  vk::PipelineViewportStateCreateInfo viewportSCI(
      vk::PipelineViewportStateCreateFlags(), 1, nullptr, 1, nullptr);
  vk::PipelineRasterizationStateCreateInfo rasterizationSCI(
      vk::PipelineRasterizationStateCreateFlags(), false, false,
      vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone,
      vk::FrontFace::eCounterClockwise, false, 0.0f, 0.0f, 0.0f, 1.0f);
  vk::PipelineMultisampleStateCreateInfo multisampleSCI(
      vk::PipelineMultisampleStateCreateFlags(), vk::SampleCountFlagBits::e1);
  vk::StencilOpState stencilOpState(vk::StencilOp::eKeep, vk::StencilOp::eKeep,
                                    vk::StencilOp::eKeep,
                                    vk::CompareOp::eAlways);
  vk::PipelineDepthStencilStateCreateInfo depthStencilSCI(
      vk::PipelineDepthStencilStateCreateFlags(), true, true,
      vk::CompareOp::eLessOrEqual, false, false, stencilOpState,
      stencilOpState);
  vk::PipelineColorBlendAttachmentState colorBlendAttachmentState(
      false, vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
      vk::BlendFactor::eZero, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
  vk::PipelineColorBlendStateCreateInfo colorBlendSCI(
      vk::PipelineColorBlendStateCreateFlags(), false, vk::LogicOp::eNoOp,
      colorBlendAttachmentState, {{1.0f, 1.0f, 1.0f, 1.0f}});
  std::array<vk::DynamicState, 2> dynamicStates = {vk::DynamicState::eViewport,
                                                   vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dynamicSCI(
      vk::PipelineDynamicStateCreateFlags(), dynamicStates);

  auto vCode =
      vgeu::readFile(getShadersPath() + "/paint_splatter/chain_line.vert.spv");
  auto fCode =
      vgeu::readFile(getShadersPath() + "/paint_splatter/chain_line.frag.spv");
  vk::raii::ShaderModule vModule = vgeu::createShaderModule(device, vCode);
  vk::raii::ShaderModule fModule = vgeu::createShaderModule(device, fCode);
  std::array<vk::PipelineShaderStageCreateInfo, 2> stageCIs{
      vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                        vk::ShaderStageFlagBits::eVertex,
                                        *vModule, "main", nullptr),
      vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                        vk::ShaderStageFlagBits::eFragment,
                                        *fModule, "main", nullptr),
  };
  vk::GraphicsPipelineCreateInfo lineCI(
      vk::PipelineCreateFlags(), stageCIs, &vertexInputSCI, &inputAssemblySCI,
      nullptr, &viewportSCI, &rasterizationSCI, &multisampleSCI,
      &depthStencilSCI, &colorBlendSCI, &dynamicSCI, *pipelineLayout,
      *renderPass);
  linePipeline = vk::raii::Pipeline(device, pipelineCache, lineCI);
}
```

- [ ] **Step 5: Wire the new setup into `prepare()`.** In `paint_splatter.cpp` `prepare()`, after `createMarkerBuffers();` (line 96) add `createChainBuffers();`. After `createParticlePipeline();` (line 103) add `createLinePipeline();`.

- [ ] **Step 6: Draw the chain.** In `paint_splatter.cpp` `buildCommandBuffers()`, after the spoid-marker draw block (ends at line 1899) add:

```cpp
  // --- Phase 2: pendulum chain (links + joints) ---
  if (showChain && spoidControlMode == SpoidControlMode::Pendulum &&
      !pendulumChains.empty()) {
    const PendulumChain& c = pendulumChains[0];
    const uint32_t nodeCount =
        std::min<uint32_t>(static_cast<uint32_t>(c.nodes.size()), kMaxChainNodes);
    if (nodeCount >= 2) {
      // joints
      Particle* jm = static_cast<Particle*>(
          jointMarkerBuffers[currentFrameIndex]->getMappedData());
      for (uint32_t i = 0; i < nodeCount; i++) {
        jm[i].pos = glm::vec4(c.nodes[i].pos, 1.f);
        jm[i].vel = glm::vec4(0.f);
        jm[i].predict = glm::vec4(0.f);
        jm[i].color = glm::vec4(0.9f, 0.9f, 0.2f, 1.f);  // joint = yellow
      }
      // links: two verts per segment (node i-1 -> node i)
      Particle* lv = static_cast<Particle*>(
          lineBuffers[currentFrameIndex]->getMappedData());
      const uint32_t links = nodeCount - 1;
      for (uint32_t i = 0; i < links; i++) {
        lv[2 * i].pos = glm::vec4(c.nodes[i].pos, 1.f);
        lv[2 * i].color = glm::vec4(0.7f, 0.7f, 0.7f, 1.f);  // string = grey
        lv[2 * i + 1].pos = glm::vec4(c.nodes[i + 1].pos, 1.f);
        lv[2 * i + 1].color = glm::vec4(0.7f, 0.7f, 0.7f, 1.f);
      }
      vk::DeviceSize off(0);
      // lines
      drawCmdBuffers[currentFrameIndex].bindPipeline(
          vk::PipelineBindPoint::eGraphics, *linePipeline);
      drawCmdBuffers[currentFrameIndex].bindDescriptorSets(
          vk::PipelineBindPoint::eGraphics, *pipelineLayout, 0,
          {*descriptorSets[currentFrameIndex]}, nullptr);
      drawCmdBuffers[currentFrameIndex].bindVertexBuffers(
          0, lineBuffers[currentFrameIndex]->getBuffer(), off);
      drawCmdBuffers[currentFrameIndex].draw(links * 2, 1, 0, 0);
      // joints (marker pipeline)
      drawCmdBuffers[currentFrameIndex].bindPipeline(
          vk::PipelineBindPoint::eGraphics, *markerPipeline);
      drawCmdBuffers[currentFrameIndex].bindVertexBuffers(
          0, jointMarkerBuffers[currentFrameIndex]->getBuffer(), off);
      drawCmdBuffers[currentFrameIndex].draw(nodeCount, 1, 0, 0);
    }
  }
```

- [ ] **Step 7: Build (with shader reconfigure) + run.** Run: `rtk cmd /c mingwBuild.bat`
If `shaders/paint_splatter/chain_line.vert.spv` and `chain_line.frag.spv` are NOT created, force a CMake reconfigure (the GLOB only re-scans on configure) and rebuild. Then `./build/paint_splatter.exe`.
Expected: **VL-CLEAN**. In pendulum mode a static double chain is visible: a grey two-segment string from (0,−2.8,0) down to the tip, with three yellow joint dots (pivot + 2 bobs). Keyboard mode shows no chain.

- [ ] **Step 8: GATE (user).** User confirms the static double-pendulum chain (lines + joints) renders correctly in pendulum mode, VL-CLEAN.

- [ ] **Step 9: Commit.**

```bash
clang-format -i src/examples/paint_splatter/paint_splatter.cpp src/examples/paint_splatter/paint_splatter.hpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): chain visualization (link lines + joint sprites) (Phase 2)"
```

---

## Task 4 — PBD pendulum step (the swing)

**Goal:** make the chain swing under gravity via PBD: predict → distance constraints → velocity → damping, with per-node inverse mass and a stability soft-clamp. Add a tip readback.

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.cpp` (`PendulumSpoidController::stepChain`, tip readback print)

- [ ] **Step 1: Implement `stepChain()`.** In `paint_splatter.cpp`, replace the placeholder `PendulumSpoidController::stepChain` (Task 2 Step 3) with:

```cpp
void PendulumSpoidController::stepChain(PendulumChain& c, float dt) {
  if (c.nodes.size() < 2 || dt <= 0.f) return;
  const int sub = std::max(1, c.substeps);
  const float sdt = dt / static_cast<float>(sub);
  const float g = gravity;  // +Y (down on screen)
  const float maxSpeed = 50.f;  // coarse CFL safety net (world u/s)
  for (int s = 0; s < sub; s++) {
    // 1. predict (skip the fixed pivot, node 0)
    for (size_t i = 1; i < c.nodes.size(); i++) {
      PendulumNode& nd = c.nodes[i];
      if (nd.invMass == 0.f) continue;  // pinned
      nd.vel.y += g * sdt;
      nd.prevPos = nd.pos;
      nd.pos += nd.vel * sdt;
    }
    // 2. distance constraints (Gauss-Seidel)
    for (int it = 0; it < std::max(1, c.iters); it++) {
      for (size_t i = 1; i < c.nodes.size(); i++) {
        PendulumNode& a = c.nodes[i - 1];
        PendulumNode& b = c.nodes[i];
        float w = a.invMass + b.invMass;
        if (w == 0.f) continue;  // both pinned
        glm::vec3 d = b.pos - a.pos;
        float len = glm::length(d);
        if (len < 1e-6f) continue;
        float L = c.linkLength[i - 1];
        glm::vec3 corr = (len - L) / (w * len) * d;
        a.pos += a.invMass * corr;
        b.pos -= b.invMass * corr;
      }
    }
    // 3. velocity update + 4. damping
    const float airK = std::max(0.f, 1.f - c.airDamping * sdt);
    for (size_t i = 1; i < c.nodes.size(); i++) {
      PendulumNode& nd = c.nodes[i];
      if (nd.invMass == 0.f) {
        nd.vel = glm::vec3(0.f);
        continue;
      }
      nd.vel = (nd.pos - nd.prevPos) / sdt;
      nd.vel *= airK;  // air resistance (global)
    }
    // joint/string friction: damp the relative velocity component ALONG each
    // link (energy lost at the string end).
    for (size_t i = 1; i < c.nodes.size(); i++) {
      PendulumNode& a = c.nodes[i - 1];
      PendulumNode& b = c.nodes[i];
      glm::vec3 d = b.pos - a.pos;
      float len = glm::length(d);
      if (len < 1e-6f) continue;
      glm::vec3 axis = d / len;
      glm::vec3 rel = b.vel - a.vel;
      float along = glm::dot(rel, axis);
      glm::vec3 damp = axis * (along * c.jointDamping);
      if (b.invMass > 0.f) b.vel -= damp;
      if (a.invMass > 0.f) a.vel += damp;
    }
    // stability: CFL speed cap + soft-clamp y above the floor (pivot exempt).
    for (size_t i = 1; i < c.nodes.size(); i++) {
      PendulumNode& nd = c.nodes[i];
      float sp = glm::length(nd.vel);
      if (sp > maxSpeed) nd.vel *= maxSpeed / sp;
      // floor y=0, ceiling y=-3 (kDomainHeight). keep bobs above the floor.
      nd.pos.y = glm::clamp(nd.pos.y, -3.0f + 0.05f, -0.1f);
    }
  }
}
```

- [ ] **Step 2: Add a tip readback print.** In `paint_splatter.cpp` `consumeParticleReadback()`, just before the terminating `<< std::endl;` of the readback `std::cout` (the existing `ext x=... z=...` line, near line 1402-ish, search for `" | ext x="`), append a pendulum tip term:

```cpp
            << (spoidControlMode == SpoidControlMode::Pendulum &&
                        !pendulumChains.empty() &&
                        pendulumChains[0].nodes.size() >= 2
                    ? " | tip=(" +
                          std::to_string(pendulumChains[0].nodes.back().pos.x) +
                          "," +
                          std::to_string(pendulumChains[0].nodes.back().pos.y) +
                          "," +
                          std::to_string(pendulumChains[0].nodes.back().pos.z) +
                          ") tipv=" +
                          std::to_string(
                              glm::length(pendulumChains[0].nodes.back().vel))
                    : std::string())
```

(If the readback uses `printf`/a different stream, instead add an equivalent `std::cout << "tip=..."` line right after the existing readback print. The intent: print tip position + speed each readback so divergence is visible numerically.)

- [ ] **Step 3: Build + run.** Run: `rtk cmd /c mingwBuild.bat` then `./build/paint_splatter.exe`
Expected: **VL-CLEAN**. In pendulum mode the double chain swings (chaotically). The tip readback `tipv` stays bounded (well under `maxSpeed`); no NaN. With stream mode on, the spoid paints a swinging trail.

- [ ] **Step 4: GATE (user).** User confirms the double pendulum swings smoothly with no divergence/NaN/device-loss; the spoid draws a swinging paint trail.

- [ ] **Step 5: Commit.**

```bash
clang-format -i src/examples/paint_splatter/paint_splatter.cpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): PBD pendulum step (swing + damping + stability) (Phase 2)"
```

---

## Task 5 — Per-spoid rotary offset (r, angle₀, ω)

**Goal:** offset the emission point from the attach node in the plane perpendicular to the string, spinning at constant ω → spirograph.

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.cpp` (`PendulumSpoidController::emissionPoint`)

- [ ] **Step 1: Replace `emissionPoint()` with the offset version.** In `paint_splatter.cpp`, replace the Task 2 `emissionPoint` body with:

```cpp
glm::vec3 PendulumSpoidController::emissionPoint(const PendulumChain& c,
                                                 const Spoid& s) const {
  if (c.nodes.size() < 2) return c.pivot;
  int ni = (s.nodeIndex < 0) ? static_cast<int>(c.nodes.size()) - 1 : s.nodeIndex;
  ni = std::max(1, std::min(ni, static_cast<int>(c.nodes.size()) - 1));
  glm::vec3 node = c.nodes[ni].pos;
  if (s.offsetR == 0.f) return node;
  // link direction at this node (node - parent).
  glm::vec3 d = node - c.nodes[ni - 1].pos;
  float dl = glm::length(d);
  if (dl < 1e-6f) return node;
  d /= dl;
  // orthonormal basis of the plane perpendicular to the string. Swap the
  // reference axis when d is nearly parallel to it (singularity guard).
  glm::vec3 ref = (std::abs(d.x) < 0.9f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 0, 1);
  glm::vec3 e1 = glm::normalize(glm::cross(d, ref));
  glm::vec3 e2 = glm::cross(d, e1);
  float ph = s.offsetPhase;  // angle0 + omega*t, advanced in update()
  return node + s.offsetR * (std::cos(ph) * e1 + std::sin(ph) * e2);
}
```

Also ensure `offsetPhase` starts at `offsetAngle0`: in `update()` it is advanced by `offsetOmega*dt` each frame, but it must be initialized. Add to `setSpoidControlMode` (when switching TO pendulum), after `resetPendulum();`:

```cpp
    for (Spoid& s : spoids) s.offsetPhase = s.offsetAngle0;
```

- [ ] **Step 2: Build + run.** Run: `rtk cmd /c mingwBuild.bat` then `./build/paint_splatter.exe`
Expected: **VL-CLEAN**. Set the selected spoid's `offsetR` > 0 and `offsetOmega` ≠ 0 (sliders added in Task 8; for now temporarily set defaults in code or test after Task 8). The spoid marker orbits around the bob; with stream mode the painted trail becomes a spirograph rosette superimposed on the swing.

> NOTE: the sliders to drive `offsetR/angle0/omega` live in Task 8. To verify this task standalone, temporarily set `Spoid` defaults `offsetR = 0.2f; offsetOmega = 6.f;` in the hpp, confirm the orbit, then revert to `0.f` before committing (Task 8 exposes them as sliders). 

- [ ] **Step 3: Commit.**

```bash
clang-format -i src/examples/paint_splatter/paint_splatter.cpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): per-spoid rotary offset (r, angle0, omega) (Phase 2)"
```

---

## Task 6 — Paint reservoir (`paintMass`) gating

**Goal:** each spoid's `paintMass` gates and is drained by emission; empty spoids stop; a UI refill resets it.

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.cpp` (stream loop at 1481-1496; a refill in the UI is added in Task 8)

- [ ] **Step 1: Gate + drain in the stream loop.** In `paint_splatter.cpp`, in the `streamMode` branch (lines 1487-1496), change the per-spoid body so emission is gated by `paintMass` and drains it. Replace the loop body:

```cpp
    const float kDrain = 1.0e-4f;  // reservoir per emitted particle
    for (Spoid& s : spoids) {
      if (!isEmitter(s)) continue;
      if (s.paintMass <= 0.f) continue;  // empty -> no emission
      s.emitAccum += streamRate * dt;
      int n = static_cast<int>(s.emitAccum);
      if (n > 0) {
        s.emitAccum -= static_cast<float>(n);
        enqueueDrop(s.pos, s.holeRadius, s.color, s.emissionVelocity,
                    s.concentration, n, s.prevPos);
        s.paintMass -= kDrain * static_cast<float>(n);
      }
    }
```

(Leave the burst/auto-emit branch unchanged; the reservoir gates only the continuous stream, which is the pendulum's emission path.)

- [ ] **Step 2: Build + run.** Run: `rtk cmd /c mingwBuild.bat` then `./build/paint_splatter.exe`
Expected: **VL-CLEAN**. In stream mode, a spoid emits until `paintMass` (1.0) drains (~10k particles at `kDrain=1e-4`), then stops. (Refill UI comes in Task 8; for now restarting the app refills.)

- [ ] **Step 3: Commit.**

```bash
clang-format -i src/examples/paint_splatter/paint_splatter.cpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): paintMass reservoir gates stream emission (Phase 2)"
```

---

## Task 7 — Top-view camera animation button

**Goal:** a button that smoothly animates the camera from the current view to an overhead top-down framing and holds it; pressing again releases back to orbit.

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.cpp` (`startTopViewAnim`, `updateCameraAnim`, render() hook, UI button)

- [ ] **Step 1: Implement the animation helpers.** In `paint_splatter.cpp`, add:

```cpp
void VgeExample::startTopViewAnim() {
  if (cameraAnim.active || cameraAnim.locked) {
    // toggle off: release back to the orbit controller.
    cameraAnim.active = false;
    cameraAnim.locked = false;
    return;
  }
  cameraAnim.fromEye = camera.getPosition();
  // current look-at point: project forward from the eye along view dir. Simpler
  // and robust: aim at the canvas centre (0,0,0), the orbit target.
  cameraAnim.fromTarget = glm::vec3(0.f);
  cameraAnim.toEye = glm::vec3(0.f, -6.f, 0.f);  // overhead (world -Y is up)
  cameraAnim.toTarget = glm::vec3(0.f);
  cameraAnim.up = glm::vec3(0.f, 0.f, -1.f);  // non-degenerate for a +Y view dir
  cameraAnim.t = 0.f;
  cameraAnim.active = true;
}

void VgeExample::updateCameraAnim() {
  if (!cameraAnim.active && !cameraAnim.locked) return;
  glm::vec3 eye = cameraAnim.toEye, target = cameraAnim.toTarget;
  if (cameraAnim.active) {
    cameraAnim.t += frameTimer;
    float u = glm::clamp(cameraAnim.t / cameraAnim.duration, 0.f, 1.f);
    float e = u * u * (3.f - 2.f * u);  // smoothstep ease in/out
    eye = glm::mix(cameraAnim.fromEye, cameraAnim.toEye, e);
    target = glm::mix(cameraAnim.fromTarget, cameraAnim.toTarget, e);
    if (u >= 1.f) {
      cameraAnim.active = false;
      cameraAnim.locked = true;
    }
  }
  camera.setViewTarget(eye, target, cameraAnim.up);
}
```

- [ ] **Step 2: Hook it into `render()`.** In `paint_splatter.cpp` `render()`, immediately BEFORE the `updateGlobalUbo();` call (line 1514), add:

```cpp
  updateCameraAnim();  // overrides the orbit view while animating/locked
```

(Order: the base loop calls the subclass `render()` at `vge_base.cpp:183` BEFORE the camera-controller update at `:196-201`. So the camera matrix in `render()` is from the previous frame's end; overriding it here — right before `updateGlobalUbo()` reads `camera.getView()` — wins for the current draw. The controller's `setViewYXZ` at frame end is harmless: while `active`/`locked` we re-override every frame, and on release the camera simply retains the controller's orbit pose, snapping back to it. `frameTimer` inside `render()` is the previous frame's dt, consistent with how `updateSpoids` already uses it.)

- [ ] **Step 3: Add the UI button.** In `paint_splatter.cpp` `onUpdateUIOverlay()`, just before the existing `Save PNG` button (line 2107), add:

```cpp
      if (uiOverlay->button(cameraAnim.locked || cameraAnim.active
                                ? "Free camera"
                                : "Top view")) {
        startTopViewAnim();
      }
```

- [ ] **Step 4: Build + run.** Run: `rtk cmd /c mingwBuild.bat` then `./build/paint_splatter.exe`
Expected: **VL-CLEAN**. Clicking "Top view" smoothly animates (~1 s) from the current orbit to an overhead view of the canvas and holds it; the button now reads "Free camera"; clicking it returns control to the orbit camera. Save PNG still works (and is camera-independent).

- [ ] **Step 5: GATE (user).** User confirms the smooth top-view animation and release.

- [ ] **Step 6: Commit.**

```bash
clang-format -i src/examples/paint_splatter/paint_splatter.cpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): smooth top-view camera animation button (Phase 2)"
```

---

## Task 8 — ImGui pendulum panel (chain config + per-spoid pendulum params)

**Goal:** expose all pendulum knobs live so the harmonograph can be tuned, plus a reservoir refill. This is the milestone where the feature becomes usable.

**Files:**
- Modify: `src/examples/paint_splatter/paint_splatter.cpp` (`onUpdateUIOverlay`)

- [ ] **Step 1: Add a pendulum config panel.** In `paint_splatter.cpp` `onUpdateUIOverlay()`, after the mode radio added in Task 2 Step 5, add (only shown in pendulum mode):

```cpp
    if (spoidControlMode == SpoidControlMode::Pendulum &&
        !pendulumChains.empty()) {
      PendulumChain& c = pendulumChains[0];
      if (ImGui::CollapsingHeader("Pendulum", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("show chain", &showChain);
        bool rebuild = false;
        rebuild |= ImGui::DragInt("links (n)", &c.numLinks, 0.1f, 1,
                                  static_cast<int>(kMaxChainNodes) - 1);
        rebuild |= ImGui::DragFloat("total length", &c.totalLength, 0.01f, 0.1f,
                                    2.8f, "%.2f");
        rebuild |= ImGui::DragFloat3("pivot", &c.pivot.x, 0.01f);
        rebuild |= ImGui::DragFloat("init theta (rad)", &c.initTheta, 0.01f,
                                    0.f, 3.14f, "%.2f");
        rebuild |= ImGui::DragFloat("init phi (rad)", &c.initPhi, 0.01f, 0.f,
                                    6.28f, "%.2f");
        rebuild |= ImGui::DragFloat("init speed", &c.initSpeed, 0.01f, 0.f, 10.f,
                                    "%.2f");
        ImGui::DragFloat("air damping (/s)", &c.airDamping, 0.005f, 0.f, 5.f,
                         "%.3f");
        ImGui::DragFloat("joint damping", &c.jointDamping, 0.005f, 0.f, 1.f,
                         "%.3f");
        ImGui::DragInt("substeps", &c.substeps, 0.2f, 1, 32);
        ImGui::DragInt("constraint iters", &c.iters, 0.1f, 1, 16);
        // per-node mass (mass<=0 => pinned). Resize to numLinks lazily.
        if (static_cast<int>(c.bobMass.size()) != c.numLinks) {
          c.bobMass.assign(std::max(1, c.numLinks), 1.0f);
        }
        for (int i = 0; i < static_cast<int>(c.bobMass.size()); i++) {
          char lbl[32];
          std::snprintf(lbl, sizeof(lbl), "bob %d mass (<=0 pin)", i + 1);
          rebuild |= ImGui::DragFloat(lbl, &c.bobMass[i], 0.02f, 0.f, 10.f,
                                      "%.2f");
        }
        if (ImGui::Button("reset pendulum") || rebuild) {
          resetPendulum();
          for (Spoid& s : spoids) s.offsetPhase = s.offsetAngle0;
        }
      }
    }
```

- [ ] **Step 2: Add per-spoid pendulum params + reservoir refill.** In `paint_splatter.cpp` `onUpdateUIOverlay()`, inside the existing "Spoids" collapsing header where per-spoid sliders live (near the `hole radius (disk)` slider, ~line 1968 region of the spoid editing block), add these sliders for the selected spoid `s` (use the same `editAllSpoids` propagation pattern already used by the existing spoid sliders):

```cpp
        if (ImGui::DragFloat("offset r", &s.offsetR, 0.005f, 0.f, 1.f, "%.3f") &&
            editAllSpoids)
          for (auto& o : spoids) o.offsetR = s.offsetR;
        if (ImGui::DragFloat("offset angle0 (rad)", &s.offsetAngle0, 0.01f, 0.f,
                             6.28f, "%.2f") &&
            editAllSpoids)
          for (auto& o : spoids) o.offsetAngle0 = s.offsetAngle0;
        if (ImGui::DragFloat("offset omega (rad/s)", &s.offsetOmega, 0.02f,
                             -20.f, 20.f, "%.2f") &&
            editAllSpoids)
          for (auto& o : spoids) o.offsetOmega = s.offsetOmega;
        ImGui::Text("paint mass : %.3f", s.paintMass);
        if (ImGui::Button("refill paint")) {
          if (editAllSpoids)
            for (auto& o : spoids) o.paintMass = 1.f;
          else
            s.paintMass = 1.f;
        }
```

(Find the selected-spoid reference `s` already established by the existing spoid-editing UI; reuse it. If the existing code indexes `spoids[selectedSpoidUi]`, use that same expression.)

- [ ] **Step 3: Build + run.** Run: `rtk cmd /c mingwBuild.bat` then `./build/paint_splatter.exe`
Expected: **VL-CLEAN**. In pendulum mode: changing `links` to 1/2/3 rebuilds the chain; setting a `bob mass` to 0 pins that node (it stops moving); `init theta/speed` change the swing; `offset r/omega` make the spoid orbit; `refill paint` resets the reservoir; `air/joint damping` visibly change how fast the swing decays.

- [ ] **Step 4: GATE (user).** User confirms live tuning works and a double pendulum + rotary offset paints a harmonograph/spirograph pattern on the canvas (turn on stream mode), and `mass=0` pins a node.

- [ ] **Step 5: Commit.**

```bash
clang-format -i src/examples/paint_splatter/paint_splatter.cpp
rtk git add -A && rtk git commit -m "feat(paint_splatter): ImGui pendulum panel + per-spoid offset/reservoir (Phase 2)"
```

---

## Task 9 — Docs

**Files:**
- Modify: `docs/09_paint_splatter.md`

- [ ] **Step 1: Add a Phase 2 section** to `docs/09_paint_splatter.md` describing: the `PendulumSpoidController`, the n-link PBD chain (predict → distance constraints → velocity → air/joint damping, `mass<=0` = pinned), the per-spoid rotary offset `(r, angle₀, ω)` producing harmonograph/spirograph patterns, the `paintMass` reservoir, the chain visualization (link lines + joint sprites), and the top-view camera animation button. Keep the style of the existing doc (overview + how-to + parameter list). Note it is host-side only (no compute/UBO change).

- [ ] **Step 2: Commit.**

```bash
rtk git add docs/09_paint_splatter.md
rtk git commit -m "docs: paint_splatter Phase 2 (PBD pendulum spoids + top view)"
```

- [ ] **GATE (user) — Phase 2 complete:** Build and run. User confirms the full feature: a double pendulum swings smoothly (pure PBD, no divergence), spoids with rotary offset paint a harmonograph/spirograph onto the canvas in stream mode, the reservoir drains/refills, the chain renders (lines + joints), `mass=0` pins a node, and the "Top view" button smoothly animates overhead before saving. VL-CLEAN throughout.

---

## Self-Review Notes (coverage vs spec)

- Spec "host-side only, no GPU-struct/UBO/shader-physics change" → Tasks 1-8 touch only host C++ + two trivial line shaders; no `static_assert`/UBO/compute change. ✔
- Spec data model (`PendulumNode`, `PendulumChain` per-node `bobMass` with `mass<=0`⇒pinned, `Spoid` Phase-2 fields) → Task 1. ✔
- Spec controller (positions spoid, stream emission, keyboard default) → Task 2 (skeleton) + Task 4 (PBD) + Task 5 (offset). ✔
- Spec PBD step (predict, distance constraints by inverse mass, velocity, air + joint damping, stability nets, pivot exempt from Phase-1 clamp) → Task 4 + Task 2 Step 4. ✔
- Spec rotary offset `(r, angle₀, ω)` perpendicular-plane geometry + singularity guard → Task 5. ✔
- Spec reservoir `paintMass` simple linear drain gating stream → Task 6 + refill in Task 8. ✔
- Spec visualization (link lines new `eLineList` pipeline + joint sprites reusing `markerPipeline`) → Task 3. ✔
- Spec top-view camera animation (orbit unchanged; button; Save PNG separate) → Task 7. ✔
- Spec verification (VL-CLEAN / tip READBACK / user GATE) → each task ends in build+run; Tasks 3,4,7,8 + final end in user GATE; Task 4 adds the tip readback. ✔
- Spec "pure PBD, no analytic" → no analytic mode anywhere. ✔
- Spec generalization hooks (`vector<PendulumChain>`, `nodeIndex`) → present in Task 1/2 (single chain used, structure ready). ✔

**Known v1 simplifications (decided, not placeholders):** uniform link length (per-link length deferred — Task 1 `linkLength` filled uniformly); single chain (`pendulumChains` is a vector but one element); `kDrain=1e-4` and `topHeight=6`/`toEye.y=-6` are concrete starting values tunable in code; offset basis reference-axis swap is the documented singularity handling. All match the spec's "Deferred" list.
