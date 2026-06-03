#pragma once

#include "vge_base.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

// std
#include <memory>
#include <vector>

namespace vge {

struct CanvasVertex {
  float position[3];
  float uv[2];
};

// GlobalUbo: camera matrices + canvas info
// Layout (std140-compatible):
//   projection  mat4  -- 0
//   view        mat4  -- 64
//   inverseView mat4  -- 128
//   canvasInfo  vec4  -- 192
//                        208 total
struct GlobalUbo {
  glm::mat4 projection{1.f};   // -- 0 --
  glm::mat4 view{1.f};         // -- 64 --
  glm::mat4 inverseView{1.f};  // -- 128 --
  glm::vec4 canvasInfo{
      0.f};  // -- 192 -- xy=half-extent, z=world size, w=density-debug flag
  glm::vec4 renderParams{1.f, 0.f, 0.f,
                         0.f};  // -- 208 -- x=particle point-size scale
  // -- 224 --
};
static_assert(sizeof(GlobalUbo) == 224, "GlobalUbo std140 size");

// Particle: std430, 64 bytes
//   pos      vec4  -- 0  -- xyz position, w=1
//   vel      vec4  -- 16 -- xyz velocity, w=wetness/alpha
//   predict  vec4  -- 32 -- xyz predicted x*, w=lambda
//   color    vec4  -- 48 -- rgba paint color (a=concentration)
//                     64
struct Particle {
  glm::vec4 pos;      // -- 0 --
  glm::vec4 vel;      // -- 16 --
  glm::vec4 predict;  // -- 32 --
  glm::vec4 color;    // -- 48 --
  // -- 64 --
};
static_assert(sizeof(Particle) == 64, "Particle std430 size");

// ComputeUbo: std140, 112 bytes. Shared by every PBF compute pass.
struct ComputeUbo {
  float dt;                // -- 0  -- substep dt
  uint32_t particleCount;  // -- 4  --
  float gravity;           // -- 8  -- +Y (down on screen)
  float h;                 // -- 12 -- smoothing radius == grid cell size
  glm::vec4 canvasMin;     // -- 16 -- xyz domain min, w unused
  glm::vec4 canvasMax;     // -- 32 -- xyz domain max (floor at y=cmax.y=0)
  glm::ivec4 gridDim;      // -- 48 -- xyz grid dims, w = numCells
  float rho0;              // -- 64 -- rest density
  float epsCFM;            // -- 68 -- constraint relaxation (Eq. 11)
  float scorrK;            // -- 72 -- artificial pressure strength (Eq. 13)
  float scorrDq;           // -- 76 -- scorr reference distance (= ratio*h)
  float scorrN;            // -- 80 -- scorr exponent
  float xsphC;             // -- 84 -- XSPH viscosity coefficient (Eq. 17)
  float kPoly6;            // -- 88 -- 315/(64 pi h^9)
  float kSpiky;            // -- 92 -- 45/(pi h^6) (gradient magnitude)
  float scorrDenom;        // -- 96 -- 1/W_poly6(scorrDq) precomputed
  float velDamp;           // -- 100 -- global velocity drag rate (per second)
  float velClampFactor;    // -- 104 -- CFL cap: maxSpeed = factor * h / dt
  float solverRelax;       // -- 108 -- Jacobi under-relaxation factor (0..1)
  uint32_t prevCount;      // -- 112 -- ping-pong: # particles carried from prev
  float dryRate;           // -- 116 -- wetness lost/sec while depositing (M6)
  float depositStrength;   // -- 120 -- stamp alpha = concentration*this (M6)
  float depositHeight;     // -- 124 -- deposit when pos.y >= -this (near floor)
  float depositRadius;     // -- 128 -- canvas stamp radius in WORLD units (M6)
  float drySettle;      // -- 132 -- 0..1: how much drying freezes motion (M6)
  float cohesionFloor;  // -- 136 -- bounded signed constraint: C = max(rho/rho0
                        // -1, -cohesionFloor). 0 = compression-only (no
                        // cohesion); ~0.3-0.5 = cohesive droplet (M8)
  float dpClampFactor;  // -- 140 -- per-iteration |dp| cap = factor*h; <=0
                        // disables. Low/off lets the incompressible rebound
                        // (crown) grow; default 0.2 = old behavior (M8)
  // -- 144 --
};
static_assert(sizeof(ComputeUbo) == 144, "ComputeUbo std140 size");

// EmitPush: push constant for the emit pass (M5). One dispatch per droplet
// burst; the host computes baseIndex (append-only live count) so no GPU atomic
// counter is needed until compaction lands in M6.
struct EmitPush {
  glm::vec4 originRadius;  // -- 0  -- xyz spoid origin (y<0), w lattice spacing
  glm::vec4 velConc;   // -- 16 -- xyz initial velocity (+Y), w concentration
  glm::vec4 color;     // -- 32 -- rgb paint color, a unused
  uint32_t baseIndex;  // -- 48 -- first particle slot written
  uint32_t count;      // -- 52 -- particles in this burst
  uint32_t seed;       // -- 56 -- rng seed (varies per burst)
  uint32_t _pad;       // -- 60 --
  // -- 64 --
};
static_assert(sizeof(EmitPush) == 64, "EmitPush push-constant size");

// Spoid (M5): host-side eyedropper POD. Selected via ImGui, moved via keyboard;
// Space releases one droplet burst of `amount` particles from each selected
// spoid. World convention: spoids live above the floor at y<0.
struct Spoid {
  glm::vec3 pos{0.f, -2.5f, 0.f};
  float holeRadius = 0.12f;
  glm::vec3 color{0.2f, 0.4f, 0.9f};
  float emissionVelocity = 2.f;  // initial downward (+Y) speed
  int amount = 300;              // particles per drop
  float concentration = 1.f;     // -> stamp alpha (M6) + opacity
  bool selected = true;
};

// Per-frame keyboard intent, already mapped to world axes (GLFW reading lives
// in the example so the controller stays input-source agnostic).
struct InputState {
  glm::vec3 move{0.f};  // world-space move direction (x,y,z), unnormalized
  bool emit = false;  // edge-triggered: true only on the frame Space goes down
};

// SpoidController abstraction (design §4): Phase 1 = keyboard, Phase 2 will add
// a pendulum-driven controller without touching the solver/emitter.
struct SpoidController {
  virtual ~SpoidController() = default;
  // Move the selected spoids and append emit triggers (spoid indices) for this
  // frame.
  virtual void update(float dt, std::vector<Spoid>& spoids,
                      const InputState& in, std::vector<int>& emitDrops) = 0;
};

// KeyboardSpoidController: moves every selected spoid together; Space releases
// one droplet burst from each selected spoid.
class KeyboardSpoidController : public SpoidController {
public:
  float moveSpeed = 1.5f;  // world units / second
  void update(float dt, std::vector<Spoid>& spoids, const InputState& in,
              std::vector<int>& emitDrops) override {
    for (auto& s : spoids) {
      if (s.selected) s.pos += in.move * (moveSpeed * dt);
    }
    if (in.emit) {
      for (int i = 0; i < static_cast<int>(spoids.size()); i++) {
        if (spoids[i].selected) emitDrops.push_back(i);
      }
    }
  }
};

// Intentionally empty for M1; simulation/spoid knobs are added in later
// milestones.
struct Options {};

class VgeExample : public VgeBase {
public:
  VgeExample();
  ~VgeExample();

  void initVulkan() override;
  void getEnabledFeatures() override;
  void prepare() override;
  void render() override;
  void viewChanged() override;
  void setupCommandLineParser(CLI::App& app) override;
  void onUpdateUIOverlay() override;
  void buildCommandBuffers() override;

  void setOptions(const std::optional<Options>& o) {
    if (o) opts = *o;
  }

  Options opts{};

private:
  void createVertexBuffer();
  void createIndexBuffer();
  void createUniformBuffers();
  void createDescriptorSetLayout();
  void createDescriptorPool();
  void createDescriptorSets();
  void createPipelines();
  void updateGlobalUbo();
  void draw();

  // ---- particle SSBO helpers ----
  void createParticleBuffers();
  void seedParticles();      // (re)fill all particle SSBOs with the lattice
  void restartSimulation();  // waitIdle + reseed + reset sync bootstrap
  void createParticlePipeline();

  // ---- compute helpers ----
  void prepareCompute();
  void createGridBuffers();
  void createComputeDescriptorSetLayout();
  void createComputeDescriptorSets();
  void createComputePipeline();  // builds all PBF compute pipelines
  void updateComputeUbo();
  void buildComputeCommandBuffers();
  // Records one full PBF substep (predict -> grid -> solve iters -> finalize).
  void recordPbfSubstep(const vk::raii::CommandBuffer& cmd, uint32_t frame);

  // ---- emit (M5) ----
  void createEmitPipeline();
  // Append one droplet burst: reserves a contiguous slot range from the single
  // global live count (ping-pong chain) and enqueues one EmitPush. Drained once
  // per frame into the current buffer; the predict pass carries it forward.
  void enqueueDrop(const glm::vec3& origin, float holeRadius,
                   const glm::vec3& color, float emissionVel,
                   float concentration, int amount);
  void recordEmit(const vk::raii::CommandBuffer& cmd, uint32_t frame);
  // Single FIFO of pending bursts (one evolving chain; drained + cleared once
  // per frame in buildComputeCommandBuffers, written into the current buffer).
  std::vector<EmitPush> pendingEmits;
  uint32_t emitSeedCounter = 1u;  // varies the rng seed per burst

  // ---- spoids (M5 Task 10) ----
  void createMarkerBuffers();
  // Read GLFW keys, run the spoid controller, and convert emit triggers into
  // droplet bursts. Called once per frame from render().
  void updateSpoids();
  // Lay the spoids out evenly on a circle in the X-Z plane (n-way split).
  // Called whenever a spoid is added/removed so they stay arranged; heights (y)
  // are preserved so the user can still raise/lower them.
  void arrangeSpoidsCircle();

  // ---- debug readback (M3): print particle y min/max ~once per second ----
  // The particle SSBO participates in the compute<->graphics queue-ownership
  // ping-pong, so an out-of-band copy would break the release/acquire pairing.
  // Instead the copy is recorded INSIDE the graphics command buffer (where the
  // buffer is already owned by graphics) and read one frame later.
  void recordParticleReadbackCopy();  // inside buildCommandBuffers
  void consumeParticleReadback();     // after waitForFences in draw()
  float readbackTimer = 0.f;
  bool readbackRequest = false;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> readbackBuffers;
  std::vector<uint8_t> readbackPending;  // per-frame: copy recorded, await read

  // geometry
  std::unique_ptr<vgeu::VgeuBuffer> vertexBuffer;
  std::unique_ptr<vgeu::VgeuBuffer> indexBuffer;

  // per-frame uniform buffers (GlobalUbo)
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> uniformBuffers;
  GlobalUbo globalUbo{};

  // descriptors (canvas / GlobalUbo)
  std::vector<vk::raii::DescriptorSet> descriptorSets;
  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;

  // pipeline (canvas quad)
  vk::raii::Pipeline pipeline = nullptr;

  // ---- canvas accumulation texture (M6) ----
  // Single RGBA8 image (the permanent painting): written by the deposit compute
  // pass (M6-B) and sampled by canvas.frag. Kept in GENERAL layout so both the
  // compute imageStore and the fragment sample are valid without per-use layout
  // transitions.
  void createCanvasImage();
  // M7 Task 14: read the canvas image back to the host and write a PNG to
  // build/paint_<timestamp>.png. device.waitIdle (simplest correct) -> copy the
  // GENERAL-layout RGBA8 image to a host-visible buffer -> stbi_write_png.
  void saveCanvasPng();
  bool saveRequested = false;  // set by the ImGui button, handled in render()
  std::string saveStatus;      // last save result, shown in the ImGui panel
  std::unique_ptr<vgeu::VgeuImage> canvasImage;
  vk::raii::Sampler canvasSampler = nullptr;
  static constexpr uint32_t kCanvasTexRes = 2048;
  // deposit params (M6-B); live in ComputeUbo pads.
  float dryRate = 2.0f;  // wetness lost per second while depositing
  // Small per-frame stamp alpha so overlapping colours ACCUMULATE smoothly
  // (mix) instead of flickering: compaction reorders particles each frame, so a
  // high-alpha order-dependent alpha-over makes a two-colour texel oscillate
  // (A-over-B vs B-over-A). A small alpha makes the order variance negligible
  // and also keeps concentration/blending visible (no instant saturation).
  float depositStrength = 0.03f;  // stamp alpha = concentration * this
  float depositHeight = 0.08f;  // deposit when pos.y >= -this (near floor y=0)
  float depositRadius = 0.02f;  // canvas stamp radius in WORLD units (M6)
  // Drying makes a near-floor particle freeze in place (paint setting): its
  // velocity is scaled toward 0 as wetness (pos.w) -> 0. 0 = no freeze, 1 =
  // fully tie motion to wetness.
  float drySettle = 1.0f;
  // Particle point-sprite size scale (GlobalUbo.renderParams.x). <1 shrinks the
  // rendered live-particle discs.
  float pointScale = 0.5f;

  // ---- particle SSBO ----
  static constexpr uint32_t kMaxParticles = 1u << 16;  // 65536
  // M6-C-2: the exact live count now lives on the GPU (liveCountBuffers). The
  // host counters below are conservative UPPER BOUNDS used only to size
  // dispatches / the point-draw; correctness comes from the in-shader liveCount
  // guards. numParticles = this frame's dispatch & draw upper bound;
  // prevParticleCount = upper bound on the prev buffer's live count (predict's
  // compaction dispatch size). Both are >= the true counts by construction.
  uint32_t numParticles = 0;
  uint32_t prevParticleCount = 0;

  // ---- live count / compaction (M6-C-2) ----
  // liveCountBuffers[i]: a uint[1] storage buffer holding the # of packed live
  // particles for slot i. Reset to 0 at the top of each frame's compute
  // (binding 9); pbf_predict (compaction) + emit atomicAdd into it; every other
  // PBF pass guards on it. The PREV slot's buffer is bound at binding 10
  // (read-only) so pbf_predict reads the exact valid range of the prev
  // (ping-pong) buffer without a host readback -- mirrors the particle 0/7
  // ping-pong. liveCount never crosses queues (compute reads/writes only).
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> liveCountBuffers;
  // Host-visible per-slot copies (one tiny uint, filled once per frame) for
  // ImGui display + to shrink the dispatch upper bound. NOT used for
  // correctness (the GPU guards handle that), only to keep dispatch tight.
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> liveCountReadbackBuffers;
  std::vector<uint8_t> liveReadbackValid;    // per slot: a copy has been made
  std::vector<uint64_t> liveCopyCumEmitted;  // cumEmitted snapshot at the copy
  uint64_t cumEmitted = 0;        // all-time emitted particle count (monotonic)
  uint32_t emittedThisFrame = 0;  // reset each frame; feeds the dispatch bound
  uint32_t liveCountDisplay = 0;  // last read-back live count (ImGui/debug)
  // Readback-derived upper bound on the live count (kMax until the first
  // readback). live_now <= R + emittedSince, so this is a valid bound that also
  // shrinks as particles dry out -> keeps numParticles from pegging at kMax.
  uint32_t liveUpperFromReadback = kMaxParticles;
  void createLiveCountBuffers();
  void consumeLiveCountReadback();
  // per-frame device-local SSBOs (written by compute, read as vertex buffer).
  // Stepped as a ping-pong chain (read prev, write cur) like particle.cpp, so
  // the two in-flight buffers form ONE evolving sim (no per-buffer divergence).
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> particleBuffers;

  // particle debug renderer (point list)
  vk::raii::Pipeline particlePipeline = nullptr;
  // spoid markers: larger, bordered discs so they stand out from the fluid
  vk::raii::Pipeline markerPipeline = nullptr;
  bool showParticles = true;

  // Per-frame-index flag: the very first compute dispatch on each particle
  // buffer must NOT acquire queue ownership (no graphics release precedes it),
  // else validation reports an unmatched ownership-acquire barrier at startup.
  std::vector<uint8_t> computeFirstUse;

  // ---- sim params (M3) ----
  float gravity = 6.0f;       // world units/s^2, +Y (down on screen)
  bool useFixedDt = true;     // fixed dt avoids frame-time spikes destabilizing
  float seedJitterXZ = 0.0f;  // random horizontal seed velocity (0 = clean dam)
  bool restartRequested = false;
  static constexpr float kFixedDt = 1.0f / 120.0f;
  static constexpr float kDomainHeight = 3.0f;  // floor y=0 .. cmin.y=-height
  // M4 test: confine the fluid (collision walls + neighbor grid) to a small box
  // in x,z so a dam-break forms a dense, visible pool. The canvas quad itself
  // stays kCanvasWorld.
  static constexpr float kFluidHalf = 1.0f;
  // M5: droplets need the full canvas footprint, so the simulation domain
  // (collision walls + neighbor grid) spans the whole canvas in x,z.
  static constexpr float kDomainHalf = 2.0f;  // == kCanvasWorld * 0.5

  // ---- emit / spoids (M5) ----
  std::vector<Spoid> spoids;
  std::unique_ptr<SpoidController> spoidController;
  static constexpr uint32_t kMaxSpoids = 16;
  // Per-frame host-visible marker buffers (Particle stride) so the spoids can
  // be drawn as points with the existing particle pipeline. Not part of the
  // compute<->graphics ping-pong (host-written each frame).
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> markerBuffers;
  bool showSpoids = true;
  bool spaceWasDown = false;   // edge-detect the emit key
  int selectedSpoidUi = 0;     // which spoid the ImGui param sliders edit
  bool editAllSpoids = false;  // edit mode: apply param edits to ALL spoids
  // Task 9: hardcoded auto-drop (a single fixed emitter) to verify the emit
  // pass before the spoid UI exists; off by default now that spoids drive it.
  bool autoEmit = true;  // default-on: spoids auto-drop so the scene is alive
  float autoEmitInterval = 0.6f;  // seconds between auto drops
  float autoEmitTimer = 0.f;
  // Particle pool exhaustion warning (set when a drop is rejected/clamped).
  bool poolFull = false;

  // ---- graphics sync semaphores (signalled by graphics, waited by compute)
  // ----
  struct {
    uint32_t queueFamilyIndex = 0;
    std::vector<vk::raii::Semaphore> semaphores;
  } graphics;

  // ---- compute ----
  struct {
    uint32_t queueFamilyIndex = 0;
    vk::raii::Queue queue = nullptr;
    vk::raii::CommandPool cmdPool = nullptr;
    vk::raii::CommandBuffers cmdBuffers = nullptr;
    std::vector<vk::raii::Semaphore> semaphores;

    // per-frame uniform buffers (ComputeUbo)
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> uniformBuffers;
    ComputeUbo ubo{};

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    // PBF pipeline stages (all share descriptorSetLayout/pipelineLayout)
    vk::raii::Pipeline pipeline = nullptr;  // predict (compaction, once/frame)
    vk::raii::Pipeline integrate =
        nullptr;  // M8: per-substep force integration
    vk::raii::Pipeline gridCount = nullptr;
    vk::raii::Pipeline gridScan = nullptr;
    vk::raii::Pipeline gridScatter = nullptr;
    vk::raii::Pipeline lambda = nullptr;
    vk::raii::Pipeline delta = nullptr;
    vk::raii::Pipeline apply = nullptr;
    vk::raii::Pipeline finalize = nullptr;
    // Emit (M5): separate pipeline layout = compute set layout + push constant.
    vk::raii::PipelineLayout emitPipelineLayout = nullptr;
    vk::raii::Pipeline emit = nullptr;
    // Deposit (M6): stamps near-floor particle colour into the canvas image.
    // Uses the shared compute pipelineLayout (binding 8 = canvas storage
    // image). DISPATCHED ON THE GRAPHICS QUEUE (in buildCommandBuffers) so the
    // canvas image stays graphics-owned -- no cross-queue ownership transfer
    // needed.
    vk::raii::Pipeline deposit = nullptr;
  } compute;

  // ---- neighbor grid (per-frame; rebuilt every substep) ----
  // cellCount[numCells], cellStart[numCells+1] (exclusive prefix sum),
  // cellOffset[numCells+1] (mutable copy advanced during scatter),
  // sortedIds[kMaxParticles], deltaP[kMaxParticles] (PBF position correction).
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> cellCountBuffers;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> cellStartBuffers;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> cellOffsetBuffers;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> sortedIdBuffers;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> deltaPBuffers;
  glm::ivec3 gridDim{0};
  uint32_t numCells = 0;

  // ---- PBF tunables (M4) ----
  // h = smoothing radius == grid CELL size. numCells ~= (domain/h)^3, so do NOT
  // shrink h (h=0.01 -> 400^3 = 48M cells -> GB buffers + serial scan stall).
  static constexpr float kSmoothingRadius = 0.1f;  // h (fixes grid size)
  // Rest particle spacing == the spacing droplets emit at, so a fresh droplet
  // sits at rest density (rho ~= rho0 ~= 8078) and the incompressibility +
  // bounded cohesion constraints are actually active (M8). 0.5h is the
  // PBF-standard rest spacing (~33 neighbours within h). (Independent of h;
  // sets rho0 + the emit lattice, not the grid.) NOTE: the old 0.005 (=0.05h)
  // made rho0 ~8e6 >> a droplet's actual density -> both density pressure and
  // scorr evaluated to ~0 -> no cohesion, no crown (see the M8 design spec).
  static constexpr float kParticleSpacing = 0.03f;
  float rho0 = 0.f;  // computed from the rest lattice in prepare()
  // SOFT constraint (M8, reference-grounded): the CPU reference uses
  // epsilon_cfm=1e5, which dominates a typical Sum|grad C|^2 (~660 at our
  // scale) by ~175x -> tiny lambda -> gentle, stable corrections. The old
  // epsCFM=100 was STIFF (<< 660) -> big lambda -> Delta p / sub_dt blew up on
  // emit ("first explosion" / "too fast"). Lower it toward ~1e3 for a stiffer,
  // crisper crown.
  float epsCFM = 1.0e5f;  // CFM relaxation (reference: 1e5)
  // Bounded signed density constraint (M8): under-dense particles get a bounded
  // attractive pull (cohesion) toward rest density; the floor caps it so a
  // sub-monolayer cannot run away. cohesionFloor=1.0 == the reference's
  // UNBOUNDED signed constraint (C never drops below -1 physically), now safe
  // because the soft epsCFM already keeps the pull gentle. 0 = compression-only
  // (no cohesion).
  float cohesionFloor = 1.0f;
  // Per-iteration Δp clamp (pbf_delta), as a multiple of h. <=0 disables it.
  // Default 0.2 preserves the old clamp; lower it toward 0 for a stronger crown
  // once substeps keep the sim stable (M8).
  float dpClampFactor = 0.2f;
  // Artificial pressure / surface tension (Eq.13). Reference-grounded (M8):
  // corr_k = m*1e-4 ~= 3e-6 (unit mass), corr_h = 0.30h, n = 4. With the soft
  // epsCFM, lambda is ~1e-5, so the old scorrK=0.1 would dominate it ~10^4x;
  // 3e-6 keeps scorr a minor surface term as in the reference (which itself
  // flags scorr as weak/heuristic). Raise it for more visible surface tension.
  float scorrK = 3.0e-6f;     // artificial pressure strength (reference ~3e-6)
  float scorrDqRatio = 0.3f;  // scorrDq = ratio * h (reference corr_h = 0.30)
  float scorrN = 4.f;
  float xsphC = 0.05f;  // XSPH viscosity (reference viscosity_coeff = 0.050)
  // Gentle per-substep drag (M8): the reference uses v*=0.999/substep. High
  // values (the old 8.0) dissipate the impact energy that launches a crown and
  // cap terminal velocity (~0.75) -- settling of DEPOSITED paint is drying's
  // job (drySettle), not bulk drag's.
  float velDamp = 0.36f;  // per-second drag; 1 - velDamp*sub_dt ~= 0.999 at
                          // sub_dt=1/360 (reference v *= 0.999 per substep)
  // CFL speed cap maxSpeed = factor * h / dt; <=0 disables it. The principled
  // approach is XPBD "small steps" (many substeps, few iters): with small dt
  // the velocity recovery v=(x*-x)/dt is already physical, so no clamp is
  // needed. Kept only as a coarse safety net for large dt.
  float velClampFactor =
      0.0f;  // off: compression-only constraint keeps it calm
  // Jacobi under-relaxation: apply only this fraction of the SPH position
  // correction per iteration. Damps the parallel-solver overshoot that makes
  // close/piled particles oscillate ("flicker") and softens close-range
  // popping.
  float solverRelax = 0.3f;
  int substeps = 3;  // M8: XPBD small-steps (stability from many small dt,
                     // not from grinding iters at a big dt; reference uses 5)
  int solverIters = 2;  // M8: fewer iters per substep (reference uses 2)
  bool colorByDensity = false;  // debug: tint particles by rho/rho0

  static constexpr float kCanvasWorld = 4.0f;
};

}  // namespace vge
