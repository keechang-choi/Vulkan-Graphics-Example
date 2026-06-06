#pragma once

#include "vge_base.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

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
  glm::vec4
      originRadius;   // -- 0  -- xyz spoid origin (y<0); w = lattice spacing
                      // (mode 0) or ball radius=holeRadius (mode 1)
  glm::vec4 velConc;  // -- 16 -- xyz initial velocity (+Y), w concentration
  glm::vec4 color;    // -- 32 -- rgb paint color, a unused
  uint32_t mode;      // -- 48 -- spawn shape: 0 = cube lattice, 1 = ball (M8)
  uint32_t count;     // -- 52 -- particles in this burst
  uint32_t seed;      // -- 56 -- rng seed (varies per burst)
  uint32_t _pad;      // -- 60 --
  glm::vec4 segPrev;  // -- 64 -- ball sweep START (xyz); == origin for a static
                      // burst, = previous hole pos for a moving stream (M8)
  // -- 80 --
};
static_assert(sizeof(EmitPush) == 80, "EmitPush push-constant size");

// Spoid (M5): host-side eyedropper POD. Selected via ImGui, moved via keyboard;
// Space releases one droplet burst of `amount` particles from each selected
// spoid. World convention: spoids live above the floor at y<0.
struct Spoid {
  glm::vec3 pos{0.f, -1.25f, 0.f};  // M8: start height halved (was -2.5)
  float holeRadius = 0.02f;
  glm::vec3 color{0.2f, 0.4f, 0.9f};
  float emissionVelocity = 2.f;  // initial downward (+Y) speed
  int amount = 300;              // particles per drop (burst mode)
  float concentration = 1.f;     // -> stamp alpha (M6) + opacity
  bool selected = true;
  // Stream-mode runtime state (M8, not UI params): prevPos = this hole's
  // position last frame (the sweep start for a continuous stream); emitAccum =
  // fractional particle carry so a non-integer per-frame rate still emits
  // evenly.
  glm::vec3 prevPos{0.f, -1.25f, 0.f};
  float emitAccum = 0.f;
  // --- Phase 2: pendulum attachment (used only by PendulumSpoidController) ---
  int nodeIndex = -1;    // attach node in the chain; -1 = tip (last node)
  float offsetR = 0.1f;  // offset distance in the plane perpendicular to the
                         // link direction at the attach node (0 = on node)
  float offsetAngle0 = 0.f;  // start angle in that perpendicular plane (rad);
                             // set per-spoid by arrangeSpoidsCircle so multiple
                             // spoids sit evenly around the ring
  float offsetOmega = 4.f;   // constant spin rate of the offset (rad/s)
  float offsetPhase = 0.f;   // runtime: angle0 + omega*t, carried across frames
  // runtime: a unit vector in the plane perpendicular to the string, parallel-
  // transported across frames so the rotary-offset basis stays CONTINUOUS as
  // the string swings (a per-frame world-axis pick stutters during a conical
  // swing).
  glm::vec3 offsetAxis{1.f, 0.f, 0.f};
  float paintMass = 1.f;  // paint reservoir; decreases on emit, 0 => stop
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

// --- Phase 2: PBD pendulum -------------------------------------------------
// One point mass in a pendulum chain. node[0] is the fixed pivot (invMass 0).
struct PendulumNode {
  glm::vec3 pos{0.f};
  glm::vec3 prevPos{0.f};  // PBD: position at the start of the substep
  glm::vec3 vel{0.f};
  float invMass = 1.f;  // 1/m; pivot = 0 (immovable)
};

// One suspended pendulum assembly (n-link chain). The example owns a vector of
// these; v1 uses a single chain. CONVENTION: bobMass[i] <= 0 => infinite mass
// => invMass 0 => node i+1 is PINNED (immovable).
struct PendulumChain {
  glm::vec3 pivot{0.f, -2.8f, 0.f};  // fixed suspension point (near ceiling)
  std::vector<PendulumNode> nodes;   // [0]=pivot, [1..numLinks]=bobs
  std::vector<float> linkLength;  // rest |node_i - node_{i-1}|, i=1..numLinks
  // config (UI-editable):
  int numLinks = 1;            // n: 1 = simple (default), 2 = double pendulum
  float totalLength = 2.0f;    // sum of links (split uniformly into n)
  std::vector<float> bobMass;  // per bob; built on reset, default uniform 1.0
  float airDamping = 0.f;      // global velocity damping /s (air resistance)
  float jointDamping = 0.f;    // damping of along-link relative velocity
  int substeps = 8;            // XPBD small-steps
  int iters = 4;               // distance-constraint Gauss-Seidel iters/substep
  // initial state (chain starts as a straight line displaced from +Y vertical):
  float initTheta = 1.3f;  // rad from the +Y (down) axis
  float initPhi = 0.5f;    // rad azimuth in X-Z
  // initial tip velocity, split into the two tangent directions of the bob's
  // sphere at the start pose (both perpendicular to the string, world u/s):
  //  - radial  = meridional e_theta (swings in the vertical plane,
  //  toward/through
  //              the bottom centre) -> a normal back-and-forth swing.
  //  - tangential = azimuthal e_phi (circles around the +Y axis) -> a conical /
  //                 precessing swing that pairs with the rotary offset.
  float initSpeedRadial = 0.f;
  float initSpeedTangential = 2.0f;

  // --- Blackburn (Lissajous) pendulum ---------------------------------------
  // Two fixed anchors A,B + a triangulated junction J (node[1]) + a single bob
  // (node[2]). The junction is rigidly triangulated in the A-B plane (so the
  // bob swings about J with the SHORT length lowerLen there) but free to swing
  // about the A-B line perpendicular to it (LONG length vDepth+lowerLen) -> two
  // different frequencies in perpendicular axes => Lissajous. Pure PBD: the two
  // upper distance constraints (A->J, B->J) produce the anisotropy. Frequency
  // ratio ~ sqrt((vDepth+lowerLen)/lowerLen).
  bool blackburn = false;
  float anchorSep = 1.0f;  // 2s: horizontal gap between anchors A,B (along X)
  float vDepth = 0.5f;     // h: junction depth below the anchor line
  float lowerLen = 1.5f;   // L: junction -> bob
  glm::vec3 pivotB{0.f};   // anchor B (computed on reset; A = `pivot`-derived)
  float upperLenB = 0.f;   // rest |J - B| (computed on reset)
};

// Drives spoids along an n-link PBD pendulum. The example owns the chains and a
// gravity value; this controller holds references to them. update() steps the
// PBD chains, advances each spoid's offset phase, and writes spoid.pos = the
// emission point (attach node + rotating perpendicular offset).
class PendulumSpoidController : public SpoidController {
public:
  PendulumSpoidController(std::vector<PendulumChain>& chains,
                          const float& gravity, const float& ceilingHeight)
      : chains(chains), gravity(gravity), ceilingHeight(ceilingHeight) {}
  std::vector<PendulumChain>& chains;
  const float& gravity;
  // = the example's kDomainHeight (scaled). Lower bound for the bob y-clamp so
  // the safety clamp tracks the world scale instead of a hard-coded floor.
  const float& ceilingHeight;
  void update(float dt, std::vector<Spoid>& spoids, const InputState& in,
              std::vector<int>& emitDrops) override;

private:
  void stepChain(PendulumChain& c, float dt);  // PBD (filled in a later task)
  // non-const Spoid&: parallel-transports s.offsetAxis to keep the basis
  // smooth.
  glm::vec3 emissionPoint(const PendulumChain& c, Spoid& s) const;
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
  // `segPrev` is the ball-sweep start (M8): pass it != origin to sweep the
  // spawn ball along [segPrev -> origin] for a continuous moving stream; pass
  // it == origin (or omit) for a static burst at `origin`.
  void enqueueDrop(const glm::vec3& origin, float holeRadius,
                   const glm::vec3& color, float emissionVel,
                   float concentration, int amount, const glm::vec3& segPrev);
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
  // Canvas accumulation-texture resolution. Allocated once at startup -> a
  // start-time-only setting (no live slider); set via --canvas-res.
  uint32_t kCanvasTexRes = 2048;
  // deposit params (M6-B); live in ComputeUbo pads.
  float dryRate = 2.0f;  // wetness lost per second while depositing
  // Small per-frame stamp alpha so overlapping colours ACCUMULATE smoothly
  // (mix) instead of flickering: compaction reorders particles each frame, so a
  // high-alpha order-dependent alpha-over makes a two-colour texel oscillate
  // (A-over-B vs B-over-A). A small alpha makes the order variance negligible
  // and also keeps concentration/blending visible (no instant saturation).
  float depositStrength = 0.01f;  // stamp alpha = concentration * this
  float depositHeight = 0.08f;  // deposit when pos.y >= -this (near floor y=0)
  float depositRadius = 0.01f;  // canvas stamp radius in WORLD units (M6)
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
  // Domain extents. Phase 2: NON-const, recomputed = base * worldScale in
  // computeScaledDims() so the whole scene scales relative to the fixed-size
  // particles. Base values are the literals in computeScaledDims().
  float kDomainHeight = 3.0f;  // floor y=0 .. cmin.y=-height
  // M4 test: confine the fluid (collision walls + neighbor grid) to a small box
  // in x,z so a dam-break forms a dense, visible pool. The canvas quad itself
  // stays kCanvasWorld.
  static constexpr float kFluidHalf = 1.0f;
  // M5: droplets need the full canvas footprint, so the simulation domain
  // (collision walls + neighbor grid) spans the whole canvas in x,z.
  float kDomainHalf = 2.0f;  // == kCanvasWorld * 0.5 (scaled by worldScale)

  // ---- emit / spoids (M5) ----
  std::vector<Spoid> spoids;
  std::unique_ptr<SpoidController> spoidController;
  // --- Phase 2: pendulum -----------------------------------------------------
  enum class SpoidControlMode { Keyboard, Pendulum };
  SpoidControlMode spoidControlMode = SpoidControlMode::Pendulum;
  std::vector<PendulumChain> pendulumChains;  // owned here; controller mutates
  static constexpr uint32_t kMaxChainNodes = 32;  // pivot + up to 31 bobs
  bool showChain = true;          // draw link lines + joint sprites
  float chainLineWidth = 2.f;     // link + spoid-arm line thickness (px)
  bool wideLinesEnabled = false;  // device supports lineWidth > 1 (set on init)
  // Per-frame host-visible buffers for chain visualization (Particle stride, so
  // they reuse the marker/line pipelines' vertex input). jointMarkerBuffers:
  // one point per node. lineBuffers: two points per link (eLineList).
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> jointMarkerBuffers;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> lineBuffers;
  vk::raii::Pipeline linePipeline = nullptr;  // eLineList, chain_line shaders
  void setSpoidControlMode(SpoidControlMode mode);
  void resetPendulum();  // (re)build pendulumChains[0] from config + init
  void createChainBuffers();
  void createLinePipeline();
  // --- Phase 2: top-view camera animation ---
  struct CameraAnim {
    bool active = false;  // an eye/target/up tween is in progress
    bool locked = false;  // holding the top view after the forward tween
    bool toTop = true;    // direction: true = going to top, false = returning
    float t = 0.f, duration = 1.0f;
    // Pose = eye + orientation quaternion. The orientation is SLERP-ed (uniform
    // angular rate) while the eye eases, so the camera rotates smoothly
    // together with the move instead of swinging hard at the end (which
    // separate eye/target/up lerps caused).
    glm::vec3 fromEye{0.f}, toEye{0.f};
    glm::quat fromQuat{1.f, 0.f, 0.f, 0.f}, toQuat{1.f, 0.f, 0.f, 0.f};
  } cameraAnim;
  void startTopViewAnim();
  void updateCameraAnim();
  // current camera eye + orientation quaternion (from the live inverse-view).
  void currentCameraEyeQuat(glm::vec3& eye, glm::quat& quat) const;
  // orientation quaternion for a look-at pose (matches setViewTarget's basis).
  glm::quat lookQuat(const glm::vec3& eye, const glm::vec3& target,
                     const glm::vec3& up) const;
  static constexpr uint32_t kMaxSpoids = 16;
  // Per-frame host-visible marker buffers (Particle stride) so the spoids can
  // be drawn as points with the existing particle pipeline. Not part of the
  // compute<->graphics ping-pong (host-written each frame).
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> markerBuffers;
  bool showSpoids = true;
  bool spaceWasDown = false;  // edge-detect the emit key
  int selectedSpoidUi = 0;    // which spoid the ImGui param sliders edit
  bool editAllSpoids = true;  // edit mode: apply param edits to ALL spoids
  // arrangeSpoidsCircle radius, as a fraction of kDomainHalf (UI-tunable). 0.3
  // reproduces the original hardcoded compact ring near the centre; >1 spreads
  // the spoids past the canvas edge.
  float spoidArrangeRadiusFrac = 0.3f;
  // Task 9: hardcoded auto-drop (a single fixed emitter) to verify the emit
  // pass before the spoid UI exists; off by default now that spoids drive it.
  bool autoEmit = true;  // default-on: spoids auto-drop so the scene is alive
  float autoEmitInterval = 0.6f;  // seconds between auto drops
  float autoEmitTimer = 0.f;
  // Spawn shape (M8): true = uniform ball of radius holeRadius (holeRadius
  // drives droplet size); false = jittered cube lattice at rest spacing (size
  // from amount). Ball became viable once the soft epsCFM stopped close-pair
  // pops.
  bool sphericalSpawn = true;
  // Continuous STREAM mode (M8): instead of discrete bursts on a timer, emit
  // streamRate particles/sec every frame from each selected spoid's hole, swept
  // along the hole's motion this frame (so a fast-moving stroke stays connected
  // instead of breaking into dots). No physical tank -- the soft-constraint
  // fluid + gravity form the falling ribbon. Takes over from autoEmit when on.
  bool streamMode = true;    // default ON (pairs with the pendulum default)
  float streamRate = 300.f;  // particles per second per streaming spoid
  // Phase 2: paint reservoir drain per emitted particle (UI-tunable). 0 = the
  // reservoir never depletes (paint never runs out).
  float massDrainRate = 0.f;
  // Phase 2: physically-motivated outflow. Real flow through a hole follows
  // Torricelli v_exit ~ sqrt(g_eff), and for a swinging bob g_eff = |g - a_bob|
  // with a_bob dominated by the centripetal term v^2/L (up toward the pivot).
  // So outflow peaks at the fast bottom of the swing and dips at the turning
  // points. When on, streamRate is scaled by 1 + flowGain*(sqrt(g_eff/g) - 1).
  bool flowFromDynamics = true;  // default ON (physical outflow)
  float flowGain = 1.f;          // 0 = constant; >1 exaggerates the variation
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
  // Not runtime-tunable (feeds rho0 + the grid at startup) -> a start-time-only
  // setting; set via --spacing.
  float kParticleSpacing = 0.03f;
  float rho0 = 0.f;  // computed from the rest lattice in prepare()
  // SOFT constraint (M8, reference-grounded): the CPU reference uses
  // epsilon_cfm=1e5, which dominates a typical Sum|grad C|^2 (~660 at our
  // scale) by ~175x -> tiny lambda -> gentle, stable corrections. The old
  // epsCFM=100 was STIFF (<< 660) -> big lambda -> Delta p / sub_dt blew up on
  // emit ("first explosion" / "too fast"). Lower it toward ~1e3 for a stiffer,
  // crisper crown.
  float epsCFM = 1.0e5f;  // CFM relaxation (reference: 1e5); used when
                          // xpbdCompliance is OFF (fixed PBF behaviour)
  // XPBD: when on, the constraint coefficient is alpha/dt^2 (substep dt)
  // instead of the fixed epsCFM, so the effective stiffness -- and the
  // spawn-relief ejection velocity -- no longer depend on the substep count.
  // complianceXPBD (alpha) ~= epsCFM * dt^2 at the default 3 substeps (1/360
  // s), so the default look is unchanged but it stays consistent as `substeps`
  // changes.
  bool xpbdCompliance = false;
  float complianceXPBD = 0.77f;
  // Bounded signed density constraint (M8): under-dense particles get a bounded
  // attractive pull (cohesion) toward rest density; the floor caps it so a
  // sub-monolayer cannot run away. cohesionFloor=1.0 == the reference's
  // UNBOUNDED signed constraint (C never drops below -1 physically), now safe
  // because the soft epsCFM already keeps the pull gentle. 0 = compression-only
  // (no cohesion).
  float cohesionFloor = 2.0f;  // raised for stronger droplet cohesion
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
  float xsphC = 0.15f;  // XSPH viscosity (raised for more coherent motion)
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
  int substeps = 1;  // M8: XPBD small-steps (stability from many small dt,
                     // not from grinding iters at a big dt; reference uses 5)
  int solverIters = 1;  // M8: fewer iters per substep (reference uses 2)
  bool colorByDensity = false;  // debug: tint particles by rho/rho0

  float kCanvasWorld = 4.0f;  // canvas side length (scaled by worldScale)

  // --- Phase 2: world scale --------------------------------------------------
  // Multiplies the canvas + sim domain + grid + pendulum geometry, NOT the
  // particle spacing / smoothing radius h / rho0 -- so the M8 fluid tuning is
  // preserved and particles simply look smaller on a bigger canvas. Applied on
  // Restart. Grid buffers are pre-sized for kMaxWorldScale so no live buffer
  // realloc / descriptor rewrite is needed (only gridDim/numCells change).
  static constexpr float kMaxWorldScale = 4.0f;
  float worldScale = 1.5f;         // UI knob; takes effect on Restart
  float appliedWorldScale = 1.5f;  // scale currently baked into the geometry
  uint32_t maxNumCells = 0;        // grid buffer capacity (sized for max scale)
  // Recompute kCanvasWorld/kDomainHalf/kDomainHeight (= base * worldScale) and
  // gridDim/numCells/maxNumCells. Pure CPU; does not touch GPU resources.
  void computeScaledDims();
};

}  // namespace vge
