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
      0.f};  // -- 192 -- xy=half-extent, z=world size, w unused
  // -- 208 --
};
static_assert(sizeof(GlobalUbo) == 208, "GlobalUbo std140 size");

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
  float _pad1;             // -- 104 --
  float _pad2;             // -- 108 --
  // -- 112 --
};
static_assert(sizeof(ComputeUbo) == 112, "ComputeUbo std140 size");

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

  // ---- particle SSBO ----
  static constexpr uint32_t kMaxParticles = 1u << 16;  // 65536
  uint32_t numParticles = 0;
  // per-frame device-local SSBOs (written by compute, read as vertex buffer)
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> particleBuffers;

  // particle debug renderer (point list)
  vk::raii::Pipeline particlePipeline = nullptr;
  bool showParticles = true;

  // Per-frame-index flag: the very first compute dispatch on each particle
  // buffer must NOT acquire queue ownership (no graphics release precedes it),
  // else validation reports an unmatched ownership-acquire barrier at startup.
  std::vector<uint8_t> computeFirstUse;

  // ---- sim params (M3) ----
  float gravity = 9.8f;       // world units/s^2, +Y (down on screen)
  bool useFixedDt = true;     // fixed dt avoids frame-time spikes destabilizing
  float seedJitterXZ = 0.0f;  // random horizontal seed velocity (0 = clean dam)
  bool restartRequested = false;
  static constexpr float kFixedDt = 1.0f / 120.0f;
  static constexpr float kDomainHeight = 3.0f;  // floor y=0 .. cmin.y=-height
  // M4 test: confine the fluid (collision walls + neighbor grid) to a small box
  // in x,z so a dam-break forms a dense, visible pool. Widened to the canvas in
  // M5 for droplets. The canvas quad itself stays kCanvasWorld.
  static constexpr float kFluidHalf = 1.0f;

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
    vk::raii::Pipeline pipeline = nullptr;  // predict
    vk::raii::Pipeline gridCount = nullptr;
    vk::raii::Pipeline gridScan = nullptr;
    vk::raii::Pipeline gridScatter = nullptr;
    vk::raii::Pipeline lambda = nullptr;
    vk::raii::Pipeline delta = nullptr;
    vk::raii::Pipeline apply = nullptr;
    vk::raii::Pipeline finalize = nullptr;
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
  static constexpr float kSmoothingRadius = 0.1f;   // h (fixes grid size)
  static constexpr float kParticleSpacing = 0.05f;  // ~0.5h
  float rho0 = 0.f;           // computed from the rest lattice in prepare()
  float epsCFM = 100.f;       // CFM relaxation
  float scorrK = 0.1f;        // artificial pressure strength
  float scorrDqRatio = 0.2f;  // scorrDq = ratio * h
  float scorrN = 4.f;
  float xsphC = 0.1f;    // XSPH viscosity (normalized)
  float velDamp = 6.0f;  // global velocity drag (per second) to settle bulk
  int substeps = 1;
  int solverIters = 4;
  bool colorByDensity = false;  // debug: tint particles by rho/rho0

  static constexpr float kCanvasWorld = 4.0f;
};

}  // namespace vge
