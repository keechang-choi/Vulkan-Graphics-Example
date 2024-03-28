#include "vge_base.hpp"

//
#include "vgeu_gltf.hpp"

// std
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace vge {

struct GlobalUbo {
  glm::mat4 projection{1.f};
  glm::mat4 view{1.f};
  glm::mat4 inverseView{1.f};
  glm::vec4 lightPos{0.f};
  // NOTE: alignment
  glm::vec2 screenDim;
  // point min size, max size
  glm::vec2 pointSize{1.f, 64.f};
};

// NOTE: std140 alignment
struct ComputeUbo {
  glm::vec4 clickData;
  glm::vec4 gravity;
  glm::ivec2 particleCount;
  float dt;
  float stiffness;
  float alpha;
  float jacobiScale;
  float thickness;
  float radius;
  float friction;
  uint32_t numSubsteps;
};

struct ComputePushConstantsData {
  glm::uvec2 constraintInfo;
};

// NOTE: simple model for circle, quad, lines
struct SimpleModel {
  SimpleModel(const vk::raii::Device& device, VmaAllocator allocator,
              const vk::raii::Queue& transferQueue,
              const vk::raii::CommandPool& commandPool);

  const vk::raii::Device& device;
  VmaAllocator allocator;
  const vk::raii::Queue& transferQueue;
  const vk::raii::CommandPool& commandPool;

  struct Vertex {
    glm::vec4 pos;
    glm::vec4 normal;
    glm::vec4 color;
    glm::vec2 uv;
  };
  bool isLines = false;
  std::unique_ptr<vgeu::VgeuBuffer> vertexBuffer;
  std::unique_ptr<vgeu::VgeuBuffer> indexBuffer;
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;

  // color.alpha=0.0 for checker board
  // color.alpha=1.0 for no lighting
  void setNgon(uint32_t n, glm::vec4 color, bool useCenter = false);
  void setLineList(const std::vector<glm::vec4>& positions,
                   const std::vector<uint32_t>& indices, glm::vec4 color);
  void createBuffers(const std::vector<SimpleModel::Vertex>& vertices,
                     const std::vector<uint32_t>& indices);
};

class SpatialHash {
 public:
  SpatialHash(const double spacing, const uint32_t maxNumObjects);
  void resetTable();
  // same adding order to addTableEntries()
  void addPos(const std::vector<glm::dvec3>& positions);
  void createPartialSum();
  // same adding order to addPos()
  void addTableEntries(const std::vector<glm::dvec3>& positions);
  void query(const glm::dvec3 pos, int maxCellDistance,
             std::vector<std::pair<uint32_t, uint32_t>>& queryIds);
  void queryTri(const glm::dvec4& aabb,
                std::vector<std::pair<uint32_t, uint32_t>>& queryIds);

 private:
  uint32_t hashDiscreteCoords(const int xi, const int yi, const int zi);
  int discreteCoord(const double coord);
  uint32_t hashPos(const glm::dvec3& pos);

  double spacing;
  uint32_t tableSize;
  std::vector<uint32_t> cellStart;
  // object index, particle index
  std::vector<std::pair<uint32_t, uint32_t>> cellEntries;
  std::vector<uint32_t> separator;
  uint32_t objectIndex;
};

struct DynamicUboElt {
  glm::mat4 modelMatrix{1.f};
  // color.alpha used for mix between color.rgb and original color
  glm::vec4 modelColor{0.f};
};

// NOTE: std430 alignment
// cloth particle
struct ParticleCalculate {
  alignas(16) glm::vec4 prevPos;
  glm::vec4 pos;
  glm::vec4 vel;
  glm::vec4 corr;
};

struct DistConstraint {
  alignas(8) glm::uvec2 constIds;
  float restLength;
};
struct ParticleRender {
  // w component as inv mass
  alignas(16) glm::vec4 pos;
  glm::vec4 normal;
  glm::vec2 uv;
};

/*
compute type:
0-> integrate
1-> solve collision
2-> solve distance constraints
3-> update vel
4-> update mesh
5-> initialize? (TODO: optional)
*/
enum class ComputeType {
  kInitializeParticles = 0,
  kInitializeConstraints,
  kIntegrate,
  kSolveCollision,
  kSolveDistanceConstraintsGauss,
  kSolveDistanceConstraintsJacobi,
  kAddCorrections,
  kUpdateVel,
  kUpdateMesh
};

struct SpecializationData {
  uint32_t sharedDataSize;
  uint32_t computeType;
  uint32_t localSizeX;
  uint32_t localSizeY;
  uint32_t localSizeZ;
};

struct Options {
  float coefficientDeltaTime = 0.5f;
  float moveSpeed = 10.f;
  float lineWidth = 2.0f;
  float pointSize[2] = {1.f, 128.f};
  int32_t desiredSharedDataSize = 64u;
  float animationSpeed = 0.5f;
  bool renderWireMesh{false};

  // save camera view. not configurable by pannel
  glm::mat4 cameraView{1.f};

  int32_t numSubsteps = 1;
  float gravity = 10.f;
  float collisionRadius = 0.5f;
  float stiffness = 1.f;
  float alpha = 1.f;
  float jacobiScale = 0.2f;
  float thickness = 0.05f;
  float friction = 0.f;
};

// NOTE: ssbo usage alignment
struct AnimatedVertex {
  alignas(16) glm::vec4 pos;
  alignas(16) glm::vec4 normal;
  alignas(16) glm::vec4 color;
  alignas(16) glm::vec4 tangent;
  alignas(8) glm::vec2 uv;
};

struct VertexInfos {
  vk::PipelineVertexInputStateCreateInfo vertexInputSCI;
  std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
  std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
};

// only support grid type now
class Cloth {
 public:
  Cloth(const vk::raii::Device& device, VmaAllocator allocator,
        const vk::raii::Queue& transferQueue,
        const uint32_t transferQueueFamilyIndex,
        const uint32_t computeQueueFamilyIndex,
        const vk::raii::CommandPool& commandPool,
        const vk::raii::DescriptorPool& descriptorPool,
        const vk::raii::DescriptorSetLayout& particleDescriptorSetLayout,
        const vk::raii::DescriptorSetLayout& constraintDescriptorSetLayout,
        const uint32_t framesInFlight);

  // not copyable
  Cloth(const Cloth&) = delete;
  Cloth& operator=(const Cloth&) = delete;
  Cloth(Cloth&&) = default;
  Cloth& operator=(Cloth&&) = default;

  // not mendatory for cloth-cloth constraints only instance
  // combined with resource creation not to store vertex data in host
  void initParticlesData(const std::vector<vgeu::glTF::Vertex>& vertices,
                         const std::vector<uint32_t>& indices,
                         const glm::mat4& translate, const glm::mat4& rotate,
                         const glm::mat4& scale);

  // default grid dist constraints using number of particles in each axis.
  // assert numParticles.
  void initDistConstraintsData(const uint32_t numX, const uint32_t numY);
  // cloth-cloth constraints or external fixed point
  void initDistConstraintsData(
      const std::vector<DistConstraint>& distConstraints);
  // TODO: dispatch group size
  // integrate and prevPos store
  void integrate(const uint32_t frameIndex,
                 const vk::raii::CommandBuffer& cmdBuffer);
  // bind two?
  // Gauss-Seidel, Jacobi, correction adder -> by specialize
  void solveConstraints(const uint32_t frameIndex,
                        const vk::raii::CommandBuffer& cmdBuffer);
  // vel update
  void updateVel(const uint32_t frameIndex,
                 const vk::raii::CommandBuffer& cmdBuffer);
  // use model->bindSSBO for getting index buffer as storage buffer
  void updateMesh(const uint32_t frameIndex,
                  const vk::raii::CommandBuffer& cmdBuffer);

  void bindVertexBuffer(const vk::raii::CommandBuffer& cmdBuffer,
                        const uint32_t frameIndex) {
    cmdBuffer.bindVertexBuffers(0, renderSBs[frameIndex]->getBuffer(), {0});
  }
  const vgeu::VgeuBuffer* getRenderSBPtr(const uint32_t frameIndex) {
    return renderSBs[frameIndex].get();
  }
  const vgeu::VgeuBuffer* getCalculateSBPtr(const uint32_t frameIndex) {
    return calculateSBs[frameIndex].get();
  }

  const vk::DescriptorSet getParticleDescriptorSet(const uint32_t frameIndex) {
    return *particleDescriptorSets[frameIndex];
  }

  const vk::DescriptorSet getConstraintDescriptorSet() {
    return *constraintDescriptorSet;
  }

  const uint32_t getNumParticles() { return numParticles; }
  const uint32_t getNumTriangles() { return numTris; }
  const uint32_t getNumConstraints() { return numConstraints; }
  const uint32_t getNumPasses() { return numPasses; }
  const uint32_t getPassSize(uint32_t passIndex) {
    assert(passIndex < numPasses);
    return passSizes[passIndex];
  }
  const bool isPassIndependent(uint32_t passIndex) {
    assert(passIndex < numPasses);
    return passIndependent[passIndex];
  }
  const glm::mat4 getInitialTransform() { return initialTransform; }

 private:
  void createParticleStorageBuffers(
      const std::vector<ParticleCalculate>& particlesCalculate);
  void createParticleDescriptorSets();

  void createDistConstraintStorageBuffers(
      const std::vector<DistConstraint>& distConstraints);
  void createDistConstraintDescriptorSets();

  const vk::raii::Device& device;
  VmaAllocator allocator;
  const vk::raii::Queue& transferQueue;
  const uint32_t transferQueueFamilyIndex;
  const uint32_t computeQueueFamilyIndex;
  const vk::raii::CommandPool& commandPool;
  const vk::raii::DescriptorPool& descriptorPool;
  const vk::raii::DescriptorSetLayout& particleDescriptorSetLayout;
  const vk::raii::DescriptorSetLayout& constraintDescriptorSetLayout;
  const uint32_t framesInFlight;

  uint32_t numParticles;
  uint32_t numConstraints;
  uint32_t numTris;

  glm::mat4 initialTransform{1.f};

  // used in both of the pipelines
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> calculateSBs;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> renderSBs;
  // TODO: use model's compute only vertex and index buffer
  // to initialize storage buffers

  // initial setting
  std::unique_ptr<vgeu::VgeuBuffer> constraintSBs;
  // descriptor sets
  // set 1 -> cal sb, render sb
  // set 2 -> constraints
  // not explicitly created
  // set 3 -> other?? cal sb, render sb
  std::vector<vk::raii::DescriptorSet> particleDescriptorSets;
  vk::raii::DescriptorSet constraintDescriptorSet = nullptr;

  bool hasParticleBuffer{false};
  bool hasConstraintBuffer{false};

  uint32_t numPasses;
  std::vector<bool> passIndependent;
  std::vector<uint32_t> passSizes;
};

// NOTE: for current animation implementation,
// each instance need its own uniformBuffers
struct ModelInstance {
  std::shared_ptr<vgeu::glTF::Model> model;
  std::shared_ptr<SimpleModel> simpleModel;
  std::unique_ptr<Cloth> clothModel;
  std::string name;
  bool isBone = false;
  int animationIndex = -1;
  float animationTime = 0.f;
  // initial offset and scale
  vgeu::TransformComponent transform;
  uint32_t getVertexCount() const;
  ModelInstance(){};
  ModelInstance(const ModelInstance& o) = delete;
  ModelInstance& operator=(const ModelInstance& other) = delete;
  ModelInstance(ModelInstance&& other);
  ModelInstance& operator=(ModelInstance&& other);
};

class VgeExample : public VgeBase {
 public:
  VgeExample();
  ~VgeExample();
  virtual void initVulkan();
  virtual void getEnabledExtensions();
  virtual void render();
  virtual void prepare();
  virtual void viewChanged();
  virtual void setupCommandLineParser(CLI::App& app);
  virtual void onUpdateUIOverlay();

  // to separate cmd line init and restart variable
  void setupCommandLineParser(CLI::App& app, Options& opts);
  // copy option values to member variables
  void initFromOptions();

  // common resources
  void prepareCommon();
  void createDescriptorPool();
  void loadAssets();
  void createVertexSCI();
  void createStorageBuffers();
  // NOTE: require UBO to be set
  void createUniformBuffers();
  void createDescriptorSetLayout();
  void createDescriptorSets();
  void initClothModels();

  // compute resources
  void prepareCompute();
  void createComputeUniformBuffers();
  void createComputeStorageBuffers();
  void createComputeDescriptorSetLayout();
  void createComputeDescriptorSets();
  void createComputePipelines();

  // graphics resources
  void prepareGraphics();
  void createGraphicsUniformBuffers();
  void createGraphicsDescriptorSetLayout();
  void createGraphicsDescriptorSets();
  void createGraphicsPipelines();

  void draw();
  void buildCommandBuffers();
  void buildComputeCommandBuffers();

  void setupDynamicUbo();
  void setupClothSSBO();
  void updateGraphicsUbo();
  void updateComputeUbo();
  void updateDynamicUbo();

  void addModelInstance(ModelInstance&& newInstance);
  const std::vector<size_t>& findInstances(const std::string& name);

  void setOptions(const std::optional<Options>& opts);

  // used in allocating descriptor pool
  const uint32_t kMaxNumClothModels = 10u;

  struct {
    // each frames in flight, each model
    std::vector<std::vector<std::unique_ptr<vgeu::VgeuBuffer>>>
        animatedVertexBuffers;

    std::vector<DynamicUboElt> dynamicUbo;
    size_t alignedSizeDynamicUboElt = 0;
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> dynamicUniformBuffers;
    std::vector<vk::raii::DescriptorSet> dynamicUboDescriptorSets;
    vk::raii::DescriptorSetLayout dynamicUboDescriptorSetLayout = nullptr;

    vk::raii::DescriptorSetLayout particleDescriptorSetLayout = nullptr;
    // concurrent frames , buffers
    std::vector<std::vector<const vgeu::VgeuBuffer*>>
        ownershipTransferBufferPtrs;
  } common;

  struct {
    uint32_t queueFamilyIndex;

    vk::raii::Queue queue = nullptr;
    vk::raii::CommandPool cmdPool = nullptr;
    vk::raii::CommandBuffers cmdBuffers = nullptr;
    // strictly, these semaphores commonly used.
    struct Semaphores {
      std::vector<vk::raii::Semaphore> ready;
      std::vector<vk::raii::Semaphore> complete;
    } semaphores;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;
    vk::raii::PipelineLayout pipelineLayout = nullptr;

    vk::raii::DescriptorSetLayout skinDescriptorSetLayout = nullptr;
    // each frames in flight, each model
    std::vector<std::vector<vk::raii::DescriptorSet>> skinDescriptorSets;

    // each model, each skins
    std::vector<std::vector<vgeu::glTF::MeshMatricesData>> skinMatricesData;
    // each frames in flight, each model
    std::vector<std::vector<std::unique_ptr<vgeu::VgeuBuffer>>>
        skinMatricesBuffers;

    struct Pipelines {
      // for compute animation
      vk::raii::Pipeline pipelineModelAnimate = nullptr;
      // cloth simulation calculation and integration
      std::vector<vk::raii::Pipeline> pipelinesCloth;
    } pipelines;

    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> uniformBuffers;
    ComputeUbo ubo;

    std::vector<bool> firstCompute;

    // cloth
    vk::raii::DescriptorSetLayout constraintDescriptorSetLayout = nullptr;
    // frames, clothes
    std::vector<std::vector<const vgeu::VgeuBuffer*>> calculateBufferPtrs;

    ComputePushConstantsData pc;

  } compute;

  struct {
    // NOTE: for simple Model
    VertexInfos simpleVertexInfos;
    // for animated vertex
    VertexInfos animatedVertexInfos;
    // for cloth model
    VertexInfos clothVertexInfos;

    // NOTE: movable element;
    uint32_t queueFamilyIndex;
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> globalUniformBuffers;
    GlobalUbo globalUbo;
    std::vector<vk::raii::DescriptorSet> globalUboDescriptorSets;
    vk::raii::DescriptorSetLayout globalUboDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;

    struct Pipelines {
      vk::raii::Pipeline pipelinePhong = nullptr;
      vk::raii::Pipeline pipelineSimpleMesh = nullptr;
      vk::raii::Pipeline pipelineWireMesh = nullptr;
      vk::raii::Pipeline pipelineSimpleLine = nullptr;
      vk::raii::Pipeline pipelineCloth = nullptr;
      vk::raii::Pipeline pipelineWireCloth = nullptr;
    } pipelines;

  } graphics;

  std::vector<ModelInstance> modelInstances;
  // saves both index for corresponding model and simple model
  std::unordered_map<std::string, std::vector<size_t>> instanceMap;

  float animationTime = 0.f;
  float animationLastTime = 0.f;
  uint32_t numParticles = 1;
  const uint32_t kMaxNumParticles = 1024u;
  uint32_t integrator = 1u;

  Options opts{};

  uint32_t desiredSharedDataSize = 64u;
  uint32_t sharedDataSize;
  uint32_t desiredCollisionWorkGroupSize = 16u;
  uint32_t collisionWorkGroupSize;
};
}  // namespace vge
