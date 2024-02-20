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

// NOTE: for current animation implementation,
// each instance need its own uniformBuffers
struct ModelInstance {
  std::shared_ptr<vgeu::glTF::Model> model;
  std::shared_ptr<SimpleModel> simpleModel;
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

struct DynamicUboElt {
  glm::mat4 modelMatrix{1.f};
  // color.alpha used for mix between color.rgb and original color
  glm::vec4 modelColor{0.f};
};

struct Particle {
  glm::dvec4 pos;
  glm::dvec4 vel;
  glm::dvec4 prevPos;
};

struct SpecializationData {
  uint32_t sharedDataSize;
  uint32_t integrator;
  uint32_t integrateStep;
  uint32_t localSizeX;
};

struct Options {
  int32_t numParticles{1};
  float coefficientDeltaTime = 0.05f;
  float gravity = 10.f;
  float power = 1.f;
  float soften = 0.001f;
  int32_t integrator = 1;
  float moveSpeed = 10.f;
  float lineWidth = 2.0f;
  float pointSize[2] = {1.f, 128.f};
  int32_t desiredSharedDataSize = 256u;
  float animationSpeed = 0.5f;
  float restitution = 1.0f;
  std::vector<bool> enableSimulation;
  bool computeModelAnimation{false};
  int32_t numSubsteps = 10;

  // save camera view. not configurable by pannel
  glm::mat4 cameraView{1.f};
  float edgeCompliance = 0.1f;
  float areaCompliance = 0.001f;

  float lengthStiffness = 0.001;
  float compressionStiffness = 0.001;
  float stretchStiffness = 0.000;
  float collisionStiffness = 0.01;
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

  void loadAssets();
  void createDescriptorPool();

  void createVertexSCI();
  void createStorageBuffers();
  void createUniformBuffers();

  // graphics resources
  void prepareGraphics();
  void createDescriptorSetLayout();
  void createDescriptorSets();
  void createPipelines();

  // compute resources
  void prepareCompute();


  struct {
    uint32_t queueFamilyIndex;
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> storageBuffers;
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> uniformBuffers;
    vk::raii::Queue queue = nullptr;
    vk::raii::CommandPool cmdPool = nullptr;
    vk::raii::CommandBuffers cmdBuffers = nullptr;
    std::vector<vk::raii::Semaphore> semaphores;

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

    // each frames in flight, each model
    std::vector<std::vector<std::unique_ptr<vgeu::VgeuBuffer>>>
        animatedVertexBuffers;

    // for compute animation
    vk::raii::Pipeline pipelineModelAnimate = nullptr;
    struct computeUbo {
      glm::vec4 clickData;
      float dt;
      uint32_t particleCount;
    } ubo;
    std::vector<bool> firstCompute;
  } compute;

  void draw();
  void buildCommandBuffers();
  void buildComputeCommandBuffers();

  void setupDynamicUbo();
  void updateGraphicsUbo();
  void updateComputeUbo();
  void updateDynamicUbo();

  void addModelInstance(ModelInstance&& newInstance);
  const std::vector<size_t>& findInstances(const std::string& name);

  void setOptions(const std::optional<Options>& opts);

  // NOTE: for simple Model
  VertexInfos simpleVertexInfos;
  // for animated vertex
  VertexInfos animatedVertexInfos;
  struct {
    // NOTE: movable element;
    uint32_t queueFamilyIndex;
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> globalUniformBuffers;
    GlobalUbo globalUbo;
    std::vector<vk::raii::DescriptorSet> globalUboDescriptorSets;
    vk::raii::DescriptorSetLayout globalUboDescriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;

    vk::raii::Pipeline pipelinePhong = nullptr;
    vk::raii::Pipeline pipelineSimpleMesh = nullptr;
    vk::raii::Pipeline pipelineWireMesh = nullptr;
    vk::raii::Pipeline pipelineSimpleLine = nullptr;

    std::vector<vk::raii::Semaphore> semaphores;
  } graphics;

  std::vector<ModelInstance> modelInstances;
  // saves both index for corresponding model and simple model
  std::unordered_map<std::string, std::vector<size_t>> instanceMap;

  std::vector<DynamicUboElt> dynamicUbo;
  size_t alignedSizeDynamicUboElt = 0;
  std::vector<std::unique_ptr<vgeu::VgeuBuffer>> dynamicUniformBuffers;
  std::vector<vk::raii::DescriptorSet> dynamicUboDescriptorSets;
  vk::raii::DescriptorSetLayout dynamicUboDescriptorSetLayout = nullptr;

  float animationTime = 0.f;
  float animationLastTime = 0.f;
  uint32_t numParticles = 1;
  const uint32_t kMaxNumParticles = 1024u;
  uint32_t integrator = 1u;

  Options opts{};

  uint32_t desiredSharedDataSize = 256u;
  uint32_t sharedDataSize;
};
}  // namespace vge
