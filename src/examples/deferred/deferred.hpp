#include "Vge_base.hpp"

//
#include "vgeu_gltf.hpp"
#include "vgeu_texture.hpp"

// std
#include <vector>

namespace vge {

#define MAX_LIGHTS 10
struct Options {
  float moveSpeed = 10.f;
  // save camera view. not configurable by panel
  glm::mat4 cameraView{1.f};
  int32_t debugDisplayTarget = 0;
  int32_t numTargets = 10;
  float farClamp = 50.f;
  int32_t modelNumX = 4;
  int32_t modelNumZ = 4;
  float spacingX = 4.f;
  float spacingZ = 4.f;
};

struct DynamicUboElt {
  glm::mat4 modelMatrix{1.f};
  // color.alpha used for mix between color.rgb and original color
  glm::vec4 modelColor{0.f};
};

struct UniformDataOffscreen {
  glm::mat4 projection{1.f};
  glm::mat4 view{1.f};
  // glm::mat4 model{1.f};
};

struct Light {
  glm::vec4 position;
  glm::vec3 color;
  float radius;
};

// NOTE: for alignment. default size: 352bytes = 32*10 + 16 + 4 + 4 + 4 + 4
struct alignas(64) UniformDataComposition {
  Light lights[MAX_LIGHTS];
  glm::vec4 viewPos;
  int debugDisplayTarget{0};
  int numLights;
  float nearPlane;
  float farPlane;
  float farClamp;
};
struct VertexInfos {
  vk::PipelineVertexInputStateCreateInfo vertexInputSCI;
  std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
  std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
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
  ModelInstance() {};
  ModelInstance(const ModelInstance& o) = delete;
  ModelInstance& operator=(const ModelInstance& other) = delete;
  ModelInstance(ModelInstance&& other);
  ModelInstance& operator=(ModelInstance&& other);
};

class VgeExample : public VgeBase {
 public:
  VgeExample();
  ~VgeExample();
  // to separate cmd line init and restart variable
  virtual void setupCommandLineParser(CLI::App& app);
  void setOptions(const std::optional<Options>& opts);

  virtual void initVulkan();
  virtual void getEnabledExtensions();
  virtual void getEnabledFeatures();
  virtual void prepare();
  virtual void render();

  virtual void viewChanged();
  virtual void onUpdateUIOverlay();

  void loadAssets();
  void setupDynamicUbo();
  void prepareOffScreenFrameBuffer();
  std::unique_ptr<vgeu::VgeuImage> createAttachment(vk::Format format,
                                                    vk::ImageUsageFlags usage);
  void prepareUniformBuffers();
  void setupDescriptors();
  void preparePipelines();

  void updateUboComposition();
  void updateUboOffScreen();
  void buildCommandBuffers();

  void draw();

  void addModelInstance(ModelInstance&& newInstance);
  const std::vector<size_t>& findInstances(const std::string& name);

  Options opts{};

  struct Textures {
    std::unique_ptr<vgeu::Texture2D> colorMap;
    std::unique_ptr<vgeu::Texture2D> normalMap;
  };
  struct {
    Textures model;
    Textures floor;
  } textures;

  std::vector<ModelInstance> modelInstances;
  // saves both index for corresponding model and simple model
  std::unordered_map<std::string, std::vector<size_t>> instanceMap;

  UniformDataOffscreen uniformDataOffscreen;

  std::vector<DynamicUboElt> dynamicUbo;
  size_t alignedSizeDynamicUboElt = 0;

  UniformDataComposition uniformDataComposition;
  struct UniformBuffers {
    std::unique_ptr<vgeu::VgeuBuffer> dynamic;
    std::unique_ptr<vgeu::VgeuBuffer> offScreen;
    std::unique_ptr<vgeu::VgeuBuffer> composition;
  };
  std::vector<UniformBuffers> uniformBuffers;

  struct {
    vk::raii::Pipeline offScreen = nullptr;
    // NOTE: check we need separate pipeline
    vk::raii::Pipeline offScreenSimpleMesh = nullptr;

    vk::raii::Pipeline composition = nullptr;
    std::vector<vk::raii::Pipeline> displayTargets;
  } pipelines;
  vk::raii::PipelineLayout pipelineLayoutOffScreen = nullptr;
  vk::raii::PipelineLayout pipelineLayoutCompoisition = nullptr;

  struct SpecializationData {
    uint32_t displayTargetIndex;
  };
  struct {
    // std::vector<vk::raii::DescriptorSet> model;
    // std::vector<vk::raii::DescriptorSet> floor;
    std::vector<vk::raii::DescriptorSet> dynamicUboDescriptorSets;
    std::vector<vk::raii::DescriptorSet> offScreenUboDescriptorSets;
    std::vector<vk::raii::DescriptorSet> composition;
  } descriptorSets;
  vk::raii::DescriptorSetLayout compositionDescriptorSetLayout = nullptr;
  vk::raii::DescriptorSetLayout offScreenUboDescriptorSetLayout = nullptr;
  vk::raii::DescriptorSetLayout dynamicUboDescriptorSetLayout = nullptr;

  struct FrameBuffer {
    uint32_t width, height;
    // TODO: check to duplicate for each frame
    std::vector<vk::raii::Framebuffer> frameBuffers;
    std::vector<std::unique_ptr<vgeu::VgeuImage>> position, normal, albedo, arm,
        emissive;
    std::vector<std::unique_ptr<vgeu::VgeuImage>> depth;
    const size_t numAttachments = 6;
    vk::raii::RenderPass renderPass = nullptr;
    std::vector<bool> isFirstFrame;
  } offScreenFrameBuf;

  // TODO: check to duplicate for each frame
  vk::raii::Sampler colorSampler = nullptr;
  // for each frame.
  // NOTE(kcchoi): use same cmd buffers and
  // implicit synchronization by subpass dependencies.
  // vk::raii::CommandBuffers offScreenCmdBuffers = nullptr;
  // std::vector<vk::raii::Semaphore> offScreenSemaphores;
};
}  // namespace vge