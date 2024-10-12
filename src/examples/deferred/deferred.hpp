#include "Vge_base.hpp"

//
#include "vgeu_gltf.hpp"
#include "vgeu_texture.hpp"

// std
#include <vector>

namespace vge {

#define MAX_LIGHTS 10
struct Options {
  int32_t debugDisplayarget = 0;
};

struct DynamicUboElt {
  glm::mat4 modelMatrix{1.f};
  // color.alpha used for mix between color.rgb and original color
  glm::vec4 modelColor{0.f};
};

struct UniformDataOffscreen {
  glm::mat4 projection{1.f};
  glm::mat4 view{1.f};
  glm::mat4 model{1.f};
};

struct Light {
  glm::vec4 position;
  glm::vec3 color;
  float radius;
};

struct UniformDataComposition {
  glm::vec4 viewPos;
  Light lights[MAX_LIGHTS];
  int numLights;
  int debugDisplayTarget{0};
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
  virtual void initVulkan();
  virtual void getEnabledExtensions();
  virtual void getEnabledFeatures();
  virtual void render();
  virtual void prepare();
  virtual void viewChanged();
  virtual void setupCommandLineParser(CLI::App& app);
  virtual void onUpdateUIOverlay();

  // to separate cmd line init and restart variable
  void setupCommandLineParser(CLI::App& app, Options& opts);
  // copy option values to member variables
  void initFromOptions();

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

  UniformDataOffscreen UniformDataOffscreen;

  std::vector<DynamicUboElt> dynamicUbo;
  size_t alignedSizeDynamicUboElt = 0;

  UniformDataComposition uniformDataComposition;
  struct {
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> dynamicUniformBuffers;
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> offScreen;
    std::vector<std::unique_ptr<vgeu::VgeuBuffer>> composition;
  } uniformBuffers;

  struct {
    vk::raii::Pipeline offScreen = nullptr;
    // NOTE: check we need separate pipeline
    vk::raii::Pipeline offScreenSimpleMesh = nullptr;

    vk::raii::Pipeline composition = nullptr;
  } pipelines;
  vk::raii::PipelineLayout pipelineLayout = nullptr;

  struct {
    std::vector<vk::raii::DescriptorSet> dynamicUboDescriptorSets;
    std::vector<vk::raii::DescriptorSet> model;
    std::vector<vk::raii::DescriptorSet> floor;
    std::vector<vk::raii::DescriptorSet> composition;
  } descriptorSets;
  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::DescriptorSetLayout dynamicUboDescriptorSetLayout = nullptr;

  struct FrameBuffer {
    uint32_t width, height;
    // TODO: check to duplicate for each frame
    std::vector<vk::raii::Framebuffer> frameBuffers;
    std::vector<std::unique_ptr<vgeu::VgeuImage>> position, normal, albedo;
    std::vector<std::unique_ptr<vgeu::VgeuImage>> depth;
    vk::raii::RenderPass renderPass = nullptr;
  } offScreenFrameBuf;

  // TODO: check to duplicate for each frame
  vk::raii::Sampler colorSampler = nullptr;
  // for each frame
  vk::raii::CommandBuffers offScreenCmdBuffers = nullptr;
  std::vector<vk::raii::Semaphore> offScreenSemaphores;
};
}  // namespace vge