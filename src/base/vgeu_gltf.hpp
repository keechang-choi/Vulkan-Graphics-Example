/*

gltf loading class based on tinyglTF

base code from
https://github.com/SaschaWillems/Vulkan/blob/master/base/VulkanglTFModel.h

*/
#pragma once

#include "vgeu_buffer.hpp"
#include "vgeu_flags.hpp"
#include "vgeu_utils.hpp"

// libs
#include "tiny_gltf.h"
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

//
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

// std
#include <limits>
#include <memory>
namespace vgeu {

enum class DescriptorBindingFlagBits : uint32_t {
  kImageBaseColor = 0x00000001,
  kImageNormalMap = 0x00000002
};
using DescriptorBindingFlags = Flags<DescriptorBindingFlagBits>;
template <>
struct FlagTraits<DescriptorBindingFlagBits> {
  static constexpr bool isBitmask = true;
};

enum class FileLoadingFlagBits : uint32_t {
  kNone = 0x00000000,
  kPreTransformVertices = 0x00000001,
  kPreMultiplyVertexColors = 0x00000002,
  kFlipY = 0x00000004,
  kDontLoadImages = 0x00000008
};
using FileLoadingFlags = Flags<FileLoadingFlagBits>;
template <>
struct FlagTraits<FileLoadingFlagBits> {
  static constexpr bool isBitmask = true;
};

enum class RenderFlagBits : uint32_t {
  kBindImages = 0x00000001,
  kRenderOpaqueNodes = 0x00000002,
  kRenderAlphaMaskedNodes = 0x00000004,
  kRenderAlphaBlendedNodes = 0x00000008
};
using RenderFlags = Flags<RenderFlagBits>;
template <>
struct FlagTraits<RenderFlagBits> {
  static constexpr bool isBitmask = true;
};
namespace glTF {

struct Node;

// modified existing structure to fit in RAII paradigm.
struct Texture {
  std::unique_ptr<vgeu::VgeuImage> vgeuImage;
  vk::ImageLayout imageLayout{};
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t mipLevels = 0;
  uint32_t layerCount = 0;
  vk::DescriptorImageInfo descriptorInfo{};
  vk::raii::Sampler sampler = nullptr;

  // fromglTFImage
  Texture(tinygltf::Image& gltfimage, std::string path,
          const vk::raii::Device& device, VmaAllocator allocator,
          const vk::raii::Queue& transferQueue,
          const vk::raii::CommandPool& commandPool);
  // empty texture
  Texture(const vk::raii::Device& device, VmaAllocator allocator,
          const vk::raii::Queue& transferQueue,
          const vk::raii::CommandPool& commandPool);

  void fromglTFImage(tinygltf::Image& gltfimage, std::string path,
                     const vk::raii::Device& device, VmaAllocator allocator,
                     const vk::raii::Queue& transferQueue,
                     const vk::raii::CommandPool& commandPool);
  void generateMipmaps(const vk::raii::CommandBuffer& cmdBuffer);
  void createEmptyTexture(const vk::raii::Device& device,
                          VmaAllocator allocator,
                          const vk::raii::Queue& transferQueue,
                          const vk::raii::CommandPool& commandPool);
  // NOTE: use mipLevels
  void createSampler(const vk::raii::Device& device);
  // NOTE: used at the end of fromglTFImage()
  void updateDescriptorInfo();
};

struct Material {
  enum class AlphaMode { kALPHAMODE_OPAQUE, kALPHAMODE_MASK, kALPHAMODE_BLEND };

  AlphaMode alphaMode = AlphaMode::kALPHAMODE_OPAQUE;
  float alphaCutoff = 1.0f;
  float metallicFactor = 1.0f;
  float roughnessFactor = 1.0f;
  glm::vec4 baseColorFactor = glm::vec4(1.0f);
  const vgeu::glTF::Texture* baseColorTexture = nullptr;
  const vgeu::glTF::Texture* metallicRoughnessTexture = nullptr;
  const vgeu::glTF::Texture* normalTexture = nullptr;
  const vgeu::glTF::Texture* occlusionTexture = nullptr;
  const vgeu::glTF::Texture* emissiveTexture = nullptr;
  // NOTE: not yet used.
  const vgeu::glTF::Texture* specularGlossinessTexture = nullptr;
  const vgeu::glTF::Texture* diffuseTexture = nullptr;

  vk::raii::DescriptorSet descriptorSet = nullptr;

  void createDescriptorSet(
      const vk::raii::Device& device,
      const vk::raii::DescriptorPool& descriptorPool,
      const vk::raii::DescriptorSetLayout& descriptorSetLayout,
      DescriptorBindingFlags descriptorBindingFlags);
};

struct Dimensions {
  glm::vec3 min = glm::vec3(std::numeric_limits<float>::max());
  glm::vec3 max = glm::vec3(-std::numeric_limits<float>::max());
  glm::vec3 size;
  glm::vec3 center;
  float radius;
};

struct Primitive {
  uint32_t firstIndex;
  uint32_t indexCount;
  uint32_t firstVertex;
  uint32_t vertexCount;
  const Material& material;

  Dimensions dimensions;

  void setDimensions(glm::vec3 min, glm::vec3 max);
  Primitive(uint32_t firstIndex, uint32_t indexCount, const Material& material)
      : firstIndex(firstIndex), indexCount(indexCount), material(material){};
};

#define MAX_JOINT_MATRICES 64
struct Mesh {
  // NOTE: vector of Primitive class itself
  std::vector<Primitive> primitives;
  std::string name;

  // NOTE: for skinning, update ubos each frame
  // we need separate uniformBuffers and descriptorSets
  std::vector<std::unique_ptr<VgeuBuffer>> uniformBuffers;
  std::vector<vk::raii::DescriptorSet> descriptorSets;
  struct UniformBlock {
    glm::mat4 matrix{1.f};
    glm::mat4 jointMatrices[MAX_JOINT_MATRICES]{};
    glm::vec4 jointcount{0.f};
  } uniformBlock;

  Mesh(VmaAllocator allocator, glm::mat4 matrix, const uint32_t framesInFlight);
};

struct Skin {
  std::string name;
  const Node* skeletonRoot = nullptr;
  std::vector<glm::mat4> inverseBindMatrices;
  std::vector<const Node*> joints;
};

// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#nodes-and-hierarchy
// node hierarchy is tree -> children can be unique_ptr
// techically forest
struct Node {
  // NTOE: raw ptr. do not have ownership of parent node.
  const Node* parent = nullptr;
  uint32_t index;
  // NOTE: unqiue_ptr since nodes are in tree structure.
  std::vector<std::unique_ptr<Node>> children;
  glm::mat4 matrix;
  std::string name;
  // NOTE: unique_ptr for lazy init.
  std::unique_ptr<Mesh> mesh;
  // NOTE: raw ptr. skin ownerships are on the Model with skins.
  const Skin* skin = nullptr;
  int32_t skinIndex = -1;
  glm::vec3 translation{};
  glm::vec3 scale{1.0f};
  glm::quat rotation{};
  glm::mat4 localMatrix() const;
  glm::mat4 getMatrix() const;
  void update(const uint32_t frameIndex, bool isBindPose = false);
};

struct AnimationChannel {
  enum class PathType { kTranslation, kRotation, kScale };
  PathType path;
  // changes node matrix, so not const
  Node* node;
  uint32_t samplerIndex;
};

struct AnimationSampler {
  enum class InterpolationType { kLinear, kStep, kCubicSpline };
  InterpolationType interpolation;
  std::vector<float> inputs;
  std::vector<glm::vec4> outputsVec4;
};

struct Animation {
  std::string name;
  std::vector<AnimationSampler> samplers;
  std::vector<AnimationChannel> channels;
  float start = std::numeric_limits<float>::max();
  float end = std::numeric_limits<float>::min();
};

enum class VertexComponent {
  kPosition,
  kNormal,
  kUV,
  kColor,
  kTangent,
  kJoint0,
  kWeight0
};

struct Vertex {
  // NOTE: vec3->vec4 for ssbo usage (last can be used to skinIndex)
  alignas(16) glm::vec4 pos;
  // NOTE: vec3->vec4 for ssbo usage
  alignas(16) glm::vec4 normal;
  alignas(16) glm::vec4 color;
  alignas(16) glm::vec4 joint0;
  alignas(16) glm::vec4 weight0;
  alignas(16) glm::vec4 tangent;
  // NOTE: moved last for ssbo usage
  alignas(8) glm::vec2 uv;

  // NOTE: thread_local
  static vk::VertexInputBindingDescription& getInputBindingDescription(
      uint32_t binding);

  static vk::VertexInputAttributeDescription getInputAttributeDescription(
      uint32_t binding, uint32_t location, VertexComponent component);

  // NOTE: thread_local
  static std::vector<vk::VertexInputAttributeDescription>&
  getInputAttributeDescriptions(uint32_t binding,
                                const std::vector<VertexComponent>& components);

  // NOTE: returns thread_local vertex input state.
  static vk::PipelineVertexInputStateCreateInfo getPipelineVertexInputState(
      const std::vector<VertexComponent>& components);
};

class Model {
 public:
  // setup common resources
  Model(const vk::raii::Device& device, VmaAllocator allocator,
        const vk::raii::Queue& transferQueue,
        const vk::raii::CommandPool& commandPool, uint32_t framesInFlight);
  ~Model();

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // NOTE: move ownership of root nodes to the "nodes"
  // move owenership of children nodes to the parent's vector
  void loadFromFile(
      std::string filename,
      FileLoadingFlags fileLoadingFlags = FileLoadingFlagBits::kNone,
      float scale = 1.0f);

  void bindBuffers(const vk::raii::CommandBuffer& cmdBuffer);
  void bindSSBO(const vk::raii::CommandBuffer& cmdBuffer,
                vk::PipelineLayout pipelineLayout, uint32_t bindSet);

  // NOTE: nullable pipelinelayout
  void drawNode(const uint32_t frameIndex, const Node* node,
                const vk::raii::CommandBuffer& cmdBuffer,
                RenderFlags renderFlags = {},
                vk::PipelineLayout pipelineLayout = VK_NULL_HANDLE,
                uint32_t bindImageSet = 1, int bindSkinSet = -1);

  void draw(const uint32_t frameIndex, const vk::raii::CommandBuffer& cmdBuffer,
            RenderFlags renderFlags = {},
            vk::PipelineLayout pipelineLayout = VK_NULL_HANDLE,
            uint32_t bindImageSet = 1, int bindSkinSet = -1);

  // TOOD: bind pose skeleton not yet supported.
  void getSkeletonMatrices(
      std::vector<std::vector<glm::mat4>>& jointMatrices) const;
  void updateAnimation(const uint32_t frameIndex, const int Animationindex,
                       const float time, const bool repeat = false);
  Dimensions getDimensions() const { return dimensions; };

  // NOTE: moved from globals to model class member.
  // TODO: all models should share those values, better to move it out of model
  vk::raii::DescriptorSetLayout descriptorSetLayoutImage = nullptr;
  vk::raii::DescriptorSetLayout descriptorSetLayoutUbo = nullptr;
  vk::raii::DescriptorSetLayout descriptorSetLayoutVertex = nullptr;

  // TODO: check instead usageFlags, for raytracing related
  vk::MemoryPropertyFlags memoryPropertyFlags{};
  // NOTE: <unresolved overloaded function type> for () constructor,
  // when used in operator|
  vk::BufferUsageFlags additionalBufferUsageFlags{};
  // NOTE: for normal map
  DescriptorBindingFlags descriptorBindingFlags =
      DescriptorBindingFlagBits::kImageBaseColor;

 private:
  // load images from files
  void loadImages(tinygltf::Model& gltfModel);

  void loadMaterials(const tinygltf::Model& gltfModel);

  void loadNode(Node* parent, const tinygltf::Node& node, uint32_t nodeIndex,
                const tinygltf::Model& model, std::vector<uint32_t>& indices,
                std::vector<Vertex>& vertices, float globalscale);
  void loadSkins(const tinygltf::Model& gltfModel);

  void loadAnimations(const tinygltf::Model& gltfModel);

  const Texture* getTexture(uint32_t index) const;
  // non-const Node ptr return for animation channel
  Node* findNode(Node* parent, uint32_t index) const;
  Node* nodeFromIndex(uint32_t index) const;
  void prepareNodeDescriptor(
      const Node* node,
      const vk::raii::DescriptorSetLayout& descriptorSetLayout);

  void getNodeDimensions(const Node* node, glm::vec3& min,
                         glm::vec3& max) const;
  void setSceneDimensions();

  const vk::raii::Device& device;
  VmaAllocator allocator;
  const vk::raii::Queue& transferQueue;
  const vk::raii::CommandPool& commandPool;
  const uint32_t framesInFlight;

  vk::raii::DescriptorPool descriptorPool = nullptr;
  std::unique_ptr<VgeuBuffer> vertexBuffer;
  std::unique_ptr<VgeuBuffer> indexBuffer;

  // NOTE: descriptorSet for animation in compute shader
  vk::raii::DescriptorSet descriptorSetVertex = nullptr;

  // NOTE: takes ownership since thoese are root nodes of each tree.
  // unique_ptr
  std::vector<std::unique_ptr<Node>> nodes;
  // all nodes without ownership
  std::vector<Node*> linearNodes;

  std::vector<Skin> skins;

  // NOTE: unique_ptr or class itself, it doesn't matter
  // since those class are safe to be moved. (by default move constructor)
  std::vector<std::unique_ptr<Texture>> textures;
  std::vector<Material> materials;

  std::vector<Animation> animations;

  Dimensions dimensions;

  std::unique_ptr<Texture> emptyTexture;
  // NOTE: not test yet.
  bool metallicRoughnessWorkflow = true;
  bool buffersBound = false;
  std::string path;
};

}  // namespace glTF
}  // namespace vgeu