/*

gltf loading class based on tinyglTF

base code from
https://github.com/SaschaWillems/Vulkan/blob/master/base/VulkanglTFModel.h

*/
#pragma once

#include "vgeu_buffer.hpp"
#include "vgeu_utils.hpp"

// libs
#include "tiny_gltf.h"
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

//
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

// std
#include <limits>
#include <memory>
#include <type_traits>
namespace vgeu {

// Reference: Vulkan-Hpp enums
template <typename BitType>
class Flags {
 public:
  using MaskType = typename std::underlying_type<BitType>::type;

  constexpr Flags() noexcept : mask_(0) {}
  constexpr Flags(BitType bit) noexcept : mask_(static_cast<MaskType>(bit)) {}
  constexpr Flags(Flags<BitType> const& rhs) noexcept = default;
  constexpr explicit Flags(MaskType flags) noexcept : mask_(flags) {}
  constexpr bool operator==(Flags<BitType> const& rhs) const noexcept {
    return mask_ == rhs.mask_;
  }
  constexpr bool operator!=(Flags<BitType> const& rhs) const noexcept {
    return mask_ != rhs.mask_;
  }
  constexpr Flags<BitType> operator&(Flags<BitType> const& rhs) const noexcept {
    return Flags<BitType>(mask_ & rhs.mask_);
  }
  constexpr Flags<BitType> operator|(Flags<BitType> const& rhs) const noexcept {
    return Flags<BitType>(mask_ | rhs.mask_);
  }
  constexpr Flags<BitType> operator^(Flags<BitType> const& rhs) const noexcept {
    return Flags<BitType>(mask_ ^ rhs.mask_);
  }
  explicit constexpr operator bool() const noexcept { return !!mask_; }
  explicit constexpr operator MaskType() const noexcept { return mask_; }

 private:
  MaskType mask_;
};

namespace glTF {
enum class DescriptorBindingFlagBits : uint32_t {
  kImageBaseColor = 0x00000001,
  kImageNormalMap = 0x00000002
};

using DescriptorBindingFlags = Flags<DescriptorBindingFlagBits>;

struct Node;

struct Texture {
  std::unique_ptr<vgeu::VgeuImage> vgeuImage;
  vk::ImageLayout imageLayout;
  uint32_t width, height;
  uint32_t mipLevels;
  uint32_t layerCount;
  vk::DescriptorImageInfo descriptorInfo;
  vk::raii::Sampler sampler = nullptr;
  // TODO: use end of fromglTFImage()
  void updateDescriptorInfo();
  void fromglTfImage(tinygltf::Image& gltfimage, std::string path,
                     const vk::raii::Device& device,
                     const vk::raii::PhysicalDevice& physicalDevice,
                     VmaAllocator allocator,
                     const vk::raii::Queue& transferQueue);
};

struct Material {
  const vk::raii::Device& device;
  enum class AlphaMode { kALPHAMODE_OPAQUE, kALPHAMODE_MASK, kALPHAMODE_BLEND };
  AlphaMode alphaMode = AlphaMode::kALPHAMODE_OPAQUE;
  float alphaCutoff = 1.0f;
  float metallicFactor = 1.0f;
  float roughnessFactor = 1.0f;
  glm::vec4 baseColorFactor = glm::vec4(1.0f);
  vgeu::glTF::Texture* baseColorTexture = nullptr;
  vgeu::glTF::Texture* metallicRoughnessTexture = nullptr;
  vgeu::glTF::Texture* normalTexture = nullptr;
  vgeu::glTF::Texture* occlusionTexture = nullptr;
  vgeu::glTF::Texture* emissiveTexture = nullptr;
  // TODO: check it used.
  vgeu::glTF::Texture* specularGlossinessTexture;
  vgeu::glTF::Texture* diffuseTexture;

  vk::raii::DescriptorSet descriptorSet = nullptr;

  Material(const vk::raii::Device& device) : device(device){};
  void createDescriptorSet(
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
  Material& material;

  Dimensions dimensions;

  void setDimensions(glm::vec3 min, glm::vec3 max);
  Primitive(uint32_t firstIndex, uint32_t indexCount, Material& material)
      : firstIndex(firstIndex), indexCount(indexCount), material(material){};
};

struct Mesh {
  // TODO: unique_ptr or class itself
  std::vector<std::unique_ptr<Primitive>> primitives;
  std::string name;

  std::unique_ptr<VgeuBuffer> uniformBuffer;
  vk::DescriptorBufferInfo descriptorInfo;
  vk::raii::DescriptorSet descriptorSet = nullptr;
  struct UniformBlock {
    glm::mat4 matrix;
    glm::mat4 jointMatrix[64]{};
    float jointcount{0};
  } uniformBlock;

  Mesh(VmaAllocator allocator, glm::mat4 matrix);
  ~Mesh();
};

// TODO: skin

// https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#nodes-and-hierarchy
// TODO: check nodes are used as DAG

// node hierarchy is tree -> children can be unique_ptr
// techically forest
struct Node {
  // TODO: raw ptr?
  Node* parent;
  uint32_t index;
  // TODO: unqiue_ptr since tree structure.
  std::vector<std::unique_ptr<Node>> children;
  glm::mat4 matrix;
  std::string name;
  // TODO: unique_ptr
  std::unique_ptr<Mesh> mesh;
  // Skin* skin;
  int32_t skinIndex = -1;
  glm::vec3 translation{};
  glm::vec3 scale{1.0f};
  glm::quat rotation{};
  glm::mat4 localMatrix();
  glm::mat4 getMatrix();
  void update();
  ~Node();
};

// TODO: animation

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
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec2 uv;
  glm::vec4 color;
  glm::vec4 joint0;
  glm::vec4 weight0;
  glm::vec4 tangent;
  static vk::VertexInputBindingDescription vertexInputBindingDescription;
  static std::vector<vk::VertexInputAttributeDescription>
      vertexInputAttributeDescriptions;
  static vk::PipelineVertexInputStateCreateInfo
      pipelineVertexInputStateCreateInfo;
  static vk::VertexInputBindingDescription inputBindingDescription(
      uint32_t binding);
  static vk::VertexInputAttributeDescription inputAttributeDescription(
      uint32_t binding, uint32_t location, VertexComponent component);
  static std::vector<vk::VertexInputAttributeDescription>
  inputAttributeDescriptions(uint32_t binding,
                             const std::vector<VertexComponent> components);

  // TODO: check pointer
  static vk::PipelineVertexInputStateCreateInfo getPipelineVertexInputState(
      const std::vector<VertexComponent> components);
};

enum class FileLoadingFlagBits : uint32_t {
  kNone = 0x00000000,
  kPreTransformVertices = 0x00000001,
  kPreMultiplyVertexColors = 0x00000002,
  kFlipY = 0x00000004,
  kDontLoadImages = 0x00000008
};
using FileLoadingFlags = Flags<FileLoadingFlagBits>;

enum class RenderFlagBits {
  kBindImages = 0x00000001,
  kRenderOpaqueNodes = 0x00000002,
  kRenderAlphaMaskedNodes = 0x00000004,
  kRenderAlphaBlendedNodes = 0x00000008
};
using RenderFlags = Flags<RenderFlagBits>;

class Model {
 public:
  // setup common resources
  Model(const vk::raii::Device& device,
        const vk::raii::PhysicalDevice& physicalDevice, VmaAllocator allocator,
        const vk::raii::Queue& transferQueue,
        const vk::raii::CommandPool& commandPool);
  ~Model();

  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // NOTE: move ownership of root nodes to the "nodes"
  // move owenership of children nodes to the parent's vector
  void loadFromFile(
      std::string filename,
      FileLoadingFlags fileLoadingFlags = FileLoadingFlagBits::kNone,
      float scale = 1.0f);

  void bindBuffers(const vk::raii::CommandBuffer& commandBuffer);

  // NOTE: nullable pipelinelayout
  void drawNode(Node* node, const vk::raii::CommandBuffer& commandBuffer,
                uint32_t renderFlags = 0,
                vk::PipelineLayout pipelineLayout = VK_NULL_HANDLE,
                uint32_t bindImageSet = 1);
  void draw(const vk::raii::CommandBuffer& commandBuffer,
            uint32_t renderFlags = 0,
            vk::PipelineLayout pipelineLayout = VK_NULL_HANDLE,
            uint32_t bindImageSet = 1);
  Dimensions getDimensions() const { return dimensions; };
  // TODO: update animation

  // TODO: moved from globals to model class member.
  // check any problems
  vk::raii::DescriptorSetLayout descriptorSetLayoutImage = nullptr;
  vk::raii::DescriptorSetLayout descriptorSetLayoutUbo = nullptr;
  // TODO: check instead usageFlags, for raytracing related
  vk::MemoryPropertyFlags memoryPropertyFlags{};
  // NOTE: <unresolved overloaded function type> for () constructor
  vk::BufferUsageFlags additionalBufferUsageFlags{};
  // NOTE: used for normal map
  DescriptorBindingFlags descriptorBindingFlags =
      DescriptorBindingFlagBits::kImageBaseColor;

  vk::raii::DescriptorPool descriptorPool = nullptr;
  std::unique_ptr<VgeuBuffer> vertexBuffer;
  std::unique_ptr<VgeuBuffer> indexBuffer;

 private:
  void createEmptyTexture();
  void loadNode(Node* parent, const tinygltf::Node& node, uint32_t nodeIndex,
                const tinygltf::Model& model, std::vector<uint32_t>& indices,
                std::vector<Vertex>& vertices, float globalscale);
  // TODO: skins
  void loadImages(tinygltf::Model& gltfModel);
  void loadMaterials(tinygltf::Model& gltfModel);
  // TODO: animation
  Texture* getTexture(uint32_t index);
  Node* findNode(const Node* parent, uint32_t index);
  Node* nodeFromIndex(uint32_t index);
  void prepareNodeDescriptor(
      const Node* node,
      const vk::raii::DescriptorSetLayout& descriptorSetLayout);

  void getNodeDimensions(const Node* node, glm::vec3& min, glm::vec3& max);
  void setSceneDimensions();

  const vk::raii::Device& device;
  const vk::raii::PhysicalDevice& physicalDevice;
  VmaAllocator allocator;
  const vk::raii::Queue& transferQueue;
  const vk::raii::CommandPool& commandPool;

  // TODO: takes ownership since root nodes or each tree.
  // unique_ptr
  std::vector<std::unique_ptr<Node>> nodes;
  // all nodes without ownership
  std::vector<Node*> linearNodes;

  // TODO: skin
  // std::vector<Skin*> skins;

  std::vector<Texture> textures;
  std::vector<Material> materials;

  // TODO: animation
  // std::vector<Animation> animations;
  Dimensions dimensions;

  Texture emptyTexture;
  // TOOD: check it to be private right.
  bool metallicRoughnessWorkflow = true;
  bool buffersBound = false;
  std::string path;
};

}  // namespace glTF
}  // namespace vgeu