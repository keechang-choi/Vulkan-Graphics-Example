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

namespace vgeu {
namespace glTF {
enum class DescriptorBindingFlags {
  kImageBaseColor = 0x00000001,
  kImageNormalMap = 0x00000002
};

struct Node;

struct Texture {
  vgeu::ImageData imageData = nullptr;
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
                     const vk::raii::Queue& copyQueue);
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

  vgeu::glTF::Texture* specularGlossinessTexture;
  vgeu::glTF::Texture* diffuseTexture;

  vk::raii::DescriptorSet descriptorSet = nullptr;

  Material(const vk::raii::Device& device) : device(device){};
  void createDescriptorSet(
      const vk::raii::DescriptorPool& descriptorPool,
      const vk::raii::DescriptorSetLayout& descriptorSetLayout,
      uint32_t descriptorBindingFlags);
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
  const vk::raii::Device& device;

  // TODO: unqiue_ptr
  std::vector<Primitive*> primitives;
  std::string name;

  std::unique_ptr<VgeuBuffer> uniformBuffer;
  vk::DescriptorBufferInfo descriptor;
  vk::raii::DescriptorSet descriptorSet = nullptr;
  struct UniformBlock {
    glm::mat4 matrix;
    glm::mat4 jointMatrix[64]{};
    float jointcount{0};
  } uniformBlock;

  Mesh(const vk::raii::Device& device, glm::mat4 matrix);
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
  std::vector<Node*> children;
  glm::mat4 matrix;
  std::string name;
  // TODO: unique_ptr
  Mesh* mesh;
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

enum class FileLoadingFlags {
  kNone = 0x00000000,
  kPreTransformVertices = 0x00000001,
  kPreMultiplyVertexColors = 0x00000002,
  kFlipY = 0x00000004,
  kDontLoadImages = 0x00000008
};

enum class RenderFlags {
  kBindImages = 0x00000001,
  kRenderOpaqueNodes = 0x00000002,
  kRenderAlphaMaskedNodes = 0x00000004,
  kRenderAlphaBlendedNodes = 0x00000008
};
class Model {
 public:
  Model(){};
  ~Model();

  void loadNode(Node* parent, const tinygltf::Node& node, uint32_t nodeIndex,
                const tinygltf::Model& model,
                std::vector<uint32_t>& indexBuffer,
                std::vector<Vertex>& vertexBuffer, float globalscale);
  // TODO: skins
  void loadImages(tinygltf::Model& gltfModel, const vk::raii::Device& device,
                  VkQueue transferQueue);
  void loadMaterials(tinygltf::Model& gltfModel);
  // TODO: animation

  // NOTE: move ownership of root nodes to the "nodes"
  // move owenership of children nodes to the parent's vector
  void loadFromFile(std::string filename, const vk::raii::Device& device,
                    const vk::raii::Queue& transferQueue,
                    FileLoadingFlags fileLoadingFlags = FileLoadingFlags::kNone,
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
  void getNodeDimensions(Node* node, glm::vec3& min, glm::vec3& max);
  void getSceneDimensions();
  // TODO: animation
  Node* findNode(Node* parent, uint32_t index);
  Node* nodeFromIndex(uint32_t index);
  void prepareNodeDescriptor(
      Node* node, const vk::raii::DescriptorSetLayout& descriptorSetLayout);

  // TODO: moved from globals to model class member.
  // check any problems
  vk::raii::DescriptorSetLayout descriptorSetLayoutImage = nullptr;
  vk::raii::DescriptorSetLayout descriptorSetLayoutUbo = nullptr;
  vk::MemoryPropertyFlags memoryPropertyFlags();
  DescriptorBindingFlags descriptorBindingFlags =
      DescriptorBindingFlags::kImageBaseColor;

  vk::raii::DescriptorPool descriptorPool = nullptr;
  std::unique_ptr<VgeuBuffer> vertices;
  std::unique_ptr<VgeuBuffer> indices;
  // TODO: takes ownership since root nodes or each tree.
  // unique_ptr
  std::vector<Node*> nodes;
  // all nodes without ownership
  std::vector<Node*> linearNodes;

  // TODO: skin
  // std::vector<Skin*> skins;

  std::vector<Texture> textures;
  std::vector<Material> materials;

  // TODO: animation
  // std::vector<Animation> animations;

  Dimensions dimensions;

  bool metallicRoughnessWorkflow = true;
  bool buffersBound = false;
  std::string path;

 private:
  void createEmptyTexture(VkQueue transferQueue);

  Texture* getTexture(uint32_t index);
  Texture emptyTexture;

  // NOTE: nullable
  vk::Device device;
};

}  // namespace glTF
}  // namespace vgeu