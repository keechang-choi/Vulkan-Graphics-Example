/*

gltf loading class based on tinyglTF

base code from
https://github.com/SaschaWillems/Vulkan/blob/master/base/VulkanglTFModel.h

*/
#pragma once

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

namespace vgeu {
namespace glTF {
enum DescriptorBindingFlags {
  ImageBaseColor = 0x00000001,
  ImageNormalMap = 0x00000002
};

extern vk::raii::DescriptorSetLayout descriptorSetLayoutImage;
extern vk::raii::DescriptorSetLayout descriptorSetLayoutUbo;
extern vk::MemoryPropertyFlags memoryPropertyFlags;
extern uint32_t descriptorBindingFlags;

struct Node;

struct Texture {
  vgeu::ImageData imageData = nullptr;
  uint32_t width, height;
  uint32_t mipLevels;
  uint32_t layerCount;
  vk::DescriptorImageInfo descriptor;
  vk::raii::Sampler sampler;
  void updateDescriptor();
  void fromglTfImage(tinygltf::Image& gltfimage, std::string path,
                     const vk::raii::Device& device,
                     const vk::raii::Queue& copyQueue);
};

struct Material {
  const vk::raii::Device& device;
  enum AlphaMode { ALPHAMODE_OPAQUE, ALPHAMODE_MASK, ALPHAMODE_BLEND };
  AlphaMode alphaMode = ALPHAMODE_OPAQUE;
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

struct Primitive {
  uint32_t firstIndex;
  uint32_t indexCount;
  uint32_t firstVertex;
  uint32_t vertexCount;
  Material& material;

  struct Dimensions {
    glm::vec3 min = glm::vec3(FLT_MAX);
    glm::vec3 max = glm::vec3(-FLT_MAX);
    glm::vec3 size;
    glm::vec3 center;
    float radius;
  } dimensions;

  void setDimensions(glm::vec3 min, glm::vec3 max);
  Primitive(uint32_t firstIndex, uint32_t indexCount, Material& material)
      : firstIndex(firstIndex), indexCount(indexCount), material(material){};
};

struct Mesh {
  const vk::raii::Device& device;

  std::vector<Primitive*> primitives;
  std::string name;

  struct UniformBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDescriptorBufferInfo descriptor;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    void* mapped;
  } uniformBuffer;

  struct UniformBlock {
    glm::mat4 matrix;
    glm::mat4 jointMatrix[64]{};
    float jointcount{0};
  } uniformBlock;

  Mesh(const vk::raii::Device& device, glm::mat4 matrix);
  ~Mesh();
};

struct Node {
  Node* parent;
  uint32_t index;
  std::vector<Node*> children;
  glm::mat4 matrix;
  std::string name;
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

enum class VertexComponent {
  Position,
  Normal,
  UV,
  Color,
  Tangent,
  Joint0,
  Weight0
};

struct Vertex {
  glm::vec3 pos;
  glm::vec3 normal;
  glm::vec2 uv;
  glm::vec4 color;
  glm::vec4 joint0;
  glm::vec4 weight0;
  glm::vec4 tangent;
  static VkVertexInputBindingDescription vertexInputBindingDescription;
  static std::vector<VkVertexInputAttributeDescription>
      vertexInputAttributeDescriptions;
  static VkPipelineVertexInputStateCreateInfo
      pipelineVertexInputStateCreateInfo;
  static VkVertexInputBindingDescription inputBindingDescription(
      uint32_t binding);
  static VkVertexInputAttributeDescription inputAttributeDescription(
      uint32_t binding, uint32_t location, VertexComponent component);
  static std::vector<VkVertexInputAttributeDescription>
  inputAttributeDescriptions(uint32_t binding,
                             const std::vector<VertexComponent> components);
  /** @brief Returns the default pipeline vertex input state create info
   * structure for the requested vertex components */
  static VkPipelineVertexInputStateCreateInfo* getPipelineVertexInputState(
      const std::vector<VertexComponent> components);
};

enum FileLoadingFlags {
  None = 0x00000000,
  PreTransformVertices = 0x00000001,
  PreMultiplyVertexColors = 0x00000002,
  FlipY = 0x00000004,
  DontLoadImages = 0x00000008
};

enum RenderFlags {
  BindImages = 0x00000001,
  RenderOpaqueNodes = 0x00000002,
  RenderAlphaMaskedNodes = 0x00000004,
  RenderAlphaBlendedNodes = 0x00000008
};

}  // namespace glTF
}  // namespace vgeu