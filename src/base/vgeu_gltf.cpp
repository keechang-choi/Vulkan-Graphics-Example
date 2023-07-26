#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vgeu_gltf.hpp"

#include "vgeu_utils.hpp"

// std
#include <stdexcept>

namespace {

bool isKtx(const tinygltf::Image& gltfImage) {
  bool isUriKtx{false};
  if (gltfImage.uri.find_last_of(".") != std::string::npos) {
    if (gltfImage.uri.substr(gltfImage.uri.find_last_of(".") + 1) == "ktx") {
      isUriKtx = true;
    }
  }
  return isUriKtx;
}
bool loadImageDataFunc(tinygltf::Image* gltfImage, const int imageIndex,
                       std::string* error, std::string* warning, int req_width,
                       int req_height, const unsigned char* bytes, int size,
                       void* userData) {
  if (isKtx(*gltfImage)) {
    return true;
  }
  return tinygltf::LoadImageData(gltfImage, imageIndex, error, warning,
                                 req_width, req_height, bytes, size, userData);
}

bool loadImageDataFuncEmpty(tinygltf::Image* image, const int imageIndex,
                            std::string* error, std::string* warning,
                            int req_width, int req_height,
                            const unsigned char* bytes, int size,
                            void* userData) {
  return true;
}
}  // namespace
namespace vgeu {
namespace glTF {
Texture::Texture(tinygltf::Image& gltfimage, std::string path,
                 const vk::raii::Device& device,
                 const vk::raii::PhysicalDevice& physicalDevice,
                 VmaAllocator allocator, const vk::raii::Queue& transferQueue) {
  fromglTfImage(gltfimage, path, device, physicalDevice, allocator,
                transferQueue);
}

Texture::Texture(const vk::raii::Device& device, VmaAllocator allocator,
                 const vk::raii::Queue& transferQueue) {
  createEmptyTexture(device, allocator, transferQueue);
}

void Texture::fromglTfImage(tinygltf::Image& gltfImage, std::string path,
                            const vk::raii::Device& device,
                            const vk::raii::PhysicalDevice& physicalDevice,
                            VmaAllocator allocator,
                            const vk::raii::Queue& copyQueue) {
  if (!::isKtx(gltfImage)) {
  } else {
  }
}

void Texture::createEmptyTexture(const vk::raii::Device& device,
                                 VmaAllocator allocator,
                                 const vk::raii::Queue& transferQueue) {
  width = 1;
  height = 1;
  layerCount = 1;
  mipLevels = 1;
  unsigned char buffer = 0u;

  {
    vgeu::VgeuBuffer stagingBuffer(
        allocator, 4, width * height, vk::BufferUsageFlagBits::eTransferSrc,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);

    std::memcpy(stagingBuffer.getMappedData(), &buffer,
                stagingBuffer.getBufferSize());

    vgeuImage = std::make_unique<VgeuImage>(
        device, allocator, vk::Format::eR8G8B8A8Unorm,
        vk::Extent2D(width, height), vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::ImageLayout::eUndefined,
        VmaAllocationCreateFlagBits::
            VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT);
  }
}

void Texture::updateDescriptorInfo() {
  descriptorInfo.sampler = *sampler;
  descriptorInfo.imageView = *vgeuImage->getImageView();
  descriptorInfo.imageLayout = imageLayout;
}
Model::Model(const vk::raii::Device& device,
             const vk::raii::PhysicalDevice& physicalDevice,
             VmaAllocator allocator, const vk::raii::Queue& transferQueue,
             const vk::raii::CommandPool& commandPool)
    : device(device),
      physicalDevice(physicalDevice),
      allocator(allocator),
      transferQueue(transferQueue),
      commandPool(commandPool) {}

void Model::loadFromFile(std::string filename,
                         FileLoadingFlags fileLoadingFlags, float scale) {
  tinygltf::Model gltfModel;
  tinygltf::TinyGLTF gltfContext;
  if (fileLoadingFlags & FileLoadingFlagBits::kDontLoadImages) {
    gltfContext.SetImageLoader(loadImageDataFuncEmpty, nullptr);
  } else {
    gltfContext.SetImageLoader(loadImageDataFunc, nullptr);
  }
  size_t pos = filename.find_last_of('/');
  path = filename.substr(0, pos);
  std::string error, warning;
  bool fileLoaded =
      gltfContext.LoadASCIIFromFile(&gltfModel, &error, &warning, filename);
  if (!fileLoaded) {
    throw std::runtime_error("failed to open file: " + filename);
  }

  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;

  if (!(fileLoadingFlags & FileLoadingFlagBits::kDontLoadImages)) {
    loadImages(gltfModel);
  }
  loadMaterials(gltfModel);
  const tinygltf::Scene& scene =
      gltfModel
          .scenes[gltfModel.defaultScene > -1 ? gltfModel.defaultScene : 0];
  for (size_t i = 0; i < scene.nodes.size(); i++) {
    const tinygltf::Node node = gltfModel.nodes[scene.nodes[i]];
    loadNode(nullptr, node, scene.nodes[i], gltfModel, indices, vertices,
             scale);
  }
  if (gltfModel.animations.size() > 0) {
    // TODO:
    // loadAnimations(gltfModel);
  }
  // TODO:
  // loadSkins(gltfModel);

  for (auto node : linearNodes) {
    // TODO:
    // Assign skins
    // if (node->skinIndex > -1) {
    //   node->skin = skins[node->skinIndex];
    // }
    // Initial pose
    if (node->mesh) {
      node->update();
    }
  }

  // Pre-Calculations for requested features
  if ((fileLoadingFlags & FileLoadingFlagBits::kPreTransformVertices) ||
      (fileLoadingFlags & FileLoadingFlagBits::kPreMultiplyVertexColors) ||
      (fileLoadingFlags & FileLoadingFlagBits::kFlipY)) {
    const bool preTransform = static_cast<bool>(
        fileLoadingFlags & FileLoadingFlagBits::kPreTransformVertices);
    const bool preMultiplyColor = static_cast<bool>(
        fileLoadingFlags & FileLoadingFlagBits::kPreMultiplyVertexColors);
    const bool flipY =
        static_cast<bool>(fileLoadingFlags & FileLoadingFlagBits::kFlipY);
    for (Node* node : linearNodes) {
      if (node->mesh.get() == nullptr) {
        continue;
      }
      const glm::mat4 localMatrix = node->getMatrix();
      for (const auto& primitive : node->mesh->primitives) {
        for (uint32_t i = 0; i < primitive->vertexCount; i++) {
          Vertex& vertex = vertices[primitive->firstVertex + i];
          // Pre-transform vertex positions by node-hierarchy
          if (preTransform) {
            vertex.pos = glm::vec3(localMatrix * glm::vec4(vertex.pos, 1.0f));
            vertex.normal =
                glm::normalize(glm::mat3(localMatrix) * vertex.normal);
          }
          // Flip Y-Axis of vertex positions
          if (flipY) {
            vertex.pos.y *= -1.0f;
            vertex.normal.y *= -1.0f;
          }
          // Pre-Multiply vertex colors with material base color
          if (preMultiplyColor) {
            vertex.color = primitive->material.baseColorFactor * vertex.color;
          }
        }
      }
    }
  }

  for (auto extension : gltfModel.extensionsUsed) {
    if (extension == "KHR_materials_pbrSpecularGlossiness") {
      std::cout << "Required extension: " << extension;
      metallicRoughnessWorkflow = false;
    }
  }

  // vertex buffer
  {
    assert(vertices.size() > 0);
    vgeu::VgeuBuffer vertexStagingBuffer(
        allocator, sizeof(Vertex), static_cast<uint32_t>(vertices.size()),
        vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    std::memcpy(vertexStagingBuffer.getMappedData(), vertices.data(),
                vertexStagingBuffer.getBufferSize());

    vk::BufferUsageFlags vertexBufferUsageFlags =
        vk::BufferUsageFlagBits::eVertexBuffer |
        vk::BufferUsageFlagBits::eTransferDst;
    vertexBuffer = std::make_unique<vgeu::VgeuBuffer>(
        allocator, sizeof(Vertex), static_cast<uint32_t>(vertices.size()),
        (vk::BufferUsageFlagBits::eVertexBuffer |
         vk::BufferUsageFlagBits::eTransferDst) |
            additionalBufferUsageFlags,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);

    // index buffer
    assert(indices.size() > 0);
    vgeu::VgeuBuffer indexStagingBuffer(
        allocator, sizeof(uint32_t), static_cast<uint32_t>(indices.size()),
        vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);

    std::memcpy(indexStagingBuffer.getMappedData(), indices.data(),
                indexStagingBuffer.getBufferSize());

    indexBuffer = std::make_unique<vgeu::VgeuBuffer>(
        allocator, sizeof(uint32_t), static_cast<uint32_t>(indices.size()),
        (vk::BufferUsageFlagBits::eIndexBuffer |
         vk::BufferUsageFlagBits::eTransferDst) |
            additionalBufferUsageFlags,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);

    // single Time command copy both buffers
    vgeu::oneTimeSubmit(
        device, commandPool, transferQueue,
        [&](const vk::raii::CommandBuffer& cmdBuffer) {
          cmdBuffer.copyBuffer(
              vertexStagingBuffer.getBuffer(), vertexBuffer->getBuffer(),
              vk::BufferCopy(0, 0, vertexStagingBuffer.getBufferSize()));
          cmdBuffer.copyBuffer(
              indexStagingBuffer.getBuffer(), indexBuffer->getBuffer(),
              vk::BufferCopy(0, 0, indexStagingBuffer.getBufferSize()));
        });
  }
  setSceneDimensions();

  // Setup descriptors
  uint32_t uboCount{0};
  uint32_t imageCount{0};
  for (auto node : linearNodes) {
    if (node->mesh) {
      uboCount++;
    }
  }
  for (auto& material : materials) {
    if (material.baseColorTexture != nullptr) {
      imageCount++;
    }
  }

  std::vector<vk::DescriptorPoolSize> poolSizes;
  poolSizes.emplace_back(vk::DescriptorType::eUniformBuffer, uboCount);

  if (imageCount > 0) {
    if (descriptorBindingFlags & DescriptorBindingFlagBits::kImageBaseColor) {
      poolSizes.emplace_back(vk::DescriptorType::eCombinedImageSampler,
                             imageCount);
    }
    if (descriptorBindingFlags & DescriptorBindingFlagBits::kImageNormalMap) {
      poolSizes.emplace_back(vk::DescriptorType::eCombinedImageSampler,
                             imageCount);
    }
  }
  // NOTE: mesh and material own descriptorSet (not sets).
  // -> maxSets: unoCount for each mesh, imageCount for each material
  // one DescriptorSet of material may contain two textures (as binding).
  vk::DescriptorPoolCreateInfo descriptorPoolCI(
      vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      uboCount + imageCount, poolSizes);
  descriptorPool = vk::raii::DescriptorPool(device, descriptorPoolCI);

  // Descriptors for per-node uniform buffers
  {
    // Layout is global, so only create if it hasn't already been created before
    if (!*descriptorSetLayoutUbo) {
      vk::DescriptorSetLayoutBinding setlayoutBinding(
          0, vk::DescriptorType::eUniformBuffer, 1,
          vk::ShaderStageFlagBits::eVertex);
      vk::DescriptorSetLayoutCreateInfo setLayoutCI({}, 1, &setlayoutBinding);
      descriptorSetLayoutUbo =
          vk::raii::DescriptorSetLayout(device, setLayoutCI);
    }
    for (const auto& node : nodes) {
      prepareNodeDescriptor(node.get(), descriptorSetLayoutUbo);
    }
  }

  // Descriptors for per-material images
  {
    // Layout is global, so only create if it hasn't already been created before
    if (!*descriptorSetLayoutImage) {
      std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings{};
      // binding 0
      if (descriptorBindingFlags & DescriptorBindingFlagBits::kImageBaseColor) {
        setLayoutBindings.emplace_back(
            static_cast<uint32_t>(setLayoutBindings.size()),
            vk::DescriptorType::eCombinedImageSampler, 1,
            vk::ShaderStageFlagBits::eFragment);
      }
      // binding 0 or 1
      if (descriptorBindingFlags & DescriptorBindingFlagBits::kImageNormalMap) {
        setLayoutBindings.emplace_back(
            static_cast<uint32_t>(setLayoutBindings.size()),
            vk::DescriptorType::eCombinedImageSampler, 1,
            vk::ShaderStageFlagBits::eFragment);
      }
      // NOTE: using constructor of vk::ArrayProxyNoTemporaries
      vk::DescriptorSetLayoutCreateInfo setLayoutCI(
          vk::DescriptorSetLayoutCreateFlags{}, setLayoutBindings);
      descriptorSetLayoutImage =
          vk::raii::DescriptorSetLayout(device, setLayoutCI);
    }
    for (auto& material : materials) {
      if (material.baseColorTexture != nullptr) {
        material.createDescriptorSet(descriptorPool, descriptorSetLayoutImage,
                                     descriptorBindingFlags);
      }
    }
  }
}

void Model::loadImages(tinygltf::Model& gltfModel) {
  for (tinygltf::Image& image : gltfModel.images) {
    Texture texture;
    texture.fromglTfImage(image, path, device, physicalDevice, allocator,
                          transferQueue);
    textures.push_back(texture);
  }
  // Create an empty texture to be used for empty material images
  createEmptyTexture(transferQueue);
}

}  // namespace glTF
}  // namespace vgeu