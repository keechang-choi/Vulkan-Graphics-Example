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
                 const vk::raii::Device& device, VmaAllocator allocator,
                 const vk::raii::Queue& transferQueue,
                 const vk::raii::CommandPool& commandPool) {
  fromglTFImage(gltfimage, path, device, allocator, transferQueue, commandPool);
}

Texture::Texture(const vk::raii::Device& device, VmaAllocator allocator,
                 const vk::raii::Queue& transferQueue,
                 const vk::raii::CommandPool& commandPool) {
  createEmptyTexture(device, allocator, transferQueue, commandPool);
}

void Texture::fromglTFImage(tinygltf::Image& gltfImage, std::string path,
                            const vk::raii::Device& device,
                            VmaAllocator allocator,
                            const vk::raii::Queue& transferQueue,
                            const vk::raii::CommandPool& commandPool) {
  if (!::isKtx(gltfImage)) {
    // NOTE: SetPreserveimageChannels false by default
    assert(gltfImage.component == 4 && "failed: image channel is not RGBA");
    vk::DeviceSize bufferSize = gltfImage.image.size();
    uint32_t pixelCount = gltfImage.width * gltfImage.height;
    uint32_t pixelSize = 4;
    assert(bufferSize == pixelCount * pixelSize);

    width = gltfImage.width;
    height = gltfImage.height;
    mipLevels = static_cast<uint32_t>(
        std::floor(std::log2(std::max(width, height))) + 1.0);
    // TODO: check physical device format properties support?

    // NOTE: create image mipLevels-count, copy 0-level image
    {
      vgeu::VgeuBuffer stagingBuffer(
          allocator, pixelSize, width * height,
          vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_AUTO,
          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
              VMA_ALLOCATION_CREATE_MAPPED_BIT);

      std::memcpy(stagingBuffer.getMappedData(), gltfImage.image.data(),
                  stagingBuffer.getBufferSize());

      vgeuImage = std::make_unique<VgeuImage>(
          device, allocator, vk::Format::eR8G8B8A8Unorm,
          vk::Extent2D(width, height), vk::ImageTiling::eOptimal,
          vk::ImageUsageFlagBits::eSampled |
              vk::ImageUsageFlagBits::eTransferSrc |
              vk::ImageUsageFlagBits::eTransferDst,
          vk::ImageLayout::eUndefined, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
          VmaAllocationCreateFlagBits::
              VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
          vk::ImageAspectFlagBits::eColor, mipLevels);

      // NOTE: row length, image height : 0 for buffer packed tightly
      vk::BufferImageCopy region(
          0, 0, 0,
          vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
          vk::Offset3D{0, 0, 0}, vk::Extent3D{width, height, 1});
      // NOTE: 0-mipLevel image copy and transition all mipLevels to dst
      oneTimeSubmit(device, commandPool, transferQueue,
                    [&](const vk::raii::CommandBuffer& cmdBuffer) {
                      setImageLayout(cmdBuffer, vgeuImage->getImage(),
                                     vgeuImage->getFormat(), 0, mipLevels,
                                     vk::ImageLayout::eUndefined,
                                     vk::ImageLayout::eTransferDstOptimal);
                      cmdBuffer.copyBufferToImage(
                          stagingBuffer.getBuffer(), vgeuImage->getImage(),
                          vk::ImageLayout::eTransferDstOptimal, region);
                    });
    }
    oneTimeSubmit(device, commandPool, transferQueue,
                  [this](const vk::raii::CommandBuffer& cmdBuffer) {
                    this->generateMipmaps(cmdBuffer);
                  });

  } else {
    // TODO: loading texture using KTX format
    assert(false && "failed: not yet implemented KTX format texture loading");
  }
  createSampler(device);
  updateDescriptorInfo();
}

void Texture::generateMipmaps(const vk::raii::CommandBuffer& cmdBuffer) {
  uint32_t mipWidth = width;
  uint32_t mipHeight = height;
  for (uint32_t i = 1; i < mipLevels; i++) {
    setImageLayout(cmdBuffer, vgeuImage->getImage(), vgeuImage->getFormat(),
                   i - 1, 1, vk::ImageLayout::eTransferDstOptimal,
                   vk::ImageLayout::eTransferSrcOptimal);
    uint32_t nextMipWidth = mipWidth > 1 ? mipWidth / 2 : 1u;
    uint32_t nextMipHeight = mipHeight > 1 ? mipHeight / 2 : 1u;

    vk::ImageBlit region(
        vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i - 1, 0,
                                   1},
        {
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{static_cast<int>(mipWidth),
                         static_cast<int>(mipHeight), 1},
        },
        vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i, 0, 1},
        {
            vk::Offset3D{0, 0, 0},
            vk::Offset3D{static_cast<int>(nextMipWidth),
                         static_cast<int>(nextMipHeight), 1},
        });

    cmdBuffer.blitImage(
        vgeuImage->getImage(), vk::ImageLayout::eTransferSrcOptimal,
        vgeuImage->getImage(), vk::ImageLayout::eTransferDstOptimal, region,
        vk::Filter::eLinear);
    setImageLayout(cmdBuffer, vgeuImage->getImage(), vgeuImage->getFormat(),
                   i - 1, 1, vk::ImageLayout::eTransferSrcOptimal,
                   vk::ImageLayout::eShaderReadOnlyOptimal);
    mipWidth = nextMipWidth;
    mipHeight = nextMipHeight;
  }
  setImageLayout(cmdBuffer, vgeuImage->getImage(), vgeuImage->getFormat(),
                 mipLevels - 1, 1, vk::ImageLayout::eTransferDstOptimal,
                 vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Texture::createEmptyTexture(const vk::raii::Device& device,
                                 VmaAllocator allocator,
                                 const vk::raii::Queue& transferQueue,
                                 const vk::raii::CommandPool& commandPool) {
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
        vk::ImageLayout::eUndefined, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
        VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        vk::ImageAspectFlagBits::eColor, mipLevels);

    // NOTE: 0 for buffer packed tightly
    vk::BufferImageCopy region(
        0, 0, 0,
        vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        vk::Offset3D{0, 0, 0}, vk::Extent3D{width, height, 1});
    // layout transition
    oneTimeSubmit(device, commandPool, transferQueue,
                  [&](const vk::raii::CommandBuffer& cmdBuffer) {
                    setImageLayout(cmdBuffer, vgeuImage->getImage(),
                                   vgeuImage->getFormat(), 0, mipLevels,
                                   vk::ImageLayout::eUndefined,
                                   vk::ImageLayout::eTransferDstOptimal);
                    cmdBuffer.copyBufferToImage(
                        stagingBuffer.getBuffer(), vgeuImage->getImage(),
                        vk::ImageLayout::eTransferDstOptimal, region);
                    setImageLayout(cmdBuffer, vgeuImage->getImage(),
                                   vgeuImage->getFormat(), 0, mipLevels,
                                   vk::ImageLayout::eTransferDstOptimal,
                                   vk::ImageLayout::eShaderReadOnlyOptimal);
                  });
    imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  }
  createSampler(device);
  updateDescriptorInfo();
}

void Texture::createSampler(const vk::raii::Device& device) {
  // NOTE: maxAnisotorpy fixed. may get it from physical Device.
  vk::SamplerCreateInfo samplerCI(
      vk::SamplerCreateFlags{}, vk::Filter::eLinear, vk::Filter::eLinear,
      vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode::eRepeat, vk::SamplerAddressMode::eRepeat, 0.f,
      true, 8.0f, false, vk::CompareOp::eNever, 0.f,
      static_cast<float>(mipLevels), vk::BorderColor::eFloatOpaqueWhite);

  sampler = vk::raii::Sampler(device, samplerCI);
}

void Texture::updateDescriptorInfo() {
  descriptorInfo.sampler = *sampler;
  descriptorInfo.imageView = *vgeuImage->getImageView();
  descriptorInfo.imageLayout = imageLayout;
}
Model::Model(const vk::raii::Device& device, VmaAllocator allocator,
             const vk::raii::Queue& transferQueue,
             const vk::raii::CommandPool& commandPool)
    : device(device),
      allocator(allocator),
      transferQueue(transferQueue),
      commandPool(commandPool) {}

Model::~Model() {}

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

  // NOTE: SetPreserveimageChannels false by default -> 4channel
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
        material.createDescriptorSet(device, descriptorPool,
                                     descriptorSetLayoutImage,
                                     descriptorBindingFlags);
      }
    }
  }
}

void Model::loadImages(tinygltf::Model& gltfModel) {
  for (tinygltf::Image& gltfImage : gltfModel.images) {
    textures.push_back(std::make_unique<Texture>(
        gltfImage, path, device, allocator, transferQueue, commandPool));
  }
  // Create an empty texture to be used for empty material images
  emptyTexture =
      std::make_unique<Texture>(device, allocator, transferQueue, commandPool);
}

void Model::loadMaterials(tinygltf::Model& gltfModel) {
  for (tinygltf::Material& mat : gltfModel.materials) {
    Material& material = materials.emplace_back();
    if (mat.values.find("baseColorTexture") != mat.values.end()) {
      material.baseColorTexture = getTexture(
          gltfModel.textures[mat.values["baseColorTexture"].TextureIndex()]
              .source);
    }
    // Metallic roughness workflow
    if (mat.values.find("metallicRoughnessTexture") != mat.values.end()) {
      material.metallicRoughnessTexture = getTexture(
          gltfModel
              .textures[mat.values["metallicRoughnessTexture"].TextureIndex()]
              .source);
    }
    if (mat.values.find("roughnessFactor") != mat.values.end()) {
      material.roughnessFactor =
          static_cast<float>(mat.values["roughnessFactor"].Factor());
    }
    if (mat.values.find("metallicFactor") != mat.values.end()) {
      material.metallicFactor =
          static_cast<float>(mat.values["metallicFactor"].Factor());
    }
    if (mat.values.find("baseColorFactor") != mat.values.end()) {
      material.baseColorFactor =
          glm::make_vec4(mat.values["baseColorFactor"].ColorFactor().data());
    }
    if (mat.additionalValues.find("normalTexture") !=
        mat.additionalValues.end()) {
      material.normalTexture = getTexture(
          gltfModel
              .textures[mat.additionalValues["normalTexture"].TextureIndex()]
              .source);
    } else {
      material.normalTexture = emptyTexture.get();
    }
    if (mat.additionalValues.find("emissiveTexture") !=
        mat.additionalValues.end()) {
      material.emissiveTexture = getTexture(
          gltfModel
              .textures[mat.additionalValues["emissiveTexture"].TextureIndex()]
              .source);
    }
    if (mat.additionalValues.find("occlusionTexture") !=
        mat.additionalValues.end()) {
      material.occlusionTexture = getTexture(
          gltfModel
              .textures[mat.additionalValues["occlusionTexture"].TextureIndex()]
              .source);
    }
    if (mat.additionalValues.find("alphaMode") != mat.additionalValues.end()) {
      tinygltf::Parameter param = mat.additionalValues["alphaMode"];
      if (param.string_value == "BLEND") {
        material.alphaMode = Material::AlphaMode::kALPHAMODE_BLEND;
      }
      if (param.string_value == "MASK") {
        material.alphaMode = Material::AlphaMode::kALPHAMODE_MASK;
      }
    }
    if (mat.additionalValues.find("alphaCutoff") !=
        mat.additionalValues.end()) {
      material.alphaCutoff =
          static_cast<float>(mat.additionalValues["alphaCutoff"].Factor());
    }
  }
  // Push a default material at the end of the list for meshes with no material
  // assigned
  materials.emplace_back();
}
void Model::loadNode(Node* parent, const tinygltf::Node& gltfNode,
                     uint32_t nodeIndex, const tinygltf::Model& gltfModel,
                     std::vector<uint32_t>& indices,
                     std::vector<Vertex>& vertices, float globalscale) {
  // To be moved into nodes or children of the other node;
  std::unique_ptr<Node> newNode(new Node{});
  newNode->index = nodeIndex;
  newNode->parent = parent;
  newNode->name = gltfNode.name;
  newNode->skinIndex = gltfNode.skin;
  newNode->matrix = glm::mat4(1.0f);

  // Generate local node matrix
  glm::vec3 translation = glm::vec3(0.0f);
  if (gltfNode.translation.size() == 3) {
    translation = glm::make_vec3(gltfNode.translation.data());
    newNode->translation = translation;
  }
  glm::mat4 rotation = glm::mat4(1.0f);
  if (gltfNode.rotation.size() == 4) {
    glm::quat q = glm::make_quat(gltfNode.rotation.data());
    newNode->rotation = glm::mat4(q);
  }
  glm::vec3 scale = glm::vec3(1.0f);
  if (gltfNode.scale.size() == 3) {
    scale = glm::make_vec3(gltfNode.scale.data());
    newNode->scale = scale;
  }
  if (gltfNode.matrix.size() == 16) {
    newNode->matrix = glm::make_mat4x4(gltfNode.matrix.data());
    if (globalscale != 1.0f) {
      // TODO: check why commented
      // newNode->matrix = glm::scale(newNode->matrix, glm::vec3(globalscale));
    }
  };

  // Node with children
  if (gltfNode.children.size() > 0) {
    for (auto i = 0; i < gltfNode.children.size(); i++) {
      loadNode(newNode.get(), gltfModel.nodes[gltfNode.children[i]],
               gltfNode.children[i], gltfModel, indices, vertices, globalscale);
    }
  }

  // Node contains mesh data
  if (gltfNode.mesh > -1) {
    const tinygltf::Mesh gltfMesh = gltfModel.meshes[gltfNode.mesh];
    newNode->mesh = std::make_unique<Mesh>(allocator, newNode->matrix);
    newNode->mesh->name = gltfMesh.name;
    for (size_t j = 0; j < gltfMesh.primitives.size(); j++) {
      const tinygltf::Primitive& gltfPrimitive = gltfMesh.primitives[j];
      if (gltfPrimitive.indices < 0) {
        continue;
      }
      uint32_t indexStart = static_cast<uint32_t>(indices.size());
      uint32_t vertexStart = static_cast<uint32_t>(vertices.size());
      uint32_t indexCount = 0;
      uint32_t vertexCount = 0;
      glm::vec3 posMin{};
      glm::vec3 posMax{};
      bool hasSkin = false;
      // Vertices
      {
        const float* bufferPos = nullptr;
        const float* bufferNormals = nullptr;
        const float* bufferTexCoords = nullptr;
        const float* bufferColors = nullptr;
        const float* bufferTangents = nullptr;
        uint32_t numColorComponents;
        const uint16_t* bufferJoints = nullptr;
        const float* bufferWeights = nullptr;

        // Position attribute is required
        assert(gltfPrimitive.attributes.find("POSITION") !=
               gltfPrimitive.attributes.end());

        const tinygltf::Accessor& posAccessor =
            gltfModel
                .accessors[gltfPrimitive.attributes.find("POSITION")->second];
        const tinygltf::BufferView& posView =
            gltfModel.bufferViews[posAccessor.bufferView];
        bufferPos = reinterpret_cast<const float*>(
            &(gltfModel.buffers[posView.buffer]
                  .data[posAccessor.byteOffset + posView.byteOffset]));
        posMin = glm::vec3(posAccessor.minValues[0], posAccessor.minValues[1],
                           posAccessor.minValues[2]);
        posMax = glm::vec3(posAccessor.maxValues[0], posAccessor.maxValues[1],
                           posAccessor.maxValues[2]);

        if (gltfPrimitive.attributes.find("NORMAL") !=
            gltfPrimitive.attributes.end()) {
          const tinygltf::Accessor& normAccessor =
              gltfModel
                  .accessors[gltfPrimitive.attributes.find("NORMAL")->second];
          const tinygltf::BufferView& normView =
              gltfModel.bufferViews[normAccessor.bufferView];
          bufferNormals = reinterpret_cast<const float*>(
              &(gltfModel.buffers[normView.buffer]
                    .data[normAccessor.byteOffset + normView.byteOffset]));
        }

        if (gltfPrimitive.attributes.find("TEXCOORD_0") !=
            gltfPrimitive.attributes.end()) {
          const tinygltf::Accessor& uvAccessor =
              gltfModel.accessors[gltfPrimitive.attributes.find("TEXCOORD_0")
                                      ->second];
          const tinygltf::BufferView& uvView =
              gltfModel.bufferViews[uvAccessor.bufferView];
          bufferTexCoords = reinterpret_cast<const float*>(
              &(gltfModel.buffers[uvView.buffer]
                    .data[uvAccessor.byteOffset + uvView.byteOffset]));
        }

        if (gltfPrimitive.attributes.find("COLOR_0") !=
            gltfPrimitive.attributes.end()) {
          const tinygltf::Accessor& colorAccessor =
              gltfModel
                  .accessors[gltfPrimitive.attributes.find("COLOR_0")->second];
          const tinygltf::BufferView& colorView =
              gltfModel.bufferViews[colorAccessor.bufferView];
          // Color buffer are either of type vec3 or vec4
          numColorComponents =
              colorAccessor.type == TINYGLTF_PARAMETER_TYPE_FLOAT_VEC3 ? 3 : 4;
          bufferColors = reinterpret_cast<const float*>(
              &(gltfModel.buffers[colorView.buffer]
                    .data[colorAccessor.byteOffset + colorView.byteOffset]));
        }

        if (gltfPrimitive.attributes.find("TANGENT") !=
            gltfPrimitive.attributes.end()) {
          const tinygltf::Accessor& tangentAccessor =
              gltfModel
                  .accessors[gltfPrimitive.attributes.find("TANGENT")->second];
          const tinygltf::BufferView& tangentView =
              gltfModel.bufferViews[tangentAccessor.bufferView];
          bufferTangents = reinterpret_cast<const float*>(&(
              gltfModel.buffers[tangentView.buffer]
                  .data[tangentAccessor.byteOffset + tangentView.byteOffset]));
        }

        // Skinning
        // Joints
        if (gltfPrimitive.attributes.find("JOINTS_0") !=
            gltfPrimitive.attributes.end()) {
          const tinygltf::Accessor& jointAccessor =
              gltfModel
                  .accessors[gltfPrimitive.attributes.find("JOINTS_0")->second];
          const tinygltf::BufferView& jointView =
              gltfModel.bufferViews[jointAccessor.bufferView];
          bufferJoints = reinterpret_cast<const uint16_t*>(
              &(gltfModel.buffers[jointView.buffer]
                    .data[jointAccessor.byteOffset + jointView.byteOffset]));
        }

        if (gltfPrimitive.attributes.find("WEIGHTS_0") !=
            gltfPrimitive.attributes.end()) {
          const tinygltf::Accessor& uvAccessor =
              gltfModel.accessors[gltfPrimitive.attributes.find("WEIGHTS_0")
                                      ->second];
          const tinygltf::BufferView& uvView =
              gltfModel.bufferViews[uvAccessor.bufferView];
          bufferWeights = reinterpret_cast<const float*>(
              &(gltfModel.buffers[uvView.buffer]
                    .data[uvAccessor.byteOffset + uvView.byteOffset]));
        }

        hasSkin = (bufferJoints && bufferWeights);

        vertexCount = static_cast<uint32_t>(posAccessor.count);

        for (size_t v = 0; v < posAccessor.count; v++) {
          Vertex vert{};
          vert.pos = glm::vec4(glm::make_vec3(&bufferPos[v * 3]), 1.0f);
          vert.normal = glm::normalize(
              glm::vec3(bufferNormals ? glm::make_vec3(&bufferNormals[v * 3])
                                      : glm::vec3(0.0f)));
          vert.uv = bufferTexCoords ? glm::make_vec2(&bufferTexCoords[v * 2])
                                    : glm::vec3(0.0f);
          if (bufferColors) {
            switch (numColorComponents) {
              case 3:
                vert.color =
                    glm::vec4(glm::make_vec3(&bufferColors[v * 3]), 1.0f);
              case 4:
                vert.color = glm::make_vec4(&bufferColors[v * 4]);
            }
          } else {
            vert.color = glm::vec4(1.0f);
          }
          vert.tangent = bufferTangents
                             ? glm::vec4(glm::make_vec4(&bufferTangents[v * 4]))
                             : glm::vec4(0.0f);
          vert.joint0 = hasSkin
                            ? glm::vec4(glm::make_vec4(&bufferJoints[v * 4]))
                            : glm::vec4(0.0f);
          vert.weight0 =
              hasSkin ? glm::make_vec4(&bufferWeights[v * 4]) : glm::vec4(0.0f);
          vertices.push_back(vert);
        }
      }
      // Indices
      {
        const tinygltf::Accessor& accessor =
            gltfModel.accessors[gltfPrimitive.indices];
        const tinygltf::BufferView& bufferView =
            gltfModel.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = gltfModel.buffers[bufferView.buffer];

        indexCount = static_cast<uint32_t>(accessor.count);

        switch (accessor.componentType) {
          case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT: {
            uint32_t* buf = new uint32_t[accessor.count];
            memcpy(buf,
                   &buffer.data[accessor.byteOffset + bufferView.byteOffset],
                   accessor.count * sizeof(uint32_t));
            for (size_t index = 0; index < accessor.count; index++) {
              indices.push_back(buf[index] + vertexStart);
            }
            delete[] buf;
            break;
          }
          case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT: {
            uint16_t* buf = new uint16_t[accessor.count];
            memcpy(buf,
                   &buffer.data[accessor.byteOffset + bufferView.byteOffset],
                   accessor.count * sizeof(uint16_t));
            for (size_t index = 0; index < accessor.count; index++) {
              indices.push_back(buf[index] + vertexStart);
            }
            delete[] buf;
            break;
          }
          case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE: {
            uint8_t* buf = new uint8_t[accessor.count];
            memcpy(buf,
                   &buffer.data[accessor.byteOffset + bufferView.byteOffset],
                   accessor.count * sizeof(uint8_t));
            for (size_t index = 0; index < accessor.count; index++) {
              indices.push_back(buf[index] + vertexStart);
            }
            delete[] buf;
            break;
          }
          default:
            std::cerr << "Index component type " << accessor.componentType
                      << " not supported!" << std::endl;
            return;
        }
      }

      newNode->mesh->primitives.push_back(std::make_unique<Primitive>(
          indexStart, indexCount,
          gltfPrimitive.material > -1 ? materials[gltfPrimitive.material]
                                      : materials.back()));

      std::unique_ptr<Primitive>& newPrimitive =
          newNode->mesh->primitives.back();
      newPrimitive->firstVertex = vertexStart;
      newPrimitive->vertexCount = vertexCount;
      newPrimitive->setDimensions(posMin, posMax);
    }
  }
  linearNodes.push_back(newNode.get());
  if (parent != nullptr) {
    parent->children.push_back(std::move(newNode));
  } else {
    nodes.push_back(std::move(newNode));
  }
}

const Texture* Model::getTexture(uint32_t index) const {
  if (index < textures.size()) {
    return textures[index].get();
  }
  return nullptr;
}

void Model::draw(const vk::raii::CommandBuffer& cmdBuffer,
                 RenderFlags renderFlags, vk::PipelineLayout pipelineLayout,
                 uint32_t bindImageSet) {
  if (!buffersBound) {
    bindBuffers(cmdBuffer);
  }
  for (const auto& node : nodes) {
    drawNode(node.get(), cmdBuffer, renderFlags, pipelineLayout, bindImageSet);
  }
}

void Model::drawNode(const Node* node, const vk::raii::CommandBuffer& cmdBuffer,
                     RenderFlags renderFlags, vk::PipelineLayout pipelineLayout,
                     uint32_t bindImageSet) {
  if (node->mesh) {
    for (const auto& primitive : node->mesh->primitives) {
      bool skip = false;
      const Material& material = primitive->material;
      if (renderFlags & RenderFlagBits::kRenderOpaqueNodes) {
        skip = (material.alphaMode != Material::AlphaMode::kALPHAMODE_OPAQUE);
      }
      if (renderFlags & RenderFlagBits::kRenderAlphaMaskedNodes) {
        skip = (material.alphaMode != Material::AlphaMode::kALPHAMODE_MASK);
      }
      if (renderFlags & RenderFlagBits::kRenderAlphaBlendedNodes) {
        skip = (material.alphaMode != Material::AlphaMode::kALPHAMODE_BLEND);
      }
      if (!skip) {
        if (renderFlags & RenderFlagBits::kBindImages) {
          cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                       pipelineLayout, bindImageSet,
                                       *material.descriptorSet, nullptr);
        }
        cmdBuffer.drawIndexed(primitive->indexCount, 1, primitive->firstIndex,
                              0, 0);
      }
    }
  }
  for (const auto& child : node->children) {
    drawNode(child.get(), cmdBuffer, renderFlags, pipelineLayout, bindImageSet);
  }
}

void Model::bindBuffers(const vk::raii::CommandBuffer& cmdBuffer) {
  vk::DeviceSize offset(0);
  cmdBuffer.bindVertexBuffers(0, vertexBuffer->getBuffer(), offset);
  buffersBound = true;
}

void Model::prepareNodeDescriptor(
    const Node* node,
    const vk::raii::DescriptorSetLayout& descriptorSetLayout) {
  if (node->mesh) {
    vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                            *descriptorSetLayout);
    node->mesh->descriptorSet =
        std::move(vk::raii::DescriptorSets(device, allocInfo).front());

    vk::WriteDescriptorSet writeDescriptorSet(
        *node->mesh->descriptorSet, 0, 0, vk::DescriptorType::eUniformBuffer,
        nullptr, node->mesh->descriptorInfo);
    device.updateDescriptorSets(writeDescriptorSet, nullptr);
  }
  for (const auto& child : node->children) {
    prepareNodeDescriptor(child.get(), descriptorSetLayout);
  }
}

void Model::getNodeDimensions(const Node* node, glm::vec3& min,
                              glm::vec3& max) const {
  if (node->mesh) {
    for (const auto& primitive : node->mesh->primitives) {
      glm::vec4 locMin =
          glm::vec4(primitive->dimensions.min, 1.0f) * node->getMatrix();
      glm::vec4 locMax =
          glm::vec4(primitive->dimensions.max, 1.0f) * node->getMatrix();
      if (locMin.x < min.x) {
        min.x = locMin.x;
      }
      if (locMin.y < min.y) {
        min.y = locMin.y;
      }
      if (locMin.z < min.z) {
        min.z = locMin.z;
      }
      if (locMax.x > max.x) {
        max.x = locMax.x;
      }
      if (locMax.y > max.y) {
        max.y = locMax.y;
      }
      if (locMax.z > max.z) {
        max.z = locMax.z;
      }
    }
  }
  for (const auto& child : node->children) {
    getNodeDimensions(child.get(), min, max);
  }
}

void Model::setSceneDimensions() {
  dimensions.min = glm::vec3(std::numeric_limits<float>::max());
  dimensions.max = glm::vec3(-std::numeric_limits<float>::max());
  for (const auto& node : nodes) {
    getNodeDimensions(node.get(), dimensions.min, dimensions.max);
  }
  dimensions.size = dimensions.max - dimensions.min;
  dimensions.center = (dimensions.min + dimensions.max) / 2.0f;
  dimensions.radius = glm::distance(dimensions.min, dimensions.max) / 2.0f;
}

const Node* Model::findNode(const Node* parent, uint32_t index) const {
  const Node* nodeFound = nullptr;
  if (parent->index == index) {
    return parent;
  }
  for (const auto& child : parent->children) {
    nodeFound = findNode(child.get(), index);
    if (nodeFound) {
      break;
    }
  }
  return nodeFound;
}

const Node* Model::nodeFromIndex(uint32_t index) const {
  const Node* nodeFound = nullptr;
  for (const auto& node : nodes) {
    nodeFound = findNode(node.get(), index);
    if (nodeFound) {
      break;
    }
  }
  return nodeFound;
}

void Material::createDescriptorSet(
    const vk::raii::Device& device,
    const vk::raii::DescriptorPool& descriptorPool,
    const vk::raii::DescriptorSetLayout& descriptorSetLayout,
    DescriptorBindingFlags descriptorBindingFlags) {
  vk::DescriptorSetAllocateInfo allocInfo(*descriptorPool,
                                          *descriptorSetLayout);
  descriptorSet =
      std::move(vk::raii::DescriptorSets(device, allocInfo).front());

  // TODO: check unused.
  std::vector<vk::DescriptorImageInfo> descriptorInfos{};
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets{};
  // writeDescriptorSets.reserve(2);
  if (descriptorBindingFlags & DescriptorBindingFlagBits::kImageBaseColor) {
    descriptorInfos.push_back(baseColorTexture->descriptorInfo);
    writeDescriptorSets.emplace_back(
        *descriptorSet, static_cast<uint32_t>(writeDescriptorSets.size()), 0,
        vk::DescriptorType::eCombinedImageSampler,
        baseColorTexture->descriptorInfo, nullptr);
  }
  if (normalTexture &&
      descriptorBindingFlags & DescriptorBindingFlagBits::kImageNormalMap) {
    descriptorInfos.push_back(normalTexture->descriptorInfo);
    writeDescriptorSets.emplace_back(
        *descriptorSet, static_cast<uint32_t>(writeDescriptorSets.size()), 0,
        vk::DescriptorType::eCombinedImageSampler,
        normalTexture->descriptorInfo, nullptr);
  }
  device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

void Primitive::setDimensions(glm::vec3 min, glm::vec3 max) {
  dimensions.min = min;
  dimensions.max = max;
  dimensions.size = max - min;
  dimensions.center = (min + max) / 2.0f;
  dimensions.radius = glm::distance(min, max) / 2.0f;
}

Mesh::Mesh(VmaAllocator allocator, glm::mat4 matrix) {
  uniformBlock.matrix = matrix;
  uniformBuffer = std::make_unique<vgeu::VgeuBuffer>(
      allocator, sizeof(uniformBlock), 1,
      vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT |
          VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT);
  descriptorInfo = uniformBuffer->descriptorInfo();
}

glm::mat4 Node::localMatrix() const {
  return glm::translate(glm::mat4(1.0f), translation) * glm::mat4(rotation) *
         glm::scale(glm::mat4(1.0f), scale) * matrix;
}

glm::mat4 Node::getMatrix() const {
  glm::mat4 m = localMatrix();
  const Node* p = parent;
  while (p) {
    m = p->localMatrix() * m;
    p = p->parent;
  }
  return m;
}

void Node::update() {
  if (mesh) {
    glm::mat4 m = getMatrix();
    // TODO: skin
    if (/*skin*/ false) {
    } else {
      memcpy(mesh->uniformBuffer->getMappedData(), &m, sizeof(glm::mat4));
    }
  }

  for (auto& child : children) {
    child->update();
  }
}

vk::VertexInputBindingDescription Vertex::getInputBindingDescription(
    uint32_t binding) {
  return vk::VertexInputBindingDescription(binding, sizeof(Vertex),
                                           vk::VertexInputRate::eVertex);
  ;
}

vk::VertexInputAttributeDescription Vertex::getInputAttributeDescription(
    uint32_t binding, uint32_t location, VertexComponent component) {
  switch (component) {
    case VertexComponent::kPosition:
      return vk::VertexInputAttributeDescription(location, binding,
                                                 vk::Format::eR32G32B32Sfloat,
                                                 offsetof(Vertex, pos));
    case VertexComponent::kNormal:
      return vk::VertexInputAttributeDescription(location, binding,
                                                 vk::Format::eR32G32B32Sfloat,
                                                 offsetof(Vertex, normal));
    case VertexComponent::kUV:
      return vk::VertexInputAttributeDescription(
          location, binding, vk::Format::eR32G32Sfloat, offsetof(Vertex, uv));
    case VertexComponent::kColor:
      return vk::VertexInputAttributeDescription(
          location, binding, vk::Format::eR32G32B32A32Sfloat,
          offsetof(Vertex, color));
    case VertexComponent::kTangent:
      return vk::VertexInputAttributeDescription(
          location, binding, vk::Format::eR32G32B32A32Sfloat,
          offsetof(Vertex, tangent));
    case VertexComponent::kJoint0:
      return vk::VertexInputAttributeDescription(
          location, binding, vk::Format::eR32G32B32A32Sfloat,
          offsetof(Vertex, joint0));
    case VertexComponent::kWeight0:
      return vk::VertexInputAttributeDescription(
          location, binding, vk::Format::eR32G32B32A32Sfloat,
          offsetof(Vertex, weight0));
    default:
      assert(false && "failed: to get vertex input attribute description");
      return vk::VertexInputAttributeDescription();
  }
}

std::vector<vk::VertexInputAttributeDescription>
Vertex::getInputAttributeDescriptions(
    uint32_t binding, const std::vector<VertexComponent>& components) {
  std::vector<vk::VertexInputAttributeDescription> result;
  uint32_t location = 0;
  for (VertexComponent component : components) {
    result.push_back(
        Vertex::getInputAttributeDescription(binding, location, component));
    location++;
  }
  return result;
}

vk::PipelineVertexInputStateCreateInfo Vertex::getPipelineVertexInputState(
    const std::vector<VertexComponent> components) {
  vk::VertexInputBindingDescription vertexInputeBindingDescription =
      getInputBindingDescription(0);
  std::vector<vk::VertexInputAttributeDescription>
      vertexInputAttributeDescriptions =
          getInputAttributeDescriptions(0, components);
  return vk::PipelineVertexInputStateCreateInfo(
      vk::PipelineVertexInputStateCreateFlags{}, vertexInputeBindingDescription,
      vertexInputAttributeDescriptions);
}

}  // namespace glTF
}  // namespace vgeu