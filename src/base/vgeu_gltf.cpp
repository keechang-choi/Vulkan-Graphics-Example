#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vgeu_gltf.hpp"

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
bool loadImageDataFunc(tinygltf::Image* image, const int imageIndex,
                       std::string* error, std::string* warning, int req_width,
                       int req_height, const unsigned char* bytes, int size,
                       void* userData) {
  if (isKtx(*image)) {
    return true;
  }
  return tinygltf::LoadImageData(image, imageIndex, error, warning, req_width,
                                 req_height, bytes, size, userData);
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
void Texture::updateDescriptorInfo() {
  descriptorInfo.sampler = *sampler;
  descriptorInfo.imageView = *vgeuImage->getImageView();
  descriptorInfo.imageLayout = imageLayout;
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

Model::Model(const vk::raii::Device& device,
             const vk::raii::PhysicalDevice& physicalDevice,
             VmaAllocator allocator, const vk::raii::Queue& transferQueue)
    : device(device),
      physicalDevice(physicalDevice),
      allocator(allocator),
      transferQueue(transferQueue) {}
}  // namespace glTF
}  // namespace vgeu