#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "vgeu_gltf.hpp"

namespace vgeu {
namespace glTF {
vk::raii::DescriptorSetLayout descriptorSetLayoutImage = nullptr;
vk::raii::DescriptorSetLayout descriptorSetLayoutUbo = nullptr;
vk::MemoryPropertyFlags memoryPropertyFlags = vk::MemoryPropertyFlags(0);
uint32_t descriptorBindingFlags = DescriptorBindingFlags::ImageBaseColor;

}  // namespace glTF
}  // namespace vgeu