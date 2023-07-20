#include "triangle.hpp"

// std
#include <array>

namespace vge {
VgeExample::VgeExample() : VgeBase() {
  title = "First Triangle Example";
  // camera setup
  camera.setViewTarget(glm::vec3{0.0f, -2.f, -2.f}, glm::vec3{0.f, 0.f, 0.f});
  camera.setPerspectiveProjection(
      60.f, static_cast<float>(width) / static_cast<float>(height), 1.f, 256.f);
}
VgeExample::~VgeExample() {}
void VgeExample::render() {}
void VgeExample::prepare() {
  VgeBase::prepare();

  std::vector<Vertex> vertices{
      {{1.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f}},
      {{-1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}},
      {{0.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}},
  };

  vertexBuffer = std::make_unique<vgeu::VgeuBuffer>(
      globalAllocator, sizeof(Vertex), static_cast<uint32_t>(vertices.size()),
      vk::BufferUsageFlagBits::eVertexBuffer |
          vk::BufferUsageFlagBits::eTransferDst,
      VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);
}

void VgeExample::createUniformBuffers() {
  // NOTE: prevent move during vector element creation.
  uniformBuffers.reserve(MAX_CONCURRENT_FRAMES);
  for (int i = 0; i < MAX_CONCURRENT_FRAMES; i++) {
    uniformBuffers.push_back(std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator, sizeof(Ubo), 1,
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT));
  }
}

void VgeExample::createDescriptorSetLayout() {
  vk::DescriptorSetLayoutBinding layoutBinding(
      0, vk::DescriptorType::eUniformBuffer, 1,
      vk::ShaderStageFlagBits::eVertex);
  vk::DescriptorSetLayoutCreateInfo layoutCI({}, 1, &layoutBinding);
  descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutCI);

  vk::PipelineLayoutCreateInfo pipelineLayoutCI({}, *descriptorSetLayout);
  pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutCI);
}

void VgeExample::createPipelines() {
  auto vertCode =
      vgeu::readFile(getShadersPath() + "triangle/triangle.vert.spv");
  auto fragCode =
      vgeu::readFile(getShadersPath() + "triangle/triangle.frag.spv");
  // NOTE: after pipeline creation, shader modules can be destroyed.
  vk::raii::ShaderModule vertShaderModule =
      vgeu::createShaderModule(device, vertCode);
  vk::raii::ShaderModule fragShaderModule =
      vgeu::createShaderModule(device, fragCode);

  std::array<vk::PipelineShaderStageCreateInfo, 2> pipelineShaderStage = {
      vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                                        *vertShaderModule, "main", nullptr),
      vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                                        *vertShaderModule, "main", nullptr),
  };
  // vk::GraphicsPipelineCreateInfo pipelineCI();
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()