/*

UI Overlay using ImGui

code base from
https://github.com/SaschaWillems/Vulkan/blob/master/base/VulkanUIOverlay.h

*/

#pragma once

#include "vgeu_buffer.hpp"
#include "vgeu_utils.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
//
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

#include "../../external/imgui/imgui.h"

// std
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iomanip>
#include <sstream>
#include <vector>

// TODO: add other platform/OS supports.

namespace vgeu {
class UIOverlay {
 public:
  UIOverlay();
  ~UIOverlay();
  UIOverlay(const UIOverlay&) = delete;
  UIOverlay& operator=(const UIOverlay&) = delete;

  void preparePipeline(const vk::raii::PipelineCache& pipelineCache,
                       const vk::raii::RenderPass& renderPass,
                       const vk::Format colorFormat,
                       const vk::Format depthFormat);
  void prepareResources();

  bool update();
  // CHECK: const ref?
  void draw(const vk::raii::CommandBuffer& commandBuffer);
  void resize(uint32_t width, uint32_t height);

  // CHECK: raii paradigm
  // void freeResources();

  bool header(const char* caption);
  bool checkBox(const char* caption, bool* value);
  bool checkBox(const char* caption, int32_t* value);
  bool radioButton(const char* caption, bool value);
  bool inputFloat(const char* caption, float* value, float step,
                  uint32_t precision);
  bool sliderFloat(const char* caption, float* value, float min, float max);
  bool sliderInt(const char* caption, int32_t* value, int32_t min, int32_t max);
  bool comboBox(const char* caption, int32_t* itemindex,
                std::vector<std::string> items);
  bool button(const char* caption);
  bool colorPicker(const char* caption, float* color);
  void text(const char* formatstr, ...);

 private:
  vk::raii::Device& device;
  vk::raii::Queue& queue;
  vk::SampleCountFlagBits rasterizationSamples = vk::SampleCountFlagBits::e1;
  uint32_t subpass = 0;

  vgeu::VgeuBuffer vertexBuffer;
  vgeu::VgeuBuffer indexBuffer;
  int32_t vertexCount = 0;
  int32_t indexCount = 0;

  std::vector<vk::PipelineShaderStageCreateInfo> shaders;

  vk::raii::DescriptorPool descriptorPool = nullptr;
  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::DescriptorSet descriptorSet = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline pipeline = nullptr;

  vgeu::ImageData fontImageData = nullptr;
  vk::raii::Sampler sampler = nullptr;

  struct PushConstBlock {
    glm::vec2 scale;
    glm::vec2 translate;
  } pushConstBlock;

  // TODO: getters and setter
  bool visible = true;
  bool updated = false;
  float scale = 1.0f;
};
}  // namespace vgeu