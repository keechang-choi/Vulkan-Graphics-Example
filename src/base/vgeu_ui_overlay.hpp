/*

UI Overlay using ImGui


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
//
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "../../external/imgui/backends/imgui_impl_glfw.h"
#include "../../external/imgui/backends/imgui_impl_vulkan.h"
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
  UIOverlay(const vk::raii::Device& device, GLFWwindow* window,
            const vk::raii::Instance& instance, const vk::raii::Queue& queue,
            const vk::raii::PhysicalDevice& physicalDevice,
            const vk::raii::RenderPass& renderPass,
            const vk::raii::PipelineCache& pipelineCache,
            const vk::raii::CommandPool& commandPool,
            const uint32_t minImageCount);
  ~UIOverlay();
  UIOverlay(const UIOverlay&) = delete;
  UIOverlay& operator=(const UIOverlay&) = delete;

  // buffers
  bool update();
  // CHECK: const ref?
  void draw(const vk::raii::CommandBuffer& cmdBuffer);
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

  bool isUpdated() { return updated; }
  bool isVisible() { return visible; }
  float getScale() { return scale; }
  void setUpdated(bool updated_) { updated = updated_; }
  void setVisible(bool visible_) { visible = visible_; }

 private:
  vk::raii::DescriptorPool descriptorPool = nullptr;

  std::unique_ptr<VgeuImage> fontImageData;
  vk::raii::Sampler sampler = nullptr;

  struct PushConstBlock {
    glm::vec2 scale;
    glm::vec2 translate;
  } pushConstBlock;

  bool visible = true;
  bool updated = false;
  float scale = 1.0f;
};
}  // namespace vgeu