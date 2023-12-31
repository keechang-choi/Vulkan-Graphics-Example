#include "vgeu_ui_overlay.hpp"

#include "vgeu_utils.hpp"

// libs
#include "../../external/imgui/backends/imgui_impl_glfw.h"
#include "../../external/imgui/backends/imgui_impl_vulkan.h"
#include "../../external/imgui/imgui.h"

// libs
#include <vector>

namespace vgeu {

UIOverlay::UIOverlay(const vk::raii::Device& device, GLFWwindow* window,
                     const vk::raii::Instance& instance,
                     const vk::raii::Queue& queue,
                     const vk::raii::PhysicalDevice& physicalDevice,
                     const vk::raii::RenderPass& renderPass,
                     const vk::raii::PipelineCache& pipelineCache,
                     const vk::raii::CommandPool& commandPool,
                     const uint32_t minImageCount) {
  // create descriptor pool for imGui
  const uint32_t oversizedPoolValue = 1000;
  vk::DescriptorPoolSize();

  std::vector<vk::DescriptorPoolSize> poolSizes{
      {vk::DescriptorType::eSampler, oversizedPoolValue},
      {vk::DescriptorType::eCombinedImageSampler, oversizedPoolValue},
      {vk::DescriptorType::eSampledImage, oversizedPoolValue},
      {vk::DescriptorType::eStorageImage, oversizedPoolValue},
      {vk::DescriptorType::eUniformTexelBuffer, oversizedPoolValue},
      {vk::DescriptorType::eStorageTexelBuffer, oversizedPoolValue},
      {vk::DescriptorType::eUniformBuffer, oversizedPoolValue},
      {vk::DescriptorType::eStorageBuffer, oversizedPoolValue},
      {vk::DescriptorType::eUniformBufferDynamic, oversizedPoolValue},
      {vk::DescriptorType::eStorageBufferDynamic, oversizedPoolValue},
      {vk::DescriptorType::eInputAttachment, oversizedPoolValue}};

  vk::DescriptorPoolCreateInfo descriptorPoolCI(vk::DescriptorPoolCreateFlags(),
                                                oversizedPoolValue, poolSizes);
  descriptorPool = vk::raii::DescriptorPool(device, descriptorPoolCI);

  // imGui init
  ImGui::CreateContext();
  // TODO: check install_callbakcs?
  ImGui_ImplGlfw_InitForVulkan(window, true);

  ImGui_ImplVulkan_InitInfo initInfo{};
  initInfo.Instance = static_cast<VkInstance>(*instance);
  initInfo.PhysicalDevice = static_cast<VkPhysicalDevice>(*physicalDevice);
  initInfo.Device = static_cast<VkDevice>(*device);
  initInfo.Queue = static_cast<VkQueue>(*queue);
  initInfo.DescriptorPool = static_cast<VkDescriptorPool>(*descriptorPool);
  initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  initInfo.PipelineCache = static_cast<VkPipelineCache>(*pipelineCache);

  // TODO: check where those image count are used.
  // check erros for
  // Cannot call vkDestroyBuffer on VkBuffer  that is currently in use by a
  // command buffer:
  // if this image Count is less than the MAX_FRAMES_IN_FLIGHT.
  // https://github.com/ocornut/imgui/issues/3690
  initInfo.ImageCount = minImageCount;
  initInfo.MinImageCount = minImageCount;

  ImGui_ImplVulkan_Init(&initInfo, static_cast<VkRenderPass>(*renderPass));

  // font
  vgeu::oneTimeSubmit(device, commandPool, queue,
                      [&](const vk::raii::CommandBuffer& cmdBuffer) {
                        ImGui_ImplVulkan_CreateFontsTexture(
                            static_cast<VkCommandBuffer>(*cmdBuffer));
                      });
  ImGui_ImplVulkan_DestroyFontUploadObjects();
}
UIOverlay::~UIOverlay() {
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

bool UIOverlay::update() { return false; }

void UIOverlay::draw(const vk::raii::CommandBuffer& cmdBuffer) {
  void* p = ImGui::GetDrawData();
  assert(p != nullptr);
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),
                                  static_cast<VkCommandBuffer>(*cmdBuffer));
}

void UIOverlay::resize(uint32_t width, uint32_t height) {
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2((float)(width), (float)(height));
}

bool UIOverlay::header(const char* caption, bool open) {
  ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_DefaultOpen;
  if (!open) {
    flags = ImGuiTreeNodeFlags_CollapsingHeader;
  }
  return ImGui::CollapsingHeader(caption, flags);
}

bool UIOverlay::sliderFloat(const char* caption, float* value, float min,
                            float max) {
  bool res = ImGui::SliderFloat(caption, value, min, max);
  if (res) {
    updated = true;
  };
  return res;
}

bool UIOverlay::sliderInt(const char* caption, int32_t* value, int32_t min,
                          int32_t max) {
  bool res = ImGui::SliderInt(caption, value, min, max);
  if (res) {
    updated = true;
  };
  return res;
}

bool UIOverlay::button(const char* caption) {
  bool res = ImGui::Button(caption);
  if (res) {
    updated = true;
  };
  return res;
}

bool UIOverlay::colorPicker(const char* caption, float* color) {
  bool res = ImGui::ColorEdit4(caption, color, ImGuiColorEditFlags_NoInputs);
  if (res) {
    updated = true;
  };
  return res;
}

bool UIOverlay::inputFloat(const char* caption, float* value, float step,
                           const char* format) {
  bool res = ImGui::InputFloat(caption, value, step, step * 10.0f, format);
  if (res) {
    updated = true;
  };
  return res;
}

bool UIOverlay::inputInt(const char* caption, int32_t* value, int32_t step) {
  bool res = ImGui::InputInt(caption, value, step, step * 10.0f);
  if (res) {
    updated = true;
  };
  return res;
}

bool UIOverlay::radioButton(const char* caption, int32_t* v, int32_t button) {
  bool res = ImGui::RadioButton(caption, v, button);
  if (res) {
    updated = true;
  };
  return res;
}
}  // namespace vgeu