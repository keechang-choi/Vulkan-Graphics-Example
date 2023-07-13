#pragma once

// libs
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

// std
#include <string>

// Vulkan Graphics Example Utils
namespace vgeu {
class VgeuWindow {
 public:
  VgeuWindow(uint32_t w, uint32_t h, std::string name);
  ~VgeuWindow();

  VgeuWindow(const VgeuWindow&) = delete;
  VgeuWindow& operator=(const VgeuWindow&) = delete;

  bool shouldClose() { return glfwWindowShouldClose(window); }
  vk::Extent2D getExtent() { return {width, height}; }

 private:
  static void framebufferResizeCallback(GLFWwindow* window, uint32_t width,
                                        uint32_t height);
  void initWindow();
  std::string windowName;
  GLFWwindow* window;
  uint32_t width, height;
  bool framebufferResized = false;
};
}  // namespace vgeu