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
  VgeuWindow(int w, int h, std::string name);
  ~VgeuWindow();

  VgeuWindow(const VgeuWindow&) = delete;
  VgeuWindow& operator=(const VgeuWindow&) = delete;

  bool shouldClose() { return glfwWindowShouldClose(window); }
  vk::Extent2D getExtent() {
    return {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
  }
  bool isPaused() { return paused; }

  void createWindowSurface(VkInstance instance, VkSurfaceKHR* surface);
  GLFWwindow* getGLFWwindow() const { return window; };

 private:
  static void framebufferResizeCallback(GLFWwindow* window, int width,
                                        int height);
  static void keyCallback(GLFWwindow* window, int key, int scancode, int action,
                          int mods);

  void initWindow();
  std::string windowName;
  GLFWwindow* window;
  int width, height;
  bool framebufferResized = false;
  bool paused = false;
};
}  // namespace vgeu