#pragma once

// libs
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
//
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

// std
#include <string>

// Vulkan Graphics Example Utils
namespace vgeu {
struct MouseData {
  bool left = false;
  bool right = false;
  bool middle = false;
  glm::vec2 mousePos;
};

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
  MouseData getMouseInputs() { return mouseData; }

  void createWindowSurface(VkInstance instance, VkSurfaceKHR* surface);
  GLFWwindow* getGLFWwindow() const { return window; };
  void waitMinimized();

 private:
  static void framebufferResizeCallback(GLFWwindow* window, int width,
                                        int height);
  static void keyCallback(GLFWwindow* window, int key, int scancode, int action,
                          int mods);
  static void mouseButtonCallback(GLFWwindow* window, int button, int action,
                                  int mods);
  static void mousePositionCallback(GLFWwindow* window, double xpos,
                                    double ypos);

  void initWindow();
  std::string windowName;
  GLFWwindow* window;
  int width, height;
  bool framebufferResized = false;
  bool paused = false;
  MouseData mouseData;
};
}  // namespace vgeu