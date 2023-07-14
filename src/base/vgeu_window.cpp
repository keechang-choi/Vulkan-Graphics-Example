#include "vgeu_window.hpp"

namespace vgeu {
VgeuWindow::VgeuWindow(int w, int h, std::string name)
    : width{w}, height{h}, windowName{name} {
  initWindow();
}

VgeuWindow::~VgeuWindow() {
  glfwDestroyWindow(window);
  glfwTerminate();
}

void VgeuWindow::initWindow() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  window =
      glfwCreateWindow(width, height, windowName.c_str(), nullptr, nullptr);
  glfwSetWindowUserPointer(window, this);
  glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void VgeuWindow::framebufferResizeCallback(GLFWwindow* window, int width,
                                           int height) {
  auto vgeuWindow =
      reinterpret_cast<VgeuWindow*>(glfwGetWindowUserPointer(window));
  vgeuWindow->framebufferResized = true;
  vgeuWindow->width = width;
  vgeuWindow->height = height;
}

void VgeuWindow::createWindowSurface(VkInstance instance,
                                     VkSurfaceKHR* surface) {
  if (glfwCreateWindowSurface(instance, window, nullptr, surface) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface");
  }
}

}  // namespace vgeu