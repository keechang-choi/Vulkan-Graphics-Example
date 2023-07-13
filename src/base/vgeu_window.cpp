#include "vgeu_window.hpp"

namespace vgeu {
VgeuWindow::VgeuWindow(uint32_t w, uint32_t h, std::string name)
    : width{w}, height{h}, windowName{name} {
  initWindow();
}

VgeuWindow::~VgeuWindow() {
  glfwDestroyWindow(window);
  glfwTerminate();
}

void VgeuWindow::initWindow() {}

void VgeuWindow::framebufferResizeCallback(GLFWwindow* window, uint32_t width,
                                           uint32_t height) {
  auto vgeuWindow =
      reinterpret_cast<VgeuWindow*>(glfwGetWindowUserPointer(window));
  vgeuWindow->framebufferResized = true;
  vgeuWindow->width = width;
  vgeuWindow->height = height;
}  // namespace lve
}  // namespace vgeu