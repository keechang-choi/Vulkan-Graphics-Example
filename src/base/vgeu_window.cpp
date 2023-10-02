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
  glfwSetKeyCallback(window, keyCallback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetCursorPosCallback(window, mousePositionCallback);
}

void VgeuWindow::framebufferResizeCallback(GLFWwindow* window, int width,
                                           int height) {
  auto vgeuWindow =
      reinterpret_cast<VgeuWindow*>(glfwGetWindowUserPointer(window));
  vgeuWindow->framebufferResized = true;
  vgeuWindow->width = width;
  vgeuWindow->height = height;
}
void VgeuWindow::keyCallback(GLFWwindow* window, int key, int scancode,
                             int action, int mods) {
  if (key == GLFW_KEY_P && action == GLFW_PRESS) {
    auto vgeuWindow =
        reinterpret_cast<VgeuWindow*>(glfwGetWindowUserPointer(window));
    vgeuWindow->paused = !vgeuWindow->paused;
  }
}

void VgeuWindow::createWindowSurface(VkInstance instance,
                                     VkSurfaceKHR* surface) {
  if (glfwCreateWindowSurface(instance, window, nullptr, surface) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create window surface");
  }
}

void VgeuWindow::mouseButtonCallback(GLFWwindow* window, int button, int action,
                                     int mods) {
  auto vgeuWindow =
      reinterpret_cast<VgeuWindow*>(glfwGetWindowUserPointer(window));
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    vgeuWindow->mouseData.left = (action == GLFW_PRESS);
  } else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
    vgeuWindow->mouseData.middle = (action == GLFW_PRESS);
  } else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
    vgeuWindow->mouseData.right = (action == GLFW_PRESS);
  }
}

void VgeuWindow::mousePositionCallback(GLFWwindow* window, double xpos,
                                       double ypos) {
  auto vgeuWindow =
      reinterpret_cast<VgeuWindow*>(glfwGetWindowUserPointer(window));
  vgeuWindow->mouseData.mousePos.x = static_cast<float>(xpos);
  vgeuWindow->mouseData.mousePos.y = static_cast<float>(ypos);
}
void VgeuWindow::waitMinimized() {
  while (width == 0 || height == 0) {
    // width and height would be changed by resizeCallback
    glfwWaitEvents();
  }
}

}  // namespace vgeu