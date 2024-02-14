#pragma once

#include "vgeu_window.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtx/euler_angles.hpp>

namespace vgeu {
struct TransformComponent {
  glm::vec3 translation{};
  glm::vec3 scale{1.f, 1.f, 1.f};
  // YXZ Euler-Angle
  glm::vec3 rotation{};

  // translation mat * Ry * Rx * Rz * scale mat
  // tait-bryan angles with YXZ
  // https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
  glm::mat4 mat4() const {
    glm::mat4 m{1.f};
    m = glm::translate(m, translation);
    m = m * glm::eulerAngleYXZ(rotation.y, rotation.x, rotation.z);
    m = glm::scale(m, scale);
    return m;
  }
  glm::mat3 normalMatrix();
};
class KeyBoardMovementController {
 public:
  struct KeyMappings {
    int moveLeft = GLFW_KEY_A;
    int moveRight = GLFW_KEY_D;
    int moveForward = GLFW_KEY_W;
    int moveBackward = GLFW_KEY_S;
    int moveUp = GLFW_KEY_E;
    int moveDown = GLFW_KEY_Q;
    int lookLeft = GLFW_KEY_LEFT;
    int lookRight = GLFW_KEY_RIGHT;
    int lookUp = GLFW_KEY_UP;
    int lookDown = GLFW_KEY_DOWN;
  };
  bool moveInPlaneXZ(GLFWwindow* window, float dt,
                     TransformComponent& transform);
  KeyMappings keys{};
  float moveSpeed{3.f};
  float lookSpeed{1.5f};
};
}  // namespace vgeu