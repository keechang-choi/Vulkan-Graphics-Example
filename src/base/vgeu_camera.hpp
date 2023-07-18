#pragma once

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

namespace vgeu {

struct TransformComponent {
  glm::vec3 translation{};
  glm::vec3 scale{1.f, 1.f, 1.f};
  glm::vec3 rotation{};

  // translation mat * Ry * Rx * Rz * scale mat
  // tait-bryan angles with YXZ
  // https://en.wikipedia.org/wiki/Euler_angles#Rotation_matrix
  glm::mat4 mat4();
  glm::mat3 normalMatrix();
};

// TODO: camera view, projection getter and setter

}  // namespace vgeu