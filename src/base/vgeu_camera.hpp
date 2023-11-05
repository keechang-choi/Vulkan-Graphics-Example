#pragma once

#include "vgeu_window.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

namespace vgeu {
class VgeuCamera {
 public:
  void setOrthographicProjection(float left, float right, float top,
                                 float botton, float near, float far);
  void setPerspectiveProjection(float fovy, float aspect, float near,
                                float far);
  void setAspectRatio(float aspect) {
    setPerspectiveProjection(fovy_, aspect, near_, far_);
  }

  void setViewDirection(glm::vec3 position, glm::vec3 direction,
                        glm::vec3 up = glm::vec3{0.f, -1.f, -0.f});
  void setViewTarget(glm::vec3 position, glm::vec3 target,
                     glm::vec3 up = glm::vec3{0.f, -1.f, 0.f});
  void setViewYXZ(glm::vec3 position, glm::vec3 rotation);
  void setViewMatrix(glm::mat4 viewMatrix) {
    this->viewMatrix = viewMatrix;
    inverseViewMatrix = glm::inverse(viewMatrix);
  }
  const glm::mat4& getProjection() const { return projectionMatrix; }
  const glm::mat4& getView() const { return viewMatrix; }
  const glm::mat4& getInverseView() const { return inverseViewMatrix; }
  const glm::vec3 getPosition() const {
    return glm::vec3(inverseViewMatrix[3]);
  }
  // Y: yaw, X: pitch, Z: roll
  const glm::vec3 getRotationYXZ() const {
    return glm::vec3{
        glm::asin(-inverseViewMatrix[2][1]),                           // X
        glm::atan(inverseViewMatrix[2][0] / inverseViewMatrix[2][2]),  // Y
        glm::atan(inverseViewMatrix[0][1] / inverseViewMatrix[1][1]),  // Z
    };
  }

 private:
  glm::mat4 projectionMatrix{1.f};
  glm::mat4 viewMatrix{1.f};
  glm::mat4 inverseViewMatrix{1.f};
  float near_ = 0.1f;
  float far_ = 100.0f;
  float fovy_ = glm::radians(50.0);
};

}  // namespace vgeu