#include "deferred.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtx/matrix_query.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

// std
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <unordered_set>

namespace vge {
VgeExample::VgeExample() : VgeBase() { title = "Cloth Example"; }
VgeExample::~VgeExample() {}
void VgeExample::setupCommandLineParser(CLI::App& app) {}
void VgeExample::setOptions(const std::optional<Options>& opts) {
  if (opts.has_value()) {
    this->opts = opts.value();
    // overwrite cli args for restart run
    cameraController.moveSpeed = this->opts.moveSpeed;
  } else {
    // save cli args for initial run
  }
}

void VgeExample::initVulkan() {
  cameraController.moveSpeed = opts.moveSpeed;
  // camera setup
  if (glm::isIdentity(opts.cameraView, 1e-6f)) {
    camera.setViewTarget(glm::vec3{0.f, -10.f, -20.f},
                         glm::vec3{0.f, 0.f, 0.f});
  } else {
    camera.setViewMatrix(opts.cameraView);
  }
  camera.setPerspectiveProjection(
      glm::radians(60.f),
      static_cast<float>(width) / (static_cast<float>(height)), 0.1f, 256.f);
  // NOTE: coordinate space in world

  VgeBase::initVulkan();
}

void VgeExample::getEnabledExtensions() {}
void VgeExample::getEnabledFeatures() {
  enabledFeatures.samplerAnisotropy =
      physicalDevice.getFeatures().samplerAnisotropy;
  enabledFeatures.fillModeNonSolid =
      physicalDevice.getFeatures().fillModeNonSolid;
}
void VgeExample::prepare() {
  VgeBase::prepare();
  loadAssets();
  prepareOffScreenFrameBuffer();
  prepareUniformBuffers();
  setupDescriptors();
  preparePipelines();
  prepared = true;
}

void VgeExample::render() {}
void VgeExample::viewChanged() {}
void VgeExample::onUpdateUIOverlay() {}

}  // namespace vge