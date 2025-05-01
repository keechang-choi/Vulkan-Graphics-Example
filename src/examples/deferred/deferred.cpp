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

void VgeExample::loadAssets() {
  // NOTE: no flip or preTransform for animation and skinning
  vgeu::FileLoadingFlags glTFLoadingFlags =
      vgeu::FileLoadingFlagBits::kPreMultiplyVertexColors;
  // | vgeu::FileLoadingFlagBits::kPreTransformVertices;
  //| vgeu::FileLoadingFlagBits::kFlipY;
  std::shared_ptr<vgeu::glTF::Model> damagedHelmet;

  damagedHelmet = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  damagedHelmet->loadFromFile(
      getAssetsPath() + "/models/DamagedHelmet/glTF/DamagedHelmet.gltf",
      glTFLoadingFlags);

  {
    ModelInstance modelInstance{};
    modelInstance.model = damagedHelmet;
    modelInstance.name = "damagedHelmet";
    addModelInstance(std::move(modelInstance));
  }
}

void VgeExample::prepareOffScreenFrameBuffer() {}
void VgeExample::prepareUniformBuffers() {}
void VgeExample::setupDescriptors() {}
void VgeExample::preparePipelines() {}

void VgeExample::buildCommandBuffers() {}
void VgeExample::buildDefferredCommandBuffers() {}

void VgeExample::addModelInstance(ModelInstance&& newInstance) {
  size_t instanceIdx = modelInstances.size();
  modelInstances.push_back(std::move(newInstance));
  instanceMap[newInstance.name].push_back(instanceIdx);
}

const std::vector<size_t>& VgeExample::findInstances(const std::string& name) {
  assert(instanceMap.find(name) != instanceMap.end() &&
         "failed to find instance by name.");
  return instanceMap.at(name);
}

ModelInstance::ModelInstance(ModelInstance&& other) {
  model = other.model;
  simpleModel = other.simpleModel;
  name = other.name;
  isBone = other.isBone;
  animationIndex = other.animationIndex;
  animationTime = other.animationTime;
  transform = other.transform;
}

ModelInstance& ModelInstance::operator=(ModelInstance&& other) {
  model = other.model;
  simpleModel = other.simpleModel;
  name = other.name;
  isBone = other.isBone;
  animationIndex = other.animationIndex;
  animationTime = other.animationTime;
  transform = other.transform;
  return *this;
}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()