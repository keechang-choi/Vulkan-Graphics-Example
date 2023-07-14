#include "vgeu_utils.hpp"

// libs
#include <GLFW/glfw3.h>

// std
#include <iostream>

namespace vgeu {
// local callback functions
static VKAPI_ATTR VkBool32 VKAPI_CALL
debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
              VkDebugUtilsMessageTypeFlagsEXT messageType,
              const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
              void* pUserData) {
  std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

  return VK_FALSE;
}
vk::raii::Instance createInstance(vk::raii::Context const& context,
                                  std::string const& appName,
                                  std::string const& engineName,
                                  uint32_t apiVersion) {
  vk::ApplicationInfo applicationInfo(appName.c_str(), 1, engineName.c_str(), 1,
                                      apiVersion);
  if (enableValidationLayers && !checkValidationLayerSupport(context)) {
    throw std::runtime_error("validation layers requested, but not available!");
  }

  std::vector<const char*> extensions = getRequiredExtensions();
  for (int i = 0; i < extensions.size(); i++) {
    std::cout << extensions[i] << std::endl;
  }

  vk::InstanceCreateInfo createInfo;
  if (enableValidationLayers) {
    vk::InstanceCreateInfo instanceCreateInfo({}, &applicationInfo,
                                              validationLayers, extensions);

    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

    vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo(
        {}, severityFlags, messageTypeFlags, debugCallback);
    vk::StructureChain<vk::InstanceCreateInfo,
                       vk::DebugUtilsMessengerCreateInfoEXT>
        structureChain(instanceCreateInfo, debugCreateInfo);
    createInfo = structureChain.get<vk::InstanceCreateInfo>();
  } else {
    vk::InstanceCreateInfo instanceCreateInfo({}, &applicationInfo, {},
                                              extensions);
    createInfo = instanceCreateInfo;
  }

  return vk::raii::Instance(context, createInfo);
}

bool checkValidationLayerSupport(vk::raii::Context const& context) {
  auto availableLayers = context.enumerateInstanceLayerProperties();

  for (const char* layerName : validationLayers) {
    bool layerFound = false;

    for (const auto& layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound) {
      return false;
    }
  }

  return true;
}

std::vector<const char*> getRequiredExtensions() {
  uint32_t glfwExtensionCount = 0;
  const char** glfwExtensions;
  glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

  std::vector<const char*> extensions(glfwExtensions,
                                      glfwExtensions + glfwExtensionCount);

  if (enableValidationLayers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

}  // namespace vgeu