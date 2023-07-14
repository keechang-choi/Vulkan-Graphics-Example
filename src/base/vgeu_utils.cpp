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

    vk::StructureChain<vk::InstanceCreateInfo,
                       vk::DebugUtilsMessengerCreateInfoEXT>
        structureChain(instanceCreateInfo, createDebugCreateInfo());
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
  std::cout << glfwExtensionCount;
  std::vector<const char*> extensions(glfwExtensions,
                                      glfwExtensions + glfwExtensionCount);

  if (enableValidationLayers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

vk::DebugUtilsMessengerCreateInfoEXT createDebugCreateInfo() {
  vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
  vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
      vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

  vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo(
      {}, severityFlags, messageTypeFlags, debugCallback);
  return debugCreateInfo;
}

vk::raii::DebugUtilsMessengerEXT setupDebugMessenger(
    vk::raii::Instance& instance) {
  return vk::raii::DebugUtilsMessengerEXT(instance,
                                          vgeu::createDebugCreateInfo());
}

vk::raii::Device createLogicalDevice(
    const vk::raii::PhysicalDevice& physicalDevice,
    const QueueFamilyIndices& queueFamilyIndices,
    const std::vector<const char*>& extensions,
    const vk::PhysicalDeviceFeatures* physicalDeviceFeatures, const void* pNext,
    bool useSwapChain, vk::QueueFlags requestedQueueTypes) {
  float queuePriority = 0.0f;
  std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos{};
  if (requestedQueueTypes & vk::QueueFlagBits::eGraphics) {
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(
        vk::DeviceQueueCreateFlags(), queueFamilyIndices.graphics, 1,
        &queuePriority);
    queueCreateInfos.push_back(deviceQueueCreateInfo);
  }

  if (requestedQueueTypes & vk::QueueFlagBits::eCompute) {
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(
        vk::DeviceQueueCreateFlags(), queueFamilyIndices.compute, 1,
        &queuePriority);
    queueCreateInfos.push_back(deviceQueueCreateInfo);
  }

  if (requestedQueueTypes & vk::QueueFlagBits::eTransfer) {
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(
        vk::DeviceQueueCreateFlags(), queueFamilyIndices.transfer, 1,
        &queuePriority);
    queueCreateInfos.push_back(deviceQueueCreateInfo);
  }

  vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(),
                                        queueCreateInfos, {}, extensions,
                                        physicalDeviceFeatures, pNext);
  return vk::raii::Device(physicalDevice, deviceCreateInfo);
}
QueueFamilyIndices findQueueFamilyIndices(
    const std::vector<vk::QueueFamilyProperties>& queueFamilyProperties,
    vk::QueueFlags requestedQueueTypes) {
  QueueFamilyIndices queueFamilyIndices;
  if (requestedQueueTypes & vk::QueueFlagBits::eGraphics) {
    queueFamilyIndices.graphics = getQueueFamilyIndex(
        queueFamilyProperties, vk::QueueFlagBits::eGraphics);
  } else {
    queueFamilyIndices.graphics = 0;
  }

  if (requestedQueueTypes & vk::QueueFlagBits::eCompute) {
    queueFamilyIndices.compute =
        getQueueFamilyIndex(queueFamilyProperties, vk::QueueFlagBits::eCompute);
  } else {
    queueFamilyIndices.compute = queueFamilyIndices.graphics;
  }

  if (requestedQueueTypes & vk::QueueFlagBits::eTransfer) {
    queueFamilyIndices.transfer = getQueueFamilyIndex(
        queueFamilyProperties, vk::QueueFlagBits::eTransfer);
  } else {
    queueFamilyIndices.transfer = queueFamilyIndices.graphics;
  }
  return queueFamilyIndices;
}

uint32_t getQueueFamilyIndex(
    const std::vector<vk::QueueFamilyProperties>& queueFamilyProperties,
    vk::QueueFlagBits queueFlag) {
  // Dedicated queue for compute
  // Try to find a queue family index that supports compute but not graphics
  if (queueFlag == vk::QueueFlagBits::eCompute) {
    for (uint32_t i = 0;
         i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
      if ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eCompute) &&
          (!(queueFamilyProperties[i].queueFlags &
             vk::QueueFlagBits::eGraphics))) {
        return i;
      }
    }
  }

  // Dedicated queue for transfer
  // Try to find a queue family index that supports transfer but not graphics
  // and compute
  if (queueFlag == vk::QueueFlagBits::eTransfer) {
    for (uint32_t i = 0;
         i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {
      if ((queueFamilyProperties[i].queueFlags &
           vk::QueueFlagBits::eTransfer) &&
          (!(queueFamilyProperties[i].queueFlags &
             vk::QueueFlagBits::eGraphics)) &&
          (!(queueFamilyProperties[i].queueFlags &
             vk::QueueFlagBits::eCompute))) {
        return i;
      }
    }
  }

  // For other queue types or if no separate compute queue is present, return
  // the first one to support the requested flags
  for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size());
       i++) {
    if (queueFamilyProperties[i].queueFlags & queueFlag) {
      return i;
    }
  }

  throw std::runtime_error("Could not find a matching queue family index");
}

}  // namespace vgeu