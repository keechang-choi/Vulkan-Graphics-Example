#pragma once

/*

refence: sample util code in vulkan-hpp
https://github.com/KhronosGroup/Vulkan-Hpp


*/

// forward declaration?
// class vk::ApplicationInfo;
// class vk::raii::Instance;
// class vk::InstanceCreateInfo;
// class vk::DebugUtilsMessengerCreateInfoEXT;
// class vk::StructureChain<>;

// libs
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>

// std
#include <memory>
#include <string>
#include <vector>

namespace vgeu {
// initVulkan - device
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

vk::raii::Instance createInstance(vk::raii::Context const& context,
                                  std::string const& appName,
                                  std::string const& engineName,
                                  uint32_t apiVersion = VK_API_VERSION_1_0);
vk::DebugUtilsMessengerCreateInfoEXT createDebugCreateInfo();
bool checkValidationLayerSupport(vk::raii::Context const& context);
std::vector<const char*> getRequiredExtensions();
std::unique_ptr<vk::raii::DebugUtilsMessengerEXT> setupDebugMessenger(
    vk::raii::Instance& instance);

}  // namespace vgeu