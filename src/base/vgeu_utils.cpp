#include "vgeu_utils.hpp"

// libs
#include <GLFW/glfw3.h>

// std
#include <iostream>
#include <limits>

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
vk::raii::Instance createInstance(const vk::raii::Context& context,
                                  const std::string& appName,
                                  const std::string& engineName,
                                  bool enableValidationLayers,
                                  uint32_t apiVersion) {
  vk::ApplicationInfo applicationInfo(appName.c_str(), 1, engineName.c_str(), 1,
                                      apiVersion);
  if (enableValidationLayers && !checkValidationLayerSupport(context)) {
    throw std::runtime_error("validation layers requested, but not available!");
  }

  std::vector<const char*> extensions =
      getRequiredExtensions(enableValidationLayers);
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

bool checkValidationLayerSupport(const vk::raii::Context& context) {
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

std::vector<const char*> getRequiredExtensions(bool enableValidationLayers) {
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
  return vk::raii::DebugUtilsMessengerEXT(instance, createDebugCreateInfo());
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

  std::vector<const char*> deviceExtensions(extensions);
  if (useSwapChain) {
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  vk::DeviceCreateInfo deviceCreateInfo(vk::DeviceCreateFlags(),
                                        queueCreateInfos, {}, deviceExtensions,
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

vk::Format pickDepthFormat(const vk::raii::PhysicalDevice& physicalDevice,
                           bool requiresStencil) {
  std::vector<vk::Format> candidates;
  if (requiresStencil) {
    std::vector<vk::Format> candidatesStencil{
        vk::Format::eD32SfloatS8Uint,
        vk::Format::eD24UnormS8Uint,
        vk::Format::eD16UnormS8Uint,
    };
    candidates = candidatesStencil;
  } else {
    std::vector<vk::Format> candidatesDepth{
        vk::Format::eD32SfloatS8Uint, vk::Format::eD32Sfloat,
        vk::Format::eD24UnormS8Uint,  vk::Format::eD16UnormS8Uint,
        vk::Format::eD16Unorm,
    };
    candidates = candidatesDepth;
  }

  for (vk::Format format : candidates) {
    vk::FormatProperties props = physicalDevice.getFormatProperties(format);

    if (props.optimalTilingFeatures &
        vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
      return format;
    }
  }
  throw std::runtime_error("failed to find supported format!");
}

SwapChainData::SwapChainData(const vk::raii::PhysicalDevice& physicalDevice,
                             const vk::raii::Device& device,
                             const vk::raii::SurfaceKHR& surface,
                             const vk::Extent2D& extent,
                             vk::ImageUsageFlags usage,
                             const vk::raii::SwapchainKHR* pOldSwapchain,
                             uint32_t graphicsQueueFamilyIndex,
                             uint32_t presentQueueFamilyIndex) {
  vk::SurfaceFormatKHR surfaceFormat =
      pickSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(*surface));
  colorFormat = surfaceFormat.format;

  vk::SurfaceCapabilitiesKHR surfaceCapabilities =
      physicalDevice.getSurfaceCapabilitiesKHR(*surface);
  if (surfaceCapabilities.currentExtent.width ==
      std::numeric_limits<uint32_t>::max()) {
    // If the surface size is undefined, the size is set to the size of the
    // images requested.
    swapChainExtent.width =
        clamp(extent.width, surfaceCapabilities.minImageExtent.width,
              surfaceCapabilities.maxImageExtent.width);
    swapChainExtent.height =
        clamp(extent.height, surfaceCapabilities.minImageExtent.height,
              surfaceCapabilities.maxImageExtent.height);
  } else {
    // If the surface size is defined, the swap chain size must match
    swapChainExtent = surfaceCapabilities.currentExtent;
  }

  vk::SurfaceTransformFlagBitsKHR preTransform =
      (surfaceCapabilities.supportedTransforms &
       vk::SurfaceTransformFlagBitsKHR::eIdentity)
          ? vk::SurfaceTransformFlagBitsKHR::eIdentity
          : surfaceCapabilities.currentTransform;
  vk::CompositeAlphaFlagBitsKHR compositeAlpha =
      (surfaceCapabilities.supportedCompositeAlpha &
       vk::CompositeAlphaFlagBitsKHR::ePreMultiplied)
          ? vk::CompositeAlphaFlagBitsKHR::ePreMultiplied
      : (surfaceCapabilities.supportedCompositeAlpha &
         vk::CompositeAlphaFlagBitsKHR::ePostMultiplied)
          ? vk::CompositeAlphaFlagBitsKHR::ePostMultiplied
      : (surfaceCapabilities.supportedCompositeAlpha &
         vk::CompositeAlphaFlagBitsKHR::eInherit)
          ? vk::CompositeAlphaFlagBitsKHR::eInherit
          : vk::CompositeAlphaFlagBitsKHR::eOpaque;
  vk::PresentModeKHR presentMode =
      pickPresentMode(physicalDevice.getSurfacePresentModesKHR(*surface));
  // NOTE: min image count
  uint32_t minImageCount = std::min(surfaceCapabilities.minImageCount + 1,
                                    surfaceCapabilities.maxImageCount);
  vk::SwapchainCreateInfoKHR swapChainCreateInfo(
      {}, *surface, minImageCount, colorFormat, surfaceFormat.colorSpace,
      swapChainExtent, 1, usage, vk::SharingMode::eExclusive, {}, preTransform,
      compositeAlpha, presentMode, true,
      pOldSwapchain ? **pOldSwapchain : nullptr);

  if (graphicsQueueFamilyIndex != presentQueueFamilyIndex) {
    uint32_t queueFamilyIndices[2] = {graphicsQueueFamilyIndex,
                                      presentQueueFamilyIndex};
    // If the graphics and present queues are from different queue families,
    // we either have to explicitly transfer ownership of images between the
    // queues, or we have to create the swapchain with imageSharingMode as
    // vk::SharingMode::eConcurrent
    swapChainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
    swapChainCreateInfo.queueFamilyIndexCount = 2;
    swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
  }

  swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);

  // NOTE: In implementation, reduce image count if the request not succeed.
  images = swapChain.getImages();

  imageViews.reserve(images.size());
  vk::ImageViewCreateInfo imageViewCreateInfo(
      {}, {}, vk::ImageViewType::e2D, colorFormat, {},
      {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
  for (auto image : images) {
    imageViewCreateInfo.image = image;
    imageViews.emplace_back(device, imageViewCreateInfo);
  }
}

vk::SurfaceFormatKHR pickSurfaceFormat(
    std::vector<vk::SurfaceFormatKHR> const& formats) {
  assert(!formats.empty());
  vk::SurfaceFormatKHR pickedFormat = formats[0];
  if (formats.size() == 1) {
    if (formats[0].format == vk::Format::eUndefined) {
      pickedFormat.format = vk::Format::eB8G8R8A8Unorm;
      pickedFormat.colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
    }
  } else {
    // request several formats, the first found will be used
    vk::Format requestedFormats[] = {
        vk::Format::eB8G8R8A8Unorm, vk::Format::eR8G8B8A8Unorm,
        vk::Format::eB8G8R8Unorm, vk::Format::eR8G8B8Unorm};
    vk::ColorSpaceKHR requestedColorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
    for (size_t i = 0;
         i < sizeof(requestedFormats) / sizeof(requestedFormats[0]); i++) {
      vk::Format requestedFormat = requestedFormats[i];
      auto it = std::find_if(formats.begin(), formats.end(),
                             [requestedFormat, requestedColorSpace](
                                 vk::SurfaceFormatKHR const& f) {
                               return (f.format == requestedFormat) &&
                                      (f.colorSpace == requestedColorSpace);
                             });
      if (it != formats.end()) {
        pickedFormat = *it;
        break;
      }
    }
  }
  assert(pickedFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear);
  return pickedFormat;
}

vk::PresentModeKHR pickPresentMode(
    std::vector<vk::PresentModeKHR> const& presentModes) {
  vk::PresentModeKHR pickedMode = vk::PresentModeKHR::eFifo;
  for (const auto& presentMode : presentModes) {
    if (presentMode == vk::PresentModeKHR::eMailbox) {
      pickedMode = presentMode;
      break;
    }

    if (presentMode == vk::PresentModeKHR::eImmediate) {
      pickedMode = presentMode;
    }
  }
  return pickedMode;
}

ImageData::ImageData(vk::raii::PhysicalDevice const& physicalDevice,
                     vk::raii::Device const& device, vk::Format format_,
                     vk::Extent2D const& extent, vk::ImageTiling tiling,
                     vk::ImageUsageFlags usage, vk::ImageLayout initialLayout,
                     vk::MemoryPropertyFlags memoryProperties,
                     vk::ImageAspectFlags aspectMask)
    : format(format_),
      image(device, {vk::ImageCreateFlags(),
                     vk::ImageType::e2D,
                     format,
                     vk::Extent3D(extent, 1),
                     1,
                     1,
                     vk::SampleCountFlagBits::e1,
                     tiling,
                     usage | vk::ImageUsageFlagBits::eSampled,
                     vk::SharingMode::eExclusive,
                     {},
                     initialLayout}),
      deviceMemory(vgeu::allocateDeviceMemory(
          device, physicalDevice.getMemoryProperties(),
          image.getMemoryRequirements(), memoryProperties)) {
  image.bindMemory(*deviceMemory, 0);
  imageView = vk::raii::ImageView(
      device, vk::ImageViewCreateInfo({}, *image, vk::ImageViewType::e2D,
                                      format, {}, {aspectMask, 0, 1, 0, 1}));
}

vk::raii::DeviceMemory allocateDeviceMemory(
    const vk::raii::Device& device,
    const vk::PhysicalDeviceMemoryProperties& memoryProperties,
    const vk::MemoryRequirements& memoryRequirements,
    vk::MemoryPropertyFlags memoryPropertyFlags) {
  uint32_t memoryTypeIndex = vgeu::findMemoryType(
      memoryProperties, memoryRequirements.memoryTypeBits, memoryPropertyFlags);
  vk::MemoryAllocateInfo memoryAllocateInfo(memoryRequirements.size,
                                            memoryTypeIndex);
  return vk::raii::DeviceMemory(device, memoryAllocateInfo);
}

uint32_t findMemoryType(
    const vk::PhysicalDeviceMemoryProperties& memoryProperties,
    uint32_t typeBits, vk::MemoryPropertyFlags requirementsMask) {
  uint32_t typeIndex = uint32_t(~0);
  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
    if ((typeBits & 1) && ((memoryProperties.memoryTypes[i].propertyFlags &
                            requirementsMask) == requirementsMask)) {
      typeIndex = i;
      break;
    }
    typeBits >>= 1;
  }
  assert(typeIndex != uint32_t(~0));
  return typeIndex;
}

vk::raii::RenderPass createRenderPass(const vk::raii::Device& device,
                                      vk::Format colorFormat,
                                      vk::Format depthFormat,
                                      vk::AttachmentLoadOp loadOp,
                                      vk::ImageLayout colorFinalLayout) {
  std::vector<vk::AttachmentDescription> attachmentDescriptions;
  assert(colorFormat != vk::Format::eUndefined);
  attachmentDescriptions.emplace_back(
      vk::AttachmentDescriptionFlags(), colorFormat,
      vk::SampleCountFlagBits::e1, loadOp, vk::AttachmentStoreOp::eStore,
      vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
      vk::ImageLayout::eUndefined, colorFinalLayout);
  if (depthFormat != vk::Format::eUndefined) {
    attachmentDescriptions.emplace_back(
        vk::AttachmentDescriptionFlags(), depthFormat,
        vk::SampleCountFlagBits::e1, loadOp, vk::AttachmentStoreOp::eDontCare,
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eUndefined,
        vk::ImageLayout::eDepthStencilAttachmentOptimal);
  }
  vk::AttachmentReference colorAttachment(
      0, vk::ImageLayout::eColorAttachmentOptimal);
  vk::AttachmentReference depthAttachment(
      1, vk::ImageLayout::eDepthStencilAttachmentOptimal);
  vk::SubpassDescription subpassDescription(
      vk::SubpassDescriptionFlags(), vk::PipelineBindPoint::eGraphics, {},
      colorAttachment, {},
      (depthFormat != vk::Format::eUndefined) ? &depthAttachment : nullptr);

  // subpass dependencies for layout transition
  std::vector<vk::SubpassDependency> dependencies;
  // for depth attachment
  dependencies.emplace_back(VK_SUBPASS_EXTERNAL, 0u,
                            vk::PipelineStageFlagBits::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits::eLateFragmentTests,
                            vk::PipelineStageFlagBits::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits::eLateFragmentTests,
                            vk::AccessFlagBits::eDepthStencilAttachmentWrite,
                            vk::AccessFlagBits::eDepthStencilAttachmentWrite |
                                vk::AccessFlagBits::eDepthStencilAttachmentRead,
                            vk::DependencyFlags()  // 0 for framebuffer-global
  );

  // for color attachment
  dependencies.emplace_back(VK_SUBPASS_EXTERNAL, 0u,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits::eColorAttachmentOutput,
                            vk::AccessFlags(),
                            vk::AccessFlagBits::eColorAttachmentWrite |
                                vk::AccessFlagBits::eColorAttachmentRead,
                            vk::DependencyFlags()  // 0 for framebuffer-global
  );

  vk::RenderPassCreateInfo renderPassCreateInfo(
      vk::RenderPassCreateFlags(), attachmentDescriptions, subpassDescription,
      dependencies);
  return vk::raii::RenderPass(device, renderPassCreateInfo);
}

std::vector<vk::raii::Framebuffer> createFramebuffers(
    const vk::raii::Device& device, const vk::raii::RenderPass& renderPass,
    const std::vector<vk::raii::ImageView>& imageViews,
    const vk::raii::ImageView* pDepthImageView, const vk::Extent2D& extent) {
  vk::ImageView attachments[2];
  attachments[1] = pDepthImageView ? **pDepthImageView : vk::ImageView();

  vk::FramebufferCreateInfo framebufferCreateInfo(
      vk::FramebufferCreateFlags(), *renderPass, pDepthImageView ? 2 : 1,
      attachments, extent.width, extent.height, 1);
  std::vector<vk::raii::Framebuffer> framebuffers;
  framebuffers.reserve(imageViews.size());
  for (auto const& imageView : imageViews) {
    attachments[0] = *imageView;
    framebuffers.push_back(
        vk::raii::Framebuffer(device, framebufferCreateInfo));
  }

  return framebuffers;
}

std::vector<char> readFile(const std::string& filepath) {
  std::ifstream file(filepath.c_str(), std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("failed to open file: " + filepath);
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  std::vector<char> buffer(fileSize);

  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();

  return buffer;
}

vk::raii::ShaderModule createShaderModule(const vk::raii::Device& device,
                                          std::vector<char>& code) {
  vk::ShaderModuleCreateInfo shaderModuleCI(
      {}, code.size(), reinterpret_cast<const uint32_t*>(code.data()));
  return vk::raii::ShaderModule(device, shaderModuleCI);
}

void setImageLayout(const vk::raii::CommandBuffer& commandBuffer,
                    vk::Image image,
                    vk::ImageSubresourceRange imageSubresourceRange,
                    vk::ImageLayout oldImageLayout,
                    vk::ImageLayout newImageLayout) {
  vk::AccessFlags srcAccessMask{};  // NOTE: 0
  switch (oldImageLayout) {
    case vk::ImageLayout::eTransferSrcOptimal:
      srcAccessMask = vk::AccessFlagBits::eTransferRead;
      break;
    case vk::ImageLayout::eTransferDstOptimal:
      srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      break;
    case vk::ImageLayout::ePreinitialized:
      srcAccessMask = vk::AccessFlagBits::eHostWrite;
      break;
    case vk::ImageLayout::eGeneral:  // srcAccessMask is empty
    case vk::ImageLayout::eUndefined:
      break;
    default:
      assert(false && "failed: to support oldImageLayout for srcAcessMask");
      break;
  }

  vk::PipelineStageFlags srcStage{};  // NOTE: 0
  switch (oldImageLayout) {
    case vk::ImageLayout::eGeneral:
    case vk::ImageLayout::ePreinitialized:
      srcStage = vk::PipelineStageFlagBits::eHost;
      break;
    case vk::ImageLayout::eTransferSrcOptimal:
    case vk::ImageLayout::eTransferDstOptimal:
      srcStage = vk::PipelineStageFlagBits::eTransfer;
      break;
    case vk::ImageLayout::eUndefined:
      srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
      break;
    default:
      assert(false && "failed: to support oldImageLayout for srcStage");
      break;
  }

  vk::AccessFlags dstAccessMask{};
  switch (newImageLayout) {
    case vk::ImageLayout::eColorAttachmentOptimal:
      dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
      break;
    case vk::ImageLayout::eDepthStencilAttachmentOptimal:
      dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentRead |
                      vk::AccessFlagBits::eDepthStencilAttachmentWrite;
      break;
    case vk::ImageLayout::eGeneral:  // empty dstAccessMask
    case vk::ImageLayout::ePresentSrcKHR:
      break;
    case vk::ImageLayout::eShaderReadOnlyOptimal:
      dstAccessMask = vk::AccessFlagBits::eShaderRead;
      break;
    case vk::ImageLayout::eTransferSrcOptimal:
      dstAccessMask = vk::AccessFlagBits::eTransferRead;
      break;
    case vk::ImageLayout::eTransferDstOptimal:
      dstAccessMask = vk::AccessFlagBits::eTransferWrite;
      break;
    default:
      assert(false && "failed: to support newImageLayout for dstAccessMask");
      break;
  }

  vk::PipelineStageFlags dstStage{};
  switch (newImageLayout) {
    case vk::ImageLayout::eColorAttachmentOptimal:
      dstStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
      break;
    case vk::ImageLayout::eDepthStencilAttachmentOptimal:
      dstStage = vk::PipelineStageFlagBits::eEarlyFragmentTests;
      break;
    case vk::ImageLayout::eGeneral:
      dstStage = vk::PipelineStageFlagBits::eHost;
      break;
    case vk::ImageLayout::ePresentSrcKHR:
      dstStage = vk::PipelineStageFlagBits::eBottomOfPipe;
      break;
    case vk::ImageLayout::eShaderReadOnlyOptimal:
      dstStage = vk::PipelineStageFlagBits::eFragmentShader;
      break;
    case vk::ImageLayout::eTransferDstOptimal:
    case vk::ImageLayout::eTransferSrcOptimal:
      dstStage = vk::PipelineStageFlagBits::eTransfer;
      break;
    default:
      assert(false && "failed: to support newImageLayout for dstStage");
      break;
  }

  vk::ImageMemoryBarrier imageMemoryBarrier(
      srcAccessMask, dstAccessMask, oldImageLayout, newImageLayout,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image,
      imageSubresourceRange);

  commandBuffer.pipelineBarrier(srcStage, dstStage, {}, nullptr, nullptr,
                                imageMemoryBarrier);
}

void setImageLayout(const vk::raii::CommandBuffer& commandBuffer,
                    vk::Image image, vk::Format format, uint32_t baseMipLevel,
                    uint32_t levelCount, vk::ImageLayout oldImageLayout,
                    vk::ImageLayout newImageLayout) {
  vk::ImageAspectFlags aspectMask;
  if (newImageLayout == vk::ImageLayout::eDepthStencilAttachmentOptimal) {
    aspectMask = vk::ImageAspectFlagBits::eDepth;
    if (format == vk::Format::eD32SfloatS8Uint ||
        format == vk::Format::eD24UnormS8Uint) {
      aspectMask |= vk::ImageAspectFlagBits::eStencil;
    }
  } else {
    aspectMask = vk::ImageAspectFlagBits::eColor;
  }

  vk::ImageSubresourceRange imageSubresourceRange(aspectMask, baseMipLevel,
                                                  levelCount, 0, 1);
  setImageLayout(commandBuffer, image, imageSubresourceRange, oldImageLayout,
                 newImageLayout);
}

}  // namespace vgeu