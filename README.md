# Vulkan-Graphics-Example
To study and implement interesting examples in Computer Graphics using Vulkan API

## Reference
1. Existing Vulkan examples repo  
    https://github.com/SaschaWillems/Vulkan
2. https://github.com/blurrypiano/littleVulkanEngine
3. https://vulkan-tutorial.com/
4. Physically-based simulations (The Ten Minute Physics)  
    https://matthias-research.github.io/pages/tenMinutePhysics/index.html

# Enviroment & Setup

supposed to run example apps on multi-platforms, but mainly tested on Windows OS first.

Prerequisites
- CMake tools
- MinGW-w64
- ninja-build (optional)
- Vulkan SDK
- third-party [externals](external)
  - Vulkan-hpp  for  vulkan c++-API wrapper
  - VMA for memory allocation
- assets [liscene](assets)

```
git submodule init
git submodule update

./mingwBuild.bat
./build/<example>.exe
```

# [Examples](src/examples)

## [triangle](src/examples/triangle)
![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/184f2e34-0c22-4939-ae92-c2fc3c03a88e)

## [pipelines](src/examples/pipelines)
![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/cd856a55-5888-4852-bcea-a8c16b5c772e)

## [animation](src/examples/animation)
- animation, skinning, dynamic uniform buffers  
![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/3dbcdfbf-a977-4924-969f-3087a8875882)


