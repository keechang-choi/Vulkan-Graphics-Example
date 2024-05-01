# Vulkan-Graphics-Example
To study and implement interesting examples in Computer Graphics using Vulkan API

## References
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

## [particle](src/examples/particle)
- particle compute shader,
- references:
  - n body system: https://github.com/SaschaWillems/Vulkan/tree/master/examples/computenbody
  - model mesh attraction: https://github.com/byumjin/WebGL2-GPU-Particle
  - skinning in compute shader: https://github.com/KaminariOS/rustracer/tree/main
  - numerical integration: https://adamsturge.github.io/Engine-Blog/index.html

|     |     | 
| --- | --- | 
| ![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/c32eebec-0b68-4a3e-9f7c-75a768202c9f)  | ![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/1664e1c8-9f7a-486f-b01a-735522c0ed20) |
| ![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/6af233a0-eb81-4b06-83e1-d515195412ca) |![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/ff3554b7-35d3-4a3d-9752-133b70aa3d79) |
| ![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/6f3b388f-1311-4aac-b34e-e766fbd3fc7c) | ![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/9a2c6c61-2e4d-4740-baa4-b0373ff5f38d) |
| ![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/59a21bf7-9177-4c69-8053-e0ce0c56cd99) | ![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/95bd0fd0-a8e1-4cb4-8942-13331bf5f7f7) |



## [PBD](src/examples/pbd)
- Position Based Dynamics on CPU
- 2d simulation examples from [The Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html)


|     |     |
| --- | --- |
| ![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/129e7a94-2e60-4199-a857-23accaf26a1b) | ![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/f6f369eb-5575-48f5-a821-6cd5e9815e3a) |
| ![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/df7958b3-45de-4bb1-8f2d-b8b43c3732f5) | ![image](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/e9966c9b-6272-4ef5-b1c0-d6c1a19ef509) |



## [cloth](src/examples/cloth)
- PBD cloth simulation on GPU (vulkan compute shader)
- geometry shader for debugging normals
- references:
  - [Vulkan/examples/computecloth/computecloth.cpp at master · SaschaWillems/Vulkan (github.com)](https://github.com/SaschaWillems/Vulkan/blob/master/examples/computecloth/computecloth.cpp)
  - [Vulkan/shaders/glsl/computecloth/cloth.comp at master · SaschaWillems/Vulkan (github.com)](https://github.com/SaschaWillems/Vulkan/blob/master/shaders/glsl/computecloth/cloth.comp)
  - lecture 14, 15, 16 from  [The Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html)
  - [python with warp code in lec 16](https://github.com/matthias-research/pages/blob/master/tenMinutePhysics/16-GPUCloth.py)


|     |     |
| --- | --- |
| ![Untitled (25)](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/eb9abc95-140c-485a-b48b-b1f2095b1065) | ![Untitled (14)](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/4906d557-bef7-4fd1-9f68-a61591ef3672) |
| ![Untitled (21)](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/33c337d0-80e4-4f45-8aaa-09e2116b97bb) | ![Untitled (26)](https://github.com/keechang-choi/Vulkan-Graphics-Example/assets/49244613/428520ff-5fd9-4595-b88a-63adac9340a4) |

