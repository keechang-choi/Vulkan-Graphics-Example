if(WIN32)
  message(STATUS "OS: windows")
else()
  message(STATUS "OS: not windows")
endif()

cmake_minimum_required(VERSION 3.11.0)

set(NAME VulkanGraphicsExample)

if(WIN32)
  if("$ENV{VK_SDK_PATH}" STREQUAL "")
    set(VK_SDK_PATH C:/VulkanSDK/1.3.216.0)
  else()
    set(VK_SDK_PATH "$ENV{VK_SDK_PATH}")
    file(TO_CMAKE_PATH ${VK_SDK_PATH} VK_SDK_PATH)
  endif()

  message("VK_SDK_PATH = ${VK_SDK_PATH}")

  message(STATUS "using ${CMAKE_GENERATOR}")

  if(CMAKE_GENERATOR STREQUAL "MinGW Makefiles")
    if("$ENV{MINGW_PATH}" STREQUAL "")
      set(MINGW_PATH "C:/Program\ Files/mingw-w64/x86_64-12.2.0-release-posix-seh-rt_v10-rev1/mingw64")
    else()
      set(MINGW_PATH "$ENV{MINGW_PATH}")
      file(TO_CMAKE_PATH ${MINGW_PATH} MINGW_PATH)
    endif()

    message("MINGW_PATH = ${MINGW_PATH}")
    set(USE_MINGW "True")
    set(CMAKE_C_COMPILER ${MINGW_PATH}/bin/gcc.exe)
    set(CMAKE_CXX_COMPILER ${MINGW_PATH}/bin/g++.exe)
  endif()
endif()

project(${NAME})

if(DEFINED VK_SDK_PATH)
  set(Vulkan_INCLUDE_DIRS "${VK_SDK_PATH}/Include")
  set(Vulkan_LIBRARIES "${VK_SDK_PATH}/Lib")
  set(Vulkan_FOUND "True")
else()
  find_package(Vulkan REQUIRED)
  message(STATUS "Found Vulkan: $ENV{VULKAN_SDK}")
endif()

if(NOT Vulkan_FOUND)
  message(FATAL_ERROR "Could not find Vulkan library!")
else()
  message(STATUS "Using vulkan lib at: ${Vulkan_LIBRARIES}")
endif()

include_directories(
  ${Vulkan_INCLUDE_DIRS}
  external
  external/glm
  external/glfw/include
  external/Vulkan-Hpp
  external/imgui
  src/base
)

if(USE_MINGW)
  include_directories(${MINGW_PATH}/include)
  link_directories(${MINGW_PATH}/lib)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/external/glfw)
add_subdirectory(src/base)
add_subdirectory(src/examples)

# ############# Build SHADERS #######################

# Find all vertex and fragment sources within shaders directory
# taken from VBlancos vulkan tutorial
# https://github.com/vblanco20-1/vulkan-guide/blob/all-chapters/CMakeLists.txt
find_program(GLSL_VALIDATOR glslangValidator HINTS
  ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}
  /usr/bin
  /usr/local/bin
  ${VK_SDK_PATH}/Bin
  ${VK_SDK_PATH}/Bin32
  $ENV{VULKAN_SDK}/Bin/
  $ENV{VULKAN_SDK}/Bin32/
)

# get all .vert and .frag files in shaders directory
file(GLOB_RECURSE GLSL_SOURCE_FILES
  "${PROJECT_SOURCE_DIR}/shaders/*.frag"
  "${PROJECT_SOURCE_DIR}/shaders/*.vert"
  "${PROJECT_SOURCE_DIR}/shaders/*.comp"
)

foreach(GLSL ${GLSL_SOURCE_FILES})
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${PROJECT_SOURCE_DIR}/shaders/${FILE_NAME}.spv")
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
    DEPENDS ${GLSL})
  list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach(GLSL)

add_custom_target(
  Shaders
  DEPENDS ${SPIRV_BINARY_FILES}
)