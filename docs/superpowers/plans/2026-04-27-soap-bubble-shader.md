# Soap Bubble Shader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 새 example `soap_bubble`을 만들어 박막 간섭(thin-film interference)을 가시광 분광 적분으로 계산해 두께·굴절률·시야각에 따른 무지개색 변화를 실시간 관찰하는 셰이더 데모를 구현. 그 전에 PBR에 묻혀있던 IBL 코드를 `vgeu_ibl` 모듈로 추출해 공유 사용.

**Architecture:** 두 phase. Phase 1은 IBL 추출 + PBR 마이그레이션 (`vgeu::IBLBaker` 클래스가 HDR→envCubemap→irradiance→prefiltered→BRDF LUT 베이킹, `vgeu::Skybox` 클래스가 매 프레임 스카이박스 그리기). Phase 1 끝에 PBR이 추출 전과 동일하게 동작하는지 검증하는 hard checkpoint. Phase 2는 새 example: pirate gold sphere 모델에 thin-film 셰이더 적용, Wyman 2013 analytical CMF로 분광 적분, Schlick Fresnel 기반 알파, ImGui 6 그룹 파라미터 패널.

**Tech Stack:** Vulkan-HPP RAII, VMA, glm, GLSL 450, glslangValidator, ImGui, CLI11, MinGW + Ninja, clang-format

**Reference Spec:** `docs/superpowers/specs/2026-04-27-soap-bubble-shader-design.md`

---

## 작업 환경 메모

- 모든 bash 명령어는 `rtk` 프리픽스 (e.g., `rtk git add`, `rtk git commit`)
- 모든 `.cpp/.hpp` 변경은 commit 전 `rtk clang-format -i <file>` 적용
- 빌드: `./mingwBuild.bat Debug` (validation layer on)
- 실행: `./build/<example_name>.exe`
- Validation layer 메시지는 콘솔에 출력. 새 VUID 에러 0개 목표 (PBR 기존 VUID-02697은 별개)

---

## 파일 맵

### Phase 1: IBL 추출

| 상태     | 경로                                                                       | 역할                                                                 |
| -------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **MOVE** | `shaders/pbr/equirect.{vert,frag}` → `shaders/common/equirect.{vert,frag}` | equirectangular HDR → cubemap face                                   |
| **MOVE** | `shaders/pbr/irradiance.frag` → `shaders/common/irradiance.frag`           | irradiance 적분                                                      |
| **MOVE** | `shaders/pbr/prefilter.frag` → `shaders/common/prefilter.frag`             | GGX prefilter                                                        |
| **MOVE** | `shaders/pbr/brdf_lut.{vert,frag}` → `shaders/common/brdf_lut.{vert,frag}` | BRDF LUT                                                             |
| **MOVE** | `shaders/pbr/skybox.{vert,frag}` → `shaders/common/skybox.{vert,frag}`     | 스카이박스                                                           |
| **신규** | `src/base/vgeu_ibl.hpp`                                                    | `IBLBaker`, `Skybox`, `IBLBakeConfig` 선언                           |
| **신규** | `src/base/vgeu_ibl.cpp`                                                    | 위 구현 (~700줄, pbr.cpp에서 추출)                                   |
| **수정** | `src/examples/pbr/pbr.hpp`                                                 | IBL/skybox 멤버를 `IBLBaker`/`Skybox`로 교체                         |
| **수정** | `src/examples/pbr/pbr.cpp`                                                 | IBL 함수 ~800줄 제거 + IBLBaker/Skybox 호출로 교체, 셰이더 경로 갱신 |

`src/base/CMakeLists.txt`는 `GLOB_RECURSE`라 신규 `.cpp` 자동 포함, 수정 불필요.

### Phase 2: soap_bubble Example

| 상태     | 경로                                       | 역할                                 |
| -------- | ------------------------------------------ | ------------------------------------ |
| **신규** | `src/examples/soap_bubble/soap_bubble.hpp` | `VgeExample`, `Options`, UBO 구조체  |
| **신규** | `src/examples/soap_bubble/soap_bubble.cpp` | example 구현 (~700~900줄)            |
| **신규** | `shaders/soap_bubble/bubble.vert`          | MVP, 월드 위치/노말/UV 패스          |
| **신규** | `shaders/soap_bubble/bubble.frag`          | 박막 간섭 + 분광 적분 + Fresnel 알파 |

`src/examples/CMakeLists.txt`도 자동 발견. CMake 수정 불필요.

---

# Phase 1: IBL 추출 (vgeu_ibl + PBR 마이그레이션)

목표: PBR이 추출 전후 시각적으로 동일하게 동작. 회귀가 발생하면 Phase 2 진입 금지.

---

## Task 1: shaders/common/ 디렉토리 만들고 IBL/skybox 셰이더 이동

**Files:**
- Move: `shaders/pbr/equirect.vert` → `shaders/common/equirect.vert`
- Move: `shaders/pbr/equirect.frag` → `shaders/common/equirect.frag`
- Move: `shaders/pbr/irradiance.frag` → `shaders/common/irradiance.frag`
- Move: `shaders/pbr/prefilter.frag` → `shaders/common/prefilter.frag`
- Move: `shaders/pbr/brdf_lut.vert` → `shaders/common/brdf_lut.vert`
- Move: `shaders/pbr/brdf_lut.frag` → `shaders/common/brdf_lut.frag`
- Move: `shaders/pbr/skybox.vert` → `shaders/common/skybox.vert`
- Move: `shaders/pbr/skybox.frag` → `shaders/common/skybox.frag`

`pbr/`에 남는 셰이더: `pbr.vert/.frag`, `mrt.vert/.frag`, `sprite.vert/.frag` (이건 PBR 전용).

- [ ] **Step 1: `shaders/common/` 디렉토리 생성**

```bash
mkdir -p shaders/common
```

- [ ] **Step 2: 8개 셰이더 파일 git mv로 이동**

```bash
rtk git mv shaders/pbr/equirect.vert    shaders/common/equirect.vert
rtk git mv shaders/pbr/equirect.frag    shaders/common/equirect.frag
rtk git mv shaders/pbr/irradiance.frag  shaders/common/irradiance.frag
rtk git mv shaders/pbr/prefilter.frag   shaders/common/prefilter.frag
rtk git mv shaders/pbr/brdf_lut.vert    shaders/common/brdf_lut.vert
rtk git mv shaders/pbr/brdf_lut.frag    shaders/common/brdf_lut.frag
rtk git mv shaders/pbr/skybox.vert      shaders/common/skybox.vert
rtk git mv shaders/pbr/skybox.frag      shaders/common/skybox.frag
```

`shaders/pbr/`의 기존 `.spv` 파일들은 빌드 시 재생성되므로 수동 삭제 불필요. 다음 빌드에서 stale `.spv`가 남을 수 있으므로 정리:

```bash
rm -f shaders/pbr/equirect.vert.spv shaders/pbr/equirect.frag.spv \
      shaders/pbr/irradiance.frag.spv shaders/pbr/prefilter.frag.spv \
      shaders/pbr/brdf_lut.vert.spv shaders/pbr/brdf_lut.frag.spv \
      shaders/pbr/skybox.vert.spv shaders/pbr/skybox.frag.spv
```

- [ ] **Step 3: 결과 확인**

```bash
rtk ls shaders/common
rtk ls shaders/pbr
```

Expected: `shaders/common/`에 8개 셰이더, `shaders/pbr/`에는 `pbr.vert/.frag`, `mrt.vert/.frag`, `sprite.vert/.frag`만 남음.

- [ ] **Step 4: 아직 commit 하지 않음** — 다음 task에서 PBR 경로 갱신과 함께 한 commit으로.

---

## Task 2: PBR의 셰이더 경로를 `common/`로 갱신

**Files:**
- Modify: `src/examples/pbr/pbr.cpp`

PBR이 IBL 셰이더를 `/pbr/`에서 읽으니까, 이동한 8개 모두 `/common/`으로 경로를 갱신. 기존 `/pbr/pbr.{vert,frag}`, `/pbr/mrt.{vert,frag}`, `/pbr/sprite.{vert,frag}`는 그대로.

- [ ] **Step 1: 8군데 경로 수정**

`src/examples/pbr/pbr.cpp`에서 다음 라인들 수정 (라인 번호는 추출 작업 전 기준, grep으로 정확히 찾기):

```bash
rtk grep -n 'getShadersPath() \+ "/pbr/' src/examples/pbr/pbr.cpp
```

각 결과에서 `/pbr/<셰이더>` → `/common/<셰이더>`로 변경 (단, `pbr.{vert,frag}`, `mrt.{vert,frag}`, `sprite.{vert,frag}`는 그대로):

| 변경 전                      | 변경 후                         |
| ---------------------------- | ------------------------------- |
| `"/pbr/equirect.vert.spv"`   | `"/common/equirect.vert.spv"`   |
| `"/pbr/equirect.frag.spv"`   | `"/common/equirect.frag.spv"`   |
| `"/pbr/irradiance.frag.spv"` | `"/common/irradiance.frag.spv"` |
| `"/pbr/prefilter.frag.spv"`  | `"/common/prefilter.frag.spv"`  |
| `"/pbr/brdf_lut.vert.spv"`   | `"/common/brdf_lut.vert.spv"`   |
| `"/pbr/brdf_lut.frag.spv"`   | `"/common/brdf_lut.frag.spv"`   |
| `"/pbr/skybox.vert.spv"`     | `"/common/skybox.vert.spv"`     |
| `"/pbr/skybox.frag.spv"`     | `"/common/skybox.frag.spv"`     |

- [ ] **Step 2: 빌드**

```bash
./mingwBuild.bat Debug
```

Expected: 컴파일 성공, 새 `shaders/common/*.spv` 생성됨, validation 메시지 없음.

- [ ] **Step 3: 실행 확인**

```bash
./build/pbr.exe
```

Expected: 추출 작업 전과 시각적으로 동일한 PBR 렌더 (skybox + sphere grid). useIBL on/off, useJitter on/off, skyboxLod 슬라이더 모두 정상 동작.

5초 이상 돌리면서 ImGui 토글들 한 번씩 굴려본 후 종료.

- [ ] **Step 4: clang-format 적용 + commit**

```bash
rtk clang-format -i src/examples/pbr/pbr.cpp
rtk git add shaders/common shaders/pbr src/examples/pbr/pbr.cpp
rtk git commit -m "refactor(shaders): move IBL+skybox shaders to shaders/common/"
```

---

## Task 3: `vgeu_ibl.hpp` 작성 — 인터페이스만

**Files:**
- Create: `src/base/vgeu_ibl.hpp`

선언만 작성. 구현은 다음 task부터.

- [ ] **Step 1: `src/base/vgeu_ibl.hpp` 작성**

```cpp
#pragma once

#include "vgeu_buffer.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <Vulkan-Hpp/vulkan/vulkan.hpp>
#include <Vulkan-Hpp/vulkan/vulkan_raii.hpp>
#include <glm/glm.hpp>

// std
#include <memory>
#include <string>
#include <vector>

namespace vgeu {

struct IBLBakeConfig {
  std::string hdrPath;
  std::string commonShadersPath;
  uint32_t envCubemapSize = 512;
  uint32_t irradianceSize = 64;
  uint32_t prefilteredSize = 128;
  uint32_t brdfLutSize = 512;
  uint32_t irradianceSamples = 2048;
  uint32_t prefilteredSamples = 1024;
  bool useJitter = true;
};

class IBLBaker {
 public:
  IBLBaker(const vk::raii::Device& device, VmaAllocator allocator,
           const vk::raii::Queue& transferQueue,
           const vk::raii::CommandPool& commandPool);

  // Full bake: HDR → envCubemap → irradiance → prefiltered → BRDF LUT.
  void bake(const IBLBakeConfig& config);

  // Re-bake irradiance + prefiltered only (e.g., on jitter toggle).
  void rebakeFiltering(const IBLBakeConfig& config);

  const VgeuImage& envCubemap() const { return *envCubemap_; }
  const VgeuImage& irradianceMap() const { return *irradianceMap_; }
  const VgeuImage& prefilteredMap() const { return *prefilteredMap_; }
  const VgeuImage& brdfLut() const { return *brdfLut_; }
  const vk::raii::Sampler& iblSampler() const { return iblSampler_; }
  const vk::raii::Sampler& hdrSampler() const { return hdrSampler_; }

 private:
  // Internal pipeline stages
  void createSamplersAndCaptureMatrices();
  void loadHdr(const std::string& path);
  void buildEnvCubemap(const IBLBakeConfig&);
  void buildIrradianceMap(const IBLBakeConfig&);
  void buildPrefilteredMap(const IBLBakeConfig&);
  void buildBrdfLut(const IBLBakeConfig&);

  const vk::raii::Device& device_;
  VmaAllocator allocator_;
  const vk::raii::Queue& transferQueue_;
  const vk::raii::CommandPool& commandPool_;

  std::unique_ptr<VgeuImage> hdrTexture_;
  std::unique_ptr<VgeuImage> envCubemap_;
  std::unique_ptr<VgeuImage> irradianceMap_;
  std::unique_ptr<VgeuImage> prefilteredMap_;
  std::unique_ptr<VgeuImage> brdfLut_;
  vk::raii::Sampler iblSampler_ = nullptr;
  vk::raii::Sampler hdrSampler_ = nullptr;
  glm::mat4 captureProj_;
  std::vector<glm::mat4> captureViews_;
};

class Skybox {
 public:
  Skybox(const vk::raii::Device& device,
         const vk::raii::PipelineCache& pipelineCache,
         const vk::raii::DescriptorPool& descPool,
         const vk::raii::RenderPass& renderPass,
         const std::string& commonShadersPath, const IBLBaker& iblBaker,
         uint32_t maxFramesInFlight);

  // Bind pipeline + descriptor + push constants and issue draw.
  void draw(const vk::raii::CommandBuffer& cmd, uint32_t frameIndex,
            const glm::mat4& view, const glm::mat4& proj, float lod = 0.0f);

 private:
  const vk::raii::Device& device_;
  vk::raii::DescriptorSetLayout descSetLayout_ = nullptr;
  vk::raii::PipelineLayout pipelineLayout_ = nullptr;
  vk::raii::Pipeline pipeline_ = nullptr;
  std::vector<vk::raii::DescriptorSet> descriptorSets_;
};

}  // namespace vgeu
```

- [ ] **Step 2: 헤더만 컴파일 확인 (구현 없음이라 link 에러는 다음 task에서)**

```bash
./mingwBuild.bat Debug
```

Expected: `vgeu_ibl.hpp`만 있고 `.cpp`가 없으니 base 라이브러리는 변화 없이 빌드됨 (헤더는 include 되지 않은 상태). PBR도 그대로 동작.

- [ ] **Step 3: clang-format + 임시 commit (다음 task와 묶을 거라 commit 안 해도 됨)**

```bash
rtk clang-format -i src/base/vgeu_ibl.hpp
```

(commit은 Task 11 끝에 모아서)

---

## Task 4: `vgeu_ibl.cpp` skeleton

**Files:**
- Create: `src/base/vgeu_ibl.cpp`

생성자, 빈 함수 본체 작성. 컴파일/링크 통과 확인.

- [ ] **Step 1: `src/base/vgeu_ibl.cpp` 작성 (skeleton)**

```cpp
#include "vgeu_ibl.hpp"

#include "vgeu_utils.hpp"

// libs
#include <glm/gtc/matrix_transform.hpp>
#define STB_IMAGE_IMPLEMENTATION
// stb_image.h가 이미 다른 곳에서 IMPLEMENTATION 정의되어 있으면 위 줄 제거.
// 이 프로젝트는 vgeu_gltf.cpp에서 STB_IMAGE_IMPLEMENTATION 이미 정의 → 제거.
#undef STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

// std
#include <cassert>
#include <cmath>
#include <cstring>

namespace vgeu {

IBLBaker::IBLBaker(const vk::raii::Device& device, VmaAllocator allocator,
                   const vk::raii::Queue& transferQueue,
                   const vk::raii::CommandPool& commandPool)
    : device_(device),
      allocator_(allocator),
      transferQueue_(transferQueue),
      commandPool_(commandPool) {
  createSamplersAndCaptureMatrices();
}

void IBLBaker::createSamplersAndCaptureMatrices() {
  // Task 5에서 이식
}

void IBLBaker::loadHdr(const std::string& path) {
  // Task 5에서 이식
}

void IBLBaker::buildEnvCubemap(const IBLBakeConfig&) {
  // Task 6에서 이식
}

void IBLBaker::buildIrradianceMap(const IBLBakeConfig&) {
  // Task 7에서 이식
}

void IBLBaker::buildPrefilteredMap(const IBLBakeConfig&) {
  // Task 8에서 이식
}

void IBLBaker::buildBrdfLut(const IBLBakeConfig&) {
  // Task 9에서 이식
}

void IBLBaker::bake(const IBLBakeConfig& config) {
  loadHdr(config.hdrPath);
  buildEnvCubemap(config);
  buildIrradianceMap(config);
  buildPrefilteredMap(config);
  buildBrdfLut(config);
}

void IBLBaker::rebakeFiltering(const IBLBakeConfig& config) {
  buildIrradianceMap(config);
  buildPrefilteredMap(config);
}

// Skybox 구현은 Task 11에서

}  // namespace vgeu
```

stb_image는 `external/`에 있고, `vgeu_gltf.cpp`가 `STB_IMAGE_IMPLEMENTATION`을 이미 정의했는지 확인 필요. 만약 그렇다면 위 코드의 `STB_IMAGE_IMPLEMENTATION` 부분 제거하고 그냥 `#include <stb_image.h>`만 (선언 사용).

- [ ] **Step 2: STB_IMAGE_IMPLEMENTATION 위치 확인**

```bash
rtk grep -n 'STB_IMAGE_IMPLEMENTATION' src external
```

Expected: 한 군데에서만 IMPLEMENTATION 정의. `vgeu_ibl.cpp`에서는 정의하지 말고 `<stb_image.h>`만 include.

- [ ] **Step 3: 빌드**

```bash
./mingwBuild.bat Debug
```

Expected: 컴파일 성공. 이 단계에선 PBR은 여전히 자기 IBL 코드를 사용 중이라 동작 변화 없음.

- [ ] **Step 4: clang-format**

```bash
rtk clang-format -i src/base/vgeu_ibl.cpp
```

---

## Task 5: `IBLBaker::createSamplersAndCaptureMatrices()` + `loadHdr()` 이식

**Files:**
- Modify: `src/base/vgeu_ibl.cpp`
- Reference: `src/examples/pbr/pbr.cpp:288-330` (loadHdrTexture), `src/examples/pbr/pbr.cpp:1087-1130` (prepareIBL)

- [ ] **Step 1: `createSamplersAndCaptureMatrices()` 본체 작성**

`src/examples/pbr/pbr.cpp:1087-1130` 의 `prepareIBL()` 본체를 그대로 이식. `iblSampler`, `hdrSampler`, `captureProj`, `captureViews` 모두 멤버 (`iblSampler_`, `hdrSampler_`, `captureProj_`, `captureViews_`)로:

```cpp
void IBLBaker::createSamplersAndCaptureMatrices() {
  vk::SamplerCreateInfo samplerCI(
      {}, vk::Filter::eLinear, vk::Filter::eLinear,
      vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge, 0.f, false, 1.f, false,
      vk::CompareOp::eNever, 0.f,
      static_cast<float>(static_cast<uint32_t>(std::floor(std::log2(512))) + 1),
      vk::BorderColor::eFloatOpaqueWhite);
  iblSampler_ = vk::raii::Sampler(device_, samplerCI);

  vk::SamplerCreateInfo hdrSamplerCI(
      {}, vk::Filter::eLinear, vk::Filter::eLinear,
      vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eRepeat,
      vk::SamplerAddressMode::eClampToEdge,
      vk::SamplerAddressMode::eClampToEdge, 0.f, false, 1.f, false,
      vk::CompareOp::eNever, 0.f,
      static_cast<float>(static_cast<uint32_t>(std::floor(std::log2(512))) + 1),
      vk::BorderColor::eFloatOpaqueWhite);
  hdrSampler_ = vk::raii::Sampler(device_, hdrSamplerCI);

  captureProj_ = glm::perspective(glm::radians(90.f), 1.f, 0.1f, 512.f);
  captureViews_ = std::vector<glm::mat4>{
      glm::lookAt(glm::vec3(0), glm::vec3(1, 0, 0), glm::vec3(0, -1, 0)),
      glm::lookAt(glm::vec3(0), glm::vec3(-1, 0, 0), glm::vec3(0, -1, 0)),
      glm::lookAt(glm::vec3(0), glm::vec3(0, 1, 0), glm::vec3(0, 0, 1)),
      glm::lookAt(glm::vec3(0), glm::vec3(0, -1, 0), glm::vec3(0, 0, -1)),
      glm::lookAt(glm::vec3(0), glm::vec3(0, 0, 1), glm::vec3(0, -1, 0)),
      glm::lookAt(glm::vec3(0), glm::vec3(0, 0, -1), glm::vec3(0, -1, 0)),
  };
}
```

PBR의 `prepareIBL()`에 후속으로 `buildEnvCubemap` 등 호출이 있을 수 있으니 거기까지가 아니라 위 부분(샘플러+캡처 매트릭스)까지만 이식.

- [ ] **Step 2: `loadHdr()` 본체 작성**

`src/examples/pbr/pbr.cpp:288-330`의 `loadHdrTexture()` 본체를 이식. 멤버 변수명을 `_` 접미사로, `globalAllocator->getAllocator()` 대신 `allocator_`로, `device`/`commandPool`/`queue` → `device_`/`commandPool_`/`transferQueue_`로 치환:

```cpp
void IBLBaker::loadHdr(const std::string& path) {
  int w, h, c;
  float* pixels = stbi_loadf(path.c_str(), &w, &h, &c, 4);
  assert(pixels && "Failed to load HDR file");

  vk::DeviceSize size = static_cast<vk::DeviceSize>(w) * h * 4 * sizeof(float);

  vgeu::VgeuBuffer staging(
      allocator_, size, 1, vk::BufferUsageFlagBits::eTransferSrc,
      VMA_MEMORY_USAGE_AUTO,
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
          VMA_ALLOCATION_CREATE_MAPPED_BIT);
  std::memcpy(staging.getMappedData(), pixels, size);
  stbi_image_free(pixels);

  hdrTexture_ = std::make_unique<vgeu::VgeuImage>(
      device_, allocator_, vk::Format::eR32G32B32A32Sfloat,
      vk::Extent2D{(uint32_t)w, (uint32_t)h}, vk::ImageTiling::eOptimal,
      vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
      vk::ImageLayout::eUndefined, VmaMemoryUsage::VMA_MEMORY_USAGE_AUTO,
      VmaAllocationCreateFlagBits::VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
      vk::ImageAspectFlagBits::eColor, 1);

  vk::BufferImageCopy region(
      0, 0, 0,
      vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      vk::Offset3D{0, 0, 0}, vk::Extent3D{(uint32_t)w, (uint32_t)h, 1});

  vgeu::oneTimeSubmit(
      device_, commandPool_, transferQueue_,
      [&](const vk::raii::CommandBuffer& cmd) {
        vgeu::setImageLayout(
            cmd, hdrTexture_->getImage(), vk::Format::eR32G32B32A32Sfloat, 0, 1,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        cmd.copyBufferToImage(staging.getBuffer(), hdrTexture_->getImage(),
                              vk::ImageLayout::eTransferDstOptimal, region);
        vgeu::setImageLayout(cmd, hdrTexture_->getImage(),
                             vk::Format::eR32G32B32A32Sfloat, 0, 1,
                             vk::ImageLayout::eTransferDstOptimal,
                             vk::ImageLayout::eShaderReadOnlyOptimal);
      });
}
```

- [ ] **Step 3: 빌드**

```bash
./mingwBuild.bat Debug
```

Expected: 컴파일 성공. PBR은 여전히 자기 IBL 코드 사용 중이라 동작 변화 없음.

- [ ] **Step 4: clang-format**

```bash
rtk clang-format -i src/base/vgeu_ibl.cpp
```

---

## Task 6: `IBLBaker::buildEnvCubemap()` 이식

**Files:**
- Modify: `src/base/vgeu_ibl.cpp`
- Reference: `src/examples/pbr/pbr.cpp:332-567` (buildEnvCubemap, ~235줄)

PBR의 `buildEnvCubemap()` 본체 이식. 셰이더 경로 `getShadersPath() + "/pbr/equirect.*.spv"` → `config.commonShadersPath + "/equirect.*.spv"`로 치환.

- [ ] **Step 1: `buildEnvCubemap()` 본체 이식**

`src/examples/pbr/pbr.cpp:332-567` 본체를 복사. 다음 치환:
- `device` → `device_`
- `globalAllocator->getAllocator()` → `allocator_`
- `commandPool` → `commandPool_`
- `queue` → `transferQueue_`
- `hdrTexture` → `hdrTexture_` (uniqueptr deref 동일)
- `envCubemap` → `envCubemap_`
- `hdrSampler` → `hdrSampler_`
- `captureProj` → `captureProj_`
- `captureViews` → `captureViews_`
- `getShadersPath() + "/pbr/equirect.vert.spv"` → `config.commonShadersPath + "/equirect.vert.spv"`
- `getShadersPath() + "/pbr/equirect.frag.spv"` → `config.commonShadersPath + "/equirect.frag.spv"`
- 하드코딩된 `dim = 512`, `numMips = log2(512)+1` 부분은 `config.envCubemapSize`와 그에 해당하는 `numMips` 계산으로 치환

함수 시그니처는 `void IBLBaker::buildEnvCubemap(const IBLBakeConfig& config)`. PBR 원본에서 `pipelineCache` 사용처가 있으면 임시로 `vk::raii::PipelineCache(device_, vk::PipelineCacheCreateInfo{})` 로컬 생성으로 교체 (성능 영향 없음, IBL 베이크는 startup 1회).

- [ ] **Step 2: 빌드**

```bash
./mingwBuild.bat Debug
```

Expected: 컴파일 성공.

- [ ] **Step 3: clang-format**

```bash
rtk clang-format -i src/base/vgeu_ibl.cpp
```

---

## Task 7: `IBLBaker::buildIrradianceMap()` 이식

**Files:**
- Modify: `src/base/vgeu_ibl.cpp`
- Reference: `src/examples/pbr/pbr.cpp:569-762` (buildIrradianceMap, ~193줄)

`buildIrradianceMap()` 이식. 동일 치환 + jitter 처리:
- `opts.useJitter` → `config.useJitter` (specialization constant 또는 push constant 위치)
- 셰이더 경로: `/pbr/irradiance.frag.spv` → `config.commonShadersPath + "/irradiance.frag.spv"`, `/pbr/equirect.vert.spv` → `config.commonShadersPath + "/equirect.vert.spv"`
- 해상도: `dim = 64` → `config.irradianceSize`
- 샘플 수: 하드코딩된 2048 → `config.irradianceSamples` (push constant 또는 specialization constant)

- [ ] **Step 1: 함수 본체 이식 + 치환**

PBR 원본을 그대로 복사 후 위 치환 적용. 복사 후 그 안의 `envCubemap` (소스로 사용) 참조가 `envCubemap_` (멤버)을 가리키도록 통일.

- [ ] **Step 2: 빌드 + clang-format**

```bash
./mingwBuild.bat Debug
rtk clang-format -i src/base/vgeu_ibl.cpp
```

Expected: 컴파일 성공.

---

## Task 8: `IBLBaker::buildPrefilteredMap()` 이식

**Files:**
- Modify: `src/base/vgeu_ibl.cpp`
- Reference: `src/examples/pbr/pbr.cpp:764-968` (buildPrefilteredMap, ~204줄)

동일 패턴. 차이점:
- 셰이더: `/pbr/prefilter.frag.spv` → `config.commonShadersPath + "/prefilter.frag.spv"`
- 해상도: 하드코딩된 prefilter 사이즈 → `config.prefilteredSize`
- 샘플: → `config.prefilteredSamples`
- jitter: → `config.useJitter`

- [ ] **Step 1: 함수 본체 이식 + 치환**

- [ ] **Step 2: 빌드 + clang-format**

```bash
./mingwBuild.bat Debug
rtk clang-format -i src/base/vgeu_ibl.cpp
```

---

## Task 9: `IBLBaker::buildBrdfLut()` 이식

**Files:**
- Modify: `src/base/vgeu_ibl.cpp`
- Reference: `src/examples/pbr/pbr.cpp:990-1085` (buildBrdfLut, ~95줄)

가장 단순. 큐브맵 아니라 2D LUT.
- 셰이더: `/pbr/brdf_lut.{vert,frag}.spv` → `config.commonShadersPath + "/brdf_lut.{vert,frag}.spv"`
- 해상도: → `config.brdfLutSize`

- [ ] **Step 1: 함수 본체 이식**

- [ ] **Step 2: 빌드 + clang-format**

```bash
./mingwBuild.bat Debug
rtk clang-format -i src/base/vgeu_ibl.cpp
```

---

## Task 10: `IBLBaker::rebakeFiltering()` 동작 검증 (스탠드얼론)

**Files:** 없음 (코드 변경 없음, 검증만)

이 task는 단독으로 IBLBaker가 제대로 컴파일되고 링크되는지 sanity check. PBR은 여전히 자기 코드를 사용 중이라 IBLBaker 클래스가 *사용*되지는 않지만, 헤더가 include되고 base 라이브러리가 링크되는지 확인.

- [ ] **Step 1: PBR이 vgeu_ibl을 헤더에서 include하지 않더라도 base 라이브러리 빌드 통과 확인**

```bash
./mingwBuild.bat Debug
```

Expected: `vgeu_ibl.cpp` 컴파일 성공, base 라이브러리 빌드 성공, PBR 동작 변화 없음.

- [ ] **Step 2: 수동 sanity (선택, 스킵 가능)**: 임시로 `pbr.cpp`의 `prepare()` 시작 부분에 `vgeu::IBLBaker tester(device, globalAllocator->getAllocator(), queue, commandPool); vgeu::IBLBakeConfig cfg{...}; tester.bake(cfg);` 추가해 IBL 베이크가 두 번 돌아도 (PBR 원본 + 새 IBLBaker) 충돌 없는지. 확인 후 즉시 제거.

이 sanity는 다음 task에서 PBR 마이그레이션과 함께 자연스럽게 검증되므로 스킵 가능.

---

## Task 11: `Skybox` 클래스 이식

**Files:**
- Modify: `src/base/vgeu_ibl.cpp`
- Reference: `src/examples/pbr/pbr.cpp:1750-1965` (skybox 셋업 부분), `pbr.cpp:2140-2160` (draw 부분)

PBR의 `prepareSkyboxPipeline()` + `prepareSkyboxDescriptors()`를 `Skybox` 생성자로, draw 코드를 `Skybox::draw()`로 옮김.

- [ ] **Step 1: `Skybox` 생성자 작성**

```cpp
Skybox::Skybox(const vk::raii::Device& device,
               const vk::raii::PipelineCache& pipelineCache,
               const vk::raii::DescriptorPool& descPool,
               const vk::raii::RenderPass& renderPass,
               const std::string& commonShadersPath, const IBLBaker& iblBaker,
               uint32_t maxFramesInFlight)
    : device_(device) {
  // Descriptor set layout: binding 0 = envCubemap (samplerCube)
  std::array<vk::DescriptorSetLayoutBinding, 1> bindings{
      vk::DescriptorSetLayoutBinding(
          0, vk::DescriptorType::eCombinedImageSampler, 1,
          vk::ShaderStageFlagBits::eFragment),
  };
  descSetLayout_ = vk::raii::DescriptorSetLayout(
      device_, vk::DescriptorSetLayoutCreateInfo({}, bindings));

  // Push constants: SkyboxPushConstants (view, proj, lod) — 동일 구조체 정의 필요.
  // PBR의 struct를 vgeu_ibl.cpp 익명 namespace에 복사:
  //   struct SkyboxPushConstants { glm::mat4 view; glm::mat4 projection;
  //   float lod; };
  vk::PushConstantRange pushRange(vk::ShaderStageFlagBits::eVertex |
                                      vk::ShaderStageFlagBits::eFragment,
                                  0, sizeof(SkyboxPushConstants));
  pipelineLayout_ = vk::raii::PipelineLayout(
      device_, vk::PipelineLayoutCreateInfo({}, *descSetLayout_, pushRange));

  // Allocate descriptor sets per frame
  descriptorSets_.reserve(maxFramesInFlight);
  for (uint32_t i = 0; i < maxFramesInFlight; ++i) {
    descriptorSets_.push_back(
        std::move(vk::raii::DescriptorSets(
                      device_, vk::DescriptorSetAllocateInfo(
                                   *descPool, *descSetLayout_))
                      .front()));
    vk::DescriptorImageInfo envInfo(*iblBaker.iblSampler(),
                                    iblBaker.envCubemap().getImageView(),
                                    vk::ImageLayout::eShaderReadOnlyOptimal);
    device_.updateDescriptorSets(
        vk::WriteDescriptorSet(*descriptorSets_[i], 0, 0,
                               vk::DescriptorType::eCombinedImageSampler,
                               envInfo),
        nullptr);
  }

  // Pipeline (PBR의 skyboxPipeline 셋업과 동일)
  // 셰이더: commonShadersPath + "/skybox.{vert,frag}.spv"
  // 정점 입력 없음 (셰이더에서 36 vertex 하드코딩, gl_VertexIndex 사용)
  // depth: VK_COMPARE_OP_LESS_OR_EQUAL (skybox.vert가 z=w 트릭)
  // cull: front (큐브 안에서 보는 거라 front face가 카메라 향함)
  // PBR pbr.cpp:1808-1963 부분을 그대로 복사 후 셰이더 경로만 commonShadersPath 사용
}
```

`SkyboxPushConstants` 구조체는 vgeu_ibl.hpp에 추가하거나 익명 namespace로:

```cpp
namespace {
struct SkyboxPushConstants {
  glm::mat4 view;
  glm::mat4 projection;
  float lod = 0.0f;
};
}
```

`익명 namespace`로 vgeu_ibl.cpp 안에 두면 외부에서 안 보임. PBR의 동일 구조체와 ABI 호환만 맞추면 됨.

- [ ] **Step 2: `Skybox::draw()` 작성**

```cpp
void Skybox::draw(const vk::raii::CommandBuffer& cmd, uint32_t frameIndex,
                  const glm::mat4& view, const glm::mat4& proj, float lod) {
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline_);
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipelineLayout_, 0,
                         {*descriptorSets_[frameIndex]}, nullptr);
  SkyboxPushConstants pc;
  pc.view = glm::mat4(glm::mat3(view));  // translation 제거
  pc.projection = proj;
  pc.lod = lod;
  cmd.pushConstants<SkyboxPushConstants>(
      *pipelineLayout_,
      vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0,
      pc);
  cmd.draw(36, 1, 0, 0);  // 큐브 36 vertex
}
```

- [ ] **Step 3: 빌드 + clang-format**

```bash
./mingwBuild.bat Debug
rtk clang-format -i src/base/vgeu_ibl.hpp src/base/vgeu_ibl.cpp
```

Expected: 컴파일 성공.

- [ ] **Step 4: 중간 commit (vgeu_ibl 모듈 자체)**

```bash
rtk git add src/base/vgeu_ibl.hpp src/base/vgeu_ibl.cpp
rtk git commit -m "feat(base): add vgeu_ibl module (IBLBaker, Skybox)"
```

이 시점에서 PBR은 아직 새 모듈을 사용 *안* 함. 다음 task에서 마이그레이션.

---

## Task 12: PBR을 `IBLBaker` + `Skybox`로 마이그레이션 (회귀 검증 — Phase 1 hard checkpoint)

**Files:**
- Modify: `src/examples/pbr/pbr.hpp`
- Modify: `src/examples/pbr/pbr.cpp`

PBR이 자기 IBL 코드 대신 IBLBaker/Skybox를 사용하도록 수정. 이 task가 Phase 1의 마지막. 끝나면 PBR이 추출 전과 동일하게 동작해야 함.

- [ ] **Step 1: `pbr.hpp` 정리 — IBL 멤버 제거 후 IBLBaker/Skybox 멤버 추가**

다음 멤버들 **제거**:
```cpp
std::unique_ptr<vgeu::VgeuImage> hdrTexture;
std::unique_ptr<vgeu::VgeuImage> envCubemap;
std::unique_ptr<vgeu::VgeuImage> irradianceMap;
std::unique_ptr<vgeu::VgeuImage> prefilteredMap;
std::unique_ptr<vgeu::VgeuImage> brdfLut;
vk::raii::Sampler iblSampler = nullptr;
vk::raii::Sampler hdrSampler = nullptr;
glm::mat4 captureProj;
std::vector<glm::mat4> captureViews;

vk::raii::Pipeline skyboxPipeline = nullptr;
vk::raii::PipelineLayout skyboxPipelineLayout = nullptr;
vk::raii::DescriptorSetLayout skyboxDescriptorSetLayout = nullptr;
std::vector<vk::raii::DescriptorSet> skyboxDescriptorSets;
std::unique_ptr<vgeu::VgeuBuffer> skyboxUboBuffer;
```

함수 선언 **제거**:
```cpp
void prepareIBL();
void loadHdrTexture();
void buildEnvCubemap();
void buildIrradianceMap();
void buildPrefilteredMap();
void rebuildIBLFiltering();
void buildBrdfLut();
void prepareSkyboxPipeline();
void prepareSkyboxDescriptors();
```

다음 멤버들 **추가** (`#include "vgeu_ibl.hpp"`도):
```cpp
std::unique_ptr<vgeu::IBLBaker> iblBaker;
std::unique_ptr<vgeu::Skybox> skybox;
vgeu::IBLBakeConfig iblConfig;  // ImGui jitter 토글 시 재사용
```

`iblDescriptorSetLayout`/`iblDescriptorSets`는 PBR composition의 IBL descriptor (set=1)이므로 그대로 유지. 단, 그 descriptor에 binding 되는 image view/sampler는 `iblBaker->irradianceMap().getImageView()` 등으로 변경.

- [ ] **Step 2: `pbr.cpp`의 IBL 함수 정의 6개 + skybox 함수 2개 모두 삭제**

다음 함수 본체 통째로 삭제 (라인 추정):
- `void VgeExample::loadHdrTexture()` (288-330)
- `void VgeExample::buildEnvCubemap()` (332-567)
- `void VgeExample::buildIrradianceMap()` (569-762)
- `void VgeExample::buildPrefilteredMap()` (764-968)
- `void VgeExample::rebuildIBLFiltering()` (970-988)
- `void VgeExample::buildBrdfLut()` (990-1085)
- `void VgeExample::prepareIBL()` (1087-1130)
- `void VgeExample::prepareSkyboxPipeline()` (~1750-1963)
- `void VgeExample::prepareSkyboxDescriptors()` (~1750 부근)

`grep`으로 정확한 라인 찾기:

```bash
rtk grep -n '^void VgeExample::\(loadHdrTexture\|buildEnv\|buildIrr\|buildPref\|rebuildIBL\|buildBrdf\|prepareIBL\|prepareSkybox\)' src/examples/pbr/pbr.cpp
```

각 함수의 시작 `{` 부터 매칭되는 `}` 까지 삭제.

- [ ] **Step 3: `prepare()` 안의 호출 갱신**

`prepare()` (또는 동등 위치)에서 기존:
```cpp
loadHdrTexture();
buildEnvCubemap();
prepareIBL();
buildIrradianceMap();
buildPrefilteredMap();
buildBrdfLut();
prepareSkyboxPipeline();
prepareSkyboxDescriptors();
```

다음으로 교체:
```cpp
iblConfig.hdrPath = getAssetsPath() + "/textures/hdr/tree_lined_driveway_4k.hdr";
iblConfig.commonShadersPath = getShadersPath() + "/common";
iblConfig.useJitter = opts.useJitter;
// 해상도/샘플은 기본값 사용 (PBR 추출 전과 동일)

iblBaker = std::make_unique<vgeu::IBLBaker>(
    device, globalAllocator->getAllocator(), queue, commandPool);
iblBaker->bake(iblConfig);

skybox = std::make_unique<vgeu::Skybox>(
    device, pipelineCache, descriptorPool, *renderPass,
    iblConfig.commonShadersPath, *iblBaker, MAX_CONCURRENT_FRAMES);
```

- [ ] **Step 4: IBL descriptor (set=1, composition pass) 재셋업**

기존 `setupDescriptors()` 안에서 IBL descriptor에 binding 하는 image view들이 `irradianceMap`, `prefilteredMap`, `brdfLut` 멤버 (지금 제거됨)를 사용 중. 이걸 모두 `iblBaker->irradianceMap()`, `iblBaker->prefilteredMap()`, `iblBaker->brdfLut()`로 교체. 샘플러도 `iblSampler` → `iblBaker->iblSampler()`.

- [ ] **Step 5: ImGui jitter 토글 핸들러 갱신**

기존 `rebuildIBLFiltering()` 호출은:
```cpp
if (ImGui::Checkbox("useJitter", &opts.useJitter)) {
  device.waitIdle();
  iblConfig.useJitter = opts.useJitter;
  iblBaker->rebakeFiltering(iblConfig);
  // IBL descriptor 재바인딩 (irradiance/prefiltered가 새로 만들어졌으므로
  // descriptor의 imageView가 stale)
  // setupDescriptors()의 IBL descriptor 작성 부분만 다시 호출하는 헬퍼 필요.
}
```

기존 `rebuildIBLFiltering`이 descriptor 재바인딩까지 했다면 그 패턴 유지. PBR 원본 동작 그대로 보존.

- [ ] **Step 6: skybox draw 호출 갱신**

기존 `buildCommandBuffers()`의 skybox 부분:
```cpp
cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *skyboxPipeline);
cmd.bindDescriptorSets(...);
cmd.pushConstants<SkyboxPushConstants>(...);
cmd.draw(36, 1, 0, 0);
```

다음으로 교체:
```cpp
glm::mat4 view = camera.getView();
glm::mat4 proj = camera.getProjection();
skybox->draw(cmd, currentFrameIndex, view, proj, opts.skyboxLod);
```

- [ ] **Step 7: 빌드 + clang-format**

```bash
rtk clang-format -i src/examples/pbr/pbr.hpp src/examples/pbr/pbr.cpp
./mingwBuild.bat Debug
```

Expected: 컴파일 성공, validation 클린, link 성공.

- [ ] **Step 8: 회귀 검증 (Phase 1 hard checkpoint)**

```bash
./build/pbr.exe
```

다음을 모두 확인:
- [ ] Skybox: 추출 전과 동일하게 렌더 (HDR 환경)
- [ ] Sphere grid: metallic/roughness 셀별 색상 동일
- [ ] `useIBL` 토글: on/off 모두 추출 전과 동일
- [ ] `useJitter` 토글: on/off 모두 추출 전과 동일 (재바이크 발생 확인 — 토글 누르면 잠시 멈춤 후 결과 변화)
- [ ] `skyboxLod` 슬라이더: mip 레벨 슬라이딩 동일
- [ ] Validation 콘솔: 새 VUID 에러 없음 (기존 VUID-02697은 OK)

회귀 발견 시: Phase 2 진입 금지, 어느 task에서 깨졌는지 역추적.

- [ ] **Step 9: Phase 1 완료 commit**

```bash
rtk git add src/examples/pbr/pbr.hpp src/examples/pbr/pbr.cpp
rtk git commit -m "refactor(pbr): migrate IBL bake and skybox to vgeu_ibl module"
```

---

# Phase 1 → Phase 2 Checkpoint

**진행하기 전 확인:**
- [ ] PBR이 추출 전과 시각적으로 동일하게 동작
- [ ] Validation layer 메시지에 새 VUID 에러 없음
- [ ] `vgeu_ibl.hpp/.cpp` commit 완료
- [ ] PBR 마이그레이션 commit 완료
- [ ] `git status`가 깨끗함

OK 면 Phase 2로.

---

# Phase 2: soap_bubble Example

목표: 새 example 디렉토리에서 박막 간섭 셰이더 구현, ImGui로 모든 파라미터 조작 가능.

---

## Task 13: example 디렉토리 + skeleton 작성

**Files:**
- Create: `src/examples/soap_bubble/soap_bubble.hpp`
- Create: `src/examples/soap_bubble/soap_bubble.cpp`

`src/examples/triangle/`을 참고로 한 최소 skeleton. CMake는 자동 발견.

- [ ] **Step 1: `soap_bubble.hpp` 작성 (skeleton)**

```cpp
#pragma once

#include "vge_base.hpp"
#include "vgeu_gltf.hpp"
#include "vgeu_ibl.hpp"
#include "vgeu_texture.hpp"

#include <memory>
#include <optional>

namespace vge {

struct Options {
  // Thin Film
  float thicknessMin = 200.f;  // nm
  float thicknessMax = 800.f;
  float n1 = 1.0f;
  float n2 = 1.33f;
  float n3 = 1.0f;
  int32_t spectralSamples = 16;
  // Thickness Source
  int32_t thicknessMode = 0;  // 0=Texture, 1=Procedural
  float gravityStrength = 1.0f;
  float noiseScale = 2.0f;
  // Animation
  bool useAnimation = false;
  float driftSpeed = 0.2f;
  // Surface & Blending
  float roughness = 0.0f;
  float alphaScale = 1.0f;
  // IBL / Env
  float iblExposure = 4.5f;
  float iblGamma = 2.2f;
  bool useJitter = true;
  float skyboxLod = 0.0f;
  // Debug
  bool showThicknessHeatmap = false;
  bool showFresnelOnly = false;
};

struct GlobalsUbo {
  glm::mat4 view{1.f};
  glm::mat4 projection{1.f};
  glm::vec4 viewPos{0.f};
};

struct BubbleParamsUbo {
  // 16-byte 정렬 중요. std140 레이아웃 가정.
  float thicknessMin;
  float thicknessMax;
  float n1;
  float n2;
  // -- 16 byte boundary --
  float n3;
  int32_t spectralSamples;
  int32_t thicknessMode;
  float gravityStrength;
  // -- 16 --
  float noiseScale;
  int32_t useAnimation;
  float driftSpeed;
  float roughness;
  // -- 16 --
  float alphaScale;
  float iblExposure;
  float iblGamma;
  float time;
  // -- 16 --
  int32_t showThicknessHeatmap;
  int32_t showFresnelOnly;
  float _pad0;
  float _pad1;
};

class VgeExample : public VgeBase {
 public:
  VgeExample();
  ~VgeExample();
  void setupCommandLineParser(CLI::App& app) override;
  void setOptions(const std::optional<Options>& opts);

  void initVulkan() override;
  void getEnabledExtensions() override;
  void getEnabledFeatures() override;
  void prepare() override;
  void render() override;
  void viewChanged() override;
  void onUpdateUIOverlay() override;

  void loadAssets();
  void prepareIBL();
  void prepareUniformBuffers();
  void setupDescriptors();
  void preparePipelines();
  void buildCommandBuffers() override;
  void draw();

  void updateGlobalsUbo();
  void updateBubbleParamsUbo();

  Options opts{};

  // IBL
  std::unique_ptr<vgeu::IBLBaker> iblBaker;
  std::unique_ptr<vgeu::Skybox> skybox;
  vgeu::IBLBakeConfig iblConfig;

  // Bubble model
  std::shared_ptr<vgeu::glTF::Model> bubbleModel;

  // Uniform buffers (per-frame)
  struct UniformBuffers {
    std::unique_ptr<vgeu::VgeuBuffer> globals;
    std::unique_ptr<vgeu::VgeuBuffer> bubbleParams;
  };
  std::vector<UniformBuffers> uniformBuffers;

  GlobalsUbo globalsUbo;
  BubbleParamsUbo bubbleParamsUbo;

  // Pipeline
  vk::raii::DescriptorSetLayout globalsSetLayout = nullptr;
  vk::raii::DescriptorSetLayout bubbleParamsSetLayout = nullptr;
  vk::raii::DescriptorSetLayout heightTexSetLayout = nullptr;
  vk::raii::DescriptorSetLayout envSetLayout = nullptr;
  vk::raii::PipelineLayout bubblePipelineLayout = nullptr;
  vk::raii::Pipeline bubblePipeline = nullptr;

  std::vector<vk::raii::DescriptorSet> globalsDescSets;
  std::vector<vk::raii::DescriptorSet> bubbleParamsDescSets;
  vk::raii::DescriptorSet heightTexDescSet = nullptr;
  std::vector<vk::raii::DescriptorSet> envDescSets;
};

}  // namespace vge
```

- [ ] **Step 2: `soap_bubble.cpp` skeleton (빈 main, base만 호출)**

```cpp
#include "soap_bubble.hpp"

#include "vgeu_utils.hpp"

namespace vge {

VgeExample::VgeExample() : VgeBase() { title = "soap_bubble"; }
VgeExample::~VgeExample() {}

void VgeExample::setupCommandLineParser(CLI::App& app) {
  // Task 26에서 작성
}

void VgeExample::setOptions(const std::optional<Options>& o) {
  if (o) opts = *o;
}

void VgeExample::initVulkan() { VgeBase::initVulkan(); }
void VgeExample::getEnabledExtensions() {}
void VgeExample::getEnabledFeatures() { enabledFeatures.samplerAnisotropy = VK_TRUE; }

void VgeExample::prepare() {
  VgeBase::prepare();
  loadAssets();
  prepareIBL();
  prepareUniformBuffers();
  setupDescriptors();
  preparePipelines();
  prepared = true;
}

void VgeExample::loadAssets() {
  // Task 15
}
void VgeExample::prepareIBL() {
  // Task 14
}
void VgeExample::prepareUniformBuffers() {
  // Task 17
}
void VgeExample::setupDescriptors() {
  // Task 17
}
void VgeExample::preparePipelines() {
  // Task 17
}
void VgeExample::buildCommandBuffers() {
  // Task 14
}
void VgeExample::draw() {}

void VgeExample::render() {
  if (!prepared) return;
  draw();
}
void VgeExample::viewChanged() {}
void VgeExample::onUpdateUIOverlay() {
  // Task 25
}
void VgeExample::updateGlobalsUbo() {}
void VgeExample::updateBubbleParamsUbo() {}

}  // namespace vge

VULKAN_EXAMPLE_MAIN()
```

- [ ] **Step 3: 빌드**

```bash
./mingwBuild.bat Debug
```

Expected: `soap_bubble` 실행 파일이 빌드되지만 실행하면 빈 창만 뜸 (prepare/render 본체가 비어있어서). 그래도 link/compile은 통과해야 함.

- [ ] **Step 4: 실행 sanity (검은 창만 뜨면 OK, 빠르게 종료)**

```bash
./build/soap_bubble.exe
```

Expected: 검은 창. ESC로 종료. validation 에러 없음 (renderpass 시작은 base가 함).

- [ ] **Step 5: clang-format + commit**

```bash
rtk clang-format -i src/examples/soap_bubble/soap_bubble.hpp src/examples/soap_bubble/soap_bubble.cpp
rtk git add src/examples/soap_bubble
rtk git commit -m "feat(soap_bubble): scaffold new example skeleton"
```

---

## Task 14: IBL 베이크 + Skybox 그리기

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.cpp`

skybox만 보이는 상태까지. 모델은 다음 task에서.

- [ ] **Step 1: `prepareIBL()` 본체 작성**

```cpp
void VgeExample::prepareIBL() {
  iblConfig.hdrPath = getAssetsPath() + "/textures/hdr/tree_lined_driveway_4k.hdr";
  iblConfig.commonShadersPath = getShadersPath() + "/common";
  iblConfig.useJitter = opts.useJitter;

  iblBaker = std::make_unique<vgeu::IBLBaker>(
      device, globalAllocator->getAllocator(), queue, commandPool);
  iblBaker->bake(iblConfig);

  skybox = std::make_unique<vgeu::Skybox>(
      device, pipelineCache, descriptorPool, *renderPass,
      iblConfig.commonShadersPath, *iblBaker, MAX_CONCURRENT_FRAMES);
}
```

- [ ] **Step 2: `buildCommandBuffers()` 작성 (skybox만)**

```cpp
void VgeExample::buildCommandBuffers() {
  for (size_t i = 0; i < drawCmdBuffers.size(); ++i) {
    const auto& cmd = drawCmdBuffers[i];
    cmd.begin(vk::CommandBufferBeginInfo());

    std::array<vk::ClearValue, 2> clearValues;
    clearValues[0].color = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    clearValues[1].depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

    vk::RenderPassBeginInfo rpBegin(*renderPass, *frameBuffers[i],
                                    {{0, 0}, {width, height}}, clearValues);
    cmd.beginRenderPass(rpBegin, vk::SubpassContents::eInline);

    cmd.setViewport(0, vk::Viewport(0.f, 0.f, (float)width, (float)height,
                                    0.f, 1.f));
    cmd.setScissor(0, vk::Rect2D({0, 0}, {width, height}));

    skybox->draw(cmd, static_cast<uint32_t>(i), camera.getView(),
                 camera.getProjection(), opts.skyboxLod);

    drawUI(cmd);
    cmd.endRenderPass();
    cmd.end();
  }
}
```

- [ ] **Step 3: `draw()` 작성 — base 패턴**

```cpp
void VgeExample::draw() {
  prepareFrame();
  buildCommandBuffers();
  vk::SubmitInfo submit({}, {}, *drawCmdBuffers[currentImageIndex]);
  queue.submit(submit);
  submitFrame();
}
```

(buildCommandBuffers를 매 프레임 호출하는 단순 패턴. PBR도 동일.)

- [ ] **Step 4: `viewChanged()` — 빈 함수 유지 OK** (카메라는 base가 처리)

- [ ] **Step 5: 빌드 + 실행**

```bash
./mingwBuild.bat Debug
./build/soap_bubble.exe
```

Expected: HDR 스카이박스가 보임. 마우스 우클릭 + 드래그로 회전 가능. 5초 돌려보고 종료. validation 클린.

- [ ] **Step 6: clang-format + commit**

```bash
rtk clang-format -i src/examples/soap_bubble/soap_bubble.cpp
rtk git add src/examples/soap_bubble/soap_bubble.cpp
rtk git commit -m "feat(soap_bubble): bake IBL and draw skybox"
```

---

## Task 15: pirate gold 모델 로드

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.cpp`

`vgeu_gltf::Model` 로드. PBR 셔이더 패턴 참고.

- [ ] **Step 1: `loadAssets()` 본체 작성**

```cpp
void VgeExample::loadAssets() {
  bubbleModel = std::make_shared<vgeu::glTF::Model>(
      device, globalAllocator->getAllocator(), queue, commandPool,
      MAX_CONCURRENT_FRAMES);
  vgeu::glTF::FileLoadingFlags loadFlags =
      vgeu::glTF::FileLoadingFlagBits::kPreTransformVertices |
      vgeu::glTF::FileLoadingFlagBits::kPreMultiplyVertexColors |
      vgeu::glTF::FileLoadingFlagBits::kFlipY;
  bubbleModel->loadFromFile(
      getAssetsPath() + "/models/pirate_gold/scene.gltf", loadFlags);
}
```

PBR 코드의 정확한 model load 호출 시그니처를 grep으로 확인 후 맞춤:

```bash
rtk grep -n 'loadFromFile' src/examples/pbr/pbr.cpp
```

- [ ] **Step 2: 빌드 sanity**

```bash
./mingwBuild.bat Debug
```

Expected: 컴파일 성공 (실행 결과는 아직 모델 안 그림 — bubble 파이프라인 없음).

- [ ] **Step 3: 실행 — 여전히 skybox만 보임**

```bash
./build/soap_bubble.exe
```

- [ ] **Step 4: clang-format + commit**

```bash
rtk clang-format -i src/examples/soap_bubble/soap_bubble.cpp
rtk git add src/examples/soap_bubble/soap_bubble.cpp
rtk git commit -m "feat(soap_bubble): load pirate gold model"
```

---

## Task 16: `bubble.vert` + 최소 `bubble.frag` (단색 출력)

**Files:**
- Create: `shaders/soap_bubble/bubble.vert`
- Create: `shaders/soap_bubble/bubble.frag`

bubble 셰이더의 최소 형태. 색은 다음 task부터 박막으로 교체.

- [ ] **Step 1: `shaders/soap_bubble/bubble.vert`**

```glsl
#version 450

layout(set = 0, binding = 0) uniform Globals {
  mat4 view;
  mat4 projection;
  vec4 viewPos;
} globals;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inColor;
layout(location = 4) in vec4 inJoint;
layout(location = 5) in vec4 inWeight;
layout(location = 6) in vec4 inTangent;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outWorldNormal;
layout(location = 2) out vec2 outUV;

void main() {
  // 모델은 PreTransformVertices 적용되어 있어 model matrix는 identity 가정
  outWorldPos = inPosition;
  outWorldNormal = inNormal;
  outUV = inUV;
  gl_Position = globals.projection * globals.view * vec4(inPosition, 1.0);
}
```

vertex 입력 layout은 `vgeu_gltf`가 정의한 7개 attribute에 맞춤. PBR의 `pbr.vert`/`mrt.vert`가 같은 패턴이니 거기 참고:

```bash
rtk grep -n 'layout(location' shaders/pbr/mrt.vert
```

- [ ] **Step 2: `shaders/soap_bubble/bubble.frag` (최소, 빨강)**

```glsl
#version 450

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec4 outColor;

void main() {
  outColor = vec4(1.0, 0.0, 0.0, 1.0);  // 일단 빨강
}
```

- [ ] **Step 3: 빌드 (셰이더 컴파일만 검증)**

```bash
./mingwBuild.bat Debug
```

Expected: `bubble.vert.spv`, `bubble.frag.spv` 생성됨.

- [ ] **Step 4: commit**

```bash
rtk git add shaders/soap_bubble
rtk git commit -m "feat(soap_bubble): add minimal bubble.vert/.frag"
```

---

## Task 17: bubble pipeline + descriptor 셋업

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.cpp`

descriptor set, pipeline layout, pipeline 생성. 실행 결과는 빨간 sphere가 skybox 위에 나타나야 함.

- [ ] **Step 1: `prepareUniformBuffers()` 작성**

```cpp
void VgeExample::prepareUniformBuffers() {
  uniformBuffers.resize(MAX_CONCURRENT_FRAMES);
  for (auto& ub : uniformBuffers) {
    ub.globals = std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(GlobalsUbo), 1,
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
    ub.bubbleParams = std::make_unique<vgeu::VgeuBuffer>(
        globalAllocator->getAllocator(), sizeof(BubbleParamsUbo), 1,
        vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT);
  }
}
```

- [ ] **Step 2: `setupDescriptors()` 작성 — 4개 descriptor set layout**

```cpp
void VgeExample::setupDescriptors() {
  // set=0 Globals UBO
  globalsSetLayout = vk::raii::DescriptorSetLayout(
      device,
      vk::DescriptorSetLayoutCreateInfo(
          {}, vk::DescriptorSetLayoutBinding(
                  0, vk::DescriptorType::eUniformBuffer, 1,
                  vk::ShaderStageFlagBits::eVertex |
                      vk::ShaderStageFlagBits::eFragment)));

  // set=1 BubbleParams UBO
  bubbleParamsSetLayout = vk::raii::DescriptorSetLayout(
      device,
      vk::DescriptorSetLayoutCreateInfo(
          {}, vk::DescriptorSetLayoutBinding(
                  0, vk::DescriptorType::eUniformBuffer, 1,
                  vk::ShaderStageFlagBits::eFragment)));

  // set=2 Height texture
  heightTexSetLayout = vk::raii::DescriptorSetLayout(
      device,
      vk::DescriptorSetLayoutCreateInfo(
          {}, vk::DescriptorSetLayoutBinding(
                  0, vk::DescriptorType::eCombinedImageSampler, 1,
                  vk::ShaderStageFlagBits::eFragment)));

  // set=3 prefilteredCubemap
  envSetLayout = vk::raii::DescriptorSetLayout(
      device,
      vk::DescriptorSetLayoutCreateInfo(
          {}, vk::DescriptorSetLayoutBinding(
                  0, vk::DescriptorType::eCombinedImageSampler, 1,
                  vk::ShaderStageFlagBits::eFragment)));

  // Allocate per-frame UBO descriptor sets
  globalsDescSets.reserve(MAX_CONCURRENT_FRAMES);
  bubbleParamsDescSets.reserve(MAX_CONCURRENT_FRAMES);
  envDescSets.reserve(MAX_CONCURRENT_FRAMES);
  for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
    // globals
    globalsDescSets.push_back(
        std::move(vk::raii::DescriptorSets(
                      device, vk::DescriptorSetAllocateInfo(
                                  *descriptorPool, *globalsSetLayout))
                      .front()));
    vk::DescriptorBufferInfo globalsBI(uniformBuffers[i].globals->getBuffer(),
                                       0, sizeof(GlobalsUbo));
    device.updateDescriptorSets(
        vk::WriteDescriptorSet(*globalsDescSets[i], 0, 0,
                               vk::DescriptorType::eUniformBuffer, {},
                               globalsBI),
        nullptr);

    // bubbleParams
    bubbleParamsDescSets.push_back(
        std::move(vk::raii::DescriptorSets(
                      device, vk::DescriptorSetAllocateInfo(
                                  *descriptorPool, *bubbleParamsSetLayout))
                      .front()));
    vk::DescriptorBufferInfo paramsBI(
        uniformBuffers[i].bubbleParams->getBuffer(), 0, sizeof(BubbleParamsUbo));
    device.updateDescriptorSets(
        vk::WriteDescriptorSet(*bubbleParamsDescSets[i], 0, 0,
                               vk::DescriptorType::eUniformBuffer, {},
                               paramsBI),
        nullptr);

    // env
    envDescSets.push_back(
        std::move(vk::raii::DescriptorSets(
                      device, vk::DescriptorSetAllocateInfo(
                                  *descriptorPool, *envSetLayout))
                      .front()));
    vk::DescriptorImageInfo envInfo(*iblBaker->iblSampler(),
                                    iblBaker->prefilteredMap().getImageView(),
                                    vk::ImageLayout::eShaderReadOnlyOptimal);
    device.updateDescriptorSets(
        vk::WriteDescriptorSet(*envDescSets[i], 0, 0,
                               vk::DescriptorType::eCombinedImageSampler,
                               envInfo),
        nullptr);
  }

  // Height texture (single set, model owns the texture; allocate once)
  heightTexDescSet =
      std::move(vk::raii::DescriptorSets(
                    device, vk::DescriptorSetAllocateInfo(
                                *descriptorPool, *heightTexSetLayout))
                    .front());
  // 모델의 height texture에 binding — vgeu::glTF::Model API에서 가져오기
  // (PBR 코드에서 pirateGoldHeightTexture 사용 패턴 grep으로 확인)
  // 여기선 임시로 model의 첫 albedo texture를 binding (Task 22에서 height로 교체)
  // 또는 모델 로드 시 별도 vgeu::Texture2D로 height만 따로 로드해 그걸 사용
}
```

`pirate_gold` 모델의 height texture를 어떻게 노출하는지는 PBR 코드 참고:

```bash
rtk grep -n 'pirateGoldHeight\|height' src/examples/pbr/pbr.cpp | head -20
```

PBR이 `vgeu::Texture2D` 별도 로드한다면 같은 패턴으로 `loadAssets()`에서 height texture만 별도로:

```cpp
heightTexture = std::make_unique<vgeu::Texture2D>(
    getAssetsPath() + "/models/pirate_gold/textures/<height_filename>.png",
    device, globalAllocator->getAllocator(), queue, commandPool, true);
```

`heightTexture` 멤버를 `.hpp`에 추가:

```cpp
std::unique_ptr<vgeu::Texture2D> heightTexture;
```

descriptor binding은 `*heightTexture->sampler`, `heightTexture->descriptorInfo.imageView` 사용.

- [ ] **Step 3: `preparePipelines()` 작성**

```cpp
void VgeExample::preparePipelines() {
  std::array<vk::DescriptorSetLayout, 4> setLayouts{
      *globalsSetLayout, *bubbleParamsSetLayout, *heightTexSetLayout,
      *envSetLayout};
  bubblePipelineLayout = vk::raii::PipelineLayout(
      device, vk::PipelineLayoutCreateInfo({}, setLayouts));

  auto vertCode = vgeu::readFile(getShadersPath() + "/soap_bubble/bubble.vert.spv");
  auto fragCode = vgeu::readFile(getShadersPath() + "/soap_bubble/bubble.frag.spv");
  vk::raii::ShaderModule vertSM(
      device, vk::ShaderModuleCreateInfo({}, vertCode.size(),
                                         (const uint32_t*)vertCode.data()));
  vk::raii::ShaderModule fragSM(
      device, vk::ShaderModuleCreateInfo({}, fragCode.size(),
                                         (const uint32_t*)fragCode.data()));

  std::array<vk::PipelineShaderStageCreateInfo, 2> stages{
      vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eVertex,
                                        *vertSM, "main"),
      vk::PipelineShaderStageCreateInfo({}, vk::ShaderStageFlagBits::eFragment,
                                        *fragSM, "main"),
  };

  // Vertex input: vgeu_gltf 표준 (PBR mrt.vert와 동일 — 7 attributes)
  // PBR의 setupVertexInput 헬퍼 또는 직접 작성. 여기선 간단히 직접:
  auto bindingDesc = vgeu::glTF::Vertex::getBindingDescriptions();
  auto attrDesc = vgeu::glTF::Vertex::getAttributeDescriptions();
  vk::PipelineVertexInputStateCreateInfo vertexInputCI({}, bindingDesc, attrDesc);

  vk::PipelineInputAssemblyStateCreateInfo iaCI(
      {}, vk::PrimitiveTopology::eTriangleList);

  vk::PipelineViewportStateCreateInfo vpCI({}, 1, {}, 1, {});

  // Front-face only (back face culling), counter-clockwise (gltf 표준)
  vk::PipelineRasterizationStateCreateInfo rsCI(
      {}, false, false, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack,
      vk::FrontFace::eCounterClockwise, false, 0.f, 0.f, 0.f, 1.f);

  vk::PipelineMultisampleStateCreateInfo msCI({}, vk::SampleCountFlagBits::e1);

  // Depth test on, depth write OFF (transparent surfaces)
  vk::PipelineDepthStencilStateCreateInfo dsCI(
      {}, true /*depthTest*/, false /*depthWrite*/, vk::CompareOp::eLess);

  // Alpha blend: srcAlpha * src + (1-srcAlpha) * dst
  vk::PipelineColorBlendAttachmentState cbAtt(
      true, vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha,
      vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero,
      vk::BlendOp::eAdd,
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
  vk::PipelineColorBlendStateCreateInfo cbCI({}, false, vk::LogicOp::eClear,
                                             cbAtt);

  std::array<vk::DynamicState, 2> dynStates{vk::DynamicState::eViewport,
                                            vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dynCI({}, dynStates);

  vk::GraphicsPipelineCreateInfo pipelineCI(
      {}, stages, &vertexInputCI, &iaCI, nullptr, &vpCI, &rsCI, &msCI, &dsCI,
      &cbCI, &dynCI, *bubblePipelineLayout, *renderPass);
  bubblePipeline = vk::raii::Pipeline(device, pipelineCache, pipelineCI);
}
```

`vgeu::glTF::Vertex::getBindingDescriptions()` / `getAttributeDescriptions()`가 vgeu_gltf에 있는지 확인:

```bash
rtk grep -n 'getBindingDescriptions\|getAttributeDescriptions' src/base/vgeu_gltf.hpp
```

없다면 PBR mrt.vert pipeline 셋업의 vertex input 코드를 그대로 복사.

- [ ] **Step 4: `buildCommandBuffers()` 갱신 — bubble draw 추가**

skybox draw 후, model draw 추가:

```cpp
// bubble pipeline
cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *bubblePipeline);
std::array<vk::DescriptorSet, 4> descSets{
    *globalsDescSets[i], *bubbleParamsDescSets[i], *heightTexDescSet,
    *envDescSets[i]};
cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                       *bubblePipelineLayout, 0, descSets, nullptr);
bubbleModel->draw(cmd);
```

`bubbleModel->draw(cmd)` 시그니처는 vgeu_gltf 확인:

```bash
rtk grep -n 'void draw\|::draw' src/base/vgeu_gltf.hpp
```

- [ ] **Step 5: `updateGlobalsUbo()`/`updateBubbleParamsUbo()` 작성 + render() 갱신**

```cpp
void VgeExample::updateGlobalsUbo() {
  globalsUbo.view = camera.getView();
  globalsUbo.projection = camera.getProjection();
  globalsUbo.viewPos = glm::vec4(camera.getPosition(), 1.0);
  std::memcpy(uniformBuffers[currentFrameIndex].globals->getMappedData(),
              &globalsUbo, sizeof(GlobalsUbo));
}

void VgeExample::updateBubbleParamsUbo() {
  bubbleParamsUbo.thicknessMin = opts.thicknessMin;
  bubbleParamsUbo.thicknessMax = opts.thicknessMax;
  bubbleParamsUbo.n1 = opts.n1;
  bubbleParamsUbo.n2 = opts.n2;
  bubbleParamsUbo.n3 = opts.n3;
  bubbleParamsUbo.spectralSamples = opts.spectralSamples;
  bubbleParamsUbo.thicknessMode = opts.thicknessMode;
  bubbleParamsUbo.gravityStrength = opts.gravityStrength;
  bubbleParamsUbo.noiseScale = opts.noiseScale;
  bubbleParamsUbo.useAnimation = opts.useAnimation ? 1 : 0;
  bubbleParamsUbo.driftSpeed = opts.driftSpeed;
  bubbleParamsUbo.roughness = opts.roughness;
  bubbleParamsUbo.alphaScale = opts.alphaScale;
  bubbleParamsUbo.iblExposure = opts.iblExposure;
  bubbleParamsUbo.iblGamma = opts.iblGamma;
  bubbleParamsUbo.time = timer;
  bubbleParamsUbo.showThicknessHeatmap = opts.showThicknessHeatmap ? 1 : 0;
  bubbleParamsUbo.showFresnelOnly = opts.showFresnelOnly ? 1 : 0;
  std::memcpy(uniformBuffers[currentFrameIndex].bubbleParams->getMappedData(),
              &bubbleParamsUbo, sizeof(BubbleParamsUbo));
}

void VgeExample::render() {
  if (!prepared) return;
  updateGlobalsUbo();
  updateBubbleParamsUbo();
  draw();
}
```

- [ ] **Step 6: 빌드 + 실행**

```bash
./mingwBuild.bat Debug
./build/soap_bubble.exe
```

Expected: skybox 위에 빨간 pirate gold sphere가 보임. 마우스/WASD로 회전·이동 가능. validation 클린.

- [ ] **Step 7: clang-format + commit**

```bash
rtk clang-format -i src/examples/soap_bubble/soap_bubble.hpp src/examples/soap_bubble/soap_bubble.cpp
rtk git add src/examples/soap_bubble
rtk git commit -m "feat(soap_bubble): set up bubble pipeline with red placeholder"
```

---

## Task 18: bubble.frag — env reflection (mirror sphere, 박막 없음)

**Files:**
- Modify: `shaders/soap_bubble/bubble.frag`

박막 수학 들어가기 전에 env IBL 반사부터 동작하는지 확인. 결과: 거울 sphere.

- [ ] **Step 1: `bubble.frag` 갱신**

```glsl
#version 450

layout(set = 0, binding = 0) uniform Globals {
  mat4 view;
  mat4 projection;
  vec4 viewPos;
} globals;

layout(set = 1, binding = 0) uniform BubbleParams {
  float thicknessMin;
  float thicknessMax;
  float n1;
  float n2;
  float n3;
  int spectralSamples;
  int thicknessMode;
  float gravityStrength;
  float noiseScale;
  int useAnimation;
  float driftSpeed;
  float roughness;
  float alphaScale;
  float iblExposure;
  float iblGamma;
  float time;
  int showThicknessHeatmap;
  int showFresnelOnly;
  float _pad0;
  float _pad1;
} params;

layout(set = 2, binding = 0) uniform sampler2D heightTex;
layout(set = 3, binding = 0) uniform samplerCube prefilteredCubemap;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec4 outColor;

void main() {
  vec3 N = normalize(inWorldNormal);
  vec3 V = normalize(globals.viewPos.xyz - inWorldPos);
  vec3 R = reflect(-V, N);

  // env reflection (LOD 0 = mirror)
  vec3 envColor = textureLod(prefilteredCubemap, R, 0.0).rgb;

  outColor = vec4(envColor * params.iblExposure, 1.0);
}
```

- [ ] **Step 2: 빌드 + 실행**

```bash
./mingwBuild.bat Debug
./build/soap_bubble.exe
```

Expected: 환경(skybox)이 반사된 거울 sphere가 보임. 카메라 회전 시 반사가 자연스럽게 따라옴. validation 클린.

- [ ] **Step 3: commit**

```bash
rtk git add shaders/soap_bubble/bubble.frag
rtk git commit -m "feat(soap_bubble): bubble frag samples env IBL (mirror)"
```

---

## Task 19: Wyman CMF + Schlick Fresnel GLSL 헬퍼

**Files:**
- Modify: `shaders/soap_bubble/bubble.frag`

박막 수식의 빌딩 블록 두 개. 다음 task에서 사용.

- [ ] **Step 1: `bubble.frag`에 헬퍼 함수 추가** (main 위쪽에)

```glsl
const float PI = 3.14159265358979323846;

// Wyman 2013 analytical fit for CIE 1931 color matching functions.
// Input: lambda in nm. Output: (x_bar, y_bar, z_bar).
vec3 wymanCMF(float lambda) {
  float x =
      0.398 * exp(-1250.0 * pow(log((lambda + 570.1) / 1014.0), 2.0)) +
      1.132 * exp(-234.0 * pow(log((1338.0 - lambda) / 743.5), 2.0));
  float y =
      1.011 * exp(-0.5 * pow((lambda - 556.1) / 46.14, 2.0));
  float z =
      2.060 * exp(-32.0 * pow(log((lambda - 265.8) / 180.4), 2.0));
  return vec3(x, y, z);
}

// Schlick Fresnel for unpolarized light at boundary n1 → n2.
float fresnelSchlick(float cosTheta, float nFrom, float nTo) {
  float f0 = (nFrom - nTo) / (nFrom + nTo);
  f0 = f0 * f0;
  return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

// XYZ → sRGB (D65, linear). Caller applies gamma after.
vec3 xyzToSrgb(vec3 xyz) {
  mat3 M = mat3(
       3.2406, -0.9689,  0.0557,
      -1.5372,  1.8758, -0.2040,
      -0.4986,  0.0415,  1.0570);
  return max(M * xyz, vec3(0.0));
}
```

- [ ] **Step 2: 빌드 sanity (셰이더 컴파일만)**

```bash
./mingwBuild.bat Debug
```

Expected: 셰이더 컴파일 성공. main이 헬퍼를 아직 사용 안 하므로 결과는 거울 sphere 그대로.

- [ ] **Step 3: 실행 확인 — 변화 없음**

```bash
./build/soap_bubble.exe
```

- [ ] **Step 4: commit**

```bash
rtk git add shaders/soap_bubble/bubble.frag
rtk git commit -m "feat(soap_bubble): add Wyman CMF + Schlick Fresnel helpers"
```

---

## Task 20: `thinFilmReflectance()` 분광 적분 + main에서 사용

**Files:**
- Modify: `shaders/soap_bubble/bubble.frag`

박막 핵심. 시야각 의존 무지개색이 나타나야 함.

- [ ] **Step 1: `thinFilmReflectance()` 추가**

main 위쪽에 (헬퍼 다음):

```glsl
// Compute per-pixel thin-film reflectance integrated over visible spectrum,
// returned as linear sRGB. d = thickness in nm, cosTheta1 = view-N dot.
vec3 thinFilmReflectance(float d, float cosTheta1) {
  // Snell
  float sinTheta1Sq = 1.0 - cosTheta1 * cosTheta1;
  float sinTheta2 = (params.n1 / params.n2) * sqrt(max(sinTheta1Sq, 0.0));
  if (sinTheta2 >= 1.0) return vec3(1.0);  // total internal reflection
  float cosTheta2 = sqrt(1.0 - sinTheta2 * sinTheta2);

  // Boundary Fresnels
  float r1 = fresnelSchlick(cosTheta1, params.n1, params.n2);
  float r2 = fresnelSchlick(cosTheta2, params.n2, params.n3);

  // Phase shifts
  float phi1 = (params.n1 < params.n2) ? PI : 0.0;
  float phi2 = (params.n2 < params.n3) ? PI : 0.0;
  float deltaPhi = phi1 - phi2;

  vec3 XYZ = vec3(0.0);
  float yWeight = 0.0;  // for normalization
  int N = clamp(params.spectralSamples, 4, 64);
  for (int i = 0; i < N; ++i) {
    float t = (float(i) + 0.5) / float(N);
    float lambda = mix(380.0, 780.0, t);  // nm
    float opdPhase = (4.0 * PI * params.n2 * d * cosTheta2) / lambda;
    float R = r1 * r1 + r2 * r2 +
              2.0 * r1 * r2 * cos(opdPhase + deltaPhi);
    vec3 cmf = wymanCMF(lambda);
    XYZ += R * cmf;
    yWeight += cmf.y;
  }
  XYZ /= max(yWeight, 1e-6);
  return xyzToSrgb(XYZ);
}
```

- [ ] **Step 2: `main()` 갱신 — env에 thinFilm 곱하기**

```glsl
void main() {
  vec3 N = normalize(inWorldNormal);
  vec3 V = normalize(globals.viewPos.xyz - inWorldPos);
  float cosTheta1 = max(dot(N, V), 0.001);
  vec3 R = reflect(-V, N);

  // 임시: 두께를 균일 (thicknessMax). Task 22에서 thicknessAt() 호출로 교체.
  float d = params.thicknessMax;

  vec3 thinFilm = thinFilmReflectance(d, cosTheta1);

  // env reflection (roughness slider not yet wired — LOD 0)
  vec3 envColor = textureLod(prefilteredCubemap, R, 0.0).rgb;

  vec3 color = thinFilm * envColor * params.iblExposure;

  outColor = vec4(color, 1.0);  // alpha는 다음 task에서
}
```

- [ ] **Step 3: 빌드 + 실행**

```bash
./mingwBuild.bat Debug
./build/soap_bubble.exe
```

Expected: 시야각에 따라 색이 바뀌는 sphere — 정면은 한 색, 가장자리는 다른 색. 무지개의 일부가 보여야 함. validation 클린.

- [ ] **Step 4: 셰이더 sanity**

`spectralSamples` 기본값 16에서 부드러운 색이 나와야 함. 만약 너무 어두우면 `iblExposure` 슬라이더(아직 ImGui 안 만들었지만 기본값 4.5)가 곱해져야 함. 안 곱해지면 default 값을 코드 추적해서 확인.

- [ ] **Step 5: commit**

```bash
rtk git add shaders/soap_bubble/bubble.frag
rtk git commit -m "feat(soap_bubble): integrate thin-film reflectance over spectrum"
```

---

## Task 21: Fresnel-driven 알파 + alphaScale + roughness LOD

**Files:**
- Modify: `shaders/soap_bubble/bubble.frag`

알파를 Fresnel 기반으로. roughness 슬라이더로 LOD 변환.

- [ ] **Step 1: `main()` 갱신**

```glsl
void main() {
  vec3 N = normalize(inWorldNormal);
  vec3 V = normalize(globals.viewPos.xyz - inWorldPos);
  float cosTheta1 = max(dot(N, V), 0.001);
  vec3 R = reflect(-V, N);

  float d = params.thicknessMax;  // Task 22에서 교체

  vec3 thinFilm = thinFilmReflectance(d, cosTheta1);

  // roughness → LOD on prefilteredCubemap
  float maxLod = float(textureQueryLevels(prefilteredCubemap) - 1);
  float lod = clamp(params.roughness, 0.0, 1.0) * maxLod;
  vec3 envColor = textureLod(prefilteredCubemap, R, lod).rgb;

  vec3 color = thinFilm * envColor * params.iblExposure;

  // Fresnel alpha at outer boundary
  float fresnel = fresnelSchlick(cosTheta1, params.n1, params.n2);
  float alpha = clamp(fresnel * params.alphaScale, 0.0, 1.0);

  outColor = vec4(color, alpha);
}
```

- [ ] **Step 2: 빌드 + 실행**

```bash
./mingwBuild.bat Debug
./build/soap_bubble.exe
```

Expected: 가운데가 거의 투명하고 (skybox 통과해 보임), 가장자리에 무지개색이 강하게 나타나는 비눗방울 룩. validation 클린.

- [ ] **Step 3: commit**

```bash
rtk git add shaders/soap_bubble/bubble.frag
rtk git commit -m "feat(soap_bubble): apply Fresnel-driven alpha and roughness LOD"
```

---

## Task 22: `thicknessAt()` Texture 모드 + height texture binding

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.{hpp,cpp}`
- Modify: `shaders/soap_bubble/bubble.frag`

Height 텍스처를 model에서 분리해 별도 Texture2D로 로드, set=2에 binding. 셰이더에서 sample.

- [ ] **Step 1: `soap_bubble.hpp`에 height texture 멤버 추가**

```cpp
std::unique_ptr<vgeu::Texture2D> heightTexture;
```

- [ ] **Step 2: `loadAssets()`에 height texture 로드 추가**

pirate gold height texture 파일명 확인:

```bash
rtk ls assets/models/pirate_gold/textures
```

찾은 파일명을 사용 (예: `pirate_gold_height.png`):

```cpp
heightTexture = std::make_unique<vgeu::Texture2D>(
    getAssetsPath() + "/models/pirate_gold/textures/<height_filename>",
    device, globalAllocator->getAllocator(), queue, commandPool, true);
```

- [ ] **Step 3: `setupDescriptors()`의 heightTex 바인딩 갱신**

Task 17의 placeholder 부분을 다음으로:

```cpp
vk::DescriptorImageInfo heightInfo(*heightTexture->sampler,
                                   heightTexture->descriptorInfo.imageView,
                                   vk::ImageLayout::eShaderReadOnlyOptimal);
device.updateDescriptorSets(
    vk::WriteDescriptorSet(*heightTexDescSet, 0, 0,
                           vk::DescriptorType::eCombinedImageSampler,
                           heightInfo),
    nullptr);
```

- [ ] **Step 4: `bubble.frag`에 `thicknessAt()` 추가 — Texture 모드만**

main 위쪽에:

```glsl
float thicknessAt(vec2 uv, vec3 worldPos, vec3 normal) {
  vec2 sampleUV = uv;
  if (params.useAnimation != 0) {
    sampleUV += vec2(params.driftSpeed * params.time, 0.0);
  }
  float h;
  if (params.thicknessMode == 0) {
    h = texture(heightTex, sampleUV).r;
  } else {
    // Procedural — Task 23
    h = 0.5;
  }
  return mix(params.thicknessMin, params.thicknessMax, h);
}
```

main의 `float d = params.thicknessMax;` → `float d = thicknessAt(inUV, inWorldPos, N);`

- [ ] **Step 5: 빌드 + 실행**

```bash
./mingwBuild.bat Debug
./build/soap_bubble.exe
```

Expected: pirate gold의 height 텍스처 패턴에 따라 두께가 변하면서 색 무늬가 표면에 나타남. 카메라 회전 시 색이 변함. validation 클린.

- [ ] **Step 6: commit**

```bash
rtk clang-format -i src/examples/soap_bubble/soap_bubble.hpp src/examples/soap_bubble/soap_bubble.cpp
rtk git add src/examples/soap_bubble shaders/soap_bubble
rtk git commit -m "feat(soap_bubble): wire height texture as thickness source"
```

---

## Task 23: `thicknessAt()` Procedural 모드 (gravity + value noise)

**Files:**
- Modify: `shaders/soap_bubble/bubble.frag`

ImGui radio로 mode 1 선택 시 분석적 두께 분포.

- [ ] **Step 1: value noise GLSL 추가** (헬퍼 영역)

```glsl
// Simple 3D value noise via hashing.
float hash3(vec3 p) {
  p = fract(p * vec3(443.8975, 397.2973, 491.1871));
  p += dot(p, p.yzx + 19.19);
  return fract((p.x + p.y) * p.z);
}

float valueNoise3D(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  vec3 u = f * f * (3.0 - 2.0 * f);  // smoothstep

  float n000 = hash3(i + vec3(0,0,0));
  float n100 = hash3(i + vec3(1,0,0));
  float n010 = hash3(i + vec3(0,1,0));
  float n110 = hash3(i + vec3(1,1,0));
  float n001 = hash3(i + vec3(0,0,1));
  float n101 = hash3(i + vec3(1,0,1));
  float n011 = hash3(i + vec3(0,1,1));
  float n111 = hash3(i + vec3(1,1,1));

  float nx00 = mix(n000, n100, u.x);
  float nx10 = mix(n010, n110, u.x);
  float nx01 = mix(n001, n101, u.x);
  float nx11 = mix(n011, n111, u.x);
  float nxy0 = mix(nx00, nx10, u.y);
  float nxy1 = mix(nx01, nx11, u.y);
  return mix(nxy0, nxy1, u.z);
}
```

- [ ] **Step 2: `thicknessAt()`의 procedural 분기 채우기**

```glsl
float thicknessAt(vec2 uv, vec3 worldPos, vec3 normal) {
  float h;
  if (params.thicknessMode == 0) {
    vec2 sampleUV = uv;
    if (params.useAnimation != 0) {
      sampleUV += vec2(params.driftSpeed * params.time, 0.0);
    }
    h = texture(heightTex, sampleUV).r;
  } else {
    // Procedural: gravity gradient + noise
    float gravity = clamp(0.5 + worldPos.y * params.gravityStrength, 0.0, 1.0);
    vec3 noisePos = worldPos * params.noiseScale;
    if (params.useAnimation != 0) {
      noisePos += vec3(0.0, 0.0, params.time * params.driftSpeed);
    }
    float n = valueNoise3D(noisePos);
    h = mix(gravity, n, 0.5);
  }
  return mix(params.thicknessMin, params.thicknessMax, h);
}
```

- [ ] **Step 3: 빌드 + sanity**

```bash
./mingwBuild.bat Debug
./build/soap_bubble.exe
```

Expected: ImGui로 thicknessMode 변경 못 하지만 (Task 25에서 추가), 코드 기본값이 0(Texture)이라 결과 동일. 빌드 통과 확인. CLI로 mode 변경하려면 Task 26 필요.

임시로 `Options::thicknessMode = 1`로 코드 수정 후 재빌드해서 procedural 모드가 동작하는지 시각 확인 후 원복:

```cpp
// Options 구조체에서 임시:
int32_t thicknessMode = 1;  // 임시 1, 검증 후 0으로
```

- [ ] **Step 4: commit**

```bash
rtk git add shaders/soap_bubble/bubble.frag
rtk git commit -m "feat(soap_bubble): add procedural thickness mode (gravity + value noise)"
```

---

## Task 24: time uniform + animation 검증

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.cpp`

이미 Task 17에서 `bubbleParamsUbo.time = timer;`로 wiring 됐음. 이 task는 검증 전용.

- [ ] **Step 1: `Options::useAnimation = true`로 임시 변경 후 빌드**

```cpp
bool useAnimation = true;  // 임시
```

- [ ] **Step 2: 실행**

```bash
./mingwBuild.bat Debug
./build/soap_bubble.exe
```

Expected: 두께 분포가 천천히 흐름 (Texture 모드: UV 스크롤, Procedural 모드: noise z축 시간 변화). 카메라 정지해도 색이 변함.

- [ ] **Step 3: 임시 변경 원복**

```cpp
bool useAnimation = false;  // 기본값으로
```

- [ ] **Step 4: commit (없음 — 임시 변경 원복했고 코드 변화 없음)**

이 task는 검증만이라 commit 없음.

---

## Task 25: ImGui 6 그룹 패널

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.cpp`

`onUpdateUIOverlay()` 본체 작성. PBR의 패턴 참고:

```bash
rtk grep -n 'onUpdateUIOverlay' src/examples/pbr/pbr.cpp
```

- [ ] **Step 1: `onUpdateUIOverlay()` 작성**

```cpp
void VgeExample::onUpdateUIOverlay() {
  if (uiOverlay->header("Thin Film")) {
    uiOverlay->sliderFloat("thicknessMin (nm)", &opts.thicknessMin, 0.f, 2000.f);
    uiOverlay->sliderFloat("thicknessMax (nm)", &opts.thicknessMax, 0.f, 2000.f);
    uiOverlay->sliderFloat("n1 (outside)", &opts.n1, 1.0f, 2.5f);
    uiOverlay->sliderFloat("n2 (film)", &opts.n2, 1.0f, 2.5f);
    uiOverlay->sliderFloat("n3 (inside)", &opts.n3, 1.0f, 2.5f);

    static const char* sampleOpts[] = {"8", "16", "32", "64"};
    static int currentIdx = 1;
    if (uiOverlay->comboBox("spectralSamples", &currentIdx, sampleOpts, 4)) {
      const int values[] = {8, 16, 32, 64};
      opts.spectralSamples = values[currentIdx];
    }
  }

  if (uiOverlay->header("Thickness Source")) {
    int mode = opts.thicknessMode;
    if (uiOverlay->radioButton("Texture", mode == 0)) opts.thicknessMode = 0;
    if (uiOverlay->radioButton("Procedural", mode == 1)) opts.thicknessMode = 1;
    if (opts.thicknessMode == 1) {
      uiOverlay->sliderFloat("gravityStrength", &opts.gravityStrength, 0.f, 5.f);
      uiOverlay->sliderFloat("noiseScale", &opts.noiseScale, 0.1f, 10.f);
    }
  }

  if (uiOverlay->header("Animation")) {
    uiOverlay->checkBox("useAnimation", &opts.useAnimation);
    if (opts.useAnimation) {
      uiOverlay->sliderFloat("driftSpeed", &opts.driftSpeed, 0.f, 2.f);
    }
  }

  if (uiOverlay->header("Surface & Blending")) {
    uiOverlay->sliderFloat("roughness", &opts.roughness, 0.f, 1.f);
    uiOverlay->sliderFloat("alphaScale", &opts.alphaScale, 0.f, 3.f);
  }

  if (uiOverlay->header("IBL / Env")) {
    uiOverlay->sliderFloat("iblExposure", &opts.iblExposure, 0.f, 10.f);
    uiOverlay->sliderFloat("iblGamma", &opts.iblGamma, 1.f, 3.f);
    uiOverlay->sliderFloat("skyboxLod", &opts.skyboxLod, 0.f, 9.f);
    if (uiOverlay->checkBox("useJitter", &opts.useJitter)) {
      device.waitIdle();
      iblConfig.useJitter = opts.useJitter;
      iblBaker->rebakeFiltering(iblConfig);
      // Re-bind env descriptor (prefilteredMap이 새로 만들어졌음)
      for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; ++i) {
        vk::DescriptorImageInfo envInfo(
            *iblBaker->iblSampler(), iblBaker->prefilteredMap().getImageView(),
            vk::ImageLayout::eShaderReadOnlyOptimal);
        device.updateDescriptorSets(
            vk::WriteDescriptorSet(*envDescSets[i], 0, 0,
                                   vk::DescriptorType::eCombinedImageSampler,
                                   envInfo),
            nullptr);
      }
    }
  }

  if (uiOverlay->header("Debug")) {
    glm::vec3 cp = camera.getPosition();
    uiOverlay->text("camera: %.2f, %.2f, %.2f", cp.x, cp.y, cp.z);
    uiOverlay->checkBox("showThicknessHeatmap", &opts.showThicknessHeatmap);
    uiOverlay->checkBox("showFresnelOnly", &opts.showFresnelOnly);
  }
}
```

`uiOverlay->header/sliderFloat/checkBox/comboBox/radioButton/text` API의 정확한 시그니처는 `src/base/vgeu_ui_overlay.hpp` 확인:

```bash
rtk grep -n 'bool header\|bool sliderFloat\|bool checkBox\|bool comboBox\|bool radioButton\|void text' src/base/vgeu_ui_overlay.hpp
```

API가 다르면 PBR `onUpdateUIOverlay`를 그대로 따라 갱신.

- [ ] **Step 2: 빌드 + 실행**

```bash
./mingwBuild.bat Debug
./build/soap_bubble.exe
```

Expected: ImGui 패널에서 모든 슬라이더가 즉시 반영됨. `useJitter` 토글 시 IBL 재바이크 (잠깐 멈춤). validation 클린.

- [ ] **Step 3: spec sanity 표 일부 검증**

| 입력                             | 기대         | 결과?    |
| -------------------------------- | ------------ | -------- |
| `thicknessMin=0, thicknessMax=0` | 검정         | [ ] 확인 |
| `n1=n2=n3=1.0`                   | 완전 투명    | [ ] 확인 |
| `n2=2.4`                         | 매우 진한 색 | [ ] 확인 |
| `alphaScale=0`                   | 안 보임      | [ ] 확인 |

- [ ] **Step 4: clang-format + commit**

```bash
rtk clang-format -i src/examples/soap_bubble/soap_bubble.cpp
rtk git add src/examples/soap_bubble/soap_bubble.cpp
rtk git commit -m "feat(soap_bubble): add ImGui parameter panel (6 groups)"
```

---

## Task 26: CLI11 옵션

**Files:**
- Modify: `src/examples/soap_bubble/soap_bubble.cpp`

PBR의 `setupCommandLineParser` 패턴.

- [ ] **Step 1: `setupCommandLineParser()` 본체 작성**

```cpp
void VgeExample::setupCommandLineParser(CLI::App& app) {
  VgeBase::setupCommandLineParser(app);
  app.add_option("--thicknessMin", opts.thicknessMin);
  app.add_option("--thicknessMax", opts.thicknessMax);
  app.add_option("--n1", opts.n1);
  app.add_option("--n2", opts.n2);
  app.add_option("--n3", opts.n3);
  app.add_option("--spectralSamples", opts.spectralSamples);
  app.add_option("--thicknessMode", opts.thicknessMode);
  app.add_option("--gravityStrength", opts.gravityStrength);
  app.add_option("--noiseScale", opts.noiseScale);
  app.add_option("--useAnimation", opts.useAnimation);
  app.add_option("--driftSpeed", opts.driftSpeed);
  app.add_option("--roughness", opts.roughness);
  app.add_option("--alphaScale", opts.alphaScale);
  app.add_option("--iblExposure", opts.iblExposure);
  app.add_option("--iblGamma", opts.iblGamma);
  app.add_option("--useJitter", opts.useJitter);
  app.add_option("--skyboxLod", opts.skyboxLod);
}
```

- [ ] **Step 2: 빌드 + CLI 동작 확인**

```bash
./mingwBuild.bat Debug
./build/soap_bubble.exe --n2 1.45 --thicknessMin 300 --thicknessMax 600 --spectralSamples 32
```

Expected: 시작값이 CLI로 덮어써짐. ImGui에서 해당 슬라이더 위치가 CLI 값.

- [ ] **Step 3: clang-format + commit**

```bash
rtk clang-format -i src/examples/soap_bubble/soap_bubble.cpp
rtk git add src/examples/soap_bubble/soap_bubble.cpp
rtk git commit -m "feat(soap_bubble): expose all params via CLI11"
```

---

## Task 27: Debug 시각화 (heatmap, fresnel-only)

**Files:**
- Modify: `shaders/soap_bubble/bubble.frag`

`showThicknessHeatmap`/`showFresnelOnly` 토글로 alternate output.

- [ ] **Step 1: `bubble.frag`의 main 분기 추가**

```glsl
void main() {
  vec3 N = normalize(inWorldNormal);
  vec3 V = normalize(globals.viewPos.xyz - inWorldPos);
  float cosTheta1 = max(dot(N, V), 0.001);
  vec3 R = reflect(-V, N);

  float d = thicknessAt(inUV, inWorldPos, N);

  // Debug: thickness heatmap (viridis approximation)
  if (params.showThicknessHeatmap != 0) {
    float t = clamp((d - params.thicknessMin) /
                        max(params.thicknessMax - params.thicknessMin, 1.0),
                    0.0, 1.0);
    // Simple viridis-like ramp
    vec3 c = vec3(0.267 + 0.5 * t,
                  0.005 + 0.9 * t,
                  0.329 + 0.4 * sin(t * 3.14159));
    outColor = vec4(c, 1.0);
    return;
  }

  // Debug: Fresnel grayscale
  if (params.showFresnelOnly != 0) {
    float f = fresnelSchlick(cosTheta1, params.n1, params.n2);
    outColor = vec4(f, f, f, 1.0);
    return;
  }

  // Normal path
  vec3 thinFilm = thinFilmReflectance(d, cosTheta1);
  float maxLod = float(textureQueryLevels(prefilteredCubemap) - 1);
  float lod = clamp(params.roughness, 0.0, 1.0) * maxLod;
  vec3 envColor = textureLod(prefilteredCubemap, R, lod).rgb;
  vec3 color = thinFilm * envColor * params.iblExposure;

  // Optional gamma (linear → display)
  color = pow(max(color, vec3(0.0)), vec3(1.0 / params.iblGamma));

  float fresnel = fresnelSchlick(cosTheta1, params.n1, params.n2);
  float alpha = clamp(fresnel * params.alphaScale, 0.0, 1.0);

  outColor = vec4(color, alpha);
}
```

- [ ] **Step 2: 빌드 + 실행**

```bash
./mingwBuild.bat Debug
./build/soap_bubble.exe
```

Expected:
- 기본: 비눗방울 정상
- showThicknessHeatmap on: 두께 분포가 컬러로 (파란~노란)
- showFresnelOnly on: 시야각 따라 회색조 (정면 0, 가장자리 1)

- [ ] **Step 3: commit**

```bash
rtk git add shaders/soap_bubble/bubble.frag
rtk git commit -m "feat(soap_bubble): add debug visualizations (heatmap, fresnel)"
```

---

## Task 28: 최종 sanity 표 + clang-format pass + 마무리

**Files:** 검증 + 정리만.

- [ ] **Step 1: spec의 sanity 표 모두 검증 (`docs/superpowers/specs/2026-04-27-soap-bubble-shader-design.md` §5.2)**

```bash
./build/soap_bubble.exe
```

ImGui 슬라이더로 다음을 한 번씩 돌려보고 기대 결과 확인:

| 입력                             | 기대                         | OK? |
| -------------------------------- | ---------------------------- | --- |
| `thicknessMin=0, thicknessMax=0` | 검정 박막                    | [ ] |
| `n1=n2=n3=1.0`                   | 완전 투명, 색 없음           | [ ] |
| `n1=1.0, n2=2.4, n3=1.0`         | 매우 진한 채도               | [ ] |
| `thicknessMin=thicknessMax=550`  | 단일 톤                      | [ ] |
| `spectralSamples` 8 → 64         | 부드러운 그라데이션          | [ ] |
| `alphaScale=0`                   | 안 보임                      | [ ] |
| 카메라 회전                      | 색 시야각 의존 변화          | [ ] |
| Procedural + `gravityStrength=4` | 위쪽 어둡고 아래쪽 색 두꺼움 | [ ] |
| Animation 토글                   | drift 보임 → 끄면 정적       | [ ] |
| `useJitter` 토글                 | 재바이크 발생, IBL 변화      | [ ] |

하나라도 fail → 해당 task 회귀해서 수정.

- [ ] **Step 2: validation layer 클린 확인**

콘솔 출력에 새 VUID 에러 없는지. 기존 PBR의 VUID-02697이 soap_bubble에서도 보이는지 check (보이면 별 task로 별개 이슈, plan 범위 밖).

- [ ] **Step 3: 60fps sanity**

`spectralSamples=64`에서도 frame timer가 16ms 이하인지. 안 되면 spec §5.5에 따라 기본값 16으로 회귀 (현재 default가 이미 16).

- [ ] **Step 4: 모든 .cpp/.hpp에 clang-format**

```bash
rtk clang-format -i src/base/vgeu_ibl.hpp src/base/vgeu_ibl.cpp
rtk clang-format -i src/examples/pbr/pbr.hpp src/examples/pbr/pbr.cpp
rtk clang-format -i src/examples/soap_bubble/soap_bubble.hpp src/examples/soap_bubble/soap_bubble.cpp
```

변경 있으면 별도 commit:

```bash
rtk git status
rtk git add -A
rtk git commit -m "style: clang-format pass on soap_bubble + vgeu_ibl + pbr"
```

변경 없으면 (이미 task별로 적용되었음) 스킵.

- [ ] **Step 5: README 업데이트 (선택)**

`README.md`의 examples 섹션에 soap_bubble 항목 추가. 스크린샷은 사용자가 별도로 첨부.

```markdown
## [soap_bubble](src/examples/soap_bubble)
- Thin-film interference shader with spectral integration (Wyman 2013 CMF)
- Uses vgeu_ibl module (extracted from pbr)
- Reference: https://en.wikipedia.org/wiki/Thin-film_interference
```

선택. 사용자 결정.

- [ ] **Step 6: 최종 commit (이전 단계가 변경 있었을 때만)**

```bash
rtk git log --oneline | head -10
```

전체 commit 흐름 확인:
1. `refactor(shaders): move IBL+skybox shaders to shaders/common/`
2. `feat(base): add vgeu_ibl module (IBLBaker, Skybox)`
3. `refactor(pbr): migrate IBL bake and skybox to vgeu_ibl module`
4. `feat(soap_bubble): scaffold new example skeleton`
5. `feat(soap_bubble): bake IBL and draw skybox`
6. `feat(soap_bubble): load pirate gold model`
7. `feat(soap_bubble): add minimal bubble.vert/.frag`
8. `feat(soap_bubble): set up bubble pipeline with red placeholder`
9. `feat(soap_bubble): bubble frag samples env IBL (mirror)`
10. `feat(soap_bubble): add Wyman CMF + Schlick Fresnel helpers`
11. `feat(soap_bubble): integrate thin-film reflectance over spectrum`
12. `feat(soap_bubble): apply Fresnel-driven alpha and roughness LOD`
13. `feat(soap_bubble): wire height texture as thickness source`
14. `feat(soap_bubble): add procedural thickness mode (gravity + value noise)`
15. `feat(soap_bubble): add ImGui parameter panel (6 groups)`
16. `feat(soap_bubble): expose all params via CLI11`
17. `feat(soap_bubble): add debug visualizations (heatmap, fresnel)`
18. (선택) `style: ...`
19. (선택) `docs(readme): add soap_bubble entry`

총 17~19개 commit. 전부 깨끗하면 plan 완료.

---

# Plan 완료 후

- 모든 sanity 통과 → user에게 피드백 요청
- 회귀 발견 → 해당 task 회귀
- 추가 기능 (refraction, instancing 등)은 spec의 Out of Scope에 따라 별도 PR/plan