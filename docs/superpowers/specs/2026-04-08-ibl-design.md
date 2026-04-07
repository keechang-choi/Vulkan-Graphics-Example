# IBL (Image-Based Lighting) — Design Spec

**Date:** 2026-04-08  
**Based on:** `src/examples/pbr` (deferred PBR)  
**Reference:** LearnOpenGL IBL, Sascha Willems pbrtexture example  
**HDR Asset:** `assets/textures/hdr/tree_lined_driveway_4k.hdr`

---

## 개요

기존 PBR 예제에 IBL을 추가한다. 환경 HDR 맵에서 irradiance 큐브맵, pre-filtered 큐브맵, BRDF LUT를 startup에 1회 생성하고, composition 패스에서 flat ambient를 IBL ambient로 교체한다. skybox는 별도 패스로 배경 렌더링. `useIBL` 단일 토글로 IBL ambient와 skybox를 동시에 on/off.

---

## 렌더링 파이프라인

```
[Startup 1회 — IBL 생성]
  ① loadHdrTexture()         HDR equirectangular → VkImage (R32G32B32A32_SFLOAT, 2D)
  ② buildEnvCubemap()        equirect → envCubemap  512×512, R16G16B16A16_SFLOAT
  ③ buildIrradianceMap()     envCubemap → irradianceMap  64×64, R32G32B32A32_SFLOAT, full mip
  ④ buildPrefilteredMap()    envCubemap → prefilteredMap 512×512, R16G16B16A16_SFLOAT, 10 mip
  ⑤ buildBrdfLut()           → brdfLut 512×512, R16G16_SFLOAT

[매 프레임]
  Pass 1: G-buffer offscreen MRT        ← 기존 유지
  Pass 2: Composition                   ← pbr.frag 수정 (IBL ambient 추가)
  Pass 3: Skybox                        ← 신규 (useIBL 시에만 draw)
  Pass 4: Sprite                        ← 기존 유지
```

---

## IBL 생성 공통 패턴 (Sascha Willems 방식)

각 큐브맵 생성 단계는 동일한 패턴을 따른다:

```
1. 최종 큐브맵 이미지 생성
   usage = SAMPLED | TRANSFER_DST
   flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT
   arrayLayers = 6

2. 임시 offscreen 2D 이미지 생성 (단면 렌더타겟)
   usage = COLOR_ATTACHMENT | TRANSFER_SRC
   arrayLayers = 1

3. 전용 renderpass + framebuffer (offscreen 2D 기준)

4. for each mip:
     for each face (0~5):
       captureUbo.mvp = proj * rotMatrices[face]
       업데이트 후 draw
       endRenderPass()
       vkCmdCopyImage(offscreen → cubemap[face][mip])

5. 큐브맵 전체를 SHADER_READ_ONLY_OPTIMAL으로 layout 전환

6. 임시 자원 (offscreen image, renderpass, framebuffer, pipeline) 해제
```

### 6면 캡처 회전 행렬

```cpp
std::vector<glm::mat4> captureViews = {
    glm::rotate(glm::rotate(glm::mat4(1.f), glm::radians( 90.f), {0,1,0}), glm::radians(180.f), {1,0,0}),  // +X
    glm::rotate(glm::rotate(glm::mat4(1.f), glm::radians(-90.f), {0,1,0}), glm::radians(180.f), {1,0,0}),  // -X
    glm::rotate(glm::mat4(1.f), glm::radians(-90.f), {1,0,0}),  // +Y
    glm::rotate(glm::mat4(1.f), glm::radians( 90.f), {1,0,0}),  // -Y
    glm::rotate(glm::mat4(1.f), glm::radians(180.f), {1,0,0}),  // +Z
    glm::rotate(glm::mat4(1.f), glm::radians(180.f), {0,0,1}),  // -Z
};
glm::mat4 captureProj = glm::perspective(glm::radians(90.f), 1.f, 0.1f, 512.f);
```

### capture UBO

```cpp
struct CaptureUbo {
    glm::mat4 mvp;  // 64바이트, per-face 업데이트
};
// HOST_VISIBLE | HOST_COHERENT, IBL 생성 완료 후 해제
```

---

## 각 단계별 상세

### ① loadHdrTexture

- `stbi_loadf()` → float* pixels
- VkImage: `VK_FORMAT_R32G32B32A32_SFLOAT`, 2D, usage = SAMPLED | TRANSFER_DST
- staging buffer → oneTimeSubmit으로 업로드

### ② buildEnvCubemap

| 항목 | 값 |
|------|----|
| 해상도 | 512×512 |
| 포맷 | R16G16B16A16_SFLOAT |
| mip | full mip chain (`vkCmdGenerateMipmaps` 후 생성) — prefilter 셰이더에서 `textureLod`로 샘플링해 밝은 점 아티팩트 방지 |
| 셰이더 | equirect.vert + equirect.frag |
| push constant | 없음 (mvp는 capture UBO) |
| descriptor | set=0 binding=0: sampler2D hdrTexture |

equirect.frag: `SampleSphericalMap(normalize(localPos))` → texture 샘플

### ③ buildIrradianceMap

| 항목 | 값 |
|------|----|
| 해상도 | 64×64 |
| 포맷 | R32G32B32A32_SFLOAT |
| mip | numMips = floor(log2(64)) + 1 = 7 |
| 셰이더 | equirect.vert (재사용) + irradiance.frag |
| push constant | `{ float deltaPhi; float deltaTheta; }` (8바이트) |
| descriptor | set=0 binding=0: samplerCube envCubemap |

### ④ buildPrefilteredMap

| 항목 | 값 |
|------|----|
| 해상도 | 512×512 |
| 포맷 | R16G16B16A16_SFLOAT |
| mip | numMips = floor(log2(512)) + 1 = 10 |
| 셰이더 | equirect.vert (재사용) + prefilter.frag |
| push constant | `{ float roughness; uint32_t numSamples; }` (8바이트) |
| descriptor | set=0 binding=0: samplerCube envCubemap |

외부 루프: mip (roughness = mip / (numMips-1)), 내부 루프: face

### ⑤ buildBrdfLut

| 항목 | 값 |
|------|----|
| 해상도 | 512×512 |
| 포맷 | R16G16_SFLOAT |
| 셰이더 | brdf_lut.vert (fullscreen triangle, 3 vertex) + brdf_lut.frag |
| vertex buffer | 없음 (gl_VertexIndex로 좌표 생성) |
| descriptor | 없음 |
| push constant | 없음 |

---

## 신규 셰이더 목록 (shaders/pbr/)

| 파일 | 용도 |
|------|------|
| `equirect.vert` | 큐브 localPos 출력, mvp는 capture UBO binding=0 |
| `equirect.frag` | equirectangular 샘플링 → cubemap face |
| `irradiance.frag` | 반구 적분 (push: deltaPhi, deltaTheta) |
| `prefilter.frag` | GGX importance sampling (push: roughness, numSamples) |
| `brdf_lut.vert` | fullscreen triangle (3 vertex, no buffer) |
| `brdf_lut.frag` | BRDF 적분 → RG |
| `skybox.vert` | 큐브 36 vertex 하드코딩, push constant: view+proj |
| `skybox.frag` | envCubemap 샘플링 + Uncharted2 tonemap + gamma |

### 수정 셰이더

| 파일 | 변경 |
|------|------|
| `pbr.frag` | set=1 IBL 텍스처 바인딩 추가, ambient 블록 교체 |

---

## Descriptor 구조

### Composition 패스

```
set=0  binding=0~7: 기존 G-buffer 텍스처 + UBO      ← 변경 없음
set=1  binding=0:   samplerCube irradianceMap        ← 신규
set=1  binding=1:   samplerCube prefilteredMap       ← 신규
set=1  binding=2:   sampler2D   brdfLut              ← 신규
```

### Skybox 패스

```
set=0  binding=0:   samplerCube envCubemap
set=0  binding=1:   UBO (exposure, gamma 등 파라미터)
push constant: { glm::mat4 view; glm::mat4 projection; }  // 128바이트
```

skybox view matrix: `glm::mat4(glm::mat3(camera.view))` (translation 제거)

---

## pbr.frag 수정

### UBO 추가 필드 (UniformDataComposition)

```cpp
int useIBL{0};
int _pad3[3];  // 16바이트 정렬
```

### 셰이더 추가

```glsl
// set=1
layout(set=1, binding=0) uniform samplerCube irradianceMap;
layout(set=1, binding=1) uniform samplerCube prefilteredMap;
layout(set=1, binding=2) uniform sampler2D   brdfLut;

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0)
           * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
```

### ambient 블록 교체

```glsl
vec3 ambient;
if (ubo.useIBL != 0) {
    vec3 F  = FresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    vec3 kD = (1.0 - F) * (1.0 - metallic);

    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuse    = irradiance * albedo;

    const float MAX_LOD = 9.0;  // numMips - 1
    vec3 R = reflect(-V, N);
    vec3 prefilteredColor =
        textureLod(prefilteredMap, R, roughness * MAX_LOD).rgb;
    vec2 envBRDF =
        texture(brdfLut, vec2(max(dot(N, V), 0.0), roughness)).rg;
    vec3 specular = prefilteredColor * (F * envBRDF.x + envBRDF.y);

    ambient = (kD * diffuse + specular) * ao;
} else {
    ambient = vec3(ubo.ambientStrength) * albedo * ao;
}
```

---

## VgeuImage 확장

현재 `VgeuImage`는 arrayLayers=1 고정. cubemap 지원 오버로드 추가:

```cpp
// vgeu_buffer.hpp / vgeu_buffer.cpp
VgeuImage(const vk::raii::Device& device, VmaAllocator allocator,
          vk::Format format, const vk::Extent2D& extent,
          vk::ImageTiling tiling, vk::ImageUsageFlags usage,
          vk::ImageLayout initialLayout, VmaMemoryUsage memUsage,
          VmaAllocationCreateFlags allocCreateFlags,
          vk::ImageAspectFlags aspectMask, uint32_t mipLevels,
          bool isCubemap);  // arrayLayers=6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT
```

image view type: `VK_IMAGE_VIEW_TYPE_CUBE` (isCubemap=true 시)

---

## C++ 구조 변경 (pbr.hpp)

### 신규 멤버

```cpp
// IBL 텍스처
std::unique_ptr<vgeu::VgeuImage> hdrTexture;       // 2D equirectangular
std::unique_ptr<vgeu::VgeuImage> envCubemap;        // 512×512
std::unique_ptr<vgeu::VgeuImage> irradianceMap;     // 64×64
std::unique_ptr<vgeu::VgeuImage> prefilteredMap;    // 512×512
std::unique_ptr<vgeu::VgeuImage> brdfLut;           // 512×512 2D
vk::raii::Sampler iblSampler = nullptr;             // IBL 전용 sampler

// IBL descriptor
vk::raii::DescriptorSetLayout iblDescriptorSetLayout = nullptr;
std::vector<vk::raii::DescriptorSet> iblDescriptorSets;

// Skybox
vk::raii::Pipeline skyboxPipeline = nullptr;
vk::raii::PipelineLayout skyboxPipelineLayout = nullptr;
vk::raii::DescriptorSetLayout skyboxDescriptorSetLayout = nullptr;
std::vector<vk::raii::DescriptorSet> skyboxDescriptorSets;
```

### Options 추가

```cpp
bool useIBL = false;
float iblExposure = 4.5f;
float iblGamma = 2.2f;
```

### 신규 함수

```cpp
void prepareIBL();
  // loadHdrTexture()
  // buildEnvCubemap()
  // buildIrradianceMap()
  // buildPrefilteredMap()
  // buildBrdfLut()

void prepareSkyboxPipeline();
void prepareSkyboxDescriptors();
void updateUboComposition();  // useIBL 필드 추가 업데이트
```

---

## 제외 범위

- IBL 환경맵 런타임 교체 (재생성 미지원)
- 동적 IBL (움직이는 환경)
- reflection capture (로컬 큐브맵)
