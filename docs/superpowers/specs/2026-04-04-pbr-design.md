# PBR Example — Design Spec

**Date:** 2026-04-04  
**Based on:** `src/examples/deferred`  
**Reference:** https://learnopengl.com/PBR/Lighting

---

## Overview

`deferred` 예제를 기반으로 Composition 패스의 Blinn-Phong 조명을 Cook-Torrance PBR BRDF로 교체하고, 라이트 위치를 Billboard Sprite로 시각화하는 예제.

에셋(metal_plate floor, DamagedHelmet grid)과 G-Buffer 구조는 deferred와 동일하게 유지.

---

## 렌더링 파이프라인

```
[Pass 1] G-Buffer (Offscreen MRT)          ← deferred 그대로 유지
  mrt.vert / mrt.frag
  Attachments: position | normal | albedo | ARM | emissive | depth

        ↓ subpass dependency

[Pass 2] Composition (Swapchain renderpass)
  pbr.vert / pbr.frag
  Cook-Torrance BRDF + Reinhard tone mapping + gamma correction

        ↓ 동일 renderpass, depth buffer 공유

[Pass 3] Forward Sprite (동일 Swapchain renderpass 이어서)
  sprite.vert / sprite.frag
  Billboard quad × numLights, depth test on / depth write off
```

Pass 2와 Pass 3은 동일한 swapchain renderpass를 공유하며, depth buffer를 유지해 sprite가 씬 오브젝트 뒤에 자연스럽게 가려진다.

---

## 파일 구조

### 신규 생성
```
src/examples/pbr/
  pbr.hpp
  pbr.cpp

shaders/pbr/
  mrt.vert          ← deferred/mrt.vert 복사 (변경 없음)
  mrt.frag          ← deferred/mrt.frag 복사 (변경 없음)
  pbr.vert          ← deferred/deferred.vert 복사 (변경 없음)
  pbr.frag          ← deferred/deferred.frag 대체 (PBR 구현)
  sprite.vert       ← 신규
  sprite.frag       ← 신규
```

### 변경 파일
```
src/examples/CMakeLists.txt   ← pbr 예제 추가
CMakeLists.txt                ← (필요 시)
```

---

## PBR Lighting (pbr.frag)

### Cook-Torrance BRDF

```
Lo = Σ (kD * albedo/π + kS * D*F*G / (4*NdotV*NdotL)) * radiance * NdotL
```

| 항목 | 구현 |
|------|------|
| **D** Normal Distribution | Trowbridge-Reitz GGX |
| **F** Fresnel | Fresnel-Schlick. F0 = mix(vec3(0.04), albedo, metallic) |
| **G** Geometry | Smith's Schlick-GGX (roughness remapping: r=(r+1)/2) |
| **kD** Diffuse ratio | `(1 - F) * (1 - metallic)` |
| **radiance** | `lightColor * (1 / distance²)` (순수 역제곱. Light.radius는 sprite 크기 스케일링에만 활용) |

### G-Buffer 입력 매핑

| G-Buffer | 채널 | PBR 파라미터 |
|----------|------|-------------|
| albedo attachment | rgb | albedo |
| ARM attachment | r | ao (ambient occlusion) |
| ARM attachment | g | roughness |
| ARM attachment | b | metallic |
| emissive attachment | rgb | emissive |

### 최종 출력

```glsl
vec3 ambient = vec3(0.03) * albedo * ao;
vec3 color = ambient + Lo + emissive;

// Reinhard tone mapping
color = color / (color + vec3(1.0));
// Gamma correction
color = pow(color, vec3(1.0/2.2));
outFragColor = vec4(color, 1.0);
```

### UBO 구조 (deferred와 동일 유지)

```glsl
struct Light {
    vec4 position;
    vec3 color;
    float radius;  // 감쇠 반경
};
#define MAX_LIGHTS 10
uniform UBO {
    Light lights[MAX_LIGHTS];
    vec4 viewPos;
    int debugDisplayTarget;
    int numLights;
    float nearPlane;
    float farPlane;
    float farClamp;
};
```

---

## 라이트 애니메이션

### 동작 방식

라이트들은 XZ 평면에서 원형 궤도를 공전. 각 라이트는 균등 위상(phase) 분배.

```cpp
for (int i = 0; i < numLights; i++) {
    float phase = (2.0f * PI * i) / numLights;
    float angle = animTime * rotationSpeed + phase;
    lights[i].position = vec4(
        orbitRadius * cos(angle),
        orbitHeight,
        orbitRadius * sin(angle),
        1.0f
    );
}
```

### 라이트 초기화

- 색상: 고정 팔레트 (deferred 예제와 유사하게 다채로운 색상)
- `animTime`: 매 프레임 deltaTime 누적

### UI 컨트롤 (ImGui)

| 항목 | 위젯 |
|------|------|
| Animate lights | checkbox (on/off) |
| Rotation speed | DragFloat |
| Orbit radius | DragFloat |
| Orbit height | DragFloat |
| Num lights (1~MAX_LIGHTS) | DragInt |
| Debug display target | RadioButton (기존 유지) |
| Far clamping | DragFloat (기존 유지) |

---

## Billboard Sprite 렌더링

### 파이프라인 설정

| 항목 | 값 |
|------|-----|
| Depth test | `eLessOrEqual` |
| Depth write | off |
| Blend | off (불투명) |
| Vertex input | 없음 (gl_VertexIndex로 quad 생성) |

### sprite.vert — Billboard 계산

```glsl
// View matrix의 right/up 벡터 추출로 카메라 방향 billboard
vec3 right = vec3(view[0][0], view[1][0], view[2][0]);
vec3 up    = vec3(view[0][1], view[1][1], view[2][1]);

// gl_VertexIndex(0~5)로 quad 오프셋 생성
vec3 worldPos = lightPos.xyz
              + right * offset.x * spriteSize
              + up    * offset.y * spriteSize;
```

- `gl_InstanceIndex`로 라이트 UBO 인덱싱
- `spriteSize`는 push constant로 전달
- quad 오프셋: `gl_VertexIndex % 6` → `{(-1,-1),(1,-1),(1,1),(-1,-1),(1,1),(-1,1)}` 하드코딩

### sprite.frag — Circle SDF

```glsl
vec2 uv = inUV * 2.0 - 1.0;  // [0,1] → [-1,1]
if (dot(uv, uv) > 1.0) discard;
float intensity = 1.0 - length(uv);  // 중심 밝고 가장자리 fade
outColor = vec4(lightColor * intensity, 1.0);
```

### Draw call

```cpp
// 별도 vertex buffer 없음
// numLights개 인스턴스, 6 vertices per quad (2 triangles)
cmdBuffer.draw(6, numLights, 0, 0);
```

Composition UBO를 sprite 파이프라인과 공유해 별도 라이트 버퍼 추가 없음.

---

## C++ 클래스 구조 (pbr.hpp)

deferred.hpp 대비 추가/변경 사항:

```cpp
// 라이트 애니메이션 상태
struct AnimationState {
    bool animate = true;
    float rotationSpeed = 0.5f;
    float orbitRadius = 8.0f;
    float orbitHeight = -2.0f;
    float animTime = 0.0f;
} lightAnim;

// Sprite 파이프라인
vk::raii::Pipeline spritePipeline = nullptr;
vk::raii::PipelineLayout spritePipelineLayout = nullptr;

// Sprite push constants
struct SpritePushConstants {
    float spriteSize;
};
```

Options 구조체에 `animationSpeed`, `orbitRadius`, `orbitHeight` 추가.

---

## 제외 범위

- IBL (Image-Based Lighting) — 별도 예제로 분리
- Shadow mapping
- Transparency / Alpha blending
- 라이트 색상 런타임 변경 UI
