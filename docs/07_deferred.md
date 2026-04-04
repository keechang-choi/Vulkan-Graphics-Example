# Deferred Shading

## 개요

Multi Render Target(MRT) 오프스크린 렌더링을 활용한 Deferred Shading 예제. G-Buffer에 기하 정보를 먼저 렌더링한 뒤, 조명 패스에서 최대 10개의 포인트 라이트를 효율적으로 처리한다.

## 핵심 개념

- **G-Buffer (Geometry Buffer)** — 오프스크린 패스에서 위치/법선/알베도/ARM/이미시브를 별도 어태치먼트에 저장
- **Composition Pass** — G-Buffer를 샘플링해 모든 라이트를 한 번에 계산
- **Multi Render Target** — 단일 렌더 패스에서 6개 어태치먼트에 동시 출력
- **Dynamic UBO** — 다수의 모델 인스턴스(그리드 배치)를 하나의 버퍼로 제어
- **Specialization Constants** — 디버그 뷰(법선/알베도 등 개별 출력) 컴파일 시 선택

## 렌더링 패스 구조

```
[Offscreen Pass]                    [Composition Pass]
  G-Buffer MRT 렌더링                  G-Buffer 샘플링
  ┌─────────────────────┐              ┌────────────────────┐
  │ position attachment  │──────────►  │                    │
  │ normal   attachment  │──────────►  │  Lighting Calc     │──► Swapchain
  │ albedo   attachment  │──────────►  │  (up to 10 lights) │
  │ ARM      attachment  │──────────►  │                    │
  │ emissive attachment  │──────────►  └────────────────────┘
  │ depth    attachment  │
  └─────────────────────┘
```

> ARM = Ambient Occlusion / Roughness / Metallic (PBR 텍스처 팩)

## G-Buffer 어태치먼트

| 어태치먼트 | 내용 |
|-----------|------|
| position | 월드 공간 위치 (vec3) |
| normal | 월드 공간 법선 (vec3) |
| albedo | 기본 색상 (RGB) |
| ARM | AO / Roughness / Metallic |
| emissive | 자체 발광 색상 |
| depth | 깊이 버퍼 |

## 데이터 구조

```cpp
#define MAX_LIGHTS 10

struct Light {
    glm::vec4 position;
    glm::vec3 color;
    float radius;
};

struct UniformDataOffscreen {
    glm::mat4 projection;
    glm::mat4 view;
};

struct UniformDataComposition {
    Light lights[MAX_LIGHTS];
    glm::vec4 viewPos;
    int debugDisplayTarget;  // 개별 버퍼 디버그 뷰
    int numLights;
    float nearPlane, farPlane, farClamp;
};
```

## 파이프라인 구성

```cpp
struct {
    vk::raii::Pipeline offScreen;          // G-Buffer 채우기
    vk::raii::Pipeline offScreenSimpleMesh; // 단순 메시 G-Buffer
    vk::raii::Pipeline composition;         // 조명 계산
    std::vector<vk::raii::Pipeline> displayTargets; // 디버그 뷰
} pipelines;
```

## 셰이더

| 파일 | 역할 |
|------|------|
| `mrt.vert` | G-Buffer 패스 정점 변환 |
| `mrt.frag` | 6개 어태치먼트에 기하 정보 출력 |
| `deferred.vert` | 풀스크린 쿼드 |
| `deferred.frag` | G-Buffer 샘플링 + 조명 계산 |

## 그리드 인스턴싱

모델을 X-Z 축으로 그리드 배치하여 대규모 라이팅 성능 테스트:
```cpp
struct Options {
    int32_t modelNumX = 4;   // X축 모델 수
    int32_t modelNumZ = 4;   // Z축 모델 수
    float spacingX = 4.f;
    float spacingZ = 4.f;
    int32_t debugDisplayTarget;  // 0=최종, 1~=개별 G-Buffer
    float farClamp;              // 깊이 시각화 클램프
};
```

## 주요 특징

- Subpass dependencies로 오프스크린 → 컴포지션 패스 암묵적 동기화
- 각 프레임마다 G-Buffer 어태치먼트 생성 (multiple frames in flight 지원)
- `debugDisplayTarget`으로 position/normal/albedo/ARM 등 개별 G-Buffer 채널 확인 가능
- PBR 텍스처(ARM 팩) 지원으로 물리 기반 재질 표현
