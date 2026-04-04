# Triangle

## 개요

Vulkan 렌더링 파이프라인의 가장 기본적인 예제. 하드코딩된 정점 데이터로 삼각형을 그리며, 전체 Vulkan 렌더링 흐름을 처음부터 구성하는 방법을 보여준다.

## 핵심 개념

- Vertex/Index Buffer 생성 및 바인딩
- Descriptor Set Layout / Pool / Set 구성
- Graphics Pipeline 생성 (단일 파이프라인)
- Uniform Buffer Object (UBO)를 통한 MVP 행렬 전달
- Command Buffer 레코딩 및 제출

## 데이터 구조

```cpp
struct Vertex {
    float position[3];
    float color[3];
};

struct GlobalUbo {
    glm::mat4 projection;
    glm::mat4 model;
    glm::mat4 view;
};
```

## 셰이더

| 파일 | 역할 |
|------|------|
| `simple_shader.vert` | 정점 위치 변환 (MVP 행렬 적용) |
| `simple_shader.frag` | 정점 색상 출력 |

## 렌더링 파이프라인

```
Vertex Buffer → Vertex Shader (MVP) → Fragment Shader (Color) → Swapchain
```

## 주요 특징

- glTF 모델 없이 순수 하드코딩 정점 데이터 사용
- Vulkan 파이프라인의 최소 구성 요소만 포함
- 이후 예제들의 기반이 되는 구조
