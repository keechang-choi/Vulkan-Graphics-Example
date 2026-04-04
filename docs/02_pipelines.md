# Pipelines

## 개요

하나의 glTF 씬을 세 가지 서로 다른 렌더링 파이프라인으로 그리는 예제. 동일한 지오메트리를 Phong 셰이딩, Toon 셰이딩, Wireframe으로 동시에 표현한다.

## 핵심 개념

- 여러 Graphics Pipeline을 동일한 씬에 적용
- glTF 모델 로딩 (`vgeu::glTF::Model`)
- 조명 정보(lightPos)를 포함한 확장된 UBO
- Normal Matrix / Inverse View Matrix 전달

## 데이터 구조

```cpp
struct GlobalUbo {
    glm::mat4 projection;
    glm::mat4 model;
    glm::mat4 view;
    glm::vec4 lightPos;
    glm::mat4 normalMatrix;
    glm::mat4 inverseView;
};
```

## 파이프라인 구성

```cpp
struct {
    vk::raii::Pipeline phong;     // Phong 셰이딩
    vk::raii::Pipeline toon;      // 툰 셰이딩
    vk::raii::Pipeline wireframe; // 와이어프레임
} pipelines;
```

## 셰이더

| 파일 | 역할 |
|------|------|
| `phong.vert/frag` | Diffuse + Specular 조명 계산 |
| `toon.vert/frag` | 단계적 음영(Cel shading) |
| `wireframe.vert/frag` | 와이어프레임 렌더링 |

## 주요 특징

- `getEnabledFeatures()`로 fillModeNonSolid 기능 활성화 (wireframe에 필요)
- 동일한 Descriptor Set Layout을 여러 파이프라인이 공유
- Triangle → Pipelines로 진화: 단순 정점에서 glTF 씬 렌더링으로 확장
