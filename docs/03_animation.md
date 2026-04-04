# Animation

## 개요

glTF 스켈레탈 애니메이션과 스키닝(Skinning)을 구현한 예제. 여러 모델 인스턴스를 Dynamic UBO로 개별 제어하며, 각 인스턴스가 독립적인 애니메이션 타임라인을 가진다.

## 핵심 개념

- glTF Skeletal Animation (Joint/Bone 기반)
- Dynamic Uniform Buffer Object — 인스턴스별 모델 행렬을 하나의 버퍼에 패킹
- Multi-instance rendering
- UBO 정렬(alignment) 패딩 처리 (`padUniformBufferSize`)

## 데이터 구조

```cpp
// 인스턴스별 동적 UBO 요소
struct DynamicUboElt {
    glm::mat4 modelMatrix;
    glm::vec4 modelColor;
};

// 모델 인스턴스 (각자 독립 애니메이션 상태 보유)
struct ModelInstance {
    std::shared_ptr<vgeu::glTF::Model> model;
    std::string name;
    bool isBone;
    int animationIndex;
    float animationTime;
};
```

## Descriptor Set 구성

| Set | 내용 |
|-----|------|
| Global UBO | Projection, View, Light |
| Dynamic UBO | 인스턴스별 Model Matrix, Color |

## 파이프라인

```cpp
struct {
    vk::raii::Pipeline phong;     // Phong 셰이딩
    vk::raii::Pipeline wireframe; // 와이어프레임
} pipelines;
```

## 셰이더

| 파일 | 역할 |
|------|------|
| `phong.vert/frag` | 스키닝 행렬 적용 + Phong 조명 |
| `wireframe.vert/frag` | 뼈대(bone) 구조 시각화 |

## 주요 특징

- `instanceMap`으로 이름 기반 인스턴스 검색
- `animationTime` 누적으로 프레임 독립적 애니메이션 재생
- GPU 메모리 정렬 요구사항을 고려한 Dynamic UBO 패딩 계산
