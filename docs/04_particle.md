# Particle

## 개요

Vulkan Compute Shader를 활용한 GPU 기반 파티클 시스템. N-body 중력 시뮬레이션, glTF 모델 메시 어트랙터, 컴퓨트 셰이더 스키닝을 모두 포함하며, 파티클 궤적 시각화(Tail)까지 구현한다.

## 핵심 개념

- **Compute Shader 기반 물리 시뮬레이션** (그래픽스 큐와 분리된 컴퓨트 큐 사용)
- **수치 적분 방법 선택** — Euler, RK4 등 (`integrator` 파라미터)
- **Specialization Constants** — 워크그룹 크기, 공유 메모리 크기를 컴파일 시 최적화
- **파티클 Tail(궤적)** — 링버퍼 구조로 이전 위치를 저장하여 꼬리 시각화
- **Compute Skinning** — GPU에서 애니메이션 정점을 계산 후 파티클에 어트랙터로 활용

## 데이터 구조

```cpp
struct Particle {
    glm::vec4 pos;
    glm::vec4 vel;
    glm::vec4 pk[4];  // RK4 중간값
    glm::vec4 vk[4];
    glm::vec4 attractionWeight;
};

struct SpecializationData {
    uint32_t sharedDataSize;  // 공유 메모리 크기
    uint32_t integrator;      // 적분 방법
    uint32_t integrateStep;
    uint32_t localSizeX;      // 워크그룹 크기
};
```

## 컴퓨트 파이프라인

| 파이프라인 | 역할 |
|-----------|------|
| `pipelineCalculate` | 파티클 간 힘 계산 (공유 메모리 활용) |
| `pipelineIntegrate` | 위치/속도 적분 |
| `pipelineModelAnimate` | glTF 스키닝 계산 |
| `pipelineModelCalculate` | 모델 메시 어트랙터 힘 계산 |
| `pipelineModelIntegrate` | 모델 어트랙터 기반 적분 |

## 셰이더

| 파일 | 역할 |
|------|------|
| `particle_calculate.comp` | N-body 힘 계산 |
| `particle_integrate.comp` | 위치 적분 |
| `particle_model_calculate.comp` | 메시 어트랙터 힘 |
| `particle_model_integrate.comp` | 메시 기반 적분 |
| `model_animate.comp` | GPU 스키닝 |
| `particle.vert/frag` | 파티클 포인트 렌더링 |
| `tail.vert/frag` | 궤적 라인 렌더링 |

## 동기화 구조

```
Compute Queue ──semaphore──► Graphics Queue
     ▲                            │
     └────────semaphore───────────┘
```

- 그래픽스/컴퓨트 큐 간 세마포어로 버퍼 접근 동기화

## 주요 설정 옵션 (Options)

| 옵션 | 설명 |
|------|------|
| `numParticles` | 파티클 수 (기본 1024) |
| `numAttractors` | 어트랙터 수 |
| `integrator` | 적분 방법 (0=Euler, 1=RK4 등) |
| `tailSize` | 궤적 길이 |
| `attractionType` | 어트랙션 유형 선택 |
| `gravity`, `power`, `soften` | 물리 파라미터 |
