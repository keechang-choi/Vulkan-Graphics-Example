# PBD (Position Based Dynamics)

## 개요

CPU 기반 Position Based Dynamics 시뮬레이션 예제. [The Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html) 강의의 2D 시뮬레이션 예제들을 Vulkan으로 구현한다. 7가지 서로 다른 시뮬레이션 시나리오를 포함한다.

## 핵심 개념

- **Position Based Dynamics (PBD)** — 힘 기반이 아닌 위치/제약 기반 물리 시뮬레이션
- **XPBD (Extended PBD)** — compliance 파라미터를 통한 부드러운 제약
- **2D Soft Body** — 삼각형 메시 기반 변형 가능 물체 (`SoftBody2D`)
- **Spatial Hashing** — O(1) 근접 충돌 쿼리를 위한 공간 해시 자료구조
- **CPU 시뮬레이션 + GPU 렌더링** — 시뮬레이션은 CPU, 렌더링은 Vulkan

## 포함된 시뮬레이션 시나리오

| 인덱스 | 내용 |
|--------|------|
| Sim 1 | 볼(Ball) 충돌 시뮬레이션 |
| Sim 2 | 다수 볼 충돌 |
| Sim 3 | 진자/체인 시뮬레이션 |
| Sim 4 | 강체 링크 |
| Sim 5 | 복합 진자 시스템 |
| Sim 6 | 2D Soft Body (변형 가능한 물체) |
| Sim 7 | 강체 원형 물체 |

## 주요 클래스

### SoftBody2D
```cpp
class SoftBody2D {
    void preSolve(dt, gravity, rectScale);      // 예측 위치 갱신
    void solve(dt, edgeCompliance, areaCompliance, collisionStiffness);  // 제약 해결
    void postSolve(dt);                          // 속도 갱신
    void startGrab / moveGrabbed / endGrab(...); // 마우스 인터랙션
};
```

### SpatialHash
공간을 균일 격자로 나누어 근접 물체를 O(1)에 탐색:
```cpp
class SpatialHash {
    void addPos(positions);       // 위치 등록
    void createPartialSum();      // 정렬
    void query(pos, maxCellDist, queryIds);  // 근접 쿼리
    void queryTri(aabb, queryIds);           // AABB 기반 삼각형 쿼리
};
```

## 제약 해결 함수들

| 함수 | 역할 |
|------|------|
| `solveDistanceConstraint` | 두 점 간 거리 제약 |
| `solveEdgePointDistanceConstraint` | 점-엣지 거리 제약 |
| `solveTrianglePointDistanceConstraint` | 점-삼각형 거리 제약 |
| `solveEdgePointCollisionConstraint` | 점-엣지 충돌 제약 |
| `handleBallCollision` | 볼-볼 충돌 처리 |
| `handleWallCollision` | 볼-벽 충돌 처리 |

## 렌더링 구조

- **SimpleModel** — 원, 사각형, 라인 등 단순 도형 렌더링용 헬퍼
- **Phong / SimpleMesh / WireMesh / SimpleLine** 파이프라인 사용
- CPU에서 갱신된 정점 데이터를 매 프레임 GPU 버퍼에 업로드 (mapped buffer)
- GPU compute shader(`model_animate.comp`)로 glTF 스키닝도 병행 지원

## 주요 설정 옵션 (Options)

| 옵션 | 설명 |
|------|------|
| `numSubsteps` | 물리 스텝 세분화 수 (기본 10) |
| `gravity` | 중력 가속도 |
| `edgeCompliance` | 엣지 제약 유연성 |
| `areaCompliance` | 면적 제약 유연성 |
| `restitution` | 충돌 반발 계수 |
| `tailSize` | 궤적 길이 |
