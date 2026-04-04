# Cloth

## 개요

GPU 기반 PBD 천(Cloth) 시뮬레이션 예제. Compute Shader로 천 물리를 계산하며, Geometry Shader로 법선 벡터를 시각화한다. [The Ten Minute Physics](https://matthias-research.github.io/pages/tenMinutePhysics/index.html) 강의 14~16번을 기반으로 구현한다.

## 핵심 개념

- **GPU PBD Cloth** — Compute Shader에서 거리 제약(Distance Constraint) 해결
- **Gauss-Seidel vs Jacobi** — 두 가지 제약 해결 방식 선택 가능
- **Atomic Float Add** — Jacobi 방식의 보정값 누산에 GPU atomic 연산 사용 (`VK_EXT_shader_atomic_float`)
- **Geometry Shader** — 법선 벡터를 선분으로 시각화
- **컴퓨트-그래픽스 큐 소유권 이전** — SSBO를 컴퓨트 큐에서 그래픽스 큐로 이전
- **SpatialHash** — 천-물체 충돌 감지

## 아키텍처

### Cloth 클래스
glTF 메시에서 파티클/제약 데이터를 초기화하고 GPU 버퍼를 관리:
```cpp
class Cloth {
    void initParticlesData(vertices, indices, translate, rotate, scale);
    void initDistConstraintsData(numX, numY);  // 그리드 제약 자동 생성
    void initDistConstraintsData(distConstraints);  // 외부 제약 직접 지정
};
```

### 파티클 데이터 구조
```cpp
// 컴퓨트용 (calculate SSBO)
struct ParticleCalculate {
    glm::vec4 prevPos;
    glm::vec4 pos;
    glm::vec4 vel;
    glm::vec4 corr;   // Jacobi 보정값
    glm::vec4 normal;
};

// 렌더링용 (render SSBO → vertex buffer)
struct ParticleRender {
    glm::vec4 pos;    // w = inv mass
    glm::vec4 normal;
    glm::vec2 uv;
};

struct DistConstraint {
    glm::uvec2 constIds;  // 두 파티클 인덱스
    float restLength;
};
```

## 컴퓨트 파이프라인 (ComputeType enum)

| ComputeType | 역할 |
|------------|------|
| `kInitializeParticles` | 파티클 초기화 |
| `kInitializeConstraints` | 제약 초기화 |
| `kIntegrate` | 예측 위치 계산 |
| `kSolveCollision` | 충돌 해결 |
| `kSolveDistanceConstraintsGauss` | Gauss-Seidel 거리 제약 |
| `kSolveDistanceConstraintsJacobi` | Jacobi 거리 제약 |
| `kAddCorrections` | Jacobi 보정값 합산 |
| `kUpdateVel` | 속도 갱신 |
| `kUpdateMesh` | 메시 위치 갱신 |
| `kUpdateNormals` | 법선 재계산 |
| `kRaycastingTriangleDistance` | 마우스 드래그용 레이캐스트 |

## 셰이더

| 파일 | 역할 |
|------|------|
| `cloth.comp` | 주 시뮬레이션 (atomic float 사용) |
| `cloth_no_atomic_add.comp` | atomic 미지원 장치용 대체 |
| `model_animate.comp` | glTF 스키닝 (충돌 물체 애니메이션) |
| `cloth.vert` + `phong.frag` | 천 렌더링 |
| `cloth_normal.vert/geom/frag` | 법선 시각화 |
| `wireCloth.vert` + `wireframe.frag` | 와이어프레임 |
| `simpleMesh/simpleLine` | 충돌 물체 렌더링 |

## 물리 파라미터 (Options)

| 옵션 | 설명 |
|------|------|
| `numSubsteps` | 서브스텝 수 |
| `gravity` | 중력 |
| `stiffness` | 제약 강성 |
| `alpha` | XPBD compliance |
| `jacobiScale` | Jacobi 감쇠 계수 |
| `thickness` | 천 두께 (충돌 반경) |
| `friction` | 마찰 계수 |
| `collisionRadius` | 충돌 감지 반경 |
| `showNormals` | 법선 시각화 토글 |

## 주요 특징

- `VK_EXT_shader_atomic_float` 확장 사용 여부에 따라 파이프라인 분기
- 컴퓨트-그래픽스 큐 패밀리 간 버퍼 소유권 이전(barrier) 처리
- 마우스 클릭으로 천 드래그 인터랙션 (레이캐스트 기반)
- 최대 10개 천 오브젝트 동시 지원 (`kMaxNumClothModels`)
