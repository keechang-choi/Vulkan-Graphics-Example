# Vulkan Graphics Example — 예제 목록 및 개요

Vulkan API를 사용해 컴퓨터 그래픽스의 다양한 기법을 구현한 예제 모음.  
C++ (Vulkan-hpp RAII), CMake, MinGW-w64 환경에서 개발되었으며 Windows 기준으로 테스트됨.

## 예제 목록

| # | 예제 | 주요 기법 | 문서 |
|---|------|-----------|------|
| 1 | [triangle](../src/examples/triangle) | Vertex/Index Buffer, 기본 파이프라인 | [01_triangle.md](01_triangle.md) |
| 2 | [pipelines](../src/examples/pipelines) | 다중 파이프라인, glTF 로딩, Phong/Toon/Wireframe | [02_pipelines.md](02_pipelines.md) |
| 3 | [animation](../src/examples/animation) | 스켈레탈 애니메이션, 스키닝, Dynamic UBO | [03_animation.md](03_animation.md) |
| 4 | [particle](../src/examples/particle) | Compute Shader, N-body, GPU 스키닝, RK4 적분 | [04_particle.md](04_particle.md) |
| 5 | [pbd](../src/examples/pbd) | CPU PBD, Soft Body, Spatial Hash, 다중 시나리오 | [05_pbd.md](05_pbd.md) |
| 6 | [cloth](../src/examples/cloth) | GPU PBD Cloth, Jacobi/Gauss-Seidel, Geometry Shader | [06_cloth.md](06_cloth.md) |
| 7 | [deferred](../src/examples/deferred) | Deferred Shading, G-Buffer MRT, 다중 조명 | [07_deferred.md](07_deferred.md) |

## 공통 기반 구조 (`src/base/`)

모든 예제는 `VgeBase`를 상속하며 아래 공통 유틸리티를 사용한다:

| 모듈 | 역할 |
|------|------|
| `vge_base` | Vulkan 초기화, 스왑체인, 렌더루프 |
| `vgeu_buffer` | VMA 기반 버퍼 추상화 |
| `vgeu_gltf` | glTF 2.0 모델 로딩 (스키닝 포함) |
| `vgeu_texture` | 텍스처 로딩 |
| `vgeu_camera` | 카메라 (Arcball / FPS) |
| `vgeu_ui_overlay` | ImGui 기반 디버그 UI |
| `vgeu_utils` | 공통 유틸리티 |

## 예제 복잡도 흐름

```
triangle
  └─► pipelines (glTF + 다중 파이프라인)
        └─► animation (스키닝 + Dynamic UBO)
              ├─► particle (Compute + GPU 스키닝)
              ├─► pbd (CPU 물리 + 다중 시뮬레이션)
              └─► cloth (GPU PBD + Geometry Shader)

deferred (독립적 렌더링 기법 — MRT + 조명)
```

## 빌드

```bash
git submodule init && git submodule update
./mingwBuild.bat
./build/<example>.exe
```
