# Soap Bubble Shader — Design Spec

**Date:** 2026-04-27
**Based on:** `src/examples/pbr` (IBL 인프라 재사용)
**Reference:** [Wikipedia — Thin-film interference](https://en.wikipedia.org/wiki/Thin-film_interference), Wyman 2013 ("Simple Analytic Approximations to the CIE XYZ Color Matching Functions")
**HDR Asset:** `assets/textures/hdr/tree_lined_driveway_4k.hdr` (PBR과 공유)
**Model Asset:** `assets/models/pirate_gold` (PBR과 공유, height texture를 두께 소스로 재활용)

---

## 개요

새 example `soap_bubble`을 추가한다. 박막 간섭(thin-film interference)을 가시광 분광 적분으로 계산해 박막의 두께·굴절률·시야각에 따른 무지개색 변화를 실시간으로 관찰 가능한 셰이더 데모. 모든 물리 파라미터를 ImGui 슬라이더로 노출해 "파라미터 굴려보기"가 주 사용 시나리오.

PBR example에 묻혀있던 IBL 베이킹/스카이박스 코드를 `vgeu_ibl` 모듈로 추출하고 PBR도 같이 마이그레이션한다. 회귀 검증은 PBR이 추출 전후 동일하게 동작하는지로.

---

## 결정사항 요약 (브레인스토밍 결과)

| 결정 | 선택 | 이유 |
|---|---|---|
| 박막 수학 | 분광 적분 + CIE XYZ → sRGB | Wikipedia 식을 셰이더에 직접 옮길 수 있고 파라미터 변화 색이 정확히 보임 |
| 색일치함수 | Wyman 2013 analytical fit | LUT/UBO 테이블 불필요, GLSL ~10줄 |
| 조명 SPD | flat (E(λ)=1) | IBL이 broadband 컬러 공급, 이중 분광 적분 불필요 |
| IBL 추출 범위 | `vgeu_ibl` 모듈로 추출 + PBR 마이그레이션 | PBR이 회귀 검증의 살아있는 테스트 |
| 지오메트리 | 기존 pirate gold sphere 모델 재사용 | 별도 asset 0개, 기존 height texture 재활용 |
| 두께 소스 | Texture (height tex 리매핑) + Procedural (gravity + noise), 토글 | 둘 비교로 두께 분포가 색에 미치는 효과 학습 |
| 투명도 | Forward + alpha blend, alpha = `Fresnel · alphaScale` | 추가 패스 없이 진짜 비눗방울 룩 |
| 굴절 처리 | 안 함 | 박막은 1μm 미만이라 굴절 offset이 sub-pixel |
| 애니메이션 | 토글 가능한 time 변조 | 정적/동적 둘 다 관찰 가능, 비용 거의 0 |

---

## 1. 아키텍처 & 파일 레이아웃

### 변경/추가 파일

```
src/base/
  vgeu_ibl.hpp              ← NEW (IBLBaker, Skybox)
  vgeu_ibl.cpp              ← NEW (~700줄, pbr.cpp에서 추출)
  CMakeLists.txt            ← MODIFY (소스 등록)

src/examples/pbr/
  pbr.hpp                   ← MODIFY (IBL 멤버를 IBLBaker/Skybox로 교체)
  pbr.cpp                   ← MODIFY (IBL 함수들 제거, IBLBaker 호출로 대체)

src/examples/soap_bubble/   ← NEW (CMake 자동 발견)
  soap_bubble.hpp           ← NEW
  soap_bubble.cpp           ← NEW (~700~900줄)

shaders/common/             ← NEW (공용 셰이더)
  equirect.vert/.frag       ← MOVE (pbr/에서 이동)
  prefilter.frag            ← MOVE
  irradiance.frag           ← MOVE
  brdf_lut.vert/.frag       ← MOVE
  skybox.vert/.frag         ← MOVE

shaders/soap_bubble/        ← NEW
  bubble.vert               ← NEW (단순 MVP, 월드 위치/노말/UV 패스)
  bubble.frag               ← NEW (박막 간섭 + 분광 적분 + Fresnel)
```

### 작업 단위 (PR 또는 commit 후보)

1. **IBL 추출 + PBR 마이그레이션**: `vgeu_ibl` 작성 + 셰이더를 `shaders/common/`로 이동 + PBR이 사용하도록 수정. PBR 회귀 없으면 통과.
2. **soap_bubble example 작성**: 1단계 완료 후 새 example 디렉토리 추가, `vgeu_ibl` 의존.

### CMake

- `src/base/CMakeLists.txt`에 `vgeu_ibl.cpp` 등록만 추가
- examples는 `src/examples/CMakeLists.txt`가 자동 발견 (디렉토리만 만들면 됨)
- 셰이더는 `shaders/**/*.frag` glob이라 `shaders/common/`, `shaders/soap_bubble/` 모두 자동

### 셰이더 경로 정책

`vgeu_ibl`은 파일 시스템 가정 안 함. caller가 `commonShadersPath`를 전달 (`getShadersPath() + "/common"` 같은 식). PBR도 `pbr/` → `common/`로 경로 갱신.

---

## 2. IBL 모듈 인터페이스 (`vgeu_ibl`)

두 클래스로 분리: `IBLBaker` (한 번 굽기 + 가끔 재굽기) / `Skybox` (매 프레임 그리기).

```cpp
namespace vgeu {

struct IBLBakeConfig {
  std::string hdrPath;            // 풀 경로
  std::string commonShadersPath;  // shaders/common/ 위치 (caller 결정)
  uint32_t envCubemapSize    = 512;
  uint32_t irradianceSize    = 64;
  uint32_t prefilteredSize   = 128;
  uint32_t brdfLutSize       = 512;
  uint32_t irradianceSamples = 2048;
  uint32_t prefilteredSamples= 1024;
  bool     useJitter         = true;
};

class IBLBaker {
 public:
  IBLBaker(const vk::raii::Device&, VmaAllocator,
           const vk::raii::Queue& transferQueue,
           const vk::raii::CommandPool&);

  // 전체 파이프라인: HDR → env cubemap → irradiance → prefiltered → BRDF LUT
  void bake(const IBLBakeConfig&);

  // ImGui jitter 토글 등 — irradiance + prefiltered만 다시
  void rebakeFiltering(const IBLBakeConfig&);

  // Descriptor binding용 const accessor (소유권 IBLBaker 유지)
  const VgeuImage&         envCubemap()    const;
  const VgeuImage&         irradianceMap() const;
  const VgeuImage&         prefilteredMap()const;
  const VgeuImage&         brdfLut()       const;
  const vk::raii::Sampler& iblSampler()    const;
  const vk::raii::Sampler& hdrSampler()    const;
};

class Skybox {
 public:
  Skybox(const vk::raii::Device&, const vk::raii::PipelineCache&,
         const vk::raii::DescriptorPool&, const vk::raii::RenderPass&,
         const std::string& commonShadersPath,
         const IBLBaker&,                // env cubemap 바인딩
         uint32_t maxFramesInFlight);

  void draw(const vk::raii::CommandBuffer&,
            const glm::mat4& view, const glm::mat4& proj,
            float lod = 0.0f);          // push constants
};

} // namespace vgeu
```

### 설계 포인트

- `IBLBakeConfig`로 모든 튜닝 파라미터 한 군데. 호출자(=example)가 `useJitter`, 해상도, 샘플 수를 조절. PBR은 ImGui jitter 변경 시 새 config로 `rebakeFiltering()` 호출.
- accessor는 const 참조만 노출. caller는 imageView/sampler를 받아 자기 descriptor에 binding.
- `Skybox::draw`는 view/proj/lod를 직접 받음 → push constants로 전달, 내부 UBO 안 만들고 단순화.
- 예외 정책: `bake()` 안에서 HDR 로드/셰이더 컴파일 실패 시 `std::runtime_error` throw. 호출 측 try-catch 안 함 (example 코드).

---

## 3. 비눗방울 셰이더

### 박막 간섭 수식 (per wavelength)

일반화된 단일 박막 (외부 n₁, 박막 n₂, 내부 n₃, 두께 d, 외부 입사각 θ₁):

```
sin θ₂ = (n₁/n₂) · sin θ₁          // Snell
δ₁ = π if n₁ < n₂ else 0           // 외부 경계 위상 점프
δ₂ = π if n₂ < n₃ else 0           // 내부 경계
Δφ = δ₁ − δ₂

φ(λ) = 4π · n₂ · d · cos θ₂ / λ    // OPD를 위상으로
r₁  = Schlick(cos θ₁, n₁, n₂)      // 외부 경계 Fresnel
r₂  = Schlick(cos θ₂, n₂, n₃)      // 내부 경계 Fresnel

R(λ) = r₁² + r₂² + 2 · r₁ · r₂ · cos(φ + Δφ)
```

비눗방울 기본값(n₁ = n₃ = 1.0, n₂ = 1.33)에서는 Δφ = π라 익숙한 `R = 4r² · sin²(φ/2)`로 환원되며, d = 0에서 R = 0 (검은 박막) — 위상 점프 처리가 정확하다는 sanity 포인트.

### 분광 적분

- λ를 380~780 nm에서 N개 균일 샘플링 (N ∈ {8, 16, 32, 64}, ImGui 콤보)
- 각 λᵢ에서 R(λᵢ) × CMF(λᵢ) 누적해 XYZ 만든 뒤 sRGB(D65)로 변환
- CMF는 **Wyman 2013 analytical fit**, 가우시안 합 GLSL ~10줄, LUT 없음
- 조명 SPD는 flat (E(λ) = 1), IBL이 broadband 컬러 공급 → `finalColor = thinFilmRGB · envColor`

XYZ → sRGB 변환:
```
M = mat3( 3.2406, -0.9689,  0.0557,
         -1.5372,  1.8758, -0.2040,
         -0.4986,  0.0415,  1.0570)
sRGB = max(M · XYZ, 0)
```

### 두께 함수 `thicknessAt(uv, worldPos, normal)`

ImGui 모드 토글로 분기:

- **Mode 0 (Texture)**: pirate gold height texture 샘플링, `mix(thicknessMin, thicknessMax, h)`로 nm 리매핑. 애니메이션 on이면 `uv += vec2(driftSpeed · time, 0)` 스크롤.
- **Mode 1 (Procedural)**: `gravity = clamp(0.5 + worldPos.y · gravityStrength, 0, 1)` (위쪽 얇음) + 3D value noise (`worldPos · noiseScale + time · driftSpeed`). `mix(gravity, noise, 0.5)`. 애니메이션 off면 time = 0 고정.

### 환경 반사

- 거울 반사 벡터 `R = reflect(−V, N)`로 `prefilteredCubemap` 샘플링 (envCubemap은 bubble shader에 binding 안 함 — Skybox만 사용)
- `roughness` 슬라이더(0~1) → LOD 변환 → 매끈/뿌연 표면. 0 ≈ 거울 (LOD 0), 1 = max LOD

### Fresnel 알파

- 외부 경계 Schlick(cos θ₁, n₁, n₂)을 알파로 (R 평균과 거의 비례, 계산 1회로 끝남)
- `alpha = clamp(fresnel · alphaScale, 0, 1)`, `alphaScale` 슬라이더 0~3

### 셰이더 입출력 요약

| 항목 | 값 |
|---|---|
| `set=0` | Globals UBO (view, proj, viewPos) |
| `set=1` | BubbleParams UBO (모든 슬라이더 값) |
| `set=2` | heightTex (sampler2D) — pirate gold |
| `set=3` | prefilteredCubemap (binding=0). LOD 0 = 거울 반사, roughness 슬라이더로 LOD 증가 |
| Output | `vec4(color, alpha)`, alpha blend, depth test on / write off, **front-face only** |

`gl_FragColor.rgb`는 premultiplied 아님 (블렌드 식: `srcAlpha · src + (1-srcAlpha) · dst`).

---

## 4. ImGui 파라미터

ImGui 컬랩서블 그룹. `Options` 구조체 양방향 바인딩, CLI11로 `--<name>` 옵션 노출.

### Group 1: Thin Film
| 파라미터 | 타입 | 범위 | 기본값 |
|---|---|---|---|
| `thicknessMin` | float (nm) | 0 ~ 2000 | 200 |
| `thicknessMax` | float (nm) | 0 ~ 2000 | 800 |
| `n1` (outside) | float | 1.0 ~ 2.5 | 1.0 |
| `n2` (film) | float | 1.0 ~ 2.5 | 1.33 |
| `n3` (inside) | float | 1.0 ~ 2.5 | 1.0 |
| `spectralSamples` | combo | {8, 16, 32, 64} | 16 |

### Group 2: Thickness Source
| 파라미터 | 타입 | 비고 |
|---|---|---|
| `thicknessMode` | radio | 0=Texture / 1=Procedural |
| `gravityStrength` | float, 0~5 | Procedural 모드만 |
| `noiseScale` | float, 0.1~10 | Procedural 모드만 |

### Group 3: Animation
| 파라미터 | 타입 | 비고 |
|---|---|---|
| `useAnimation` | checkbox | 토글 |
| `driftSpeed` | float, 0~2 | 토글 on일 때만 의미 |

### Group 4: Surface & Blending
| 파라미터 | 타입 | 범위 |
|---|---|---|
| `roughness` | float | 0~1 (LOD on prefilteredCubemap) |
| `alphaScale` | float | 0~3 (Fresnel × alphaScale) |

### Group 5: Environment / IBL
| 파라미터 | 타입 | 비고 |
|---|---|---|
| `iblExposure` | float, 0~10 | env 컬러 곱셈 |
| `iblGamma` | float, 1.0~3.0 | tone mapping |
| `useJitter` | checkbox | 토글 시 `IBLBaker::rebakeFiltering()` |
| `skyboxLod` | float, 0~max | env mip 디버그 |

### Group 6: Debug
| 항목 | 비고 |
|---|---|
| `cameraPos` | text (read-only) |
| `frameTimer` / FPS | base에서 이미 표시 |
| `showThicknessHeatmap` | checkbox — 두께(nm) → viridis 시각화 |
| `showFresnelOnly` | checkbox — Fresnel(cos θ₁) grayscale 출력 |

### CLI

PBR 동일 패턴:
```
soap_bubble --n2 1.45 --thicknessMin 300 --thicknessMax 600 --spectralSamples 32
```

### 인터랙션 정책

- 대부분 슬라이더는 즉시 반영 (셰이더가 매 프레임 UBO 읽음)
- `useJitter` 변경 시에만 `device.waitIdle()` + IBL 재굽기
- 카메라는 PBR과 동일 `KeyBoardMovementController` (마우스 우클릭 + WASD)

---

## 5. 검증 계획

### 5.1 PBR 회귀 검증 (IBL 추출 정확성)
추출 전후 PBR을 동일 설정으로 실행, 시각적으로 동일한지 확인.

- [ ] Skybox: 같은 HDR로 동일하게 렌더되는가
- [ ] Sphere grid: metallic/roughness 셀별 색상이 추출 전과 동일
- [ ] `useJitter` 토글: 두 상태 모두 추출 전과 동일
- [ ] `skyboxLod` 슬라이더: mip 레벨 슬라이딩이 동일
- [ ] Validation layer: 새로 추가된 VUID 에러 없음 (기존 VUID-02697은 별개)

회귀가 깨지면 비눗방울 작업 진입 전 무조건 해결. 비눗방울 디버깅하면서 추출 잘못된 걸 찾기는 시간 낭비.

### 5.2 비눗방울 셰이더 sanity check

| 입력 | 기대 결과 | 검증 의미 |
|---|---|---|
| `thicknessMin=0, thicknessMax=0` | 두께 0 → **검정** (φ=0, Δφ=π → R=0) | 위상 점프 정확 |
| `n1=n2=n3=1.0` | Fresnel=0 → 완전 투명, 색 없음 | 굴절률 분기 정확 |
| `n1=1.0, n2=2.4, n3=1.0` (다이아몬드 박막) | 매우 색이 진하고 채도 높음 | r 큰 케이스 |
| `thicknessMin=thicknessMax=550nm` | 고정된 단일 톤 색 | OPD 계산 정확 |
| `spectralSamples` 8 → 32 변경 | 더 부드러운 색 그라데이션 | 적분 수렴 |
| `alphaScale=0` | 비눗방울 완전 안 보임 | 알파 적용 정확 |
| 카메라 회전 | 같은 점 색이 미묘하게 변함 (cos θ 의존) | 시야각 의존성 |
| Procedural + `gravityStrength` 큰 값 | 위쪽 어둡고 아래쪽 색 두꺼움 | 두께 분포 |
| Animation 토글 | drift 보임 → 끄면 즉시 정적 | time uniform |

### 5.3 Vulkan validation 청결

- 모든 빌드/실행을 validation layer on (debug 빌드)으로 진행
- 새 VUID 에러 0개 목표 (PBR의 기존 VUID-02697은 별개 이슈로 남김)

### 5.4 빌드/플랫폼

- Windows 11 + MinGW (이 repo의 표준), validation on debug 빌드
- Linux/macOS는 범위 밖

### 5.5 성능 sanity

- 1280×1080, 단일 비눗방울 + skybox, `spectralSamples=32`에서 60 fps 이상. 안 되면 spectral 루프 unroll 또는 기본값 16으로 회귀.

### 5.6 코드 스타일

- `.cpp/.hpp` 변경은 `clang-format -i` 적용 후 commit (이 repo의 룰)
- IBLBaker 인터페이스는 PBR/soap_bubble 두 example에서 모두 사용되는 게 인터페이스 합리성의 살아있는 증거

---

## Out of Scope

- **굴절(refraction with screen-space sampling)**: 박막 두께가 1μm 미만이라 굴절 offset이 sub-pixel이라 효과 미미. 별도 PR/example로 분리.
- **Full transmission (T(λ) = 1 − R)**: 위와 함께 별도 PR.
- **여러 비눗방울 인스턴싱 / cluster**: 본 example의 "단일 모델 파라미터 탐색" 목적과 충돌. 후속 example로.
- **Compute 기반 두께 시뮬 (gravity drain dynamics)**: 별도 example급 작업.
- **Belcour-Barla 2017 closed-form spectral integration**: Wikipedia 이론 직접 구현이 본 example의 교육 목표.
- **Linux/macOS 빌드 검증**.

---

## 작업 순서 (writing-plans 단계로 넘기기 전 가이드)

1. `vgeu_ibl.hpp/.cpp` 작성 (PBR에서 IBL 함수 옮김, 셰이더 경로는 config로)
2. `shaders/common/` 디렉토리 만들고 PBR의 IBL 셰이더 6개 이동
3. PBR을 `IBLBaker` + `Skybox` 사용하도록 수정, 회귀 검증 (5.1)
4. `src/examples/soap_bubble/` 디렉토리 + skeleton (`vgeu_ibl` 사용, 단일 pirate gold 모델 로드, 빈 forward 패스)
5. `shaders/soap_bubble/bubble.vert/.frag` 작성, 박막 간섭 수식 + 분광 적분 + Wyman CMF + Fresnel 알파
6. ImGui 6 그룹 + Options + CLI11 옵션
7. Procedural 두께 모드 추가 (gravity + value noise)
8. Animation 토글
9. 디버그 시각화 (heatmap, fresnel-only)
10. sanity 표(5.2) 통과 확인, validation 청결 확인
