# Paint Splatter

## 개요

GPU **Position Based Fluids (PBF, Macklin & Müller 2013)** 기반의 물감 시뮬레이션.
스포이드(spoid)에서 떨어진 물감 방울이 3D로 낙하·충돌·튀어오르며, 바닥(캔버스)에
닿는 순간 색을 영구 누적 텍스처에 입힌다. 누적된 그림은 PNG로 저장할 수 있다.
컴퓨트 큐에서 PBF 솔버가 살아있는 입자를 굴리고, 그래픽스 큐에서 캔버스 텍스처와
입자를 렌더한다 (`particle`/`cloth` 예제의 컴퓨트↔그래픽스 동기화 패턴을 그대로 따름).

설계/이론 문서:
- 설계 스펙: `docs/superpowers/specs/2026-05-31-paint-splatter-pbf-design.md` (Phase 1),
  `docs/superpowers/specs/2026-06-03-paint-splatter-cohesion-crown-design.md` (M8)
- 디버깅 기록: `docs/paint_splatter_debug_log.md`
- 레퍼런스 분석(CPU PBF 대조): `docs/paint_splatter_pbf_reference_analysis.md`

## 핵심 개념

- **PBF (위치 기반 유체)** — 밀도 제약(비압축성)을 위치 보정으로 직접 푸는 방식.
  3D MAC 그리드/압력 투영 없이 GPU 친화적이며, `pbd`의 공간 해시 패턴을 컴퓨트로 포팅.
- **이중 레이어** — (L1) 살아있는 PBF 입자, (L2) 영구 캔버스 누적 텍스처(그림).
- **퇴적(deposit) + 건조(dry)** — 바닥 근처 입자가 색을 캔버스에 stamp하고, wetness를
  소진하면 "흡수"되어 GPU compaction으로 제거됨 → 입자 수가 유계로 유지된다.
- **월드 규약(고정)** — 이 엔진은 화면 위 = 월드 **−Y**로 렌더한다. 따라서 중력은
  **+Y**(화면 아래), 캔버스 바닥은 y=0의 X-Z 평면, 유체/스포이드는 y<0에 산다.
- **응집 + 크라운 (M8)** — 방울이 낙하 중 뭉쳐 있다가 충돌 시 radial하게 퍼지도록
  한 튜닝 마일스톤. CPU 레퍼런스 대조로 파라미터를 보정했다(아래 "M8" 절).
- **연속 분사 stream (M8)** — 이동하는 스포이드에서 매 프레임 연속 분사 + 경로 sweep으로
  빠른 스트로크도 점이 아닌 끊김 없는 줄로 그려진다.

## 데이터 구조

```cpp
// 살아있는 입자 (std430, 64B). pos.w = wetness(건조/제거 신호), vel.w = rho/rho0(디버그)
struct Particle { glm::vec4 pos, vel, predict, color; };

// 모든 PBF 컴퓨트 패스가 공유하는 UBO (std140, 144B)
struct ComputeUbo {
  float dt; uint particleCount; float gravity, h;      // h = smoothing radius = 그리드 셀
  glm::vec4 canvasMin, canvasMax; glm::ivec4 gridDim;  // 도메인 박스 + 그리드
  float rho0, epsCFM, scorrK, scorrDq, scorrN, xsphC;  // PBF 상수
  float kPoly6, kSpiky, scorrDenom;                    // 커널 상수(호스트 계산)
  float velDamp, velClampFactor, solverRelax;          // 안정화
  uint  prevCount; float dryRate, depositStrength, depositHeight, depositRadius, drySettle;
  float cohesionFloor;  // M8: 경계 있는 부호 제약 C=max(rho/rho0-1, -cohesionFloor)
  float dpClampFactor;  // M8: Δp clamp 배수(<=0이면 끔)
};

// 스포이드(호스트 POD). holeRadius = ball 분사 반경(=방울 크기), amount = 버스트 입자수
struct Spoid { glm::vec3 pos; float holeRadius; glm::vec3 color;
               float emissionVelocity; int amount; float concentration; bool selected;
               glm::vec3 prevPos; float emitAccum; /* stream 런타임 상태 */ };
```

GPU 공유 구조체는 전부 `static_assert(sizeof == N)`로 std140/std430 크기를 못박는다.

## 컴퓨트 파이프라인 (PBF Algorithm 1)

프레임당: `predict(압축)` → `emit` → [substep N회: `integrate → grid → solve iters → finalize`].

| 파이프라인 / 셰이더 | 역할 |
|---|---|
| `pbf_predict` | **compaction**: 이전 프레임(ping-pong) 생존 입자를 atomicAdd로 앞으로 압축, 건조(pos.w≤0) 입자 제거. 프레임당 1회 |
| `emit` | 스포이드 버스트/스트림을 ball 또는 lattice로 분사 (atomicAdd로 슬롯 확보) |
| `integrate` | **substep마다** 중력 적분 + x* = x + v·dt (M8: predict에서 분리해 진짜 substep을 가능케 함) |
| `grid_count / grid_scan / grid_scatter` | 균일 그리드 이웃 탐색 (counting sort, `pbd` 부분합 패턴) |
| `pbf_lambda` | 밀도 제약 λ (Eq.11). M8: `C=max(rho/rho0-1, -cohesionFloor)` 경계 있는 부호 제약 |
| `pbf_delta` | Δp + scorr(표면장력) + 충돌. y만 clamp(바닥/천장), x/z 자유 |
| `pbf_apply` | x* += Δp (Jacobi 분리 적용) |
| `pbf_finalize` | v=(x*−x)/dt, XSPH 점성, 건조, 캔버스 밖 착지 입자 cull, pos 커밋 |
| `deposit` | 바닥 근처 입자 색을 캔버스 이미지에 alpha-over stamp (**그래픽스 큐**에서 dispatch) |

라이브 카운트(`liveCount`)는 컴퓨트 안에서만 읽고/쓰며, ping-pong된 prev 버퍼의 정확한
범위는 GPU 카운터(binding 10)로 읽어 readback 지연과 무관하게 정확하다.

## 셰이더

| 파일 | 역할 |
|---|---|
| `canvas.vert/frag` | 캔버스 쿼드 + 누적 텍스처 샘플 |
| `particle.vert/frag` | 입자 포인트/디스크 렌더 |
| `marker.vert/frag` | 스포이드 마커 |
| `pbf_predict / integrate / grid_* / pbf_lambda / pbf_delta / pbf_apply / pbf_finalize` | PBF 솔버 |
| `emit` | 분사(ball sweep 포함) |
| `deposit` | 캔버스 퇴적 |

## 동기화 구조

```
Compute Queue ──semaphore──► Graphics Queue
     ▲                            │
     └────────semaphore───────────┘
```

- 입자 SSBO는 컴퓨트↔그래픽스 큐 소유권 ping-pong에 참여 (`particle.cpp` 패턴).
- `deposit`는 캔버스 이미지가 그래픽스 소유로 남도록 **그래픽스 큐**에서 실행(이미지 QFOT 회피).
- 디버그 readback 복사는 그래픽스 커맨드 버퍼 안에서 기록해 release/acquire 짝을 깨지 않음.

## M8 — 응집 방울 + 크라운 + 연속 분사

CPU 레퍼런스(`yuki-koyama/position-based-fluids`)와 대조해 커널 수식이 정확함을 확인하고,
파라미터 스케일을 보정한 마일스톤. 핵심:

- **rho0 ↔ emit 간격 일치**: `kParticleSpacing=0.03`(=0.3h)로 방울이 rest density 근처에서
  분사되어 비압축성+표면장력이 살아남. (예전 0.005는 rho0를 8e6으로 띄워 모든 힘을 0으로 만듦)
- **soft constraint**: `epsCFM≈1e5`가 분모를 압도 → λ가 작아 부드럽고 안정 (충돌/과속 폭발 제거).
- **경계 있는 부호 제약**: `cohesionFloor`가 under-dense 입자를 rest density로 당겨 **낙하 중 응집**.
- **XPBD small-steps**: `integrate`를 substep으로 분리, `substeps≈3 / solverIters≈2`,
  `velDamp≈0.36`(레퍼런스 v*=0.999/substep). Δp/CFL clamp는 기본 약하게(크라운 rebound 보존).
- **ball 분사**: `holeRadius`가 방울 반경. **stream 모드**: 매 프레임 `streamRate`만큼 분사 +
  이동 경로 sweep으로 빠른 스트로크 연결. **캔버스 밖**: 입자는 벽에 안 갇히고 자유낙하 후
  캔버스 밖 바닥에 닿으면 제거.

## Phase 2 — PBD 진자 구동 스포이드 + 탑뷰

스포이드를 **n-링크 PBD 진자**(기본 더블 펜듈럼)의 운동에 매달아, 흔들림 × 회전
오프셋으로 하모노그래프/스피로그래프 패턴을 그리는 마일스톤. **호스트 사이드 전용**:
컴퓨트 패스/UBO/GPU 구조체 변경 없이 컨트롤러가 매 프레임 `spoid.pos`만 갱신하고,
기존 PBF/emit/stream/deposit/PNG 경로가 그 값을 그대로 재사용한다. 추가 셰이더는
체인 라인용 `chain_line.{vert,frag}` 둘뿐.

- **PendulumSpoidController**: 키보드 모드와 토글(라디오 버튼). 펜듈럼 모드에서 각
  스포이드는 체인의 한 노드(기본=tip)에 붙어 emit 지점을 따라간다. Phase-1 스포이드
  위치 clamp는 펜듈럼 모드에서 면제(진자 물리가 위치를 소유).
- **PBD 스텝**: predict → 거리 제약(Gauss-Seidel, per-node 역질량) → 속도 갱신 →
  air/joint 감쇠, substep + CFL 속도캡 + 바닥 위 soft-clamp. **순수 PBD**(해석해 폴백
  없음). 노드별 질량 `bobMass<=0` ⇒ 역질량 0 ⇒ **해당 노드 고정(pin)**.
- **회전 오프셋 `(r, angle₀, ω)`**: 부착 노드에서 줄 방향에 수직인 평면 위로 `r`만큼
  떨어진 점을 `ω`로 회전(특이점 가드 포함). 흔들림 × 회전 = 스피로그래프.
- **paintMass 저수조**: stream 분사가 입자당 선형 소모(기본 `kDrain=1e-4`), 0이면 분사
  중단. UI `refill paint`로 리셋.
- **체인 시각화**: 링크는 새 `eLineList` 파이프라인(회색 선), 조인트는 `markerPipeline`
  재사용(노란 점).
- **탑뷰 카메라**: 오비트 카메라는 그대로 두고, 버튼이 현재 시점에서 머리 위
  부감으로 부드럽게(smoothstep) 애니메이션 후 고정. 다시 누르면 오비트로 복귀.
  Save PNG는 카메라와 무관.
- **월드 스케일**: `world scale`(1~4) 노브가 캔버스 + 시뮬 도메인 + 이웃 그리드를
  배율(입자 크기/h/rho0/spacing은 불변)로 키운다. Restart 시 적용(그리드 버퍼는 최대
  배율로 미리 할당 → 재할당/디스크립터 재작성 없이 gridDim/numCells만 갱신). Restart는
  펜듈럼도 초기 상태로 되돌린다.

## 주요 설정 (ImGui 라이브)

| 옵션 | 설명 |
|---|---|
| `substeps` / `solverIters` | XPBD 스텝/반복 (안정 ↔ 비용). 권장 3~5 / 2 |
| `epsCFM` | 제약 부드러움. 높을수록(1e5) 안정, 낮을수록(1e3) 단단/크라운 강함 |
| `cohesion floor` | 낙하 중 응집 세기(0=압축 전용, ~1=레퍼런스의 무경계 부호 제약) |
| `dp clamp` / `vel damping` | Δp 상한(0=끔) / 전역 감쇠. 낮을수록 크라운 강함 |
| `stream mode` / `stream rate` | 연속 분사 토글 / 초당 입자수(=stream의 양) |
| `spherical spawn` / `hole radius` | ball 분사 토글 / 방울 반경 |
| `amount (burst)` / `emission vel` / `concentration` / `color` | 버스트 입자수 / 분사속도 / 농도 / 색 |
| `deposit strength` / `dry rate` / `deposit radius` | 퇴적 알파 / 건조 속도 / stamp 반경 |
| `keyboard` / `pendulum` | 스포이드 제어 모드 토글 (Phase 2) |
| `links (n)` / `total length` / `pivot` | 진자 링크 수 / 전체 길이 / 매단 점 |
| `init theta/phi/speed` | 초기 각도(수직 +Y 기준)·방위·tip 접선속도 |
| `air/joint damping` / `substeps` / `constraint iters` | 공기·조인트 감쇠 / PBD substep / 제약 반복 |
| `bob N mass (<=0 pin)` | 노드별 질량(0 이하면 해당 노드 고정) |
| `offset r` / `offset angle0` / `offset omega` | 회전 오프셋 반경 / 시작각 / 각속도(스피로그래프) |
| `paint mass` / `refill paint` | 스포이드 저수조 잔량 / 리필 |
| `world scale` | 월드 배율(1~4, Restart 적용). 입자 크기는 불변 |
| Top view / Free camera | 머리 위 부감으로 부드럽게 전환 / 오비트 복귀 |
| Save PNG | 캔버스를 `build/paint_<timestamp>.png`로 저장 |

## 참고 문헌

- Macklin, Müller. *Position Based Fluids.* ACM TOG 32(4), 2013.
- Müller et al. *Position Based Dynamics.* 2007.
- Macklin et al. *Small Steps in Physics Simulation.* SCA 2019 (substep > iteration).
- 프로젝트 선례: `particle`(컴퓨트 멀티 파이프라인+동기화), `pbd`(공간 해시), `soap_bubble`(오프스크린 캡처).
