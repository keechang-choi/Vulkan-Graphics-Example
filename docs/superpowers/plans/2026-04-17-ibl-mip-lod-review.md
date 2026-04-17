# IBL PDF-based Mip LOD 검증 및 개선

**날짜**: 2026-04-17
**선행 문서**: `2026-04-11-ibl-cubemap-fix.md`
**배경**: jitter + 1024 samples 적용 후에도 구 표면 aliasing이 미세하게 남아있음.
PDF-based mip filtering이 기준 문서대로 정확히 적용되었는지 GPU Gems 3로 교차 검증.

## 참고 자료
- NVIDIA GPU Gems 3, Ch.20 "GPU-Based Importance Sampling"
  https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
- Chetan Jags, "Image Based Lighting" (2015)
- LearnOpenGL, "Specular IBL"

## GPU Gems 3 핵심 공식 (원문)

- **Eq 11**: `Ωs = 1 / (p(u,v) · N)` — sample 당 solid angle
- **Eq 12**: `Ωp = 4π · d(u) / (w · h)` — texel 당 solid angle (d는 매핑 왜곡 인자)
- **Eq 13**: `level = 0.5 · log₂(Ωs / Ωp)`

**Bias 권장 (원문 인용)**:
> "In practice, we found that introducing small amounts of overlap between samples
> (for example, biasing Ωs/Ωp by a constant) produces smoother, more visually
> acceptable results. Specifically, we perform the bias by adding 1 to the
> calculated mipmap level defined by Equation 13."

## 현 shader 대조

`shaders/pbr/irradiance.frag`, `shaders/pbr/prefilter.frag`:
```glsl
float saTexel  = 4.0 * PI / (6.0 * envResolution * envResolution);   // Ωp
float saSample = 1.0 / (float(N) * pdf + 0.0001);                    // Ωs
float mipLevel = 0.5 * log2(saSample / saTexel);                     // Eq 13
```

| 항목 | GPU Gems 3 | 현 shader | 평가 |
|------|-----------|-----------|------|
| `Ωs = 1/(pN)` | ✓ | ✓ | 일치 |
| `0.5·log₂(Ωs/Ωp)` | ✓ | ✓ | 일치 |
| Cubemap Ωp: `4π/(6wh)` | `d(u)` 포함 | `d(u)=1` 근사 | 표준 단순화 (LearnOpenGL/UE4/Chetan 동일) |
| **+1 LOD bias** | **권장** | **없음** | ★ 누락 |

**검증 결과**: 기본 공식은 정확히 적용되었음. 다만 원문이 명시적으로 권장한
"+1 bias"가 누락.

## 남은 aliasing과 관련된 쟁점

**A. `+1` LOD bias 누락 (이번에 실험)**
- GPU Gems 원저자가 "smoother, more visually acceptable results"라고 명시 권장
- 효과: 한 mip level 더 내려가서 4× 더 많은 texel을 평균 → 샘플 간 overlap
- Chetan Jags/LearnOpenGL에는 없음 (유래: GPU Gems 3 원문)
- **trade-off**: sharpness 소폭 감소 vs aliasing 감소

**B. Cubemap 분포 왜곡 `d(u)` 무시**
- Face 중심 vs 모서리에서 실제 texel solid angle 최대 2× 차이
- 엄밀: `d(u) = 1/((face_u² + face_v² + 1)^1.5)`
- 표준 관행에서 무시 — aliasing 주원인일 가능성 낮음

**C. Prefilter `roughness == 0` 분기가 mipLevel=0 강제**
- 모든 sample L=R로 수렴 → env mip 0 그대로 → HDR bright 픽셀 직접 사용
- GPU Gems formulation이 roughness=0에서 p→∞라 무너짐 → 별도 처리 필요
- 대안:
  1. `roughness = max(roughness, 0.04)` floor (GGX 계산 시)
  2. prefilteredMap mip 0 = envCubemap mip 0 직접 copy

## 실험 계획

**이번 커밋 (A: +1 bias)**:
1. `irradiance.frag`: `mipLevel = max(0.5 * log2(saSample/saTexel) + 1.0, 0.0);`
2. `prefilter.frag`: `mipLevel = push.roughness == 0.0 ? 0.0 : 0.5 * log2(saSample/saTexel) + 1.0;`

결과 확인 후:
- 개선 없음 → C(roughness=0 분기) 시도
- 개선 + 너무 흐려짐 → +1 → +0.5로 조정
- 개선 + 만족 → 다음 단계(SH irradiance 또는 일반 feature)로 이동
