# PBR Example — Bugfix 기록

PBR deferred rendering 예제(`src/examples/pbr/`) 구현 과정에서 발생한 버그들과 수정 내용 정리.

---

## 1. Vulkan pushConstants RAII 호출 오류

**Commit:** `eed9bd6`

**증상:** 컴파일 에러 또는 런타임 crash.

**원인:** `vk::raii` 환경에서 `pushConstants` 호출 시 non-RAII 형식 사용.

**수정:** Vulkan-Hpp RAII 템플릿 형식으로 교체.

---

## 2. DistributionGGX 분모 zero-division

**Commit:** `29bce89`

**증상:** roughness=0인 경우 specular 계산에서 NaN 또는 Inf 발생.

**원인:** `pbr.frag`의 GGX Normal Distribution 분모가 0이 될 수 있음.

**수정:** 분모에 `1e-7` epsilon 추가.

```glsl
// Before
return a2 / (PI * denom * denom);
// After
return a2 / (PI * denom * denom + 1e-7);
```

---

## 3. VUID-vkCmdDraw-None-02697 validation error

**Commit:** `e81e651`

**증상:** 실행 시 Vulkan validation error — composition pass에서 descriptor set이 이전 offscreen pass 것으로 바인딩된 채로 draw 호출.

**원인:** offscreen renderpass 이후 composition pass 시작 전 composition descriptor set을 rebind하지 않음.

**수정:** composition renderpass 시작 후 composition descriptor set을 명시적으로 rebind.

---

## 4. RenderFlags 타입 불일치

**Commit:** `5f76bdd`

**증상:** 컴파일 경고 또는 암묵적 변환 오류.

**원인:** empty render flags 자리에 정수 `0`을 직접 전달.

**수정:** `0` → `vgeu::RenderFlags{}` 로 교체.

---

## 5. VUID-vkCmdBindDescriptorSets-00358 — descriptor set layout mismatch (1차)

**Commit:** `227ca8b`

**증상:**
```
VUID-vkCmdBindDescriptorSets-pDescriptorSets-00358:
  bound set reports 1 descriptor, pipeline expected 4
```

**원인:** pipeline layout의 set 2를 생성할 때 `modelInstances[0].model->descriptorSetLayoutImage` (floor 모델 내부 layout, binding 1개)를 재사용함. sphere dummy descriptor set 할당도 같은 layout 사용 → binding 수 불일치.

**수정:** 전용 `sphereImageSetLayout` (4개의 `eFragment CombinedImageSampler` binding)을 별도 생성하고, pipeline layout 생성과 dummy descriptor set 할당 모두 이 layout 사용.

---

## 6. VUID-vkCmdBindDescriptorSets-00358 — ModelInstance sceneMode 누락 (2차, 근본 원인)

**Commit:** `4ec74a3`

**증상:** 위 5번 수정 후에도 동일한 validation error가 다른 VkDescriptorSet handle로 계속 발생. `sphereDummyDescriptorSet` handle과 에러에서 보이는 handle이 다름.

**원인 분석:**
- `addModelInstance(std::move(inst))` 호출 시 sphere 인스턴스의 `sceneMode`가 `kSphereOnly`로 설정되어 있었으나
- `ModelInstance`의 move constructor와 move assignment operator가 `sceneMode` 필드를 복사하지 않음
- move 후 모든 sphere 인스턴스가 default `kModelOnly`로 리셋
- `buildCommandBuffers`에서 sphere가 `kBindImages` 경로로 draw → sphere 모델의 1-binding material descriptor set이 바인딩 → pipeline layout(4-binding) 불일치

**수정:** move constructor/assignment에 `sceneMode = other.sceneMode;` 추가.

```cpp
ModelInstance::ModelInstance(ModelInstance&& other) {
  // ... 기존 필드들 ...
  sceneMode = other.sceneMode;  // 추가
}
ModelInstance& ModelInstance::operator=(ModelInstance&& other) {
  // ... 기존 필드들 ...
  sceneMode = other.sceneMode;  // 추가
  return *this;
}
```

---

## 7. Sphere albedo color picker 런타임 미적용

**Commit:** `df4f7f0`

**증상:** UI에서 "Sphere Albedo" color picker를 변경해도 화면에 반영되지 않음.

**원인:** `mrt.frag`에서 `color = mix(albedo.rgb, inColor.rgb, inColor.a)`로 모델 색상을 혼합한 뒤, 정작 G-buffer albedo 출력에는 원본 `albedo`를 그대로 씀.

```glsl
// Before (bug)
vec3 color = mix(albedo.rgb, inColor.rgb, inColor.a);
outAlbedo = albedo;  // mix 결과 버려짐

// After (fix)
outAlbedo = vec4(color, albedo.a);
```

---

## 8. Helmet/Floor에 붉은 tint 발생

**Commit:** `80448401`

**증상:** 위 7번 수정 후 helmet과 floor가 붉은 색으로 표시됨.

**원인:** `setupDynamicUbo()`에서 floor/helmet 인스턴스의 `modelColor`가 디버그용 `{1, 0, 0, 0.3f}` (30% 빨강 혼합)으로 설정된 채 남아 있었음. 7번 수정으로 mix()가 실제로 적용되면서 붉은 tint가 나타남.

**수정:** 비-sphere 인스턴스의 `modelColor` alpha를 0으로 설정 → mix()가 no-op이 됨.

```cpp
// Before (debug leftover)
dynamicUbo[idx].modelColor = glm::vec4{1.f, 0.f, 0.f, 0.3f};
// After
dynamicUbo[idx].modelColor = glm::vec4{0.f, 0.f, 0.f, 0.f};
```

---

## 9. Sphere ambient 완전히 꺼짐 + terminator 경계선

**Commit:** `ba25c9a`

**증상:**
- Sphere의 어두운 면이 완전히 검음 (ambient = 0)
- Sphere 표면에 lighting 경계선이 날카롭게 보임

**원인:**
- `sphereDummyMetRough` 텍스처를 `{0, 128, 0, 255}`로 생성 → R채널(AO) = 0
- `ambient = 0.03 * albedo * ao = 0.03 * albedo * 0 = 0`
- `pbr.frag`의 ambient가 `0.03` 하드코딩으로 UI에서 조절 불가

**수정:**
1. ARM dummy 텍스처 AO=255로 수정: `{255, 128, 0, 255}` (R=AO=1, G=roughness=0.5, B=metallic=0)
2. `ambientStrength`를 UBO에 추가, UI 슬라이더로 조절 가능하도록
3. specular 분모 epsilon 개선: `+ 0.0001` → `max(..., 0.001)` (terminator 근처 안정화)

```cpp
// pbr.cpp
sphereDummyMetRough = createDummyTexture({255, 128, 0, 255});
```

```glsl
// pbr.frag
vec3 ambient = vec3(ubo.ambientStrength) * albedo * ao;
```

---

## 10. Directional light 방향이 실행마다 랜덤

**Commit:** `576570a`

**증상:** directional light 켜면 실행마다 결과가 다름 — 아예 안 보이거나, 엉뚱한 방향에서만 비치거나.

**원인:** `ambientStrength` 필드(4 bytes)를 `UniformDataComposition`의 `useDirectionalLight`와 `_pad` 사이에 추가한 후, C++와 GLSL std140의 `dirLightDir` 오프셋이 어긋남.

| 위치 | `_pad` offset | `dirLightDir` offset |
|------|--------------|---------------------|
| C++ (`glm::vec2` align=4) | 364 | **372** |
| GLSL std140 (`vec2` align=8) | 368 | **384** |

GLSL이 offset 384에서 `dirLightDir`를 읽으려 하나, C++은 372에 썼음. 384는 C++ struct의 compiler padding 영역(초기화 안 된 쓰레기 값) → 랜덤 동작.

**수정:** `dirLightDir`에 `alignas(16)` 추가 → C++도 16-byte boundary(offset 384)에 강제 배치.

```cpp
// pbr.hpp
// Before
glm::vec4 dirLightDir;
// After
alignas(16) glm::vec4 dirLightDir;
```

**교훈:** GLSL uniform block에 필드를 추가할 때는 std140 alignment 규칙과 C++ struct layout이 항상 일치하는지 검증 필요. `vec2`/`vec4`/`vec3`는 std140에서 각각 8/16/16 byte alignment를 가지지만, C++ `glm::` 타입은 보통 4 byte alignment만 가짐. 중간에 scalar를 끼워넣으면 이후 필드의 오프셋이 달라질 수 있음.
