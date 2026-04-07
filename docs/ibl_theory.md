# IBL (Image-Based Lighting) 이론 정리

**참고:** https://learnopengl.com/PBR/IBL/Diffuse-irradiance  
**참고:** https://learnopengl.com/PBR/IBL/Specular-IBL

---

## 개요

IBL은 주변 환경 전체를 하나의 광원으로 취급하는 기법이다. HDR 환경맵에서 간접광(ambient)을 사전계산해 PBR 셰이더에 적용하면 훨씬 사실적인 결과를 얻을 수 있다.

반사 방정식에서 diffuse와 specular를 분리하면:

```
Lo(p,ωo) = ∫Ω (kD·c/π + kS·DFG/(4·(ωo·n)·(ωi·n))) · Li(p,ωi) · (n·ωi) dωi
```

- **Diffuse 항**: 환경의 irradiance를 사전적분 → irradiance 큐브맵
- **Specular 항**: Split-sum 근사로 두 텍스처에 분리 → pre-filtered 큐브맵 + BRDF LUT

---

## 1단계: HDR 환경맵 로딩

HDR(.hdr) 파일은 `stbi_loadf()`로 부동소수점 픽셀로 로딩한다. 등장방형(equirectangular) 투영 형식이며 큐브맵으로 변환이 필요하다.

### 등장방형 → 큐브맵 변환

큐브의 6면을 각각 렌더타겟으로 사용해 90도 FOV로 6방향을 렌더링한다.

```glsl
// equirect.frag — 방향 벡터 → UV 변환
const vec2 invAtan = vec2(0.1591, 0.3183);
vec2 SampleSphericalMap(vec3 v) {
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= invAtan;
    uv += 0.5;
    return uv;
}
void main() {
    vec2 uv = SampleSphericalMap(normalize(localPos));
    FragColor = vec4(texture(equirectangularMap, uv).rgb, 1.0);
}
```

캡처용 projection/view:
```cpp
glm::mat4 captureProjection = glm::perspective(glm::radians(90.f), 1.f, 0.1f, 10.f);
// 6방향 회전 행렬로 각 면 캡처 (glm::rotate 조합)
```

---

## 2단계: Diffuse Irradiance 맵

### 이론

Diffuse 반사는 입사광 방향에 독립적이므로, 법선 방향 N에 대해 반구 전체의 빛을 적분한 결과를 사전계산할 수 있다.

```
irradiance(N) = ∫₀²π ∫₀^(π/2) Li(ωi) · cos(θ) · sin(θ) dθ dφ
```

`sin(θ)` 가중치는 극점 근처 샘플 면적이 작아지는 것을 보정한다.

### 셰이더 구현

```glsl
// irradiance.frag
void main() {
    vec3 normal = normalize(localPos);
    vec3 irradiance = vec3(0.0);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, normal));
    up = normalize(cross(normal, right));

    float sampleDelta = 0.025;
    float nrSamples = 0.0;
    for (float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta) {
        for (float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta) {
            vec3 tangentSample = vec3(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta));
            vec3 sampleVec = tangentSample.x * right
                           + tangentSample.y * up
                           + tangentSample.z * normal;
            irradiance += texture(environmentMap, sampleVec).rgb
                        * cos(theta) * sin(theta);
            nrSamples++;
        }
    }
    irradiance = PI * irradiance / float(nrSamples);
    FragColor = vec4(irradiance, 1.0);
}
```

- 해상도: **64×64** (고주파 성분이 없어 저해상도로 충분)
- 출력: 큐브맵 (arrayLayers=6, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)

### PBR 셰이더 적용

```glsl
vec3 kS = FresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
vec3 kD = (1.0 - kS) * (1.0 - metallic);
vec3 irradiance = texture(irradianceMap, N).rgb;
vec3 diffuse = irradiance * albedo;
vec3 ambient = kD * diffuse * ao;
```

roughness를 반영한 Fresnel:
```glsl
vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    return F0 + (max(vec3(1.0 - roughness), F0) - F0)
           * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
```

---

## 3단계: Specular IBL — Split-Sum 근사

Specular 항은 뷰 방향 의존성 때문에 직접 사전계산이 불가능하다. Epic Games의 Split-Sum 근사로 두 부분으로 분리한다:

```
∫ Li(ωi)·f(ωi,ωo)·cos(θ) dωi ≈ [∫ Li(ωi) dωi] · [∫ f(ωi,ωo)·cos(θ) dωi]
                                    ↑                  ↑
                          pre-filtered 큐브맵         BRDF LUT
```

### 3-1: Pre-filtered 환경 큐브맵

roughness별로 GGX importance sampling으로 환경을 컨볼루션한다. roughness 0~1을 mip 레벨 0~(numMips-1)에 저장.

**Hammersley 저불일치 수열:**
```glsl
float RadicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}
vec2 Hammersley(uint i, uint N) {
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
}
```

**GGX Importance Sampling:**
```glsl
vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    vec3 H = vec3(cos(phi)*sinTheta, sin(phi)*sinTheta, cosTheta);
    vec3 up = abs(N.z) < 0.999 ? vec3(0,0,1) : vec3(1,0,0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}
```

- 해상도: **512×512**, mip 레벨 수 = `floor(log2(512)) + 1 = 10`
- 샘플 수: 32 (push constant로 전달)
- 포맷: `R16G16B16A16_SFLOAT`

### 3-2: BRDF LUT

NdotV와 roughness를 입력으로 Fresnel scale(R)과 bias(G)를 출력하는 512×512 2D 텍스처.

```glsl
// brdf_lut.frag
vec2 IntegrateBRDF(float NdotV, float roughness) {
    vec3 V = vec3(sqrt(1.0 - NdotV*NdotV), 0.0, NdotV);
    float A = 0.0, B = 0.0;
    vec3 N = vec3(0.0, 0.0, 1.0);
    const uint SAMPLE_COUNT = 1024u;
    for (uint i = 0u; i < SAMPLE_COUNT; ++i) {
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 H = ImportanceSampleGGX(Xi, N, roughness);
        vec3 L = normalize(2.0 * dot(V, H) * H - V);
        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);
        if (NdotL > 0.0) {
            float G = GeometrySmith_IBL(N, V, L, roughness);
            float G_Vis = (G * VdotH) / (NdotH * NdotV);
            float Fc = pow(1.0 - VdotH, 5.0);
            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    return vec2(A, B) / float(SAMPLE_COUNT);
}
```

Geometry 함수는 IBL용 k = roughness²/2 사용 (direct lighting의 k = (roughness+1)²/8과 다름):
```glsl
float GeometrySchlickGGX_IBL(float NdotV, float roughness) {
    float a = roughness;
    float k = (a * a) / 2.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}
```

- 포맷: `R16G16_SFLOAT`
- 해상도: **512×512**

### Specular PBR 셰이더 적용

```glsl
const float MAX_REFLECTION_LOD = float(numMips - 1);  // 9.0
vec3 R = reflect(-V, N);
vec3 prefilteredColor =
    textureLod(prefilteredMap, R, roughness * MAX_REFLECTION_LOD).rgb;
vec2 envBRDF =
    texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
vec3 specular = prefilteredColor * (F * envBRDF.x + envBRDF.y);

vec3 ambient = (kD * diffuse + specular) * ao;
```

---

## 구현 시 주의사항

1. **Vulkan cubemap seamless filtering**: `VkPhysicalDeviceFeatures::imageCubeArray` 및 `shaderSampledImageArrayDynamicIndexing` 확인
2. **큐브맵 생성 패턴**: framebuffer에 face를 직접 붙이는 대신, **offscreen 2D image에 렌더 → vkCmdCopyImage** 방식이 안전하다 (Sascha Willems 방식)
3. **irradiance mip**: 큐브맵 샘플러의 `minLod/maxLod` 설정으로 전체 mip chain 샘플링 가능하게 할 것
4. **BRDF LUT 경계**: `addressMode = CLAMP_TO_EDGE` 필수 (엣지 아티팩트 방지)
5. **sRGB 주의**: HDR 환경맵은 linear 공간. albedo만 sRGB→linear 변환 필요, IBL 텍스처는 변환 불필요
