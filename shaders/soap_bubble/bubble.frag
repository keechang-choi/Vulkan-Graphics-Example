#version 450

layout(set = 0, binding = 0) uniform Globals {
  mat4 view;
  mat4 projection;
  mat4 model;
  vec4 viewPos;
} globals;

layout(set = 1, binding = 0) uniform BubbleParams {
  float thicknessMin;
  float thicknessMax;
  float n1;
  float n2;
  float n3;
  int spectralSamples;
  int thicknessMode;
  float gravityStrength;
  float noiseScale;
  int useAnimation;
  float driftSpeed;
  float roughness;
  float iblExposure;
  float iblGamma;
  float time;
  int rtMode;                // 0=both, 1=R-only, 2=T-only
  int showThicknessHeatmap;
  int showFresnelOnly;
  int showNormal;
  int _pad0;
} params;

layout(set = 2, binding = 0) uniform sampler2D heightTex;
layout(set = 3, binding = 0) uniform samplerCube prefilteredCubemap;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846;

vec3 wymanCMF(float lambda) {
  float x = 0.398 * exp(-1250.0 * pow(log((lambda + 570.1) / 1014.0), 2.0)) +
            1.132 * exp(-234.0 * pow(log((1338.0 - lambda) / 743.5), 2.0));
  float y = 1.011 * exp(-0.5 * pow((lambda - 556.1) / 46.14, 2.0));
  float z = 2.060 * exp(-32.0 * pow(log((lambda - 265.8) / 180.4), 2.0));
  return vec3(x, y, z);
}

float fresnelSchlick(float cosTheta, float nFrom, float nTo) {
  float f0 = (nFrom - nTo) / (nFrom + nTo);
  f0 = f0 * f0;
  return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

// XYZ -> linear sRGB (D65). max(0) clamp deferred to after composite so
// out-of-gamut channels can cancel against env contributions.
vec3 xyzToSrgb(vec3 xyz) {
  mat3 M = mat3(
       3.2406, -0.9689,  0.0557,
      -1.5372,  1.8758, -0.2040,
      -0.4986,  0.0415,  1.0570);
  return M * xyz;
}

// Simplex 3D noise. Tetrahedral grid removes the axis-aligned cell-boundary
// artifacts that both value noise and Perlin gradient noise show on a sphere
// (visible as discontinuous circles along x/y/z axes when noiseScale grows).
// Reference: Stefan Gustavson "Simplex noise demystified" (2005), GLSL port by
// Ian McEwan / Ashima Arts (2011), public domain.
// Native range is roughly [-1, 1]; remapped to [0, 1] below.
vec3 mod289v3(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289v4(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289v4(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) {
  return 1.79284291400159 - 0.85373472095314 * r;
}

float simplexNoise3D(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  // First corner
  vec3 i = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);

  // Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  // Permutations
  i = mod289v3(i);
  vec4 p = permute(permute(permute(i.z + vec4(0.0, i1.z, i2.z, 1.0)) + i.y +
                           vec4(0.0, i1.y, i2.y, 1.0)) +
                   i.x + vec4(0.0, i1.x, i2.x, 1.0));

  // Gradients: 7x7 points over a square, mapped onto an octahedron
  float n_ = 0.142857142857;  // 1/7
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);

  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);

  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  // Normalise gradients
  vec4 norm = taylorInvSqrt(
      vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

  // Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)),
               0.0);
  m = m * m;
  float n = 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2),
                                   dot(p3, x3)));
  return clamp(n * 0.5 + 0.5, 0.0, 1.0);
}

float thicknessAt(vec2 uv, vec3 worldPos, vec3 normal) {
  float h;
  if (params.thicknessMode == 0) {
    vec2 sampleUV = uv;
    if (params.useAnimation != 0) {
      sampleUV += vec2(params.driftSpeed * params.time, 0.0);
    }
    h = texture(heightTex, sampleUV).r;
  } else {
    float gravity = clamp(0.5 + worldPos.y * params.gravityStrength, 0.0, 1.0);
    vec3 noisePos = worldPos * params.noiseScale;
    if (params.useAnimation != 0) {
      noisePos += vec3(0.0, 0.0, params.time * params.driftSpeed);
    }
    float n = simplexNoise3D(noisePos);
    h = mix(gravity, n, 0.5);
  }
  return mix(params.thicknessMin, params.thicknessMax, h);
}

// Per-wavelength reflectance R(lambda). Amplitude form: R = R1 + R2 +
// 2*sqrt(R1*R2)*cos(opdPhase + deltaPhi). R1, R2 are wavelength-independent
// under Schlick (no dispersion) so passed in.
float thinFilmR(float d, float cosTheta2, float deltaPhi, float lambda,
                float R1, float R2) {
  float opdPhase = (4.0 * PI * params.n2 * d * cosTheta2) / lambda;
  return R1 + R2 + 2.0 * sqrt(max(R1 * R2, 0.0)) * cos(opdPhase + deltaPhi);
}

void main() {
  vec3 N = normalize(inWorldNormal);
  vec3 V = normalize(globals.viewPos.xyz - inWorldPos);

  if (params.showNormal != 0) {
    outColor = vec4(N * 0.5 + 0.5, 1.0);
    return;
  }

  float cosTheta1 = max(dot(N, V), 0.001);
  float sinTheta1Sq = 1.0 - cosTheta1 * cosTheta1;
  float sinTheta2 = (params.n1 / params.n2) * sqrt(max(sinTheta1Sq, 0.0));
  float cosTheta2 = (sinTheta2 >= 1.0) ? 0.0
                                       : sqrt(1.0 - sinTheta2 * sinTheta2);

  float R1 = fresnelSchlick(cosTheta1, params.n1, params.n2);
  float R2 = (sinTheta2 >= 1.0)
                 ? 1.0
                 : fresnelSchlick(cosTheta2, params.n2, params.n3);
  float phi1 = (params.n1 < params.n2) ? PI : 0.0;
  float phi2 = (params.n2 < params.n3) ? PI : 0.0;
  float deltaPhi = phi1 - phi2;

  float d = thicknessAt(inUV, inWorldPos, N);

  if (params.showThicknessHeatmap != 0) {
    float t = clamp((d - params.thicknessMin) /
                        max(params.thicknessMax - params.thicknessMin, 1.0),
                    0.0, 1.0);
    vec3 c = vec3(0.267 + 0.5 * t, 0.005 + 0.9 * t,
                  0.329 + 0.4 * sin(t * PI));
    outColor = vec4(c, 1.0);
    return;
  }
  if (params.showFresnelOnly != 0) {
    outColor = vec4(R1, R1, R1, 1.0);
    return;
  }

  // Spectral integration: accumulate XYZ for both R(lambda) and T(lambda)=1-R.
  vec3 XYZ_R = vec3(0.0);
  vec3 XYZ_T = vec3(0.0);
  float yWeight = 0.0;
  int N_samples = clamp(params.spectralSamples, 4, 64);
  for (int i = 0; i < N_samples; ++i) {
    float lambda =
        mix(380.0, 780.0, (float(i) + 0.5) / float(N_samples));
    float R = (sinTheta2 >= 1.0)
                  ? 1.0
                  : thinFilmR(d, cosTheta2, deltaPhi, lambda, R1, R2);
    R = clamp(R, 0.0, 1.0);
    vec3 cmf = wymanCMF(lambda);
    XYZ_R += R * cmf;
    XYZ_T += (1.0 - R) * cmf;
    yWeight += cmf.y;
  }
  float invY = 1.0 / max(yWeight, 1e-6);
  vec3 rgbR = xyzToSrgb(XYZ_R * invY);
  vec3 rgbT = xyzToSrgb(XYZ_T * invY);

  vec3 R_dir = reflect(-V, N);
  vec3 T_dir = refract(-V, N, params.n1 / params.n2);
  if (length(T_dir) < 1e-3) T_dir = -V;

  float maxLod = float(textureQueryLevels(prefilteredCubemap) - 1);
  float lod = clamp(params.roughness, 0.0, 1.0) * maxLod;
  vec3 envR = textureLod(prefilteredCubemap, R_dir, lod).rgb;
  vec3 envT = textureLod(prefilteredCubemap, T_dir, lod).rgb;

  vec3 colorR = rgbR * envR;
  vec3 colorT = rgbT * envT;
  vec3 color;
  if (params.rtMode == 1) {
    color = colorR;
  } else if (params.rtMode == 2) {
    color = colorT;
  } else {
    color = colorR + colorT;
  }
  color = max(color, vec3(0.0)) * params.iblExposure;
  color = pow(color, vec3(1.0 / params.iblGamma));

  outColor = vec4(color, 1.0);
}
