#version 450

layout(set = 0, binding = 0) uniform Globals {
  mat4 view;
  mat4 projection;
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
  float alphaScale;
  float iblExposure;
  float iblGamma;
  float time;
  int showThicknessHeatmap;
  int showFresnelOnly;
  float _pad0;
  float _pad1;
} params;

layout(set = 2, binding = 0) uniform sampler2D heightTex;
layout(set = 3, binding = 0) uniform samplerCube prefilteredCubemap;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inWorldNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846;

// Wyman 2013 analytical fit for CIE 1931 color matching functions.
// Input: lambda in nm. Output: (x_bar, y_bar, z_bar).
vec3 wymanCMF(float lambda) {
  float x = 0.398 * exp(-1250.0 * pow(log((lambda + 570.1) / 1014.0), 2.0)) +
            1.132 * exp(-234.0 * pow(log((1338.0 - lambda) / 743.5), 2.0));
  float y = 1.011 * exp(-0.5 * pow((lambda - 556.1) / 46.14, 2.0));
  float z = 2.060 * exp(-32.0 * pow(log((lambda - 265.8) / 180.4), 2.0));
  return vec3(x, y, z);
}

// Schlick Fresnel for unpolarized light at boundary nFrom -> nTo.
float fresnelSchlick(float cosTheta, float nFrom, float nTo) {
  float f0 = (nFrom - nTo) / (nFrom + nTo);
  f0 = f0 * f0;
  return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

// XYZ -> linear sRGB (D65). Caller applies any gamma after.
vec3 xyzToSrgb(vec3 xyz) {
  mat3 M = mat3(
       3.2406, -0.9689,  0.0557,
      -1.5372,  1.8758, -0.2040,
      -0.4986,  0.0415,  1.0570);
  return max(M * xyz, vec3(0.0));
}

void main() {
  vec3 N = normalize(inWorldNormal);
  vec3 V = normalize(globals.viewPos.xyz - inWorldPos);
  vec3 R = reflect(-V, N);

  vec3 envColor = textureLod(prefilteredCubemap, R, 0.0).rgb;

  outColor = vec4(envColor * params.iblExposure, 1.0);
}
