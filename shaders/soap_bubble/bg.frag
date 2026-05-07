#version 450

layout(set = 1, binding = 0) uniform samplerCube irradianceMap;

layout(location = 0) in vec3 inWorldNormal;
layout(location = 1) in vec3 inBaseColor;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846;

void main() {
  vec3 N = normalize(inWorldNormal);
  vec3 irradiance = texture(irradianceMap, N).rgb;
  vec3 diffuse = irradiance * inBaseColor / PI;
  vec3 color = pow(max(diffuse, vec3(0.0)), vec3(1.0 / 2.2));
  outColor = vec4(color, 1.0);
}
