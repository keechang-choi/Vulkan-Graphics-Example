#version 450

layout(set = 0, binding = 0) uniform Globals {
  mat4 view;
  mat4 projection;
  mat4 model;       // unused in bg pass; bubble pass owns this slot
  vec4 viewPos;
} globals;

layout(push_constant) uniform PC {
  mat4 model;
  vec4 baseColor;
} pc;

layout(location = 0) in vec4 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec3 inNormal;
layout(location = 4) in vec4 inTangent;

layout(location = 0) out vec3 outWorldNormal;
layout(location = 1) out vec3 outBaseColor;

void main() {
  vec4 wp = pc.model * vec4(inPos.xyz, 1.0);
  outWorldNormal = mat3(pc.model) * inNormal;
  outBaseColor = pc.baseColor.rgb;
  gl_Position = globals.projection * globals.view * wp;
}
