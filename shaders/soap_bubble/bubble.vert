#version 450

layout(set = 0, binding = 0) uniform Globals {
  mat4 view;
  mat4 projection;
  vec4 viewPos;
} globals;

layout(location = 0) in vec4 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec3 inNormal;
layout(location = 4) in vec4 inTangent;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outWorldNormal;
layout(location = 2) out vec2 outUV;

void main() {
  outWorldPos = inPos.xyz;
  outWorldNormal = inNormal;
  outUV = inUV;
  gl_Position = globals.projection * globals.view * vec4(inPos.xyz, 1.0);
}
