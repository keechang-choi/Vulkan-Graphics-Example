#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUv;

layout(set = 0, binding = 0) uniform GlobalUbo {
  mat4 projection;
  mat4 view;
  mat4 inverseView;
  vec4 canvasInfo;
}
ubo;

layout(location = 0) out vec2 vUv;

void main() {
  vUv = inUv;
  gl_Position = ubo.projection * ubo.view * vec4(inPos, 1.0);
}
