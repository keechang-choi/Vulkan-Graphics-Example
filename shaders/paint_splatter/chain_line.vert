#version 450
layout(location = 0) in vec4 inPos;
layout(location = 1) in vec4 inVel;
layout(location = 2) in vec4 inPredict;
layout(location = 3) in vec4 inColor;
layout(set = 0, binding = 0) uniform GlobalUbo {
  mat4 projection;
  mat4 view;
  mat4 inverseView;
  vec4 canvasInfo;
}
ubo;
layout(location = 0) out vec4 vColor;
void main() {
  gl_Position = ubo.projection * ubo.view * vec4(inPos.xyz, 1.0);
  vColor = inColor;
}
