#version 450
layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;

void main() {
  // Discard fragments outside the disk
  vec2 coord = gl_PointCoord - vec2(0.5);
  if (length(coord) > 0.5) discard;
  outColor = vColor;
}
