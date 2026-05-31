#version 450

layout(location = 0) in vec2 vUv;

layout(location = 0) out vec4 outColor;

void main() {
  vec2 g = step(0.5, fract(vUv * 8.0));
  float c = abs(g.x - g.y);
  outColor = vec4(mix(vec3(0.85), vec3(0.25), c), 1.0);
}
