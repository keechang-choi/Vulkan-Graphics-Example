#version 450

layout(location = 0) in vec2 vUv;

layout(location = 0) out vec4 outColor;

// Canvas accumulation texture (M6): the permanent painting, written by the
// deposit compute pass and sampled here. Bound in GENERAL layout.
layout(set = 0, binding = 1) uniform sampler2D canvasTex;

void main() { outColor = texture(canvasTex, vUv); }
