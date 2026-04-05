#version 450

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main() {
    vec2 uv = inUV * 2.0 - 1.0;
    float dist = length(uv);
    if (dist > 1.0) discard;
    float intensity = 1.0 - dist;
    outFragColor = vec4(inColor * intensity, intensity);
}
