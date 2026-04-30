// shaders/common/brdf_lut.vert
#version 450

layout(location = 0) out vec2 outUV;

void main() {
    // Fullscreen triangle: gl_VertexIndex 0,1,2
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
}
