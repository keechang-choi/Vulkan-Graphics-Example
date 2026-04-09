// shaders/pbr/skybox.vert
#version 450

layout(push_constant) uniform PushBlock {
    mat4 view;
    mat4 projection;
} push;

layout(location = 0) out vec3 outUVW;

const vec3 cubeVertices[36] = vec3[36](
    // +X
    vec3( 1,-1,-1), vec3( 1,-1, 1), vec3( 1, 1, 1),
    vec3( 1, 1, 1), vec3( 1, 1,-1), vec3( 1,-1,-1),
    // -X
    vec3(-1,-1, 1), vec3(-1,-1,-1), vec3(-1, 1,-1),
    vec3(-1, 1,-1), vec3(-1, 1, 1), vec3(-1,-1, 1),
    // +Y
    vec3(-1, 1,-1), vec3( 1, 1,-1), vec3( 1, 1, 1),
    vec3( 1, 1, 1), vec3(-1, 1, 1), vec3(-1, 1,-1),
    // -Y
    vec3(-1,-1, 1), vec3( 1,-1, 1), vec3( 1,-1,-1),
    vec3( 1,-1,-1), vec3(-1,-1,-1), vec3(-1,-1, 1),
    // +Z
    vec3(-1,-1, 1), vec3(-1, 1, 1), vec3( 1, 1, 1),
    vec3( 1, 1, 1), vec3( 1,-1, 1), vec3(-1,-1, 1),
    // -Z
    vec3( 1,-1,-1), vec3( 1, 1,-1), vec3(-1, 1,-1),
    vec3(-1, 1,-1), vec3(-1,-1,-1), vec3( 1,-1,-1)
);

void main() {
    vec3 pos = cubeVertices[gl_VertexIndex];
    outUVW = pos;
    // depth trick: set z=w so depth=1.0 after perspective divide
    gl_Position = (push.projection * push.view * vec4(pos, 1.0)).xyww;
}
