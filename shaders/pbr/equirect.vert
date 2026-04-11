// shaders/pbr/equirect.vert
#version 450

layout(push_constant) uniform CapturePush {
    mat4 mvp;
} capture;

layout(location = 0) out vec3 outLocalPos;

const vec3 positions[36] = vec3[](
    // +X
    vec3( 1,-1, 1),vec3( 1,-1,-1),vec3( 1, 1,-1),
    vec3( 1, 1,-1),vec3( 1, 1, 1),vec3( 1,-1, 1),
    // -X
    vec3(-1,-1,-1),vec3(-1,-1, 1),vec3(-1, 1, 1),
    vec3(-1, 1, 1),vec3(-1, 1,-1),vec3(-1,-1,-1),
    // +Y
    vec3(-1, 1, 1),vec3( 1, 1, 1),vec3( 1, 1,-1),
    vec3( 1, 1,-1),vec3(-1, 1,-1),vec3(-1, 1, 1),
    // -Y
    vec3(-1,-1,-1),vec3( 1,-1,-1),vec3( 1,-1, 1),
    vec3( 1,-1, 1),vec3(-1,-1, 1),vec3(-1,-1,-1),
    // +Z
    vec3(-1,-1, 1),vec3(-1, 1, 1),vec3( 1, 1, 1),
    vec3( 1, 1, 1),vec3( 1,-1, 1),vec3(-1,-1, 1),
    // -Z
    vec3( 1,-1,-1),vec3( 1, 1,-1),vec3(-1, 1,-1),
    vec3(-1, 1,-1),vec3(-1,-1,-1),vec3( 1,-1,-1)
);

void main() {
    outLocalPos = positions[gl_VertexIndex];
    gl_Position = capture.mvp * vec4(outLocalPos, 1.0);
}
