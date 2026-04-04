#version 450

layout (set = 0, binding = 0) uniform UniformDataOffscreen {
    mat4 projection;
    mat4 view;
} offscreenUbo;

struct Light {
    vec4 position;
    vec3 color;
    float radius;
};
#define MAX_LIGHTS 10
layout (set = 1, binding = 0) uniform CompositionUBO {
    Light lights[MAX_LIGHTS];
    vec4 viewPos;
    int debugDisplayTarget;
    int numLights;
    float nearPlane;
    float farPlane;
    float farClamp;
} compositionUbo;

layout (push_constant) uniform PushConstants {
    float spriteSize;
} pc;

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;

const vec2 quadOffsets[6] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0,  1.0)
);

void main() {
    int lightIdx = gl_InstanceIndex;
    vec2 offset  = quadOffsets[gl_VertexIndex % 6];

    vec3 lightPos = compositionUbo.lights[lightIdx].position.xyz;
    outColor = compositionUbo.lights[lightIdx].color;

    // Extract camera right and up from view matrix (GLM column-major)
    vec3 right = vec3(offscreenUbo.view[0][0], offscreenUbo.view[1][0], offscreenUbo.view[2][0]);
    vec3 up    = vec3(offscreenUbo.view[0][1], offscreenUbo.view[1][1], offscreenUbo.view[2][1]);

    vec3 worldPos = lightPos
        + right * offset.x * pc.spriteSize
        + up    * offset.y * pc.spriteSize;

    outUV = offset * 0.5 + 0.5;
    gl_Position = offscreenUbo.projection * offscreenUbo.view * vec4(worldPos, 1.0);
}
