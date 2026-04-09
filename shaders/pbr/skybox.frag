// shaders/pbr/skybox.frag
#version 450

layout(set = 0, binding = 0) uniform samplerCube envMap;

layout(push_constant) uniform PushBlock {
    mat4 view;
    mat4 projection;
} push;

layout(location = 0) in  vec3 inUVW;
layout(location = 0) out vec4 outColor;

// Uncharted2 filmic tonemapping
vec3 Uncharted2Tonemap(vec3 x) {
    float A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

void main() {
    vec3 color = texture(envMap, inUVW).rgb;
    // Tone-map + gamma
    float exposure = 4.5;
    color = Uncharted2Tonemap(color * exposure);
    vec3 whiteScale = 1.0 / Uncharted2Tonemap(vec3(11.2));
    color *= whiteScale;
    color = pow(color, vec3(1.0 / 2.2));
    outColor = vec4(color, 1.0);
}
