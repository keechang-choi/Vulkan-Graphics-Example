// shaders/pbr/prefilter.frag
#version 450

layout(set = 0, binding = 0) uniform samplerCube envMap;

layout(push_constant) uniform PushBlock {
    layout(offset = 64) float roughness;
    layout(offset = 68) uint  numSamples;
} push;

layout(location = 0) in  vec3 inLocalPos;
layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

float RadicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

vec2 Hammersley(uint i, uint N) {
    return vec2(float(i) / float(N), RadicalInverse_VdC(i));
}

vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    vec3 H = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    vec3 up = abs(N.z) < 0.999 ? vec3(0, 0, 1) : vec3(1, 0, 0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

float DistributionGGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float denom = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom);
}

void main() {
    vec3 N = normalize(inLocalPos);
    vec3 R = N;
    vec3 V = R;

    vec3  prefilteredColor = vec3(0.0);
    float totalWeight = 0.0;
    float envResolution = 512.0;

    for (uint i = 0u; i < push.numSamples; ++i) {
        vec2 Xi = Hammersley(i, push.numSamples);
        vec3 H  = ImportanceSampleGGX(Xi, N, push.roughness);
        vec3 L  = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(dot(N, L), 0.0);
        if (NdotL > 0.0) {
            float NdotH = max(dot(N, H), 0.0);
            float HdotV = max(dot(H, V), 0.0);
            float D = DistributionGGX(NdotH, push.roughness);
            float pdf = (D * NdotH / (4.0 * HdotV)) + 0.0001;
            float saTexel  = 4.0 * PI / (6.0 * envResolution * envResolution);
            float saSample = 1.0 / (float(push.numSamples) * pdf + 0.0001);
            float mipLevel = push.roughness == 0.0
                ? 0.0 : 0.5 * log2(saSample / saTexel);
            prefilteredColor +=
                textureLod(envMap, L, mipLevel).rgb * NdotL;
            totalWeight += NdotL;
        }
    }
    outColor = vec4(prefilteredColor / totalWeight, 1.0);
}
