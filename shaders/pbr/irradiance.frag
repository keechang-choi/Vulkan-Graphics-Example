// shaders/pbr/irradiance.frag
#version 450

layout(set = 0, binding = 0) uniform samplerCube envMap;

layout(push_constant) uniform PushBlock {
    layout(offset = 64) uint numSamples;
} push;

layout(location = 0) in  vec3 inLocalPos;
layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

// Van der Corput radical inverse (base 2)
float radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

// Hammersley low-discrepancy sequence
vec2 hammersley(uint i, uint N) {
    return vec2(float(i) / float(N), radicalInverse_VdC(i));
}

// Per-texel hash → random phi offset. Chetan Jags (2015): per-texel azimuthal
// jitter converts the fixed Hammersley grid's structured aliasing into
// high-frequency noise, which the eye averages out.
float hash31(vec3 p) {
    p = fract(p * vec3(443.8975, 397.2973, 491.1871));
    p += dot(p, p.yzx + 19.19);
    return fract((p.x + p.y) * p.z);
}

// Cosine-weighted hemisphere sample in world space around N
// PDF = cos(theta) / PI  =>  integral * (1/N) * sum(L(L_i) * PI) = PI * avg
vec3 importanceSampleCosine(vec2 xi, vec3 N, float phiJitter) {
    float phi      = 2.0 * PI * xi.x + phiJitter;
    float cosTheta = sqrt(1.0 - xi.y);
    float sinTheta = sqrt(xi.y);

    vec3 H = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    // Build TBN avoiding degenerate up vector
    vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

void main() {
    vec3 N = normalize(inLocalPos);

    vec3  irradiance    = vec3(0.0);
    float envResolution = 512.0;
    float saTexel       = 4.0 * PI / (6.0 * envResolution * envResolution);
    float phiJitter     = hash31(N) * 2.0 * PI;

    for (uint i = 0u; i < push.numSamples; ++i) {
        vec2 xi = hammersley(i, push.numSamples);
        vec3 L  = importanceSampleCosine(xi, N, phiJitter);

        // cosTheta = sqrt(1 - xi.y) from cosine sampling; PDF = NdotL / PI
        float NdotL   = sqrt(1.0 - xi.y);
        float pdf     = NdotL / PI + 0.0001;
        float saSample = 1.0 / (float(push.numSamples) * pdf + 0.0001);
        // +1 LOD bias per GPU Gems 3 Ch.20 for smoother, sample-overlapping filtering
        float mipLevel = max(0.5 * log2(saSample / saTexel) + 1.0, 0.0);

        irradiance += textureLod(envMap, L, mipLevel).rgb;
    }

    // Cosine-weighted PDF cancels NdotL: irradiance = PI * (1/N) * sum(L(L_i))
    irradiance = PI * irradiance / float(push.numSamples);
    outColor = vec4(irradiance, 1.0);
}
