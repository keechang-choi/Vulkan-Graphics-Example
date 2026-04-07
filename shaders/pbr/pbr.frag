#version 450

layout (set = 0, binding = 0) uniform sampler2D samplerPosition;
layout (set = 0, binding = 1) uniform sampler2D samplerNormal;
layout (set = 0, binding = 2) uniform sampler2D samplerAlbedo;
layout (set = 0, binding = 3) uniform sampler2D samplerArm;
layout (set = 0, binding = 4) uniform sampler2D samplerEmissive;
layout (set = 0, binding = 5) uniform sampler2D samplerDepth;
layout (set = 0, binding = 7) uniform sampler2D samplerHeight;

layout (location = 0) in vec2 inUV;
layout (constant_id = 0) const int DISPLAY_TARGET_INDEX = 0;
layout (location = 0) out vec4 outFragColor;

#define PI 3.14159265359

struct Light {
    vec4 position;
    vec3 color;
    float radius;
};
#define MAX_LIGHTS 10
layout (set = 0, binding = 6) uniform UBO {
    Light lights[MAX_LIGHTS];
    vec4 viewPos;
    int displayDebugTarget;
    int numLights;
    float nearPlane;
    float farPlane;
    float farClamp;
    int useDirectionalLight;
    float ambientStrength;
    vec2 _pad;
    vec4 dirLightDir;    // xyz = direction toward light (normalized)
    vec3 dirLightColor;
    float _pad2;
} ubo;

// Normal Distribution: Trowbridge-Reitz GGX
float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a   = roughness * roughness;
    float a2  = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float denom  = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    return a2 / (PI * denom * denom + 1e-7);
}

// Geometry: Schlick-GGX single term
float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

// Geometry: Smith's method
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);
}

// Fresnel-Schlick
vec3 FresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 PBR(vec3 N, vec3 V, vec3 L, vec3 radiance,
         vec3 albedo, float roughness, float metallic, vec3 F0) {
    vec3 H = normalize(V + L);
    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(N, V, L, roughness);
    vec3  F   = FresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);
    float NdotL = max(dot(N, L), 0.0);
    vec3 specular = (NDF * G * F) /
        max(4.0 * max(dot(N, V), 0.0) * NdotL, 0.001);

    return (kD * albedo / PI + specular) * radiance * NdotL;
}

void main() {
    vec3 fragPos = texture(samplerPosition, inUV).rgb;
    vec3 N       = normalize(texture(samplerNormal, inUV).rgb);
    vec4 albedoSample = texture(samplerAlbedo, inUV);
    // sRGB -> linear
    vec3 albedo   = pow(albedoSample.rgb, vec3(2.2));
    vec3 arm      = texture(samplerArm, inUV).rgb;
    float ao        = arm.r;
    float roughness = arm.g;
    float metallic  = arm.b;
    vec3 emissive = texture(samplerEmissive, inUV).rgb;

    // Debug display target
    int displayTargetIndex = DISPLAY_TARGET_INDEX;
    if (displayTargetIndex == 0) displayTargetIndex = ubo.displayDebugTarget;
    if (displayTargetIndex > 0) {
        switch (displayTargetIndex) {
            case 1: outFragColor.rgb = fragPos; break;
            case 2: outFragColor.rgb = N * 0.5 + 0.5; break;
            case 3: outFragColor.rgb = albedoSample.rgb; break;
            case 4: outFragColor.rgb = arm.rgb; break;
            case 5: outFragColor.rgb = vec3(ao); break;
            case 6: outFragColor.rgb = vec3(roughness); break;
            case 7: outFragColor.rgb = vec3(metallic); break;
            case 8: outFragColor.rgb = emissive; break;
            case 9: {
                float depth = texture(samplerDepth, inUV).r;
                float sd = depth * 2.0 - 1.0;
                float ld = (2.0 * ubo.nearPlane * ubo.farPlane) /
                    (ubo.farPlane + ubo.nearPlane - sd * (ubo.farPlane - ubo.nearPlane));
                ld = (ld - ubo.nearPlane) / (ubo.farPlane - ubo.nearPlane);
                ld = ld * (ubo.farPlane - ubo.nearPlane) / (ubo.farClamp - ubo.nearPlane);
                outFragColor.rgb = vec3(1.0 - clamp(ld, 0.0, 1.0));
                break;
            }
            case 10: outFragColor.rgb = texture(samplerHeight, inUV).rrr; break;
        }
        outFragColor.a = 1.0;
        return;
    }

    // PBR: Cook-Torrance BRDF
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 V = normalize(ubo.viewPos.xyz - fragPos);

    vec3 Lo = vec3(0.0);
    if (ubo.useDirectionalLight != 0) {
        // Directional light: fixed direction, no attenuation
        vec3 L = normalize(ubo.dirLightDir.xyz);
        Lo = PBR(N, V, L, ubo.dirLightColor, albedo, roughness, metallic, F0);
    } else {
        for (int i = 0; i < ubo.numLights; i++) {
            vec3 L    = normalize(ubo.lights[i].position.xyz - fragPos);
            float dist = length(ubo.lights[i].position.xyz - fragPos);
            float attenuation = ubo.lights[i].radius / (dist * dist + 1.0);
            vec3 radiance = ubo.lights[i].color * attenuation;
            Lo += PBR(N, V, L, radiance, albedo, roughness, metallic, F0);
        }
    }

    vec3 ambient = vec3(ubo.ambientStrength) * albedo * ao;
    vec3 color = ambient + Lo + emissive;

    // Reinhard tone mapping
    color = color / (color + vec3(1.0));
    // Gamma correction (linear -> sRGB)
    color = pow(color, vec3(1.0 / 2.2));

    outFragColor = vec4(color, 1.0);
}
