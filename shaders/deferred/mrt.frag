#version 450

layout (set = 2, binding = 0) uniform sampler2D samplerColorMap;
layout (set = 2, binding = 1) uniform sampler2D samplerNormalMap;
layout (set = 2, binding = 2) uniform sampler2D samplerMetallicRoughnessMap;
// layout (set = 2, binding = 3) uniform sampler2D samplerOcclusionMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec4 inColor;
layout (location = 3) in vec4 inWorldPos;
layout (location = 4) in vec3 inTangent;

layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outAlbedo;
layout (location = 3) out vec4 outArm;

void main() 
{
	// TODO: Desaturate color
	vec4 albedo = texture(samplerColorMap, inUV);
	vec3 color = mix(albedo.rgb, inColor.rgb, inColor.a);
	outAlbedo = albedo;

	outPosition = inWorldPos;

	// Calculate normal in tangent space
	vec3 N = normalize(inNormal);
	vec3 T = normalize(inTangent);
	vec3 B = normalize(cross(N, T));
	mat3 TBN = mat3(T, B, N);
	// range [0,1] -> [-1,1]
	vec3 normalMapSample = texture(samplerNormalMap, inUV).xyz * 2.0 - vec3(1.0);
	vec3 tnorm = normalize(TBN * normalMapSample);
	outNormal = vec4(tnorm, 1.0);
	vec3 arm = vec3(0.0);
	arm.rgb = texture(samplerMetallicRoughnessMap, inUV).rgb; // metallic roughness
	// arm.r = texture(samplerOcclusionMap, inUV).r; // occlusion
	outArm = vec4(arm, 1.0);
}