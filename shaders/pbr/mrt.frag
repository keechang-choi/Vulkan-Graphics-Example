#version 450

layout (set = 2, binding = 0) uniform sampler2D samplerColorMap;
layout (set = 2, binding = 1) uniform sampler2D samplerNormalMap;
layout (set = 2, binding = 2) uniform sampler2D samplerMetallicRoughnessMap;
layout (set = 2, binding = 3) uniform sampler2D samplerEmissionMap;

layout (set = 1, binding = 0) uniform ModelUbo {
	mat4 modelMatrix;
	vec4 modelColor;
	vec4 pbrOverride;  // x=metallic, y=roughness, z=useOverride(0/1), w=unused
} modelUbo;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec4 inColor;
layout (location = 3) in vec4 inWorldPos;
layout (location = 4) in vec3 inTangent;

layout (location = 0) out vec4 outPosition;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outAlbedo;
layout (location = 3) out vec4 outArm;
layout (location = 4) out vec4 outEmissive;

void main()
{
	vec4 albedo = texture(samplerColorMap, inUV);
	vec3 color = mix(albedo.rgb, inColor.rgb, inColor.a);
	outAlbedo = albedo;

	outPosition = inWorldPos;

	// Normal: use tangent-space mapping only when tangent is valid.
	// The sphere model has no TANGENT attribute (zero tangent), so we fall
	// back to the geometric normal to avoid NaN from normalize(vec3(0)).
	vec3 N = normalize(inNormal);
	if (length(inTangent) > 0.001) {
		vec3 T = normalize(inTangent);
		vec3 B = normalize(cross(N, T));
		mat3 TBN = mat3(T, B, N);
		vec3 normalMapSample = texture(samplerNormalMap, inUV).xyz * 2.0 - vec3(1.0);
		N = normalize(TBN * normalMapSample);
	}
	outNormal = vec4(N, 1.0);

	// ARM (AO/Roughness/Metallic): override per-instance when useSpheres is active.
	// pbrOverride: x=metallic, y=roughness, z=useOverride
	vec3 arm = vec3(0.0);
	if (modelUbo.pbrOverride.z > 0.5) {
		// glTF metallicRoughness convention: g=roughness, b=metallic
		arm = vec3(0.0, modelUbo.pbrOverride.y, modelUbo.pbrOverride.x);
	} else {
		arm.rgb = texture(samplerMetallicRoughnessMap, inUV).rgb;
		// arm.r = texture(samplerOcclusionMap, inUV).r; // occlusion
	}
	outArm = vec4(arm, 1.0);

	vec3 emissive = vec3(0.0);
	emissive.rgb = texture(samplerEmissionMap, inUV).rgb;
	outEmissive = vec4(emissive, 1.0);
}
