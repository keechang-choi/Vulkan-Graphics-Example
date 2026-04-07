#version 450

layout (set = 2, binding = 0) uniform sampler2D samplerColorMap;
layout (set = 2, binding = 1) uniform sampler2D samplerNormalMap;
layout (set = 2, binding = 2) uniform sampler2D samplerMetallicRoughnessMap;
layout (set = 2, binding = 3) uniform sampler2D samplerEmissionMap;

layout (set = 4, binding = 0) uniform sampler2D samplerHeightMap;

layout (set = 0, binding = 0) uniform UBO {
	mat4 projection;
	mat4 view;
	vec4 viewPos;
	float heightScale;
} ubo;

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
layout (location = 5) out vec4 outHeight;

// Parallax Occlusion Mapping (learnopengl.com/Advanced-Lighting/Parallax-Mapping)
// Uses a height map (bright=raised). Internally converts to depth = 1 - height.
// White dummy (height=1, depth=0) exits immediately with no UV offset.
vec2 parallaxOcclusionMapping(vec2 texCoords, vec3 viewDir) {
    // Dynamic layer count: more layers at grazing angles for quality
    const float minLayers = 8.0;
    const float maxLayers = 32.0;
    float numLayers = mix(maxLayers, minLayers, abs(dot(vec3(0.0, 0.0, 1.0), viewDir)));

    float layerDepth = 1.0 / numLayers;
    float currentLayerDepth = 0.0;

    // Total UV shift proportional to view angle and heightScale
    vec2 P = viewDir.xy * ubo.heightScale;
    vec2 deltaTexCoords = P / numLayers;

    vec2  currentTexCoords    = texCoords;
    float currentDepth        = 1.0 - texture(samplerHeightMap, currentTexCoords).r;

    // Step through layers until ray hits the surface
    for (int i = 0; i < int(maxLayers); i++) {
        if (currentLayerDepth >= currentDepth) break;
        currentTexCoords  -= deltaTexCoords;
        currentDepth       = 1.0 - texture(samplerHeightMap, currentTexCoords).r;
        currentLayerDepth += layerDepth;
    }

    // Linear interpolation between the layer before and after the intersection
    vec2  prevTexCoords = currentTexCoords + deltaTexCoords;
    float depthAfter    = currentDepth - currentLayerDepth;
    float depthBefore   = (1.0 - texture(samplerHeightMap, prevTexCoords).r)
                          - (currentLayerDepth - layerDepth);
    float weight = depthAfter / (depthAfter - depthBefore);
    return mix(currentTexCoords, prevTexCoords, weight);
}

void main()
{
	vec3 N = normalize(inNormal);
	vec2 uv = inUV;

	// Apply POM only when tangent is valid (pirate-gold has tangents; smooth_sphere does not).
	// White dummy height map exits immediately with no offset.
	if (length(inTangent) > 0.001) {
		vec3 T = normalize(inTangent);
		vec3 B = normalize(cross(N, T));
		mat3 TBN_inv = transpose(mat3(T, B, N));  // world → tangent space

		vec3 viewDir_world   = normalize(ubo.viewPos.xyz - inWorldPos.xyz);
		vec3 viewDir_tangent = normalize(TBN_inv * viewDir_world);

		uv = parallaxOcclusionMapping(inUV, viewDir_tangent);
	}

	// Output raw height at original UV to G-buffer for debug display (target 10)
	float h = texture(samplerHeightMap, inUV).r;
	outHeight = vec4(h, h, h, 1.0);

	vec4 albedo = texture(samplerColorMap, uv);
	vec3 color  = mix(albedo.rgb, inColor.rgb, inColor.a);
	outAlbedo   = vec4(color, albedo.a);

	outPosition = inWorldPos;

	// Normal map in tangent space (tangent guard same as POM above)
	if (length(inTangent) > 0.001) {
		vec3 T = normalize(inTangent);
		vec3 B = normalize(cross(N, T));
		mat3 TBN = mat3(T, B, N);
		vec3 normalSample = texture(samplerNormalMap, uv).xyz * 2.0 - vec3(1.0);
		N = normalize(TBN * normalSample);
	}
	outNormal = vec4(N, 1.0);

	// ARM: override or from texture
	vec3 arm = vec3(0.0);
	if (modelUbo.pbrOverride.z > 0.5) {
		arm = vec3(1.0, modelUbo.pbrOverride.y, modelUbo.pbrOverride.x);
	} else {
		arm = texture(samplerMetallicRoughnessMap, uv).rgb;
	}
	outArm = vec4(arm, 1.0);

	outEmissive = vec4(texture(samplerEmissionMap, uv).rgb, 1.0);
}
