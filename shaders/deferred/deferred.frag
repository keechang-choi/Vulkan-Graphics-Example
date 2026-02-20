#version 450

layout (set = 0, binding = 0) uniform sampler2D samplerPosition;
layout (set = 0, binding = 1) uniform sampler2D samplerNormal;
layout (set = 0, binding = 2) uniform sampler2D samplerAlbedo;
layout (set = 0, binding = 3) uniform sampler2D samplerArm;
layout (set = 0, binding = 4) uniform sampler2D samplerEmissive;
layout (set = 0, binding = 5) uniform sampler2D samplerDepth;


layout (location = 0) in vec2 inUV;

layout (constant_id = 0) const int DISPLAY_TARGET_INDEX = 0;

layout (location = 0) out vec4 outFragColor;

struct Light {
	vec4 position;
	vec3 color;
	float radius;
};
#define MAX_LIGHTS 10
layout (set = 0, binding = 6) uniform UBO
{
	Light lights[MAX_LIGHTS];
	vec4 viewPos;
	int displayDebugTarget;
	int numLights;
	float nearPlane;
	float farPlane;
	float farClamp;
} ubo;

void main() 
{
	// G buffer reading
	vec3 fragPos = texture(samplerPosition, inUV).rgb;
	vec3 normal = texture(samplerNormal, inUV).rgb;
	vec4 albedo = texture(samplerAlbedo, inUV);
	vec3 arm = texture(samplerArm, inUV).rgb;
	// display target. NOTE: specialization constant.
	int displayTargetIndex = DISPLAY_TARGET_INDEX;
	if(displayTargetIndex == 0){
		displayTargetIndex = ubo.displayDebugTarget;
	}
	if (displayTargetIndex > 0) {
		switch (displayTargetIndex) {
			case 1: 
				outFragColor.rgb = fragPos;
				break;
			case 2: 
				// tested by vertex normal.[-1,1] to [0,1]
				vec3 normal_color = normal * vec3(1.0, +1.0, 1.0);
				normal_color += vec3(1.0, 1.0, 1.0);
				normal_color *= vec3(0.5, 0.5, 0.5);
				outFragColor.rgb = normal_color;
				break;
			case 3: 
				outFragColor.rgb = albedo.rgb;
				break;
			case 4: 
				outFragColor.rgb = arm.rgb;
				break;
			case 5: 
				outFragColor.rgb = arm.rrr;
				break;
			case 6: 
				outFragColor.rgb = arm.ggg;
				break;
			case 7: 
				outFragColor.rgb = arm.bbb;
				break;
			case 8:
				outFragColor.rgb = texture(samplerEmissive, inUV).rgb;
				break;
			case 9:
				vec4 depthRead = texture(samplerDepth, inUV);
				float depth = depthRead.r;
				// linearize depth
				float signedDepth = depth * 2.0 - 1.0; // back to [-1,1]
				float linearDepth = (2.0 * ubo.nearPlane * ubo.farPlane) / (ubo.farPlane + ubo.nearPlane - signedDepth * (ubo.farPlane - ubo.nearPlane));
				// remap to [0,1]
				linearDepth = (linearDepth - ubo.nearPlane) / (ubo.farPlane - ubo.nearPlane);
				linearDepth = linearDepth * (ubo.farPlane - ubo.nearPlane) / (ubo.farClamp - ubo.nearPlane);
				linearDepth = clamp(linearDepth, 0.0, 1.0);
				outFragColor.rgb = vec3(1.0-linearDepth);
				break;
		}		
		outFragColor.a = 1.0;
		return;
	}

	// composition 
#define ambientIntensity 0.15

	vec3 ambient = albedo.rgb * ambientIntensity;
	vec3 fragColor = ambient;
	for(int i=0; i<ubo.numLights; i++){
		vec3 L = ubo.lights[i].position.xyz - fragPos;
		float distFragToLight = length(L);
		//if(distFragToLight < ubo.lights[i].radius)
		{
			vec3 V = ubo.viewPos.xyz - fragPos;
			L = normalize(L);
			V = normalize(V);

			float atten = ubo.lights[i].radius / (pow(distFragToLight, 2.0) + 1.0);
			vec3 N = normalize(normal);
			float NdotL = max(dot(N, L), 0.0);
			vec3 diffuse = ubo.lights[i].color * albedo.rgb * NdotL * atten;

			// vec3 R = reflect(-L, N); // phong
			// float RdotV = max(dot(R, V), 0.0);
			vec3 H = normalize(L + V); // blinn-phong
			float NdotH = max(dot(H, N), 0.0);

			float shininess = 16.0;
			// TODO: check albedo alpha ->
			// specular intensity or roughness.
			albedo.a = 0.3;
			vec3 specular = ubo.lights[i].color * albedo.a * pow(NdotH, shininess) * atten;
			fragColor += diffuse + specular;
		}
	}
	outFragColor = vec4(fragColor, 1.0);		
}