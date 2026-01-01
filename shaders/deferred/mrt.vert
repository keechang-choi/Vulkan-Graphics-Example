#version 450

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec2 inUV;
layout (location = 2) in vec4 inColor;
layout (location = 3) in vec3 inNormal;
layout (location = 4) in vec3 inTangent;

layout (set = 0, binding = 0) uniform UBO 
{
	mat4 projection;
    mat4 view;
} ubo;

layout (set = 1, binding = 0) uniform ModelUbo 
{
	mat4 modelMatrix;
	vec4 modelColor;
} modelUbo;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outUV;
layout (location = 2) out vec4 outColor;
layout (location = 3) out vec4 outWorldPos;
layout (location = 4) out vec3 outTangent;

void main() 
{
	// gl_InstanceIndex
	// discard vertex color
	outColor = modelUbo.modelColor;
	outColor.a = clamp(outColor.a, 0.0, 1.0);
	outUV = inUV;
	mat4 worldTransform =  modelUbo.modelMatrix;
	outWorldPos = worldTransform * vec4(inPos.rgb, 1.0);
	gl_Position = ubo.projection * ubo.view * outWorldPos;
	
	mat3 normalMatrix = inverse(transpose(mat3(worldTransform)));
	outNormal = normalize(normalMatrix * inNormal);
	outTangent = normalize(normalMatrix * inTangent);
}