#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inColor;

layout (set = 0, binding = 0) uniform GlobalUbo 
{
	mat4 projection;
    mat4 view;
	vec4 lightPos;
	mat4 inverseView;
} globalUbo;

layout (set = 1, binding = 0) uniform ModelUbo 
{
	mat4 model;
	vec4 modelColor;
} modelUbo;

layout (std430, set = 3, binding = 0) uniform MeshUbo
{
	mat4 nodeMatrix;
	mat4 jointMatrices[64];
	float jointCount;
} meshUbo;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outColor;
layout (location = 2) out vec2 outUV;
layout (location = 3) out vec3 outViewVec;
layout (location = 4) out vec3 outLightVec;

void main() 
{
	outColor = inColor;
	outUV = inUV;
	gl_Position = globalUbo.projection * globalUbo.view * modelUbo.model  * vec4(inPos.xyz, 1.0);
	
	vec4 pos = modelUbo.model * vec4(inPos, 1.0);
	mat3 normalMatrix = inverse(transpose(mat3(modelUbo.model)));
	outNormal = normalize(normalMatrix * inNormal);
	vec3 lPos = globalUbo.lightPos.xyz;
	outLightVec = normalize(lPos - pos.xyz);
	outViewVec = normalize(globalUbo.inverseView[3].xyz - pos.xyz);		
}