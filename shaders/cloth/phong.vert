#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec4 inColor;
layout (location = 3) in vec3 inTangent;
layout (location = 4) in vec2 inUV;

layout (set = 0, binding = 0) uniform GlobalUbo 
{
	mat4 projection;
    mat4 view;
	mat4 inverseView;
	vec4 lightPos;
    vec2 screenDim;
	vec2 pointSize;
} globalUbo;

layout (set = 1, binding = 0) uniform ModelUbo 
{
	mat4 modelMatrix;
	vec4 modelColor;
} modelUbo;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec4 outColor;
layout (location = 2) out vec2 outUV;
layout (location = 3) out vec3 outViewVec;
layout (location = 4) out vec3 outLightVec;

void main() 
{
	outColor = modelUbo.modelColor;
	outUV = inUV;
	// TODO: animation pre-compute
	mat4 worldTransform = modelUbo.modelMatrix;
	vec4 pos = worldTransform * vec4(inPos, 1.0);
	gl_Position = globalUbo.projection * globalUbo.view * pos;
	
	mat3 normalMatrix = inverse(transpose(mat3(worldTransform)));
	outNormal = normalize(normalMatrix * inNormal);
	vec3 lPos = globalUbo.lightPos.xyz;
	outLightVec = normalize(lPos - pos.xyz);
	outViewVec = normalize(globalUbo.inverseView[3].xyz - pos.xyz);		
}