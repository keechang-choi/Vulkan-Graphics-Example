#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inColor;
layout (location = 4) in vec4 inJointIndices;
layout (location = 5) in vec4 inJointWeights;


layout (set = 0, binding = 0) uniform GlobalUbo 
{
	mat4 projection;
    mat4 view;
	vec4 lightPos;
	mat4 inverseView;
} globalUbo;

layout (set = 1, binding = 0) uniform ModelUbo 
{
	mat4 modelMatrix;
	vec4 modelColor;
} modelUbo;

layout (set = 3, binding = 0) uniform MeshUbo
{
	mat4 nodeMatrix;
	mat4 jointMatrices[64];
	vec4 jointCount;
} meshUbo;

layout (location = 0) out vec4 outColor;
layout (location = 1) out vec2 outUV;

void main() 
{
	outColor.rgb = inColor*(1.0-modelUbo.modelColor.a) + modelUbo.modelColor.rgb*(modelUbo.modelColor.a);
	outColor.a = modelUbo.modelColor.a;
	outUV = inUV;
	mat4 skinMatrix = 		
		inJointWeights.x * meshUbo.jointMatrices[int(inJointIndices.x)] +
		inJointWeights.y * meshUbo.jointMatrices[int(inJointIndices.y)] +
		inJointWeights.z * meshUbo.jointMatrices[int(inJointIndices.z)] +
		inJointWeights.w * meshUbo.jointMatrices[int(inJointIndices.w)];
	
	float skinned = float(meshUbo.jointCount.x > 0.0f);
	skinMatrix = skinMatrix * skinned  + mat4(1.0f)*(1.0f-skinned);

	gl_Position = globalUbo.projection * globalUbo.view * modelUbo.modelMatrix * meshUbo.nodeMatrix * skinMatrix * vec4(inPos.xyz, 1.0);	
}