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

layout (location = 0) out vec3 outColor;
layout (location = 1) out vec2 outUV;

void main() 
{
	outColor = inColor;
	outUV = inUV;
	gl_Position = globalUbo.projection * globalUbo.view * modelUbo.model  * vec4(inPos.xyz, 1.0);
	
}