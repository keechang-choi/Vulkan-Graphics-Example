#version 450

layout (set = 2, binding = 0) uniform sampler2D samplerColorMap;

layout (location = 0) in vec4 inColor;
layout (location = 1) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	
	// vec3 color = vec3(texture(samplerColorMap, inUV)) * (1-inColor.a) + inColor.rgb*inColor.a;
	vec3 color = mix(vec3(texture(samplerColorMap, inUV)), inColor.rgb, inColor.a);
	outFragColor.rgb = color;
}