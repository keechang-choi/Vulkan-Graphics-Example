#version 450

layout (set = 1, binding = 0) uniform sampler2D samplerColorMap;

layout (location = 0) in vec3 inColor;
layout (location = 1) in vec2 inUV;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	// Desaturate color
    // vec3 color = vec3(mix(inColor, vec3(dot(vec3(0.2126,0.7152,0.0722), inColor)), 0.65));	
	vec3 color = vec3(texture(samplerColorMap, inUV));

	outFragColor.rgb = color;
}