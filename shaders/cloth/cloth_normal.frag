#version 450

layout (set = 2, binding = 0) uniform sampler2D samplerColorMap;

layout (location = 0) in vec3 inColor;

layout (location = 0) out vec4 outFragColor;

void main(void)
{
	outFragColor = vec4(inColor, 1.0);
}