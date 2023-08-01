#version 450

layout (set = 1, binding = 0) uniform sampler2D samplerColorMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	// Desaturate color
    // vec3 color = vec3(mix(inColor, vec3(dot(vec3(0.2126,0.7152,0.0722), inColor)), 0.65));	
	// vec3 color = vec3(texture(samplerColorMap, inUV));

	vec3 N = normalize(inNormal);
	vec3 L = normalize(inLightVec);

	float intensity = dot(N, L);
	float shade = 1.0;
	shade = intensity < 0.80 ? 0.9 : shade;
	shade = intensity < 0.65 ? 0.75 : shade;
	shade = intensity < 0.35 ? 0.45 : shade;
	shade = intensity < 0.1 ? 0.15 : shade;

	outFragColor = vec4(inColor*shade, 1.0);		
}