#version 450

layout (set = 2, binding = 0) uniform sampler2D samplerColorMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec4 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	// Desaturate color
    // vec3 color = vec3(mix(inColor, vec3(dot(vec3(0.2126,0.7152,0.0722), inColor)), 0.65));	
	vec3 color = mix(vec3(texture(samplerColorMap, inUV)), inColor.rgb, inColor.a);
	if(!gl_FrontFacing){
		color = vec3(0.0, 1.0, 0.0);
	}

	vec3 ambient = color * vec3(0.1);
	vec3 N = normalize(inNormal);
	vec3 L = normalize(inLightVec);
	vec3 V = normalize(inViewVec);
	vec3 R = reflect(-L, N);
	vec3 halfAngle = normalize(L + V);
	vec3 diffuse = max(dot(N, L), 0.0) * color;
	// vec3 specular = pow(max(dot(R, V), 0.0), 64.0) * vec3(0.35);
	vec3 specular = pow(max(dot(halfAngle, N), 0.0), 64.0) * vec3(0.35);
	outFragColor = vec4(ambient + diffuse + specular, 1.0);		
}