#version 450

layout (set = 2, binding = 0) uniform sampler2D samplerColorMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec4 inColor;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec3 inViewVec;
layout (location = 4) in vec3 inLightVec;

layout (location = 0) out vec4 outFragColor;

const float checkBoardSize = 0.05;
void main() 
{
	vec3 color = inColor.xyz;

	// High ambient colors because mesh materials are pretty dark
	vec3 ambient = color * vec3(0.3);
	vec3 N = normalize(inNormal);
	vec3 L = normalize(inLightVec);
	vec3 V = normalize(inViewVec);
	vec3 R = reflect(-L, N);
	vec3 halfAngle = normalize(L + V);
	vec3 diffuse = max(dot(N, L), 0.0) * color;
	// vec3 specular = pow(max(dot(R, V), 0.0), 64.0) * vec3(0.35);
	vec3 specular = pow(max(dot(halfAngle, N), 0.0), 64.0) * vec3(0.35);
	float alpha = 1.0;
	vec4 outColor = vec4(ambient + diffuse + specular, alpha);	

	vec2 modUV;
	modUV.x = mod(inUV.x, checkBoardSize*2.0);
	modUV.y = mod(inUV.y, checkBoardSize*2.0);
	
	if((modUV.x < checkBoardSize) ^^ (modUV.y < checkBoardSize)){
		outColor = vec4(outColor.xyz, 0.1);
	}
	outFragColor = outColor;
}