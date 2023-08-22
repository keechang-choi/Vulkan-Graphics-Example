#version 450

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec4 inVel;

layout (location = 0) out float outPackedColor;

layout (set = 0, binding = 0) uniform GlobalUbo 
{
	mat4 projection;
    mat4 view;
	vec4 lightPos;
	mat4 inverseView;
    vec2 screenDim;
} globalUbo;

out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};

void main () 
{
	const float spriteSize = 0.005 * inPos.w; // Point size influenced by mass (stored in inPos.w);

	vec4 eyePos = globalUbo.view * vec4(inPos.x, inPos.y, inPos.z, 1.0); 
	vec4 projectedCorner = globalUbo.projection * vec4(0.5 * spriteSize, 0.5 * spriteSize, eyePos.z, eyePos.w);
	gl_PointSize = clamp(globalUbo.screenDim.x * projectedCorner.x / projectedCorner.w, 1.0, 128.0);
	
	gl_Position = globalUbo.projection * eyePos;

	outPackedColor = inVel.w;
}