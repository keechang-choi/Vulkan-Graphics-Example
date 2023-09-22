#version 450

layout (location = 0) in vec4 inPos;
layout (location = 1) in vec4 inVel;

layout (location = 0) out vec3 outColor;

layout (set = 0, binding = 0) uniform GlobalUbo 
{
	mat4 projection;
    mat4 view;
	mat4 inverseView;
    vec2 screenDim;
	vec2 tailInfo;
} globalUbo;

out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};


// https://stackoverflow.com/questions/6893302/decode-rgb-value-to-single-float-without-bit-shift-in-glsl
float packColor(in vec3 color) {
    return color.r + color.g * 256.0 + color.b * 256.0 * 256.0;
}

vec3 unpackColor(in float f) {
    vec3 color;
    color.b = floor(f / 256.0 / 256.0);
    color.g = floor((f - color.b * 256.0 * 256.0) / 256.0);
    color.r = floor(f - color.b * 256.0 * 256.0 - color.g * 256.0);
    // now we have a vec3 with the 3 components in range [0..255]. Let's normalize it!
    return color / 255.0;
}


void main () 
{
	const float spriteSize = 0.005 * inPos.w; // Point size influenced by mass (stored in inPos.w);

	vec4 eyePos = globalUbo.view * vec4(inPos.x, inPos.y, inPos.z, 1.0); 
	vec4 projectedCorner = globalUbo.projection * vec4(0.5 * spriteSize, 0.5 * spriteSize, eyePos.z, eyePos.w);
	// gl_PointSize = clamp(globalUbo.screenDim.x * projectedCorner.x / projectedCorner.w, 1.0, 16.0);
	gl_PointSize = 3.0;

	gl_Position = globalUbo.projection * eyePos;

	outColor = unpackColor(inVel.w);
}