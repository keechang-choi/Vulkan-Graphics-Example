#version 450

layout (location = 0) in vec4 inPos;

layout (set = 0, binding = 0) uniform GlobalUbo 
{
	mat4 projection;
    mat4 view;
	mat4 inverseView;
	vec4 lightPos;
	vec4 tailInfo;
    vec2 screenDim;
	vec2 pointSize;
} globalUbo;


layout (location = 0) out vec3 outColor;

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
	vec4 eyePos = globalUbo.view * vec4(inPos.x, inPos.y, inPos.z, 1.0); 
	gl_Position = globalUbo.projection * eyePos;
	outColor = unpackColor(inPos.w);
    float tailSize = globalUbo.tailInfo.x;
	float tailFrontIndex = globalUbo.tailInfo.y;
    float tailIntensity = globalUbo.tailInfo.z;
    float tailFadeOut = globalUbo.tailInfo.w;
	float currentFront = mod((tailFrontIndex-1 + tailSize), tailSize);
	float index = mod(float(gl_VertexIndex), tailSize);
    float brightness = (tailSize - mod(tailSize + currentFront - index, tailSize))/tailSize;
	outColor = tailIntensity * unpackColor(inPos.w) * pow(brightness, tailFadeOut);
}