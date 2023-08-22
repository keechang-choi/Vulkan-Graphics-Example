#version 450

layout (location = 0) in float inPackedColor;

layout (location = 0) out vec4 outFragColor;

float packColor(vec3 color) {
    return color.r + color.g * 256.0 + color.b * 256.0 * 256.0;
}

vec3 unpackColor(float f) {
    vec3 color;
    color.b = floor(f / 256.0 / 256.0);
    color.g = floor((f - color.b * 256.0 * 256.0) / 256.0);
    color.r = floor(f - color.b * 256.0 * 256.0 - color.g * 256.0);
    // now we have a vec3 with the 3 components in range [0..255]. Let's normalize it!
    return color / 255.0;
}

void main () 
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    float alpha = max(0.25 - dot(coord, coord), 0.0) * 4.0;
	outFragColor = vec4(unpackColor(inPackedColor), alpha);
}