#version 450

layout (location = 0) in vec3 inColor;

layout (location = 0) out vec4 outFragColor;


void main () 
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    // high power -> full circle sprite
    float alpha = max(1.0 - pow(4.0*dot(coord, coord), 0.5), 0.0);
	outFragColor = vec4(inColor * alpha, 1.0);
}