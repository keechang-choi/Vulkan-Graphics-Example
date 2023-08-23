#version 450

layout (location = 0) in vec3 inColor;

layout (location = 0) out vec4 outFragColor;


void main () 
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    float alpha = max(0.25 - dot(coord, coord), 0.0) * 4.0;
	outFragColor = vec4(inColor * alpha, 1.0);
}