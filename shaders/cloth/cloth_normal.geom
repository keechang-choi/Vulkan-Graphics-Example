#version 450


layout (triangles) in;
layout (line_strip, max_vertices = 6) out;

layout (set = 0, binding = 0) uniform GlobalUbo 
{
	mat4 projection;
    mat4 view;
	mat4 inverseView;
	vec4 lightPos;
    vec2 screenDim;
	vec2 pointSize;
} globalUbo;

// triangle normals
layout (location = 0) in vec3 inNormals[];
layout (location = 0) out vec3 outColor;


void main() 
{
	float normalLength = 0.1;
	for(int i=0; i<gl_in.length(); i++)
	{
		vec3 pos = gl_in[i].gl_Position.xyz;
		vec3 normal = inNormals[i].xyz;

		mat4 transform = globalUbo.projection * globalUbo.view;
		gl_Position = transform * vec4(pos, 1.0);
		outColor = vec3(1.0, 0.0, 0.0);
		EmitVertex();

		gl_Position = transform * vec4(pos + normal * normalLength, 1.0);
		outColor = vec3(1.0, 1.0, 0.0);
		EmitVertex();

		EndPrimitive();
	}
}