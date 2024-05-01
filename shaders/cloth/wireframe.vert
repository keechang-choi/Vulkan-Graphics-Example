#version 450

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec4 inColor;
layout (location = 3) in vec3 inTangent;
layout (location = 4) in vec2 inUV;

layout (set = 0, binding = 0) uniform GlobalUbo 
{
	mat4 projection;
    mat4 view;
	mat4 inverseView;
	vec4 lightPos;
    vec2 screenDim;
	vec2 pointSize;
} globalUbo;

layout (set = 1, binding = 0) uniform ModelUbo 
{
	mat4 modelMatrix;
	vec4 modelColor;
} modelUbo;


layout (location = 0) out vec4 outColor;
layout (location = 1) out vec2 outUV;

void main() 
{
	if(modelUbo.modelColor.a <= -1.0){
		outColor = vec4(vec3(modelUbo.modelColor), 1.0);
	}else{
		outColor = modelUbo.modelColor;
	}
	outUV = inUV;
	// applied in compute
	mat4 worldTransform = mat4(1.0);// modelUbo.modelMatrix ;
	gl_Position = globalUbo.projection * globalUbo.view * worldTransform * vec4(inPos.xyz, 1.0);
}