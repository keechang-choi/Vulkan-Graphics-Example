#version 450

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec2 uv;

layout (location = 0) out vec3 fragColor;
layout (location = 1) out vec3 fragPosWorld;
layout (location = 2) out vec3 fragNormalWorld;
layout (location = 3) out vec2 fragTexCoord;

struct PointLight{
  vec4 position; // ignore w
  vec4 color; // w as intensity
};

layout (set = 0, binding = 0) uniform GlobalUbo{
    mat4 projection;
    mat4 view;
    mat4 invView;
    vec4 ambientLightColor; // w as intensity
    PointLight pointLights[10];
    int numLights;
} ubo;

layout (push_constant) uniform Push{
	mat4 modelMatrix; 
    mat4 normalMatrix;
} push;

void main() {
  vec4 positionWorld = push.modelMatrix * vec4(position, 1.0);
  gl_Position = ubo.projection * ubo.view * positionWorld;

  // temp
  // vec3 normalWorldSpace = normalize(mat3(push.modelMatrix) * normal);
  // vec3 normalWorldSpace = normalize((push.modelMatrix * vec4(normal, 0.0)).xyz);
  // mat3 normalMatrix = transpose(inverse(mat3(push.modelMatrix));
  fragNormalWorld = normalize(mat3(push.normalMatrix) * normal);
  fragPosWorld = positionWorld.xyz;
  fragColor = color;
  fragTexCoord = uv;
}