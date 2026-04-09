// shaders/pbr/irradiance.frag
#version 450

layout(set = 1, binding = 0) uniform samplerCube envMap;

layout(push_constant) uniform PushBlock {
    float deltaPhi;
    float deltaTheta;
} push;

layout(location = 0) in  vec3 inLocalPos;
layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;

void main() {
    vec3 N = normalize(inLocalPos);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, N));
    up = normalize(cross(N, right));

    vec3 irradiance = vec3(0.0);
    float nrSamples = 0.0;

    for (float phi = 0.0; phi < 2.0 * PI; phi += push.deltaPhi) {
        for (float theta = 0.0; theta < 0.5 * PI; theta += push.deltaTheta) {
            vec3 tangentSample = vec3(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta));
            vec3 sampleVec = tangentSample.x * right
                           + tangentSample.y * up
                           + tangentSample.z * N;
            irradiance += texture(envMap, sampleVec).rgb
                        * cos(theta) * sin(theta);
            nrSamples++;
        }
    }
    irradiance = PI * irradiance / nrSamples;
    outColor = vec4(irradiance, 1.0);
}
