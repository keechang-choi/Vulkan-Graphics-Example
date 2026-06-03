#version 450
layout(location = 0) in vec4 inPos;
layout(location = 1) in vec4 inVel;
layout(location = 2) in vec4 inPredict;
layout(location = 3) in vec4 inColor;

layout(set = 0, binding = 0) uniform GlobalUbo {
  mat4 projection;
  mat4 view;
  mat4 inverseView;
  vec4 canvasInfo;
  vec4 renderParams;  // x = particle point-size scale
}
ubo;

layout(location = 0) out vec4 vColor;

// blue (low) -> green -> red (high), like the FLIP/PBF density view
vec3 sciColor(float t) {
  t = clamp(t, 0.0, 1.0) * 4.0;
  int seg = int(t);
  float f = t - float(seg);
  if (seg == 0) return vec3(0.0, f, 1.0);
  if (seg == 1) return vec3(0.0, 1.0, 1.0 - f);
  if (seg == 2) return vec3(f, 1.0, 0.0);
  return vec3(1.0, 1.0 - f, 0.0);
}

void main() {
  vec4 clip = ubo.projection * ubo.view * vec4(inPos.xyz, 1.0);
  gl_Position = clip;
  // renderParams.x scales the sprite size (live-particle disc); <1 shrinks it.
  gl_PointSize =
      clamp(ubo.renderParams.x * 16.0 / max(clip.w, 0.001), 1.0, 32.0);
  if (ubo.canvasInfo.w > 0.5) {
    // density debug: inVel.w holds rho/rho0; center the map around rest (1.0)
    vColor = vec4(sciColor((inVel.w - 0.5) / 1.0), 1.0);
  } else {
    vColor = inColor;
  }
}
