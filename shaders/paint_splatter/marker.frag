#version 450
layout(location = 0) in vec4 vColor;
layout(location = 0) out vec4 outColor;

// Filled disc in the spoid's paint colour, ringed by a contrasting border so
// the markers are easy to tell apart from each other and from the fluid.
void main() {
  vec2 c = (gl_PointCoord - vec2(0.5)) * 2.0;  // -1..1 across the point
  float d = length(c);
  if (d > 1.0) discard;

  // Border colour contrasts with the fill luminance (dark ring on light paint,
  // white ring on dark paint).
  float lum = dot(vColor.rgb, vec3(0.299, 0.587, 0.114));
  vec3 border = (lum > 0.5) ? vec3(0.05) : vec3(1.0);

  // Outer ~24% of the disc is the border.
  float ring = smoothstep(0.72, 0.80, d);
  vec3 rgb = mix(vColor.rgb, border, ring);
  outColor = vec4(rgb, 1.0);
}
