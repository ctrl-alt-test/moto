#version 150

in vec4 a_position;

#include "shared.h"

#ifdef USE_VERTEX_SHADER
out float camFishEye;
out float camFoV;
out float camMotoSpace; // bool - 0. or 1.
out float camProjectionRatio;
out float camShowDriver; // bool - 0. or 1.
out vec3 camPos;
out vec3 camTa;
out float wallHeight;
out float guardrailHeight;
// x: actual width
// y: width + transition
// z: max width
out vec3 roadWidthInMeters;

const int SPLINE_SIZE = 13;
out vec2 spline[SPLINE_SIZE];
out float motoDistanceOnCurve;

uniform float iTime;
float time;

const float INF = 1e6;


float hash11(float x) { return fract(sin(x) * 43758.5453); }
vec2 hash12(float x) { float h = hash11(x); return vec2(h, hash11(h)); }

mat2 Rotation(float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, s, -s, c);
}

vec2 valueNoise2(float p)
{
    float p0 = floor(p);
    float p1 = p0 + 1.;

    vec2 v0 = hash12(p0);
    vec2 v1 = hash12(p1);

    float fp = p - p0;
    fp = fp*fp * (3.0 - 2.0 * fp);

    return mix(v0, v1, fp);
}

#include "camera.frag"

#endif

void main(void)
{
    gl_Position = a_position;
#ifdef USE_VERTEX_SHADER
    time = iTime;
    selectShot();
#endif
}
