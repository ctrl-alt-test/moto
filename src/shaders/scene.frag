#version 150

// #define DEBUG 1
const bool ENABLE_STOCHASTIC_MOTION_BLUR = true;
// #define ENABLE_STEP_COUNT
// #define ENABLE_DAY_MODE
// #define DISABLE_MOTO
// #define DISABLE_MOTO_DRIVER
// #define DISABLE_TERRAIN
// #define DISABLE_TREES

// Constants:
const int MAX_ROAD_LIGHTS = 2 * 8; // Pairs of lights
const int MAX_RAY_MARCH_STEPS = 200;
const float MAX_RAY_MARCH_DIST = 1000.0;
const int MAX_SHADOW_STEPS = 30;
const float MAX_SHADOW_DIST = 5.0;
const float NORMAL_DP = 2.*1e-3;
const float BOUNCE_OFFSET = 1e-3;
const float GAMMA = 2.2;
const int SPLINE_SIZE = 13;
const float INF = 1e6;
#include "shared.h"
vec2 iResolution = vec2(XRES, YRES);

const float DISTANCE_BETWEEN_LAMPS = 50.;

// Uniforms:
uniform float iTime;
uniform sampler2D tex;

#ifdef USE_VERTEX_SHADER
// Inputs:
in float camFishEye;
in float camFoV;
in float camMotoSpace;
in float camProjectionRatio;
in float camShowDriver;
in vec3 camPos;
in vec3 camTa;
in float wallHeight;
in float guardrailHeight;
in vec3 roadWidthInMeters;

in vec2 spline[SPLINE_SIZE];

#else

float camFishEye;
float camFoV;
float camMotoSpace;
float camProjectionRatio;
float camShowDriver;
vec3 camPos;
vec3 camTa;
float wallHeight;
float guardrailHeight;
// x: actual width
// y: width + transition
// z: max width
vec3 roadWidthInMeters;

vec2 spline[SPLINE_SIZE];

#endif

// Outputs:
out vec4 fragColor;

// Semantic constants:
#ifdef USE_VERTEX_SHADER
float PIXEL_ANGLE = camFoV / iResolution.x;
#else
float PIXEL_ANGLE;
#endif
float time;

#include "common.frag"
#include "ids.frag"
#include "backgroundContent.frag"
#include "roadContent.frag"
#include "motoContent.frag"
#include "rendering.frag"
#ifndef USE_VERTEX_SHADER
#include "camera.frag"
#endif

void main()
{
    ComputeBezierSegmentsLengthAndAABB();
    vec2 texCoord = gl_FragCoord.xy/iResolution.xy;
    vec2 uv = (texCoord * 2. - 1.) * vec2(1., iResolution.y / iResolution.x);

    if (ENABLE_STOCHASTIC_MOTION_BLUR) {
        time = iTime + hash31(vec3(gl_FragCoord.xy, 1e-3*iTime)) * 0.008;
    } else {
        time = iTime;
    }

    computeMotoPosition();
#ifndef USE_VERTEX_SHADER
    selectShot();
#endif

    // Compute moto position

    // camPos and camTa are passed by the vertex shader
    vec3 ro;
    vec3 rd;
    vec3 cameraPosition = camPos;
    vec3 cameraTarget = camTa;
    vec3 cameraUp = vec3(0., 1., 0.);
    if (camMotoSpace > 0.5) {
        cameraPosition = motoToWorld(camPos, true);
        cameraTarget = motoToWorld(camTa, true);
        //cameraUp = motoToWorld(cameraUp, false);
    }
    setupCamera(uv, cameraPosition, cameraTarget, cameraUp, ro, rd);

    // View moto from front
    // motoCamera(uv, vec3(1.26, 1.07, 0.05), vec3(-10.,0.,0), ro, rd);

    // First-person view
    // motoCamera(uv, vec3(0.02, 1.2, 0.05), vec3(10.,0.,0.), ro, rd);

    // Third-person view, near ground
    // motoCamera(uv, vec3(-2., 0.5, -0.2), vec3(10.,0.,0.), ro, rd);


    vec3 p;
#ifdef ENABLE_STEP_COUNT
    int steps = 0;
    vec2 t = rayMarchScene(ro, rd, MAX_RAY_MARCH_DIST, MAX_RAY_MARCH_STEPS, p, steps);
    fragColor = vec4(stepsToColor(steps), 1.);
    return;
#else
    vec2 t = rayMarchScene(ro, rd, MAX_RAY_MARCH_DIST, MAX_RAY_MARCH_STEPS, p);
#endif
    vec3 i_N = evalNormal(p, t.x);
    vec3 i_radiance = evalRadiance(t, p, -rd, i_N);
    
    vec3 i_color = pow(i_radiance, vec3(1. / GAMMA)) *
        smoothstep(0., 4., iTime) * // fade in
        smoothstep(138., 132., iTime); // fade out
    fragColor = vec4(mix(i_color, texture(tex, texCoord).rgb, 0.2)+vec3(
        hash21(fract(uv+iTime)),
        hash21(fract(uv-iTime)),
        hash21(fract(uv.yx+iTime))
    )*.025, 1.);
}
