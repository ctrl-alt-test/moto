#version 150

//#define DEBUG 1
#define ENABLE_STOCHASTIC_MOTION_BLUR
// #define ENABLE_STEP_COUNT
// #define ENABLE_DAY_MODE
// #define DISABLE_MOTO
// #define DISABLE_MOTO_DRIVER
// #define DISABLE_TERRAIN
// #define DISABLE_TREES

const int MAX_RAY_MARCH_STEPS = 200;
const float MAX_RAY_MARCH_DIST = 100.0;
const int MAX_SHADOW_STEPS = 30;
const float MAX_SHADOW_DIST = 5.0;

const float NORMAL_DP = 2.*1e-3;
const float BOUNCE_OFFSET = 1e-3;

const float GAMMA = 2.2;

in vec3 camPos;
in vec3 camTa;
in float camMotoSpace;
in float camFoV;
in float camProjectionRatio;
in float camFishEye;
in float camShowDriver;

out vec4 fragColor;
const vec2 iResolution = vec2(1920.,1080.);

uniform float iTime;

float PIXEL_ANGLE = camFoV / iResolution.x;

#define ZERO(iTime) min(0, int(iTime))
#include "common.frag"

vec3 motoPos, motoDir;

const float NO_ID = -1.;
const float GROUND_ID = 0.;
const float MOTO_ID = 1.;
const float MOTO_HEAD_LIGHT_ID = 2.;
const float MOTO_BREAK_LIGHT_ID = 3.;
const float MOTO_WHEEL_ID = 4.;
const float MOTO_MOTOR_ID = 5.;
const float MOTO_EXHAUST_ID = 6.;
const float MOTO_DRIVER_ID = 7.;
const float MOTO_DRIVER_HELMET_ID = 8.;
const float CITY_ID = 9.;

bool IsMoto(float mid)
{
    return mid >= MOTO_ID && mid <= MOTO_DRIVER_HELMET_ID;
}

#include "backgroundContent.frag"
#include "roadContent.frag"
#include "motoContent.frag"
#include "moto.frag"
