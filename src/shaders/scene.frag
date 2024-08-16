#version 150

//#define DEBUG 1

out vec4 fragColor;
const vec2 iResolution = vec2(1920.,1080.);
vec2 iMouse = vec2(700., 900.);
uniform float iTime;

#define ZERO min(0, int(iTime))
#include "common.frag"

vec3 motoPos, motoDir;

#define NO_ID              -1.
#define GROUND_ID           0.
#define MOTO_ID             1.
#define MOTO_HEAD_LIGHT_ID  2.
#define MOTO_BREAK_LIGHT_ID 3.
#define MOTO_WHEEL_ID       4.
#define MOTO_MOTOR_ID       5.
#define DRIVER_ID           6.
#define DRIVER_HELMET_ID    7.

#include "roadContent.frag"
#include "motoContent.frag"
#include "moto.frag"
