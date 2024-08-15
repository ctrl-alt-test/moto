#version 150

//#define DEBUG 1

out vec4 fragColor;
const vec2 iResolution = vec2(1920.,1080.);
vec2 iMouse = vec2(700., 900.);
uniform float iTime;

#define ZERO min(0, int(iTime))
#include "common.frag"

#include "moto.frag"
