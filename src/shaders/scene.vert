#version 150

in vec4 a_position;

out vec3 sunDir;
out vec3 camPos;
out vec3 camTa;
out float camFoV;
out float camProjectionRatio;
out float camFishEye;
out float camMotoSpace; // bool - 0. or 1.
out float camShowDriver; // bool - 0. or 1.

uniform float iTime;

const float INFINITE = 9e7;

float hash11(float x) { return fract(sin(x) * 43758.5453); }
vec2 hash12(float x) { float h = hash11(x); return vec2(h, hash11(h)); }

vec2 valueNoise(float p)
{
    float p0 = floor(p);
    float p1 = p0 + 1.;

    vec2 v0 = hash12(p0);
    vec2 v1 = hash12(p1);

    float fp = p - p0;
    fp = fp*fp * (3.0 - 2.0 * fp);

    return mix(v0, v1, fp);
}

float verticalBump()
{
    return valueNoise(6.*iTime).x;
}

void sideShotFront()
{
    vec2 p = vec2(0.95, 0.5);
    p.x += mix(-0.5, 1., valueNoise(0.5*iTime).y);
    p.y += 0.05 * verticalBump();
    camPos = vec3(p, 2.8);
    camTa = vec3(p.x, p.y + 0.1, 0.);
    camProjectionRatio = 2.;
}

void sideShotRear()
{
    vec2 p = vec2(-1., 0.5);
    p.x += mix(-1.2, 0.5, valueNoise(0.5*iTime).y);
    p.y += 0.05 * verticalBump();
    camPos = vec3(p, 2.8);
    camTa = vec3(p.x, p.y + 0.1, 0.);
    camProjectionRatio = 2.;
}

void fpsDashboardShot()
{
    vec2 noise = valueNoise(5.*iTime);
    vec2 slowNoise = valueNoise(0.1*iTime);

    camPos = vec3(0.1, 1.12, 0.);
    camPos.z += mix(-0.02, 0.02, slowNoise.x);
    camPos.y += 0.01 * noise.y;
    camTa = vec3(5, 1, 0.);
    camProjectionRatio = 0.6;
}

// t should go from 0 to 1 in roughly 4 seconds
void dashBoardUnderTheShoulderShot(float t)
{
    float bump = 0.02 * verticalBump();
    camPos = vec3(-0.2 - 0.6 * t, 0.88 + 0.35*t + bump, 0.42);
    camTa = vec3(0.5, 1. + 0.2 * t + bump, 0.25);
    camProjectionRatio = 1.5;
}

void frontWheelCloseUpShot()
{
    camPos = vec3(-0.1, 0.5, 0.5);
    camTa = vec3(0.9, 0.35, 0.2);
    vec2 vibration = 0.005 * valueNoise(40.*iTime);
    float bump = 0.02 * verticalBump();
    vibration.x += bump;
    camPos.yz += vibration;
    camTa.yz += vibration;
    camProjectionRatio = 1.6;
    camShowDriver = 0.;
}

void overTheHeadShot()
{
    camPos = vec3(-1.4, 1.7, 0.);
    camTa = vec3(0.05, 1.45, 0.);
    float bump = 0.01 * verticalBump();
    camPos.y += bump;
    camTa.y += bump;
    camProjectionRatio = 2.;
}

void main(void)
{
    gl_Position = a_position;
    float time = iTime;

    camProjectionRatio = 1.;
    camFishEye = 0.1;
    camMotoSpace = 1.;
    camShowDriver = 1.;

    // list of camera shots
    /*
    if (time < 5.) {
        camPos = vec3(1.26, 1.07, -0.5);
        camPos = vec3(2., 1.07, 1.5);
        camTa = vec3(-2., 0., -5.);
    } else if (time < 10.) {
        camPos = vec3(0.02, 1.2, 0.05);
        camTa = vec3(10.,0.,0.);
    } else if (time < 15.) {
        camPos = vec3(-1.1, 1.2, -0.8);
        camTa = vec3(0.,0.,10.);
    } else if (time < 20.) {
        camPos = vec3(-3., 2.5, -0.2);
        camTa = vec3(1.,-0.5,0.);
    } else {
        camMotoSpace = 0.;
        camPos = vec3(0.02, 5.2+iTime*0.05, 4.05);
        camTa = vec3(10.,2.,1.);
    }
    */

    float shotDuration = 6.;
    float numberOfDifferentShots = 6.;

    float t = fract((iTime / shotDuration) / numberOfDifferentShots) * numberOfDifferentShots;
    float shot = floor(t);
    float t_in_shot = fract(t);

    if (shot == 0.)
    {
        sideShotRear();
    }
    if (shot == 1.)
    {
        sideShotFront();
    }
    if (shot == 2.)
    {
        frontWheelCloseUpShot();
    }
    if (shot == 3.)
    {
        overTheHeadShot();
    }
    if (shot == 4.)
    {
        fpsDashboardShot();
    }
    if (shot == 5.)
    {
        dashBoardUnderTheShoulderShot(t_in_shot);
    }
    camFoV = atan(1. / camProjectionRatio);
}
