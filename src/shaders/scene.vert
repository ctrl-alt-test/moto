#version 150

in vec4 a_position;

out float camFishEye;
out float camFoV;
out float camMotoSpace; // bool - 0. or 1.
out float camProjectionRatio;
out float camShowDriver; // bool - 0. or 1.
out vec3 camPos;
out vec3 camTa;

const int SPLINE_SIZE = 13;
out vec2 spline[SPLINE_SIZE];

uniform float iTime;

const float INF = 1e6;

float hash11(float x) { return fract(sin(x) * 43758.5453); }
vec2 hash12(float x) { float h = hash11(x); return vec2(h, hash11(h)); }

mat2 Rotation(float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, s, -s, c);
}

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

void GenerateSpline(float maxCurvature, float segmentLength, float seed)
{
    vec2 direction = vec2(hash11(seed), hash11(seed + 1.0)) * 2.0 - 1.0;
    direction = normalize(direction);
    vec2 point = vec2(0.);
    for(int i = 0; i < SPLINE_SIZE; i++) {
        if (i % 2 == 0) {
            spline[i] = point + 0.5*segmentLength*direction;
            continue;
        }
        float ha = hash11(seed + float(i) * 3.0);
        point += direction * segmentLength;
        float angle = mix(-maxCurvature, maxCurvature, ha);
        direction *= Rotation(angle);
        spline[i] = point;
    }
}

float verticalBump()
{
    return valueNoise(6.*iTime).x;
}

const int SHOT_SIDE_FRONT = 0;
void sideShotFront()
{
    vec2 p = vec2(0.95, 0.5);
    p.x += mix(-0.5, 1., valueNoise(0.5*iTime).y);
    p.y += 0.05 * verticalBump();
    camPos = vec3(p, 1.5);
    camTa = vec3(p.x, p.y + 0.1, 0.);
    camProjectionRatio = 1.2;
}

void sideShotRear()
{
    vec2 p = vec2(-1., 0.5);
    p.x += mix(-0.2, 0.2, valueNoise(0.5*iTime).y);
    p.y += 0.05 * verticalBump();
    camPos = vec3(p, 1.5);
    camTa = vec3(p.x, p.y + 0.1, 0.);
    camProjectionRatio = 1.2;
}

void fpsDashboardShot()
{
    vec2 noise = valueNoise(5.*iTime);
    vec2 slowNoise = valueNoise(0.1*iTime);

    camPos = vec3(0.1, 1.12, 0.);
    camPos.z += mix(-0.02, 0.02, slowNoise.x);
    camPos.y += 0.01 * noise.y;
    camTa = vec3(5, 1, 0.);
    camProjectionRatio = 0.7;
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
    camPos = vec3(-1.8, 1.7, 0.);
    camTa = vec3(0.05, 1.45, 0.);
    float bump = 0.01 * verticalBump();
    camPos.y += bump;
    camTa.y += bump;
    camProjectionRatio = 3.;
}

void topDownView() // useful for debugging & visualizing the spline
{
    camPos = vec3(-5., 37., 0.);
    camTa = vec3(1.05, 1.45, 0.);
    float bump = 0.01 * verticalBump();
    camPos.y += bump;
    camTa.y += bump;
    camProjectionRatio = 0.5;
}

void viewFromBehind(float t_in_shot)
{
    camTa = vec3(1., 1., 0.);
    camPos = vec3(-2. - 2.5*t_in_shot, 1.5, sin(t_in_shot));
    camProjectionRatio = 1.;
}

void faceView(float t_in_shot)
{
    camTa = vec3(1., 1.5, 0.);
    camPos = vec3(1. + 3.*t_in_shot, 1.5, 1);
    camProjectionRatio = 1.;
}

void openingShot(float t_in_shot) {
    camTa = vec3(10., 12. - mix(0., 10., min(1.,t_in_shot/6.)), 1.);
    camPos = vec3(5, 7. - min(t_in_shot, 6.), 1.); // vec3(1. + 3.*time, 1.5, 1);
    camProjectionRatio = 1.;
}

bool get_shot(inout float time, float duration) {
    if (time < duration) {
        return true;
    }
    time -= duration;
    return false;
}

void main() {
    gl_Position = a_position;
    float time = iTime;
    GenerateSpline(1.8/*curvature*/, 80./*scale*/, 2.+floor(iTime / 20)/*seed*/);

    camProjectionRatio = 1.;
    camFishEye = 0.1;
    camMotoSpace = 1.;
    camShowDriver = 1.;
    camFoV = atan(1. / camProjectionRatio);

    if (get_shot(time, 10.5)) {
        openingShot(time);
    } else if (get_shot(time, 6.)) {
        sideShotRear();
    } else if (get_shot(time, 5.)) {
        sideShotFront();
    } else if (get_shot(time, 8.)) {
        frontWheelCloseUpShot();
    } else if (get_shot(time, 8.)) {
        overTheHeadShot();
    } else if (get_shot(time, 8.)) {
        fpsDashboardShot();
    } else if (get_shot(time, 8.)) {
        dashBoardUnderTheShoulderShot(time);
    } else if (get_shot(time, 8.)) {
        viewFromBehind(time);
    } else if (get_shot(time, 8.)) {
        faceView(time);
    } else if (get_shot(time, 8.)) {
        sideShotRear();
    } else if (get_shot(time, 8.)) {
        sideShotFront();
    } else if (get_shot(time, 8.)) {
        frontWheelCloseUpShot();
    } else if (get_shot(time, 8.)) {
        overTheHeadShot();
    } else if (get_shot(time, 8.)) {
        fpsDashboardShot();
    } else if (get_shot(time, 8.)) {
        dashBoardUnderTheShoulderShot(time);
    } else if (get_shot(time, 8.)) {
        viewFromBehind(time);
    } else {
        overTheHeadShot();
    }
}

void main_old(void)
{
    gl_Position = a_position;
    float time = iTime;
    GenerateSpline(1.8/*curvature*/, 80./*scale*/, 2.+floor(iTime / 20)/*seed*/);

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
    float numberOfDifferentShots = 8.;

    float t = fract((iTime / shotDuration) / numberOfDifferentShots) * numberOfDifferentShots;
    int shot = int(t);
    float t_in_shot = fract(t);

    if (shot == 0)
    {
        openingShot(t_in_shot);
    }
    if (shot == 1)
    {
        sideShotRear();
    }
    if (shot == 2)
    {
        sideShotFront();
    }
    if (shot == 3)
    {
        frontWheelCloseUpShot();
    }
    if (shot == 4)
    {
        overTheHeadShot();
    }
    if (shot == 5)
    {
        fpsDashboardShot();
    }
    if (shot == 6)
    {
        dashBoardUnderTheShoulderShot(t_in_shot);
    }
    if (shot == 7)
    {
        viewFromBehind(t_in_shot);
    }
    if (shot == 8)
    {
        faceView(t_in_shot);
    }
    camFoV = atan(1. / camProjectionRatio);
}
