#version 150

in vec4 a_position;

out vec3 sunDir;
out vec3 camPos;
out vec3 camTa;
out float camProjectionRatio;
out float camFishEye;
out float camMotoSpace; // bool - 0. or 1.

uniform float iTime;

const float INFINITE = 9e7;

float hash(float x) { return fract(sin(x) * 43758.5453); }
vec2 hash2(float x) { float h = hash(x); return vec2(h, hash(h)); }

vec2 valueNoise(float p)
{
    float p0 = floor(p);
    float p1 = p0 + 1.;

    vec2 v0 = hash2(p0);
    vec2 v1 = hash2(p1);

    float fp = p - p0;
    fp = fp*fp * (3.0 - 2.0 * fp);

    return mix(v0, v1, fp);
}

void sideShotFront()
{
    vec2 p = vec2(0.95, 0.5);
    p.x += mix(-0.5, 1., valueNoise(0.5*iTime).y);
    p.y += 0.05 * valueNoise(6.*iTime).x;
    camPos = vec3(p, 2.8);
    camTa = vec3(p.x, p.y + 0.1, 0.);
    camProjectionRatio = 2.;
}

void sideShotRear()
{
    vec2 p = vec2(-1., 0.5);
    p.x += mix(-1.2, 0.5, valueNoise(0.5*iTime).y);
    p.y += 0.05 * valueNoise(6.*iTime).x;
    camPos = vec3(p, 2.8);
    camTa = vec3(p.x, p.y + 0.1, 0.);
    camProjectionRatio = 2.;
}

void main(void)
{
    gl_Position = a_position;
    float time = iTime;

    camProjectionRatio = 1.;
    camFishEye = 0.1;
    camMotoSpace = 1.;

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
    if (int(iTime / 4.) % 2 == 0)
        sideShotFront();
    else
        sideShotRear();
}
