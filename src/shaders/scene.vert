#version 150

in vec4 a_position;

out vec3 sunDir;
out vec3 camPos;
out vec3 camTa;
out float camMotoSpace; // bool - 0. or 1.
out float camFocal;
out float fishEyeFactor;

uniform float iTime;

const float INFINITE = 9e7;

void main(void)
{
    gl_Position = a_position;
    float time = iTime;
    
    camFocal = 2.;
    fishEyeFactor = 0.;
    camMotoSpace = 1.;

    // list of camera shots
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
}
