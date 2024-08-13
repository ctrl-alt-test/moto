#version 150

in vec4 a_position;

out vec3 sunDir;
out vec3 camPos;
out vec3 camTa;
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
    sunDir = normalize(vec3(3.5,1.,-1.));

    // list of camera shots
    if (time < 5.) {
        camPos = vec3(5.+time*.5, 2., 0.+time*.5);
        camTa = vec3(0., 2., 0.);
    } else {
        camPos = vec3(7.5, 2., 2.5);
        camTa = vec3(0., 2., 0.);
    }
}
