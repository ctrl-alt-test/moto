#version 150

in vec4 a_position;

out vec3 sunDir;
out vec3 camPos;
out vec3 camTa;
out float camFocal;

uniform float iTime;

const float INFINITE = 9e7;

void main(void)
{
    gl_Position = a_position;
    float time = iTime;
    
    camFocal = 2.;
    sunDir = normalize(vec3(3.5,1.,-1.));
    
    camPos = vec3(5.+time*.5, 2., 0.+time*.5);
    camTa = vec3(0., 2., 0.);
}
