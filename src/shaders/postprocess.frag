#version 150
out vec4 fragColor;
#include "iResolution.inc.frag"

uniform sampler2D tex;

void main(void)
{
    vec2 rcpFrame = 1./iResolution;
    vec2 texcoord = gl_FragCoord.xy * rcpFrame;
    vec4 uv = vec4( texcoord, texcoord - (rcpFrame * 0.5));
    
    fragColor = texture(tex, uv.xy);
}