#version 150
out vec4 fragColor;
const vec2 iResolution = vec2(1920.,1080.);

//----------------------------------------------------------------------
// Vertex/Fragment IO
//----------------------------------------------------------------------
in vec3 anvilPos;
in vec3 sunDir;
in vec3 camPos;
in vec3 camTa;
in float camFocal;

uniform float iTime;


//----------------------------------------------------------------------
// Maths function
//----------------------------------------------------------------------

const float PI = acos(-1.);
const float INFINITE = 9e7;

mat3 lookat(vec3 ro, vec3 ta);
mat2 rot(float v);


// ---------------------------------------------
// Distance field toolbox
// ---------------------------------------------
float box(vec3 p, vec3 b);
float smin(float d1, float d2, float k);
float smax(float a, float b, float k);


// ---------------------------------------------
// Distance field 
// ---------------------------------------------
vec2 map(vec3 p);
float shadow( vec3 ro, vec3 rd);

// Materials
const float GROUND = 0.;
const float BLACK_METAL = 11.;

vec2 dmin(vec2 a, vec2 b) {
    return a.x<b.x ? a : b;
}

vec2 anvil(vec3 p) {
    p -= anvilPos;
    p.xz = rot(1.)*p.xz;
    float h = pow(clamp(p.y-1.,0.004,1.),.5);
    float d = box(p-vec3(0.,1.,0.), vec3(1.5-h,1.,2.5-h));
    if (d<10.) {
        d = min(d, box(p-vec3(0.,3.,0.), vec3(2.,1.,3.)));
        
        float d2 = length((p.yz-vec2(4.5,3.))*vec2(1.,.8))-2.;
        d2 = max(d2, abs(p.x)-.5);
        d2 = max(d2, p.y-3.5);
        d = min(d, d2);
        return vec2(d-.1, BLACK_METAL);
    }
    return vec2(INFINITE,GROUND);
}

vec2 map(vec3 p) {
    return dmin(vec2(p.y, GROUND), anvil(p));
}

float fastAO( in vec3 pos, in vec3 nor, float maxDist, float falloff ) {
    float occ1 = .5*maxDist - map(pos + nor*maxDist *.5).x;
    float occ2 = .95*(maxDist - map(pos + nor*maxDist).x);
    return clamp(1. - falloff*1.5*(occ1 + occ2), 0., 1.);
}

float trace(vec3 ro, vec3 rd) {
    float t = 0.01;
    for(int i=0; i<128; i++) {
        float d = map(ro+rd*t).x;
        t += d;
        if (t > 100. || abs(d) < 0.001) break;
    }
    
    return t;
}

void main()
{
    vec2 uv = (gl_FragCoord.xy) / iResolution;
    vec2 v = uv*2.-1.;
    v.x *= iResolution.x / iResolution.y;
        
    // Setup ray
    vec3 ro = camPos;
    vec3 ta = camTa;
    vec3 rd = lookat(ro, ta) * normalize(vec3(v,camFocal));
        
    // Trace : intersection point + normal
    float t = trace(ro,rd);
    vec3 p = ro + rd * t;
    vec2 dmat = map(p);
    vec2 eps = vec2(0.0001,0.0);
    vec3 n = normalize(vec3(dmat.x - map(p - eps.xyy).x, dmat.x - map(p - eps.yxy).x, dmat.x - map(p - eps.yyx).x));
    
    
    // ----------------------------------------------------------------
    // Shade
    // ----------------------------------------------------------------
    
    float ao = fastAO(p, n, .15, 1.) * fastAO(p, n, 1., .1)*.5;
    
    float shad = shadow(p, sunDir);
    float fre = 1.0+dot(rd,n);
    
    vec3 diff = vec3(1.,.8,.7) * max(dot(n,sunDir), 0.) * pow(vec3(shad), vec3(1.,1.2,1.5));
    vec3 bnc = vec3(1.,.8,.7)*.1 * max(dot(n,-sunDir), 0.) * ao;
    vec3 sss = vec3(.5) * mix(fastAO(p, rd, .3, .75), fastAO(p, sunDir, .3, .75), 0.5);
    vec3 spe = vec3(1.) * max(dot(reflect(rd,n), sunDir),0.);
    vec3 envm = vec3(0.);
    
    vec3 amb = vec3(.4,.45,.5)*1. * ao;
    vec3 emi = vec3(0.);
    
    vec3 albedo = vec3(0.);
    if(dmat.y == GROUND) {
        albedo = vec3(3.);
        sss *= 0.;
        spe *= 0.;
    } else if(dmat.y == BLACK_METAL) {
        albedo = vec3(1.);
        diff *= vec3(.1)*fre;
        amb *= vec3(.1)*fre;
        bnc *= 0.;
        sss *= 0.;
        spe = pow(spe, vec3(100.))*fre*2.;
    }
    
    // fog
    vec3 skyColor = vec3(0.7,0.8,1.0);
    vec3 col = clamp(
        mix(
            (albedo * (amb*1. + diff*.5 + bnc*2. + sss*2. ) + envm + spe*shad + emi),
            skyColor,
            smoothstep(90.,100.,t)),
        0., 1.);

    // ----------------------------------------------------------------
    // Post processing pass
    // ----------------------------------------------------------------

    const float endTime = 146.;
    // gamma correction & color grading
    col = pow(pow(col, vec3(1./2.2)), vec3(1.0,1.05,1.1));
    
    // Circle to black
    float circle = length(gl_FragCoord.xy/iResolution.xx - vec2(.5,.3));
    float tt = max(.137, smoothstep(endTime+1., endTime, iTime));
    col *= smoothstep(tt, tt-.005, circle);
 
    // vignetting
    fragColor = vec4(col / (1.+pow(length(uv*2.-1.),4.)*.04),1.);
}



// ---------------------------------------------
// Raytracing toolbox
// ---------------------------------------------

// https://www.shadertoy.com/view/lsKcDD
float shadow( vec3 ro, vec3 rd)
{
    float res = 1.0;
    float t = 0.08;
    for( int i=0; i<64; i++ )
    {
        float h = map( ro + rd*t ).x;
        res = min( res, 30.0*h/t );
        t += h;
        
        if( res<0.0001 || t>50. ) break;
        
    }
    return clamp( res, 0.0, 1.0 );
}

// ---------------------------------------------
// Math
// ---------------------------------------------
mat3 lookat(vec3 ro, vec3 ta)
{
    const vec3 up = vec3(0.,1.,0.);
    vec3 fw = normalize(ta-ro);
    vec3 rt = normalize( cross(fw, normalize(up)) );
    return mat3( rt, cross(rt, fw), fw );
}

mat2 rot(float v) {
    float a = cos(v);
    float b = sin(v);
    return mat2(a,b,-b,a);
}

// ---------------------------------------------
// Distance field toolbox
// ---------------------------------------------
float box( vec3 p, vec3 b )
{
    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float smin(float d1, float d2, float k)
{
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h);
}

float smax(float a, float b, float k)
{
    k *= 1.4;
    float h = max(k-abs(a-b),0.0);
    return max(a, b) + h*h*h/(6.0*k*k);
}
