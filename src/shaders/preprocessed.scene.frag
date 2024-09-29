#version 150


const bool ENABLE_STOCHASTIC_MOTION_BLUR = true;








const int MAX_LIGHTS = 3;
const int MAX_RAY_MARCH_STEPS = 200;
const float MAX_RAY_MARCH_DIST = 100.0;
const int MAX_SHADOW_STEPS = 30;
const float MAX_SHADOW_DIST = 5.0;
const float NORMAL_DP = 2.*1e-3;
const float BOUNCE_OFFSET = 1e-3;
const float GAMMA = 2.2;
const vec2 iResolution = vec2(1920.,1080.);
const int SPLINE_SIZE = 13;
const float INF = 1e6;


uniform float iTime;
uniform sampler2D tex;



in vec3 camPos;
in vec3 camTa;
in float camMotoSpace;
in float camFoV;
in float camProjectionRatio;
in float camFishEye;
in float camShowDriver;
in vec2 spline[SPLINE_SIZE];


out vec4 fragColor;


float PIXEL_ANGLE = camFoV / iResolution.x;
float time;
#define ZERO(iTime) min(0, int(iTime))

// Include begin: common.frag
// --------------------------------------------------------------------
const bool ENABLE_SMOOTHER_STEP_NOISE = false;
const float PI = acos(-1.);






vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d)
{
    return a + b * cos(2. * PI * (c * t + d));
}




// Inactive conditional block: #ifdef ENABLE_STEP_COUNT





struct light
{
    vec3 P; 
    vec3 Q; 
    vec3 C; 
    float A; 
    float B; 
    float F; 
    float I; 
};

const int MATERIAL_TYPE_DIELECTRIC = 0;
const int MATERIAL_TYPE_METALLIC = 1;
const int MATERIAL_TYPE_EMISSIVE = 2;
const int MATERIAL_TYPE_RETROREFLECTIVE = 3;

struct material
{
    int T; 
    vec3 C; 
    float R; 
};




float invV1(float NdotV, float sqrAlpha)
{
    return NdotV + sqrt(sqrAlpha + (1.0 - sqrAlpha) * NdotV * NdotV);
}

vec3 cookTorrance(
    vec3 f0,
	float roughness,
	vec3 NcrossH,
	float VdotH,
    float NdotL,
    float NdotV)
{
	float alpha = roughness * roughness;
    float sqrAlpha = alpha * alpha;

    
	float distribution = dot(NcrossH, NcrossH) * (1. - sqrAlpha) + sqrAlpha;
	float D = sqrAlpha / (PI * distribution * distribution);

    
    float V = 1.0 / (invV1(NdotV, sqrAlpha) * invV1(NdotL, sqrAlpha));

    
	float x = 1. - VdotH;
	vec3 F = x + f0 * (1. - x*x*x*x*x);

	return F * D * V;
}

vec3 lightContribution(light l, vec3 p, vec3 V, vec3 N, vec3 albedo, vec3 f0, float roughness)
{
    vec3 L, irradiance;
    float NdotL;

    if (l.A != -1.0)
    {
        
        
        
        vec3 L0 = l.P - p;
        float d0 = length(L0);
        L = L0 / d0;

        NdotL = dot(N, L);

        float LdotD = dot(L, -l.Q);
        vec3 freq = 180. + vec3(0., .4, .8);
        float angleFallOff = smoothstep(l.A, l.B, LdotD);
        angleFallOff *= angleFallOff;
        angleFallOff *= angleFallOff;
        angleFallOff *= angleFallOff;

        vec3 radiant_intensity = l.C * l.I;
        radiant_intensity *= (sin(freq/LdotD) * 0.5 + 0.5) * 0.2 + 0.8;
        radiant_intensity *= angleFallOff;
        radiant_intensity *= (1.0 + l.F) / (l.F + d0 * d0);
        irradiance = radiant_intensity * NdotL;
    }
    else
    {
        
        
        
        vec3 L0 = l.P - p;
        vec3 L1 = l.Q - p;
        float d0 = length(L0);
        float d1 = length(L1);

        
        
        
        
        
        
        float falloff = 1.0;

        
        
    
        float NdotL0 = dot(N, L0) / d0;
        float NdotL1 = dot(N, L1) / d1;

        float angularContribution = 2. * clamp((NdotL0 + NdotL1) / 2., 0., 1.);
        float geometricAttenuation = d0 * d1 + dot(L0, L1);
        float contribution = angularContribution / geometricAttenuation;
        
        
        
        

        contribution *= falloff;
        if (contribution <= 0.)
        {
            return vec3(0.);
        }

        vec3 radiant_intensity = l.C * l.I;
        irradiance = radiant_intensity * contribution;

        vec3 Ld = l.Q - l.P;
        vec3 R = reflect(-V, N);
        float RdotLd = dot(R, Ld);
    
        float t = clamp((dot(R, L0) * RdotLd - dot(L0, Ld)) / (dot(Ld, Ld) - RdotLd * RdotLd), 0., 1.);

        
        L = normalize(mix(L0, L1, t));
        NdotL = dot(N, L);
    }

    if (NdotL <= 0.)
        return vec3(0.);

    vec3 H = normalize(L + V);
    vec3 NcrossH = cross(N, H);
    float VdotH = clamp(dot(V, H), 0., 1.);
    float NdotV = clamp(dot(N, V), 0., 1.);

    vec3 spec = cookTorrance(f0, roughness, NcrossH, VdotH, NdotL, NdotV);

    return irradiance * (albedo + spec);
}




float hash11(float x) { return fract(sin(x) * 43758.5453); }
float hash21(vec2 xy) { return fract(sin(dot(xy, vec2(12.9898, 78.233))) * 43758.5453); }
float hash31(vec3 xyz) { return hash21(vec2(hash21(xyz.xy), xyz.z)); }
vec2 hash22(vec2 xy) { return fract(sin(vec2(dot(xy, vec2(127.1,311.7)), dot(xy, vec2(269.5,183.3)))) * 43758.5453); }

float valueNoise(vec2 p)
{
    vec2 p00 = floor(p);
    vec2 p10 = p00 + vec2(1.0, 0.0);
    vec2 p01 = p00 + vec2(0.0, 1.0);
    vec2 p11 = p00 + vec2(1.0, 1.0);

    float v00 = hash21(p00);
    float v10 = hash21(p10);
    float v01 = hash21(p01);
    float v11 = hash21(p11);

    vec2 fp = p - p00;
    if (ENABLE_SMOOTHER_STEP_NOISE)
    {
        fp = fp*fp*fp* (fp* (fp * 6.0 - 15.0) + 10.0);
    }
    else
    {
        fp = fp*fp * (3.0 - 2.0 * fp);
    }

    return mix(
        mix(v00, v10, fp.x),
        mix(v01, v11, fp.x),
    fp.y);
}

float fBm(vec2 p, int iterations, float weight_param, float frequency_param)
{
    float v = 0.;
    float weight = 1.0;
    float frequency = 1.0;
    float offset = 0.0;

    for (int i = ZERO(iTime); i < iterations; ++i)
    {
        float noise = valueNoise(p * frequency + offset) * 2. - 1.;
        v += weight * noise;
        weight *= clamp(weight_param, 0., 1.);
        frequency *= 1.0 + 2.0 * clamp(frequency_param, 0., 1.);
        offset += 1.0;
    }
    return v;
}




float smin(float a, float b, float k)
{
    k /= 1.0 - sqrt(0.5);
    return max(k, min(a, b)) - length(max(k - vec2(a, b), 0.0));
}

float Box2(vec2 p, vec2 size, float corner)
{
   p = abs(p) - size + corner;
   return length(max(p, 0.)) + min(max(p.x, p.y), 0.) - corner;
}

float Box3(vec3 p, vec3 size, float corner)
{
   p = abs(p) - size + corner;
   return length(max(p, 0.)) + min(max(max(p.x, p.y), p.z), 0.) - corner;
}

float Ellipsoid( in vec3 p, in vec3 r )
{
    float k0 = length(p/r);
    float k1 = length(p/(r*r));
    return k0*(k0-1.0)/k1;
}

float Segment3(vec3 p, vec3 a, vec3 b, out float h)
{
	vec3 ap = p - a;
	vec3 ab = b - a;
	h = clamp(dot(ap, ab) / dot(ab, ab), 0., 1.);
	return length(ap - ab * h);
}

float Capsule(vec3 p, float h, float r)
{
    p.y += clamp(-p.y, 0., h);
    return length(p) - r;
}

float Torus( vec3 p, vec2 t )
{
    return length( vec2(length(p.xz)-t.x,p.y) )-t.y;
}

mat2 Rotation(float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, s, -s, c);
}

float DistanceFromAABB(vec2 p, vec4 aabb)
{
    vec2 center = (aabb.xy + aabb.zw) / 2.0;
    vec2 size = aabb.zw - aabb.xy;

    return Box2(p - center, size / 2.0, 0.0);
}








vec2 Bezier(vec2 A, vec2 B, vec2 C, float t)
{
    vec2 AB = mix(A, B, t);
    vec2 BC = mix(B, C, t);
    return mix(AB, BC, t);
}



vec4 FindCubicRoots(float a, float b, float c)
{
    float p = b - a * a / 3.0, p3 = p * p * p;
    float q = a * (2.0 * a * a - 9.0 * b) / 27.0 + c;
    float d = q * q + 4.0 * p3 / 27.0;
    float offset = -a / 3.0;
    if (d >= 0.0) {
        float z = sqrt(d);
        vec2 x = (vec2(z, -z) - q) / 2.0;
        vec2 uv = sign(x) * pow(abs(x), vec2(1.0, 1.0) / 3.0);
        return vec4(offset + uv.x + uv.y, 0, 0, 1.0);
    }

    float v = acos(-sqrt(-27.0 / p3) * q / 2.0) / 3.0;
    float m = cos(v);
    float n = sin(v) * sqrt(3.0);
    return vec4(vec3(m + m, -n - m, n - m) * sqrt(-p / 3.0) + offset, 3.0);
}


float GetWinding(vec2 a, vec2 b)
{
    return 2.0 * step(a.x * b.y, a.y * b.x) - 1.0;
}



vec2 BezierSDF(vec2 A, vec2 B, vec2 C, vec2 p)
{
    vec2 a = B - A, b = A - B * 2.0 + C, c = a * 2.0, d = A - p;
    float dotbb = dot(b, b);

    vec3 k = vec3(3.0 * dot(a, b), 2. * dot(a, a) + dot(d, b), dot(d, a)) / dotbb;
    vec4 t = FindCubicRoots(k.x, k.y, k.z);
    vec2 tsat = clamp(t.xy, 0., 1.);

    vec2 dp1 = d + (c + b * tsat.x) * tsat.x;
    float d1 = dot(dp1, dp1);
    vec2 dp2 = d + (c + b * tsat.y) * tsat.y;
    float d2 = dot(dp2, dp2);

    
    vec4 r = (d1 < d2) ? vec4(d1, t.x, dp1) : vec4(d2, t.y, dp2);

    
    float s = GetWinding(r.zw, 2.0 * b * r.y + c);
    
    return vec2(s * sqrt(r.x), r.y);
}




float BezierCurveLengthAt(vec2 A, vec2 B, vec2 C, float t)
{
    
    

    
    
    vec2 a1 = 2.0 * (A - 2.0 * B + C);
    vec2 b1 = 2.0 * (B - A);

    
    
    float k1 = dot(a1, a1);
    float k2 = 2.0 * dot(a1, b1);
    float k3 = dot(b1, b1);

    
    
    float k4 = 0.5 * k2 / k1;
    float k5 = k3 / k1 - k4 * k4;

    
    
    
    
    vec2 ti = vec2(0.0, t); 
    vec2 vt = sqrt((ti + k4) * (ti + k4) + k5);
    vec2 sdft = sqrt(k1) * 0.5 * ((k4 + ti) * vt + k5 * log(abs(k4 + ti + vt)));
    return sdft.y - sdft.x;
}



vec4 BezierAABB(vec2 A, vec2 B, vec2 C)
{
    
    vec4 res = vec4(min(A, C), max(A, C));

    
    if (B.x < res.x || B.x > res.z ||
        B.y < res.y || B.y > res.w)
    {
        
        
        

        vec2 t = clamp((A - B) / (A - 2.0*B+C),0.0,1.0);
        vec2 s = 1.0 - t;
        vec2 q = s*s*A + 2.0*s*t*B + t*t*C;
        
        res.xy = min(res.xy, q);
        res.zw = max(res.zw, q);
    }
    
    return res;
}






vec2 MinDist(vec2 d1, vec2 d2)
{
    return d1.x < d2.x ? d1 : d2;
}




 

void setupCamera(vec2 uv, vec3 cameraPosition, vec3 cameraTarget, vec3 cameraUp, out vec3 ro, out vec3 rd)
{
    vec3 cameraForward = normalize(cameraTarget - cameraPosition);
    if (abs(dot(cameraForward, cameraUp)) > 0.99)
    {
        cameraUp = vec3(1., 0., 0.);
    }
    vec3 cameraRight = normalize(cross(cameraForward, cameraUp));
    cameraUp = normalize(cross(cameraRight, cameraForward));

    
    uv *= mix(1., length(uv), camFishEye);
    ro = cameraPosition;
    rd = normalize(cameraForward * camProjectionRatio + uv.x * cameraRight + uv.y * cameraUp);
}


// --------------------------------------------------------------------
// Include end: common.frag

// Include begin: ids.frag
// --------------------------------------------------------------------
const float NO_ID = -1.;
const float GROUND_ID = 0.;
const float MOTO_ID = 1.;
const float MOTO_HEAD_LIGHT_ID = 2.;
const float MOTO_BREAK_LIGHT_ID = 3.;
const float MOTO_WHEEL_ID = 4.;
const float MOTO_MOTOR_ID = 5.;
const float MOTO_EXHAUST_ID = 6.;
const float MOTO_DRIVER_ID = 7.;
const float MOTO_DRIVER_HELMET_ID = 8.;
const float CITY_ID = 9.;
const float ROAD_REFLECTOR_ID = 10.;
// Inactive conditional block: #ifdef DEBUG


bool IsMoto(float mid)
{
    return mid >= MOTO_ID && mid <= MOTO_DRIVER_HELMET_ID;
}
// --------------------------------------------------------------------
// Include end: ids.frag

// Include begin: backgroundContent.frag
// --------------------------------------------------------------------
// Inactive conditional block: #ifdef ENABLE_DAY_MODE
// Active conditional block: #else
vec3 nightHorizonLight = 0.01 * vec3(0.07, 0.1, 1.0);
vec3 moonLightColor = vec3(0.2, 0.8, 1.0);
// End of active block.

vec3 moonDirection = normalize(vec3(-1., 0.3, 0.4));

vec3 sky(vec3 V)
{
    vec3 darkest = vec3(1., 4.6, 97.);
    vec3 deepest = vec3(1., 282., 1777.);
    vec3 clearest = vec3(34., 728., 1910.);
    vec3 greenest = vec3(332., 1190., 1777.);
    vec3 yellow = vec3(990., 1527., 1297.);
    vec3 red = vec3(898., 898., 810.);
    vec3 fog = vec3(126., 387., 728.);

    float direction = clamp(dot(V, normalize(vec3(0., 1., 0.25))), 0., 1.);
    vec3 color = mix(clearest, darkest, pow(direction, 0.5));

    color = mix(greenest, color, smoothstep(0., 0.3, mix(V.y, direction, 0.5)));
    color = mix(yellow, color, smoothstep(0., 0.15, mix(V.y, direction, 0.2)));
    color = mix(fog, color, smoothstep(0.05, 0.15, V.y));

    float moon = clamp(dot(V, moonDirection), 0., 1.);
    float dmoon = 2.*fwidth(moon);
    moon = smoothstep(-dmoon, dmoon, moon - 0.9999);

    if (moon > 0.)
    {
        float pattern = smoothstep(-0.5, 0.5, fBm(V.xy * 100., 4, 0.65, 0.7) + 0.13);
        vec3 moonColor = clearest * 5. * mix(1., 2., pattern);
        color = mix(color, moonColor, 0.9 * moon);
    }

    float cloud = fBm(0.015*time+V.xz/(0.01 + V.y) * 0.5, 5, 0.55, 0.7);
    cloud = smoothstep(0., 1., cloud+1.);

    color *= mix(0.1, 1., pow(cloud, 2.));
    color *= smoothstep(-0.1, 0., V.y);
    return color / 197000.;
}

vec3 cityLights(vec2 p)
{
    vec3 ctex=vec3(0);
    for(int i=0;i<3;i++) {
        float fi=float(i);
        vec2 xp=p*Rotation(max(fi-3.,0.)*.5)*(1.+fi*.3),mp=mod(xp,10.)-5.;
        
        float a = smoothstep(.6+fi*.1,0.,min(abs(mp.x),abs(mp.y)))*max(
            smoothstep(.7+fi*.1,.5,length(mod(p,2.)-1.))*smoothstep(.5,.7,valueNoise(p)-.15)
            ,pow(valueNoise(xp*.5),10.)
        );
        ctex += valueNoise(xp*.5)*mix(
            mix(vec3(.56,.32,.18)*min(a,.5)*2.,vec3(.88,.81,.54),max(a-.5,0.)*2.),
            mix(vec3(.45,.44,.6)*min(a,.5)*2.,vec3(.80,.89,.93),max(a-.5,0.)*2.),
            step(.5,valueNoise(p*2.))
        );
    }
    return ctex*5.;
}

vec2 cityShape(vec3 p){
    vec3 o=p;
    
    float len = Box3(p - vec3(150, 0, 0), vec3(1., 200., 200.), 0.01);
    if (len > 10.) return vec2(len-5., CITY_ID);

    
    float seed=hash21(floor(o.xz/14.));
    p.xz=mod(p.xz*Rotation(.7)+seed*(6.-3.)*5.,14.)-7.;
    float buildingCutouts = max(max(abs(p.x),abs(p.z))-2.,p.y-seed*5.);
    p.xz=mod(o.xz+6.,14.)-7.;
    buildingCutouts = min(buildingCutouts,max(max(abs(p.x),abs(p.z))-2.,p.y-seed*5.));
    return
        vec2(max(min(buildingCutouts*.5,p.y),o.z),
            CITY_ID);
}
// --------------------------------------------------------------------
// Include end: backgroundContent.frag

// Include begin: roadContent.frag
// --------------------------------------------------------------------
vec4 splineAABB;
vec4 splineSegmentAABBs[SPLINE_SIZE / 2];
vec2 splineSegmentDistances[SPLINE_SIZE / 2]; 






void ComputeBezierSegmentsLengthAndAABB()
{
    float splineLength = 0.0;
    splineAABB = vec4(INF, INF, -INF, -INF);

    for (int i = ZERO(iTime); i < SPLINE_SIZE / 2; ++i)
    {
        int index = 2 * i;
        vec2 A = spline[index + 0];
        vec2 B = spline[index + 1];
        vec2 C = spline[index + 2];
        float segmentLength = BezierCurveLengthAt(A, B, C, 1.0);
        splineSegmentDistances[i].x = splineLength;
        splineLength += segmentLength;
        splineSegmentDistances[i].y = splineLength;

        vec4 AABB = BezierAABB(A, B, C);
        splineSegmentAABBs[i] = AABB;
        splineAABB.xy = min(splineAABB.xy, AABB.xy);
        splineAABB.zw = max(splineAABB.zw, AABB.zw);
    }
}












vec4 ToSplineLocalSpace(vec2 p, float splineWidth)
{
    vec4 splineUV = vec4(INF, 0, 0, 0);

    if (DistanceFromAABB(p, splineAABB) > splineWidth)
    {
        return splineUV;
    }

    
    for (int i = ZERO(iTime); i < SPLINE_SIZE / 2; ++i)
    {
        int index = 2 * i;
        vec2 A = spline[index + 0];
        vec2 B = spline[index + 1];
        vec2 C = spline[index + 2];

        if (DistanceFromAABB(p, BezierAABB(A, B, C)) > splineWidth)
        {
            continue;
        }

        
        B = mix(B + vec2(1e-4), B, abs(sign(B * 2.0 - A - C))); 
        
        vec2 bezierSDF = BezierSDF(A, B, C, p);

        if (abs(bezierSDF.x) < abs(splineUV.x))
        {
            float lengthInSegment = BezierCurveLengthAt(A, B, C, clamp(bezierSDF.y, 0., 1.));
            float lengthInSpline = splineSegmentDistances[i].x + lengthInSegment;
            splineUV = vec4(
                bezierSDF.x,
                clamp(bezierSDF.y, 0., 1.),
                lengthInSpline,
                float(index));
        }
    }

    return splineUV;
}









vec2 GetPositionOnSplineFromIndex(vec2 spline_t_and_index)
{
    float t = spline_t_and_index.x;
    int index = int(spline_t_and_index.y);
    vec2 A = spline[index + 0];
    vec2 B = spline[index + 1];
    vec2 C = spline[index + 2];

    return Bezier(A, B, C, t);
}





vec2 GetPositionOnSpline(float t)
{
    
    float targetLength = t * splineSegmentDistances[SPLINE_SIZE / 2 - 1].y;
    
    
    int index = 0;
    while (index < SPLINE_SIZE / 2 && targetLength > splineSegmentDistances[index].y)
    {
        ++index;
    }

    float segmentStartDistance = splineSegmentDistances[index].x;
    float segmentEndDistance = splineSegmentDistances[index].y;
    
    
    float segmentT = (targetLength - segmentStartDistance) / (segmentEndDistance - segmentStartDistance);

    return GetPositionOnSplineFromIndex(vec2(segmentT, index * 2.0));
}




vec3 roadWidthInMeters = vec3(4.0, 8.0, 8.0);
const float laneWidth = 3.5;

float roadMarkings(vec2 uv, float width, vec2 params)
{
    
    vec2 t1  = vec2(26.0 / 2.0, 3.0);
    vec2 t1b = vec2(26.0 / 4.0, 1.5);
    vec2 t2  = vec2(26.0 / 4.0, 3.0);
    vec2 t3  = vec2(26.0 / 6.0, 3.0);
    vec2 t3b = vec2(26.0 / 1.0, 20.0);
    vec2 continuous = vec2(100.0, 100.0);

    vec2 separationLineParams = t1;
    if (params.x > 0.25) separationLineParams = t1b;
    if (params.x > 0.50) separationLineParams = t3;
    if (params.x > 0.75) separationLineParams = continuous;

    vec2 sideLineParams = t2;
    if (width > 4.0) sideLineParams = t3b;

    float tileY = uv.y - floor(clamp(uv.y, 3.5-width, width) / 3.5) * 3.5;
    vec2 separationTileUV = vec2(fract(uv.x / separationLineParams.x) * separationLineParams.x, tileY);
    vec2 sideTileUV = vec2(fract((uv.x + 0.4) / sideLineParams.x) * sideLineParams.x, uv.y);

    float sideLine1 = Box2(sideTileUV - vec2(0.5 * sideLineParams.y, width), vec2(0.5 * sideLineParams.y, 0.10), 0.03);
    float sideLine2 = Box2(sideTileUV - vec2(0.5 * sideLineParams.y, -width), vec2(0.5 * sideLineParams.y, 0.10), 0.03);

    float separationLine1 = Box2(separationTileUV - vec2(0.5 * separationLineParams.y, 0.0), vec2(0.5 * separationLineParams.y, 0.10), 0.01);

    float pattern = min(min(sideLine1, sideLine2), separationLine1);

    return 1.-smoothstep(-0.01, 0.01, pattern);
}

material roadMaterial(vec2 uv, float width, vec2 params)
{
    vec2 laneUV = uv / laneWidth;

    float tireTrails = sin((laneUV.x-0.125) * 4. * PI) * 0.5 + 0.5;
    tireTrails = mix(tireTrails, smoothstep(0., 1., tireTrails), 0.25);

    float largeScaleNoise = smoothstep(-0.25, 1., fBm(laneUV * vec2(15., 0.1), 2, .7, .4));
    tireTrails = mix(tireTrails, largeScaleNoise, 0.2);

    float highFreqNoise = fBm(laneUV * vec2(150., 6.), 1, 1., 1.);
    tireTrails = mix(tireTrails, highFreqNoise, 0.1);

    float roughness = mix(0.8, 0.4, tireTrails);
    vec3 color = vec3(mix(vec3(0.11, 0.105, 0.1), vec3(0.15), tireTrails));


    float paint = roadMarkings(uv.yx, width, params);
    color = mix(color, vec3(0.5), paint);
    roughness = mix(roughness, 0.7, paint);

    
     
    

    return material(paint > 0.5 ? MATERIAL_TYPE_RETROREFLECTIVE : MATERIAL_TYPE_DIELECTRIC, color, roughness);
}

const float terrain_fBm_weight_param = 0.6;
const float terrain_fBm_frequency_param = 0.5;

float smoothTerrainHeight(vec2 p)
{
    float hillHeightInMeters = 200.;
    float hillLengthInMeters = 2000.;

    return 0.5 * hillHeightInMeters * fBm(p * 2. / hillLengthInMeters, 3, terrain_fBm_weight_param, terrain_fBm_frequency_param);
}

float terrainDetailHeight(vec2 p)
{
    float detailHeightInMeters = 1.;
    float detailLengthInMeters = 100.;

    return 0.5 * detailHeightInMeters * fBm(p * 2. / detailLengthInMeters, 1, terrain_fBm_weight_param, terrain_fBm_frequency_param);
}

float roadBumpHeight(float d)
{
    float x = clamp(abs(d / roadWidthInMeters.x), 0., 1.);
    return 0.2 * (1. - x * x * x);
}

vec2 roadSideItems(vec4 splineUV, float relativeHeight) {
    vec3 pRoad = vec3(abs(splineUV.x), relativeHeight, splineUV.z);

	
    vec3 pObj = vec3(pRoad.x - 4.2, pRoad.y - 0.8, 0.);
    float len = Box3(pObj, vec3(0.1, 0.2, 0.1), 0.05);

    pObj = vec3(pRoad.x - 4.1, pRoad.y - 0.8, 0.);
    len = max(len, -Box3(pObj, vec3(0.1, 0.1, 0.1), 0.1));

    pObj = vec3(pRoad.x - 4.3, pRoad.y - 0.5, round(pRoad.z * 0.5) / 0.5 - pRoad.z);
    len = min(len, Box3(pObj, vec3(0.05, 0.5, 0.05), 0.01));

    float reflector = Box3(pObj - vec3(-0.1, 0.3, 0.0), vec3(0.04, 0.06, 0.03), 0.01);
    vec2 res = MinDist(vec2(len, MOTO_EXHAUST_ID), vec2(reflector, ROAD_REFLECTOR_ID));

    
    pObj = vec3(pRoad.x - 4.5, pRoad.y - 1.5, round(pRoad.z / 30.) * 30. - pRoad.z);
    len = Box3(pObj, vec3(0.1, 3., 0.1), 0.1);
    res = MinDist(res, vec2(len, MOTO_EXHAUST_ID));

    pObj = vec3(pRoad.x - 4.3, pRoad.y - 4., pObj.z);
    len = Box3(pObj, vec3(0.2, 0.1, 0.1), 0.1);
    res = MinDist(res, vec2(len, MOTO_BREAK_LIGHT_ID)); 

    return res;
}

vec2 terrainShape(vec3 p, vec4 splineUV)
{
    float heightToDistanceFactor = 0.75;

    
    float terrainHeight = smoothTerrainHeight(p.xz);
    float relativeHeight = p.y - terrainHeight;

    
    if (relativeHeight > 5.5)
    {
        return vec2(heightToDistanceFactor * relativeHeight, GROUND_ID);
    }

    
    float isRoad = 1.0 - smoothstep(roadWidthInMeters.x, roadWidthInMeters.y, abs(splineUV.x));

    
    if (isRoad < 1.0)
    {
        terrainHeight += terrainDetailHeight(p.xz);
    }

    
    float roadHeight = terrainHeight;
    if (isRoad > 0.0)
    {
        
        vec2 positionOnSpline = GetPositionOnSplineFromIndex(splineUV.yw);

        
        roadHeight = smoothTerrainHeight(positionOnSpline);
        roadHeight += roadBumpHeight(splineUV.x);
    }

    
    float height = mix(terrainHeight, roadHeight, isRoad);

    relativeHeight = p.y - height;
    
    vec2 d = vec2(heightToDistanceFactor * relativeHeight, GROUND_ID);

    d = MinDist(d, roadSideItems(splineUV, p.y - roadHeight));
    return d;
}

float tree(vec3 globalP, vec3 localP, vec2 id, vec4 splineUV, float current_t) {
    float h1 = hash21(id);
    float h2 = hash11(h1);

    
    float presence = smoothstep(-0.7, 0.7, fBm(id / 500., 2, 0.5, 0.3));
    if (h1 < presence)
    {
        return INF;
    }

    
    if (abs(splineUV.x) < roadWidthInMeters.y) return INF;

    
    
    
    
    
    
    
    
    
    

    float treeHeight = mix(5., 20., 1.-h1*h1);
    float treeWidth = treeHeight * mix(0.3, 0.5, h2*h2);
    float terrainHeight = smoothTerrainHeight(id);

    localP.y -= terrainHeight + 0.5 * treeHeight;
    localP.xz += (vec2(h1, h2)*2. - 1.) * 2.;

    float d = Ellipsoid(localP, 0.5*vec3(treeWidth, treeHeight, treeWidth));

    float leaves = 1. - smoothstep(50., 200., current_t);
    if (d < 2. && leaves > 0.)
    {
        d += leaves * fBm(5. * vec2(2.*atan(localP.z, localP.x), localP.y) + id, 2, 0.5, 0.5) * 0.5;
    }

    return d;
}

vec2 treesShape(vec3 p, vec4 splineUV, float current_t)
{
    float spacing = 10.;

    
    
    vec2 id = round(p.xz / spacing) * spacing;
    vec3 localP = p;
    localP.xz -= id;
    return vec2(tree(p, localP, id, splineUV, current_t), GROUND_ID);
}
// --------------------------------------------------------------------
// Include end: roadContent.frag

// Include begin: motoContent.frag
// --------------------------------------------------------------------
vec3 motoPos;
vec3 motoDir;
vec3 headLightOffsetFromMotoRoot = vec3(0.53, 0.98, 0.0);
vec3 breakLightOffsetFromMotoRoot = vec3(-1.14, 0.55, 0.0);
vec3 dirHeadLight = normalize(vec3(1.0, -0.15, 0.0));
vec3 dirBreakLight = normalize(vec3(-1.0, -0.5, 0.0));
float motoYaw;
float motoPitch;
float motoRoll;

















void computeMotoPosition()
{
    float distanceOnCurve = fract(time*0.1);

    vec3 nextPos = motoPos *= 0.;
    motoPos.xz = GetPositionOnSpline(distanceOnCurve);
    motoPos.y = smoothTerrainHeight(motoPos.xz);

    nextPos.xz = GetPositionOnSpline(distanceOnCurve + 0.0001);
    nextPos.y = smoothTerrainHeight(nextPos.xz);

    motoDir = normalize(nextPos - motoPos);

    vec2 motoRight = vec2(-motoDir.z, motoDir.x);
    float rightOffset = 2.0 + 1.5*sin(time);
    motoPos.xz += motoRight * rightOffset;
    motoPos.y += roadBumpHeight(abs(rightOffset));

    motoYaw = atan(motoDir.z, motoDir.x);
    motoPitch = atan(motoDir.y, length(motoDir.zx));
    motoRoll = 0.0;
}

vec3 motoToWorld(vec3 v, bool isPos)
{
    v.xy *= Rotation(-motoPitch);
    v.yz *= Rotation(-motoRoll);
    v.xz *= Rotation(-motoYaw);
    if (isPos)
    {
        v += motoPos;
    }
    return v;
}

vec3 worldToMoto(vec3 v, bool isPos)
{
    if (isPos)
    {
        v -= motoPos;
    }
    v.xz *= Rotation(motoYaw);
    v.yz *= Rotation(motoRoll);
    v.xy *= Rotation(motoPitch);
    return v;
}






vec3 meter3(vec2 uv, float value) {
    
    
    

    float verticalLength = 0.04 + 0.15 * smoothstep(0.1, 0.4, uv.x);

    float r = Box2(uv, vec2(0.5, verticalLength), 0.01);
    

    float lines = smoothstep(0.5, 0.7, fract(uv.x * 30.));
    lines *= smoothstep(0.1, 0.3, fract(uv.y/verticalLength*2.));

    vec3 baseCol =
        mix(vec3(0.7, 0.9, 0.8),
            vec3(0.8, 0., 0.), smoothstep(0.4, 0.41, uv.x));

    baseCol = mix(vec3(0.01), baseCol, 0.15+0.85*smoothstep(0., 0.001, value*0.5 - uv.x));
    vec3 col = lines * baseCol;
    return smoothstep(0.001, 0., r) * float(uv.y > 0.) * col;
}

vec3 meter4(vec2 uv, float value) {
  float len = length(uv);
  float angle = atan(uv.y, uv.x);

  float lines =
    smoothstep(0.7, 1., mod(angle, 0.25)/0.25) *
    smoothstep(0., 0.01, abs(angle + 0.7) - 0.7) * 
    smoothstep(0., 0.01, 0.1 - length(uv)) *
    smoothstep(0., 0.01, length(uv) - 0.06);

  value = (value * 1.5 - 1.) * PI;
  vec2 point = vec2(sin(value), cos(value)) * 0.07;
  float dummy;
  float line = smoothstep(0.004, 0.002, Segment3(uv.xyy, vec3(0), point.xyy, dummy));
  vec3 col = vec3(0.36, 0.16, 0.12) * lines;
  col += vec3(0.7) * line;
  return col;
}

float digit(int n, vec2 p)
{
    vec2 size = vec2(0.2, 0.35);
    const float thickness = 0.065;
    const float gap = 0.0125;
    const float slant = 0.15;
    const float roundOuterCorners = 0.5;
    const float roundInterCorners = 0.15;
    const float spacing = 0.67;

    bool A = (n != 1 && n != 4);
    bool i_B = (n != 5 && n != 6);
    bool i_C = (n != 2);
    bool i_D = (A && n != 7);
    bool i_E = (A && n % 2 == 0);
    bool i_F = (n != 1 && n != 2 && n != 3 && n != 7);
    bool i_G = (n > 1 && n != 7);

    p.x -= p.y * slant;
    float boundingBox = Box2(p, size, size.x * roundOuterCorners);
    float innerBox = -Box2(p, size - thickness, size.x * roundInterCorners);
    float d = INF;

    
    if (A)
    {
        float sA = innerBox;
        sA = max(sA, gap + (p.x - p.y - size.x + size.y));
        sA = max(sA, gap - (p.x + p.y + size.x - size.y));
        d = min(d, sA);
    }

    
    if (i_B)
    {
        float sB = innerBox;
        sB = max(sB, gap - (p.x - p.y - size.x + size.y));
        sB = max(sB, gap - (p.x + p.y + size.x - size.y));
        sB = max(sB, p.x - (p.y) - (size.x + thickness) / 2.);
        d = min(d, sB);
    }

    
    if (i_C)
    {
        float sC = innerBox;
        sC = max(sC, gap - (p.x - p.y + size.x - size.y));
        sC = max(sC, gap - (p.x + p.y - size.x + size.y));
        sC = max(sC, p.x + (p.y) - (size.x + thickness) / 2.);
        d = min(d, sC);
    }

    
    if (i_D)
    {
        float sD = innerBox;
        sD = max(sD, gap - (p.x - p.y + size.x - size.y));
        sD = max(sD, gap + (p.x + p.y - size.x + size.y));
        d = min(d, sD);
    }

    
    if (i_E)
    {
        float sE = innerBox;
        sE = max(sE, gap + (p.x - p.y + size.x - size.y));
        sE = max(sE, gap + (p.x + p.y - size.x + size.y));
        sE = max(sE, -p.x + (p.y) - (size.x + thickness) / 2.);
        d = min(d, sE);
    }

    
    if (i_F)
    {
        float sF = innerBox;
        sF = max(sF, gap + (p.x - p.y - size.x + size.y));
        sF = max(sF, gap + (p.x + p.y + size.x - size.y));
        sF = max(sF, -p.x - (p.y) - (size.x + thickness) / 2.);
        d = min(d, sF);
    }

    
    if (i_G)
    {
        float sG = -(thickness - abs(p.y) * 2.);
        sG = max(sG, gap + (p.x - p.y + size.x - size.y));
        sG = max(sG, gap - (p.x - p.y - size.x + size.y));
        sG = max(sG, gap + (p.x + p.y + size.x - size.y));
        sG = max(sG, gap - (p.x + p.y - size.x + size.y));
        d = min(d, sG);
    }

    return max(d, boundingBox);
}

vec3 glowy(float d)
{
    float dd = fwidth(d);
    float brightness = smoothstep(-dd, +dd, d);
    vec3 segment = vec3(0.3, 0.5, 0.4);

    vec3 innerColor = mix(vec3(0.2), segment, 1. / exp(50. * max(0., -d)));
    vec3 outerColor = mix(vec3(0.), segment, 1. / exp(200. * max(0., d)));
    return mix(innerColor, outerColor, brightness);
}

vec3 motoDashboard(vec2 uv)
{
    int speed = 105 + int(sin(time*.5) * 10.);
    vec2 uvSpeed = uv * 3. - vec2(0.4, 1.95);

    float numbers =
        min(min(min(
        
        digit(5, uv * 8. - vec2(0.7,2.4)),
        
        (float(speed<100) + digit(speed/100, uvSpeed))),
        digit((speed/10)%10, uvSpeed - vec2(.5,0))),
        digit(speed%10, uvSpeed - vec2(1.,0)));

    return
        meter3(uv * 0.6 - vec2(0.09, 0.05), 0.7+0.3*sin(time*0.5)) +
        meter4(uv * .7 - vec2(0.6, 0.45), 0.4) +
        glowy(numbers);
}





material motoMaterial(float mid, vec3 p, vec3 N)
{
    if (mid == MOTO_HEAD_LIGHT_ID)
    {
        float isLight = smoothstep(0.9, 0.95, N.x);
        vec3 luminance = isLight * vec3(1., 0.95, 0.9);

        float isDashboard = smoothstep(0.9, 0.95, -N.x + 0.4 * N.y - 0.07);
        if (isDashboard > 0.)
        {
            vec3 color = motoDashboard(p.zy * 5.5 + vec2(0.5, -5.));
            luminance = mix(vec3(0), color, isDashboard);
        }

        return material(MATERIAL_TYPE_EMISSIVE, luminance, 0.15);
    }
    if (mid == MOTO_BREAK_LIGHT_ID)
    {
        float isLight = smoothstep(0.9, 0.95, -N.x);
        vec2 lightUV = fract(68.*p.yz + vec2(0.6, 0.)) * 2. - 1.;
        float pattern = smoothstep(0.2, 1., sqrt(length(lightUV)));
        vec3 luminance = mix(vec3(1., 0.005, 0.02), vec3(0.02, 0., 0.), pattern);
        return material(MATERIAL_TYPE_EMISSIVE, isLight * luminance, 0.5);
    }
    if (mid == MOTO_EXHAUST_ID)
    {
        return material(MATERIAL_TYPE_METALLIC, vec3(1.), 0.2);
    }
    if (mid == MOTO_MOTOR_ID)
    {
        return material(MATERIAL_TYPE_DIELECTRIC, vec3(0.), 0.3);
    }
    if (mid == MOTO_WHEEL_ID)
    {
        return material(MATERIAL_TYPE_DIELECTRIC, vec3(0.008), 0.8);
    }

    if (mid == MOTO_DRIVER_ID)
    {
        return material(MATERIAL_TYPE_DIELECTRIC, vec3(0.02, 0.025, 0.04), 0.6);
    }
    if (mid == MOTO_DRIVER_HELMET_ID)
    {
        return material(MATERIAL_TYPE_DIELECTRIC, vec3(0.), 0.25);
    }

    return material(MATERIAL_TYPE_DIELECTRIC, vec3(0.), 0.15);
}

vec2 driverShape(vec3 p)
{
    p = worldToMoto(p, true);

    
    p -= vec3(-0.35, 0.78, 0.0);

    float d = length(p);
    if (d > 1.2)
        return vec2(d, MOTO_DRIVER_ID);

    vec3 simP = p;
    simP.z = abs(simP.z);

    float wind = fBm((p.xy + time) * 12., 1, 0.5, 0.5);

    
    if (true && d < 0.8)
    {
        vec3 pBody = simP;
        pBody.z -= 0.02;
        pBody.xy *= Rotation(3.1);
        pBody.yz *= Rotation(-0.1);
        d = smin(d, Capsule(pBody, 0.12, 0.12), 0.1);

        pBody.y += 0.2;
        pBody.xy *= Rotation(-0.6);
        d = smin(d, Capsule(pBody, 0.12, 0.11), 0.02);

        pBody.y += 0.2;
        pBody.xy *= Rotation(-0.3);
        pBody.yz *= Rotation(-0.2);
        d = smin(d, Capsule(pBody, 0.12, 0.12), 0.02);

        pBody.y += 0.1;
        pBody.yz *= Rotation(1.7);
        d = smin(d, Capsule(pBody, 0.12, 0.1), 0.015);
    }
    d += 0.005 * wind;
    
    
    if (true)
    {
        vec3 pArm = simP;

        pArm -= vec3(0.23, 0.45, 0.18);
        pArm.yz *= Rotation(-0.6);
        pArm.xy *= Rotation(0.2);
        float arms = Capsule(pArm, 0.29, 0.06);
        d = smin(d, arms, 0.005);

        pArm.y += 0.32;
        pArm.xy *= Rotation(1.5);
        arms = Capsule(pArm, 0.28, 0.04);
        d = smin(d, arms, 0.005);
    }
    d += 0.01 * wind;

    
    if (true)
    {
        vec3 pLeg = simP;

        pLeg -= vec3(0.0, 0.0, 0.13);
        pLeg.xy *= Rotation(1.55);
        pLeg.yz *= Rotation(-0.45);
        float h2 = Capsule(pLeg, 0.35, 0.09);
        d = smin(d, h2, 0.01);

        pLeg.y += 0.4;
        pLeg.xy *= Rotation(-1.5);
        float legs = Capsule(pLeg, 0.4, 0.06);
        d = smin(d, legs, 0.01);

        pLeg.y += 0.45;
        pLeg.xy *= Rotation(1.75);
        pLeg.yz *= Rotation(0.25);
        float feet = Capsule(pLeg, 0.2, 0.04);
        d = smin(d, feet, 0.01);
    }
    d += 0.002 * wind;

    
    if (true)
    {
        vec3 pHead = p - vec3(0.39, 0.6, 0.0);
        float head = length(pHead) - 0.15;

        if (head < d)
        {
            return vec2(head, MOTO_DRIVER_HELMET_ID);
        }
    }

    return vec2(d, MOTO_DRIVER_ID);
}

vec2 wheelShape(vec3 p, float wheelRadius, float tireRadius, float innerRadius)
{
    vec2 d = vec2(1e6, MOTO_WHEEL_ID);
    float wheel = Torus(p.yzx, vec2(wheelRadius, tireRadius));

    if (wheel < 0.25)
    {
        p.z = abs(p.z);
        float h;
        float cyl = Segment3(p, vec3(0.0), vec3(0.0, 0.0, 1.0), h);
        wheel = -smin(-wheel, cyl - innerRadius, 0.01);

         
        
        wheel = min(wheel, -min(min(min(0.15 - cyl, cyl - 0.08), p.z - 0.04), -p.z + 0.05));
         
    }
    return vec2(wheel, MOTO_WHEEL_ID);
}

vec2 motoShape(vec3 p)
{
    p = worldToMoto(p, true);

    float boundingSphere = length(p);
    if (boundingSphere > 2.0)
        return vec2(boundingSphere - 1.5, MOTO_ID);

    vec2 d = vec2(1e6, MOTO_ID);

// Inactive conditional block: #ifdef DEBUG


    float h;
    float cyl;

    float frontWheelTireRadius = 0.14/2.0;
    float frontWheelRadius = 0.33 - frontWheelTireRadius;
    float rearWheelTireRadius = 0.3/2.0;
    float rearWheelRadius = 0.32 - rearWheelTireRadius;
    vec3 frontWheelPos = vec3(0.9, frontWheelRadius + frontWheelTireRadius, 0.0);

    
    if (true)
    {
        d = MinDist(d, wheelShape(p - frontWheelPos, frontWheelRadius, frontWheelTireRadius, 0.22));
    }

    
    if (true)
    {
        d = MinDist(d, wheelShape(p - vec3(-0.85, rearWheelRadius + rearWheelTireRadius, 0.0), rearWheelRadius, rearWheelTireRadius, 0.18));
    
        
        if (true)
        {
            vec3 pBreak = p - breakLightOffsetFromMotoRoot;
            float breakBlock = Box3(pBreak, vec3(0.02, 0.025, 0.1), 0.02);
            d = MinDist(d, vec2(breakBlock, MOTO_BREAK_LIGHT_ID));
        }
    }

    
    if (true)
    {
        float forkThickness = 0.025;
        vec3 pFork = p;
        vec3 pForkTop = vec3(-0.48, 0.66, 0.0);
        vec3 pForkAngle = pForkTop + vec3(-0.14, 0.04, 0.05);
        pFork.z = abs(pFork.z);
        pFork -= frontWheelPos + vec3(0.0, 0.0, frontWheelTireRadius + 2. * forkThickness);
        float fork = Segment3(pFork, pForkTop, vec3(0.0), h) - forkThickness;

        
        fork = min(fork, Segment3(pFork, pForkTop, pForkAngle, h) - forkThickness * 0.7);

        
        float handle = Segment3(pFork, pForkAngle, pForkAngle + vec3(-0.08, -0.07, 0.3), h);
        fork = min(fork, handle - mix(0.035, 0.02, smoothstep(0.25, 0.4, h)));

        
        vec3 pMirror = pFork - pForkAngle - vec3(0.0, 0.1, 0.15);
        pMirror.xz *= Rotation(0.2);
        pMirror.xy *= Rotation(-0.2);
        
        float mirror = pMirror.x - 0.02;
        pMirror.xz *= Rotation(0.25);

        mirror = -min(mirror, -Ellipsoid(pMirror, vec3(0.04, 0.05, 0.08)));
        fork = min(fork, mirror);

        d = MinDist(d, vec2(fork, MOTO_ID));
    }

    
    if (true)
    {
        vec3 pHead = p - headLightOffsetFromMotoRoot;
        float headBlock = Ellipsoid(pHead, vec3(0.15, 0.2, 0.15));
        
        if (headBlock < 0.2)
        {
            vec3 pHeadTopBottom = pHead;

            
            pHeadTopBottom.xy *= Rotation(-0.15);
            headBlock = -min(-headBlock, -Ellipsoid(pHeadTopBottom - vec3(-0.2, -0.05, 0.0), vec3(0.35, 0.16, 0.25)));

            
            headBlock = -min(-headBlock, -Ellipsoid(pHead - vec3(-0.2, -0.08, 0.0), vec3(0.35, 0.25, 0.13)));

            
            headBlock = -min(-headBlock, -Ellipsoid(pHead - vec3(-0.1, -0.05, 0.0), vec3(0.2, 0.2, 0.3)));

            
            pHead.xy *= Rotation(-0.4);
            headBlock = -min(-headBlock, -Ellipsoid(pHead - vec3(0.1, 0.0, 0.0), vec3(0.2, 0.3, 0.4)));
        }

        d = MinDist(d, vec2(headBlock, MOTO_HEAD_LIGHT_ID));

        float joint = Box3(p - vec3(0.4, 0.82, 0.0), vec3(0.04, 0.1, 0.08), 0.02);
        d = MinDist(d, vec2(joint, MOTO_MOTOR_ID));
    }

    
    if (true)
    {
        vec3 pTank = p - vec3(0.1, 0.74, 0.0);
        vec3 pTankR = pTank;
        pTankR.xy *= Rotation(0.45);
        pTankR.x += 0.05;
        float tank = Ellipsoid(pTankR, vec3(0.35, 0.2, 0.42));

        if (tank < 0.1)
        {
            
            float tankCut = Ellipsoid(pTankR + vec3(0.0, 0.13, 0.0), vec3(0.5, 0.35, 0.22));
            tank = -min(-tank, -tankCut);
            

            
            float tankCut2 = Ellipsoid(pTank - vec3(0.0, 0.3, 0.0), vec3(0.6, 0.35, 0.4));
            tank = -min(-tank, -tankCut2);
            
        }
        d = MinDist(d, vec2(tank, MOTO_ID));
    }

    
    if (true)
    {
        vec3 pMotor = p - vec3(-0.08, 0.44, 0.0);
        
        
        vec3 pMotorSkewd = pMotor;
        pMotorSkewd.x *= 1. - pMotorSkewd.y * 0.4;
        pMotorSkewd.x += pMotorSkewd.y * 0.1;
        float motorBlock = Box3(pMotorSkewd, vec3(0.44, 0.29, 0.11), 0.02);
        
        if (motorBlock < 0.5)
        {
            
            vec3 pMotor1 = pMotor - vec3(0.27, 0.12, 0.0);
            vec3 pMotor2 = pMotor - vec3(0.00, 0.12, 0.0);
            pMotor1.xy *= Rotation(-0.35);
            pMotor2.xy *= Rotation(0.35);
            motorBlock = min(motorBlock, Box3(pMotor1, vec3(0.1, 0.12, 0.20), 0.04));
            motorBlock = min(motorBlock, Box3(pMotor2, vec3(0.1, 0.12, 0.20), 0.04));

            
            vec3 pGearBox = pMotor - vec3(-0.15, -0.12, -0.125);
            pGearBox.xy *= Rotation(-0.15);
            float gearBox = Segment3(pGearBox, vec3(0.2, 0.0, 0.0), vec3(-0.15, 0.0, 0.0), h);
            gearBox -= mix(0.08, 0.15, h);
            
            pGearBox.x += 0.13;
            float gearBoxCut = -pGearBox.z - 0.05;
            gearBoxCut = min(gearBoxCut, Box3(pGearBox, vec3(0.16, 0.08, 0.1), 0.04));
            gearBox = -min(-gearBox, -gearBoxCut);

            motorBlock = min(motorBlock, gearBox);

            
            vec3 pPedals = pMotor - vec3(0.24, -0.13, 0.0);
            float pedals = Segment3(pPedals, vec3(0.0, 0.0, .4), vec3(0.0, 0.0, -.4), h) - 0.02;
            motorBlock = min(motorBlock, pedals);
        }
        d = MinDist(d, vec2(motorBlock, MOTO_MOTOR_ID));
    }

    
    if (true)
    {
        vec3 pExhaust = p;
        pExhaust -= vec3(0.0, 0.0, rearWheelTireRadius + 0.05);
        float exhaust = Segment3(pExhaust, vec3(0.24, 0.25, 0.0), vec3(-0.7, 0.3, 0.05), h);

        if (exhaust < 0.6)
        {
            exhaust -= mix(0.04, 0.08, mix(h, smoothstep(0.5, 0.7, h), 0.5));
            exhaust = -min(-exhaust, p.x - 0.7 * p.y + 0.9);
            exhaust = min(exhaust, Segment3(pExhaust, vec3(0.24, 0.25, 0.0), vec3(0.32, 0.55, -0.02), h) - 0.04);
            exhaust = min(exhaust, Segment3(pExhaust, vec3(0.22, 0.32, -0.02), vec3(-0.4, 0.37, 0.02), h) - 0.04);
        }
        d = MinDist(d, vec2(exhaust, MOTO_ID));
    }

    
    if (true)
    {
        vec3 pSeat = p - vec3(-0.44, 0.44, 0.0);
        float seat = Ellipsoid(pSeat, vec3(0.8, 0.4, 0.2));
        float seatRearCut = length(p + vec3(1.05, -0.1, 0.0)) - 0.7;
        seat = max(seat, -seatRearCut);

        if (seat < 0.2)
        {
            vec3 pSaddle = pSeat - vec3(0.35, 0.57, 0.0);
            pSaddle.xy *= Rotation(0.4);
            float seatSaddleCut = Ellipsoid(pSaddle, vec3(0.5, 0.15, 0.6));
            seat = -min(-seat, seatSaddleCut);
            seat = -smin(-seat, seatSaddleCut, 0.02);

            vec3 pSeatBottom = pSeat + vec3(0.0, -0.55, 0.0);
            pSeatBottom.xy *= Rotation(0.5);
            float seatBottomCut = Ellipsoid(pSeatBottom, vec3(0.8, 0.4, 0.4));
            seat = -min(-seat, -seatBottomCut);
        }
        d = MinDist(d, vec2(seat, MOTO_ID));
    }

    return d;
}
// --------------------------------------------------------------------
// Include end: motoContent.frag

// Include begin: rendering.frag
// --------------------------------------------------------------------
light lights[MAX_LIGHTS];




material computeMaterial(float mid, vec3 p, vec3 N)
{
// Inactive conditional block: #ifdef DEBUG


    if (mid == GROUND_ID)
    {
        vec4 splineUV = ToSplineLocalSpace(p.xz, roadWidthInMeters.z);
        float isRoad = 1.0 - smoothstep(roadWidthInMeters.x, roadWidthInMeters.y, abs(splineUV.x));
        if (isRoad > 0.0)
        {
            return roadMaterial(splineUV.xz, 3.5, vec2(0.7, 0.0));
        }
        
        vec3 color = pow(vec3(67., 81., 70.) / 255. * 1.5, vec3(GAMMA));
        return material(MATERIAL_TYPE_DIELECTRIC, color, 0.5);
    }

    if (IsMoto(mid))
    {
        p = worldToMoto(p, true);
        N = worldToMoto(N, false);
        
        return motoMaterial(mid, p, N);
    }

    if (mid == ROAD_REFLECTOR_ID)
    {
        return material(MATERIAL_TYPE_RETROREFLECTIVE, vec3(1., 0.4, 0.05), 0.2);
    }

    return material(MATERIAL_TYPE_DIELECTRIC, fract(p.xyz), 1.0);
}

vec2 sceneSDF(vec3 p, float current_t)
{
    vec2 d = vec2(INF, NO_ID);

    vec4 splineUV = ToSplineLocalSpace(p.xz, roadWidthInMeters.z);

// Active conditional block: #ifndef DISABLE_MOTO
    d = MinDist(d, motoShape(p));
// End of active block.
// Active conditional block: #ifndef DISABLE_MOTO_DRIVER
    if (camShowDriver > 0.5)
    {
        d = MinDist(d, driverShape(p));
    }
// End of active block.
// Active conditional block: #ifndef DISABLE_TERRAIN
    d = MinDist(d, terrainShape(p, splineUV));
// End of active block.
// Active conditional block: #ifndef DISABLE_TREES
    d = MinDist(d, treesShape(p, splineUV, current_t));
// End of active block.

    return d;
}

void setLights()
{
// Inactive conditional block: #ifdef ENABLE_DAY_MODE
// Active conditional block: #else
    lights[0] = light(moonDirection * 1e3, -moonDirection, moonLightColor, 0., 0., 1e10, 0.005);
// End of active block.

    vec3 posHeadLight = motoToWorld(headLightOffsetFromMotoRoot + vec3(0.1, 0., 0.), true);
    vec3 posBreakLight = motoToWorld(breakLightOffsetFromMotoRoot, true);
    dirHeadLight = motoToWorld(dirHeadLight, false);
    dirBreakLight = motoToWorld(dirBreakLight, false);

    vec3 intensityHeadLight = vec3(1.);
    lights[1] = light(posHeadLight, dirHeadLight, intensityHeadLight, 0.75, 0.95, 10.0, 20.);

    vec3 intensityBreakLight = vec3(1., 0., 0.);
    lights[2] = light(posBreakLight, dirBreakLight, intensityBreakLight, 0.3, 0.9, 2.0, 0.05);
}




vec3 evalNormal(vec3 p, float t)
{
    
    const float h = NORMAL_DP;
    vec2 k = vec2(1., -1.);
    return normalize(
        k.xyy * sceneSDF(p + k.xyy * h, t).x + 
        k.yyx * sceneSDF(p + k.yyx * h, t).x + 
        k.yxy * sceneSDF(p + k.yxy * h, t).x + 
        k.xxx * sceneSDF(p + k.xxx * h, t).x
    );
}

vec2 rayMarchScene(vec3 ro, vec3 rd, float tMax, int max_steps, out vec3 p
// Inactive conditional block: #ifdef ENABLE_STEP_COUNT

)
{
    p = ro;
    float t = 0.;
    vec2 d;

    for (int i = ZERO(iTime); i < max_steps; ++i)
    {
// Inactive conditional block: #ifdef ENABLE_STEP_COUNT


        d = sceneSDF(p, t);
        t += d.x;
        p = ro + t * rd;

        
        
        float epsilon = t * PIXEL_ANGLE;
        if (d.x < epsilon)
        {
            return vec2(t, d.y);
        }
        if (t >= tMax)
        {
            return vec2(t, NO_ID);
        }
    }
    return vec2(t, d.y);
}

float castShadowRay(vec3 p, vec3 N, vec3 rd)
{
// Inactive conditional block: #ifdef ENABLE_STEP_COUNT
// Active conditional block: #else
    vec2 t = rayMarchScene(p + BOUNCE_OFFSET * N, rd, MAX_SHADOW_DIST, MAX_SHADOW_STEPS, p);
// End of active block.

    return smoothstep(MAX_SHADOW_DIST/2., MAX_SHADOW_DIST, t.x);
}

vec3 evalRadiance(vec2 t, vec3 p, vec3 V, vec3 N)
{
    float mid = t.y;
    if (mid == CITY_ID) {
        return mix(
            abs(N.y)>.8?cityLights(p.xz*2.):vec3(0.),
            mix(vec3(0), vec3(.06,.04,.03),V.y),
            min(t.x*.001,1.));
    }

    if (mid == NO_ID)
    {
        
        return sky(-V);
    }

    material m = computeMaterial(mid, p, N);

    vec3 emissive = vec3(0.);
    if (m.T == MATERIAL_TYPE_EMISSIVE)
    {
        
        
        float aligned = clamp(dot(V, N), 0., 1.);
        float aligned4 = aligned * aligned * aligned * aligned;
        float aligned8 = aligned4 * aligned4 * aligned4 * aligned4 * aligned4 * aligned4 * aligned4 * aligned4;
        emissive = m.C * mix(aligned*0.1 + aligned4, 1., aligned8);
    }

    vec3 albedo = vec3(0.);
    if (m.T == MATERIAL_TYPE_DIELECTRIC)
    {
        albedo = m.C;
    }

    vec3 f0 = vec3(0.04);
    if (m.T == MATERIAL_TYPE_METALLIC)
    {
        f0 = m.C;
    }

    if (m.T == MATERIAL_TYPE_RETROREFLECTIVE)
    {
        f0 = m.C;
        N = V;
    }

    vec3 radiance = emissive;

    
// Inactive conditional block: #ifdef ENABLE_DAY_MODE
// Active conditional block: #else
    
    vec3 I0 = nightHorizonLight * mix(1.0, 0.1, N.y * N.y) * (N.x * 0.5 + 0.5);
// End of active block.
    radiance += I0 * albedo;

    
    if (m.R < 0.25) 
    {
        vec3 L = reflect(-V, N);
        
	    
	    
	    
        radiance += f0 * sky(L);
    }

    
    for (int i = 0; i < MAX_LIGHTS; ++i)
    {
        radiance += lightContribution(lights[i], p, V, N, albedo, f0, m.R);
    }

    float fogAmount = 1.0 - exp(-t.x*0.01);
    vec3 fogColor = vec3(0.001,0.001,0.005);
    radiance = mix(radiance, fogColor, fogAmount);

    return radiance;
}
// --------------------------------------------------------------------
// Include end: rendering.frag

// Include begin: moto.frag
// --------------------------------------------------------------------
// --------------------------------------------------------------------
// Include end: moto.frag


void main()
{
    ComputeBezierSegmentsLengthAndAABB();

    vec2 texCoord = gl_FragCoord.xy/iResolution.xy;
    vec2 uv = (texCoord * 2. - 1.) * vec2(1., iResolution.y / iResolution.x);

    time = iTime;
    if (ENABLE_STOCHASTIC_MOTION_BLUR) {
        time += hash31(vec3(gl_FragCoord.xy, 1e-3*iTime)) * 0.008;
    }

    
    computeMotoPosition();

    setLights();

    
    vec3 ro;
    vec3 rd;
    vec3 cameraPosition = camPos;
    vec3 cameraTarget = camTa;
    vec3 cameraUp = vec3(0., 1., 0.);
    if (camMotoSpace > 0.5) {
        cameraPosition = motoToWorld(camPos, true);
        cameraTarget = motoToWorld(camTa, true);
        
    }
    setupCamera(uv, cameraPosition, cameraTarget, cameraUp, ro, rd);

    
    

    
    

    
    


    vec3 p;
// Inactive conditional block: #ifdef ENABLE_STEP_COUNT
// Active conditional block: #else
    vec2 t = rayMarchScene(ro, rd, MAX_RAY_MARCH_DIST, MAX_RAY_MARCH_STEPS, p);
// End of active block.
    vec3 N = evalNormal(p, t.x);

    vec3 radiance = evalRadiance(t, p, -rd, N);
    
    vec3 color = pow(radiance, vec3(1. / GAMMA));
    color = mix(color, texture(tex, texCoord).rgb, 0.2);
    fragColor = vec4(color, 1.);
}
