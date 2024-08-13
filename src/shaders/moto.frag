#version 150
out vec4 fragColor;
const vec2 iResolution = vec2(1920.,1080.);
vec2 iMouse = vec2(700., 900.);
uniform float iTime;
#define iFrame 0

in vec3 camPos;
in vec3 camTa;
in float camFocal;
in float fishEyeFactor;

#define MAX_SDF_STEPS 100
#define MAX_HM_STEPS 50
#define MAX_SHADOW_STEPS 32
#define MAX_DIST 60.0
#define MAX_SHADOW_DIST 10.0
#define MAX_LIGHTS 6
#define EPSILON 1e-4
#define BOUNCE_OFFSET 1e-3
#define PI acos(-1.)
#define GAMMA 2.2
#define ZERO min(iFrame, 0)

#define SPLINE_ABOUT_FOURTY_POINTS 0

#define ENABLE_STOCHASTIC_MOTION_BLUR 1

#define GROUND_ID 0
#define BIDULE_ID 1

vec2 segmentPoints[] = vec2[](vec2(-5.0), vec2(5.0));

vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d)
{
    return a + b * cos(2. * PI * (c * t + d));
}

vec3 debugPalette(float t)
{
    return cos(2. * PI * (t + vec3(0.1, 0.2, 0.3))) * 0.5 + 0.5;
}

// -------------------------------------------------------
// Noise functions

float hash(float x) { return fract(sin(x) * 43758.5453); }
float hash(vec2 xy) { return fract(sin(dot(xy, vec2(12.9898, 78.233))) * 43758.5453); }
float hash(vec3 xyz) { return hash(vec2(hash(xyz.xy), xyz.z)); }
vec2 hash2(vec2 xy) { return fract(sin(vec2(dot(xy, vec2(127.1,311.7)), dot(xy, vec2(269.5,183.3)))) * 43758.5453); }

float valueNoise(vec2 p)
{
    vec2 p00 = floor(p);

    vec2 fp = p - p00;
    fp = fp*fp * (3.0 - 2.0 * fp);

    vec2 p10 = p00 + vec2(1.0, 0.0);
    vec2 p01 = p00 + vec2(0.0, 1.0);
    vec2 p11 = p00 + vec2(1.0, 1.0);

    float v00 = hash(p00);
    float v10 = hash(p10);
    float v01 = hash(p01);
    float v11 = hash(p11);

    float v = mix(mix(v00, v10, fp.x), mix(v01, v11, fp.x), fp.y);
    return v;// - 0.5; <---- FIXME
}

float fBm(vec2 p, int iterations, float a, float b)
{
    float v = 0.;
    float weight = 1.0;
    float frequency = 1.0;
    float offset = 0.0;

    for (int i = ZERO; i < iterations; ++i)
    {
        v += weight * valueNoise(frequency * p + offset);
        weight *= clamp(a, 0., 1.);
        frequency *= 1.0 + 2.0 * clamp(b, 0., 1.);
        offset += 1.0;
    }
    return v;
}

// -------------------------------------------------------
// Camera
vec3 sphericalToCartesian(float r, float phi, float theta)
{
    float cosTheta = cos(theta);
    float x = cosTheta * cos(phi);
    float y = sin(theta);
    float z = cosTheta * sin(phi);
    return r * vec3(x, y, z);
}

vec3 worldToCubeMap(vec3 v)
{
    return vec3(v.x, v.y, -v.z);
}

void orbitalCamera(vec2 uv, vec2 mouseInput, out vec3 ro, out vec3 rd)
{
    mouseInput.y = clamp(mouseInput.y, 0.0, 1.0);
    vec3 cameraPosition = sphericalToCartesian(10.0, 2.0 * PI * mouseInput.x, PI * (0.5 - mouseInput.y));
    vec3 cameraTarget = vec3(0.);
    vec3 cameraForward = normalize(cameraTarget - cameraPosition);
    vec3 cameraUp = vec3(0., 1., 0.);
    if (abs(dot(cameraForward, cameraUp)) > 0.99)
    {
        cameraUp = vec3(1., 0., 0.);
    }
    vec3 cameraRight = normalize(cross(cameraForward, cameraUp));
    cameraUp = normalize(cross(cameraRight, cameraForward));

    ro = cameraPosition;
    rd = normalize(cameraForward + uv.x * cameraRight + uv.y * cameraUp);
}

// -------------------------------------------------------
// SDF functions

float smin(float a, float b, float k)
{
    k *= 1.0 / (1.0 - sqrt(0.5));
    return max(k, min(a, b)) - length(max(k - vec2(a, b), 0.0));
}

float Box(vec2 p, vec2 size, float corner)
{
   p = abs(p) - size + corner;
   return length(max(p, 0.)) + min(max(p.x, p.y), 0.) - corner;
}

float Box(vec3 p, vec3 size, float corner)
{
   p = abs(p) - size + corner;
   return length(max(p, 0.)) + min(max(max(p.x, p.y), p.z), 0.) - corner;
}

float Segment(vec2 p, vec2 a, vec2 b, out float h)
{
	vec2 ap = p - a;
	vec2 ab = b - a;
	h = clamp(dot(ap, ab) / dot(ab, ab), 0., 1.);
	return length(ap - ab * h);
}

// -------------------------------------------------------
// Spline function


#if SPLINE_ABOUT_FOURTY_POINTS
#define SPLINE_SIZE 37
#else
#define SPLINE_SIZE 13
#endif

float splineSegmentDistances[SPLINE_SIZE / 2];

vec2 spline[SPLINE_SIZE] = vec2[](
    vec2(-5.0, -5.0),
    vec2(-3.0, -2.0),

    vec2(-5.0,  0.0),
    vec2(-7.0,  2.0),

    vec2(-5.0,  5.0),
    vec2(-3.0,  7.0),

    vec2( 0.0,  5.0),
    vec2( 1.0,  4.0),

    vec2( 2.0,  5.0),
    vec2( 4.0,  6.0),

    vec2( 4.0,  3.0),
    vec2( 4.0, -2.0),

    vec2( 8.0, 3.0)
);

// Piece of code copy-pasted from:
// https://www.shadertoy.com/view/NltBRB
// Credit: MMatteini

// Roots of the cubic equation for the closest point to a bezier.
// From: https://www.shadertoy.com/view/MdXBzB by tomkh
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

// Returns 1.0 if the two vector are clockwise sorted, -1.0 otherwise
float GetWinding(vec2 a, vec2 b)
{
    return 2.0 * step(a.x * b.y, a.y * b.x) - 1.0;
}

// Returns the signed distance from a point to a bezier curve
// Mostly from: https://www.shadertoy.com/view/MdXBzB by tomkh
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

    // Find closest distance and t
    vec4 r = (d1 < d2) ? vec4(d1, t.x, dp1) : vec4(d2, t.y, dp2);

    // Check the distance sign
    float s = GetWinding(r.zw, 2.0 * b * r.y + c);
    
    return vec2(s * sqrt(r.x), r.y);
}

// Calc the length of a quadratic bezier at the start/end points and at the specified value for the parameter t.
// X = length of the bezier up to "t" 
// Y = total length of the curve
float BezierCurveLengthAt(vec2 A, vec2 B, vec2 C, float t)
{
    // Bezier curve function:
    // f(t) = t^2(A - 2B + C) + t(2B - 2A) + A

    // Calc the bezier curve derivative (velocity function):
    // f'(t) = t(2A-4B+2C) + 2B - 2A = a1t + b1
    vec2 a1 = 2.0 * (A - 2.0 * B + C);
    vec2 b1 = 2.0 * (B - A);

    // Calc the velocity function magnitude:
    // ||f'(t)|| = sqrt(t^2 * k1 + t * k2 + k3)
    float k1 = dot(a1, a1);
    float k2 = 2.0 * dot(a1, b1);
    float k3 = dot(b1, b1);

    // Reparametrize for easier integration
    // t^2k1 + tk2 + k3 = k1((t + k4)^2 + k5)
    float k4 = 0.5f * k2 / k1;
    float k5 = k3 / k1 - k4 * k4;

    // Calculate the definite integrals of the velocity function to obtain the distance function
    // solution to this integral form is from:
    // https://en.wikipedia.org/wiki/List_of_integrals_of_irrational_functions
    // S ||f'(t)|| dt = 0.5 * sqrt(k1) * [(k4 + t) * sqrt((t + k4)^2 + k5) + k5 ln(k4 + t + sqrt((t + k4)^2 + k5))]  
    vec2 ti = vec2(0.0, t); // calc at both integration bounds at once
    vec2 vt = sqrt((ti + k4) * (ti + k4) + k5);
    vec2 sdft = sqrt(k1) * 0.5 * ((k4 + ti) * vt + k5 * log(abs(k4 + ti + vt)));
    return sdft.y - sdft.x;
}

void ComputeBezierSegmentsLength()
{
    float splineLength = 0.0;
    for (int i = 0; i < SPLINE_SIZE - 1; i += 2)
    {
        vec2 A = spline[i + 0];
        vec2 B = spline[i + 1];
        vec2 C = spline[i + 2];
        float segmentLength = BezierCurveLengthAt(A, B, C, 1.0);
        splineSegmentDistances[i / 2] = splineLength;
        splineLength += segmentLength;
    }
}

// Quadradic B�zier curve exact bounding box from IQ:
// https://www.shadertoy.com/view/XdVBWd
vec4 BezierAABB(vec2 A, vec2 B, vec2 C)
{
    // extremes
    vec2 mi = min(A, C);
    vec2 ma = max(A, C);

    // maxima/minima point, if p1 is outside the current bbox/hull
    if (B.x < mi.x || B.x > ma.x ||
        B.y < mi.y || B.y > ma.y)
    {
        // p = (1-t)^2*p0 + 2(1-t)t*p1 + t^2*p2
        // dp/dt = 2(t-1)*p0 + 2(1-2t)*p1 + 2t*p2 = t*(2*p0-4*p1+2*p2) + 2*(p1-p0)
        // dp/dt = 0 -> t*(p0-2*p1+p2) = (p0-p1);

        vec2 t = clamp((A - B) / (A - 2.0*B+C),0.0,1.0);
        vec2 s = 1.0 - t;
        vec2 q = s*s*A + 2.0*s*t*B + t*t*C;
        
        mi = min(mi, q);
        ma = max(ma, q);
    }
    
    return vec4(mi, ma);
}

float DistanceFromBezierAABB(vec2 p, vec2 A, vec2 B, vec2 C)
{
    vec4 aabb = BezierAABB(A, B, C);

    vec2 center = (aabb.xy + aabb.zw) / 2.0;
    vec2 size = aabb.zw - aabb.xy;

    return Box(p - center, size / 2.0, 0.0);
}

// Decompose a give location into its distance from the closest point on the spline. 
// and the length of the spline up to that point.
// Returns a vector where:
// X = signed distance from the spline
// Y = spline parameter t at the closest point [0; 1]
// Z = spline length at the closest point
vec3 ToSplineLocalSpace(vec2 p, float splineWidth)
{
    vec3 splineUV = vec3(3e10, 0, 0);

    // For each bezier segment
    for (int i = 0; i < SPLINE_SIZE - 1; i += 2)
    {
        vec2 A = spline[i + 0];
        vec2 B = spline[i + 1];
        vec2 C = spline[i + 2];
        
        if (DistanceFromBezierAABB(p, A, B, C) - splineWidth <= 0.0)
        {
            // This is to prevent 3 colinear points, but there should be better solution to it.
            B = mix(B + vec2(1e-4), B, abs(sign(B * 2.0 - A - C))); 
            // Current bezier curve SDF
            vec2 bezierSDF = BezierSDF(A, B, C, p);

            if (abs(bezierSDF.x) < abs(splineUV.x))
            {
                float lengthInSegment = BezierCurveLengthAt(A, B, C, clamp(bezierSDF.y, 0., 1.));
                float lengthInSpline = splineSegmentDistances[i / 2] + lengthInSegment;
                splineUV = vec3(bezierSDF.x, (clamp(bezierSDF.y, 0., 1.) + 0.5 * float(i)) / float(SPLINE_SIZE), lengthInSpline);
            }
        }
    }

    return splineUV;
}

// -------------------------------------------------------
// Scene description functions

vec3 daySkyDomeLight = 0.5 * vec3(0.25, 0.5, 1.0);
vec3 sunLight = vec3(1.0, 0.85, 0.7);

vec3 nightHorizonLight = 0.01 * vec3(0.07, 0.1, 1.0);
vec3 moonLight = 0.02 * vec3(0.2, 0.8, 1.0);

float maxRoadWidth = 1.0;

struct material
{
    vec3 emissive;
    vec3 albedo;
    float roughness;
};

vec3 roadPattern(vec2 uv, float width, vec2 params)
{
    // Total interval, line length
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

    float sideLine1 = Box(sideTileUV - vec2(0.5 * sideLineParams.y, width), vec2(0.5 * sideLineParams.y, 0.10), 0.03);
    float sideLine2 = Box(sideTileUV - vec2(0.5 * sideLineParams.y, -width), vec2(0.5 * sideLineParams.y, 0.10), 0.03);

    float separationLine1 = Box(separationTileUV - vec2(0.5 * separationLineParams.y, 0.0), vec2(0.5 * separationLineParams.y, 0.10), 0.01);

    float pattern = min(min(sideLine1, sideLine2), separationLine1);

    float duv = length(fwidth(uv));
    vec3 test = /**/ vec3(1.0 - smoothstep(-duv, 0.0, pattern)); /*/ debugDistance(iResolution, 10.*pattern) /**/;
    return mix(test, vec3(fract(uv), separationTileUV.x), 0.0);
}

material computeMaterial(int mid, vec3 p, vec3 N)
{
    if (mid == GROUND_ID)
    {
        vec2 cellPos = floor(1.*p.xz + 0.5);
        float cellID = hash(cellPos);
        
        /*
        vec3 color = palette(cellID,
            vec3(0.15,0.35,0.18),
            vec3(0.15, 0.15, 0.1),
            vec3(3.0,5.0,7.0),
            vec3(0.0));
        */
        vec3 color = pow(vec3(67., 81., 70.) / 255. * 1.5, vec3(GAMMA));
        //color = debugPalette(cellID);

        vec3 splineUV = ToSplineLocalSpace(p.xz, maxRoadWidth);
        float isRoad = 1.0 - smoothstep(0.5, 0.6, abs(splineUV.x));
        vec3 roadColor = vec3(0.0);
        if (isRoad > 0.0)
        {
            roadColor = roadPattern(splineUV.zx * 8., 3.5, vec2(0.7, 0.0));
        }
        color = mix(color, roadColor, isRoad);
        return material(vec3(0.0), color, 0.5);
    }
    else if (mid == BIDULE_ID)
    {
        vec3 color = vec3(1.0);
        return material(vec3(0.0), color, 0.25);
    }
    
    return material(vec3(0.0), fract(p.xyz), 1.0);
}

float sceneSDF(vec3 p, float dMax, out int mid)
{
    float d = dMax;

    /*
    float ground = p.y + 0.0;
    if (ground < d)
    {
        d = ground;
        mid = GROUND_ID;
    }
    //*/

    p -= vec3(0.0 , 1.0, 0.0);
    float sphere = length(p) - 1.;
    float box = Box(p, vec3(1.25, 0.4, 0.9), 0.1);
    float bidule = smin(sphere, box, 0.01);
    if (bidule < d)
    {
        d = bidule;
        mid = BIDULE_ID;
    }

    if (d > EPSILON)
    {
        mid = -1;
    }

    return d;
}

float sceneHeightMap(vec2 p, int maxIter, bool computeRoad)
{
    vec2 params = iMouse.xy / iResolution.xy;

    float isRoad = 0.0;
    if (true)//computeRoad) // FIXME: broken shadows on road sides
    {
        vec3 splineUV = ToSplineLocalSpace(p, maxRoadWidth);
        isRoad = 1.0 - smoothstep(0.5, 1.0, abs(splineUV.x));
    }

    float terrain = 0.0;
    if (isRoad < 1.0)
    {
        terrain = 5.*fBm(p * 0.1, maxIter, params.x, params.y) - 5.;
    }

    float road = 0.0;
    if (isRoad >= 0.0)
    {
        road = 5.*fBm(p * 0.1, 2, params.x, params.y) - 5. + 0.2;
    }
    
    return mix(terrain, road, isRoad);
}

// -------------------------------------------------------
// Rendering functions

// TODO: Ideally h would depend on the screen space projected size.
vec3 evalNormal(vec3 p, float useSDF, float h)
{
    if (useSDF > 0.)
    {
        const vec2 k = vec2(1., -1.);
        int dummy;
        return normalize(
            k.xyy * sceneSDF(p + k.xyy * h, 1.0, dummy) + 
            k.yyx * sceneSDF(p + k.yyx * h, 1.0, dummy) + 
            k.yxy * sceneSDF(p + k.yxy * h, 1.0, dummy) + 
            k.xxx * sceneSDF(p + k.xxx * h, 1.0, dummy)
        );
    }
    else
    {
        vec3 p10 = p + vec3(h, 0., 0.);
        p10.y = sceneHeightMap(p10.xz, 6, true);

        vec3 p01 = p + vec3(0., 0., h);
        p01.y = sceneHeightMap(p01.xz, 6, true);

        return normalize(cross(p01 - p, p10 - p));
    }
}

float rayMarchSceneSDF(vec3 ro, vec3 rd, float tMax, int max_steps, out vec3 p, out int mid)
{
    p = ro;
    float t = 0.;
    mid = -1;

    for (int i = ZERO; i < max_steps; ++i)
    {
        float d = sceneSDF(p, tMax - t, mid);
        t += d;
        p = ro + t * rd;

        if (d < EPSILON || t >= tMax)
        {
            break;
        }
    }
    return t;
}

float rayMarchSceneHeightMap(vec3 ro, vec3 rd, float tMax, int max_steps, out vec3 p, out int mid)
{
    float tPrev;
    float dPrev;
    float t = 0.;
    float d = tMax;
    p = ro;
    mid = -1;

    for (int i = ZERO; i < max_steps; ++i)
    {
        dPrev = d;
        tPrev = t;

        d = p.y - sceneHeightMap(p.xz, d < 2. ? 6 : 4, d < 4.);

        if (d < 0.)
        {
            float w = d / dPrev;
            t = (t - w * tPrev) / (1. - w);
            p = ro + t * rd;
            
            // TODO: it should be possible to use only the first harmonics
            // for large distance, and compute the finer details only when
            // getting close.
            break;
        }

        // Assume the height is a good approximation of the distance that can be marched.
        t = tPrev + d * 0.8;
        p = ro + t * rd;
    }
    p.y = sceneHeightMap(p.xz, 6, true);
    mid = GROUND_ID;
        
    return t;
}

vec2 rayMarchScene(vec3 ro, vec3 rd, float tMax, int max_sdf_steps, int max_hm_steps, out vec3 p, out int mid)
{
    vec3 pSDF;
    int midSDF;
    float tSDF = rayMarchSceneSDF(ro, rd, tMax, max_sdf_steps, pSDF, midSDF);

    vec3 pHM;
    int midHM;
    float tHM = rayMarchSceneHeightMap(ro, rd, tMax, max_hm_steps, pHM, midHM);

    float t;
    if (tSDF < tHM)
    {
        t = tSDF;
        p = pSDF;
        mid = midSDF;
    }
    else
    {
        t = tHM;
        p = pHM;
        mid = midHM;
    }
    return vec2(t, float(tSDF < tHM));
}

float castShadowRay(vec3 p, vec3 N, vec3 rd)
{
    int mid = -1;
    vec2 t = rayMarchScene(p + BOUNCE_OFFSET * N, rd, MAX_SHADOW_DIST, MAX_SHADOW_STEPS, MAX_SHADOW_STEPS, p, mid);

    return smoothstep(EPSILON, 1.0, t.x);
}

vec3 evalRadiance(int mid, vec3 p, vec3 V, vec3 N)
{
    if (mid < 0)
    {
        // Background
        return vec3(0.);//texture(iChannel0, worldToCubeMap(-V)).rgb;
    }

    material m = computeMaterial(mid, p, N);


    // Global illumination coming from the sky dome:

    // Direct light:
    vec3 L1 = normalize(vec3(0.4, 0.6*1.0, 0.8));
    float NdotL1 = clamp(dot(N, L1), 0.0, 1.0);
    float visibility_L1 = castShadowRay(p, N, L1);

#if 1
    // Day version:
    vec3 I0 = daySkyDomeLight * (N.y * 0.5 + 0.5);
    vec3 I1 = sunLight * NdotL1 * visibility_L1;
#else
    // Night version:
    vec3 I0 = nightHorizonLight * mix(1.0, 0.1, N.y * N.y) * (N.x * 0.5 + 0.5);
    vec3 I1 = moonLight * NdotL1 * visibility_L1;
#endif

    // Env map:
    vec3 L2 = reflect(-V, N);
    vec3 H = normalize(L2 + V);
	float x = 1.0 - dot(V, H);
	x = x*x*x*x*x;
	float F = x + 0.04 * (1.0 - x);

    //vec3 I2 = texture(iChannel0, worldToCubeMap(L2)).rgb;

    vec3 radiance = vec3(0.);
    radiance += m.emissive;
    radiance += (I0 + I1) * m.albedo;
    //radiance += F * I2;
    return radiance;
}

// -------------------------------------------------------

mat3 lookat(vec3 ro, vec3 ta)
{
    const vec3 up = vec3(0.,1.,0.);
    vec3 fw = normalize(ta-ro);
    vec3 rt = normalize(cross(fw, normalize(up)));
    return mat3(rt, cross(rt, fw), fw);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    ComputeBezierSegmentsLength();

    vec2 uv = (fragCoord/iResolution.xy * 2. - 1.) * vec2(1., iResolution.y / iResolution.x);
    
    vec2 mouseInput = iMouse.xy / iResolution.xy;

    float time = iTime;
#if ENABLE_STOCHASTIC_MOTION_BLUR
    time += hash(vec3(fragCoord, 1e-3*iTime)) * 0.008;
#endif
    // orbitalCamera(uv, vec2(0.02 * time, 0.3) /*mouseInput*/, ro, rd);
    vec2 v = uv*2.-1.;
    vec3 ro = camPos;
    vec3 rd = lookat(ro, camTa) * normalize(vec3(v, camFocal - length(v) * fishEyeFactor));

    vec3 p;
    int mid;
    vec2 t = rayMarchScene(ro, rd, MAX_DIST, MAX_SDF_STEPS, MAX_HM_STEPS, p, mid);
    vec3 N = evalNormal(p, t.y, 1e-2);

    vec3 radiance = evalRadiance(mid, p, -rd, N);
    
    vec3 color = pow(radiance, vec3(1. / GAMMA));
    //color = fract(0.1*p);
    //color = vec3(debugPalette(t.x));
    //color = N * 0.5 + 0.5;
    //color = vec3((N * 0.5 + 0.5).z);
    fragColor = vec4(color, 1.);
}


void main() {
    mainImage(fragColor, gl_FragCoord.xy);
    // fragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
