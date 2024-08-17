#ifndef ZERO
#define ZERO 0
#endif

const bool ENABLE_SMOOTHER_STEP_NOISE = false;
const float PI = acos(-1.);

// -------------------------------------------------------
// Palette function
// Code by IQ
// See: https://iquilezles.org/articles/palettes/

vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d)
{
    return a + b * cos(2. * PI * (c * t + d));
}

// -------------------------------------------------------
// Scene description functions

struct light
{
    vec3 p0;
    vec3 p1;
    vec3 color;
    float cosAngle;
    float collimation;
    float luminance;
};

struct material
{
    vec3 emissive;
    vec3 albedo;
    float roughness;
};

// -------------------------------------------------------
// Shading functions

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

	float SmithL = (2. * NdotL) / max(1e-8, NdotL + sqrt(NdotL * NdotL * (1. - sqrAlpha) + sqrAlpha));
	float SmithV = (2. * NdotV) / max(1e-8, NdotV + sqrt(NdotV * NdotV * (1. - sqrAlpha) + sqrAlpha));
	float G = SmithL * SmithV;

	float x = 1. - VdotH;
	x = x*x*x*x*x;
	vec3 F = x + f0 * (1. - x);

	return F * (D * G * 0.25 / max(1e-8, NdotV * NdotL));
}

vec3 coneLightContribution(material m, light l, vec3 p, vec3 N, vec3 V)
{
    vec3 L0 = l.p0 - p;
    float d0 = length(L0);
    vec3 L = L0 / d0;

    float NdotL = dot(N, L);
    if (NdotL <= 0.)
        return vec3(0.);

    vec3 intensity = l.color * l.luminance;
    intensity *= smoothstep(l.cosAngle, mix(l.cosAngle, 1.0, 0.5), dot(L, -l.p1));
    intensity *= (1.0 + l.collimation) / (l.collimation + d0 * d0);
    vec3 radiance = intensity * NdotL;

    vec3 H = normalize(L + V);
    vec3 NcrossH = cross(N, H);
    float VdotH = clamp(dot(V, H), 0., 1.);
    float NdotV = clamp(dot(N, V), 0., 1.);

    vec3 diff = m.albedo;
    vec3 spec = cookTorrance(vec3(0.04), m.roughness, NcrossH, VdotH, NdotL, NdotV);

    return radiance * (diff + spec);
}

vec3 rodLightContribution(material m, light l, vec3 p, vec3 N, vec3 V)
{
    vec3 L0 = l.p0 - p;
    vec3 L1 = l.p1 - p;
    float d0 = length(L0);
    float d1 = length(L1);

    vec3 luminance = l.color * l.luminance;

    // Approximating what the impact of the light will be.
    // This approximation assumes a diffuse material. In case
    // of artifacts with shiny materials, maybe the area can
    // be elongated based on the roughness.
    //float roughContribution = dot(luminance, vec3(1.0)) * (0.5 * dot(N, L0) / dot(L0, L0));
    //if (roughContribution * 1000.0 < 1.0) return vec3(0.);
    float falloff = 1.0;//smoothstep(0.001, 0.002, roughContribution);

    // DEBUG:
    //if (p.x < 0.) return vec3(roughContribution);
    
    float NdotL0 = dot(N, L0) / d0;
    float NdotL1 = dot(N, L1) / d1;

    float topPart = 2. * clamp((NdotL0 + NdotL1) / 2., 0., 1.);
    float bottomPart = d0 * d1 + dot(L0, L1);
    float contribution = topPart / bottomPart;
    // The Karis paper has a +2 term in the bottom part,
    // but I found the result to better match point lights
    // when the length is zero.
    // Then again, there could be an implementation error.

    if (contribution <= 0.)
    {
        return vec3(0.);
    }

    vec3 irradiance = luminance * contribution * falloff;

    vec3 Ld = l.p1 - l.p0;
    vec3 R = reflect(-V, N);
    float RdotLd = dot(R, Ld);
    
    float t = clamp((dot(R, L0) * RdotLd - dot(L0, Ld)) / (dot(Ld, Ld) - RdotLd * RdotLd), 0., 1.);
    vec3 Lmrp = mix(L0, L1, t);

    vec3 L = normalize(Lmrp);
    float NdotL = clamp(dot(N, L), 0., 1.);

    vec3 H = normalize(L + V);
    vec3 NcrossH = cross(N, H);
    float VdotH = clamp(dot(V, H), 0., 1.);
    float NdotV = clamp(dot(N, V), 0., 1.);

    vec3 diff = m.albedo;
    vec3 spec = cookTorrance(vec3(0.04), m.roughness, NcrossH, VdotH, NdotL, NdotV);

    return irradiance * (diff + spec);
}

// -------------------------------------------------------
// Noise functions

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

    for (int i = ZERO; i < iterations; ++i)
    {
        float noise = valueNoise(p * frequency + offset) * 2. - 1.;
        v += weight * noise;
        weight *= clamp(weight_param, 0., 1.);
        frequency *= 1.0 + 2.0 * clamp(frequency_param, 0., 1.);
        offset += 1.0;
    }
    return v;
}

// -------------------------------------------------------
// SDF functions

float smin(float a, float b, float k)
{
    k *= 1.0 / (1.0 - sqrt(0.5));
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

float Segment2(vec2 p, vec2 a, vec2 b, out float h)
{
	vec2 ap = p - a;
	vec2 ab = b - a;
	h = clamp(dot(ap, ab) / dot(ab, ab), 0., 1.);
	return length(ap - ab * h);
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

// -------------------------------------------------------
// Bézier functions

// Piece of code copy-pasted from:
// https://www.shadertoy.com/view/NltBRB
// Credit: MMatteini

vec2 Bezier(vec2 A, vec2 B, vec2 C, float t)
{
    vec2 AB = mix(A, B, t);
    vec2 BC = mix(B, C, t);
    return mix(AB, BC, t);
}

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
    float k4 = 0.5 * k2 / k1;
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

// Quadradic Bézier curve exact bounding box from IQ:
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

    return Box2(p - center, size / 2.0, 0.0);
}

// -------------------------------------------------------
// Raymarching functions

// x: distance
// y: ID
vec2 MinDist(vec2 d1, vec2 d2)
{
    return d1.x < d2.x ? d1 : d2;
}

// -------------------------------------------------------
// Camera functions

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

void orbitalCamera(vec2 uv, float dist, float lat, float lon, out vec3 ro, out vec3 rd)
{
    lat = clamp(lat, -PI, PI);
    vec3 cameraPosition = sphericalToCartesian(dist, lon, lat);
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

void setupCamera(vec2 uv, vec3 cameraPosition, vec3 cameraTarget, vec3 cameraUp, float projectionRatio, float camFishEye, out vec3 ro, out vec3 rd)
{
    vec3 cameraForward = normalize(cameraTarget - cameraPosition);
    if (abs(dot(cameraForward, cameraUp)) > 0.99)
    {
        cameraUp = vec3(1., 0., 0.);
    }
    vec3 cameraRight = normalize(cross(cameraForward, cameraUp));
    cameraUp = normalize(cross(cameraRight, cameraForward));

    // meh. FIXME
    uv *= mix(1., length(uv), camFishEye);
    ro = cameraPosition;
    rd = normalize(cameraForward * projectionRatio + uv.x * cameraRight + uv.y * cameraUp);
}

// -------------------------------------------------------
