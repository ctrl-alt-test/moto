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

//
// Debug palette to show the number of steps
//
#ifdef ENABLE_STEP_COUNT
vec3 stepsToColor(int steps)
{
    vec3 colorCodedCount = vec3(0.);

    vec3 colorCodes[] = vec3[](
        vec3(0.),
        vec3(0., 0., 1.),
        vec3(0., 1., 1.),
        vec3(0., 1., 0.),
        vec3(1., 1., 0.),
        vec3(1., 0., 0.),
        vec3(1., 0.4, 1.)
    );
    if (steps <= 10)
    {
        colorCodedCount = mix(colorCodes[0], colorCodes[1], clamp(float(steps) / 10, 0., 1.));
    }
    else if (steps <= 50)
    {
        colorCodedCount = mix(colorCodes[1], colorCodes[2], clamp(float(steps - 10) / 40, 0., 1.));
    }
    else if (steps <= 100)
    {
        colorCodedCount = mix(colorCodes[2], colorCodes[3], clamp(float(steps - 50) / 50, 0., 1.));
    }
    else if (steps <= 150)
    {
        colorCodedCount = mix(colorCodes[2], colorCodes[3], clamp(float(steps - 100) / 50, 0., 1.));
    }
    else if (steps <= 200)
    {
        colorCodedCount = mix(colorCodes[3], colorCodes[4], clamp(float(steps - 150) / 50, 0., 1.));
    }
    else if (steps <= 250)
    {
        colorCodedCount = mix(colorCodes[4], colorCodes[5], clamp(float(steps - 200) / 50, 0., 1.));
    }
    else
    {
        colorCodedCount = mix(colorCodes[5], colorCodes[6], clamp(float(steps - 250) / 50, 0., 1.));
    }

    return colorCodedCount;
}
#endif

// -------------------------------------------------------
// Scene description functions

struct L // light
{
    vec3 P; // Start position for rod light, position for cone light.
    vec3 Q; // End position for rod light, direction for cone light.
    vec3 C; // Color
    float A; // Start cos angle; -1 for rod light.
    float B; // End cos angle.
    float F; // Collimation factor (point light vs focused torch light).
};

const int MATERIAL_TYPE_DIELECTRIC = 0;
const int MATERIAL_TYPE_METALLIC = 1;
const int MATERIAL_TYPE_EMISSIVE = 2;
const int MATERIAL_TYPE_RETROREFLECTIVE = 3;

struct M // material
{
    int T; // Type
    vec3 C; // Albedo for dielectric, metallic and retroreflective, luminance for emissive.
    float R; // Roughness
};

// -------------------------------------------------------
// Shading functions

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

    // Normal distribution term D:
	float distribution = dot(NcrossH, NcrossH) * (1. - sqrAlpha) + sqrAlpha;
	float D = sqrAlpha / (PI * distribution * distribution);

    // Visibility term V:
    float V = 1.0 / (invV1(NdotV, sqrAlpha) * invV1(NdotL, sqrAlpha));

    // Fresnel term F:
	float x = 1. - VdotH;
	vec3 F = x + f0 * (1. - x*x*x*x*x);

	return F * D * V;
}

vec3 lightContribution(L l, vec3 p, vec3 V, vec3 N, vec3 albedo, vec3 f0, float roughness)
{
    vec3 L, irradiance;
    vec3 L0 = l.P - p;
    float d0 = length(L0);
    float NdotL;

    if (l.A != -1.0)
    {
        //
        // Cone light contribution
        //
        L = L0 / d0;

        NdotL = dot(N, L);

        float LdotD = dot(L, -l.Q);
        vec3 freq = 180. + vec3(0., .4, .8);
        float angleFallOff = smoothstep(l.A, l.B, LdotD);
        angleFallOff *= angleFallOff;
        angleFallOff *= angleFallOff;
        angleFallOff *= angleFallOff;

        vec3 radiant_intensity = l.C;
        radiant_intensity *= (sin(freq/LdotD) * 0.5 + 0.5) * 0.2 + 0.8;
        radiant_intensity *= angleFallOff;
        radiant_intensity *= (1.0 + l.F) / (l.F + d0 * d0);
        irradiance = radiant_intensity * NdotL;
    }
    else
    {
        //
        // Rod light contribution
        //
        vec3 L1 = l.Q - p;
        float d1 = length(L1);

        // Approximating what the impact of the light will be.
        // This approximation assumes a diffuse material. In case
        // of artifacts with shiny materials, maybe the area can
        // be elongated based on the roughness.
        //float roughContribution = dot(intensity, vec3(1.0)) * (0.5 * dot(N, L0) / dot(L0, L0));
        //if (roughContribution * 1000.0 < 1.0) return vec3(0.);
        float falloff = 1.0;//smoothstep(0.001, 0.002, roughContribution);

        // DEBUG:
        //if (p.x < 0.) return vec3(roughContribution);
    
        float NdotL0 = dot(N, L0) / d0;
        float NdotL1 = dot(N, L1) / d1;

        float angularContribution = max(0., NdotL0 + NdotL1);
        float geometricAttenuation = d0 * d1 + dot(L0, L1);
        float contribution = angularContribution / geometricAttenuation;
        // The Karis paper has a +2 term in the bottom part,
        // but I found the result to better match point lights
        // when the length is zero.
        // Then again, there could be an implementation error.

        contribution *= falloff;
        if (contribution <= 0.)
        {
            return vec3(0.);
        }

        vec3 radiant_intensity = l.C;
        irradiance = radiant_intensity * contribution;

        vec3 Ld = l.Q - l.P;
        vec3 R = reflect(-V, N);
        float RdotLd = dot(R, Ld);
    
        float t = clamp((dot(R, L0) * RdotLd - dot(L0, Ld)) / (dot(Ld, Ld) - RdotLd * RdotLd), 0., 1.);

        // Most representative light direction
        L = normalize(mix(L0, L1, t));
        NdotL = dot(N, L);
    }

    if (NdotL <= 0.)
        return vec3(0.);

    vec3 H = normalize(L + V);
    vec3 NcrossH = cross(N, H);
    float VdotH = max(0., dot(V, H));
    float NdotV = max(0., dot(N, V));

    vec3 spec = cookTorrance(f0, roughness, NcrossH, VdotH, NdotL, NdotV);

    return irradiance * (albedo + spec);
}

// -------------------------------------------------------
// Noise functions

// TODO: try to reduce the number of hash functions?
float hash11(float x) { return fract(sin(x) * 43758.5453); }
float hash21(vec2 xy) { return fract(sin(dot(xy, vec2(12.9898, 78.233))) * 43758.5453); }
float hash31(vec3 xyz) { return hash21(vec2(hash21(xyz.xy), xyz.z)); }
vec2 hash22(vec2 xy) { return fract(sin(vec2(dot(xy, vec2(127.1,311.7)), dot(xy, vec2(269.5,183.3)))) * 43758.5453); }
vec2 hash12(float x) { float h = hash11(x); return vec2(h, hash11(h)); }

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

// TODO: merge with previous function?
vec2 valueNoise2(float p)
{
    float p0 = floor(p);
    float p1 = p0 + 1.;

    vec2 v0 = hash12(p0);
    vec2 v1 = hash12(p1);

    float fp = p - p0;
    fp = fp*fp * (3.0 - 2.0 * fp);

    return mix(v0, v1, fp);
}

float fBm(vec2 p, int iterations, float weight_param, float frequency_param)
{
    float v = 0.;
    float weight = 1.0;
    float frequency = 1.0;
    float offset = 0.0;

    for (int i = 0; i < iterations; ++i)
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

// -------------------------------------------------------
// Bézier functions

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

// Returns the signed distance from a point to a Bezier curve
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
    // extremes (min: xy, max: zw)
    vec4 res = vec4(min(A, C), max(A, C));

    // maxima/minima point, if p1 is outside the current bbox/hull
    if (B.x < res.x || B.x > res.z ||
        B.y < res.y || B.y > res.w)
    {
        // p = (1-t)^2*p0 + 2(1-t)t*p1 + t^2*p2
        // dp/dt = 2(t-1)*p0 + 2(1-2t)*p1 + 2t*p2 = t*(2*p0-4*p1+2*p2) + 2*(p1-p0)
        // dp/dt = 0 -> t*(p0-2*p1+p2) = (p0-p1);

        vec2 t = clamp((A - B) / (A - 2.0*B+C),0.0,1.0);
        vec2 s = 1.0 - t;
        vec2 q = s*s*A + 2.0*s*t*B + t*t*C;
        
        res.xy = min(res.xy, q);
        res.zw = max(res.zw, q);
    }
    
    return res;
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

/*
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
*/

void setupCamera(vec2 uv, vec3 cameraPosition, vec3 cameraTarget, vec3 cameraUp, out vec3 ro, out vec3 rd)
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
    rd = normalize(cameraForward * camProjectionRatio + uv.x * cameraRight + uv.y * cameraUp);
}

// -------------------------------------------------------
