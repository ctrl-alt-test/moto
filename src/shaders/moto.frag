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
#define GAMMA 2.2

#define ENABLE_STOCHASTIC_MOTION_BLUR 1

#define NO_ID       -1.
#define GROUND_ID   0.
#define BIDULE_ID   1.

vec2 segmentPoints[] = vec2[](vec2(-5.0), vec2(5.0));

// -------------------------------------------------------
// Scene description functions

vec3 daySkyDomeLight = 0.5 * vec3(0.25, 0.5, 1.0);
vec3 sunLight = vec3(1.0, 0.85, 0.7);

vec3 nightHorizonLight = 0.01 * vec3(0.07, 0.1, 1.0);
vec3 moonLight = 0.02 * vec3(0.2, 0.8, 1.0);

struct material
{
    vec3 emissive;
    vec3 albedo;
    float roughness;
};

material computeMaterial(float mid, vec3 p, vec3 N)
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

vec2 sceneSDF(vec3 p)
{
    vec2 d = vec2(1e6, NO_ID);

    p -= vec3(0.0 , 1.0, 0.0);
    float sphere = length(p) - 1.;
    float box = Box(p, vec3(1.25, 0.4, 0.9), 0.1);
    float bidule = smin(sphere, box, 0.01);
    if (bidule < d.x)
    {
        d = vec2(bidule, BIDULE_ID);
    }

    if (d.x > EPSILON)
    {
        d.y = NO_ID;
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
        return normalize(
            k.xyy * sceneSDF(p + k.xyy * h).x + 
            k.yyx * sceneSDF(p + k.yyx * h).x + 
            k.yxy * sceneSDF(p + k.yxy * h).x + 
            k.xxx * sceneSDF(p + k.xxx * h).x
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

vec2 rayMarchSceneSDF(vec3 ro, vec3 rd, float tMax, int max_steps, out vec3 p
#ifdef ENABLE_STEP_COUNT
, out int steps
#endif
)
{
    p = ro;
    float t = 0.;
    float mid = NO_ID;

    for (int i = ZERO; i < max_steps; ++i)
    {
#ifdef ENABLE_STEP_COUNT
        steps = i + 1;
#endif

        vec2 d = sceneSDF(p);
        t += d.x;
        mid = d.y;
        p = ro + t * rd;

        if (d.x < EPSILON || t >= tMax)
        {
            break;
        }
    }
    if (t >= tMax)
    {
        mid = NO_ID;
    }
    return vec2(t, mid);
}

vec2 rayMarchSceneHeightMap(vec3 ro, vec3 rd, float tMax, int max_steps, out vec3 p)
{
    float mid;
    float tPrev;
    float dPrev;
    float t = 0.;
    float d = tMax;
    p = ro;
    mid = NO_ID;

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
        
    return vec2(t, mid);
}

vec3 rayMarchScene(vec3 ro, vec3 rd, float tMax, int max_sdf_steps, int max_hm_steps, out vec3 p)
{
    vec3 pSDF;
    vec2 tSDF = rayMarchSceneSDF(ro, rd, tMax, max_sdf_steps, pSDF);

    vec3 pHM;
    vec2 tHM = rayMarchSceneHeightMap(ro, rd, tMax, max_hm_steps, pHM);

    vec2 t;
    if (tSDF.x < tHM.x)
    {
        t = tSDF;
        p = pSDF;
    }
    else
    {
        t = tHM;
        p = pHM;
    }
    return vec3(t, float(tSDF.x < tHM.x));
}

float castShadowRay(vec3 p, vec3 N, vec3 rd)
{
    vec3 t = rayMarchScene(p + BOUNCE_OFFSET * N, rd, MAX_SHADOW_DIST, MAX_SHADOW_STEPS, MAX_SHADOW_STEPS, p);

    return smoothstep(EPSILON, 1.0, t.x);
}

vec3 evalRadiance(float mid, vec3 p, vec3 V, vec3 N)
{
    if (mid == NO_ID)
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
    vec3 t = rayMarchScene(ro, rd, MAX_DIST, MAX_SDF_STEPS, MAX_HM_STEPS, p);
    vec3 N = evalNormal(p, t.z, 1e-2);

    vec3 radiance = evalRadiance(t.y, p, -rd, N);
    
    vec3 color = pow(radiance, vec3(1. / GAMMA));
    fragColor = vec4(color, 1.);
}


void main() {
    mainImage(fragColor, gl_FragCoord.xy);
}
