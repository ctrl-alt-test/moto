light lights[MAX_LIGHTS];

// -------------------------------------------------------
// Scene description functions

material computeMaterial(float mid, vec3 p, vec3 N)
{
    if (mid == GROUND_ID)
    {
        vec4 splineUV = ToSplineLocalSpace(p.xz, roadWidthInMeters.z);
        float isRoad = 1.0 - smoothstep(roadWidthInMeters.x, roadWidthInMeters.y, abs(splineUV.x));
        if (isRoad > 0.0)
        {
            return roadMaterial(splineUV.xz, 3.5, vec2(0.7, 0.0));
        }
        // Terrain
        vec3 color = pow(vec3(67., 81., 70.) / 255. * 1.5, vec3(GAMMA));
        return material(MATERIAL_TYPE_DIELECTRIC, color, 0.5);
    }

    if (IsMoto(mid))
    {
        p = worldToMoto(p, true, iTime);
        N = worldToMoto(N, false, iTime);
        //return material(MATERIAL_TYPE_EMISSIVE, N * 0.5 + 0.5, 0.15);
        return motoMaterial(mid, p, N, iTime);
    }

    if (mid == ROAD_REFLECTOR_ID)
    {
        return material(MATERIAL_TYPE_RETROREFLECTIVE, vec3(1., 0.4, 0.), 0.2);
    }

    return material(MATERIAL_TYPE_DIELECTRIC, fract(p.xyz), 1.0);
}

vec2 sceneSDF(vec3 p, float current_t)
{
    vec2 d = vec2(INF, NO_ID);

    vec4 splineUV = ToSplineLocalSpace(p.xz, roadWidthInMeters.z);

#ifndef DISABLE_MOTO
    d = MinDist(d, motoShape(p));
#endif
#ifndef DISABLE_MOTO_DRIVER
    if (camShowDriver > 0.5)
    {
        d = MinDist(d, driverShape(p));
    }
#endif
#ifndef DISABLE_TERRAIN
    d = MinDist(d, terrainShape(p, splineUV));
#endif
#ifndef DISABLE_TREES
    d = MinDist(d, treesShape(p, splineUV, current_t));
#endif

    return d;
}

void setLights()
{
#ifdef ENABLE_DAY_MODE
    lights[0] = light(moonDirection * 1e3, moonDirection, sunLightColor, 0., 1e3, 10.);
#else
    lights[0] = light(moonDirection * 1e3, moonDirection, moonLightColor, -2., 1e10, 0.02);
#endif

    vec3 posHeadLight = motoToWorld(headLightOffsetFromMotoRoot, true, iTime);
    vec3 posBreakLight = motoToWorld(breakLightOffsetFromMotoRoot, true, iTime);
    dirHeadLight = motoToWorld(dirHeadLight, false, iTime);
    dirBreakLight = motoToWorld(dirBreakLight, false, iTime);

    vec3 intensityHeadLight = vec3(1.);
    lights[1] = light(posHeadLight, dirHeadLight, intensityHeadLight, 0.93, 10.0, 20.);

    vec3 intensityBreakLight = vec3(1., 0., 0.);
    lights[2] = light(posBreakLight, dirBreakLight, intensityBreakLight, 0.7, 2.0, 0.1);
}

// -------------------------------------------------------
// Rendering functions

vec3 evalNormal(vec3 p, float t)
{
    // TODO: Ideally h would depend on the screen space projected size.
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
#ifdef ENABLE_STEP_COUNT
, out int steps
#endif
)
{
    p = ro;
    float t = 0.;
    vec2 d;

    for (int i = ZERO(iTime); i < max_steps; ++i)
    {
#ifdef ENABLE_STEP_COUNT
        steps = i + 1;
#endif

        d = sceneSDF(p, t);
        t += d.x;
        p = ro + t * rd;

        // The minimum distance is t.sin(pixel_angle), but the angle is
        // so small we can safely approximate with sin(x) = x.
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
#ifdef ENABLE_STEP_COUNT
    int steps;
    vec2 t = rayMarchScene(p + BOUNCE_OFFSET * N, rd, MAX_SHADOW_DIST, MAX_SHADOW_STEPS, p, steps);
#else
    vec2 t = rayMarchScene(p + BOUNCE_OFFSET * N, rd, MAX_SHADOW_DIST, MAX_SHADOW_STEPS, p);
#endif

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
        // Background / sky
        return sky(-V);
    }

    material m = computeMaterial(mid, p, N);

    vec3 emissive = vec3(0.);
    if (m.type == MATERIAL_TYPE_EMISSIVE)
    {
        // Stronger luminance in the direction of the normal:
        // https://graphtoy.com/?f1(x,t)=mix(0.1*x+pow(x,4),1,pow(x,8))&coords=0.4,0.4,1
        float aligned = clamp(dot(V, N), 0., 1.);
        float aligned4 = aligned * aligned;
        aligned4 *= aligned4;
        float aligned8 = aligned4 * aligned4;
        aligned8 *= aligned8;
        aligned8 *= aligned8;
        emissive = m.color * mix(aligned*0.1 + aligned4, 1., aligned8);
    }

    vec3 albedo = vec3(0.);
    if (m.type == MATERIAL_TYPE_DIELECTRIC)
    {
        albedo = m.color;
    }

    vec3 f0 = vec3(0.04);
    if (m.type == MATERIAL_TYPE_METALLIC)
    {
        f0 = m.color;
    }

    if (m.type == MATERIAL_TYPE_RETROREFLECTIVE)
    {
        f0 = m.color;
        N = V;
    }

    vec3 radiance = emissive;

    // Crude global illumination coming from the sky dome:
#ifdef ENABLE_DAY_MODE
    // Day version:
    vec3 I0 = daySkyDomeLight * (N.y * 0.5 + 0.5);
#else
    // Night version:
    vec3 I0 = nightHorizonLight * mix(1.0, 0.1, N.y * N.y) * (N.x * 0.5 + 0.5);
#endif
    radiance += I0 * albedo;

    // Env map:
    if (m.roughness < 0.25) // Brutal if until there's a roughness dependent reflection.
    {
        vec3 L = reflect(-V, N);
        // vec3 H = normalize(L + V);
	    // float x = 1.0 - dot(V, H);
	    // x = x*x*x*x*x;
	    // vec3 F = x + f0 * (1.0 - x);
        radiance += f0 * sky(L);
    }

    // Direct lighting:
    for (int i = 0; i < MAX_LIGHTS; ++i)
    {
        radiance += lightContribution(lights[i], p, V, N, albedo, f0, m.roughness);
    }

    float fogAmount = 1.0 - exp(-t.x*0.01);
    vec3 fogColor = vec3(0.001,0.001,0.005);
    radiance = mix(radiance, fogColor, fogAmount);

    return radiance;
}
