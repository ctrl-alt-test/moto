// -------------------------------------------------------
// Scene description functions

M computeMaterial(int mid, vec3 p, vec3 N)
{
#ifdef DEBUG
    if (mid == DEBUG_ID)
    {
        return M(MATERIAL_TYPE_EMISSIVE, N * 0.5 + 0.5, 1.0);
    }
#endif

    vec4 splineUV;
    vec3 pRoad = p;
    if (IsRoad(mid))
    {
        splineUV = ToSplineLocalSpace(p.xz, roadWidthInMeters.z);
        pRoad.xz = splineUV.xz;
    }

    if (mid == TREE_ID)
    {
        return M(MATERIAL_TYPE_DIELECTRIC, vec3(0.1, 0.2, 0.05), 0.7);
    }
        
    if (mid == GROUND_ID)
    {
        float isRoad = 1.0 - smoothstep(roadWidthInMeters.x, roadWidthInMeters.y, abs(splineUV.x));
        vec3 grassColor = vec3(0.22, 0.21, 0.04);
        if (isRoad > 0.)
        {
            M m = roadMaterial(splineUV.xz, 3.5, vec2(0.7, 0.0));
            m.C = mix(grassColor, m.C, isRoad);
            return m;
        }
        // Terrain
        return M(MATERIAL_TYPE_DIELECTRIC, grassColor, 0.8);
    }

    if (IsMoto(mid))
    {
        p = worldToMoto(p, true);
        N = worldToMoto(N, false);
        return motoMaterial(mid, p, N);
    }


    M utility = M(MATERIAL_TYPE_METALLIC, vec3(0.9), 0.7);
    if (mid == ROAD_UTILITY_ID)
    {
        return utility;
    }
    if (mid == ROAD_LIGHT_ID)
    {
        if (N.y > -0.5)
        {
            return utility;
        }
        return M(MATERIAL_TYPE_EMISSIVE, vec3(5., 3., 0.1), 0.4);
    }

    if (mid == ROAD_WALL_ID)
    {
        return M(MATERIAL_TYPE_DIELECTRIC, vec3(.5)+fBm(pRoad.yz*vec2(.2,1)+valueNoise(pRoad.xz),3,.6,.9)*.15, .6);
    }

    if (mid == ROAD_REFLECTOR_ID)
    {
        return M(MATERIAL_TYPE_RETROREFLECTIVE, vec3(1., 0.4, 0.05), 0.2);
    }

    return M(MATERIAL_TYPE_DIELECTRIC, fract(p.xyz), 1.0);
}

vec2 sceneSDF(vec3 p, float current_t)
{
    vec4 splineUV = ToSplineLocalSpace(p.xz, roadWidthInMeters.z);

#ifndef DISABLE_MOTO
    vec2 d = motoShape(p);
#else
    vec2 d = vec2(INF, NO_ID);
#endif
#ifndef DISABLE_MOTO_DRIVER
    d = MinDist(d, driverShape(p));
#endif
#ifndef DISABLE_TERRAIN
    d = MinDist(d, terrainShape(p, splineUV));
#endif
#ifndef DISABLE_TREES
    d = MinDist(d, treesShape(p, splineUV, current_t));
#endif

    return d;
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

    for (int i = min(0, int(iTime)); i < max_steps; ++i)
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
    int mid = int(t.y);
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

    M m = computeMaterial(mid, p, N);

    vec3 emissive = vec3(0.);
    if (m.T == MATERIAL_TYPE_EMISSIVE)
    {
        // Stronger luminance in the direction of the normal:
        // https://graphtoy.com/?f1(x,t)=mix(0.1*x+pow(x,4),1,pow(x,8))&coords=0.4,0.4,1
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

    // Crude global illumination coming from the sky dome:
#ifdef ENABLE_DAY_MODE
    // Day version:
    vec3 daySkyDomeLight = 0.8 * vec3(0.25, 0.5, 1.0);
    vec3 I0 = daySkyDomeLight * (N.y * 0.5 + 0.5);
#else
    vec3 nightHorizonLight = 0.01 * vec3(0.07, 0.1, 1.0);
    // Night version:
    vec3 I0 = nightHorizonLight * mix(1.0, 0.1, N.y * N.y) * (N.x * 0.5 + 0.5);
#endif
    radiance += I0 * albedo;

    // Env map:
    if (m.R < 0.25) // Brutal if until there's a roughness dependent reflection.
    {
        vec3 L = reflect(-V, N);
        // vec3 H = normalize(L + V);
	    // float x = 1.0 - dot(V, H);
	    // x = x*x*x*x*x;
	    // vec3 F = x + f0 * (1.0 - x);
        radiance += f0 * sky(L);
    }

    // Direct lighting:
    for (int i = 0; i < MAX_ROAD_LIGHTS + 3; ++i)
    {
        L light;

        // Environment light
        if (i == MAX_ROAD_LIGHTS)
        {
#ifdef ENABLE_DAY_MODE
            vec3 sunLightColor = vec3(1.0, 0.85, 0.7);
            light = L(moonDirection * 1e3, -moonDirection, sunLightColor, 0., 0., 1e10, 5.);
#else
            vec3 moonLightColor = vec3(0.2, 0.8, 1.0);
            light = L(moonDirection * 1e3, -moonDirection, moonLightColor, 0., 0., 1e10, 0.005);
#endif
        }

        // Head light
        if (i == MAX_ROAD_LIGHTS + 1)
        {
            vec3 pos = motoToWorld(headLightOffsetFromMotoRoot + vec3(0.1, 0., 0.), true);
            vec3 dirHeadLight = normalize(vec3(1.0, -0.15, 0.0));
            vec3 dir = motoToWorld(dirHeadLight, false);
            light = L(pos, dir, vec3(1.), 0.75, 0.95, 10.0, 5.);
        }

        // Break light
        if (i == MAX_ROAD_LIGHTS + 2)
        {
            vec3 pos = motoToWorld(breakLightOffsetFromMotoRoot, true);
            vec3 dirBreakLight = normalize(vec3(-1.0, -0.5, 0.0));
            vec3 dir = motoToWorld(dirBreakLight, false);
            light = L(pos, dir, vec3(1., 0., 0.), 0.3, 0.9, 2.0, 0.05);
        }

        // Road lights
        if (i < MAX_ROAD_LIGHTS)
        {
            float t = float(i/2 - MAX_ROAD_LIGHTS/4 + 1);

            float roadLength = splineSegmentDistances[SPLINE_SIZE / 2 - 1].y;
            float motoDistanceOnRoad = motoDistanceOnCurve * roadLength;

            float distanceOfFirstLamp = floor(motoDistanceOnRoad / DISTANCE_BETWEEN_LAMPS) * DISTANCE_BETWEEN_LAMPS;
            float distanceOfCurrentLamp = distanceOfFirstLamp + t * DISTANCE_BETWEEN_LAMPS;

            float distanceOnCurve = distanceOfCurrentLamp / roadLength;
            if (distanceOnCurve >= 1.)
            {
                // End of the road.
                // (hard coded 0.97 value because I couldn't determine
                // the exact condition; I suspect the +0.02 in
                // getRoadDirectionAndPosition)
                continue;
            }
            vec3 pos;
            vec4 roadDirAndCurve = getRoadPositionDirectionAndCurvature(distanceOnCurve, pos);
            roadDirAndCurve.y = 0.;

            pos.x += (roadWidthInMeters.x - 1.) * 1.2 * (float(i % 2) * 2. - 1.);
            pos.y += 5.;
            // 90� rotation:
            //roadDirAndCurve.xz = vec2(roadDirAndCurve.z, -roadDirAndCurve.x);

            vec3 coldNeon = vec3(0.8, 0.9, 1.);
            vec3 warmNeon = vec3(1., 0.9, 0.7);
            vec3 sodium = vec3(1., 0.3, 0.0);
            light = L(pos, pos + roadDirAndCurve.xyz, sodium, -1.0, 0.0, 0.0, 10.0);
        }

        radiance += lightContribution(light, p, V, N, albedo, f0, m.R);
    }

    float fogAmount = 1.0 - exp(-t.x*0.01);
    vec3 fogColor = vec3(0.001,0.001,0.005);
    radiance = mix(radiance, fogColor, fogAmount);

    return radiance*2;
}
