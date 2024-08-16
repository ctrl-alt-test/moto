vec3 evalNormal(vec3 p)
{
    // TODO: Ideally h would depend on the screen space projected size.
    const float h = NORMAL_DP;
    const vec2 k = vec2(1., -1.);
    return normalize(
        k.xyy * sceneSDF(p + k.xyy * h).x + 
        k.yyx * sceneSDF(p + k.yyx * h).x + 
        k.yxy * sceneSDF(p + k.yxy * h).x + 
        k.xxx * sceneSDF(p + k.xxx * h).x
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

    for (int i = ZERO; i < max_steps; ++i)
    {
#ifdef ENABLE_STEP_COUNT
        steps = i + 1;
#endif

        d = sceneSDF(p);
        t += d.x;
        p = ro + t * rd;

        if (d.x < EPSILON)
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
    if (mid == NO_ID) // background / sky
    {
        vec3 rd = -V;
        float y = max(rd.y, 0.01);
        vec3 col = mix(vec3(0.03,0.002,0.01), vec3(0,0,0.05), y);
        vec3 p = normalize(rd*vec3(0.1,1,0.1));
        float den = exp(-1. * fBm(p.xz*3. + vec2(iTime*0.01), 5, 0.6, 5.5));
        col = mix(col, col+vec3(0.01,0.01,0.02),
            smoothstep(.1, 1., den) * (1. - clamp(1. / (10.*y),0.,1.)));
        
        
        float moonDistance = distance(rd, normalize(vec3(0.5,0.3,-1.)));
        vec3 moonColor = vec3(1.,1.,1.) * den;
        float moon = smoothstep(0.02,0.018, moonDistance);
        col = mix(col, moonColor, moon*.1);
        float halo = smoothstep(0.08,0., moonDistance);
        col = mix(col, moonColor, halo*.02);
        return col;
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

    float fogAmount = 1.0 - exp(-t.x*0.03);
    vec3 fogColor = vec3(0,0,0.005)+vec3(0.01,0.01,0.02)*0.1;
    radiance = mix(radiance, fogColor, fogAmount);

    return radiance;
}
