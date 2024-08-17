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

        if (d.x < MOTO_EPSILON || (d.x < EPSILON && !IsMoto(d.y)))
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

vec3 cityLights(vec2 p)
{
    vec3 ctex=vec3(0);
    for(int i=0;i<3;i++) {
        float fi=float(i);
        vec2 xp=p*Rotation(max(fi-3.,0.)*.5)*(1.+fi*.3),mp=mod(xp,10.)-5.;
        // adjust this value  ↓  based on depth to reduce shimmering
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


    // Global illumination coming from the sky dome:

    // Direct light:
    vec3 L1 = normalize(vec3(0.4, 0.6*1.0, 0.8));
    float NdotL1 = clamp(dot(N, L1), 0.0, 1.0);
    float visibility_L1 = castShadowRay(p, N, L1);

#ifdef ENABLE_DAY_MODE
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
