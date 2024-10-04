#ifdef ENABLE_DAY_MODE
vec3 daySkyDomeLight = 0.8 * vec3(0.25, 0.5, 1.0);
vec3 sunLightColor = vec3(1.0, 0.85, 0.7);
#else
vec3 nightHorizonLight = 0.01 * vec3(0.07, 0.1, 1.0);
vec3 moonLightColor = vec3(0.2, 0.8, 1.0);
#endif

vec3 moonDirection = normalize(vec3(-1., 0.3, 0.4));

vec3 sky(vec3 V)
{
#ifdef ENABLE_DAY_MODE
    return mix(vec3(0.6, 0.8, 1.), vec3(0.01, 0.35, 1.), pow(smoothstep(0.15, 1., V.y), 0.4));
#endif
    vec3 darkest = vec3(0, 0, 2);
    vec3 deepest = vec3(0, 3, 18);
    vec3 clearest = vec3(3, 7, 19);
    vec3 greenest = vec3(3, 12, 18);
    vec3 fog = vec3(1, 4, 7);
    vec3 moonColor = vec3(10., 15, 13);

    float direction = clamp(dot(V, normalize(vec3(0., 1., 0.25))), 0., 1.);
    vec3 color = mix(clearest, darkest, sqrt(direction));
    color = mix(greenest, color, smoothstep(0., 0.3, mix(V.y, direction, 0.5)));
    color = mix(fog, color, smoothstep(0.05, 0.15, V.y));

    float cloud = fBm(0.015*time+V.xz/(0.01 + V.y) * 0.5, 5, 0.55, 0.7);
    cloud = smoothstep(0., 1., cloud+1.);
    cloud *= cloud;

    float moon = dot(V, moonDirection) - 1.;
    moon = smoothstep(0., 0.4, moon*10000.+1.);
    if (moon > 0.)
    {
        float pattern = .5 + smoothstep(-1., 1., -fBm(V.yz * 100., 2, .9, 0.5));
        color = mix(color, moonColor * 100. * pattern*cloud*cloud, moon);
    }

    return color
        * mix(0.1, 1., cloud)
        * smoothstep(-0.1, 0., V.y)
        / 2000.;
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

vec2 cityShape(vec3 p){
    vec3 o=p;
    // Put the city in a box
    float len = Box3(p - vec3(150, 0, 0), vec3(1., 200., 200.), 0.01);
    if (len > 10.) return vec2(len-5., CITY_ID);

    // LJ
    float seed=hash21(floor(o.xz/14.));
    p.xz=mod(p.xz*Rotation(.7)+seed*(6.-3.)*5.,14.)-7.;
    float buildingCutouts = max(max(abs(p.x),abs(p.z))-2.,p.y-seed*5.);
    p.xz=mod(o.xz+6.,14.)-7.;
    buildingCutouts = min(buildingCutouts,max(max(abs(p.x),abs(p.z))-2.,p.y-seed*5.));
    return
        vec2(max(min(buildingCutouts*.5,p.y),o.z),
            CITY_ID);
}
