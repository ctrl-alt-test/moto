#ifdef ENABLE_DAY_MODE
vec3 daySkyDomeLight = 0.5 * vec3(0.25, 0.5, 1.0);
vec3 sunLight = vec3(1.0, 0.85, 0.7);
#else
vec3 nightHorizonLight = 0.01 * vec3(0.07, 0.1, 1.0);
vec3 moonLight = 0.02 * vec3(0.2, 0.8, 1.0);
#endif

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

    float cloud = fBm(0.015*iTime+V.xz/(0.01 + V.y) * 0.5, 5, 0.55, 0.7);
    cloud = smoothstep(0., 1., cloud+1.);

    color *= mix(0.1, 1., pow(cloud, 2.));
    color *= smoothstep(-0.1, 0., V.y);
    return color / 197000.;
}
