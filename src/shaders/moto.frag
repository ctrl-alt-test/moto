// -------------------------------------------------------
// Scene description functions

material computeMaterial(float mid, vec3 p, vec3 N)
{
    if (mid == GROUND_ID)
    {
        vec3 color = pow(vec3(67., 81., 70.) / 255. * 1.5, vec3(GAMMA));

        vec4 splineUV = ToSplineLocalSpace(p.xz, roadWidthInMeters.z);
        float isRoad = 1.0 - smoothstep(roadWidthInMeters.x, roadWidthInMeters.y, abs(splineUV.x));
        vec3 roadColor = vec3(0.0);
        if (isRoad > 0.0)
        {
            roadColor = roadPattern(splineUV.zx, 3.5, vec2(0.7, 0.0));
        }
        color = mix(color, roadColor, isRoad);
        return material(vec3(0.0), color, 0.5);
    }

    if (IsMoto(mid))
    {
        p = worldToMoto(p, true, iTime);
        N = worldToMoto(N, false, iTime);
        //return material(N * 0.5 + 0.5, vec3(0.), 0.15);
        return motoMaterial(mid, p, N, iTime);
    }

    return material(vec3(0.0), fract(p.xyz), 1.0);
}

float tree(vec3 globalP, vec3 localP, vec2 id, vec4 splineUV, float current_t) {
    float h1 = hash21(id);
    float h2 = hash11(h1);

    // Define if the area has trees
    float presence = smoothstep(-0.7, 0.7, fBm(id / 500., 2, 0.5, 0.3));
    if (h1 < presence)
    {
        return 1e6;
    }

    // Clear trees close to the road
    if (abs(splineUV.x) < roadWidthInMeters.y) return 1e6;

    //
    // FIXME: the splineUV is relative to the current position, not relative
    // to the tree position.
    // This will probably need some coordinate trickery to know if there is
    // a tree or not.
    // But if that doesn't work, we can still use splineUV to ignore cases in
    // which we are sure there is or there is no tree. Then for cases in
    // between, we can evaluate the spline relative to the tree position.
    // That should still be a lot fewer spline evaluations.
    //

    float treeHeight = mix(5., 20., 1.-h1*h1);
    float treeWidth = treeHeight * mix(0.3, 0.5, h2*h2);
    float terrainHeight = smoothTerrainHeight(id);

    localP.y -= terrainHeight + 0.5 * treeHeight;
    localP.xz += (vec2(h1, h2)*2. - 1.) * 2.;

    float d = Ellipsoid(localP, 0.5*vec3(treeWidth, treeHeight, treeWidth));

    float leaves = 1. - smoothstep(50., 200., current_t);
    if (d < 2. && leaves > 0.)
    {
        d += leaves * fBm(5. * vec2(2.*atan(localP.z, localP.x), localP.y) + id, 2, 0.5, 0.5) * 0.5;
    }

    return d;
}

vec2 treesShape(vec3 p, vec4 splineUV, float current_t)
{
    float spacing = 10.;

    // iq - repeated_ONLY_SYMMETRIC_SDFS (https://iquilezles.org/articles/sdfrepetition/)
    //vec3 lim = vec3(1e8,0,1e8);
    vec2 id = round(p.xz / spacing) * spacing;
    vec3 localP = p;
    localP.xz -= id;
    return vec2(tree(p, localP, id, splineUV, current_t), GROUND_ID);
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

vec2 sceneSDF(vec3 p, float current_t)
{
    vec2 d = vec2(1e6, NO_ID);

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

// -------------------------------------------------------
// Rendering functions

#include "rendering.frag"

// -------------------------------------------------------

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    GenerateSpline(PI/*curvature*/, 10./*scale*/, 1./*seed*/);
    ComputeBezierSegmentsLengthAndAABB();

    vec2 uv = (fragCoord/iResolution.xy * 2. - 1.) * vec2(1., iResolution.y / iResolution.x);
    
    vec2 mouseInput = iMouse.xy / iResolution.xy;

    float time = iTime;
#if ENABLE_STOCHASTIC_MOTION_BLUR
    time += hash31(vec3(fragCoord, 1e-3*iTime)) * 0.008;
#endif
    // orbitalCamera(uv, vec2(0.02 * time, 0.3) /*mouseInput*/, ro, rd);

    float ti = fract(iTime * 0.1);
    motoPos.xz = GetPositionOnCurve(ti);
    motoPos.y = smoothTerrainHeight(motoPos.xz);
    vec3 nextPos;
    nextPos.xz = GetPositionOnCurve(ti+0.01);
    nextPos.y = smoothTerrainHeight(nextPos.xz);
    motoDir = normalize(nextPos - motoPos);

    // camPos and camTa are passed by the vertex shader
    vec3 ro;
    vec3 rd;
    vec3 cameraPosition = camPos;
    vec3 cameraTarget = camTa;
    vec3 cameraUp = vec3(0., 1., 0.);
    if (camMotoSpace > 0.5) {
        cameraPosition = motoToWorld(camPos, true, iTime);
        cameraTarget = motoToWorld(camTa, true, iTime);
        //cameraUp = motoToWorld(cameraUp, false, iTime);
    }
    setupCamera(uv, cameraPosition, cameraTarget, cameraUp, camProjectionRatio, camFishEye, ro, rd);

    // View moto from front
    // motoCamera(uv, vec3(1.26, 1.07, 0.05), vec3(-10.,0.,0), ro, rd);

    // First-person view
    // motoCamera(uv, vec3(0.02, 1.2, 0.05), vec3(10.,0.,0.), ro, rd);

    // Third-person view, near ground
    // motoCamera(uv, vec3(-2., 0.5, -0.2), vec3(10.,0.,0.), ro, rd);


    vec3 p;
    vec2 t = rayMarchScene(ro, rd, MAX_RAY_MARCH_DIST, MAX_RAY_MARCH_STEPS, p);
    vec3 N = evalNormal(p, t.x);

    vec3 radiance = evalRadiance(t, p, -rd, N);
    
    vec3 color = pow(radiance, vec3(1. / GAMMA));
    fragColor = vec4(color, 1.);
}


void main() {
    mainImage(fragColor, gl_FragCoord.xy);
}
