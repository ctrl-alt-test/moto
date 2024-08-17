in vec3 camPos;
in vec3 camTa;
in float camMotoSpace;
in float camFocal;
in float fishEyeFactor;

#define MAX_HM_STEPS 50

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

float tree(vec3 p, vec3 globalP, vec3 id, vec4 splineUV) {
    float ha = hash(id);

    // Remove half of the trees
    if (hash(ha) < .5) return 0.5;
    // and trees near the road
    if (abs(splineUV.x) < 5.5) return 0.5;

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

    float y = smoothTerrainHeight(globalP.xz);
    float height = 6. - 4.5*ha;
    p -= vec3(.8*(ha-0.5), y + 0.5, 1.2*(ha-0.5));
    float d = Ellipsoid(p, vec3(0.2, 1., 0.2));
    d = min(d, Ellipsoid(p + vec3(0,-2.- height/2.,0), vec3(0.8, height, 0.8)));
    d += fBm(p.xy + p.yz + id.xz, 4, 1., .5) * 0.05;

    return d;
}

vec2 treesShape(vec3 p, vec4 splineUV) {
    float spacing = 3.;

    // iq - repeated_ONLY_SYMMETRIC_SDFS (https://iquilezles.org/articles/sdfrepetition/)
    vec3 lim = vec3(1e8,0,1e8);
    vec3 id = clamp(round(p / spacing), -lim, lim);
    vec3 localP = p - spacing * id;
    return vec2(tree(localP, p, id, splineUV), GROUND_ID);
}


vec2 sceneSDF(vec3 p)
{
    vec2 d = vec2(1e6, NO_ID);

    vec4 splineUV = ToSplineLocalSpace(p.xz, roadWidthInMeters.z);

    d = MinDist(d, motoShape(p));
    d = MinDist(d, driverShape(p));
    d = MinDist(d, terrainShape(p, splineUV));
    d = MinDist(d, treesShape(p, splineUV));

    return d;
}

// -------------------------------------------------------
// Rendering functions

#include "rendering.frag"

// -------------------------------------------------------

void motoCamera(vec2 uv, vec3 relativePos, vec3 relativeTa, out vec3 ro, out vec3 rd)
{
    vec3 cameraPosition = motoToWorld(relativePos, true, iTime);

    vec3 cameraTarget = cameraPosition + relativeTa;
    vec3 cameraForward = normalize(cameraTarget - cameraPosition);
    vec3 cameraUp = vec3(0., 1., 0.);
    cameraUp = motoToWorld(cameraUp, false, iTime);
    if (abs(dot(cameraForward, cameraUp)) > 0.99)
    {
        cameraUp = vec3(1., 0., 0.);
    }
    vec3 cameraRight = normalize(cross(cameraForward, cameraUp));
    cameraUp = normalize(cross(cameraRight, cameraForward));

    ro = cameraPosition;
    rd = normalize(cameraForward + uv.x * cameraRight + uv.y * cameraUp);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    ComputeBezierSegmentsLengthAndAABB();
    GenerateSpline(1.2/*curvature*/, 4./*scale*/, 1./*seed*/);

    vec2 uv = (fragCoord/iResolution.xy * 2. - 1.) * vec2(1., iResolution.y / iResolution.x);
    
    vec2 mouseInput = iMouse.xy / iResolution.xy;

    float time = iTime;
#if ENABLE_STOCHASTIC_MOTION_BLUR
    time += hash(vec3(fragCoord, 1e-3*iTime)) * 0.008;
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
    vec2 v = uv*2.-1.;
    vec3 ro = camPos;
    vec3 rd;
    if (camMotoSpace > 0.5) {
        motoCamera(uv, camPos, camTa, ro, rd);
    } else {
        rd = lookat(ro, camTa) * normalize(vec3(v, camFocal - length(v) * fishEyeFactor));
    }

    // View moto from front
    // motoCamera(uv, vec3(1.26, 1.07, 0.05), vec3(-10.,0.,0), ro, rd);

    // First-person view
    // motoCamera(uv, vec3(0.02, 1.2, 0.05), vec3(10.,0.,0.), ro, rd);

    // Third-person view, near ground
    // motoCamera(uv, vec3(-2., 0.5, -0.2), vec3(10.,0.,0.), ro, rd);


    vec3 p;
    vec2 t = rayMarchScene(ro, rd, MAX_RAY_MARCH_DIST, MAX_RAY_MARCH_STEPS, p);
    vec3 N = evalNormal(p);

    vec3 radiance = evalRadiance(t, p, -rd, N);
    
    vec3 color = pow(radiance, vec3(1. / GAMMA));
    fragColor = vec4(color, 1.);
}


void main() {
    mainImage(fragColor, gl_FragCoord.xy);
}
