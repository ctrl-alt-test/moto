in vec3 camPos;
in vec3 camTa;
in float camMotoSpace;
in float camFocal;
in float fishEyeFactor;

#define MAX_HM_STEPS 50

// -------------------------------------------------------
// Scene description functions

vec3 daySkyDomeLight = 0.5 * vec3(0.25, 0.5, 1.0);
vec3 sunLight = vec3(1.0, 0.85, 0.7);

vec3 nightHorizonLight = 0.01 * vec3(0.07, 0.1, 1.0);
vec3 moonLight = 0.02 * vec3(0.2, 0.8, 1.0);

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

    if (mid >= MOTO_ID && mid <= MOTO_MOTOR_ID)
    {
        p = worldToMoto(p, true, iTime);
        N = worldToMoto(N, false, iTime);
        //return material(N * 0.5 + 0.5, vec3(0.), 0.15);
        return motoMaterial(mid, p, N, iTime);
    }

    if (mid == DRIVER_ID)
    {
        return material(vec3(0.0), vec3(0.02, 0.025, 0.04), 0.6);
    }
    if (mid == DRIVER_HELMET_ID)
    {
        return material(vec3(0.0), vec3(0.), 0.25);
    }

    return material(vec3(0.0), fract(p.xyz), 1.0);
}

float tree(vec3 p, vec3 globalP, vec3 id) {
    float ha = hash(id);

    // Remove half of the trees
    if (hash(ha) < .5) return 0.5;
    // and trees near the road
    vec4 splineUV = ToSplineLocalSpace(id.xz, roadWidthInMeters.z);
    if (abs(splineUV.x) < 5.5) return 0.5;
    
    float y = smoothTerrainHeight(globalP.xz);
    float height = 6. - 4.5*ha;
    p -= vec3(.8*(ha-0.5), y + 0.5, 1.2*(ha-0.5));
    float d = Ellipsoid(p, vec3(0.2, 1., 0.2));
    d = min(d, Ellipsoid(p + vec3(0,-2.- height/2.,0), vec3(0.8, height, 0.8)));
    d += fBm(p.xy + p.yz + id.xz, 4, 1., .5) * 0.05;

    return d;
}

vec2 trees(vec3 p) {
    float spacing = 3.;

    // iq - repeated_ONLY_SYMMETRIC_SDFS (https://iquilezles.org/articles/sdfrepetition/)
    vec3 lim = vec3(1e8,0,1e8);
    vec3 id = clamp(round(p / spacing), -lim, lim);
    vec3 localP = p - spacing * id;
    return vec2(tree(localP, p, id), GROUND_ID);
}


vec2 sceneSDF(vec3 p)
{
    vec2 d = vec2(1e6, NO_ID);

    d = MinDist(d, motoShape(p));
    d = MinDist(d, driverShape(p));
    d = MinDist(d, terrainShape(p));
    d = MinDist(d, trees(p));

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
    ComputeBezierSegmentsLength();
    GenerateSpline(1.2/*curvature*/, 2./*scale*/, 1./*seed*/);

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
