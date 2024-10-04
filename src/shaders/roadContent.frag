vec4 splineAABB;
vec4 splineSegmentAABBs[SPLINE_SIZE / 2];
vec2 splineSegmentDistances[SPLINE_SIZE / 2]; // x: start; y: end


// Piece of code copy-pasted from:
// https://www.shadertoy.com/view/NltBRB
// Credit: MMatteini

void ComputeBezierSegmentsLengthAndAABB()
{
    float splineLength = 0.0;
    splineAABB = vec4(INF, INF, -INF, -INF);

    for (int i = min(0, int(iTime)); i < SPLINE_SIZE / 2; ++i)
    {
        int index = 2 * i;
        vec2 A = spline[index + 0];
        vec2 B = spline[index + 1];
        vec2 C = spline[index + 2];
        float segmentLength = BezierCurveLengthAt(A, B, C, 1.0);
        splineSegmentDistances[i].x = splineLength;
        splineLength += segmentLength;
        splineSegmentDistances[i].y = splineLength;

        vec4 AABB = BezierAABB(A, B, C);
        splineSegmentAABBs[i] = AABB;
        splineAABB.xy = min(splineAABB.xy, AABB.xy);
        splineAABB.zw = max(splineAABB.zw, AABB.zw);
    }
}

// Decompose a give location into its distance from the closest point on the spline. 
// and the length of the spline up to that point.
// Returns a vector where:
// X = signed distance from the spline
// Y = spline parameter t in the Bezier segment [0; 1]
// Z = spline length at the closest point
// W = spline segment index
//
// To get t of the entire spline:
// (Y + 0.5 * W) / float(SPLINE_SIZE)
//
vec4 ToSplineLocalSpace(vec2 p, float splineWidth)
{
    vec4 splineUV = vec4(INF, 0, 0, 0);

    if (DistanceFromAABB(p, splineAABB) > splineWidth)
    {
        return splineUV;
    }

    // For each bezier segment
    for (int i = min(0, int(iTime)); i < SPLINE_SIZE / 2; ++i)
    {
        int index = 2 * i;
        vec2 A = spline[index + 0];
        vec2 B = spline[index + 1];
        vec2 C = spline[index + 2];

        if (DistanceFromAABB(p, BezierAABB(A, B, C)) > splineWidth)
        {
            continue;
        }

        // This is to prevent 3 colinear points, but there should be better solution to it.
        B = mix(B + vec2(1e-4), B, abs(sign(B * 2.0 - A - C))); 
        // Current bezier curve SDF
        vec2 bezierSDF = BezierSDF(A, B, C, p);

        if (abs(bezierSDF.x) < abs(splineUV.x))
        {
            float lengthInSegment = BezierCurveLengthAt(A, B, C, clamp(bezierSDF.y, 0., 1.));
            float lengthInSpline = splineSegmentDistances[i].x + lengthInSegment;
            splineUV = vec4(
                bezierSDF.x,
                clamp(bezierSDF.y, 0., 1.),
                lengthInSpline,
                float(index));
        }
    }

    return splineUV;
}

//
// 2D position on a given Bezier curve of the spline.
// - x in [0, 1] of that curve.
// - y the index of the curve in the spline.
//
// If you have a splineUV, call:
// position = GetPositionOnSpline(splineUV.yw, directionAndCurvature)
//
// If you don't, get the pair with GetTAndIndex(t)
//
vec2 GetPositionOnSpline(vec2 spline_t_and_index, out vec3 directionAndCurvature)
{
    float t = spline_t_and_index.x;
    int index = int(spline_t_and_index.y);
    vec2 A = spline[index + 0];
    vec2 B = spline[index + 1];
    vec2 C = spline[index + 2];

    vec2 AB = mix(A, B, t);
    vec2 BC = mix(B, C, t);

    directionAndCurvature.xy = 2.0 * (BC - AB);

    // Second derivative
    vec2 d2 = 2.0 * (C - 2.0 * B + A);     
    // Determinant (cross product)
    // -----------
    //   |d1|^3
    float norm = length(directionAndCurvature.xy);
    directionAndCurvature.z = directionAndCurvature.x * d2.y - directionAndCurvature.y * d2.x;
    directionAndCurvature.z /= norm*norm*norm;

    // Position:
    return mix(AB, BC, t);
}

//
// 2D position on the curve.
// - t in [0, 1]
//
vec2 GetTAndIndex(float t)
{
    // Desired length along the spline for the given t
    float targetLength = t * splineSegmentDistances[SPLINE_SIZE / 2 - 1].y;
    
    // Find the segment corresponding to this target length
    int index = 0;
    while (index < SPLINE_SIZE / 2 && targetLength > splineSegmentDistances[index].y)
    {
        ++index;
    }

    float segmentStartDistance = splineSegmentDistances[index].x;
    float segmentEndDistance = splineSegmentDistances[index].y;
    
    // Calculate how far along the segment we are
    float segmentT = (targetLength - segmentStartDistance) / (segmentEndDistance - segmentStartDistance);

    return vec2(segmentT, index * 2.0);
}

// x: actual width
// y: width + transition
// z: max width
vec3 roadWidthInMeters = vec3(4.0, 8.0, 8.0);
const float laneWidth = 3.5;

float roadMarkings(vec2 uv, float width, vec2 params)
{
    // Total interval, line length
    vec2 t1  = vec2(26.0 / 2.0, 3.0);
    vec2 t1b = vec2(26.0 / 4.0, 1.5);
    vec2 t2  = vec2(26.0 / 4.0, 3.0);
    vec2 t3  = vec2(26.0 / 6.0, 3.0);
    vec2 t3b = vec2(26.0 / 1.0, 20.0);
    vec2 continuous = vec2(100.0, 100.0);

    vec2 separationLineParams = t1;
    if (params.x > 0.25) separationLineParams = t1b;
    if (params.x > 0.50) separationLineParams = t3;
    if (params.x > 0.75) separationLineParams = continuous;

    vec2 sideLineParams = t2;
    if (width > 4.0) sideLineParams = t3b;

    float tileY = uv.y - floor(clamp(uv.y, 3.5-width, width) / 3.5) * 3.5;
    vec2 separationTileUV = vec2(fract(uv.x / separationLineParams.x) * separationLineParams.x, tileY);
    vec2 sideTileUV = vec2(fract((uv.x + 0.4) / sideLineParams.x) * sideLineParams.x, uv.y);

    float sideLine1 = Box2(sideTileUV - vec2(0.5 * sideLineParams.y, width), vec2(0.5 * sideLineParams.y, 0.10), 0.03);
    float sideLine2 = Box2(sideTileUV - vec2(0.5 * sideLineParams.y, -width), vec2(0.5 * sideLineParams.y, 0.10), 0.03);

    float separationLine1 = Box2(separationTileUV - vec2(0.5 * separationLineParams.y, 0.0), vec2(0.5 * separationLineParams.y, 0.10), 0.01);

    float pattern = min(min(sideLine1, sideLine2), separationLine1);

    return 1.-smoothstep(-0.01, 0.01, pattern+valueNoise(uv*30)*.03*valueNoise(uv));
}

M roadMaterial(vec2 uv, float width, vec2 params)
{
    vec2 laneUV = uv / laneWidth;

    float tireTrails = sin((laneUV.x-0.125) * 4. * PI) * 0.5 + 0.5;
    tireTrails = mix(tireTrails, smoothstep(0., 1., tireTrails), 0.25);

    float largeScaleNoise = smoothstep(-0.25, 1., fBm(laneUV * vec2(15., 0.1), 2, .7, .4));
    tireTrails = mix(tireTrails, largeScaleNoise, 0.2);

    float highFreqNoise = fBm(laneUV * vec2(150., 6.), 1, 1., 1.);
    tireTrails = mix(tireTrails, highFreqNoise, 0.1);

    float roughness = mix(0.8, 0.4, tireTrails);
    vec3 color = vec3(mix(vec3(0.11, 0.105, 0.1), vec3(0.15), tireTrails));


    float paint = roadMarkings(uv.yx, width, params);
    color = mix(color, vec3(1), paint);
    roughness = mix(roughness, .7, paint);

    // DEBUG --------
    /*
    vec2 marks = abs(fract(laneUV) * 2. - 1.);
    vec2 dmarks = fwidth(marks);
    marks = smoothstep(-dmarks, dmarks, marks - 0.99);
    vec3 debugUV = vec3(laneUV, 0.);
    debugUV = fract(debugUV);
    debugUV = clamp(debugUV, 0., 1.);
    debugUV = mix(debugUV * 0.5, vec3(1.), max(marks.x, marks.y));
    if (laneUV.x < 0.)
    {
        color = debugUV;
    }
    else
    {
        color = mix(color, vec3(1.), marks.x);
    }
    /**/
    // --------------

    return M(paint > 0.5 ? MATERIAL_TYPE_RETROREFLECTIVE : MATERIAL_TYPE_DIELECTRIC, color, roughness);
}

const float terrain_fBm_weight_param = 0.6;
const float terrain_fBm_frequency_param = 0.5;

float smoothTerrainHeight(vec2 p)
{
    float hillHeightInMeters = 100.;
    float hillLengthInMeters = 2000.;

    return 0.5 * hillHeightInMeters * fBm(p * 2. / hillLengthInMeters, 3, terrain_fBm_weight_param, terrain_fBm_frequency_param);
}

#pragma function inline
float terrainDetailHeight(vec2 p)
{
    float detailHeightInMeters = 1.;
    float detailLengthInMeters = 100.;

    return valueNoise(p*10.)*0.1 + 0.5 * detailHeightInMeters * fBm(p * 2. / detailLengthInMeters, 1, terrain_fBm_weight_param, terrain_fBm_frequency_param);
}

float roadBumpHeight(float d)
{
    float x = clamp(abs(d / roadWidthInMeters.x), 0., 1.);
    return 0.2 * (1. - x * x * x);
}

//
// Returns the 3D direction and curvature as a 4D return value, and the
// 3D position as an out argument, on the road spline at t in [0, 1].
//
vec4 getRoadPositionDirectionAndCurvature(float t, out vec3 position)
{
    vec4 directionAndCurvature;
    position.xz = GetPositionOnSpline(GetTAndIndex(t), directionAndCurvature.xzw);
    position.y = smoothTerrainHeight(position.xz);
    directionAndCurvature.y = smoothTerrainHeight(position.xz + directionAndCurvature.xz) - position.y;

    directionAndCurvature.xyz = normalize(directionAndCurvature.xyz);
    return directionAndCurvature;
}

vec2 roadSideItems(vec4 splineUV, float relativeHeight) {
    vec2 res = vec2(1e6, NO_ID);
    vec3 pRoad = vec3(abs(splineUV.x), relativeHeight, splineUV.z);

    pRoad.x -= roadWidthInMeters.x * 1.2;
    // float guardrailHeight = smoothstep(0., 1., abs(fract(splineUV.y*2.) * 2. - 1.)*2.) *2. - 1.;

    if (wallHeight >= 0.) {
        guardrailHeight = -1.;
    }
    float lampHeight = 7.;

    vec3 pReflector = vec3(pRoad.x, pRoad.y - 0.8, round(pRoad.z / 4.) * 4. - pRoad.z);

	// Traffic barrier
    if (guardrailHeight > -0.5)
    {
        float height = 0.8 * guardrailHeight;
        vec3 pObj = vec3(pRoad.x, pRoad.y - height, 0.);
        float len = Box3(pObj, vec3(0.1, 0.2, 0.1), 0.05);

        pObj = vec3(pRoad.x + 0.1, pRoad.y - height, 0.);
        len = max(len, -Box3(pObj, vec3(0.1), 0.1));

        pObj = vec3(pRoad.x - 0.1, pRoad.y - height + 0.5, pReflector.z);
        len = min(len, Box3(pObj, vec3(0.05, 0.5, 0.05), 0.01));
        res = MinDist(res, vec2(len, ROAD_UTILITY_ID));
    }

    float reflector = Box3(pReflector, vec3(0.05), 0.01);
    if (guardrailHeight >= 1. || wallHeight > 0.7)
    {
        res = MinDist(res, vec2(reflector, ROAD_REFLECTOR_ID));
    }

    // street lamp
    if (lampHeight > 0.)
    {
        vec3 pObj = vec3(pRoad.x - 0.7, pRoad.y, round(pRoad.z / DISTANCE_BETWEEN_LAMPS) * DISTANCE_BETWEEN_LAMPS - pRoad.z);
        float len = Box3(pObj, vec3(0.1, lampHeight, 0.1), 0.1);

        pObj = vec3(pRoad.x + 0.7, pRoad.y - lampHeight, pObj.z);
        pObj.xy *= Rotation(-0.2);
        len = min(len, Box3(pObj, vec3(1.8, 0.05, 0.05), 0.1));
        res = MinDist(res, vec2(len, ROAD_UTILITY_ID));

        pObj.x += 1.2;
        len = Box3(pObj, vec3(0.7, 0.1, 0.1), 0.1);
        res = MinDist(res, vec2(len, ROAD_LIGHT_ID));
    }

    if (wallHeight > 0.)
    {
        bool isTunnel = wallHeight >= 4.;
        float len = max(-pRoad.x+max((isTunnel?0.:1.)*pRoad.y-1.,0)*.5,pRoad.y-wallHeight);
        float d = min(
                len,
                max(len-.2,abs(mod(pRoad.z,10)-5)-.2)
            );
        if (isTunnel) {
            d = smin(d, wallHeight - pRoad.y, 0.4);
        }
        res = MinDist(res, vec2(d, ROAD_WALL_ID));
    }

    return res;
}

vec2 terrainShape(vec3 p, vec4 splineUV)
{
    float heightToDistanceFactor = 0.75;
    // First, compute the smooth terrain
    float terrainHeight = smoothTerrainHeight(p.xz);
    float relativeHeight = p.y - terrainHeight;

    // If the distance is sufficiently large, stop there
    if (relativeHeight > 10.)
    {
        return vec2(heightToDistanceFactor * relativeHeight, GROUND_ID);
    }

    vec2 d = vec2(1e6, GROUND_ID);

    // Compute the road presence
    float isRoad = 1.0 - smoothstep(roadWidthInMeters.x, roadWidthInMeters.y, abs(splineUV.x));

    // If (even partly) on the terrain, add detail to the terrain
    if (isRoad < 1.0)
    {
        terrainHeight += terrainDetailHeight(p.xz);
    }

    // If (even partly) on the road, flatten road
    float roadHeight = terrainHeight;
    if (isRoad > 0.0)
    {
        // Get the point on the center line of the spline
        vec3 directionAndCurvature;
        vec2 positionOnSpline = GetPositionOnSpline(splineUV.yw, directionAndCurvature);

        // Get the terrain height at the center line
        roadHeight = smoothTerrainHeight(positionOnSpline);
        d = MinDist(d, roadSideItems(splineUV, p.y - roadHeight));

        roadHeight += roadBumpHeight(splineUV.x)+pow(valueNoise(mod(p.xz*40,100)),.01)*.1;
    }

    // Combine terrain height and road heigt
    float height = mix(terrainHeight, roadHeight, isRoad);

    relativeHeight = p.y - height;
    
    d = MinDist(d, vec2(heightToDistanceFactor * relativeHeight, GROUND_ID));

    return d;
}

const float halfTreeSpace = 5.;
const float maxTreeHeight = 20.;

float tree(vec3 globalP, vec3 localP, vec2 id, vec4 splineUV, float current_t) {
    float h1 = hash21(id);
    float h2 = hash11(h1);
    float terrainHeight = smoothTerrainHeight(id);

    float verticalClearance = globalP.y - terrainHeight - maxTreeHeight;
    if (verticalClearance > 0.)
    {
        // The conservative value to return is verticalClearance, but
        // doing so we run out of steps and have artifacts in the sky.
        // So instead we assume we're not going to hit any tree closer
        // than what the scene SDF is.
        return INF;
    }

    float d = halfTreeSpace;

    // Define if the area has trees
    float presence = 1.;//smoothstep(-0.7, 0.7, fBm(id / 500., 2, 0.5, 0.3));
    if (h1 >= presence)
    {
        // We'll have to try the next cell.
        return d;
    }

    // Opportunity for early out: there should be no tree part on the road.
    if (abs(splineUV.x) < roadWidthInMeters.x) return d;

    // Clear trees too close to the road.
    //
    // The splineUV is relative to the current position, but we have to
    // check the distance of the road from the position of the potential
    // tree.
    float treeClearance = roadWidthInMeters.y + halfTreeSpace;
    vec4 splineUVatTree = ToSplineLocalSpace(id, treeClearance);
    if (abs(splineUVatTree.x) < treeClearance) return d;

    float treeHeight = mix(5., maxTreeHeight, 1.-h1*h1);
    float treeWidth = treeHeight * mix(0.3, 0.5, h2*h2);

    localP.y -= terrainHeight + 0.5 * treeHeight;
    localP.xz += (vec2(h1, h2) - 0.5) * 1.5; // We cannot move the trees too much due to artifacts.

    d = min(d, Ellipsoid(localP, 0.5*vec3(treeWidth, treeHeight, treeWidth)));

    float leaves = 1. - smoothstep(50., 200., current_t);
    if (d < 2. && leaves > 0.)
    {
        d += leaves * fBm(5. * vec2(2.*atan(localP.z, localP.x), localP.y) + id, 2, 0.5, 0.5) * 0.5;
    }

    return d;
}

vec2 treesShape(vec3 p, vec4 splineUV, float current_t)
{
    // iq - repeated_ONLY_SYMMETRIC_SDFS (https://iquilezles.org/articles/sdfrepetition/)
    //vec3 lim = vec3(1e8,0,1e8);
    vec2 id = round(p.xz / (halfTreeSpace * 2.)) * (halfTreeSpace * 2.);
    vec3 localP = p;
    localP.xz -= id;
    return vec2(tree(p, localP, id, splineUV, current_t), GROUND_ID);
}
