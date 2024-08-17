const int SPLINE_SIZE = 13;

float splineSegmentDistances[SPLINE_SIZE / 2];
vec4 splineAABB;

vec2 spline[SPLINE_SIZE] = vec2[](
    vec2(-5.0, -5.0),
    vec2(-3.0, -2.0),

    vec2(-5.0,  0.0),
    vec2(-7.0,  2.0),

    vec2(-5.0,  5.0),
    vec2(-3.0,  7.0),

    vec2( 0.0,  5.0),
    vec2( 1.0,  4.0),

    vec2( 2.0,  5.0),
    vec2( 4.0,  6.0),

    vec2( 4.0,  3.0),
    vec2( 4.0, -2.0),

    vec2( 8.0, 3.0)
);

// Piece of code copy-pasted from:
// https://www.shadertoy.com/view/NltBRB
// Credit: MMatteini

void ComputeBezierSegmentsLengthAndAABB()
{
    vec2 mi = vec2(1e6);
    vec2 ma = vec2(-1e6);

    float splineLength = 0.0;
    for (int i = 0; i < SPLINE_SIZE - 1; i += 2)
    {
        vec2 A = spline[i + 0];
        vec2 B = spline[i + 1];
        vec2 C = spline[i + 2];
        float segmentLength = BezierCurveLengthAt(A, B, C, 1.0);
        splineSegmentDistances[i / 2] = splineLength;
        splineLength += segmentLength;

        vec4 aabb = BezierAABB(A, B, C);
        mi = min(mi, aabb.xy);
        ma = max(ma, aabb.zw);
    }
    splineAABB = vec4(mi, ma);
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
    vec4 splineUV = vec4(1e6, 0, 0, 0);

    float d = DistanceFromAABB(p, splineAABB);
    if (d > splineWidth)
    {
        return splineUV;
    }

    // For each bezier segment
    for (int i = ZERO; i < SPLINE_SIZE - 1; i += 2)
    {
        vec2 A = spline[i + 0];
        vec2 B = spline[i + 1];
        vec2 C = spline[i + 2];

        d = DistanceFromAABB(p, BezierAABB(A, B, C));
        if (d < splineWidth)
        {
            // This is to prevent 3 colinear points, but there should be better solution to it.
            B = mix(B + vec2(1e-4), B, abs(sign(B * 2.0 - A - C))); 
            // Current bezier curve SDF
            vec2 bezierSDF = BezierSDF(A, B, C, p);

            if (abs(bezierSDF.x) < abs(splineUV.x))
            {
                float lengthInSegment = BezierCurveLengthAt(A, B, C, clamp(bezierSDF.y, 0., 1.));
                float lengthInSpline = splineSegmentDistances[i / 2] + lengthInSegment;
                splineUV = vec4(
                    bezierSDF.x,
                    clamp(bezierSDF.y, 0., 1.),
                    lengthInSpline,
                    float(i));
            }
        }
    }

    return splineUV;
}

void GenerateSpline(float maxCurvature, float segmentLength, float seed)
{
    vec2 direction = vec2(hash(seed), hash(seed + 1.0)) * 2.0 - 1.0;
    direction = normalize(direction);
    vec2 point = vec2(0.);
    for(int i = 0; i < SPLINE_SIZE; i++) {
        if (i % 2 == 0) {
            spline[i] = point + 1.*direction;
            continue;
        }
        float ha = hash(seed + float(i) * 3.0);
        point += direction * segmentLength;
        float angle = mix(-maxCurvature, maxCurvature, ha);
        direction *= Rotation(angle);
        spline[i] = point;
    }
}

// Generated by ChatGPT.
vec2 GetPositionOnCurve(float t)
{
    // Total spline length
    float totalLength = splineSegmentDistances[SPLINE_SIZE / 2 - 1];
    
    // Desired length along the spline for the given t
    float targetLength = t * totalLength;
    
    // Find the segment corresponding to this target length
    int segmentIndex = 0;
    for (int i = ZERO; i < SPLINE_SIZE / 2 - 1; ++i) {
        if (splineSegmentDistances[i] <= targetLength && splineSegmentDistances[i + 1] > targetLength) {
            segmentIndex = i;
            break;
        }
    }
    
    // Calculate how far along the segment we are
    float segmentStartLength = splineSegmentDistances[segmentIndex];
    float segmentEndLength = splineSegmentDistances[segmentIndex + 1];
    float segmentLength = segmentEndLength - segmentStartLength;
    float segmentT = (targetLength - segmentStartLength) / segmentLength;
    
    // Get the control points of the segment
    vec2 A = spline[segmentIndex * 2];
    vec2 B = spline[segmentIndex * 2 + 1];
    vec2 C = spline[segmentIndex * 2 + 2];

    return Bezier(A, B, C, segmentT);
}

//
// If you have a splineUV, call:
// GetPositionOnSpline(splineUV.yw)
//
// This function does the same as the ChatGPT one, but without the for loop.
//
vec2 GetPositionOnSpline(vec2 spline_t_and_index)
{
    float t = spline_t_and_index.x;
    int i = int(spline_t_and_index.y);
    vec2 A = spline[i + 0];
    vec2 B = spline[i + 1];
    vec2 C = spline[i + 2];

    return Bezier(A, B, C, t);
}

// x: actual width
// y: width + transition
// z: max width
vec3 roadWidthInMeters = vec3(4.0, 8.0, 8.0);

vec3 roadPattern(vec2 uv, float width, vec2 params)
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

    float sideLine1 = Box(sideTileUV - vec2(0.5 * sideLineParams.y, width), vec2(0.5 * sideLineParams.y, 0.10), 0.03);
    float sideLine2 = Box(sideTileUV - vec2(0.5 * sideLineParams.y, -width), vec2(0.5 * sideLineParams.y, 0.10), 0.03);

    float separationLine1 = Box(separationTileUV - vec2(0.5 * separationLineParams.y, 0.0), vec2(0.5 * separationLineParams.y, 0.10), 0.01);

    float pattern = min(min(sideLine1, sideLine2), separationLine1);

    float duv = length(fwidth(uv));
    vec3 test = /**/ vec3(1.0 - smoothstep(-duv, 0.0, pattern)); /*/ debugDistance(iResolution, 10.*pattern) /**/;
    return mix(test, vec3(fract(uv), separationTileUV.x), 0.0);
}

const float terrain_fBm_weight_param = 0.6;
const float terrain_fBm_frequency_param = 0.5;

float smoothTerrainHeight(vec2 p)
{
    float hillHeightInMeters = 200.;
    float hillLengthInMeters = 2000.;

    return 0.5 * hillHeightInMeters * fBm(p * 2. / hillLengthInMeters, 3, terrain_fBm_weight_param, terrain_fBm_frequency_param);
}

float terrainDetailHeight(vec2 p)
{
    float detailHeightInMeters = 1.;
    float detailLengthInMeters = 100.;

    return 0.5 * detailHeightInMeters * fBm(p * 2. / detailLengthInMeters, 1, terrain_fBm_weight_param, terrain_fBm_frequency_param);
}

vec2 terrainShape(vec3 p, vec4 splineUV)
{
    float heightToDistanceFactor = 0.75;

    // First, compute the smooth terrain
    float terrainHeight = smoothTerrainHeight(p.xz);
    float relativeHeight = p.y - terrainHeight;

    // If the distance is sufficiently large, stop there
    if (relativeHeight > 4.)
    {
        return vec2(heightToDistanceFactor * relativeHeight, GROUND_ID);
    }

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
        vec2 positionOnSpline = GetPositionOnSpline(splineUV.yw);

        // Get the terrain height at the center line
        roadHeight = smoothTerrainHeight(positionOnSpline);
        float x = clamp(abs(splineUV.x / roadWidthInMeters.x), 0., 1.);
        roadHeight += 0.2 * (1. - x * x * x);
    }

    // Combine terrain height and road heigt
    float height = mix(terrainHeight, roadHeight, isRoad);

    relativeHeight = p.y - height;
    
    vec2 d = vec2(heightToDistanceFactor * relativeHeight, GROUND_ID);

    return d;
}
