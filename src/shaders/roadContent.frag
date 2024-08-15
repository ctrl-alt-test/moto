const int SPLINE_SIZE = 13;

float maxRoadWidth = 1.0;

float splineSegmentDistances[SPLINE_SIZE / 2];

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

void ComputeBezierSegmentsLength()
{
    float splineLength = 0.0;
    for (int i = 0; i < SPLINE_SIZE - 1; i += 2)
    {
        vec2 A = spline[i + 0];
        vec2 B = spline[i + 1];
        vec2 C = spline[i + 2];
        float segmentLength = BezierCurveLengthAt(A, B, C, 1.0);
        splineSegmentDistances[i / 2] = splineLength;
        splineLength += segmentLength;
    }
}

// Decompose a give location into its distance from the closest point on the spline. 
// and the length of the spline up to that point.
// Returns a vector where:
// X = signed distance from the spline
// Y = spline parameter t at the closest point [0; 1]
// Z = spline length at the closest point
vec3 ToSplineLocalSpace(vec2 p, float splineWidth)
{
    vec3 splineUV = vec3(3e10, 0, 0);

    // For each bezier segment
    for (int i = 0; i < SPLINE_SIZE - 1; i += 2)
    {
        vec2 A = spline[i + 0];
        vec2 B = spline[i + 1];
        vec2 C = spline[i + 2];
        
        if (DistanceFromBezierAABB(p, A, B, C) - splineWidth <= 0.0)
        {
            // This is to prevent 3 colinear points, but there should be better solution to it.
            B = mix(B + vec2(1e-4), B, abs(sign(B * 2.0 - A - C))); 
            // Current bezier curve SDF
            vec2 bezierSDF = BezierSDF(A, B, C, p);

            if (abs(bezierSDF.x) < abs(splineUV.x))
            {
                float lengthInSegment = BezierCurveLengthAt(A, B, C, clamp(bezierSDF.y, 0., 1.));
                float lengthInSpline = splineSegmentDistances[i / 2] + lengthInSegment;
                splineUV = vec3(bezierSDF.x, (clamp(bezierSDF.y, 0., 1.) + 0.5 * float(i)) / float(SPLINE_SIZE), lengthInSpline);
            }
        }
    }

    return splineUV;
}

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
