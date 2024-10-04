void GenerateSpline(float maxCurvature, float segmentLength, float seed)
{
    vec2 direction = vec2(hash11(seed), hash11(seed + 1.0)) * 2.0 - 1.0;
    direction = normalize(direction);
    vec2 point = vec2(0.);
    for(int i = 0; i < SPLINE_SIZE; i++) {
        if (i % 2 == 0) {
            spline[i] = point + 0.5*segmentLength*direction;
            continue;
        }
        float ha = hash11(seed + float(i) * 3.0);
        point += direction * segmentLength;
        float angle = mix(-maxCurvature, maxCurvature, ha);
        direction *= Rotation(angle);
        spline[i] = point;
    }
}

float verticalBump()
{
    return valueNoise2(6.*time).x;
}

const int SHOT_SIDE_FRONT = 0;
void sideShotFront()
{
    vec2 p = vec2(0.95, 0.5);
    p.x += mix(-0.5, 1., valueNoise2(0.5*time).y);
    p.x += mix(-0.01, 0.01, valueNoise2(600.*time).y);
    p.y += 0.05 * verticalBump();
    camPos = vec3(p, 1.5);
    camTa = vec3(p.x, p.y + 0.1, 0.);
    camProjectionRatio = 1.2;
}

void sideShotRear()
{
    vec2 p = vec2(-1., 0.5);
    p.x += mix(-0.2, 0.2, valueNoise2(0.5*time).y);
    p.x += mix(-0.01, 0.01, valueNoise2(600.*time).y);
    p.y += 0.05 * verticalBump();
    camPos = vec3(p, 1.5);
    camTa = vec3(p.x, p.y + 0.1, 0.);
    camProjectionRatio = 1.2;
}

void fpsDashboardShot()
{
    camPos = vec3(0.1, 1.12, 0.);
    camPos.z += mix(-0.02, 0.02, valueNoise2(0.1*time).x);
    camPos.y += 0.01 * valueNoise2(5.*time).y;
    camTa = vec3(5, 1, 0.);
    camProjectionRatio = 0.7;
}

// t should go from 0 to 1 in roughly 4 seconds
void dashBoardUnderTheShoulderShot(float t)
{
    float bump = 0.02 * verticalBump();
    camPos = vec3(-0.2 - 0.6 * t, 0.88 + 0.35*t + bump, 0.42);
    camTa = vec3(0.5, 1. + 0.2 * t + bump, 0.25);
    camProjectionRatio = 1.5;
}

void frontWheelCloseUpShot()
{
    camPos = vec3(-0.1, 0.5, 0.5);
    camTa = vec3(0.9, 0.35, 0.2);
    vec2 vibration = 0.005 * valueNoise2(40.*time);
    float bump = 0.02 * verticalBump();
    vibration.x += bump;
    camPos.yz += vibration;
    vibration.x += mix(-0.01, 0.01, valueNoise2(600.*time).y);
    camTa.yz += vibration;
    camProjectionRatio = 1.6;
    camShowDriver = 0.;
}

void overTheHeadShot()
{
    camPos = vec3(-1.8, 1.7, 0.);
    camTa = vec3(0.05, 1.45, 0.);
    float bump = 0.01 * verticalBump();
    camPos.y += bump;
    camTa.y += bump;
    camProjectionRatio = 3.;
}

 // Also useful for debugging & visualizing the spline
 void topDownView()
{
    camPos = vec3(0., 300., 0.);
    camTa = vec3(0.);
    camProjectionRatio = 4.;
}

void viewFromBehind(float t_in_shot)
{
    camTa = vec3(1., 1., 0.);
    camPos = vec3(-2. - 2.5*t_in_shot, 1.5, sin(t_in_shot));
    camProjectionRatio = 1.;
}

void faceView(float t_in_shot)
{
    camTa = vec3(0., 1.5, 0.);
    camPos = vec3(1. + 2.5*t_in_shot, 1.5, -2);
    camProjectionRatio = 1.;
}

void openingShot(float t_in_shot) {
    camTa = vec3(10., 12. - mix(0., 10., min(1.,t_in_shot/6.)), 1.);
    camPos = vec3(5, 7. - min(t_in_shot, 6.), 1.); // vec3(1. + 3.*time, 1.5, 1);
    camProjectionRatio = 1.;
}

void introShotFromFar(float t_in_shot)
{
    camTa = vec3(0., 1., 0.);
    camPos = vec3(60. - 3.*t_in_shot, 2., -7+t_in_shot);
    camProjectionRatio = 1.;
}

bool get_shot(inout float time, float duration) {
    if (time < duration) {
        return true;
    }
    time -= duration;
    return false;
}

void selectShot() {
    float time = iTime;
    // We currently generate a new spline every 20 seconds.
    // Bug: this can change during a shot.
    GenerateSpline(1.8/*curvature*/, 80./*scale*/, 2.+floor(iTime / 20)/*seed*/);

    wallHeight = -1.;
    guardrailHeight = 0.;
    if (time < 80.) {
        wallHeight = -1.;
    } else if (time < 100.) {
        wallHeight = 3.;
    } else if (time < 120.) {
        wallHeight = 4.;
    } else {
        wallHeight = -1.;
        guardrailHeight = 1.;
    }

    camProjectionRatio = 1.;
    camFishEye = 0.1;
    camMotoSpace = 1.;
    camShowDriver = 1.;
    camFoV = atan(1. / camProjectionRatio);

    if (get_shot(time, 10.5)) {
        openingShot(time);
    } else if (get_shot(time, 9.5)) {
        introShotFromFar(time);
    } else if (get_shot(time, 6.)) {
        sideShotRear();
    } else if (get_shot(time, 5.)) {
        sideShotFront();
    } else if (get_shot(time, 8.)) {
        frontWheelCloseUpShot();
    } else if (get_shot(time, 8.)) {
        overTheHeadShot();
    } else if (get_shot(time, 8.)) {
        fpsDashboardShot();
    } else if (get_shot(time, 8.)) {
        dashBoardUnderTheShoulderShot(time);
    } else if (get_shot(time, 8.)) {
        viewFromBehind(time);
    } else if (get_shot(time, 4.)) {
        sideShotRear();
    } else if (get_shot(time, 8.)) {
        faceView(time);
    } else if (get_shot(time, 4.)) {
        sideShotFront();
    } else if (get_shot(time, 8.)) {
        overTheHeadShot();
    } else if (get_shot(time, 8.)) {
        frontWheelCloseUpShot();
    } else if (get_shot(time, 8.)) {
        fpsDashboardShot();
    } else if (get_shot(time, 8.)) {
        dashBoardUnderTheShoulderShot(time);
    } else if (get_shot(time, 8.)) {
        viewFromBehind(time);
    } else {
        overTheHeadShot();
    }
}
