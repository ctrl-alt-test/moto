vec3 motoPos;
vec3 motoDir;
vec3 headLightOffsetFromMotoRoot = vec3(0.53, 0.98, 0.0);
vec3 breakLightOffsetFromMotoRoot = vec3(-1.14, 0.55, 0.0);
vec3 dirHeadLight = normalize(vec3(1.0, -0.15, 0.0));
vec3 dirBreakLight = normalize(vec3(-1.0, -0.5, 0.0));
float motoYaw;
float motoPitch;
float motoRoll;

//
// Moto position functions
//

// Moto local space:
// - The moto is aligned with the X axis.
// - The anchor point is on the ground, roughly at the center (engine).
//        Y
//        ^
//        |
//        |_o<)
//    _--<| /\_  ==>
//   (o)\ |/ (o)
// -------+--------> X
//

vec3 motoToWorld(vec3 v, bool isPos, float time)
{
    v.xy *= Rotation(-motoPitch);
    v.yz *= Rotation(-motoRoll);
    v.xz *= Rotation(-motoYaw);

    if (isPos)
    {
        v += motoPos;
        v.z += 2. + 0.5*sin(time);
        // v.z -= 1.5*sin(time);
    }
    return v;
}

vec3 worldToMoto(vec3 v, bool isPos, float time)
{
    if (isPos)
    {
        v -= motoPos;
        v.z -= 2. + 0.5*sin(time);
    }
    v.xz *= Rotation(motoYaw);
    v.yz *= Rotation(motoRoll);
    v.xy *= Rotation(motoPitch);
    return v;
}

//
// Dashboard
//

float rect(vec2 uv, float x1, float y1, float x2, float y2) {
  return float(uv.x > x1 && uv.x < x2 && uv.y > y1 && uv.y < y2);
}

// vertical bars
vec3 meter3(vec2 uv, float value) {
    // 0.69, 0.87, 0.79
    // 0.29, 0.63, 0.43
    // 0.36, 0.16, 0.12

    float verticalLength = 0.04 + 0.15 * smoothstep(0.1, 0.4, uv.x);

    float r = Box2(uv, vec2(0.5, verticalLength), 0.01);
    // if (r > 0. || uv.y < -0.02) return vec3(0.);

    float lines = smoothstep(0.5, 0.7, fract(uv.x * 30.));
    lines *= smoothstep(0.1, 0.3, fract(uv.y/verticalLength*2.));

    vec3 baseCol =
        mix(vec3(0.7, 0.9, 0.8),
            vec3(0.8, 0., 0.), smoothstep(0.4, 0.41, uv.x));

    value *= 0.5;
    baseCol = mix(vec3(0.01), baseCol, 0.15+0.85*smoothstep(0., 0.001, value - uv.x));
    vec3 col = lines * baseCol;
    return smoothstep(0.001, 0., r) * float(uv.y > 0.) * col;
}

vec3 meter4(vec2 uv, float value) {
  float len = length(uv);
  float angle = atan(uv.y, uv.x);

  float lines =
    smoothstep(0.7, 1., mod(angle, 0.25)/0.25) *
    smoothstep(0., 0.01, abs(angle + 0.7) - 0.7) * // hide bottom-right
    smoothstep(0., 0.01, 0.1 - length(uv)) *
    smoothstep(0., 0.01, length(uv) - 0.06);

  value = (value * 1.5 - 1.) * PI;
  vec2 point = vec2(sin(value), cos(value)) * 0.07;
  float dummy;
  float line = smoothstep(0.004, 0.002, Segment3(uv.xyy, vec3(0), point.xyy, dummy));
  vec3 col = vec3(0.36, 0.16, 0.12) * lines;
  col += vec3(0.7) * line;
  return col;
}

float digit(int n, vec2 p)
{
    vec2 size = vec2(0.2, 0.35);
    const float thickness = 0.065;
    const float gap = 0.0125;
    const float slant = 0.15;
    const float roundOuterCorners = 0.5;
    const float roundInterCorners = 0.15;
    const float spacing = 0.67;

    bool A = (n != 1 && n != 4);
    bool i_B = (n != 5 && n != 6);
    bool i_C = (n != 2);
    bool i_D = (A && n != 7);
    bool i_E = (A && n % 2 == 0);
    bool i_F = (n != 1 && n != 2 && n != 3 && n != 7);
    bool i_G = (n > 1 && n != 7);

    p.x -= p.y * slant;
    float boundingBox = Box2(p, size, size.x * roundOuterCorners);
    float innerBox = -Box2(p, size - thickness, size.x * roundInterCorners);
    float d = INF;

    // Segment A
    if (A)
    {
        float sA = innerBox;
        sA = max(sA, gap + (p.x - p.y - size.x + size.y));
        sA = max(sA, gap - (p.x + p.y + size.x - size.y));
        d = min(d, sA);
    }

    // Segment B
    if (i_B)
    {
        float sB = innerBox;
        sB = max(sB, gap - (p.x - p.y - size.x + size.y));
        sB = max(sB, gap - (p.x + p.y + size.x - size.y));
        sB = max(sB, p.x - (p.y) - (size.x + thickness) / 2.);
        d = min(d, sB);
    }

    // Segment C
    if (i_C)
    {
        float sC = innerBox;
        sC = max(sC, gap - (p.x - p.y + size.x - size.y));
        sC = max(sC, gap - (p.x + p.y - size.x + size.y));
        sC = max(sC, p.x + (p.y) - (size.x + thickness) / 2.);
        d = min(d, sC);
    }

    // Segment D
    if (i_D)
    {
        float sD = innerBox;
        sD = max(sD, gap - (p.x - p.y + size.x - size.y));
        sD = max(sD, gap + (p.x + p.y - size.x + size.y));
        d = min(d, sD);
    }

    // Segment E
    if (i_E)
    {
        float sE = innerBox;
        sE = max(sE, gap + (p.x - p.y + size.x - size.y));
        sE = max(sE, gap + (p.x + p.y - size.x + size.y));
        sE = max(sE, -p.x + (p.y) - (size.x + thickness) / 2.);
        d = min(d, sE);
    }

    // Segment F
    if (i_F)
    {
        float sF = innerBox;
        sF = max(sF, gap + (p.x - p.y - size.x + size.y));
        sF = max(sF, gap + (p.x + p.y + size.x - size.y));
        sF = max(sF, -p.x - (p.y) - (size.x + thickness) / 2.);
        d = min(d, sF);
    }

    // Segment G
    if (i_G)
    {
        float sG = -(thickness - abs(p.y) * 2.);
        sG = max(sG, gap + (p.x - p.y + size.x - size.y));
        sG = max(sG, gap - (p.x - p.y - size.x + size.y));
        sG = max(sG, gap + (p.x + p.y + size.x - size.y));
        sG = max(sG, gap - (p.x + p.y - size.x + size.y));
        d = min(d, sG);
    }

    return max(d, boundingBox);
}

vec3 glowy(float d)
{
    float dd = fwidth(d);
    float brightness = smoothstep(-dd, +dd, d);
    vec3 segment = vec3(0.3, 0.5, 0.4);

    vec3 innerColor = mix(vec3(0.2), segment, 1. / exp(50. * max(0., -d)));
    vec3 outerColor = mix(vec3(0.), segment, 1. / exp(200. * max(0., d)));
    return mix(innerColor, outerColor, brightness);
}

vec3 motoDashboard(vec2 uv)
{
    int speed = 105 + int(sin(iTime*.5) * 10.);
    vec2 uvSpeed = uv * 3. - vec2(0.4, 1.95);

    float numbers =
        min(min(min(
        // gear
        digit(5, uv * 8. - vec2(0.7,2.4)),
        // speed
        (float(speed<100) + digit(speed/100, uvSpeed))),
        digit((speed/10)%10, uvSpeed - vec2(.5,0))),
        digit(speed%10, uvSpeed - vec2(1.,0)));

    return
        meter3(uv * 0.6 - vec2(0.09, 0.05), 0.7+0.3*sin(iTime*0.5)) +
        meter4(uv * .7 - vec2(0.6, 0.45), 0.4) +
        glowy(numbers);
}

//
// Moto and driver
//

material motoMaterial(float mid, vec3 p, vec3 N, float time)
{
    if (mid == MOTO_HEAD_LIGHT_ID)
    {
        float isLight = smoothstep(0.9, 0.95, N.x);
        vec3 luminance = isLight * vec3(1., 0.95, 0.9);

        float isDashboard = smoothstep(0.9, 0.95, -N.x + 0.4 * N.y - 0.07);
        if (isDashboard > 0.)
        {
            vec3 color = motoDashboard(p.zy * 5.5 + vec2(0.5, -5.));
            luminance = mix(vec3(0), color, isDashboard);
        }

        return material(MATERIAL_TYPE_EMISSIVE, luminance, 0.15);
    }
    if (mid == MOTO_BREAK_LIGHT_ID)
    {
        float isLight = smoothstep(0.9, 0.95, -N.x);
        vec2 lightUV = fract(68.*p.yz + vec2(0.6, 0.)) * 2. - 1.;
        float pattern = smoothstep(0.2, 1., sqrt(length(lightUV)));
        vec3 luminance = mix(vec3(1., 0.005, 0.02), vec3(0.02, 0., 0.), pattern);
        return material(MATERIAL_TYPE_EMISSIVE, isLight * luminance, 0.5);
    }
    if (mid == MOTO_EXHAUST_ID)
    {
        return material(MATERIAL_TYPE_METALLIC, vec3(1.), 0.2);
    }
    if (mid == MOTO_MOTOR_ID)
    {
        return material(MATERIAL_TYPE_DIELECTRIC, vec3(0.), 0.3);
    }
    if (mid == MOTO_WHEEL_ID)
    {
        return material(MATERIAL_TYPE_DIELECTRIC, vec3(0.008), 0.8);
    }

    if (mid == MOTO_DRIVER_ID)
    {
        return material(MATERIAL_TYPE_DIELECTRIC, vec3(0.02, 0.025, 0.04), 0.6);
    }
    if (mid == MOTO_DRIVER_HELMET_ID)
    {
        return material(MATERIAL_TYPE_DIELECTRIC, vec3(0.), 0.25);
    }

    return material(MATERIAL_TYPE_DIELECTRIC, vec3(0.), 0.15);
}

vec2 driverShape(vec3 p)
{
    p = worldToMoto(p, true, iTime);

    // Place roughly on the seat
    p -= vec3(-0.35, 0.78, 0.0);

    float d = length(p);
    if (d > 1.2)
        return vec2(d, MOTO_DRIVER_ID);

    vec3 simP = p;
    simP.z = abs(simP.z);

    float wind = fBm((p.xy + iTime) * 12., 1, 0.5, 0.5);

    // upper body
    if (true && d < 0.8)
    {
        vec3 pBody = simP;
        pBody.z -= 0.02;
        pBody.xy *= Rotation(3.1);
        pBody.yz *= Rotation(-0.1);
        d = smin(d, Capsule(pBody, 0.12, 0.12), 0.1);

        pBody.y += 0.2;
        pBody.xy *= Rotation(-0.6);
        d = smin(d, Capsule(pBody, 0.12, 0.11), 0.02);

        pBody.y += 0.2;
        pBody.xy *= Rotation(-0.3);
        pBody.yz *= Rotation(-0.2);
        d = smin(d, Capsule(pBody, 0.12, 0.12), 0.02);

        pBody.y += 0.1;
        pBody.yz *= Rotation(1.7);
        d = smin(d, Capsule(pBody, 0.12, 0.1), 0.015);
    }
    d += 0.005 * wind;
    
    // arms
    if (true)
    {
        vec3 pArm = simP;

        pArm -= vec3(0.23, 0.45, 0.18);
        pArm.yz *= Rotation(-0.6);
        pArm.xy *= Rotation(0.2);
        float arms = Capsule(pArm, 0.29, 0.06);
        d = smin(d, arms, 0.005);

        pArm.y += 0.32;
        pArm.xy *= Rotation(1.5);
        arms = Capsule(pArm, 0.28, 0.04);
        d = smin(d, arms, 0.005);
    }
    d += 0.01 * wind;

    // lower body
    if (true)
    {
        vec3 pLeg = simP;

        pLeg -= vec3(0.0, 0.0, 0.13);
        pLeg.xy *= Rotation(1.55);
        pLeg.yz *= Rotation(-0.45);
        float h2 = Capsule(pLeg, 0.35, 0.09);
        d = smin(d, h2, 0.01);

        pLeg.y += 0.4;
        pLeg.xy *= Rotation(-1.5);
        float legs = Capsule(pLeg, 0.4, 0.06);
        d = smin(d, legs, 0.01);

        pLeg.y += 0.45;
        pLeg.xy *= Rotation(1.75);
        pLeg.yz *= Rotation(0.25);
        float feet = Capsule(pLeg, 0.2, 0.04);
        d = smin(d, feet, 0.01);
    }
    d += 0.002 * wind;

    // head
    if (true)
    {
        vec3 pHead = p - vec3(0.39, 0.6, 0.0);
        float head = length(pHead) - 0.15;

        if (head < d)
        {
            return vec2(head, MOTO_DRIVER_HELMET_ID);
        }
    }

    return vec2(d, MOTO_DRIVER_ID);
}

vec2 wheelShape(vec3 p, float wheelRadius, float tireRadius, float innerRadius)
{
    vec2 d = vec2(1e6, MOTO_WHEEL_ID);
    float wheel = Torus(p.yzx, vec2(wheelRadius, tireRadius));

    if (wheel < 0.25)
    {
        p.z = abs(p.z);
        float h;
        float cyl = Segment3(p, vec3(0.0), vec3(0.0, 0.0, 1.0), h);
        wheel = -smin(-wheel, cyl - innerRadius, 0.01);

        /**/
        // Note: the following group of lines is 1 byte smaller written as
        wheel = min(wheel, -min(min(min(0.15 - cyl, cyl - 0.08), p.z - 0.04), -p.z + 0.05));
        /*/
        float breakDisc = cyl - 0.15;
        breakDisc = -min(-breakDisc, cyl - 0.08);
        breakDisc = -min(-breakDisc, -p.z + 0.05);
        breakDisc = -min(-breakDisc, p.z - 0.04);
        wheel = min(wheel, breakDisc);
        /**/
    }
    return vec2(wheel, MOTO_WHEEL_ID);
}

vec2 motoShape(vec3 p)
{
    p = worldToMoto(p, true, iTime);

    float boundingSphere = length(p);
    if (boundingSphere > 2.0)
        return vec2(boundingSphere - 1.5, MOTO_ID);

    vec2 d = vec2(1e6, MOTO_ID);

#ifdef DEBUG
    // Show moto coordinates:
    d = MinDist(d, vec2(Box3(p, vec3(1.00, 0.01, 0.01), 0.001), DEBUG_ID));
    d = MinDist(d, vec2(Box3(p, vec3(0.01, 1.00, 0.01), 0.001), DEBUG_ID));
    d = MinDist(d, vec2(Box3(p, vec3(0.01, 0.01, 1.00), 0.001), DEBUG_ID));
#endif

    float h;
    float cyl;

    float frontWheelTireRadius = 0.14/2.0;
    float frontWheelRadius = 0.33 - frontWheelTireRadius;
    float rearWheelTireRadius = 0.3/2.0;
    float rearWheelRadius = 0.32 - rearWheelTireRadius;
    vec3 frontWheelPos = vec3(0.9, frontWheelRadius + frontWheelTireRadius, 0.0);

    // Front wheel
    if (true)
    {
        d = MinDist(d, wheelShape(p - frontWheelPos, frontWheelRadius, frontWheelTireRadius, 0.22));
    }

    // Rear wheel and its break light
    if (true)
    {
        d = MinDist(d, wheelShape(p - vec3(-0.85, rearWheelRadius + rearWheelTireRadius, 0.0), rearWheelRadius, rearWheelTireRadius, 0.18));
    
        // Break light
        if (true)
        {
            vec3 pBreak = p - breakLightOffsetFromMotoRoot;
            float breakBlock = Box3(pBreak, vec3(0.02, 0.025, 0.1), 0.02);
            d = MinDist(d, vec2(breakBlock, MOTO_BREAK_LIGHT_ID));
        }
    }

    // Front wheel fork
    if (true)
    {
        float forkThickness = 0.025;
        vec3 pFork = p;
        vec3 pForkTop = vec3(-0.48, 0.66, 0.0);
        vec3 pForkAngle = pForkTop + vec3(-0.14, 0.04, 0.05);
        pFork.z = abs(pFork.z);
        pFork -= frontWheelPos + vec3(0.0, 0.0, frontWheelTireRadius + 2. * forkThickness);
        float fork = Segment3(pFork, pForkTop, vec3(0.0), h) - forkThickness;

        // Join between the fork and the handle
        fork = min(fork, Segment3(pFork, pForkTop, pForkAngle, h) - forkThickness * 0.7);

        // Handle
        float handle = Segment3(pFork, pForkAngle, pForkAngle + vec3(-0.08, -0.07, 0.3), h);
        fork = min(fork, handle - mix(0.035, 0.02, smoothstep(0.25, 0.4, h)));

        // Mirror
        vec3 pMirror = pFork - pForkAngle - vec3(0.0, 0.1, 0.15);
        pMirror.xz *= Rotation(0.2);
        pMirror.xy *= Rotation(-0.2);
        
        float mirror = pMirror.x - 0.02;
        pMirror.xz *= Rotation(0.25);

        mirror = -min(mirror, -Ellipsoid(pMirror, vec3(0.04, 0.05, 0.08)));
        fork = min(fork, mirror);

        d = MinDist(d, vec2(fork, MOTO_ID));
    }

    // Head light and junction to the body
    if (true)
    {
        vec3 pHead = p - headLightOffsetFromMotoRoot;
        float headBlock = Ellipsoid(pHead, vec3(0.15, 0.2, 0.15));
        
        if (headBlock < 0.2)
        {
            vec3 pHeadTopBottom = pHead;

            // Top and bottom cuts
            pHeadTopBottom.xy *= Rotation(-0.15);
            headBlock = -min(-headBlock, -Ellipsoid(pHeadTopBottom - vec3(-0.2, -0.05, 0.0), vec3(0.35, 0.16, 0.25)));

            // Left and right cuts
            headBlock = -min(-headBlock, -Ellipsoid(pHead - vec3(-0.2, -0.08, 0.0), vec3(0.35, 0.25, 0.13)));

            // Front cut
            headBlock = -min(-headBlock, -Ellipsoid(pHead - vec3(-0.1, -0.05, 0.0), vec3(0.2, 0.2, 0.3)));

            // Dashboard
            pHead.xy *= Rotation(-0.4);
            headBlock = -min(-headBlock, -Ellipsoid(pHead - vec3(0.1, 0.0, 0.0), vec3(0.2, 0.3, 0.4)));
        }

        d = MinDist(d, vec2(headBlock, MOTO_HEAD_LIGHT_ID));

        float joint = Box3(p - vec3(0.4, 0.82, 0.0), vec3(0.04, 0.1, 0.08), 0.02);
        d = MinDist(d, vec2(joint, MOTO_MOTOR_ID));
    }

    // Fuel tank
    if (true)
    {
        vec3 pTank = p - vec3(0.1, 0.74, 0.0);
        vec3 pTankR = pTank;
        pTankR.xy *= Rotation(0.45);
        pTankR.x += 0.05;
        float tank = Ellipsoid(pTankR, vec3(0.35, 0.2, 0.42));

        if (tank < 0.1)
        {
            // Sides cut
            float tankCut = Ellipsoid(pTankR + vec3(0.0, 0.13, 0.0), vec3(0.5, 0.35, 0.22));
            tank = -min(-tank, -tankCut);
            //tank = -smin(-tank, -tankCut, 0.025);

            // Bottom cut
            float tankCut2 = Ellipsoid(pTank - vec3(0.0, 0.3, 0.0), vec3(0.6, 0.35, 0.4));
            tank = -min(-tank, -tankCut2);
            //tank = -smin(-tank, -tankCut2, 0.01);
        }
        d = MinDist(d, vec2(tank, MOTO_ID));
    }

    // Motor
    if (true)
    {
        vec3 pMotor = p - vec3(-0.08, 0.44, 0.0);
        
        // Main block
        vec3 pMotorSkewd = pMotor;
        pMotorSkewd.x *= 1. - pMotorSkewd.y * 0.4;
        pMotorSkewd.x += pMotorSkewd.y * 0.1;
        float motorBlock = Box3(pMotorSkewd, vec3(0.44, 0.29, 0.11), 0.02);
        
        if (motorBlock < 0.5)
        {
            // Pistons
            vec3 pMotor1 = pMotor - vec3(0.27, 0.12, 0.0);
            vec3 pMotor2 = pMotor - vec3(0.00, 0.12, 0.0);
            pMotor1.xy *= Rotation(-0.35);
            pMotor2.xy *= Rotation(0.35);
            motorBlock = min(motorBlock, Box3(pMotor1, vec3(0.1, 0.12, 0.20), 0.04));
            motorBlock = min(motorBlock, Box3(pMotor2, vec3(0.1, 0.12, 0.20), 0.04));

            // Gear box on the side
            vec3 pGearBox = pMotor - vec3(-0.15, -0.12, -0.125);
            pGearBox.xy *= Rotation(-0.15);
            float gearBox = Segment3(pGearBox, vec3(0.2, 0.0, 0.0), vec3(-0.15, 0.0, 0.0), h);
            gearBox -= mix(0.08, 0.15, h);
            
            pGearBox.x += 0.13;
            float gearBoxCut = -pGearBox.z - 0.05;
            gearBoxCut = min(gearBoxCut, Box3(pGearBox, vec3(0.16, 0.08, 0.1), 0.04));
            gearBox = -min(-gearBox, -gearBoxCut);

            motorBlock = min(motorBlock, gearBox);

            // Pedals
            vec3 pPedals = pMotor - vec3(0.24, -0.13, 0.0);
            float pedals = Segment3(pPedals, vec3(0.0, 0.0, .4), vec3(0.0, 0.0, -.4), h) - 0.02;
            motorBlock = min(motorBlock, pedals);
        }
        d = MinDist(d, vec2(motorBlock, MOTO_MOTOR_ID));
    }

    // Exhaust pipes
    if (true)
    {
        vec3 pExhaust = p;
        pExhaust -= vec3(0.0, 0.0, rearWheelTireRadius + 0.05);
        float exhaust = Segment3(pExhaust, vec3(0.24, 0.25, 0.0), vec3(-0.7, 0.3, 0.05), h);

        if (exhaust < 0.6)
        {
            exhaust -= mix(0.04, 0.08, mix(h, smoothstep(0.5, 0.7, h), 0.5));
            exhaust = -min(-exhaust, p.x - 0.7 * p.y + 0.9);
            exhaust = min(exhaust, Segment3(pExhaust, vec3(0.24, 0.25, 0.0), vec3(0.32, 0.55, -0.02), h) - 0.04);
            exhaust = min(exhaust, Segment3(pExhaust, vec3(0.22, 0.32, -0.02), vec3(-0.4, 0.37, 0.02), h) - 0.04);
        }
        d = MinDist(d, vec2(exhaust, MOTO_ID));
    }

    // Seat
    if (true)
    {
        vec3 pSeat = p - vec3(-0.44, 0.44, 0.0);
        float seat = Ellipsoid(pSeat, vec3(0.8, 0.4, 0.2));
        float seatRearCut = length(p + vec3(1.05, -0.1, 0.0)) - 0.7;
        seat = max(seat, -seatRearCut);

        if (seat < 0.2)
        {
            vec3 pSaddle = pSeat - vec3(0.35, 0.57, 0.0);
            pSaddle.xy *= Rotation(0.4);
            float seatSaddleCut = Ellipsoid(pSaddle, vec3(0.5, 0.15, 0.6));
            seat = -min(-seat, seatSaddleCut);
            seat = -smin(-seat, seatSaddleCut, 0.02);

            vec3 pSeatBottom = pSeat + vec3(0.0, -0.55, 0.0);
            pSeatBottom.xy *= Rotation(0.5);
            float seatBottomCut = Ellipsoid(pSeatBottom, vec3(0.8, 0.4, 0.4));
            seat = -min(-seat, -seatBottomCut);
        }
        d = MinDist(d, vec2(seat, MOTO_ID));
    }

    return d;
}
