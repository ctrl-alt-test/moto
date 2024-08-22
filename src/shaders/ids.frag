const float NO_ID = -1.;
const float GROUND_ID = 0.;
const float MOTO_ID = 1.;
const float MOTO_HEAD_LIGHT_ID = 2.;
const float MOTO_BREAK_LIGHT_ID = 3.;
const float MOTO_WHEEL_ID = 4.;
const float MOTO_MOTOR_ID = 5.;
const float MOTO_EXHAUST_ID = 6.;
const float MOTO_DRIVER_ID = 7.;
const float MOTO_DRIVER_HELMET_ID = 8.;
const float CITY_ID = 9.;
const float ROAD_REFLECTOR_ID = 10.;

bool IsMoto(float mid)
{
    return mid >= MOTO_ID && mid <= MOTO_DRIVER_HELMET_ID;
}
