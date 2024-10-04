const int NO_ID = -1;

const int MOTO_ID = 0;
const int MOTO_WHEEL_ID = 1;
const int MOTO_MOTOR_ID = 2;
const int MOTO_EXHAUST_ID = 3;
const int MOTO_BREAK_LIGHT_ID = 4;
const int MOTO_HEAD_LIGHT_ID = 5;
const int MOTO_DRIVER_ID = 6;
const int MOTO_DRIVER_HELMET_ID = 7;

const int CITY_ID = 8;
const int TREE_ID = 9;

const int GROUND_ID = 10;
const int ROAD_UTILITY_ID = 11;
const int ROAD_WALL_ID = 12;
const int ROAD_REFLECTOR_ID = 13;
const int ROAD_LIGHT_ID = 14;

#ifdef DEBUG
const int DEBUG_ID = 9999;
#endif

bool IsMoto(int mid)
{
    // Assumes mid != NO_ID
    return mid <= MOTO_DRIVER_HELMET_ID;
}

bool IsRoad(int mid)
{
    return mid >= GROUND_ID;
}
