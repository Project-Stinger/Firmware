#include "utils/fixedPointInt.h"
#include "utils/quaternion.h"
#include <Arduino.h>
#if HW_VERSION == 2

constexpr f32 RAW_TO_RAD_PER_SEC = PI * 4000 / 65536 / 180; // 2000deg per second, but raw is only +/-.5
constexpr fix32 RAW_TO_M_PER_SEC2 = (9.81 * 32 + 0.5) / 65536; // +/-16g (0.5 for rounding)

extern fix32 roll, pitch, yaw; // Euler angles of the blaster
extern PT1 accelDataFiltered[3]; // PT1 filters for the accelerometer data
extern PT2 vAccel, rAccel, fAccel;
extern PT2 vAccelSlow;
extern fix32 upVel;

extern bool freeFallDetectionEnabled;
extern bool freeFallDetected;

/**
 * @brief initialize the IMU
 * @details setting start values for quaternion, attitude angles and mag filter rollover
 */
void imuInit();
/**
 * @brief update the attitude of the blaster
 * @details 1. feeds gyro data into the attitude quaternion, 2. filters and feeds accelerometer values into the quaternion to prevent drift, 3. updates roll, pitch and yaw values, as well as combined heading, altitude and vVel via the filtered data
 */
void updateAtti1();
void updateAtti2();
void freeFallDetection();
#endif