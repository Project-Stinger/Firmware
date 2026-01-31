#pragma once
#if HW_VERSION == 2

#ifdef USE_ML_LOG
#ifdef ENABLE_DEBUG_GYRO
#error "USE_ML_LOG and ENABLE_DEBUG_GYRO both use Serial input — enable only one"
#endif

#include "utils/typedefs.h"

// 17 bytes/sample @ 100 Hz ~= 1.7 kB/s.
// Log format is consumed by `python/ml_log_convert.py`.
struct __attribute__((packed)) MlSample {
	u32 timestamp_ms;
	i16 ax, ay, az;
	i16 gx, gy, gz;
	u8 trigger;
};

/// @brief Called once from setup() on core 0.
void mlLogInit();

/// @brief Called from loop1() (core 1). Captures 100Hz samples into a RAM buffer.
void mlLogLoop();

/// @brief Called from loop() (core 0). Flushes RAM buffer to flash when safe.
void mlLogSlowLoop();

#endif // USE_ML_LOG
#endif // HW_VERSION == 2

