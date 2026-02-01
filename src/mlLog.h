#pragma once
#if HW_VERSION == 2

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

/// @brief Called once from setup() on core 0. Mounts filesystem only.
void mlLogInit();

	/// @brief Called from menu to start recording.
	void mlLogStartRecording();

	/// @brief Called from menu to stop recording (flushes remaining buffered samples when safe).
	void mlLogStopRecording();

	/// @brief Called from loop1() (core 1). Captures 100Hz samples into a RAM buffer.
	void mlLogLoop();

/// @brief Called from loop() (core 0). Flushes RAM buffer to flash when safe.
void mlLogSlowLoop();

/// @brief Returns true if ML logging is actively recording.
bool mlLogIsActive();

/// @brief Returns flash usage as 0–100 percentage.
u8 mlLogFlashPercent();

#endif // HW_VERSION == 2
