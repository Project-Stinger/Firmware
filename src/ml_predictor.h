/**
 * ML Predictor Module
 * Pre-fire prediction system using Random Forest classifier
 *
 * Predicts trigger pulls 100-500ms in advance based on IMU motion patterns
 */

#pragma once

#include "global.h"

#if HW_VERSION == 2

// Forward declarations
class MLPredictor {
public:
    // Initialize predictor
    static void init();

    // Update with new IMU sample (called at 1600 Hz from main loop)
    static void updateIMU(int16_t accel_x, int16_t accel_y, int16_t accel_z,
                          int16_t gyro_x, int16_t gyro_y, int16_t gyro_z);

    // Run inference and return prediction (called every 10-20ms)
    static bool predict();

    // Get pre-fire state (for operationSm to check)
    static bool isPreSpinning();

    // State access for external code (operationSm needs to check this)
    static bool shouldPreSpin();

    // Update pre-spin timeout logic
    static void updateTimeout();

    // Reset predictor state
    static void reset();

    // Configuration
    static void setTimeout(uint16_t timeout_ms);
    static uint16_t getTimeout();
    static void setConsecutiveRequired(uint8_t count);
    static uint8_t getConsecutiveRequired();

    // UI Feedback - for confidence bar display
    static uint8_t getConsecutiveCount();  // Current consecutive predictions
    static float getLastProbability();     // Last raw prediction probability (0.0-1.0)

#ifdef ENABLE_ML_LOGGER
    // Data logging for training
    static void logSample(int16_t accel_x, int16_t accel_y, int16_t accel_z,
                          int16_t gyro_x, int16_t gyro_y, int16_t gyro_z,
                          uint8_t trigger_state);
#endif

private:
    // Feature extraction
    static void extractFeatures(float* features);
    static void updateRunningStats();
};

#endif // HW_VERSION == 2
