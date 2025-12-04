/**
 * ML Predictor Implementation
 * Optimized for real-time inference on RP2040
 */

#include "ml_predictor.h"

#if HW_VERSION == 2

#include "model.h"
#include <cfloat>

// Configuration
static const int FEATURE_WINDOW_SIZE = 50;
static const int ML_INFERENCE_INTERVAL_MS = 6;  // ~167Hz inference (matches training stride of 10 samples at 1600Hz)
static const uint16_t DEFAULT_TIMEOUT_MS = 500;
static const uint8_t DEFAULT_CONSECUTIVE_REQUIRED = 20;  // 120ms latency (20 * 6ms)

// State
static bool preFireActive = false;
static uint16_t timeout_ms = DEFAULT_TIMEOUT_MS;
static uint8_t consecutiveRequired = DEFAULT_CONSECUTIVE_REQUIRED;
static uint8_t consecutiveCount = 0;
static float lastProbability = 0.0f;
static elapsedMillis timeoutTimer = 0;
static elapsedMillis inferenceTimer = 0;

// Circular buffers
static float accel_x_buffer[FEATURE_WINDOW_SIZE] = {0};
static float accel_y_buffer[FEATURE_WINDOW_SIZE] = {0};
static float accel_z_buffer[FEATURE_WINDOW_SIZE] = {0};
static float gyro_x_buffer[FEATURE_WINDOW_SIZE] = {0};
static float gyro_y_buffer[FEATURE_WINDOW_SIZE] = {0};
static float gyro_z_buffer[FEATURE_WINDOW_SIZE] = {0};
static int buffer_index = 0;
static bool buffer_filled = false;

// Running statistics for incremental feature calculation
struct RunningStats {
    float sum;
    float sumSq;
    float min;
    float max;
    int samples_seen;
    bool needs_recalc;
};

static RunningStats accel_x_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
static RunningStats accel_y_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
static RunningStats accel_z_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
static RunningStats gyro_x_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
static RunningStats gyro_y_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
static RunningStats gyro_z_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};

#ifdef ENABLE_ML_LOGGER
// Data logging state
static elapsedMicros lastLogTime = 0;
static const uint32_t LOG_INTERVAL_US = 625; // 1600 Hz = 625us period
static bool loggingEnabled = false;
static uint32_t samplesLogged = 0;
#endif

// Fast square root (Quake algorithm)
inline float fast_sqrt(float x) {
    if (x <= 0.0f) return 0.0f;
    union {
        float f;
        uint32_t i;
    } conv = {.f = x};
    conv.i = 0x5f3759df - (conv.i >> 1);
    float y = conv.f;
    y = y * (1.5f - (x * 0.5f * y * y));
    return x * y;
}

// Update running statistics incrementally
inline void update_stats(RunningStats *stats, float *buffer, float new_val, float old_val) {
    // Warmup phase
    if (stats->samples_seen < FEATURE_WINDOW_SIZE) {
        stats->samples_seen++;
        stats->sum += new_val;
        stats->sumSq += new_val * new_val;
        if (new_val < stats->min) stats->min = new_val;
        if (new_val > stats->max) stats->max = new_val;
        return;
    }

    // Sliding window update
    stats->sum = stats->sum - old_val + new_val;
    stats->sumSq = stats->sumSq - (old_val * old_val) + (new_val * new_val);

    // Check if min/max needs recalculation
    const float epsilon = 0.001f;
    if (fabsf(old_val - stats->min) < epsilon || fabsf(old_val - stats->max) < epsilon) {
        stats->needs_recalc = true;
    }

    // Lazy min/max recalculation
    if (stats->needs_recalc) {
        stats->min = buffer[0];
        stats->max = buffer[0];
        for (int i = 1; i < FEATURE_WINDOW_SIZE; i++) {
            if (buffer[i] < stats->min) stats->min = buffer[i];
            if (buffer[i] > stats->max) stats->max = buffer[i];
        }
        stats->needs_recalc = false;
    } else {
        if (new_val < stats->min) stats->min = new_val;
        if (new_val > stats->max) stats->max = new_val;
    }
}

// Public API Implementation

void MLPredictor::init() {
    // Reset all state
    reset();
    Serial.println("ML: Predictor initialized");
}

void MLPredictor::updateIMU(int16_t accel_x, int16_t accel_y, int16_t accel_z,
                             int16_t gyro_x, int16_t gyro_y, int16_t gyro_z) {
    // Convert to float
    float ax = (float)accel_x;
    float ay = (float)accel_y;
    float az = (float)accel_z;
    float gx = (float)gyro_x;
    float gy = (float)gyro_y;
    float gz = (float)gyro_z;

    // Get old values from buffer
    float old_ax = accel_x_buffer[buffer_index];
    float old_ay = accel_y_buffer[buffer_index];
    float old_az = accel_z_buffer[buffer_index];
    float old_gx = gyro_x_buffer[buffer_index];
    float old_gy = gyro_y_buffer[buffer_index];
    float old_gz = gyro_z_buffer[buffer_index];

    // Update buffers
    accel_x_buffer[buffer_index] = ax;
    accel_y_buffer[buffer_index] = ay;
    accel_z_buffer[buffer_index] = az;
    gyro_x_buffer[buffer_index] = gx;
    gyro_y_buffer[buffer_index] = gy;
    gyro_z_buffer[buffer_index] = gz;

    // Update statistics
    update_stats(&accel_x_stats, accel_x_buffer, ax, old_ax);
    update_stats(&accel_y_stats, accel_y_buffer, ay, old_ay);
    update_stats(&accel_z_stats, accel_z_buffer, az, old_az);
    update_stats(&gyro_x_stats, gyro_x_buffer, gx, old_gx);
    update_stats(&gyro_y_stats, gyro_y_buffer, gy, old_gy);
    update_stats(&gyro_z_stats, gyro_z_buffer, gz, old_gz);

    // Advance buffer index
    buffer_index = (buffer_index + 1) % FEATURE_WINDOW_SIZE;
    if (buffer_index == 0 && !buffer_filled) {
        buffer_filled = true;
        Serial.println("ML: Buffer filled, predictions starting");
    }
}

void MLPredictor::extractFeatures(float* features) {
    // Only extract if buffer is filled
    if (!buffer_filled) {
        for (int i = 0; i < 42; i++) features[i] = 0.0f;
        return;
    }

    int idx = 0;

    // Helper to extract stats
    auto extract_stats = [&](RunningStats *stats) {
        int n = stats->samples_seen;
        if (n == 0) {
            features[idx++] = 0.0f; // mean
            features[idx++] = 0.0f; // std
            features[idx++] = 0.0f; // min
            features[idx++] = 0.0f; // max
            return;
        }

        float mean = stats->sum / n;
        float variance = (stats->sumSq / n) - (mean * mean);
        float std = (variance > 0.0f) ? fast_sqrt(variance) : 0.0f;

        features[idx++] = mean;
        features[idx++] = std;
        features[idx++] = stats->min;
        features[idx++] = stats->max;
    };

    // Extract 24 basic features (6 axes × 4 stats)
    extract_stats(&accel_x_stats);
    extract_stats(&accel_y_stats);
    extract_stats(&accel_z_stats);
    extract_stats(&gyro_x_stats);
    extract_stats(&gyro_y_stats);
    extract_stats(&gyro_z_stats);

    // Derivative features (12 total: 6 axes × 2 stats)
    // Use last 10 samples to compute 9 differences (matching Python np.diff)
    auto calc_derivative_stats = [&](float *buffer) {
        float diff_sum = 0.0f;
        float diff_sumSq = 0.0f;
        static const int derivative_window = 10;
        static const int n_diffs = derivative_window - 1;  // 9 differences from 10 samples

        // Loop from i=0 to i<9 to match Python: np.diff(data[-10:])
        // i=0 gives diff between most recent (buffer_index-1) and 2nd most recent (buffer_index-2)
        for (int i = 0; i < n_diffs; i++) {
            int idx_curr = (buffer_index - 1 - i + FEATURE_WINDOW_SIZE) % FEATURE_WINDOW_SIZE;
            int idx_prev = (buffer_index - 2 - i + FEATURE_WINDOW_SIZE) % FEATURE_WINDOW_SIZE;
            float diff = buffer[idx_curr] - buffer[idx_prev];
            diff_sum += diff;
            diff_sumSq += diff * diff;
        }

        float diff_mean = diff_sum / n_diffs;
        float diff_variance = (diff_sumSq / n_diffs) - (diff_mean * diff_mean);
        float diff_std = (diff_variance > 0.0f) ? fast_sqrt(diff_variance) : 0.0f;

        features[idx++] = diff_mean;
        features[idx++] = diff_std;
    };

    calc_derivative_stats(accel_x_buffer);
    calc_derivative_stats(accel_y_buffer);
    calc_derivative_stats(accel_z_buffer);
    calc_derivative_stats(gyro_x_buffer);
    calc_derivative_stats(gyro_y_buffer);
    calc_derivative_stats(gyro_z_buffer);

    // Magnitude features (6 total: 2 magnitudes × 3 stats)
    // Accelerometer magnitude
    float accel_mag_sum = 0.0f;
    float accel_mag_max = 0.0f;
    float accel_mag_sumSq = 0.0f;

    for (int i = 0; i < FEATURE_WINDOW_SIZE; i++) {
        float mag = fast_sqrt(accel_x_buffer[i]*accel_x_buffer[i] +
                              accel_y_buffer[i]*accel_y_buffer[i] +
                              accel_z_buffer[i]*accel_z_buffer[i]);
        accel_mag_sum += mag;
        accel_mag_sumSq += mag * mag;
        if (mag > accel_mag_max) accel_mag_max = mag;
    }

    float accel_mag_mean = accel_mag_sum / FEATURE_WINDOW_SIZE;
    float accel_mag_variance = (accel_mag_sumSq / FEATURE_WINDOW_SIZE) - (accel_mag_mean * accel_mag_mean);
    float accel_mag_std = (accel_mag_variance > 0.0f) ? fast_sqrt(accel_mag_variance) : 0.0f;

    features[idx++] = accel_mag_mean;
    features[idx++] = accel_mag_std;
    features[idx++] = accel_mag_max;

    // Gyro magnitude
    float gyro_mag_sum = 0.0f;
    float gyro_mag_max = 0.0f;
    float gyro_mag_sumSq = 0.0f;

    for (int i = 0; i < FEATURE_WINDOW_SIZE; i++) {
        float mag = fast_sqrt(gyro_x_buffer[i]*gyro_x_buffer[i] +
                              gyro_y_buffer[i]*gyro_y_buffer[i] +
                              gyro_z_buffer[i]*gyro_z_buffer[i]);
        gyro_mag_sum += mag;
        gyro_mag_sumSq += mag * mag;
        if (mag > gyro_mag_max) gyro_mag_max = mag;
    }

    float gyro_mag_mean = gyro_mag_sum / FEATURE_WINDOW_SIZE;
    float gyro_mag_variance = (gyro_mag_sumSq / FEATURE_WINDOW_SIZE) - (gyro_mag_mean * gyro_mag_mean);
    float gyro_mag_std = (gyro_mag_variance > 0.0f) ? fast_sqrt(gyro_mag_variance) : 0.0f;

    features[idx++] = gyro_mag_mean;
    features[idx++] = gyro_mag_std;
    features[idx++] = gyro_mag_max;
}

bool MLPredictor::predict() {
    // Throttle inference rate
    if (inferenceTimer < ML_INFERENCE_INTERVAL_MS) {
        return preFireActive;
    }
    inferenceTimer = 0;

    // Check if buffer is filled
    if (!buffer_filled) {
        return false;
    }

    // Extract features
    float features[42];
    extractFeatures(features);

    // Run model using the auto-generated Random Forest functions
    float prob = predict_prefire_probability(features);
    int prediction = (prob >= RF_THRESHOLD) ? 1 : 0;

    // Store probability for UI feedback
    lastProbability = prob;

    // Apply consecutive filtering
    if (prediction == 1) {
        consecutiveCount++;
        if (consecutiveCount >= consecutiveRequired) {
            preFireActive = true;
            timeoutTimer = 0; // Reset timeout
        }
    } else {
        consecutiveCount = 0;
    }

    return preFireActive;
}

void MLPredictor::updateTimeout() {
    if (preFireActive && timeoutTimer >= timeout_ms) {
        preFireActive = false;
    }
}

bool MLPredictor::isPreSpinning() {
    return preFireActive;
}

bool MLPredictor::shouldPreSpin() {
    return preFireActive;
}

void MLPredictor::reset() {
    preFireActive = false;
    consecutiveCount = 0;
    lastProbability = 0.0f;
    buffer_index = 0;
    buffer_filled = false;
    timeoutTimer = 0;
    inferenceTimer = 0;

    // Reset buffers
    memset(accel_x_buffer, 0, sizeof(accel_x_buffer));
    memset(accel_y_buffer, 0, sizeof(accel_y_buffer));
    memset(accel_z_buffer, 0, sizeof(accel_z_buffer));
    memset(gyro_x_buffer, 0, sizeof(gyro_x_buffer));
    memset(gyro_y_buffer, 0, sizeof(gyro_y_buffer));
    memset(gyro_z_buffer, 0, sizeof(gyro_z_buffer));

    // Reset stats
    accel_x_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
    accel_y_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
    accel_z_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
    gyro_x_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
    gyro_y_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
    gyro_z_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
}

void MLPredictor::setTimeout(uint16_t timeout_ms_val) {
    timeout_ms = timeout_ms_val;
}

uint16_t MLPredictor::getTimeout() {
    return timeout_ms;
}

void MLPredictor::setConsecutiveRequired(uint8_t count) {
    // Clamp to reasonable range: 5-50 (at 10ms inference = 50-500ms latency)
    if (count < 5) count = 5;
    if (count > 50) count = 50;
    consecutiveRequired = count;
    // Reset consecutive count when changing requirement
    consecutiveCount = 0;
}

uint8_t MLPredictor::getConsecutiveRequired() {
    return consecutiveRequired;
}

uint8_t MLPredictor::getConsecutiveCount() {
    return consecutiveCount;
}

float MLPredictor::getLastProbability() {
    return lastProbability;
}

#ifdef ENABLE_ML_LOGGER
void MLPredictor::logSample(int16_t accel_x, int16_t accel_y, int16_t accel_z,
                             int16_t gyro_x, int16_t gyro_y, int16_t gyro_z,
                             uint8_t trigger_state) {
    // Check for serial commands to start/stop logging
    if (Serial.available()) {
        char cmd = Serial.read();
        if (cmd == '1' && !loggingEnabled) {
            loggingEnabled = true;
            samplesLogged = 0;
            Serial.println("--- ML DATA LOGGING STARTED ---");
            Serial.println("accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,trigger_state");
        } else if (cmd == '0' && loggingEnabled) {
            loggingEnabled = false;
            Serial.println("--- ML DATA LOGGING STOPPED ---");
            Serial.printf("Total samples logged: %u\n", samplesLogged);
        }
    }

    // Only log if enabled
    if (!loggingEnabled) {
        return;
    }

    // Rate limit to 1600 Hz (avoid duplicates)
    if (lastLogTime < LOG_INTERVAL_US) {
        return;
    }
    lastLogTime = 0;

    // Output CSV format
    Serial.printf("%d,%d,%d,%d,%d,%d,%d\n",
                  accel_x, accel_y, accel_z,
                  gyro_x, gyro_y, gyro_z,
                  trigger_state);

    samplesLogged++;

    // Print progress every 1000 samples
    if (samplesLogged % 1000 == 0) {
        Serial.printf("--- Logged %u samples ---\n", samplesLogged);
    }
}
#endif

#endif // HW_VERSION == 2
