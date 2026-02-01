#pragma once
#if HW_VERSION == 2

#include "utils/typedefs.h"

enum MlModel : u8 {
	ML_MODEL_LOGREG = 0,
	ML_MODEL_MLP = 1,
};

/// @brief Called once from setup() on core 0. Zeroes the sliding window buffer.
void mlInferInit();

/// @brief Called from loop1() (core 1) at 100 Hz (decimated from gyro rate).
/// Pushes one sample into the sliding window.
void mlInferPushSample(i16 ax, i16 ay, i16 az, i16 gx, i16 gy, i16 gz);

/// @brief Run inference on the current sliding window.
/// @param model Which model to use.
/// @return Probability [0.0, 1.0] that a shot is imminent.
float mlInferPredict(MlModel model);

/// @brief Called from loop() (core 0). Runs the expensive inference and caches the output.
/// This keeps core 1 timing clean; core 1 should only call mlInferGetCachedProb().
/// @param enable Whether to run inference this tick.
/// @param model Which model to run if enabled.
void mlInferSlowLoop(bool enable, MlModel model);

/// @brief Get the cached probability (computed on core 0).
float mlInferGetCachedProb(MlModel model);

/// @brief Age in ms since last cached probability update (core 0).
u32 mlInferCachedAgeMs();

/// @brief Returns true if a user model file exists in LittleFS.
bool mlInferUserModelExists();

/// @brief Load user model from LittleFS (default path) into RAM and start using it.
/// This loads either LR or MLP based on the file header.
bool mlInferLoadUserModel();

/// @brief Load a specific user model type from its dedicated file.
/// - LR: /ml_model_lr.bin
/// - MLP: /ml_model_mlp.bin
bool mlInferLoadUserModelType(MlModel model);

/// @brief Clear user model file from LittleFS.
bool mlInferDeleteUserModel();

/// @brief Print model info to Serial (core 0).
void mlInferPrintInfo();

/// @brief Called from loop1() (core 1) on gyro cycles. Handles decimation
/// and pushes samples into the sliding window at 100 Hz.
void mlInferLoop();

/// @brief Returns true when the sliding window has been filled at least once.
bool mlInferReady();

#endif // HW_VERSION == 2
