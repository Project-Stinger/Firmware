#include "Fonts/FreeSans9pt7b.h"
#include "global.h"
#include "model.h" // Our exported ML model
#include <cfloat> // For FLT_MAX

volatile u8 setupDone = 0b00;
volatile bool setup1CanStart = false;

#if HW_VERSION == 2 // All ML-related code is specific to v2 hardware

// --- ML Model Globals and Declarations ---

extern volatile u32 forceNewOpState;
extern i32 targetRpm;
extern i32 idleRpm;
extern i32 rearRpm;

// Remove the enum definition from here - it's now in operationSm.h
PreFireState preFireState = PREFIRE_STATE_OFF;

const int CONFIRMATION_THRESHOLD_MS = 10;  // Reduced from 20ms to 10ms - even faster!
const int ML_INFERENCE_INTERVAL_MS = 10;
// DEFAULT_PRESPIN_TIMEOUT_MS is already defined in operationSm.h, don't redefine it here
int predictionStreakCounter = 0;
elapsedMillis preSpinTimer = 0; // Add timer for pre-spin timeout

// Make timeout configurable
u16 mlPreSpinTimeout = DEFAULT_PRESPIN_TIMEOUT_MS;

const int FEATURE_WINDOW_SIZE = 50;
float accel_x_buffer[FEATURE_WINDOW_SIZE] = {0};
float accel_y_buffer[FEATURE_WINDOW_SIZE] = {0};
float accel_z_buffer[FEATURE_WINDOW_SIZE] = {0};
float gyro_x_buffer[FEATURE_WINDOW_SIZE] = {0};
float gyro_y_buffer[FEATURE_WINDOW_SIZE] = {0};
float gyro_z_buffer[FEATURE_WINDOW_SIZE] = {0};
int buffer_index = 0;
float features[24];

// Instead of calculating all 24 features every 10ms, calculate incrementally:

struct RunningStats {
	float sum;
	float sumSq;
	float min;
	float max;
	int samples_seen; // Track how many real samples we've collected
	bool needs_recalc;
};

RunningStats accel_x_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
RunningStats accel_y_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
RunningStats accel_z_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
RunningStats gyro_x_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
RunningStats gyro_y_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};
RunningStats gyro_z_stats = {0, 0, FLT_MAX, -FLT_MAX, 0, false};

inline float fast_sqrt(float x) {
	// Quake fast inverse square root adaptation
	union {
		float f;
		uint32_t i;
	} conv = {.f = x};
	conv.i = 0x5f3759df - (conv.i >> 1);
	float y = conv.f;
	y = y * (1.5f - (x * 0.5f * y * y));
	return x * y; // Return sqrt instead of inverse sqrt
}

inline void update_stats(RunningStats *stats, float *buffer, float new_val, float old_val) {
	// Handle initial fill-up phase
	if (stats->samples_seen < FEATURE_WINDOW_SIZE) {
		stats->samples_seen++;
		// During warmup, just add without subtracting
		stats->sum += new_val;
		stats->sumSq += new_val * new_val;

		// Update min/max
		if (new_val < stats->min) stats->min = new_val;
		if (new_val > stats->max) stats->max = new_val;
		return;
	}

	// Normal operation: sliding window
	stats->sum -= old_val;
	stats->sumSq -= old_val * old_val;
	stats->sum += new_val;
	stats->sumSq += new_val * new_val;

	// Check if we need to recalculate min/max
	// Use small epsilon for float comparison
	const float epsilon = 0.001f;
	if (fabsf(old_val - stats->min) < epsilon || fabsf(old_val - stats->max) < epsilon) {
		stats->needs_recalc = true;
	}

	// Update min/max
	if (stats->needs_recalc) {
		// Full recalculation
		stats->min = buffer[0];
		stats->max = buffer[0];
		for (int i = 1; i < FEATURE_WINDOW_SIZE; i++) {
			if (buffer[i] < stats->min) stats->min = buffer[i];
			if (buffer[i] > stats->max) stats->max = buffer[i];
		}
		stats->needs_recalc = false;
	} else {
		// Fast path
		if (new_val < stats->min) stats->min = new_val;
		if (new_val > stats->max) stats->max = new_val;
	}
}

inline void get_features_from_stats(RunningStats *stats, float *features, int offset) {
	// Only calculate valid features if we have enough samples
	if (stats->samples_seen < FEATURE_WINDOW_SIZE) {
		// Not enough data yet - return zeros
		features[offset + 0] = 0.0f;
		features[offset + 1] = 0.0f;
		features[offset + 2] = 0.0f;
		features[offset + 3] = 0.0f;
		return;
	}

	float mean = stats->sum / FEATURE_WINDOW_SIZE;
	features[offset + 0] = mean;
	features[offset + 1] = fast_sqrt((stats->sumSq / FEATURE_WINDOW_SIZE) - (mean * mean));
	features[offset + 2] = stats->min;
	features[offset + 3] = stats->max;
}

// --- Helper functions to calculate statistics ---
float calculate_mean(float *arr, int size) {
	float sum = 0;
	for (int i = 0; i < size; i++)
		sum += arr[i];
	return sum / size;
}

float calculate_std(float *arr, int size, float mean) {
	float sum = 0;
	for (int i = 0; i < size; i++)
		sum += (arr[i] - mean) * (arr[i] - mean);
	return sqrt(sum / size);
}

float calculate_min(float *arr, int size) {
	float min_val = arr[0];
	for (int i = 1; i < size; i++)
		if (arr[i] < min_val) min_val = arr[i];
	return min_val;
}

float calculate_max(float *arr, int size) {
	float max_val = arr[0];
	for (int i = 1; i < size; i++)
		if (arr[i] > max_val) max_val = arr[i];
	return max_val;
}

#endif // HW_VERSION == 2

// --- SETUP AND LOOP FUNCTIONS (Original) ---

void setup() {
	if (powerOnResetMagicNumber == 0xdeadbeefdeadbeef)
		bootReason = rebootReason;
	else
		bootReason = BootReason::POR;
	powerOnResetMagicNumber = 0xdeadbeefdeadbeef;
	rebootReason = BootReason::WATCHDOG;
	Serial.begin(115200);
	DEBUG_PRINTSLN("start");
#if HW_VERSION == 2
	initStandbySwitch();
	initSpeaker();
	ledInit();
#endif
	initAnalog();
	if (bootReason == BootReason::TO_ESC_PASSTHROUGH) {
		initDisplay();
		u8 pins[4] = {PIN_MOTOR_BASE, PIN_MOTOR_BASE + 1, PIN_MOTOR_BASE + 2, PIN_MOTOR_BASE + 3};
		beginPassthrough(pins, 4);
		triggerInit();
		tft.setFont(&FreeSans9pt7b);
		tft.fillScreen(ST77XX_BLACK);
		tft.setTextColor(ST77XX_WHITE);
		printCentered("ESC Passthrough", SCREEN_WIDTH / 2, 15, SCREEN_WIDTH, 1, 22, ClipBehavior::PRINT_LAST_LINE_CENTERED);
		SET_DEFAULT_FONT;
#if HW_VERSION == 1
		printCentered("Use BLHeliSuite32 to configure ESCs. Click disconnect or hold the trigger for 3 sec to boot into normal mode again.", SCREEN_WIDTH / 2, 30, SCREEN_WIDTH, 5, YADVANCE, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
		printCentered("Use AM32 Configurator to configure ESCs. Click disconnect or hold the trigger for 3 seconds to boot into normal mode again.", SCREEN_WIDTH / 2, 40, SCREEN_WIDTH, 5, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
		elapsedMillis triggerTimer = 0;
		elapsedMillis batteryTimer = 2000;
		while (processPassthrough()) {
			static elapsedMicros adcTimer = 0;
			if (adcTimer >= 1000000 / PID_RATE) {
				adcTimer -= 1000;
				adc_run(false);
				sleep_us(3);
				analogLoop();
			}
			batLoop();
			triggerLoop();
			if (triggerState) {
				if (triggerUpdateFlag) {
					triggerUpdateFlag = false;
					triggerTimer = 0;
				}
				if (triggerTimer >= 3000)
					break;
			} else {
				triggerUpdateFlag = false;
			}
			if (batteryTimer >= 2000) {
				char buf[32];
				snprintf(buf, 32, "Battery: %4.1fV", fix32(batVoltage).getf32());
				tft.fillRect(0, SCREEN_HEIGHT - YADVANCE, 100, YADVANCE, ST77XX_BLACK);
				tft.setCursor(0, SCREEN_HEIGHT - YADVANCE);
				tft.print(buf);
				batteryTimer = 0;
			}
		}
		sleep_ms(100);
		rebootReason = BootReason::FROM_ESC_PASSTHROUGH;
		rp2040.reboot();
	}
	eepromInit();
#ifdef USE_TOF
	initTof();
#endif
	setup1CanStart = true;
	initDisplay();
	initESCs();
	initBat();
	tournamentInit();
#if HW_VERSION == 2
	initGyroSpi();
	gyroInit();
	imuInit();
#endif
	setupDone |= 0b01;
	while (setupDone != 0b11) {
		bootTimer = 0;
		tight_loop_contents();
	}
	bootTimer = 0;
	DEBUG_PRINTSLN("Setup done");
#if HW_VERSION == 2
	playStartupSound();
#endif
}

void loop() {
#ifdef USE_TOF
	tofLoop();
#endif
	batLoop();
#if HW_VERSION == 2
	if (!speakerLoopOnFastCore && !speakerLoopOnFastCore2)
		speakerLoop();
	ledLoop();
#endif
	displayLoop();
	if (openedMenu != nullptr && operationState == STATE_MENU
#if HW_VERSION == 2
		&& !standbyOn
#endif
	) {
		openedMenu->loop();
	}
	tournamentLoop();
}

void setup1() {
	while (!setup1CanStart) {
		tight_loop_contents();
	}
	triggerInit();
	initPusher();
	setupDone |= 0b10;
	while (setupDone != 0b11) {
		tight_loop_contents();
	}
}

elapsedMicros pidCycleTimer = 0;
#if HW_VERSION == 2
bool gyroCycle = true;
#endif

void loop1() {
	triggerLoop();
	if (pidCycleTimer >= 1000000 / PID_RATE) {
		pidCycleTimer -= 1000000 / PID_RATE;
		if (pidCycleTimer > 3000) pidCycleTimer = 3000;
#if HW_VERSION == 2
		if (standbyOn) {
			standbyOnLoop();
			return;
		}
#endif
		decodeErpm();
		adc_run(false);
		checkTelemetry();
		runOperationSm();
		analogLoop();
		joystickLoop();
#if HW_VERSION == 2
		pusherLoop();
		batCurrLoop();
		if (gyroCycle) {
			gyroLoop();
			freeFallDetection();
			updateAtti1();
			gyroCycle = false;
		} else {
			updateAtti2();
			gyroCycle = true;
		}
		if (!standbyOn)
			standbyOffLoop();
		if (speakerLoopOnFastCore || speakerLoopOnFastCore2)
			speakerLoop();

#ifndef ENABLE_ML_LOGGER
		// =================================================================
		// =========== MACHINE LEARNING PRE-FIRE INFERENCE LOGIC ===========
		// =================================================================
		static elapsedMillis ml_timer = 0;

		// Store old values before updating buffers
		float old_accel_x = accel_x_buffer[buffer_index];
		float old_accel_y = accel_y_buffer[buffer_index];
		float old_accel_z = accel_z_buffer[buffer_index];
		float old_gyro_x = gyro_x_buffer[buffer_index];
		float old_gyro_y = gyro_y_buffer[buffer_index];
		float old_gyro_z = gyro_z_buffer[buffer_index];

		// Update buffers with new values
		accel_x_buffer[buffer_index] = accelDataRaw[0];
		accel_y_buffer[buffer_index] = accelDataRaw[1];
		accel_z_buffer[buffer_index] = accelDataRaw[2];
		gyro_x_buffer[buffer_index] = gyroDataRaw[0];
		gyro_y_buffer[buffer_index] = gyroDataRaw[1];
		gyro_z_buffer[buffer_index] = gyroDataRaw[2];

		// Update running statistics
		update_stats(&accel_x_stats, accel_x_buffer, accelDataRaw[0], old_accel_x);
		update_stats(&accel_y_stats, accel_y_buffer, accelDataRaw[1], old_accel_y);
		update_stats(&accel_z_stats, accel_z_buffer, accelDataRaw[2], old_accel_z);
		update_stats(&gyro_x_stats, gyro_x_buffer, gyroDataRaw[0], old_gyro_x);
		update_stats(&gyro_y_stats, gyro_y_buffer, gyroDataRaw[1], old_gyro_y);
		update_stats(&gyro_z_stats, gyro_z_buffer, gyroDataRaw[2], old_gyro_z);

		buffer_index = (buffer_index + 1) % FEATURE_WINDOW_SIZE;

		// Adaptive inference rate
		int current_ml_interval = ML_INFERENCE_INTERVAL_MS;
		if (preFireState == PREFIRE_STATE_OFF) {
			current_ml_interval = 20; // Check less frequently when idle
		} else {
			current_ml_interval = ML_INFERENCE_INTERVAL_MS;
		}

		if (ml_timer >= current_ml_interval) {
			ml_timer = 0;

			// Only run inference if we have enough samples
			if (accel_x_stats.samples_seen < FEATURE_WINDOW_SIZE) {
				// Still warming up, skip inference - do nothing this cycle
			} else {
				// Use optimized feature calculation
				get_features_from_stats(&accel_x_stats, features, 0);
				get_features_from_stats(&accel_y_stats, features, 4);
				get_features_from_stats(&accel_z_stats, features, 8);
				get_features_from_stats(&gyro_x_stats, features, 12);
				get_features_from_stats(&gyro_y_stats, features, 16);
				get_features_from_stats(&gyro_z_stats, features, 20);

#ifdef PROFILE_ML
				static uint32_t max_inference_us = 0;
				uint32_t start = micros();
				int prediction = eloquent::ml::port::RandomForest().predict(features);
				uint32_t elapsed = micros() - start;
				if (elapsed > max_inference_us) {
					max_inference_us = elapsed;
					Serial.printf("ML inference: %d us (max: %d)\n", elapsed, max_inference_us);
				}
#else
				int prediction = eloquent::ml::port::RandomForest().predict(features);
#endif

				// Only manage pre-spin when in OFF state AND in ML idle mode
				if (operationState == STATE_OFF && idleEnabled == 8) {
					switch (preFireState) {
					case PREFIRE_STATE_OFF:
						if (prediction == 1) {
							preFireState = PREFIRE_STATE_SPINNING; // Jump directly to spinning
							preSpinTimer = 0;
						}
						break;

					case PREFIRE_STATE_PREDICTED:
					case PREFIRE_STATE_CONFIRMED:
						// These states are no longer used for ML idle mode
						// Jump directly to spinning
						preFireState = PREFIRE_STATE_SPINNING;
						preSpinTimer = 0;
						break;

					case PREFIRE_STATE_SPINNING:
						// Retriggerable idle: any new prediction restarts the timeout
						if (prediction == 1) {
							preSpinTimer = 0; // Reset timer - keep spinning
						}
						
						// Use configurable timeout
						if (preSpinTimer >= mlPreSpinTimeout) {
							// Timeout - return to OFF
							preFireState = PREFIRE_STATE_OFF;
							predictionStreakCounter = 0;
						}
						break;
					}
				} else if (idleEnabled != 8) {
					// If not in ML mode, always clear pre-fire state
					if (preFireState != PREFIRE_STATE_OFF) {
						preFireState = PREFIRE_STATE_OFF;
						predictionStreakCounter = 0;
					}
				} else if (operationState != STATE_OFF) {
					// If we're firing or in any other state, clear the pre-fire
					// This prevents twitching when returning from firing
					preFireState = PREFIRE_STATE_OFF;
					predictionStreakCounter = 0;
				}
			}
		}
#endif

#if ENABLE_ML_LOGGER
		static bool loggingActive = false;

		if (!loggingActive && Serial.available() > 0) {
			Serial.read();
			loggingActive = true;
			Serial.println("--- ML Data Stream Started ---");
		}

		if (loggingActive) {
			Serial.printf("%d;%d;%d;%d;%d;%d;%d\n",
						  accelDataRaw[0], accelDataRaw[1], accelDataRaw[2],
						  gyroDataRaw[0], gyroDataRaw[1], gyroDataRaw[2],
						  triggerState);
		}
#endif

#if ENABLE_DEBUG_GYRO
		static elapsedMillis c = 0;
		if (c >= 5) {
			c = 0;
			static u8 gyro = false;
			if (Serial.available()) {
				Serial.read();
				++gyro;
			}
			switch (gyro) {
			case 1:
				Serial.printf("%d %d %d\n", gyroDataRaw[0], gyroDataRaw[1], gyroDataRaw[2]);
				break;
			case 2:
				Serial.printf("%d %d %d\n", accelDataRaw[0], accelDataRaw[1], accelDataRaw[2]);
				break;
			case 3:
				Serial.printf("%f %f %f\n", roll.getf32(), pitch.getf32(), yaw.getf32());
				break;
			case 4:
				Serial.printf("%.3f %.3f %.3f\n", fix32(vAccel).getf32(), fix32(rAccel).getf32(), fix32(fAccel).getf32());
				break;
			case 5: {
				if (operationState >= STATE_PUSH && operationState <= STATE_RETRACT) {
					Serial.printf("%d %d\n", (solenoidCurrent * 1000).geti32(), accelDataRaw[1]);
				}
			} break;
			case 6:
				Serial.printf("%d %d\n", adcConversions[CONV_RESULT_ISOLENOID], adcConversions[CONV_RESULT_IBAT]);
				break;
			default:
				gyro = 0;
				break;
			}
		}
#endif
#endif
	}
}