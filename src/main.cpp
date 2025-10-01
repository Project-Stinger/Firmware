#include "Fonts/FreeSans9pt7b.h"
#include "global.h"
#include "model.h" // Our exported ML model

volatile u8 setupDone = 0b00;
volatile bool setup1CanStart = false;

#if HW_VERSION == 2 // All ML-related code is specific to v2 hardware

// --- ML Model Globals and Declarations ---

extern volatile u32 forceNewOpState;
extern i32 targetRpm;
extern i32 idleRpm;
extern i32 rearRpm;

enum PreFireState { PREFIRE_STATE_OFF,
					PREFIRE_STATE_PREDICTED,
					PREFIRE_STATE_CONFIRMED };
PreFireState preFireState = PREFIRE_STATE_OFF;

const int CONFIRMATION_THRESHOLD_MS = 50;
const int ML_INFERENCE_INTERVAL_MS = 10;
int predictionStreakCounter = 0;

const int FEATURE_WINDOW_SIZE = 50;
float accel_x_buffer[FEATURE_WINDOW_SIZE] = {0};
float accel_y_buffer[FEATURE_WINDOW_SIZE] = {0};
float accel_z_buffer[FEATURE_WINDOW_SIZE] = {0};
float gyro_x_buffer[FEATURE_WINDOW_SIZE] = {0};
float gyro_y_buffer[FEATURE_WINDOW_SIZE] = {0};
float gyro_z_buffer[FEATURE_WINDOW_SIZE] = {0};
int buffer_index = 0;
float features[24];

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

		accel_x_buffer[buffer_index] = accelDataRaw[0];
		accel_y_buffer[buffer_index] = accelDataRaw[1];
		accel_z_buffer[buffer_index] = accelDataRaw[2];
		gyro_x_buffer[buffer_index] = gyroDataRaw[0];
		gyro_y_buffer[buffer_index] = gyroDataRaw[1];
		gyro_z_buffer[buffer_index] = gyroDataRaw[2];
		buffer_index = (buffer_index + 1) % FEATURE_WINDOW_SIZE;

		if (ml_timer >= ML_INFERENCE_INTERVAL_MS) {
			ml_timer = 0;

			features[0] = calculate_mean(accel_x_buffer, FEATURE_WINDOW_SIZE);
			features[1] = calculate_std(accel_x_buffer, FEATURE_WINDOW_SIZE, features[0]);
			features[2] = calculate_min(accel_x_buffer, FEATURE_WINDOW_SIZE);
			features[3] = calculate_max(accel_x_buffer, FEATURE_WINDOW_SIZE);
			features[4] = calculate_mean(accel_y_buffer, FEATURE_WINDOW_SIZE);
			features[5] = calculate_std(accel_y_buffer, FEATURE_WINDOW_SIZE, features[4]);
			features[6] = calculate_min(accel_y_buffer, FEATURE_WINDOW_SIZE);
			features[7] = calculate_max(accel_y_buffer, FEATURE_WINDOW_SIZE);
			features[8] = calculate_mean(accel_z_buffer, FEATURE_WINDOW_SIZE);
			features[9] = calculate_std(accel_z_buffer, FEATURE_WINDOW_SIZE, features[8]);
			features[10] = calculate_min(accel_z_buffer, FEATURE_WINDOW_SIZE);
			features[11] = calculate_max(accel_z_buffer, FEATURE_WINDOW_SIZE);
			features[12] = calculate_mean(gyro_x_buffer, FEATURE_WINDOW_SIZE);
			features[13] = calculate_std(gyro_x_buffer, FEATURE_WINDOW_SIZE, features[12]);
			features[14] = calculate_min(gyro_x_buffer, FEATURE_WINDOW_SIZE);
			features[15] = calculate_max(gyro_x_buffer, FEATURE_WINDOW_SIZE);
			features[16] = calculate_mean(gyro_y_buffer, FEATURE_WINDOW_SIZE);
			features[17] = calculate_std(gyro_y_buffer, FEATURE_WINDOW_SIZE, features[16]);
			features[18] = calculate_min(gyro_y_buffer, FEATURE_WINDOW_SIZE);
			features[19] = calculate_max(gyro_y_buffer, FEATURE_WINDOW_SIZE);
			features[20] = calculate_mean(gyro_z_buffer, FEATURE_WINDOW_SIZE);
			features[21] = calculate_std(gyro_z_buffer, FEATURE_WINDOW_SIZE, features[20]);
			features[22] = calculate_min(gyro_z_buffer, FEATURE_WINDOW_SIZE);
			features[23] = calculate_max(gyro_z_buffer, FEATURE_WINDOW_SIZE);

			int prediction = eloquent::ml::port::RandomForest().predict(features);

			if (operationState == STATE_OFF) {
				switch (preFireState) {
				case PREFIRE_STATE_OFF:
					if (prediction == 1) {
						preFireState = PREFIRE_STATE_PREDICTED;
						forceNewOpState = STATE_RAMPUP; // REVERTED: Use state machine to ramp up
						predictionStreakCounter = 1;
					}
					break;
				case PREFIRE_STATE_PREDICTED:
					if (prediction == 1) {
						predictionStreakCounter++;
						if (predictionStreakCounter * ML_INFERENCE_INTERVAL_MS >= CONFIRMATION_THRESHOLD_MS) {
							preFireState = PREFIRE_STATE_CONFIRMED;
						}
					} else {
						preFireState = PREFIRE_STATE_OFF;
						forceNewOpState = STATE_RAMPDOWN; // REVERTED: Use state machine to ramp down
						predictionStreakCounter = 0;
					}
					break;
				case PREFIRE_STATE_CONFIRMED:
					if (prediction == 0) {
						preFireState = PREFIRE_STATE_OFF;
						forceNewOpState = STATE_RAMPDOWN; // REVERTED: Use state machine to ramp down
						predictionStreakCounter = 0;
					}
					break;
				}
			}
		}

		if (operationState != STATE_OFF && preFireState != PREFIRE_STATE_OFF) {
			preFireState = PREFIRE_STATE_OFF;
			predictionStreakCounter = 0;
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