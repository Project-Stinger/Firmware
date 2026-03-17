#include "Fonts/FreeSans9pt7b.h"
#include "global.h"

volatile u8 setupDone = 0b00;
volatile bool setup1CanStart = false;

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
	initDisplay(); // display reserves pin 12 as the default SPI 1 MISO pin
	initESCs(); // ESCs must be initiated after the display to avoid pin conflicts
	initBat();
	tournamentInit();
#if HW_VERSION == 2
	initGyroSpi();
	gyroInit();
	imuInit();
	mlLogInit();
	mlInferInit();
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
	usbSessionInit();
#endif
}

/**
 * @brief non time critical loop
 * @details Serial communication, display updates, menu etc.
 */
void loop() {
#ifdef USE_TOF
	tofLoop();
#endif
	batLoop();
#if HW_VERSION == 2
	usbSessionLoop0();
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
#if HW_VERSION == 2
	mlLogSlowLoop();
	{
		const bool enable = (idleEnabled == 8 || idleEnabled == 9) && !mlLogIsActive() && !usbCdcActive();
		const MlModel model = (idleEnabled == 8) ? ML_MODEL_LOGREG : ML_MODEL_MLP;
		mlInferSlowLoop(enable, model);
	}
#endif
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

/**
 * @brief time critical loop
 * @details ESC communication, control loop, trigger pin etc.
 */
void loop1() {
	triggerLoop();
	if (pidCycleTimer >= 1000000 / PID_RATE) {
		pidCycleTimer -= 1000000 / PID_RATE;
		if (pidCycleTimer > 3000) pidCycleTimer = 3000; // maximum time to catch up is 3ms
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
			mlLogLoop();
			mlInferLoop();
			gyroCycle = false;
		} else {
			updateAtti2();
			gyroCycle = true;
		}
		if (!standbyOn)
			standbyOffLoop();
		if (speakerLoopOnFastCore || speakerLoopOnFastCore2)
			speakerLoop();
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
#endif // ENABLE_DEBUG_GYRO
#endif // HW_VERSION == 2
	}
}
