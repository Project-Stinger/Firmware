#include "Fonts/FreeSans9pt7b.h"
#include "Fonts/FreeSansBold12pt7b.h"
#include "global.h"

PT1 batVoltage(10, 100);
u8 batState = BAT_SETUP;
elapsedMicros batReadTimer;
u16 readings[10];
u8 batCellCount = 6;
u8 batCellsSettings = 0;
bool batWarning = false;
char cellSettings[5][9] = {"Auto 4/6", "3S", "4S", "5S", "6S"};
u8 batWarnVoltage;
u8 batShutdownVoltage = 50;
fix32 batWarnVoltageFix;
fix32 batShutdownVoltageFix;
fix32 lowBatSchmittWidth;
u8 powerSource = 0; // 0 = unknown, 1 = battery, 2 = USB
#if HW_VERSION == 2
fix64 mahUsed = 0;
fix32 currentOffset = 0;
fix32 escCurrentAdc = 0; // A
fix32 batCurrent = 0; // A
#endif
elapsedMillis batPluginTimer;
i8 batCalibrationOffset = 0;
elapsedMillis settleTimer = 0;
#if HW_VERSION == 2
volatile bool forceInitBat = false;
#endif

void initBat() {
	motorDisableFlags |= MD_BATTERY_EMPTY;
	batPluginTimer = 0;
	batVoltage.set(0);
	batVoltage.updateCutoffFreq(10);
#if HW_VERSION == 2
	pusherEnabled = false;
	releaseLightId(LIGHT_ID::BAT_WARN);
	releaseLightId(LIGHT_ID::BAT_LOW);
#endif
	batState = BAT_SETUP;
	powerSource = 0;
	settleTimer = 0;
}

void onBatSettingChange(MenuItem *_item) {
	if (batState == BAT_RUNNING && operationState != STATE_PROFILE_SELECT) {
		batCellCount = batCellsSettings + 2;
		if (!batCellsSettings) {
			if (fix32(batVoltage) > 17.4) // 4.35V @ 4S
				batCellCount = 6;
			else
				batCellCount = 4;
		}
		batWarnVoltageFix = fix32(3 + batWarnVoltage / 100.f) * batCellCount;
		batShutdownVoltageFix = fix32(3 + batShutdownVoltage / 100.f) * batCellCount;
		lowBatSchmittWidth = fix32(0.05f) * batCellCount;
		DEBUG_PRINTF("cell count: %d, warn: %.2f, shutdown: %.2f\n", batCellCount, batWarnVoltageFix.getf32(), batShutdownVoltageFix.getf32());
	}
}

void batLoop() {
	static u8 counter = 0;
#if HW_VERSION == 2
	if (forceInitBat) {
		forceInitBat = false;
		initBat();
	}
#endif
	if (batReadTimer >= 1000) {
		batReadTimer = 0;
		readings[counter++] = adcConversions[CONV_RESULT_VBAT];
	}
	if (counter == 10) {
		counter = 0;
		u32 sum = 0;
		qsort(readings, 10, sizeof(u16), [](const void *a, const void *b) -> int {
			return (*(u16 *)a - *(u16 *)b);
		});
		for (int i = 2; i < 8; i++) {
			sum += readings[i];
		}
		sum /= 6;
#if HW_VERSION == 1
		fix32 newVoltage = sum * 0.00887f + 0.88f + 0.01f * batCalibrationOffset; // linear approximation
#elif HW_VERSION == 2
		fix32 newVoltage = sum * 0.00886f - 0.1f + 0.01f * batCalibrationOffset; // linear approximation
#endif
		if (newVoltage < 0) newVoltage = 0;
		batVoltage.update(newVoltage);

		switch (batState) {
		case BAT_SETUP:
#if HW_VERSION == 1
			if (fix32(batVoltage) < 10)
#elif HW_VERSION == 2
			if (fix32(batVoltage) < 5)
#endif
			{
				batPluginTimer = 0;
				if (settleTimer > 1000 && powerSource == 0
#if HW_VERSION == 2
					&& !standbyOn
#endif
				) {
					powerSource = 2;
				}
			}
			if (batPluginTimer > 2000) {
				batState = BAT_RUNNING;
				powerSource = 1;
				batVoltage.updateCutoffFreq(.3f);
				onBatSettingChange(nullptr);
#if HW_VERSION == 2
				enablePusher();
#endif
			}
			break;
		case BAT_RUNNING: {
			static u8 counter = 0;
#if HW_VERSION == 2
			static elapsedMillis lastBatWarningBeep = 30000;
			static elapsedMillis batWarningTimer = 1000;
			static elapsedMillis lastBatErrorBeep = 10000;
#endif
			if (counter++ == 20) {
				counter = 0;
			}
			if (fix32(batVoltage) < batShutdownVoltageFix - lowBatSchmittWidth) {
				motorDisableFlags |= MD_BATTERY_EMPTY;
#if HW_VERSION == 2
				ledSetMode(LED_MODE::BLINK_FAST, LIGHT_ID::BAT_LOW, 0, 255, 0, 0, false);
				if (lastBatErrorBeep >= 10000) {
					lastBatErrorBeep = 0;
					makeRtttlSound("batError:d=4,o=7,b=200:g,c,f#6");
				}
#endif
			} else if (fix32(batVoltage) > batShutdownVoltageFix + lowBatSchmittWidth) {
#if HW_VERSION == 2
				releaseLightId(LIGHT_ID::BAT_LOW);
#endif
				motorDisableFlags &= ~MD_BATTERY_EMPTY;
			}
			if (fix32(batVoltage) < batWarnVoltageFix) {
				batWarning = true;
#if HW_VERSION == 2
				ledSetMode(LED_MODE::BLINK, LIGHT_ID::BAT_WARN, 0, 255, 64, 0, false);
				if (lastBatWarningBeep >= 30000 && batWarningTimer >= 1000 && !(motorDisableFlags & MD_BATTERY_EMPTY)) {
					lastBatWarningBeep = 0;
					makeRtttlSound("batWarn:d=4,o=6,b=200:b,2a#");
				}
#endif
			} else {
				batWarning = false;
#if HW_VERSION == 2
				releaseLightId(LIGHT_ID::BAT_WARN);
				batWarningTimer = 0; // prevent immediate beep after warning
#endif
			}
#if HW_VERSION == 1
			if (fix32(batVoltage) < 10)
#elif HW_VERSION == 2
			if (fix32(batVoltage) < 5)
#endif
			{
				batState = BAT_SETUP;
				powerSource = 0;
				settleTimer = 0;
#if HW_VERSION == 2
				pusherEnabled = false;
				releaseLightId(LIGHT_ID::BAT_WARN);
				releaseLightId(LIGHT_ID::BAT_LOW);
#endif
				motorDisableFlags |= MD_BATTERY_EMPTY;
				batVoltage.updateCutoffFreq(10);
			}
		} break;
		}
	}
}

#if HW_VERSION == 2
void batCurrLoop() {
	escCurrentAdc = ESC_CURR_CONV * adcConversions[CONV_RESULT_IBAT];
	batCurrent = escCurrentAdc + solenoidCurrent + fix32(0.11f) + currentOffset;
	mahUsed = mahUsed + fix64(batCurrent) * 1000 / PID_RATE / 3600;
}
#endif

u8 storageModeState = STORAGE_FINISH;
u8 lastStorageModeState = STORAGE_FINISH;
u8 perCellTarget = 80; // 3.8V
i16 storageThrottle = 400;
void updatePerCellTarget(u8 perCellTarget) {
	tft.setTextColor(ST77XX_WHITE);
	SET_DEFAULT_FONT;
#if HW_VERSION == 1
	tft.fillRect(22, 64, 12, 8, ST77XX_BLACK);
	tft.setCursor(22, 64);
	tft.printf("%02d", perCellTarget);
#elif HW_VERSION == 2
	char buf[15];
	snprintf(buf, 15, "3.%02dV/cell", perCellTarget);
	tft.fillRect(10, 111, 100, YADVANCE, ST77XX_BLACK);
	printCentered(buf, 60, 111, 100, 1, 12, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
}
void updateStorageThrottle(i16 storageThrottle) {
	tft.setTextColor(ST77XX_WHITE);
	SET_DEFAULT_FONT;
#if HW_VERSION == 1
	char buf[8];
	snprintf(buf, 8, "%.1f%%", storageThrottle / 20.f);
	tft.fillRect(105, 64, 30, 8, ST77XX_BLACK);
	printCentered(buf, 120, 64, 30, 1, 8, ClipBehavior::PRINT_LAST_LINE_LEFT);
#elif HW_VERSION == 2
	char buf[15];
	snprintf(buf, 15, "<  %.1f%%  >", storageThrottle / 20.f);
	tft.fillRect(130, 111, 100, YADVANCE, ST77XX_BLACK);
	printCentered(buf, 180, 111, 100, 1, 12, ClipBehavior::PRINT_LAST_LINE_LEFT);
#endif
}

bool storageModeStart(MenuItem *_item) {
	storageModeState = STORAGE_START;
	lastStorageModeState = STORAGE_FINISH;
	perCellTarget = 80;
	storageThrottle = 400;
	triggerUpdateFlag = false;
	return true;
}

bool onStorageUp(MenuItem *_item) {
	if (storageModeState < STORAGE_FINISH) {
#if HW_VERSION == 1
		if (storageModeState == STORAGE_START) MenuItem::scheduleBeep(SETTINGS_BEEP_PERIOD, 3);
#elif HW_VERSION == 2
		MenuItem::beep(SETTINGS_BEEP_MAX_FREQ);
#endif
		perCellTarget++;
		if (perCellTarget > 90)
			perCellTarget = 60;
		DEBUG_PRINTSLN(perCellTarget);
		updatePerCellTarget(perCellTarget);
	}
	return false;
}
bool onStorageDown(MenuItem *_item) {
	if (storageModeState < STORAGE_FINISH) {
#if HW_VERSION == 1
		if (storageModeState == STORAGE_START) MenuItem::scheduleBeep(SETTINGS_BEEP_PERIOD, 0);
#elif HW_VERSION == 2
		MenuItem::beep(SETTINGS_BEEP_MIN_FREQ);
#endif
		perCellTarget--;
		if (perCellTarget < 60)
			perCellTarget = 90;
		DEBUG_PRINTSLN(perCellTarget);
		updatePerCellTarget(perCellTarget);
	}
	return false;
}
bool onStorageRight(MenuItem *item) {
	switch (storageModeState) {
	case STORAGE_START:
#if HW_VERSION == 1
		MenuItem::scheduleBeep(SETTINGS_BEEP_PERIOD, 3);
#elif HW_VERSION == 2
		MenuItem::beep(SETTINGS_BEEP_MAX_FREQ);
#endif
	case STORAGE_SPIN: {
		storageThrottle += 10;
		if (storageThrottle > 600)
			storageThrottle = 100;
		DEBUG_PRINTSLN(storageThrottle);
		updateStorageThrottle(storageThrottle);
	} break;
	case STORAGE_FINISH: {
		item->onExit();
	} break;
	}
	return false;
}
bool onStorageLeft(MenuItem *_item) {
	switch (storageModeState) {
	case STORAGE_START:
#if HW_VERSION == 1
		MenuItem::scheduleBeep(SETTINGS_BEEP_PERIOD, 0);
#elif HW_VERSION == 2
		MenuItem::beep(SETTINGS_BEEP_MIN_FREQ);
#endif
	case STORAGE_SPIN: {
		storageThrottle -= 10;
		if (storageThrottle < 100)
			storageThrottle = 600;
		DEBUG_PRINTSLN(storageThrottle);
		updateStorageThrottle(storageThrottle);
	} break;
	case STORAGE_FINISH: {
		storageModeState = STORAGE_SPIN;
		if (fix32(batVoltage).getf32() < (300 + (i32)perCellTarget) / 100.f * batCellCount)
			storageModeState = STORAGE_START;
	} break;
	}
	return false;
}

bool storageModeLoop(MenuItem *item) {
	bool newState = false;
	if (lastStorageModeState != storageModeState || item->fullRedraw) {
		item->fullRedraw = false;
		newState = true;
		lastStorageModeState = storageModeState;
	}
	switch (storageModeState) {
	case STORAGE_START: {
		static elapsedMillis displayTimer = 1000;
		if (newState) {
			displayTimer = 1000;
			tft.fillScreen(ST77XX_BLACK);
			tft.setTextColor(ST77XX_WHITE);
#if HW_VERSION == 1
			SET_DEFAULT_FONT;
			printCentered("Storage Mode", SCREEN_WIDTH / 2, 0, SCREEN_WIDTH, 1, 8, ClipBehavior::PRINT_LAST_LINE_DOTS);
			tft.setCursor(0, 11);
			tft.setTextWrap(true);
			printCentered("Fully charged batteries degrade quickly. Spin the motors to discharge.", SCREEN_WIDTH / 2, 11, SCREEN_WIDTH, 3, YADVANCE, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			// Target voltage
			printCentered("Setpoint", SCREEN_WIDTH / 4, 46, SCREEN_WIDTH / 2, 1, 8, ClipBehavior::PRINT_LAST_LINE_DOTS);
			tft.setCursor(10, 64);
			tft.print("3.  V/cell");
			tft.drawLine(37, 74, 40, 78, ST77XX_WHITE);
			tft.drawLine(40, 78, 43, 74, ST77XX_WHITE);
			tft.drawLine(37, 61, 40, 57, ST77XX_WHITE);
			tft.drawLine(40, 57, 43, 61, ST77XX_WHITE);
			updatePerCellTarget(perCellTarget);

			// Throttle
			printCentered("Throttle", SCREEN_WIDTH * 3 / 4, 46, SCREEN_WIDTH / 2, 1, 8, ClipBehavior::PRINT_LAST_LINE_DOTS);
			tft.setCursor(93, 64);
			tft.print("<       >");
			updateStorageThrottle(storageThrottle);
#elif HW_VERSION == 2
			speakerLoopOnFastCore = true;
			tft.setFont(&FreeSansBold12pt7b);
			printCentered("Storage Mode", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			SET_DEFAULT_FONT;
			printCentered("Fully charged batteries degrade quickly. Spin the motors to discharge.", SCREEN_WIDTH / 2, 32, SCREEN_WIDTH, 2, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			// Target voltage
			printCentered("Setpoint", SCREEN_WIDTH / 4, 85, SCREEN_WIDTH / 2, 1, 8, ClipBehavior::PRINT_LAST_LINE_DOTS);
			printCentered("Throttle", SCREEN_WIDTH * 3 / 4, 85, SCREEN_WIDTH / 2, 1, 8, ClipBehavior::PRINT_LAST_LINE_DOTS);
			tft.drawLine(55, 125, 60, 131, ST77XX_WHITE);
			tft.drawLine(60, 131, 65, 125, ST77XX_WHITE);
			tft.drawLine(55, 106, 60, 100, ST77XX_WHITE);
			tft.drawLine(60, 100, 65, 106, ST77XX_WHITE);
			updatePerCellTarget(perCellTarget);
			updateStorageThrottle(storageThrottle);
			speakerLoopOnFastCore = false;
#endif
			triggerUpdateFlag = false;
		}
		static elapsedMillis triggerTimer = 0;
		if (triggerUpdateFlag) {
			triggerUpdateFlag = false;
			if (triggerState == 0) {
				if (triggerTimer < 1000) {
					storageModeState = STORAGE_SPIN;
					if (fix32(batVoltage).getf32() < (300 + (i32)perCellTarget) / 100.f * batCellCount) {
						storageModeState = STORAGE_FINISH;
					}
				}
			} else {
				triggerTimer = 0;
			}
		}
		if (triggerTimer >= 1000 && triggerState == 1) {
			item->onExit();
		}
		if (displayTimer >= 1000) {
			static bool longPull = false;
			displayTimer = 0;
			SET_DEFAULT_FONT;
			tft.setTextColor(ST77XX_WHITE);
			tft.fillRect(0, HW_VERSION == 1 ? 35 : 65, SCREEN_WIDTH, YADVANCE, ST77XX_BLACK);
			printCentered(longPull ? "Long pull to abort" : "Pull trigger to start.", SCREEN_WIDTH / 2, HW_VERSION == 1 ? 35 : 65, SCREEN_WIDTH, 1, YADVANCE, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			longPull = !longPull;
		}
	} break;
	case STORAGE_SPIN: {
		static elapsedMillis x = 0;
		if (triggerUpdateFlag) {
			triggerUpdateFlag = false;
			if (triggerState == 1) {
				storageModeState = STORAGE_FINISH;
			}
		}
		menuOverrideEsc[0] = storageThrottle;
		menuOverrideEsc[1] = storageThrottle;
		menuOverrideEsc[2] = storageThrottle;
		menuOverrideEsc[3] = storageThrottle;
		menuOverrideTimer = 0;
		inactivityTimer = 0;
		if (fix32(batVoltage).getf32() < (300 + (i32)perCellTarget) / 100.f * batCellCount) {
			storageModeState = STORAGE_FINISH;
			menuOverrideEsc[0] = 0;
			menuOverrideEsc[1] = 0;
			menuOverrideEsc[2] = 0;
			menuOverrideEsc[3] = 0;
			menuOverrideTimer = 101;
		}
		if (newState) {
			tft.setTextColor(ST77XX_WHITE);
			tft.setTextWrap(true);
			SET_DEFAULT_FONT;
#if HW_VERSION == 1
			tft.setCursor(0, 15);
			tft.fillRect(0, 11, SCREEN_WIDTH, 32, ST77XX_BLACK);
			tft.printf("Discharging... %.2fV\n\nPull Trigger to abort", fix32(batVoltage).getf32() / batCellCount);
#elif HW_VERSION == 2
			tft.fillRect(0, 32, SCREEN_WIDTH, 48, ST77XX_BLACK);
			tft.setCursor(0, 40);
			tft.printf("Discharging... %.2fV\n\nPull Trigger to abort", fix32(batVoltage).getf32() / batCellCount);
#endif
		}
		if (x > 500) {
			x = 0;
			tft.setTextColor(ST77XX_WHITE);
			tft.setTextWrap(true);
			SET_DEFAULT_FONT;
#if HW_VERSION == 1
			tft.setCursor(90, 15);
			tft.fillRect(90, 15, 30, 8, ST77XX_BLACK);
			tft.printf("%.2fV", fix32(batVoltage).getf32() / batCellCount);
#elif HW_VERSION == 2
			tft.setCursor(0, 40);
			tft.fillRect(0, 40, 150, YADVANCE, ST77XX_BLACK);
			tft.printf("Discharging... %.2fV", fix32(batVoltage).getf32() / batCellCount);
#endif
		}
	} break;
	case STORAGE_FINISH: {
		static elapsedMillis finishedTimer = 0;
		static u8 finishBeeps = 0;
		if (newState) {
			finishedTimer = HW_VERSION == 1 ? 0 : 30000;
			finishBeeps = 0;
			tft.fillScreen(ST77XX_BLACK);
			tft.setTextColor(ST77XX_WHITE);
			SET_DEFAULT_FONT;
			char buf[40];
			snprintf(buf, 40, "Discharging to 3.%02dV finished.", perCellTarget);
#if HW_VERSION == 1
			printCentered("Storage Mode", SCREEN_WIDTH / 2, 0, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			printCentered(buf, SCREEN_WIDTH / 2, 15, SCREEN_WIDTH, 2, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_DOTS);
			printCentered("Press right to exit.", SCREEN_WIDTH / 2, 50, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
#elif HW_VERSION == 2
			printCentered(buf, SCREEN_WIDTH / 2, 40, SCREEN_WIDTH, 2, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_DOTS);
			printCentered("Press right to exit.", SCREEN_WIDTH / 2, 80, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			tft.setFont(&FreeSansBold12pt7b);
			printCentered("Storage Mode", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
#endif
		}
#if HW_VERSION == 1
		if (finishedTimer > 1000 && finishBeeps < 5) {
			finishBeeps++;
			finishedTimer = 0;
			pushToAllCommandBufs(DSHOT_CMD_BEACON4);
		}
#elif HW_VERSION == 2
		if (finishedTimer > 30000 && finishBeeps < 7) {
			finishBeeps++;
			finishedTimer = 0;
			makeRtttlSound("storageFinished:d=4,o=5,b=250:d,g,b,d6,4p,b,2d6,4d6");
		}
#endif
	} break;
	}
	return true;
}
