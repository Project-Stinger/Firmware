
#include "Fonts/FreeSans9pt7b.h"
#include "Fonts/FreeSansBold12pt7b.h"
#include "VL53L0X.h"
#include "global.h"

i32 dartCount = 0;

#ifdef USE_TOF

#if HW_VERSION == 1
#define NO_MAG_THRESHOLD 65
#define MAG_THRESHOLD 55
#elif HW_VERSION == 2
#define NO_MAG_THRESHOLD 55
#define MAG_THRESHOLD 50
#endif

VL53L0X tof;
PT1 tofDistance(3, 30);
bool foundTof = false;
elapsedMicros tofTimer = 0;
u8 tofCalibrationState = 255; // 0 = insert mag and press right, 1 = remove mag and press right, 2 = test, press right to continue (left to retry), 3 = error (left to retry, right to skip)
u8 tofThresLow = MAG_THRESHOLD;
u8 tofThresHigh = NO_MAG_THRESHOLD;
bool magPresent = false;
u8 magSize = 18;
bool fireWoMag = false;
bool fireWoDarts = false;
#if HW_VERSION == 2
bool beepOnMagChange = true;
u8 tofTries = 0;
#endif
u8 tofState = 0; // every I2C transaction takes about 100us, so spread them out

void initTof() {
	WIRE_TOF.setSDA(PIN_SDA);
	WIRE_TOF.setSCL(PIN_SCL);
	WIRE_TOF.setClock(400000);
	WIRE_TOF.begin();
	tof.setBus(&WIRE_TOF);
	tof.setTimeout(30);
	if (!firstBoot) {
		EEPROM.get(EEPROM_POS_TOF_HIGH, tofThresHigh);
		EEPROM.get(EEPROM_POS_TOF_LOW, tofThresLow);
	}
	if (!tof.init()) {
		int i = 0;
		for (; i < 5; i++) {
			delay(100);
			if (tof.init()) {
				break;
			}
		}
		if (i == 5) {
			Serial.println("Failed to boot VL53L0X");
			return;
		}
	}
	foundTof = true;
	tofState = 1;
	DEBUG_PRINTSLN("Found TOF");
	tof.startContinuous();
	tof.setTimeout(0);
	magPresent = false;
	tofDistance.set(100);
}

void tofLoop() {
	switch (tofState) {
	case 1:
		// waiting for result
		if (tofTimer >= 15000) {
			tofTimer = 0;
			if (tof.readReg(VL53L0X::regAddr::RESULT_INTERRUPT_STATUS) & 0x07)
				tofState = 2;
		}
		break;
	case 2: {
		// result ready
		u16 read = tof.readReg16Bit(VL53L0X::regAddr::RESULT_RANGE_STATUS + 10);
		tofDistance.update(read);
		if (magPresent && ((fix32)tofDistance).geti32() > tofThresHigh) {
			magPresent = false;
#if HW_VERSION == 2
			if ((beepOnMagChange && tofCalibrationState == 255) || tofCalibrationState == 2)
				makeRtttlSound("MagOut:d=4,o=4,b=160:d5-,0a");
#endif
			DEBUG_PRINTSLN("Magazine removed");
			dartCount = 0;
		} else if (!magPresent && ((fix32)tofDistance).geti32() <= tofThresLow) {
			magPresent = true;
#if HW_VERSION == 2
			if ((beepOnMagChange && tofCalibrationState == 255) || tofCalibrationState == 2)
				makeRtttlSound("MagIn:d=4,o=4,b=160:a-,0d5");
#endif
			DEBUG_PRINTSLN("Magazine inserted");
			dartCount = magSize;
		}
		tofState = 3;
	} break;
	case 3:
		// clear interrupt
		tof.writeReg(VL53L0X::regAddr::SYSTEM_INTERRUPT_CLEAR, 0x01);
		tofState = foundTof ? 1 : 0; // if disableTof was called on the other core, this will force it into off state
		break;
	default:
		// e.g. 0: not found
		break;
	}
}

void disableTof() {
	tofState = 0;
	foundTof = false;
	WIRE_TOF.end();
	gpio_set_function(PIN_SCL, GPIO_FUNC_NULL);
	gpio_set_function(PIN_SDA, GPIO_FUNC_NULL);
}

u16 tofMmHigh = 0, tofMmLow = 65535;
u8 prevTofHigh = 0, prevTofLow = 0;

void finishOnboarding() {
	tofCalibrationState = 255;
	EEPROM.commit();
	firstBoot = false;
	onboardingMenu->onEnter();
	openedMenu = onboardingMenu;
}

bool startTofCalibration(MenuItem *item) {
	if (!foundTof) {
		if (!firstBoot) {
			return false;
		}
		tofThresHigh = NO_MAG_THRESHOLD;
		tofThresLow = MAG_THRESHOLD;
		EEPROM.put(EEPROM_POS_TOF_LOW, tofThresLow);
		EEPROM.put(EEPROM_POS_TOF_HIGH, tofThresHigh);

		finishOnboarding();
		return false;
	}
	prevTofHigh = tofThresHigh;
	prevTofLow = tofThresLow;
	tofCalibrationState = 0;
	tofMmLow = 65535;
	tofMmHigh = 0;
	item->triggerFullRedraw();
	return true;
}

bool onTofCalibrationRight(MenuItem *item) {
	if (lastGesture.type != GESTURE_PRESS) return false;
	switch (tofCalibrationState) {
	case 0:
		tofMmLow = fix32(tofDistance).geti32();
		tofCalibrationState = 1;
		item->triggerFullRedraw();
		break;
	case 1:
		tofMmHigh = fix32(tofDistance).geti32();
		tofMmHigh = constrain(tofMmHigh, 0, 255);
		tofMmLow = constrain(tofMmLow, 0, 255);
		if (tofMmHigh >= tofMmLow + 12) {
			tofCalibrationState = 2;
			item->triggerFullRedraw();
			u8 newTofHigh = ((i32)tofMmHigh + (i32)tofMmLow) / 2 + 3;
			u8 newTofLow = ((i32)tofMmHigh + (i32)tofMmLow) / 2 - 3;
			tofThresHigh = newTofHigh;
			tofThresLow = newTofLow;
		} else {
			tofCalibrationState = 3;
			item->triggerFullRedraw();
		}
		break;
	case 2:
		tofCalibrationState = 255;
		EEPROM.put(EEPROM_POS_TOF_LOW, tofThresLow);
		EEPROM.put(EEPROM_POS_TOF_HIGH, tofThresHigh);
		item->onExit();
		break;
	case 3:
		tofThresHigh = NO_MAG_THRESHOLD;
		tofThresLow = MAG_THRESHOLD;
		EEPROM.put(EEPROM_POS_TOF_LOW, tofThresLow);
		EEPROM.put(EEPROM_POS_TOF_HIGH, tofThresHigh);
		item->onExit();
		break;
	}
	return false;
}

bool onTofCalibrationLeft(MenuItem *item) {
	switch (tofCalibrationState) {
	case 0:
		if (!firstBoot) {
			tofThresHigh = prevTofHigh;
			tofThresLow = prevTofLow;
			item->onExit();
		}
		break;
	case 2:
	case 3:
		startTofCalibration(item);
		item->triggerFullRedraw();
		break;
	}
	return false;
}

void drawTofCalibration(MenuItem *item) {
	if (item->fullRedraw) {
		item->fullRedraw = false;
		tft.fillScreen(ST77XX_BLACK);
		tft.setTextColor(ST77XX_WHITE);
#if HW_VERSION == 1
		SET_DEFAULT_FONT;
		printCentered("Magazine Detection", SCREEN_WIDTH / 2, 0, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_CENTERED);
		switch (tofCalibrationState) {
		case 0:
			printCentered("Insert magazine and press right.", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 2, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			if (!firstBoot) {
				printCentered("Press left to cancel", SCREEN_WIDTH / 2, 50, SCREEN_WIDTH, 1, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			}
			break;
		case 1:
			printCentered("Remove magazine and press right.", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 2, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			break;
		case 2:
			printCentered("Test the calibration by listening to the sounds. Press left to retry, or press right to continue.", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 4, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			break;
		case 3:
			printCentered("Calibration failed. Press left to retry, or press right to skip and use defaults.", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 4, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			break;
		}
#elif HW_VERSION == 2
		tft.setFont(&FreeSansBold12pt7b);
		printCentered("Magazine Detection", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
		tft.setFont(&FreeSans9pt7b);
		switch (tofCalibrationState) {
		case 0:
			printCentered("Insert magazine and press right.", SCREEN_WIDTH / 2, 55, SCREEN_WIDTH, 2, 22, ClipBehavior::PRINT_LAST_LINE_DOTS);
			if (!firstBoot) {
				printCentered("Press left to cancel", SCREEN_WIDTH / 2, 105, SCREEN_WIDTH, 1, 22, ClipBehavior::PRINT_LAST_LINE_DOTS);
			}
			break;
		case 1:
			printCentered("Remove magazine and press right.", SCREEN_WIDTH / 2, 55, SCREEN_WIDTH, 2, 22, ClipBehavior::PRINT_LAST_LINE_DOTS);
			break;
		case 2:
			printCentered("Test the calibration by listening to the sounds. Press left to retry, or press right to continue.", SCREEN_WIDTH / 2, 55, SCREEN_WIDTH, 4, 22, ClipBehavior::PRINT_LAST_LINE_DOTS);
			break;
		case 3:
			printCentered("Calibration failed. Press left to retry, or press right to skip and use defaults.", SCREEN_WIDTH / 2, 55, SCREEN_WIDTH, 3, 22, ClipBehavior::PRINT_LAST_LINE_DOTS);
			break;
		}
#endif
	}
}

bool onTofCalibrationExit(MenuItem *item) {
	if (firstBoot) {
		finishOnboarding();
	}
	return true;
}

#endif // USE_TOF
