#include "../global.h"
#include "Fonts/FreeSans9pt7b.h"
#include "Fonts/FreeSansBold12pt7b.h"

u8 joystickCalState = JOYSTICK_CAL_DONE;
i16 joystickCalX[3] = {512, 200, 824}; // center, min, max
i16 joystickCalY[3] = {512, 200, 824}; // center, min, max
PT1 joystickXPosRaw(JOYSTICK_LPF_CUTOFF_HZ, PID_RATE);
PT1 joystickYPosRaw(JOYSTICK_LPF_CUTOFF_HZ, PID_RATE);
i16 joystickCalXNew[3] = {0};
i16 joystickCalYNew[3] = {0};
i32 joystickXPos = 0; // -100 (left) to 100 (right)
i32 joystickYPos = 0; // -100 (down) to 100 (up)
fix32 joystickAngle = 0; // 0 to 2PI, 0 = left, PI/2 = down, PI = right, 3PI/2 = up
fix32 joystickTravelAngle = 0; // angle travelled since start of gesture, ccw up
fix32 joystickGestureLastAngle = 0; // angle at start of gesture, 0 to 2PI, 0 = left, PI/2 = down, PI = right, 3PI/2 = up
i16 joystickAngleDeg = 0; // 0 to 360, 0 = left, 90 = down, 180 = right, 270 = up
i16 joystickTravelAngleDeg = 0; // angle travelled since start of gesture, ccw up
i32 joystickMagnitude = 0; // 0 to 100
u8 lastJoystickCalDrawState = 0xFF;
Gesture lastGesture = {
	.angle = 0,
	.duration = 0,
	.count = 0,
	.direction = Direction::LEFT,
	.type = GESTURE_RELEASE,
	.prevType = GESTURE_PRESS,
};
i16 joystickRotationTicks = 0;
volatile bool gestureUpdated = false;
elapsedMillis joystickCalibrationTimer = 0;
const fix32 ANGLE_SCHMITT_BARRIER = JOYSTICK_SCHMITT_BARRIER_DEG * DEG_TO_RAD;

bool startJcal(MenuItem *item) {
	if (tournamentMode) return false;
	if (item == nullptr) operationState = STATE_JOYSTICK_CAL;
	joystickCalState = JOYSTICK_CAL_START;
	joystickCalibrationTimer = 0;
	return true;
}

void applyCalibration(fix32 xRaw, fix32 yRaw, i32 &calX, i32 &calY, fix32 &angle, i32 &magnitude) {
	i16 *currentXCal = (joystickCalState == JOYSTICK_CAL_CONFIRM || joystickCalState == JOYSTICK_CAL_LIMITS) ? joystickCalXNew : joystickCalX;
	i16 *currentYCal = (joystickCalState == JOYSTICK_CAL_CONFIRM || joystickCalState == JOYSTICK_CAL_LIMITS) ? joystickCalYNew : joystickCalY;
	if (xRaw < currentXCal[1]) xRaw = currentXCal[1];
	if (xRaw > currentXCal[2]) xRaw = currentXCal[2];
	if (yRaw < currentYCal[1]) yRaw = currentYCal[1];
	if (yRaw > currentYCal[2]) yRaw = currentYCal[2];
	if (xRaw > (i32)currentXCal[0]) {
		calX = (xRaw.geti32() - currentXCal[0]) * 100 / (currentXCal[2] - currentXCal[0]);
	} else {
		calX = (xRaw.geti32() - currentXCal[0]) * 100 / (currentXCal[0] - currentXCal[1]);
	}
	if (yRaw > (i32)currentYCal[0]) {
		calY = (yRaw.geti32() - currentYCal[0]) * 100 / (currentYCal[2] - currentYCal[0]);
	} else {
		calY = (yRaw.geti32() - currentYCal[0]) * 100 / (currentYCal[0] - currentYCal[1]);
	}
	angle = atan2f(calY, calX) + (float)M_PI;
	magnitude = min(sqrtf(calX * calX + calY * calY), 100);
}

void putJoystickValsInEeprom() {
	EEPROM.put(EEPROM_POS_JCAL_CENTER_X, joystickCalX[0]);
	EEPROM.put(EEPROM_POS_JCAL_MIN_X, joystickCalX[1]);
	EEPROM.put(EEPROM_POS_JCAL_MAX_X, joystickCalX[2]);
	EEPROM.put(EEPROM_POS_JCAL_CENTER_Y, joystickCalY[0]);
	EEPROM.put(EEPROM_POS_JCAL_MIN_Y, joystickCalY[1]);
	EEPROM.put(EEPROM_POS_JCAL_MAX_Y, joystickCalY[2]);
}

void joystickInit(bool fromReboot) {
	if (!firstBoot) {
		EEPROM.get(EEPROM_POS_JCAL_CENTER_X, joystickCalX[0]);
		EEPROM.get(EEPROM_POS_JCAL_MIN_X, joystickCalX[1]);
		EEPROM.get(EEPROM_POS_JCAL_MAX_X, joystickCalX[2]);
		EEPROM.get(EEPROM_POS_JCAL_CENTER_Y, joystickCalY[0]);
		EEPROM.get(EEPROM_POS_JCAL_MIN_Y, joystickCalY[1]);
		EEPROM.get(EEPROM_POS_JCAL_MAX_Y, joystickCalY[2]);
	}
	if (fromReboot) {
		// start in the center so that there are no artifacts when booting up
		joystickXPosRaw.set(joystickCalX[0]);
		joystickYPosRaw.set(joystickCalY[0]);
	}
}

void joystickLoop() {
	static elapsedMillis gestureTimer;
	static fix32 lastAngle = 0;
	static fix32 lastMagnitude = 0;
	static bool inCenter = true;
	static fix32 angleThresLow = 0;
	static fix32 angleThresHigh = 0;
	i32 jX, jY;
	jY = 4096 - adcConversions[CONV_RESULT_JOYSTICK_Y];
	jX = adcConversions[CONV_RESULT_JOYSTICK_X];
	joystickXPosRaw.update(jX >> 2);
	joystickYPosRaw.update(jY >> 2);
	applyCalibration((fix32)joystickXPosRaw, (fix32)joystickYPosRaw, joystickXPos, joystickYPos, joystickAngle, joystickMagnitude);
	if (joystickMagnitude > 100) joystickMagnitude = 100;
	joystickAngleDeg = (joystickAngle * FIX_RAD_TO_DEG).geti32();
	if (inCenter && joystickMagnitude > GESTURE_CENTER_LARGE_PCT) {
		// moved out of center, trigger press gesture
		inCenter = false;
		gestureTimer = 0;
		Direction newDir = Direction::LEFT;
		if (joystickAngle > FIX_PI_8)
			newDir = Direction::DOWN_LEFT;
		if (joystickAngle > FIX_PI_8 * 3)
			newDir = Direction::DOWN;
		if (joystickAngle > FIX_PI_8 * 5)
			newDir = Direction::DOWN_RIGHT;
		if (joystickAngle > FIX_PI_8 * 7)
			newDir = Direction::RIGHT;
		if (joystickAngle > FIX_PI_8 * 9)
			newDir = Direction::UP_RIGHT;
		if (joystickAngle > FIX_PI_8 * 11)
			newDir = Direction::UP;
		if (joystickAngle > FIX_PI_8 * 13)
			newDir = Direction::UP_LEFT;
		if (joystickAngle > FIX_PI_8 * 15)
			newDir = Direction::LEFT;
		angleThresHigh = FIX_PI_4 * (i32)newDir + FIX_PI_8 + ANGLE_SCHMITT_BARRIER;
		angleThresLow = FIX_PI_4 * (i32)newDir - FIX_PI_8 - ANGLE_SCHMITT_BARRIER;
		if (angleThresLow < 0) angleThresLow += FIX_2PI;
		Gesture newGesture = {
			.angle = (i16)(joystickAngle * FIX_RAD_TO_DEG).geti32(),
			.duration = 0,
			.count = 1,
			.direction = newDir,
			.type = GESTURE_PRESS,
			.prevType = lastGesture.type,
		};
		gestureUpdated = false;
		lastGesture = newGesture;
		joystickGestureLastAngle = joystickAngle;
		joystickTravelAngle = 0;
		joystickTravelAngleDeg = 0;
		joystickRotationTicks = 0;
		inactivityTimer = 0;
		gestureUpdated = true;
	} else if (!inCenter) {
		if (joystickMagnitude < GESTURE_CENTER_SMALL_PCT) {
			// moved back to center, trigger release gesture
			inCenter = true;
			Gesture newGesture = {
				.angle = (i16)(lastAngle * FIX_RAD_TO_DEG).geti32(),
				.duration = gestureTimer,
				.count = 1,
				.direction = lastGesture.direction,
				.type = GESTURE_RELEASE,
				.prevType = lastGesture.type,
			};
			gestureUpdated = false;
			lastGesture = newGesture;
			inactivityTimer = 0;
			gestureUpdated = true;
		} else {
			const i16 degreesPerTick = 32 - rotationTickSensitivity * 8;
			static i16 highAngleThresDeg = 0;
			static i16 lowAngleThresDeg = 0;
			if (joystickTravelAngle == 0) {
				// first time, set the thresholds
				highAngleThresDeg = degreesPerTick / 2 + 4;
				lowAngleThresDeg = -degreesPerTick / 2 - 4;
			}
			fix32 diff = joystickAngle - joystickGestureLastAngle;
			joystickGestureLastAngle = joystickAngle;
			if (diff > FIX_PI) diff -= FIX_2PI;
			if (diff <= -FIX_PI) diff += FIX_2PI;
			joystickTravelAngle += diff;
			joystickTravelAngleDeg = (joystickTravelAngle * FIX_RAD_TO_DEG).geti32();
			while (joystickTravelAngleDeg > highAngleThresDeg) {
				joystickRotationTicks++;
				highAngleThresDeg += degreesPerTick;
				lowAngleThresDeg += degreesPerTick;
			}
			while (joystickTravelAngleDeg < lowAngleThresDeg) {
				joystickRotationTicks--;
				highAngleThresDeg -= degreesPerTick;
				lowAngleThresDeg -= degreesPerTick;
			}
			if (((angleThresLow < angleThresHigh) && (joystickAngle > angleThresHigh || joystickAngle < angleThresLow)) ||
				((angleThresLow >= angleThresHigh) && (joystickAngle > angleThresHigh && joystickAngle < angleThresLow))) {
				// moved to a new direction, trigger another press gesture
				gestureTimer = 0;
				Direction newDir = Direction::LEFT;
				if (joystickAngle > FIX_PI_8)
					newDir = Direction::DOWN_LEFT;
				if (joystickAngle > FIX_PI_8 * 3)
					newDir = Direction::DOWN;
				if (joystickAngle > FIX_PI_8 * 5)
					newDir = Direction::DOWN_RIGHT;
				if (joystickAngle > FIX_PI_8 * 7)
					newDir = Direction::RIGHT;
				if (joystickAngle > FIX_PI_8 * 9)
					newDir = Direction::UP_RIGHT;
				if (joystickAngle > FIX_PI_8 * 11)
					newDir = Direction::UP;
				if (joystickAngle > FIX_PI_8 * 13)
					newDir = Direction::UP_LEFT;
				if (joystickAngle > FIX_PI_8 * 15)
					newDir = Direction::LEFT;
				angleThresHigh = FIX_PI_4 * (i32)newDir + FIX_PI_8 + ANGLE_SCHMITT_BARRIER;
				angleThresLow = FIX_PI_4 * (i32)newDir - FIX_PI_8 - ANGLE_SCHMITT_BARRIER;
				if (angleThresLow < 0) angleThresLow += FIX_2PI;
				gestureUpdated = false;
				Gesture newGesture = {
					.angle = (i16)(joystickAngle * FIX_RAD_TO_DEG).geti32(),
					.duration = 0,
					.count = 1,
					.direction = newDir,
					.type = GESTURE_PRESS,
					.prevType = lastGesture.type,
				};
				lastGesture = newGesture;
				gestureUpdated = true;
				inactivityTimer = 0;
			} else if ((lastGesture.type == GESTURE_PRESS && gestureTimer >= GESTURE_INIT_WAIT) ||
					   (lastGesture.type == GESTURE_HOLD && gestureTimer >= GESTURE_REPEAT_WAIT)) {
				// hold gesture after 700ms, and then repeatedly every 80ms
				gestureUpdated = false;
				lastGesture.count++;
				lastGesture.angle = (i16)(joystickAngle * FIX_RAD_TO_DEG).geti32();
				lastGesture.duration += gestureTimer;
				lastGesture.type = GESTURE_HOLD;
				lastGesture.prevType = lastGesture.type;
				gestureUpdated = true;
				inactivityTimer = 0;
				gestureTimer = 0;
			}
		}
	}
	lastAngle = joystickAngle;
	lastMagnitude = joystickMagnitude;
}

bool calJoystick(MenuItem *item) {
	if (joystickCalibrationTimer > 20000 && item != nullptr) {
		joystickCalState = JOYSTICK_CAL_ABORT;
#if HW_VERSION == 2
		ledSetMode(LED_MODE::OFF, LIGHT_ID::MENU);
#endif
		joystickCalibrationTimer = 0;
	};
	switch (joystickCalState) {
	case JOYSTICK_CAL_CENTER_TOP:
		// move joystick up and let it snap back to the center
		// pull trigger to confirm
		if (triggerUpdateFlag && triggerState && joystickCalibrationTimer > 700) {
			joystickCalYNew[0] = ((fix32)joystickYPosRaw).geti32();
			DEBUG_PRINTF("jcal: top %d\n", joystickCalYNew[0]);
			joystickCalState = JOYSTICK_CAL_CENTER_BOTTOM;
			joystickCalibrationTimer = 0;
			triggerUpdateFlag = false;
		}
		break;
	case JOYSTICK_CAL_CENTER_BOTTOM:
		// move joystick down and let it snap back to the center
		// pull trigger to confirm
		if (triggerUpdateFlag && triggerState && joystickCalibrationTimer > 700) {
			joystickCalYNew[0] = (joystickCalYNew[0] + ((fix32)joystickYPosRaw).geti32()) / 2;
			DEBUG_PRINTF("jcal: bottom %d\n", fix32(joystickYPosRaw).geti32());
			DEBUG_PRINTF("jcal: y center %d\n", joystickCalYNew[0]);
			joystickCalState = JOYSTICK_CAL_CENTER_LEFT;
			joystickCalibrationTimer = 0;
			triggerUpdateFlag = false;
		}
		break;
	case JOYSTICK_CAL_CENTER_LEFT:
		// move joystick left and let it snap back to the center
		// pull trigger to confirm
		if (triggerUpdateFlag && triggerState && joystickCalibrationTimer > 700) {
			joystickCalXNew[0] = ((fix32)joystickXPosRaw).geti32();
			DEBUG_PRINTF("jcal: left %d\n", joystickCalXNew[0]);
			joystickCalState = JOYSTICK_CAL_CENTER_RIGHT;
			joystickCalibrationTimer = 0;
			triggerUpdateFlag = false;
		}
		break;
	case JOYSTICK_CAL_CENTER_RIGHT:
		// move joystick right and let it snap back to the center
		// pull trigger to confirm
		if (triggerUpdateFlag && triggerState && joystickCalibrationTimer > 700) {
			joystickCalXNew[0] = (joystickCalXNew[0] + ((fix32)joystickXPosRaw).geti32()) / 2;
			DEBUG_PRINTF("jcal: right %d\n", fix32(joystickXPosRaw).geti32());
			DEBUG_PRINTF("jcal: x center %d\n", joystickCalXNew[0]);
			joystickCalXNew[1] = joystickCalXNew[0];
			joystickCalXNew[2] = joystickCalXNew[0];
			joystickCalYNew[1] = joystickCalYNew[0];
			joystickCalYNew[2] = joystickCalYNew[0];
			joystickCalState = JOYSTICK_CAL_LIMITS;
			joystickCalibrationTimer = 0;
			triggerUpdateFlag = false;
		}
		break;
	case JOYSTICK_CAL_LIMITS:
		// circulate joystick to find min and max values
		// pull trigger to confirm
		if (triggerUpdateFlag && triggerState && joystickCalibrationTimer > 2000) {
			joystickCalState = JOYSTICK_CAL_CONFIRM;
			joystickCalibrationTimer = 0;
			triggerUpdateFlag = false;
		} else {
			i16 rawX = ((fix32)joystickXPosRaw).geti32();
			i16 rawY = ((fix32)joystickYPosRaw).geti32();
			if (rawX < joystickCalXNew[1]) joystickCalXNew[1] = rawX;
			if (rawX > joystickCalXNew[2]) joystickCalXNew[2] = rawX;
			if (rawY < joystickCalYNew[1]) joystickCalYNew[1] = rawY;
			if (rawY > joystickCalYNew[2]) joystickCalYNew[2] = rawY;
#if HW_VERSION == 2
			u8 r, g, b;
			i32 m = joystickMagnitude - 20;
			if (m < 0) m = 0;
			m = m * 128 / 80;
			hslToRgb(255 * joystickAngleDeg / 360, 255, m, r, g, b);
			ledSetMode(LED_MODE::STATIC, LIGHT_ID::MENU, 0, r, g, b);
#endif
		}
		break;
	case JOYSTICK_CAL_CONFIRM: {
		// move joystick around, check if calibration is correct
		// pull trigger to confirm
		if (triggerUpdateFlag && triggerState && joystickCalibrationTimer > 2000) {
			memcpy(joystickCalX, joystickCalXNew, sizeof(joystickCalX));
			memcpy(joystickCalY, joystickCalYNew, sizeof(joystickCalY));
			if (item == nullptr) {
				putJoystickValsInEeprom();
				if (!firstBoot) EEPROM.commit();
			}
			joystickCalState = JOYSTICK_CAL_DONE;
			joystickCalibrationTimer = 0;
			triggerUpdateFlag = false;
		}
#if HW_VERSION == 2
		u8 r, g, b;
		i32 m = joystickMagnitude - 20;
		if (m < 0) m = 0;
		m = m * 128 / 80;
		hslToRgb(255 * joystickAngleDeg / 360, 255, m, r, g, b);
		ledSetMode(LED_MODE::STATIC, LIGHT_ID::MENU, 0, r, g, b);
#endif
	} break;
	case JOYSTICK_CAL_DONE:
		// joystick calibration done
		if (item == nullptr) {
#if HW_VERSION == 2
			releaseLightId(LIGHT_ID::MENU);
#endif
			if (firstBoot) {
				MenuItem *remap = mainMenu->search("remapMotors");
				if (remap != nullptr) {
					remap->onEnter();
					openedMenu = remap;
					menuOverrideTimer = 100;
					operationState = STATE_MENU;
					return false;
				}
			}
			return true;
		}
		item->onExit();
		break;
	case JOYSTICK_CAL_ABORT:
		// joystick calibration aborted
		if (joystickCalibrationTimer > 2000)
			item->onExit();
		break;
	}
	// return is interpreted as follows
	// if item is nullptr, returning true means we have finished the calibration process (which we don't want)
	// if item is not nullptr, returning true means we want to draw the calibration screen and interpret the joystick actions with the standard procedure (which we don't want, but we want to draw the screen)
	if (item != nullptr)
		drawJoystickCalibration(item);
	return false;
}

void drawJoystickCalibration(MenuItem *item) {
	bool firstTime = false;
	if (item != nullptr) {
		if (item->fullRedraw) {
			item->fullRedraw = false;
			firstTime = true;
		}
	}
	if (lastJoystickCalDrawState != joystickCalState) {
		lastJoystickCalDrawState = joystickCalState;
		firstTime = true;
	}
	switch (joystickCalState) {
	case JOYSTICK_CAL_CENTER_TOP:
		if (firstTime) {
			tft.fillScreen(ST77XX_BLACK);
			tft.setTextColor(ST77XX_WHITE);
#if HW_VERSION == 1
			SET_DEFAULT_FONT;
			printCentered("Joystick Calibration", SCREEN_WIDTH / 2, 0, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			printCentered("Move the joystick up and let it snap back to the center.", SCREEN_WIDTH / 2, 12, SCREEN_WIDTH, 4, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Pull the trigger to continue.", SCREEN_WIDTH / 2, 52, SCREEN_WIDTH, 2, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
			tft.setFont(&FreeSansBold12pt7b);
			printCentered("Joystick Calibration", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			tft.setFont(&FreeSans9pt7b);
			printCentered("Move the joystick up and let it snap back to the center.", SCREEN_WIDTH / 2, 57, SCREEN_WIDTH, 4, 22, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Pull the trigger to continue.", SCREEN_WIDTH / 2, 120, SCREEN_WIDTH, 2, 18, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
		}
		break;
	case JOYSTICK_CAL_CENTER_BOTTOM:
		if (firstTime) {
			tft.fillScreen(ST77XX_BLACK);
			tft.setTextColor(ST77XX_WHITE);
#if HW_VERSION == 1
			SET_DEFAULT_FONT;
			printCentered("Joystick Calibration", SCREEN_WIDTH / 2, 0, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			printCentered("Move the joystick down and let it snap back to the center.", SCREEN_WIDTH / 2, 12, SCREEN_WIDTH, 4, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Pull the trigger to continue.", SCREEN_WIDTH / 2, 52, SCREEN_WIDTH, 2, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
			tft.setFont(&FreeSansBold12pt7b);
			printCentered("Joystick Calibration", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			tft.setFont(&FreeSans9pt7b);
			printCentered("Move the joystick down and let it snap back to the center.", SCREEN_WIDTH / 2, 57, SCREEN_WIDTH, 4, 22, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Pull the trigger to continue.", SCREEN_WIDTH / 2, 120, SCREEN_WIDTH, 2, 18, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
		}
		break;
	case JOYSTICK_CAL_CENTER_LEFT:
		if (firstTime) {
			tft.fillScreen(ST77XX_BLACK);
			tft.setTextColor(ST77XX_WHITE);
#if HW_VERSION == 1
			SET_DEFAULT_FONT;
			printCentered("Joystick Calibration", SCREEN_WIDTH / 2, 0, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			printCentered("Move the joystick left and let it snap back to the center.", SCREEN_WIDTH / 2, 12, SCREEN_WIDTH, 4, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Pull the trigger to continue.", SCREEN_WIDTH / 2, 52, SCREEN_WIDTH, 2, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
			tft.setFont(&FreeSansBold12pt7b);
			printCentered("Joystick Calibration", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			tft.setFont(&FreeSans9pt7b);
			printCentered("Move the joystick left and let it snap back to the center.", SCREEN_WIDTH / 2, 57, SCREEN_WIDTH, 4, 22, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Pull the trigger to continue.", SCREEN_WIDTH / 2, 120, SCREEN_WIDTH, 2, 18, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
		}
		break;
	case JOYSTICK_CAL_CENTER_RIGHT:
		if (firstTime) {
			tft.fillScreen(ST77XX_BLACK);
			tft.setTextColor(ST77XX_WHITE);
#if HW_VERSION == 1
			SET_DEFAULT_FONT;
			printCentered("Joystick Calibration", SCREEN_WIDTH / 2, 0, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			printCentered("Move the joystick right and let it snap back to the center.", SCREEN_WIDTH / 2, 12, SCREEN_WIDTH, 4, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Pull the trigger to continue.", SCREEN_WIDTH / 2, 52, SCREEN_WIDTH, 2, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
			tft.setFont(&FreeSansBold12pt7b);
			printCentered("Joystick Calibration", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			tft.setFont(&FreeSans9pt7b);
			printCentered("Move the joystick right and let it snap back to the center.", SCREEN_WIDTH / 2, 57, SCREEN_WIDTH, 4, 22, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Pull the trigger to continue.", SCREEN_WIDTH / 2, 120, SCREEN_WIDTH, 2, 18, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
		}
		break;
	case JOYSTICK_CAL_LIMITS: {
		static elapsedMillis updateTimer = 100;
		static u8 lastX = 160, lastY = 80;
		if (firstTime) {
			tft.fillScreen(ST77XX_BLACK);
			tft.setTextColor(ST77XX_WHITE);
#if HW_VERSION == 1
			SET_DEFAULT_FONT;
			printCentered("Circulate the joystick gently to find the min and max values.", SCREEN_WIDTH * 3 / 8, 3, SCREEN_WIDTH * 3 / 4, 6, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Pull the trigger to continue", SCREEN_WIDTH * 3 / 8, 60, 120, 2, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
			tft.setFont(&FreeSansBold12pt7b);
			printCentered("Joystick Calibration", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			tft.setFont(&FreeSans9pt7b);
			printCentered("Circulate the joystick gently to find the min and max values.", SCREEN_WIDTH / 2, 44, SCREEN_WIDTH, 6, 19, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Pull the trigger to continue", SCREEN_WIDTH / 4, 95, SCREEN_WIDTH / 2, 2, 19, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
		}
		if (updateTimer >= 17) {
			updateTimer = 0;
#if HW_VERSION == 1
			// X: 140 +- 15, Y: 40 +- 15
			u8 newX = 140 - (joystickMagnitude * 15 / 100) * cosf(joystickAngle.getf32()) + .5f;
			u8 newY = 40 + (joystickMagnitude * 15 / 100) * sinf(joystickAngle.getf32()) + .5f;
			if (newX != lastX || newY != lastY) {
				tft.fillCircle(lastX, lastY, 2, ST77XX_BLACK);
				tft.fillCircle(newX, newY, 2, ST77XX_WHITE);
				tft.drawCircle(140, 40, 15, tft.color565(192, 192, 192));
				lastX = newX;
				lastY = newY;
			}
#elif HW_VERSION == 2
			// X: 180 +- 29, Y: 101 +- 29
			u8 newX = 180 - (joystickMagnitude * 29 / 100) * cosf(joystickAngle.getf32()) + .5f;
			u8 newY = 101 + (joystickMagnitude * 29 / 100) * sinf(joystickAngle.getf32()) + .5f;
			if (newX != lastX || newY != lastY) {
				tft.fillCircle(lastX, lastY, 3, ST77XX_BLACK);
				tft.fillCircle(newX, newY, 3, ST77XX_WHITE);
				tft.drawCircle(180, 101, 29, tft.color565(192, 192, 192));
				lastX = newX;
				lastY = newY;
			}
#endif
		}
	} break;
	case JOYSTICK_CAL_CONFIRM: {
		static elapsedMillis updateTimer = 100;
		static u8 lastX = 160, lastY = 80;
		if (firstTime) {
			tft.fillScreen(ST77XX_BLACK);
			tft.setTextColor(ST77XX_WHITE);
#if HW_VERSION == 1
			SET_DEFAULT_FONT;
			printCentered("Test the joystick calibration.", 51, 3, 102, 6, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Pull the trigger to continue", 51, 60, 102, 2, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
			tft.setFont(&FreeSansBold12pt7b);
			printCentered("Joystick Calibration", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			tft.setFont(&FreeSans9pt7b);
			printCentered("Test the joystick calibration.", SCREEN_WIDTH / 4, 44, SCREEN_WIDTH / 2, 6, 19, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Pull the trigger to continue", SCREEN_WIDTH / 4, 107, SCREEN_WIDTH / 2, 2, 19, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
		}
		if (updateTimer >= 17) {
			updateTimer = 0;
#if HW_VERSION == 1
			// X: 131 +- 25, Y: 40 +- 25
			u8 newX = 131 - (joystickMagnitude * 25 / 100) * cosf(joystickAngle.getf32()) + .5f;
			u8 newY = 40 + (joystickMagnitude * 25 / 100) * sinf(joystickAngle.getf32()) + .5f;
			if (newX != lastX || newY != lastY) {
				tft.fillCircle(lastX, lastY, 3, ST77XX_BLACK);
				tft.fillCircle(newX, newY, 3, ST77XX_WHITE);
				tft.drawCircle(131, 40, 25, tft.color565(192, 192, 192));
				lastX = newX;
				lastY = newY;
			}
#elif HW_VERSION == 2
			// X: 180 +- 45, Y: 80 +- 45
			u8 newX = 180 - (joystickMagnitude * 45 / 100) * cosf(joystickAngle.getf32()) + .5f;
			u8 newY = 80 + (joystickMagnitude * 45 / 100) * sinf(joystickAngle.getf32()) + .5f;
			if (newX != lastX || newY != lastY) {
				tft.fillCircle(lastX, lastY, 4, ST77XX_BLACK);
				tft.fillCircle(newX, newY, 4, ST77XX_WHITE);
				tft.drawCircle(180, 80, 45, tft.color565(192, 192, 192));
				lastX = newX;
				lastY = newY;
			}
#endif
		}
	} break;
	case JOYSTICK_CAL_ABORT: {
		if (firstTime) {
			tft.fillScreen(ST77XX_BLACK);
			tft.setTextColor(ST77XX_WHITE);
#if HW_VERSION == 1
			SET_DEFAULT_FONT;
			printCentered("Joystick Calibration", SCREEN_WIDTH / 2, 0, SCREEN_WIDTH, 1, 8, ClipBehavior::PRINT_LAST_LINE_DOTS);
			printCentered("Calibration aborted.", SCREEN_WIDTH / 2, 30, SCREEN_WIDTH, 2, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
			tft.setFont(&FreeSansBold12pt7b);
			printCentered("Joystick Calibration", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			tft.setFont(&FreeSans9pt7b);
			printCentered("Calibration aborted.", SCREEN_WIDTH / 2, 57, SCREEN_WIDTH, 2, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
		}
	} break;
	}
}
