#include "global.h"

/*
General layout:

VBat: 2956 -> 22.8V		RotX:  0 -> 0.0°/s
IEsc: 200 -> 17.6A		RotY:  0 -> 0.0°/s
ISol: 20 -> 0.07A		RotZ:  0 -> 0.0°/s
JoyX: 2048 -> 0.0%		AccX:  0 -> 0m/s²
JoyY: 2048 -> 0.0%		AccY:  2048 -> 9.81m/s²
	  180°, 0.0%			AccZ:  0 -> 0m/s²
Trig:  0				R/P/Y: 0°, 0°, 0°
ESC:  34/32/33/35°C		upVel:  0.5m/s
TOF:  70mm -> Not inserted
*/

class DiagText {
public:
	DiagText(i32 x, i32 y) : x(x), y(y), x2(-1) {
		val[0] = '\0';
		val2[0] = '\0';
	}
	DiagText(i32 x, i32 y, i32 x2) : x(x), y(y), x2(x2) {
		val[0] = '\0';
		val2[0] = '\0';
	}
	void updateValue(char *val) {
		if (!enabled)
			return;
		i32 c = strncmp(this->val, val, 32);
		if (c != 0) {
			SET_DEFAULT_FONT;
			tft.setCursor(x, y);
			tft.setTextColor(ST77XX_BLACK);
			tft.print(this->val);
			strncpy(this->val, val, 32);
			tft.setCursor(x, y);
			tft.setTextColor(ST77XX_WHITE);
			tft.print(val);
		}
	}
	void updateValue2(char *val) {
		if (!enabled || !enabled2 || x2 == -1)
			return;
		i32 c = strncmp(this->val2, val, 32);
		if (c != 0) {
			char buf[32];
			snprintf(buf, 32, "-> %s", this->val2);
			SET_DEFAULT_FONT;
			tft.setCursor(x2, y);
			tft.setTextColor(ST77XX_BLACK);
			tft.print(buf);
			strncpy(this->val2, val, 32);
			snprintf(buf, 32, "-> %s", val);
			tft.setCursor(x2, y);
			tft.setTextColor(ST77XX_WHITE);
			tft.print(buf);
		}
	}
	void draw() {
		SET_DEFAULT_FONT;
		if (enabled) {
			tft.setCursor(x, y);
			tft.setTextColor(ST77XX_WHITE);
			tft.print(this->val);
			if (enabled2 && x2 != -1) {
				tft.setCursor(x2, y);
				tft.setTextColor(ST77XX_WHITE);
				char buf[32];
				snprintf(buf, 32, "-> %s", this->val2);
				tft.print(buf);
			}
		}
	}
	void enable() {
		enabled = true;
	}
	void disable() {
		enabled = false;
	}
	void enable2() {
		enabled2 = true;
	}
	void disable2() {
		enabled2 = false;
	}

private:
	const i32 x, y, x2;
	bool enabled = true;
	bool enabled2 = true;
	char val[32];
	char val2[32];
};

#if HW_VERSION == 1
#define XDIM 48
#define XDIM2 96
#define DEGREE_SYMBOL "\xF8"
DiagText textVBat(XDIM, YADVANCE * 2, XDIM2);
DiagText textJoyX(XDIM, YADVANCE * 3, XDIM2);
DiagText textJoyY(XDIM, YADVANCE * 4, XDIM2);
DiagText textAngMag(XDIM, YADVANCE * 5);
DiagText textTrig(XDIM, YADVANCE * 6);
DiagText textESC(XDIM, YADVANCE * 7);
DiagText textTOF(XDIM, YADVANCE * 8, XDIM2);
const char titles[8][10] = {"VBat:", "JoyX:", "JoyY:", "", "Trig:", "ESC:", "TOF:"};
#elif HW_VERSION == 2
i8 page = 1;
#define XDIM 52
#define XDIM2 104
#define DEGREE_SYMBOL "°"
DiagText textVBat(XDIM, YADVANCE * 2, XDIM2);
DiagText textIEsc(XDIM, YADVANCE * 3, XDIM2);
DiagText textISol(XDIM, YADVANCE * 4, XDIM2);
DiagText textJoyX(XDIM, YADVANCE * 5, XDIM2);
DiagText textJoyY(XDIM, YADVANCE * 6, XDIM2);
DiagText textAngMag(XDIM, YADVANCE * 7);
DiagText textTrig(XDIM, YADVANCE * 8);
DiagText textESC(XDIM, YADVANCE * 9);
DiagText textTOF(XDIM, YADVANCE * 10, XDIM2);
DiagText textRotX(XDIM, YADVANCE * 2, XDIM2);
DiagText textRotY(XDIM, YADVANCE * 3, XDIM2);
DiagText textRotZ(XDIM, YADVANCE * 4, XDIM2);
DiagText textAccX(XDIM, YADVANCE * 5, XDIM2);
DiagText textAccY(XDIM, YADVANCE * 6, XDIM2);
DiagText textAccZ(XDIM, YADVANCE * 7, XDIM2);
DiagText textRPY(XDIM, YADVANCE * 8);
DiagText textVVel(XDIM, YADVANCE * 9);
DiagText *textsPage1[9] = {&textVBat, &textIEsc, &textISol, &textJoyX, &textJoyY, &textAngMag, &textTrig, &textESC, &textTOF};
DiagText *textsPage2[8] = {&textRotX, &textRotY, &textRotZ, &textAccX, &textAccY, &textAccZ, &textRPY, &textVVel};
const char titles1[9][10] = {"VBat:", "IEsc:", "ISol:", "JoyX:", "JoyY:", "", "Trig:", "ESC:", "TOF:"};
const char titles2[8][10] = {"RotX:", "RotY:", "RotZ:", "AccX:", "AccY:", "AccZ:", "R/P/Y:", "upVel:"};
#endif

void drawInputDiagnostics(MenuItem *item) {
	static elapsedMillis lastUpdate = 0;
	if (item->fullRedraw) {
		tft.fillScreen(ST77XX_BLACK);
		tft.setTextColor(ST77XX_WHITE);
		SET_DEFAULT_FONT;
		printCentered("Input Diagnostics", SCREEN_WIDTH / 2, 0, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_CENTERED);
		tft.setCursor(0, YADVANCE);
#if HW_VERSION == 1
		tft.print("Hold left to exit");
		for (u8 i = 0; i < 8; i++) {
			tft.setCursor(0, YADVANCE * (i + 2));
			tft.print(titles[i]);
		}
#elif HW_VERSION == 2
		tft.printf("Hold left to exit, hold right for page %d", page == 1 ? 2 : 1);
		for (u8 i = 0; i < 9; i++) {
			if (page == 1) {
				tft.setCursor(0, YADVANCE * (i + 2));
				tft.print(titles1[i]);
				textsPage1[i]->enable();
				textsPage1[i]->draw();
			} else if (page == 2) {
				textsPage1[i]->disable();
			}
		}
		for (u8 i = 0; i < 8; i++) {
			if (page == 2) {
				tft.setCursor(0, YADVANCE * (i + 2));
				tft.print(titles2[i]);
				textsPage2[i]->enable();
				textsPage2[i]->draw();
			} else if (page == 1) {
				textsPage2[i]->disable();
			}
		}
#endif
		item->fullRedraw = false;
	}
	if (lastUpdate >= 100) {
		lastUpdate = 0;
		char buf[32];
		snprintf(buf, 32, "%d", adcConversions[CONV_RESULT_VBAT]);
		textVBat.updateValue(buf);
		snprintf(buf, 32, "%.1fV", fix32(batVoltage).getf32());
		textVBat.updateValue2(buf);

#if HW_VERSION == 2
		snprintf(buf, 32, "%d", adcConversions[CONV_RESULT_IBAT]);
		textIEsc.updateValue(buf);
		snprintf(buf, 32, "%.1fA", fix32(escCurrentAdc).getf32());
		textIEsc.updateValue2(buf);

		snprintf(buf, 32, "%d", adcConversions[CONV_RESULT_ISOLENOID]);
		textISol.updateValue(buf);
		snprintf(buf, 32, "%.2fA", solenoidCurrent.getf32());
		textISol.updateValue2(buf);
#endif

		snprintf(buf, 32, "%d", adcConversions[CONV_RESULT_JOYSTICK_X]);
		textJoyX.updateValue(buf);
		snprintf(buf, 32, "%d%%", joystickXPos);
		textJoyX.updateValue2(buf);

		snprintf(buf, 32, "%d", adcConversions[CONV_RESULT_JOYSTICK_Y]);
		textJoyY.updateValue(buf);
		snprintf(buf, 32, "%d%%", joystickYPos);
		textJoyY.updateValue2(buf);

		snprintf(buf, 32, "%d" DEGREE_SYMBOL ", %d%%", joystickAngleDeg, joystickMagnitude);
		textAngMag.updateValue(buf);

#if USE_TOF
		if (foundTof) {
			textTOF.enable2();
			snprintf(buf, 32, "%dmm", fix32(tofDistance).geti32());
			textTOF.updateValue(buf);
			snprintf(buf, 32, "%s", magPresent ? "Mag" : "No Mag");
			textTOF.updateValue2(buf);
		} else {
			textTOF.disable2();
			snprintf(buf, 32, "Error");
			textTOF.updateValue(buf);
		}
#else
		textTOF.disable2();
		snprintf(buf, 32, "Disabled");
		textTOF.updateValue(buf);
#endif

		snprintf(buf, 32, "%d, %d, %d, %d" DEGREE_SYMBOL "C", escTemp[0], escTemp[1], escTemp[2], escTemp[3]);
		textESC.updateValue(buf);

		snprintf(buf, 32, "%d", triggerState);
		textTrig.updateValue(buf);

#if HW_VERSION == 2
		// all HW2 specific values
		snprintf(buf, 32, "%d", gyroDataRaw[0]);
		textRotX.updateValue(buf);
		snprintf(buf, 32, "%.1f°/s", gyroDataRaw[0] * RAW_TO_RAD_PER_SEC * (float)RAD_TO_DEG);
		textRotX.updateValue2(buf);

		snprintf(buf, 32, "%d", gyroDataRaw[1]);
		textRotY.updateValue(buf);
		snprintf(buf, 32, "%.1f°/s", gyroDataRaw[1] * RAW_TO_RAD_PER_SEC * (float)RAD_TO_DEG);
		textRotY.updateValue2(buf);

		snprintf(buf, 32, "%d", gyroDataRaw[2]);
		textRotZ.updateValue(buf);
		snprintf(buf, 32, "%.1f°/s", gyroDataRaw[2] * RAW_TO_RAD_PER_SEC * (float)RAD_TO_DEG);
		textRotZ.updateValue2(buf);

		snprintf(buf, 32, "%d", accelDataRaw[0]);
		textAccX.updateValue(buf);
		snprintf(buf, 32, "%.1fm/s^2", fix32(RAW_TO_M_PER_SEC2 * accelDataRaw[0]).getf32());
		textAccX.updateValue2(buf);

		snprintf(buf, 32, "%d", accelDataRaw[1]);
		textAccY.updateValue(buf);
		snprintf(buf, 32, "%.1fm/s^2", fix32(RAW_TO_M_PER_SEC2 * accelDataRaw[1]).getf32());
		textAccY.updateValue2(buf);

		snprintf(buf, 32, "%d", accelDataRaw[2]);
		textAccZ.updateValue(buf);
		snprintf(buf, 32, "%.1fm/s^2", fix32(RAW_TO_M_PER_SEC2 * accelDataRaw[2]).getf32());
		textAccZ.updateValue2(buf);

		snprintf(buf, 32, "%.1f°, %.1f°, %.1f°", (roll * FIX_RAD_TO_DEG).getf32(), (pitch * FIX_RAD_TO_DEG).getf32(), (yaw * FIX_RAD_TO_DEG).getf32());
		textRPY.updateValue(buf);

		snprintf(buf, 32, "%.1fm/s", upVel.getf32());
		textVVel.updateValue(buf);
#endif
	}
}

bool onInputDiagLeft(MenuItem *item) {
	if (lastGesture.direction == Direction::LEFT && lastGesture.type == GESTURE_HOLD && lastGesture.duration > 2000) {
		if (item->parent != nullptr) {
			item->onExit();
		} else {
			rp2040.reboot();
		}
	}
	return false;
}

bool onInputDiagRight(MenuItem *item) {
#if HW_VERSION == 2
	static bool allowSwitch = false;
	if (lastGesture.direction == Direction::RIGHT && lastGesture.type == GESTURE_PRESS) allowSwitch = true;
	if (lastGesture.direction == Direction::RIGHT && lastGesture.type == GESTURE_HOLD && lastGesture.duration > 2000 && allowSwitch) {
		page = page == 1 ? 2 : 1;
		item->fullRedraw = true;
		allowSwitch = false;
	}
#endif
	return false;
}