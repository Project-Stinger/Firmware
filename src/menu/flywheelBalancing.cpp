#include "Fonts/FreeSans9pt7b.h"
#include "Fonts/FreeSansBold12pt7b.h"
#include "global.h"

#if HW_VERSION == 1
#define MOTOR_SCREEN_X_1 21
#define MOTOR_SCREEN_X_2 59
#define MOTOR_SCREEN_Y_1 21
#define MOTOR_SCREEN_Y_2 59
#define MOTOR_SCREEN_RADIUS 16
#elif HW_VERSION == 2
#define MOTOR_SCREEN_X_1 35
#define MOTOR_SCREEN_X_2 85
#define MOTOR_SCREEN_Y_1 56
#define MOTOR_SCREEN_Y_2 106
#define MOTOR_SCREEN_RADIUS 22
#endif

i16 testingThrottlePct = 50;
u8 activeSide = 0; // 0 = throttle setting, 1 = spinning
u8 lastSide = 255;

void drawMotor(u8 motorId, u16 color); // from motorRemap.cpp

void drawMotorOutlines(u16 color) {
	tft.drawCircle(MOTOR_SCREEN_X_1, MOTOR_SCREEN_Y_1, MOTOR_SCREEN_RADIUS + 1, color);
	tft.drawCircle(MOTOR_SCREEN_X_1, MOTOR_SCREEN_Y_2, MOTOR_SCREEN_RADIUS + 1, color);
	tft.drawCircle(MOTOR_SCREEN_X_2, MOTOR_SCREEN_Y_1, MOTOR_SCREEN_RADIUS + 1, color);
	tft.drawCircle(MOTOR_SCREEN_X_2, MOTOR_SCREEN_Y_2, MOTOR_SCREEN_RADIUS + 1, color);
}

bool enterBalancing(MenuItem *item) {
	activeSide = 0;
	lastSide = 255;
	menuOverrideEsc[0] = 0;
	menuOverrideEsc[1] = 0;
	menuOverrideEsc[2] = 0;
	menuOverrideEsc[3] = 0;
	return true;
}

bool balancingLoop(MenuItem *item) {
	// Display: Move the joystick around to test the layout, hold right/left to save/discard the new layout.
	bool redraw = item->fullRedraw;
	item->fullRedraw = false;
	if (redraw) {
		tft.fillScreen(ST77XX_BLACK);
		tft.setTextColor(ST77XX_WHITE);
		SET_DEFAULT_FONT;
#if HW_VERSION == 1
		printCentered("Pull trigger to switch between throttle and spinning. Hold left to exit.", SCREEN_WIDTH * 3 / 4, 0, SCREEN_WIDTH / 2, 7, YADVANCE, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
		printCentered("Pull the trigger to switch between throttle setting and spinning. Hold left to exit.", SCREEN_WIDTH * 3 / 4, 35, SCREEN_WIDTH / 2, 5, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
		tft.setFont(&FreeSansBold12pt7b);
		printCentered("Flywheel Tester", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
#endif
	}
	if (triggerUpdateFlag) {
		triggerUpdateFlag = false;
		if (triggerState)
			activeSide = 1 - activeSide;
	}
	if (activeSide == 0) {
		bool drawValue = false;
		if (gestureUpdated) {
			gestureUpdated = false;
			if (lastGesture.type == GESTURE_PRESS || lastGesture.type == GESTURE_HOLD) {
				if (lastGesture.direction == Direction::UP) {
					testingThrottlePct++;
					if (testingThrottlePct > 80) testingThrottlePct = 80;
					drawValue = true;
				} else if (lastGesture.direction == Direction::DOWN) {
					testingThrottlePct--;
					if (testingThrottlePct < 10) testingThrottlePct = 10;
					drawValue = true;
				} else if (lastGesture.direction == Direction::LEFT && lastGesture.duration > 1500) {
					item->onExit();
				}
			}
		}
		if (redraw || lastSide != activeSide) {
			drawMotorOutlines(tft.color565(150, 150, 150));
			drawValue = true;
#if HW_VERSION == 1
			tft.drawRect(SCREEN_WIDTH * 3 / 4 - 16, 60, 32, 14, ST77XX_WHITE);
#elif HW_VERSION == 2
			tft.drawRect(SCREEN_WIDTH * 3 / 4 - 20, 114, 40, 18, ST77XX_WHITE);
#endif
		}
		if (drawValue) {
			char buf[5];
			sprintf(buf, "%3d%%", testingThrottlePct);
			tft.setTextColor(ST77XX_WHITE);
			SET_DEFAULT_FONT;
#if HW_VERSION == 1
			tft.fillRect(SCREEN_WIDTH * 3 / 4 - 15, 61, 30, 12, ST77XX_BLACK);
			printCentered(buf, SCREEN_WIDTH * 3 / 4, 63, 30, 1, YADVANCE, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
			tft.fillRect(SCREEN_WIDTH * 3 / 4 - 19, 115, 38, 16, ST77XX_BLACK);
			printCentered(buf, SCREEN_WIDTH * 3 / 4, 117, 38, 1, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
		}
	} else if (activeSide == 1) {
		if (redraw || activeSide != lastSide) {
			drawMotorOutlines(ST77XX_WHITE);
			char buf[5];
			sprintf(buf, "%3d%%", testingThrottlePct);
			tft.setTextColor(ST77XX_WHITE);
			SET_DEFAULT_FONT;
#if HW_VERSION == 1
			tft.fillRect(SCREEN_WIDTH * 3 / 4 - 16, 60, 32, 14, ST77XX_BLACK);
			printCentered(buf, SCREEN_WIDTH * 3 / 4, 63, 30, 1, YADVANCE, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
			tft.fillRect(SCREEN_WIDTH * 3 / 4 - 20, 114, 40, 18, ST77XX_BLACK);
			printCentered(buf, SCREEN_WIDTH * 3 / 4, 117, 38, 1, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
		}
		if (gestureUpdated) {
			gestureUpdated = false;
			static u8 lastEsc = 0;
			if (lastGesture.type == GESTURE_PRESS) {
				drawMotor(lastEsc, ST77XX_BLACK);
				switch (lastGesture.direction) {
				case Direction::UP_LEFT:
					drawMotor(0, tft.color565(150, 150, 150));
					lastEsc = 0;
					menuOverrideEsc[0] = testingThrottlePct * 20;
					menuOverrideEsc[1] = 0;
					menuOverrideEsc[2] = 0;
					menuOverrideEsc[3] = 0;
					break;
				case Direction::DOWN_LEFT:
					drawMotor(1, tft.color565(150, 150, 150));
					lastEsc = 1;
					menuOverrideEsc[0] = 0;
					menuOverrideEsc[1] = testingThrottlePct * 20;
					menuOverrideEsc[2] = 0;
					menuOverrideEsc[3] = 0;
					break;
				case Direction::UP_RIGHT:
					drawMotor(2, tft.color565(150, 150, 150));
					lastEsc = 2;
					menuOverrideEsc[0] = 0;
					menuOverrideEsc[1] = 0;
					menuOverrideEsc[2] = testingThrottlePct * 20;
					menuOverrideEsc[3] = 0;
					break;
				case Direction::DOWN_RIGHT:
					drawMotor(3, tft.color565(150, 150, 150));
					lastEsc = 3;
					menuOverrideEsc[0] = 0;
					menuOverrideEsc[1] = 0;
					menuOverrideEsc[2] = 0;
					menuOverrideEsc[3] = testingThrottlePct * 20;
					break;
				default:
					menuOverrideEsc[0] = 0;
					menuOverrideEsc[1] = 0;
					menuOverrideEsc[2] = 0;
					menuOverrideEsc[3] = 0;
					break;
				}
			} else if (lastGesture.type == GESTURE_RELEASE) {
				drawMotor(lastEsc, ST77XX_BLACK);
				menuOverrideEsc[0] = 0;
				menuOverrideEsc[1] = 0;
				menuOverrideEsc[2] = 0;
				menuOverrideEsc[3] = 0;
			} else if (lastGesture.type == GESTURE_HOLD) {
				if (lastGesture.direction == Direction::LEFT && lastGesture.duration > 1500) {
					item->onExit();
				}
			}
		}
	}
	menuOverrideTimer = 0;
	lastSide = activeSide;
	return false;
}