#include "Fonts/FreeSans9pt7b.h"
#include "Fonts/FreeSansBold12pt7b.h"
#include "global.h"

u8 remapState = 0;
u8 newPinout[4] = {255, 255, 255, 255};
bool calibrateSuccess = false;
enum RemapState : u8 {
	REMAP_START = 0,
	REMAP_ESC1,
	REMAP_ESC2,
	REMAP_ESC3,
	REMAP_ESC4,
	REMAP_CONFIRM,
	REMAP_ERROR
};
static u8 lastRemapState = REMAP_ERROR;
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

bool startRemap(MenuItem *_item) {
	if (tournamentMode) return false;
	remapState = REMAP_START;
	lastRemapState = REMAP_ERROR;
	calibrateSuccess = false;
	volatile u8 newPins[4];
	for (int i = 0; i < 4; i++) {
		newPins[i] = PIN_MOTOR_BASE + i;
		newPinout[i] = 255;
	}
	updateEscPins(newPins);
	return true;
}

bool stopRemap(MenuItem *item) {
	if (calibrateSuccess) {
		saveEscLayout();
		if (firstBoot) {
			MenuItem *tofCalib = mainMenu->search("calibrateTof");
			tofCalib->onEnter();
			if (openedMenu == item) // if tofCalib is skipped, it is indicated with an overwritten openedMenu
				openedMenu = tofCalib;
		}
	} else {
		initEscLayout();
		updateEscPins(escPins);
		if (firstBoot) {
			startRemap(nullptr);
			return false;
		}
	}
	return true;
}

void drawMotor(u8 motorId, u16 color) {
	switch (motorId) {
	case 0:
		tft.fillCircle(MOTOR_SCREEN_X_1, MOTOR_SCREEN_Y_1, MOTOR_SCREEN_RADIUS, color);
		break;
	case 1:
		tft.fillCircle(MOTOR_SCREEN_X_1, MOTOR_SCREEN_Y_2, MOTOR_SCREEN_RADIUS, color);
		break;
	case 2:
		tft.fillCircle(MOTOR_SCREEN_X_2, MOTOR_SCREEN_Y_1, MOTOR_SCREEN_RADIUS, color);
		break;
	case 3:
		tft.fillCircle(MOTOR_SCREEN_X_2, MOTOR_SCREEN_Y_2, MOTOR_SCREEN_RADIUS, color);
		break;
	}
}

bool remapLoop(MenuItem *item) {
	bool newState = false;
	if (lastRemapState != remapState) {
		newState = true;
		lastRemapState = remapState;
	}
	switch (remapState) {
	case REMAP_START: {
		// Display: One motor will spin up at a time. Push the joystick into the direction of the spinning motor. Hold left to discard any changes.
		if (newState || item->fullRedraw) {
			tft.fillScreen(ST77XX_BLACK);
			tft.setTextColor(ST77XX_WHITE);
#if HW_VERSION == 1
			tft.setFont(&FreeSans9pt7b);
			printCentered("Motor mapping", SCREEN_WIDTH / 2, 12, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			tft.setTextWrap(true);
			SET_DEFAULT_FONT;
			tft.setCursor(0, 30);
			tft.print("Move the joystick in the\ndirection of the spinning\nmotor.\nPress right to start.\n");
#elif HW_VERSION == 2
			tft.setFont(&FreeSansBold12pt7b);
			printCentered("Motor mapping", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			tft.setFont(&FreeSans9pt7b);
			tft.setTextWrap(true);
			SET_DEFAULT_FONT;
			tft.setCursor(0, 40);
			tft.print("Move the joystick in the direction\nof the spinning motor.\nPress right to start.\n");
#endif
			if (!firstBoot) tft.print("Hold left to exit.");
		}
		if (gestureUpdated) {
			gestureUpdated = false;
			if (lastGesture.type == GESTURE_PRESS && lastGesture.direction == Direction::RIGHT) {
				remapState = REMAP_ESC1;
				newPinout[0] = 255;
				newPinout[1] = 255;
				newPinout[2] = 255;
				newPinout[3] = 255;
			} else if (lastGesture.type == GESTURE_HOLD && lastGesture.direction == Direction::LEFT && lastGesture.duration > 1500 && !firstBoot) {
				item->onExit();
			}
		}
	} break;
	case REMAP_ESC1:
	case REMAP_ESC2:
	case REMAP_ESC3:
	case REMAP_ESC4: {
		static u8 selectedEsc;
		if ((newState && remapState == REMAP_ESC1) || item->fullRedraw) {
			tft.fillScreen(ST77XX_BLACK);
			tft.setTextColor(ST77XX_WHITE);
#if HW_VERSION == 2
			tft.setFont(&FreeSansBold12pt7b);
			printCentered("Motor mapping", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
			SET_DEFAULT_FONT;
			if (firstBoot)
				printCentered("Which motor is spinning? Hold left to restart.", SCREEN_WIDTH * 3 / 4, HW_VERSION == 1 ? 20 : 50, SCREEN_WIDTH / 2, 8, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			else
				printCentered("Which motor is spinning? Hold left to exit.", SCREEN_WIDTH * 3 / 4, HW_VERSION == 1 ? 20 : 50, SCREEN_WIDTH / 2, 8, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			tft.drawCircle(MOTOR_SCREEN_X_1, MOTOR_SCREEN_Y_1, MOTOR_SCREEN_RADIUS + 1, ST77XX_WHITE);
			tft.drawCircle(MOTOR_SCREEN_X_1, MOTOR_SCREEN_Y_2, MOTOR_SCREEN_RADIUS + 1, ST77XX_WHITE);
			tft.drawCircle(MOTOR_SCREEN_X_2, MOTOR_SCREEN_Y_1, MOTOR_SCREEN_RADIUS + 1, ST77XX_WHITE);
			tft.drawCircle(MOTOR_SCREEN_X_2, MOTOR_SCREEN_Y_2, MOTOR_SCREEN_RADIUS + 1, ST77XX_WHITE);
		}
		if (newState) {
			selectedEsc = 255;
			for (int i = 0; i < 4; i++)
				menuOverrideEsc[i] = 0;
		}
		menuOverrideEsc[remapState - REMAP_ESC1] = minThrottle;
		menuOverrideTimer = 0;
		if (gestureUpdated) {
			gestureUpdated = false;
			if (lastGesture.type == GESTURE_RELEASE && selectedEsc < 4) {
				newPinout[selectedEsc] = PIN_MOTOR_BASE + remapState - REMAP_ESC1;
				menuOverrideEsc[remapState - REMAP_ESC1] = 0;
				remapState++;
				drawMotor(selectedEsc, ST77XX_BLACK);
			} else if (lastGesture.type == GESTURE_PRESS) {
				drawMotor(selectedEsc, ST77XX_BLACK);
				switch (lastGesture.direction) {
				case Direction::UP_LEFT:
					selectedEsc = 0;
					break;
				case Direction::DOWN_LEFT:
					selectedEsc = 1;
					break;
				case Direction::UP_RIGHT:
					selectedEsc = 2;
					break;
				case Direction::DOWN_RIGHT:
					selectedEsc = 3;
					break;
				default:
					selectedEsc = 255;
					break;
				}
				DEBUG_PRINTF("Selected ESC: %d\n", selectedEsc);
				drawMotor(selectedEsc, tft.color565(150, 150, 150));
			} else if (lastGesture.type == GESTURE_HOLD) {
				if (lastGesture.direction == Direction::LEFT && lastGesture.duration > 1500) {
					item->onExit();
				}
			}
		}
	} break;
	case REMAP_CONFIRM: {
		// Display: Move the joystick around to test the layout, hold right/left to save/discard the new layout.
		if (newState) {
			bool fail = false;
			for (int i = 0; i < 4; i++) {
				if (newPinout[i] == 255) {
					fail = true;
					break;
				}
			}
			if (fail) {
				remapState = REMAP_ERROR;
				break;
			}
			updateEscPins(newPinout);
		}
		if (newState || item->fullRedraw) {
			tft.fillScreen(ST77XX_BLACK);
			tft.setTextColor(ST77XX_WHITE);
			SET_DEFAULT_FONT;
#if HW_VERSION == 1
			printCentered("Test the new config.", SCREEN_WIDTH * 3 / 4, 0, SCREEN_WIDTH / 2, 2, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Hold right to confirm", SCREEN_WIDTH * 3 / 4, 25, SCREEN_WIDTH / 2, 2, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Hold left to discard", SCREEN_WIDTH * 3 / 4, 50, SCREEN_WIDTH / 2, 2, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
			printCentered("Test the new config.", SCREEN_WIDTH * 3 / 4, 40, SCREEN_WIDTH / 2, 2, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Hold right to confirm", SCREEN_WIDTH * 3 / 4, 73, SCREEN_WIDTH / 2, 2, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			printCentered("Hold left to discard", SCREEN_WIDTH * 3 / 4, 106, SCREEN_WIDTH / 2, 2, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			tft.setFont(&FreeSansBold12pt7b);
			printCentered("Motor mapping", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
#endif
			tft.drawCircle(MOTOR_SCREEN_X_1, MOTOR_SCREEN_Y_1, MOTOR_SCREEN_RADIUS + 1, ST77XX_WHITE);
			tft.drawCircle(MOTOR_SCREEN_X_1, MOTOR_SCREEN_Y_2, MOTOR_SCREEN_RADIUS + 1, ST77XX_WHITE);
			tft.drawCircle(MOTOR_SCREEN_X_2, MOTOR_SCREEN_Y_1, MOTOR_SCREEN_RADIUS + 1, ST77XX_WHITE);
			tft.drawCircle(MOTOR_SCREEN_X_2, MOTOR_SCREEN_Y_2, MOTOR_SCREEN_RADIUS + 1, ST77XX_WHITE);
		}
		if (gestureUpdated) {
			DEBUG_PRINTF("Gesture: %d %d %d\n", lastGesture.type, lastGesture.direction, lastGesture.duration);
			gestureUpdated = false;
			static u8 lastEsc = 255;
			if (lastGesture.type == GESTURE_PRESS) {
				drawMotor(lastEsc, ST77XX_BLACK);
				switch (lastGesture.direction) {
				case Direction::UP_LEFT:
					lastEsc = 0;
					break;
				case Direction::DOWN_LEFT:
					lastEsc = 1;
					break;
				case Direction::UP_RIGHT:
					lastEsc = 2;
					break;
				case Direction::DOWN_RIGHT:
					lastEsc = 3;
					break;
				default:
					lastEsc = 255;
					break;
				}
				for (int i = 0; i < 4; i++) {
					menuOverrideEsc[i] = i == lastEsc ? minThrottle : 0;
				}
				drawMotor(lastEsc, tft.color565(150, 150, 150));
			} else if (lastGesture.type == GESTURE_RELEASE) {
				drawMotor(lastEsc, ST77XX_BLACK);
				menuOverrideEsc[0] = 0;
				menuOverrideEsc[1] = 0;
				menuOverrideEsc[2] = 0;
				menuOverrideEsc[3] = 0;
			} else if (lastGesture.type == GESTURE_HOLD) {
				if (lastGesture.direction == Direction::LEFT && lastGesture.duration > 1500) {
					item->onExit();
				} else if (lastGesture.direction == Direction::RIGHT && lastGesture.duration > 1500) {
					calibrateSuccess = true;
					item->onExit();
				}
			}
		}
		menuOverrideTimer = 0;
	} break;
	case REMAP_ERROR: {
		// Display: Error, invalid motor config.
		if (newState || item->fullRedraw) {
			tft.fillScreen(ST77XX_BLACK);
			tft.setTextColor(ST77XX_RED);
			tft.setTextWrap(true);
			SET_DEFAULT_FONT;
#if HW_VERSION == 1
			printCentered("Invalid motor config", SCREEN_WIDTH / 2, 3, SCREEN_WIDTH, 1, 8, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			tft.setTextColor(ST77XX_WHITE);
			if (firstBoot)
				printCentered("Retry >", SCREEN_WIDTH / 2, 40, SCREEN_WIDTH, 1, 8, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			else
				printCentered("< Exit   Retry >", SCREEN_WIDTH / 2, 40, SCREEN_WIDTH, 1, 8, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
			printCentered("Invalid motor config", SCREEN_WIDTH / 2, 40, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			tft.setTextColor(ST77XX_WHITE);
			if (firstBoot)
				printCentered("Retry >", SCREEN_WIDTH / 2, 70, SCREEN_WIDTH, 1, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			else
				printCentered("< Exit   Retry >", SCREEN_WIDTH / 2, 70, SCREEN_WIDTH, 1, 9, ClipBehavior::PRINT_LAST_LINE_CENTERED);
			tft.setFont(&FreeSansBold12pt7b);
			printCentered("Motor mapping", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_DOTS);
			DEBUG_PRINTSLN("Invalid motor config, remap aborted.");
#endif
		}
		if (gestureUpdated) {
			gestureUpdated = false;
			if (lastGesture.type == GESTURE_PRESS) {
				if (lastGesture.direction == Direction::RIGHT) {
					stopRemap(nullptr);
					startRemap(nullptr);
				} else if (lastGesture.direction == Direction::LEFT && !firstBoot) {
					item->onExit();
				}
			}
		}
	} break;
	}
	item->fullRedraw = false;
	return false;
}
