#include "Fonts/FreeSansBold12pt7b.h"
#include "Fonts/FreeSansBold9pt7b.h"
#include "global.h"

enum BootSelections {
	BOOT_NORMAL,
	BOOT_JOYSTICK_CALIBRATION,
	BOOT_FIRMWARE_UPDATE,
	BOOT_INPUT_DIAGNOSTICS,
	BOOT_FACTORY_RESET,
	BOOT_SELECTION_COUNT
};

#define BOOT_SELECTION_DURATION 4000
#define BOOT_SELECTION_PAUSE 1000

#if HW_VERSION == 1
#define BOOT_OPTION_START_Y 40
#define BOOT_TIMER_START_X 125
#define BOOT_TIMER_END_X 155
#elif HW_VERSION == 2
#define BOOT_OPTION_START_Y 75
#define BOOT_TIMER_START_X 160
#define BOOT_TIMER_END_X 235
#endif

static u8 currentSelection = 0;
static elapsedMillis selectionTimer = 0;
static bool isPaused = false;

bool reboot(MenuItem *_item) {
	rp2040.reboot();
	return false;
}

void drawBootSelect() {
	static bool firstRun = true;
#if HW_VERSION == 2
	speakerLoopOnFastCore = true;
#endif
	if (firstRun) {
		firstRun = false;
		tft.fillScreen(ST77XX_BLACK);
#if HW_VERSION == 1
		SET_DEFAULT_FONT;
		printCentered("Boot Selection", SCREEN_WIDTH / 2, 0, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_CENTERED);
		printCentered("Pull the trigger while your preferred option is selected.", SCREEN_WIDTH / 2, 12, SCREEN_WIDTH, 3, YADVANCE, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
		tft.setFont(&FreeSansBold12pt7b);
		printCentered("Boot Selection", SCREEN_WIDTH / 2, 20, SCREEN_WIDTH, 1, 0, ClipBehavior::PRINT_LAST_LINE_CENTERED);
		SET_DEFAULT_FONT;
		printCentered("Pull the trigger while your preferred option is selected.", SCREEN_WIDTH / 2, 37, SCREEN_WIDTH, 2, YADVANCE, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
		tft.setCursor(0, BOOT_OPTION_START_Y);
		tft.print("Normal Boot\nJoystick Calibration\nFirmware Update\nInput Diagnostics\nFactory Reset");
	}
	static u8 lastDrawnSelection = 0;
	static bool state = 1;
	if (state == 0 && isPaused) {
		tft.fillRect(BOOT_TIMER_START_X, BOOT_OPTION_START_Y + lastDrawnSelection * YADVANCE, (BOOT_TIMER_END_X - BOOT_TIMER_START_X), YADVANCE, ST77XX_BLACK);
		state = 1;
	}
	if (state == 1 && !isPaused) {
		tft.drawRect(BOOT_TIMER_START_X, BOOT_OPTION_START_Y + currentSelection * YADVANCE + 2, (BOOT_TIMER_END_X - BOOT_TIMER_START_X), YADVANCE - 4, ST77XX_WHITE);
		lastDrawnSelection = currentSelection;
		state = 0;
	}
	static elapsedMillis lastDraw = 0;
	if (lastDraw >= 17 && !isPaused && state == 0) {
		lastDraw = 0;
		tft.fillRect(BOOT_TIMER_START_X, BOOT_OPTION_START_Y + currentSelection * YADVANCE + 2, (BOOT_TIMER_END_X - BOOT_TIMER_START_X) * selectionTimer / BOOT_SELECTION_DURATION, YADVANCE - 4, ST77XX_WHITE);
	}
#if HW_VERSION == 2
	speakerLoopOnFastCore = false;
#endif
}
void runBootSelect() {
	static bool firstRun = true;
	if (firstRun) {
		firstRun = false;
		triggerUpdateFlag = false;
	}
	if (isPaused) {
		if (selectionTimer > BOOT_SELECTION_PAUSE) {
			currentSelection++;
			selectionTimer = 0;
			isPaused = false;
			if (currentSelection >= BOOT_SELECTION_COUNT) {
				currentSelection = 0;
			}
		}
	} else {
		if (selectionTimer > BOOT_SELECTION_DURATION) {
			isPaused = true;
			selectionTimer = 0;
		}
	}
	if (!isPaused && triggerUpdateFlag && triggerState) {
		// select the current option
		switch (currentSelection) {
		case BOOT_NORMAL:
			rebootReason = BootReason::FROM_BOOT_SELECTION;
			rp2040.reboot();
			break;
		case BOOT_JOYSTICK_CALIBRATION:
			if (!tournamentMode && !firstBoot)
				startJcal();
			break;
		case BOOT_FIRMWARE_UPDATE:
			rp2040.rebootToBootloader();
			break;
		case BOOT_INPUT_DIAGNOSTICS: {
			MenuItem *inputDiagnostics = mainMenu->search("inputDiagnostics");
			inputDiagnostics
				->setParent(nullptr)
				->onEnter();
			openedMenu = inputDiagnostics;
			operationState = STATE_MENU;
			break;
		}
		case BOOT_FACTORY_RESET: {
			MenuItem *resetMenu = mainMenu->search("resetMenu");
			resetMenu->search("cancelReset")->setOnEnterFunction(reboot);
			resetMenu
				->setOnLeftFunction(reboot)
				->onEnter();
			openedMenu = resetMenu;
			operationState = STATE_MENU;
			break;
		}
		}
	}
}