#include "Fonts/FreeMonoBoldOblique18pt7b.h"
#include "Fonts/FreeMonoBoldOblique24pt7b.h"
#include "Fonts/FreeSans12pt7b.h"
#include "Fonts/FreeSans18pt7b.h"
#include "Fonts/FreeSans9pt7b.h"
#include "Fonts/TomThumb.h"
#include "global.h"
#include <vector>
using std::vector;

#if HW_VERSION == 1
Adafruit_ST7735 tft = Adafruit_ST7735(PIN_TFT_CS, PIN_TFT_DC, PIN_TFT_RST);
#elif HW_VERSION == 2
Adafruit_ST7789 tft = Adafruit_ST7789(&SPI_DISPLAY_ARDUINO, PIN_TFT_CS, PIN_TFT_DC, PIN_TFT_RST);
#endif

#define HOME_BACKGROUND_COLOR (tournamentMode ? (tournamentInvertScreen ? ST77XX_BLACK : tft.color565(0, 0, 32)) : ST77XX_BLACK)

u32 lastStateDisplay = 0xFFFFFFFF;
bool forceFullUpdate = false;
const char *bootMsgs[4];
u8 currentBootMsgCount = 0;
bool newBootMsg = false;
void addBootMsg(const char *msg) {
	currentBootMsgCount++;
	if (currentBootMsgCount > 4) {
		for (u8 i = 0; i < 3; i++) {
			bootMsgs[i] = bootMsgs[i + 1];
		}
		currentBootMsgCount = 4;
	}
	bootMsgs[currentBootMsgCount - 1] = msg;
	DEBUG_PRINTSLN(msg);
	newBootMsg = true;
}
void printBootMsgs() {
#if HW_VERSION == 1
	tft.fillRect(0, 55, SCREEN_WIDTH - 20, 24, ST77XX_BLACK);
	tft.setFont(&TomThumb);
	tft.setCursor(0, 60);
	for (u8 i = 0; i < currentBootMsgCount; i++) {
		tft.println(bootMsgs[i]);
	}
#elif HW_VERSION == 2
	tft.fillRect(0, 102, SCREEN_WIDTH - 20, 32, ST77XX_BLACK);
	tft.setFont();
	tft.setCursor(0, 102);
	for (u8 i = 0; i < currentBootMsgCount; i++) {
		tft.println(bootMsgs[i]);
	}
#endif
}

void initDisplay() {
	SPI_DISPLAY_ARDUINO.setTX(PIN_TFT_MOSI);
	SPI_DISPLAY_ARDUINO.setSCK(PIN_TFT_SCK);
#if HW_VERSION == 1
	tft.initR(INITR_MINI160x80_PLUGIN);
	tft.setRotation(1);
	spi_set_baudrate(SPI_DISPLAY, 12000000);
#elif HW_VERSION == 2
	tft.init(SCREEN_HEIGHT, SCREEN_WIDTH);
	tft.setRotation(3);
	clock_configure(clk_peri, 0, CLOCKS_CLK_PERI_CTRL_AUXSRC_VALUE_CLK_SYS, F_CPU, F_CPU);
	spi_set_baudrate(SPI_DISPLAY, 66000000);
#endif
	tft.cp437(true);
	tft.fillScreen(ST77XX_BLACK);
	tft.setTextColor(ST77XX_WHITE);
	tft.enableTearing(false);
}

void triggerFullRedraw() {
	if (lastStateDisplay == STATE_JOYSTICK_CAL)
		lastJoystickCalDrawState = 0xFF;
	forceFullUpdate = true;
	if (openedMenu != nullptr) {
		openedMenu->triggerFullRedraw();
	}
}

i32 findFirstSpace(const char *str) {
	i32 i = 0;
	while (str[i] != ' ') {
		if (str[i] == '\0') return -1;
		i++;
	}
	return i;
}

bool printCentered(const char *str, u8 x, u8 y, u8 maxWidth, u8 maxLines, u8 lineHeight, ClipBehavior b, u8 *usedLines) {
	vector<char> buf;
	while (str[0] == ' ')
		str++;
	u8 len = strlen(str);
	u16 width, height;
	i16 x1, y1;
	tft.setTextWrap(false);
	for (u8 i = 0; i < len; i++) {
		buf.push_back(str[i]);
	}
	while (buf.size() && buf[buf.size() - 1] == ' ')
		buf.pop_back();
	buf.push_back('\0');
	len = buf.size() - 1;
	if (len == 0) {
		if (usedLines != nullptr)
			*usedLines = 0;
		return true;
	}
	tft.getTextBounds(buf.data(), 0, 0, &x1, &y1, &width, &height);
	if (maxLines == 1 && maxWidth < width) {
		switch (b) {
		case ClipBehavior::CLIP_TEXT: {
			while (width > maxWidth) {
				buf.pop_back();
				buf.pop_back();
				buf.push_back('\0');
				tft.getTextBounds(buf.data(), 0, 0, &x1, &y1, &width, &height);
			}
			tft.setCursor(x - width / 2, y);
		} break;
		case ClipBehavior::PRINT_LAST_LINE_DOTS: {
			while (width > maxWidth) {
				buf.pop_back();
				buf.pop_back();
				buf.pop_back();
				buf.pop_back();
				buf.pop_back();
				buf.push_back('.');
				buf.push_back('.');
				buf.push_back('.');
				buf.push_back('\0');
				tft.getTextBounds(buf.data(), 0, 0, &x1, &y1, &width, &height);
			}
			tft.setCursor(x - width / 2, y);
		} break;
		case ClipBehavior::PRINT_LAST_LINE_DOTS_PLUS_ONE: {
			char c = buf[buf.size() - 2];
			while (width > maxWidth) {
				buf.pop_back();
				buf.pop_back();
				buf.pop_back();
				buf.pop_back();
				buf.pop_back();
				buf.pop_back();
				buf.push_back('.');
				buf.push_back('.');
				buf.push_back('.');
				buf.push_back(c);
				buf.push_back('\0');
				tft.getTextBounds(buf.data(), 0, 0, &x1, &y1, &width, &height);
			}
			tft.setCursor(x - width / 2, y);
		} break;
		case ClipBehavior::PRINT_LAST_LINE_CENTERED: {
			tft.setCursor(x - width / 2, y);
		} break;
		case ClipBehavior::PRINT_LAST_LINE_LEFT: {
			tft.setCursor(x - maxWidth / 2, y);
		} break;
		case ClipBehavior::DONT_PRINT: {
			if (usedLines != nullptr)
				*usedLines = 0;
			return false;
		} break;
		}
		tft.print(buf.data());
		if (usedLines != nullptr)
			*usedLines = 1;
		return true;
	} else if (maxLines == 1) {
		tft.setCursor(x - width / 2, y);
		tft.print(buf.data());
		if (usedLines != nullptr)
			*usedLines = 1;
		return true;
	} else {
		// leading and trailing spaces are removed beforehand, we can assume that the first character is a real one
		u32 printLength = 0; // char count that is definitely being printed
		u32 nextSpace = 0; // char count until the next space
		vector<char> line;
		width = 0;
		while (width < maxWidth) {
			printLength += nextSpace;
			if (printLength) {
				if (printLength++ == len) break;
				line.pop_back();
				line.push_back(' ');
			}
			nextSpace = findFirstSpace(buf.data() + printLength);
			if (nextSpace == -1) {
				nextSpace = len - printLength;
			}
			for (u32 i = 0; i < nextSpace; i++) {
				line.push_back(buf[printLength + i]);
			}
			line.push_back('\0');
			tft.getTextBounds(line.data(), 0, 0, &x1, &y1, &width, &height);
		}
		if (printLength) {
			// at least one word fits, print all fitting words
			printLength--;
			if (len == printLength || printCentered(str + printLength, x, y + lineHeight, maxWidth, maxLines - 1, lineHeight, b, usedLines)) {
				line.clear();
				for (u32 i = 0; i < printLength; i++) {
					line.push_back(buf[i]);
				}
				line.push_back('\0');
				tft.getTextBounds(line.data(), 0, 0, &x1, &y1, &width, &height);
				tft.setCursor(x - width / 2, y);
				tft.print(line.data());
				if (usedLines != nullptr)
					(*usedLines)++;
				return true;
			}
			return false;
		}
		// not a whole word fits, try to fit as many chars as possible
		printLength = 0;
		line.clear();
		line.push_back('\0');
		width = 0;
		while (width <= maxWidth && printLength < len) {
			line.pop_back();
			line.push_back(str[printLength++]);
			line.push_back('\0');
			tft.getTextBounds(line.data(), 0, 0, &x1, &y1, &width, &height);
		}
		line.pop_back();
		line.pop_back();
		line.push_back('\0');
		printLength--;
		if (!printLength) {
			// no chars fit, don't print anything
			if (usedLines != nullptr)
				*usedLines = 0;
			return b != ClipBehavior::DONT_PRINT;
		}
		if (printCentered(str + printLength, x, y + lineHeight, maxWidth, maxLines - 1, lineHeight, b, usedLines)) {
			tft.getTextBounds(line.data(), 0, 0, &x1, &y1, &width, &height);
			tft.setCursor(x - width / 2, y);
			tft.print(line.data());
			if (usedLines != nullptr)
				(*usedLines)++;
			return true;
		}
		return false;
	}
}

void printJoystickGuide() {
	SET_DEFAULT_FONT;
	tft.setTextColor(ST77XX_WHITE);
#if HW_VERSION == 1
	tft.fillCircle(130, 59, 11, tft.color565(192, 192, 192));
	tft.fillTriangle(130, 58, 145, 58, 145, 51, HOME_BACKGROUND_COLOR);
	tft.fillTriangle(130, 58, 115, 58, 115, 51, HOME_BACKGROUND_COLOR);
	tft.fillTriangle(130, 59, 132, 47, 128, 47, HOME_BACKGROUND_COLOR);
	tft.fillCircle(130, 59, 6, HOME_BACKGROUND_COLOR);
	tft.drawCircle(130, 59, 11, ST77XX_WHITE);
	tft.drawCircle(130, 59, 7, ST77XX_WHITE);
	printCentered("Profiles", 130, 72, 60, 1, 8, ClipBehavior::PRINT_LAST_LINE_CENTERED);
	printCentered("Menu", 148, 39, 24, 1, 8, ClipBehavior::PRINT_LAST_LINE_CENTERED);
	printCentered("Darts", 108, 39, 30, 1, 8, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
	tft.fillCircle(200, 95, 17, tft.color565(165, 165, 165));
	tft.fillTriangle(199, 95, 225, 95, 225, 82, HOME_BACKGROUND_COLOR);
	tft.fillTriangle(201, 95, 175, 95, 175, 82, HOME_BACKGROUND_COLOR);
	tft.fillTriangle(200, 93, 204, 78, 196, 78, HOME_BACKGROUND_COLOR);
	tft.fillCircle(200, 95, 10, HOME_BACKGROUND_COLOR);
	tft.drawCircle(200, 95, 17, ST77XX_WHITE);
	tft.drawCircle(200, 95, 11, ST77XX_WHITE);
	printCentered("Profiles", 200, 117, 70, 1, 12, ClipBehavior::PRINT_LAST_LINE_CENTERED);
	printCentered("Menu", 220, 64, 40, 1, 12, ClipBehavior::PRINT_LAST_LINE_CENTERED);
	printCentered("Darts", 175, 64, 40, 1, 12, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
}

#if HW_VERSION == 2
void printJoystickGuideDot(bool force = false) {
	static u8 lastX = 0, lastY = 0;
	static i32 lastMagnitude = 0;
	static fix32 lastAngle = 0;
	if (joystickAngle == lastAngle && joystickMagnitude == lastMagnitude && !force) {
		return;
	}
	lastAngle = joystickAngle;
	i32 thisMag = constrain(joystickMagnitude - 10, 0, GESTURE_CENTER_LARGE_PCT - 20);
	thisMag = map(thisMag, 0, GESTURE_CENTER_LARGE_PCT - 20, 0, GESTURE_CENTER_LARGE_PCT);
	startFixTrig();
	u8 x = 200 - (i32)(cosFix(lastAngle) * thisMag * 9 / GESTURE_CENTER_LARGE_PCT + fix32(0.5f)).geti32();
	u8 y = 95 + (i32)(sinFix(lastAngle) * thisMag * 9 / GESTURE_CENTER_LARGE_PCT + fix32(0.5f)).geti32();
	if (lastX == x && lastY == y && !force) {
		return;
	}
	tft.fillCircle(lastX, lastY, 1, HOME_BACKGROUND_COLOR);
	if (lastMagnitude >= GESTURE_CENTER_LARGE_PCT * 8 / 9) {
		tft.drawCircle(200, 95, 11, ST77XX_WHITE);
	}
	tft.fillCircle(x, y, 1, ST77XX_WHITE);
	lastMagnitude = thisMag;
	lastX = x;
	lastY = y;
}
#endif

void printDcGuide() {
	tft.setTextColor(ST77XX_WHITE);
	SET_DEFAULT_FONT;
#if HW_VERSION == 1
	tft.fillRect(90, 38, 70, 42, HOME_BACKGROUND_COLOR);
	tft.drawCircle(135, 95, 50, ST77XX_WHITE);
	tft.fillRect(134, 45, 26, 10, HOME_BACKGROUND_COLOR);
	tft.fillRect(85, 60, 15, 20, HOME_BACKGROUND_COLOR);
	tft.drawLine(135, 45, 132, 42, ST77XX_WHITE);
	tft.drawLine(135, 45, 132, 48, ST77XX_WHITE);
	tft.drawLine(99, 60, 99, 55, ST77XX_WHITE);
	tft.drawLine(99, 60, 104, 60, ST77XX_WHITE);
	tft.setCursor(142, 42);
	tft.print('+');
	tft.setCursor(95, 65);
	tft.print('-');
	tft.setCursor(120, 57);
	tft.print("Dart");
	tft.setCursor(120, 67);
	tft.print("Count");
#elif HW_VERSION == 2
	tft.fillRect(136, 64, 104, 71, HOME_BACKGROUND_COLOR);
	tft.drawCircle(204, 160, 85, ST77XX_WHITE);
	tft.fillRect(203, 75, 37, 15, HOME_BACKGROUND_COLOR);
	tft.fillRect(120, 100, 25, 35, HOME_BACKGROUND_COLOR);
	tft.drawLine(204, 75, 199, 70, ST77XX_WHITE);
	tft.drawLine(204, 75, 199, 80, ST77XX_WHITE);
	tft.drawLine(144, 100, 144, 92, ST77XX_WHITE);
	tft.drawLine(144, 100, 152, 100, ST77XX_WHITE);
	tft.setCursor(215, 71);
	tft.print('+');
	tft.setCursor(138, 108);
	tft.print('-');
	tft.setCursor(170, 105);
	tft.print("Dart Count");
#endif
}

void printJoystickLockout() {
	SET_DEFAULT_FONT;
	tft.setTextColor(ST77XX_WHITE);
#if HW_VERSION == 1
	tft.fillRect(90, 38, 70, 42, HOME_BACKGROUND_COLOR);
	printCentered("LOCKED!", 125, 39, 70, 1, 8, ClipBehavior::PRINT_LAST_LINE_CENTERED);
	printCentered("Hold UP to unlock joystick.", 125, 51, 70, 3, 8, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#elif HW_VERSION == 2
	tft.fillRect(136, 64, 104, 71, HOME_BACKGROUND_COLOR);
	printCentered("LOCKED!", 188, 78, 104, 1, 12, ClipBehavior::PRINT_LAST_LINE_CENTERED);
	printCentered("Long press UP to unlock joystick.", 188, 98, 104, 4, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
#endif
}

void eraseBottomRightWidget() {
#if HW_VERSION == 1
	tft.fillRect(90, 38, 70, 42, HOME_BACKGROUND_COLOR);
	printJoystickGuide();
#elif HW_VERSION == 2
	tft.fillRect(136, 64, 104, 71, HOME_BACKGROUND_COLOR);
	printJoystickGuide();
	printJoystickGuideDot(true);
#endif
}

void drawProfileSelectFrame(bool drawIndex) {
	tft.fillRect(5, (SCREEN_HEIGHT - 8), 1, 7, ST77XX_WHITE);
	u8 widthPerProfile = (SCREEN_WIDTH - 10) / enabledProfiles;
	SET_DEFAULT_FONT;
	tft.setTextColor(ST77XX_WHITE);
	for (u8 i = 1; i <= enabledProfiles; i++) {
		tft.fillRect(5 + i * (SCREEN_WIDTH - 10) / enabledProfiles, (SCREEN_HEIGHT - 8), 1, 7, ST77XX_WHITE);
		if (drawIndex) {
#if HW_VERSION == 1
			tft.setCursor(2 + i * widthPerProfile - widthPerProfile / 2, 62);
#elif HW_VERSION == 2
			tft.setCursor(2 + i * widthPerProfile - widthPerProfile / 2, 113);
#endif
			tft.print(i);
		}
	}
}

void drawProfileSpecs(const char *p00, const char *p01, const char *p10, const char *p11) {
#if HW_VERSION == 1
	printCentered(p00, SCREEN_WIDTH / 4, 33, SCREEN_WIDTH / 2, 1, 8, ClipBehavior::CLIP_TEXT);
	printCentered(p01, SCREEN_WIDTH * 3 / 4, 33, SCREEN_WIDTH / 2, 1, 8, ClipBehavior::CLIP_TEXT);
	printCentered(p10, SCREEN_WIDTH / 4, 46, SCREEN_WIDTH / 2, 1, 8, ClipBehavior::CLIP_TEXT);
	printCentered(p11, SCREEN_WIDTH * 3 / 4, 46, SCREEN_WIDTH / 2, 1, 8, ClipBehavior::CLIP_TEXT);
#elif HW_VERSION == 2
	printCentered(p00, SCREEN_WIDTH / 4, 50, SCREEN_WIDTH / 2, 1, 12, ClipBehavior::CLIP_TEXT);
	printCentered(p01, SCREEN_WIDTH * 3 / 4, 50, SCREEN_WIDTH / 2, 1, 12, ClipBehavior::CLIP_TEXT);
	printCentered(p10, SCREEN_WIDTH / 4, 68, SCREEN_WIDTH / 2, 1, 12, ClipBehavior::CLIP_TEXT);
	printCentered(p11, SCREEN_WIDTH * 3 / 4, 68, SCREEN_WIDTH / 2, 1, 12, ClipBehavior::CLIP_TEXT);
#endif
}

void displayLoop() {
	volatile u32 state = operationState; // cache the operationState because the other core might change it
	if (forceNewOpState != 0xFFFFFFFFUL) return; // avoid redundant display updates
	bool firstRun = false;
	if (state != lastStateDisplay || forceFullUpdate) {
		if (forceFullUpdate) {
			forceFullUpdate = false;
			lastStateDisplay = 0xFFFFFFFF;
		}
		firstRun = true;
	}
	switch (state) {
	case STATE_SETUP: {
#if HW_VERSION == 2
		speakerLoopOnFastCore = true;
#endif
		if (firstRun) {
			tft.fillScreen(ST77XX_BLACK);
#if HW_VERSION == 1
			tft.setFont(&FreeSans9pt7b);
			printCentered(deviceName, SCREEN_WIDTH / 2 - 10, 15, SCREEN_WIDTH - 20, 1, 22, ClipBehavior::PRINT_LAST_LINE_DOTS);
			SET_DEFAULT_FONT;
			printCentered(ownerName, 70, 26, SCREEN_WIDTH - 20, 1, 8, ClipBehavior::PRINT_LAST_LINE_DOTS);
			printCentered(ownerContact, 70, 34, 140, 2, 8, ClipBehavior::CLIP_TEXT);
			tft.fillRect(SCREEN_WIDTH - 16, 0, 1, SCREEN_HEIGHT, ST77XX_WHITE);
#elif HW_VERSION == 2
			tft.setFont(&FreeSans12pt7b);
			printCentered(deviceName, SCREEN_WIDTH / 2 - 10, 30, SCREEN_WIDTH - 20, 1, 22, ClipBehavior::PRINT_LAST_LINE_DOTS);
			SET_DEFAULT_FONT;
			printCentered(ownerName, SCREEN_WIDTH / 2 - 10, 50, SCREEN_WIDTH - 20, 1, 12, ClipBehavior::PRINT_LAST_LINE_DOTS);
			printCentered(ownerContact, SCREEN_WIDTH / 2 - 10, 70, SCREEN_WIDTH - 20, 2, 12, ClipBehavior::CLIP_TEXT);
			tft.fillRect(SCREEN_WIDTH - 16, 0, 1, SCREEN_HEIGHT, ST77XX_WHITE);
#endif
		}
		tft.fillRect(SCREEN_WIDTH - 15, SCREEN_HEIGHT * (100 - bootProgress) / 100, 15, SCREEN_HEIGHT * bootProgress / 100, ST77XX_WHITE);
		tft.fillRect(SCREEN_WIDTH - 15, 0, 15, SCREEN_HEIGHT * (100 - bootProgress) / 100, ST77XX_BLACK);
		if (newBootMsg) {
			newBootMsg = false;
			printBootMsgs();
		}
	} break;
	case STATE_PROFILE_SELECT: {
		static const GFXfont *lastFont = &FreeSans9pt7b;
		static u8 yOff = 0;
		static ClipBehavior lastClipBehavior = ClipBehavior::DONT_PRINT;
		static char lastProfileName[16] = "\0";
		static char profileSpec00[13] = "\0";
		static char profileSpec01[13] = "\0";
		static char profileSpec10[13] = "\0";
		static char profileSpec11[13] = "\0";
		if (firstRun) {
			tft.fillScreen(ST77XX_BLACK);
			drawProfileSelectFrame(true);
		}
		volatile bool npfCache = false;
		if (newProfileFlag) {
			newProfileFlag = false;
			npfCache = true;
			// erase last profile name
			tft.setTextColor(ST77XX_BLACK);
			tft.setFont(lastFont);
			printCentered(lastProfileName, SCREEN_WIDTH / 2, yOff, SCREEN_WIDTH, 1, 29, lastClipBehavior);
			SET_DEFAULT_FONT;
			drawProfileSpecs(profileSpec00, profileSpec01, profileSpec10, profileSpec11);
		}
		if (npfCache || firstRun) {
			SET_DEFAULT_FONT;
			tft.setTextColor(profileColor565);
			snprintf(profileSpec00, 13, "%dk RPM", targetRpm / 1000);
			if (fireMode == FIRE_BURST)
				snprintf(profileSpec01, 13, "%s: %d", fireModeNames[fireMode], burstCount);
			else
				strcpy(profileSpec01, fireModeNames[fireMode]);
			if (fireMode == FIRE_BURST || fireMode == FIRE_CONTINUOUS) {
#if HW_VERSION == 2
				if (autoPusherTiming)
					snprintf(profileSpec11, 13, "Max. %d DPS", dpsLimit);
				else
#endif
					snprintf(profileSpec11, 13, "%d DPS", (1000000 + (pushDuration + retractDuration) / 2) / (pushDuration + retractDuration));
			} else {
				profileSpec11[0] = '\0';
			}
			snprintf(profileSpec10, 13, "F/R %d%%", 100 + frontRearRatio);
			drawProfileSpecs(profileSpec00, profileSpec01, profileSpec10, profileSpec11);
#if HW_VERSION == 1
			tft.setFont(&FreeSans12pt7b);
			lastFont = &FreeSans12pt7b;
			yOff = 22;
#elif HW_VERSION == 2
			tft.setFont(&FreeSans18pt7b);
			lastFont = &FreeSans18pt7b;
			yOff = 30;
#endif
			lastClipBehavior = ClipBehavior::DONT_PRINT;
			memcpy(lastProfileName, profileName, 16);
			if (!printCentered(lastProfileName, SCREEN_WIDTH / 2, yOff, SCREEN_WIDTH, 1, 29, ClipBehavior::DONT_PRINT)) {
#if HW_VERSION == 1
				tft.setFont(&FreeSans9pt7b);
				lastFont = &FreeSans9pt7b;
				yOff = 22;
#elif HW_VERSION == 2
				tft.setFont(&FreeSans12pt7b);
				lastFont = &FreeSans12pt7b;
				yOff = 30;
#endif
				printCentered(lastProfileName, SCREEN_WIDTH / 2, yOff, SCREEN_WIDTH, 1, 22, ClipBehavior::PRINT_LAST_LINE_DOTS_PLUS_ONE);
				lastClipBehavior = ClipBehavior::PRINT_LAST_LINE_DOTS_PLUS_ONE;
			}
		}
		static u8 lastDotPos = 0;
		u8 dotPos = profileSelectionJoystickAngle * (SCREEN_WIDTH - 10) / (PROFILE_SELECTION_RANGE - 1) + 5;
		if (dotPos != lastDotPos) {
			tft.fillCircle(lastDotPos, SCREEN_HEIGHT - 5, 2, ST77XX_BLACK);
			tft.fillCircle(dotPos, SCREEN_HEIGHT - 5, 2, ST77XX_WHITE);
			drawProfileSelectFrame(false);
			lastDotPos = dotPos;
		}
	} break;
	case STATE_OPEN_MENU:
		if (firstRun) {
			mainMenu->onEnter();
#if HW_VERSION == 1
			MenuItem::scheduleBeep(0, 1);
#elif HW_VERSION == 2
			MenuItem::beep(SETTINGS_BEEP_MIN_FREQ);
#endif
			openedMenu = mainMenu;
			menuOverrideTimer = 100;
		}
		break;
	case STATE_MENU:
		break;
	case STATE_OFF:
	case STATE_RAMPUP:
	case STATE_PUSH:
	case STATE_RETRACT:
	case STATE_RAMPDOWN: {
		static char lastBatString[16] = {0};
		static u8 lastBatWidth = 0;
		static bool lastDcUpdating = false;
		static bool lastJoystickLockout = false;
		u16 width, height;
		i16 x1, y1;
		char buf[32];
		bool batUpdate = false;
		bool forceBatUpdate = false;
		static char lastRpm0[16] = {0};
		static char lastRpm1[16] = {0};
		bool rpmUpdate = false;
		bool forceRpmUpdate = false;
		bool profileUpdate = false;
		bool fireModeUpdate = false;
		bool dcUpdate = false;
		static elapsedMillis dcUpdateTimer = 0;

		// print dart count guide or erase it (force transparent updates on other items)
		if (lastDcUpdating != updatingDartCount && dcUpdateTimer >= 17) {
			dcUpdateTimer = 0;
			lastDcUpdating = updatingDartCount;
			forceBatUpdate = true;
			profileUpdate = true;
			fireModeUpdate = true;
			forceRpmUpdate = true;
			dcUpdate = true;
#if HW_VERSION == 2
			speakerLoopOnFastCore = true;
#endif
			if (lastDcUpdating) {
				printDcGuide();
			} else {
				eraseBottomRightWidget();
			}
		}

		// force all updates when entering the home screen or when the profile changes (when it wasn't painted)
		if (newProfileFlag || (firstRun && (lastStateDisplay < STATE_OFF || lastStateDisplay > STATE_RAMPDOWN))) {
#if HW_VERSION == 2
			speakerLoopOnFastCore = true;
#endif
			newProfileFlag = false;
			tft.fillScreen(HOME_BACKGROUND_COLOR);
			profileUpdate = true;
			if (joystickLockout) {
				printJoystickLockout();
			} else {
				printJoystickGuide();
#if HW_VERSION == 2
				printJoystickGuideDot(true);
#endif
			}
			fireModeUpdate = true;
			dcUpdate = true;
			batUpdate = true;
			lastBatString[0] = '\0';
			rpmUpdate = true;
			lastRpm0[0] = '\0';
			lastRpm1[0] = '\0';
		}

		// print joystick guide when returning from firing
		if (firstRun && lastStateDisplay > STATE_RAMPUP && state == STATE_OFF && !joystickLockout) {
			if (joystickLockout) {
				printJoystickLockout();
			} else {
#if HW_VERSION == 1
				tft.fillRect(90, 47, 70, 22, HOME_BACKGROUND_COLOR);
				printJoystickGuide();
#elif HW_VERSION == 2
				speakerLoopOnFastCore = true;
				tft.fillRect(136, 80, 104, 22, HOME_BACKGROUND_COLOR);
				printJoystickGuide();
				printJoystickGuideDot(true);
#endif
			}
		}

		// show rampup time while firing
		if (firstRun && (lastStateDisplay < STATE_PUSH || lastStateDisplay > STATE_RAMPDOWN) && (state >= STATE_PUSH && state <= STATE_RAMPDOWN)) {
#if HW_VERSION == 1
			tft.fillRect(90, 38, 70, 42, HOME_BACKGROUND_COLOR);
			tft.setTextColor(ST77XX_WHITE);
			tft.setFont(&FreeSans9pt7b);
			tft.setCursor(90, 60);
			if (thisRampupDuration != 0xFFFFFFFFUL) {
				tft.printf("%d ms", thisRampupDuration);
			} else {
				tft.print("Timeout");
			}
#elif HW_VERSION == 2
			speakerLoopOnFastCore = true;
			tft.fillRect(136, 64, 104, 71, HOME_BACKGROUND_COLOR);
			tft.setTextColor(ST77XX_WHITE);
			tft.setFont(&FreeSans12pt7b);
			tft.setCursor(136, 100);
			if (thisRampupDuration != 0xFFFFFFFFUL) {
				tft.printf("%d ms", thisRampupDuration);
			} else {
				tft.print("Timeout");
			}
#endif
		}

		// show darts per second on rampdown
#if HW_VERSION == 2
		if (firstRun && state == STATE_RAMPDOWN && pushCount > 1) {
			speakerLoopOnFastCore = true;
			tft.fillRect(136, 64, 104, 71, HOME_BACKGROUND_COLOR);
			tft.setTextColor(ST77XX_WHITE);
			tft.setFont(&FreeSans12pt7b);
			tft.setCursor(136, 100);
			tft.printf("%.1f DPS", (actualDps + 0.05f).getf32());
		}
#endif

		// print profile name
		if (profileUpdate) {
#if HW_VERSION == 1
			tft.setFont();
			tft.setCursor(0, 0);
#elif HW_VERSION == 2
			tft.setFont(&FreeSans9pt7b);
			tft.setCursor(4, 13);
			speakerLoopOnFastCore = true;
#endif
			tft.setTextColor(lastDcUpdating ? ((profileColor565 >> 1) & 0b0111101111101111) : profileColor565);
			tft.print(profileName);
		}

		// print fire modes
		if (fireModeUpdate) {
			SET_DEFAULT_FONT;
			if (fireMode == FIRE_BURST)
				snprintf(buf, 13, "%s: %d", fireModeNames[fireMode], burstCount);
			else
				strcpy(buf, fireModeNames[fireMode]);
#if HW_VERSION == 1
			tft.setCursor(SCREEN_WIDTH - strlen(buf) * 6, 0);
#elif HW_VERSION == 2
			tft.setCursor(SCREEN_WIDTH - strlen(buf) * 13 / 2, 0);
			speakerLoopOnFastCore = true;
#endif
			tft.setTextColor(lastDcUpdating ? ((ST7735_WHITE >> 1) & 0b0111101111101111) : ST7735_WHITE);
			tft.print(buf);
		}

#if HW_VERSION == 2
		// print ML log percentage indicator
		{
			static elapsedMillis mlUpdateTimer = 1000;
			static char lastMlStr[8] = "";
			if (mlUpdateTimer > 1000) {
				mlUpdateTimer = 0;
				char mlStr[8] = "";
				if (mlLogIsActive()) {
					snprintf(mlStr, sizeof(mlStr), "R%d%%", mlLogFlashPercent());
				}
				if (strcmp(mlStr, lastMlStr) != 0) {
					SET_DEFAULT_FONT;
					speakerLoopOnFastCore = true;
					// erase old
					tft.fillRect(105, 0, 30, YADVANCE, HOME_BACKGROUND_COLOR);
					if (mlStr[0]) {
						tft.setCursor(105, 0);
						tft.setTextColor(ST77XX_RED);
						tft.print(mlStr);
					}
					strcpy(lastMlStr, mlStr);
				}
			}
		}
#endif

		// print or erase joystick lockout message
		if (joystickLockout != lastJoystickLockout) {
			lastJoystickLockout = joystickLockout;
#if HW_VERSION == 2
			speakerLoopOnFastCore = true;
#endif
			if (lastJoystickLockout) {
				printJoystickLockout();
			} else {
				eraseBottomRightWidget();
			}
		}

		// print battery voltage
		static elapsedMillis batUpdateTimer = 200;
		if (batUpdateTimer > 200) {
			batUpdateTimer = 0;
			batUpdate = true;
		}
		if (batUpdate || forceBatUpdate) {
#if HW_VERSION == 2
			speakerLoopOnFastCore = true;
#endif
			batUpdate = false;
			if (batState == BAT_RUNNING)
				snprintf(buf, 16, "%.2fV, %dS", ((fix32)batVoltage).getf32() / batCellCount, batCellCount);
			else
				snprintf(buf, 16, "%.2fV", ((fix32)batVoltage).getf32());
			if (strcmp(buf, lastBatString) != 0 || forceBatUpdate) {
				SET_DEFAULT_FONT;
				tft.setTextColor(HOME_BACKGROUND_COLOR);
				tft.setCursor(0, SCREEN_HEIGHT - YADVANCE);
				tft.print(lastBatString);
				tft.setTextColor((motorDisableFlags & MD_BATTERY_EMPTY) ? ST77XX_RED : (batWarning ? ST77XX_YELLOW : (lastDcUpdating ? ((ST7735_WHITE >> 1) & 0b0111101111101111) : ST7735_WHITE)));
				tft.getTextBounds(buf, 0, 0, &x1, &y1, &width, &height);
				tft.setCursor(0, SCREEN_HEIGHT - YADVANCE);
				tft.print(buf);
				lastBatWidth = width;
				memcpy(lastBatString, buf, 16);
			}
		}

		// flag for printing dart count on mag change
#ifdef USE_TOF
		static bool lastMagPresent = false;
		if (lastMagPresent != magPresent) {
			lastMagPresent = magPresent;
			dcUpdate = true;
		}
		static bool lastTofFound = false;
		if (lastTofFound != foundTof) {
			lastTofFound = foundTof;
			dcUpdate = true;
		}
#endif

		// print dart count
		static u8 lastDartCount = 0;
		if (lastDartCount != dartCount) {
			lastDartCount = dartCount;
			dcUpdate = true;
		}
		static char lastDartCountString[3] = {0};
		static u8 lastDartCountWidth = 0;
		if (dcUpdate) {
#if HW_VERSION == 2
			speakerLoopOnFastCore = true;
			tft.setTextSize(2);
#endif
			dcUpdate = false;
			tft.setFont(HW_VERSION == 2 ? &FreeMonoBoldOblique18pt7b : &FreeMonoBoldOblique24pt7b);
			tft.setTextColor(HOME_BACKGROUND_COLOR);
			tft.setCursor(0, HW_VERSION == 2 ? 80 : 45);
			tft.print(lastDartCountString);
			if (dartCount)
				snprintf(buf, 3, "%2d", dartCount);
#ifdef USE_TOF
			else if (!foundTof)
				strcpy(buf, "er");
			else if (magPresent)
				strcpy(buf, " 0");
#endif
			else
				strcpy(buf, "--");
			tft.getTextBounds(buf, 0, 0, &x1, &y1, &width, &height);
			lastDartCountWidth = width;
			tft.setCursor(0, HW_VERSION == 2 ? 80 : 45);
			tft.setTextColor(ST7735_WHITE);
			tft.print(buf);
#if HW_VERSION == 2
			tft.setTextSize(1);
#endif
			lastDartCountString[0] = buf[0];
			lastDartCountString[1] = buf[1];
			lastDartCountString[2] = '\0';
		}

		// print RPM
		static elapsedMillis rpmUpdateTimer = 50;
		if (rpmUpdateTimer >= 50) {
			rpmUpdateTimer = 0;
			rpmUpdate = true;
		}
		if (rpmUpdate || forceRpmUpdate) {
#if HW_VERSION == 2
			speakerLoopOnFastCore = true;
#endif
			rpmUpdate = false;
			SET_DEFAULT_FONT;
			tft.setTextColor(lastDcUpdating ? ((ST7735_WHITE >> 1) & 0b0111101111101111) : ST7735_WHITE);
			if (motorDisableFlags & MD_ESC_OVERTEMP)
				snprintf(buf, 12, "%3d C %3d C", escTemp[0], escTemp[2]);
			else
				snprintf(buf, 12, "%4.1fk %4.1fk", escRpm[0] / 1000.f, escRpm[2] / 1000.f);
			if (strcmp(buf, lastRpm0) != 0 || forceRpmUpdate) {
#if HW_VERSION == 1
				tft.fillRect(94, 14, 66, 8, HOME_BACKGROUND_COLOR);
				tft.setCursor(94, 14);
#elif HW_VERSION == 2
				tft.fillRect(165, 24, 75, 12, HOME_BACKGROUND_COLOR);
				tft.setCursor(165, 24);
#endif
				tft.print(buf);
				strncpy(lastRpm0, buf, 16);
			}
			if (motorDisableFlags & MD_ESC_OVERTEMP)
				snprintf(buf, 12, "%3d C %3d C", escTemp[1], escTemp[3]);
			else
				snprintf(buf, 12, "%4.1fk %4.1fk", escRpm[1] / 1000.f, escRpm[3] / 1000.f);
			if (strcmp(buf, lastRpm1) != 0 || forceRpmUpdate) {
#if HW_VERSION == 1
				tft.fillRect(94, 24, 66, 8, HOME_BACKGROUND_COLOR);
				tft.setCursor(94, 24);
#elif HW_VERSION == 2
				tft.fillRect(165, 42, 75, 12, HOME_BACKGROUND_COLOR);
				tft.setCursor(165, 42);
#endif
				tft.print(buf);
				strncpy(lastRpm1, buf, 16);
			}
		}

		// print joystick guide dot
#if HW_VERSION == 2
		static elapsedMillis updateTimer = 100;
		if (updateTimer >= 17 && !updatingDartCount && !joystickLockout && state == STATE_OFF) {
			updateTimer = 0;
			speakerLoopOnFastCore = true;
			printJoystickGuideDot();
		}
#endif
	} break;
	case STATE_JOYSTICK_CAL: {
		drawJoystickCalibration();
	} break;
	case STATE_FALL_DETECTED: {
		if (firstRun) {
#if HW_VERSION == 2
			speakerLoopOnFastCore = true;
#endif
			tft.fillScreen(ST77XX_BLACK);
			tft.setFont(&FreeSans9pt7b);
			tft.setTextColor(ST77XX_WHITE);
			printCentered("Fall Detected", SCREEN_WIDTH / 2, 13, SCREEN_WIDTH, 1, 22, ClipBehavior::PRINT_LAST_LINE_DOTS);
			SET_DEFAULT_FONT;
			printCentered("Push the joystick outwards and then rotate it half a turn to go back.", SCREEN_WIDTH / 2, 25, SCREEN_WIDTH, 3, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
		}
		static elapsedMillis updateTimer = 500;
		if (updateTimer > 500) {
			updateTimer = 0;
#if HW_VERSION == 2
			speakerLoopOnFastCore = true;
#endif
			tft.fillRect(0, SCREEN_HEIGHT * 3 / 4, SCREEN_WIDTH, YADVANCE, ST77XX_BLACK);
			char buf[16];
			snprintf(buf, 26, "%dS, %.2f V/cell, %.2fV", batCellCount, ((fix32)batVoltage).getf32() / batCellCount, ((fix32)batVoltage).getf32());
			SET_DEFAULT_FONT;
			printCentered(buf, SCREEN_WIDTH / 2, SCREEN_HEIGHT * 3 / 4, SCREEN_HEIGHT, 1, YADVANCE, ClipBehavior::PRINT_LAST_LINE_CENTERED);
		}
	} break;
	case STATE_SAFE: {
		if (firstRun) {
#if HW_VERSION == 2
			speakerLoopOnFastCore = true;
#endif
			tft.fillScreen(ST77XX_BLACK);
			tft.setFont(&FreeSans18pt7b);
			tft.setTextColor(ST77XX_WHITE);
			printCentered("SAFE", SCREEN_WIDTH / 2, 26, SCREEN_WIDTH, 1, 42, ClipBehavior::PRINT_LAST_LINE_DOTS);
			SET_DEFAULT_FONT;
			printCentered("Push the joystick outwards and then rotate it half a turn to continue.", SCREEN_WIDTH / 2, HW_VERSION == 1 ? 50 : 65, SCREEN_WIDTH, 3, YADVANCE_RELAXED, ClipBehavior::PRINT_LAST_LINE_CENTERED);
		}
	} break;
	case STATE_BOOT_SELECT: {
		drawBootSelect();
	} break;
	}
#if HW_VERSION == 2
	speakerLoopOnFastCore = false;
#endif
	lastStateDisplay = state;
}