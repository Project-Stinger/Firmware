#include "Adafruit_ST7735.h"
#include "Arduino.h"

enum class ClipBehavior {
	CLIP_TEXT,
	PRINT_LAST_LINE_DOTS,
	PRINT_LAST_LINE_DOTS_PLUS_ONE,
	PRINT_LAST_LINE_CENTERED,
	PRINT_LAST_LINE_LEFT,
	DONT_PRINT
};
#if HW_VERSION == 1
extern Adafruit_ST7735 tft;
#define YADVANCE 8
#define YADVANCE_RELAXED 10
#define MAX_SCREEN_LINES 10
#define SCREEN_WIDTH 160
#define SCREEN_HEIGHT 80
#define SET_DEFAULT_FONT tft.setFont()
#define MENU_START_VALUE_X 109
#elif HW_VERSION == 2
extern Adafruit_ST7789 tft;
#define YADVANCE 12
#define YADVANCE_RELAXED 15
#define MAX_SCREEN_LINES 11
#define SCREEN_WIDTH 240
#define SCREEN_HEIGHT 135
#define SET_DEFAULT_FONT tft.setFont(&Bass12px)
#define MENU_START_VALUE_X 130
#endif
void initDisplay();
void triggerFullRedraw();
void displayLoop();
void addBootMsg(const char *msg);

/**
 * @brief prints a string centered in the described origin
 *
 * @param str the string to print
 * @param x center X coordinate of the string
 * @param y start Y coordinate of the string
 * @param maxWidth maximum width in pixels
 * @param maxLines maximum line count to write
 * @param lineHeight height of a line in pixels
 * @param b How to clip text, if it doesn't fit in the described area
 * @param usedLines pointer to a variable to store the amount of lines used, nullptr if not needed
 * @return true if stuff was printed, false if nothing was printed (ClipBehavior::DONT_PRINT and the string didn't fit)
 */
bool printCentered(const char *str, u8 x, u8 y, u8 maxWidth, u8 maxLines, u8 lineHeight, ClipBehavior b, u8 *usedLines = nullptr);