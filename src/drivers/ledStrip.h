#if HW_VERSION == 2
#pragma once
#include "menu/menuItem.h"
#include "utils/typedefs.h"

enum class LED_MODE : u8 {
	OFF,
	STATIC,
	BLINK,
	BLINK_FAST,
	FADE,
	RAINBOW,
	MODE_LENGTH
};

enum class LIGHT_ID : u8 {
	NONE = 0,
	HOMESCREEN,
	BOOT,
	MENU,
	FIRSTBOOT,
	BAT_WARN,
	BAT_LOW,
	STANDBY,
};

void ledInit();
void ledLoop();

void hslToRgb(u8 h, u8 s, u8 l, u8 &r, u8 &g, u8 &b);
void rgbToHsl(u8 r, u8 g, u8 b, u8 &h, u8 &s, u8 &l);
void applyBrightness(u8 &r, u8 &g, u8 &b, u8 brightness);

bool onBrightnessEnter(MenuItem *_item);
bool onBrightnessExit(MenuItem *_item);
void onBrightnessChange(MenuItem *_item);

extern u8 brightnessMenu;

/**
 * @brief Set the mode of the LED strip
 *
 * @param mode LED_MODE:: mode
 * @param id LIGHT_ID:: id
 * @param duration in ms, 0 for infinite
 * @param r 0-255
 * @param g 0-255
 * @param b 0-255
 */
void ledSetMode(LED_MODE mode, LIGHT_ID id, u16 duration = 0, u8 r = 0, u8 g = 0, u8 b = 0, bool overrideSame = true);

/**
 * @brief Set the brightness of the LED strip
 *
 * @param brightness 0-255
 */
void setMasterBrightness(u8 brightness);

void releaseLightId(LIGHT_ID id);

void ledNewClock();

#endif