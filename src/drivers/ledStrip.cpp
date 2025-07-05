#if HW_VERSION == 2

#include "global.h"

u8 ledRed = 0;
u8 ledGreen = 0;
u8 ledBlue = 0;
u8 masterBrightness = 255;
u8 brightnessMenu = 255;
// program colors
u8 progRed = 0, progGreen = 0, progBlue = 0;
u16 ledDuration = 0;
bool ledUpdated = false;
LED_MODE ledMode = LED_MODE::OFF;
elapsedMillis ledTimer;
LIGHT_ID lightId = LIGHT_ID::NONE; // higher = more important, 255 can overwrite any but is then reset to 0

struct LedProgram {
	LIGHT_ID id;
	LED_MODE mode;
	u16 duration;
	u8 r, g, b;
};

static NeoPixelConnect leds(PIN_LED_STRIP, 2, pio1, 0);

// Helper: given p, q and a “shifted” hue t (in 0–255), compute one color channel.
static inline int hue2rgb(int p, int q, int t) {
	// Wrap t if needed:
	if (t < 0) t += 255;
	if (t > 255) t -= 255;
	// The 1/6, 1/2 and 2/3 breakpoints in 0–1 correspond to ~43, 128 and 170 in 0–255.
	if (t < 43)
		return p + ((q - p) * t * 6 + 127) / 255;
	if (t < 128)
		return q;
	if (t < 170)
		return p + ((q - p) * (170 - t) * 6 + 127) / 255;
	return p;
}

/**
 * @brief RGB to HSL conversion
 *
 * @param r input Red (0-255)
 * @param g input Green (0-255)
 * @param b input Blue (0-255)
 * @param h output Hue (0-255)
 * @param s output Saturation (0-255)
 * @param l output Luminance (0-255)
 */
void rgbToHsl(u8 r, u8 g, u8 b, u8 &h, u8 &s, u8 &l) {
	// Find min and max of r, g, b:
	u8 maxVal = r, minVal = r;
	if (g > maxVal) maxVal = g;
	if (b > maxVal) maxVal = b;
	if (g < minVal) minVal = g;
	if (b < minVal) minVal = b;

	// Lightness:
	l = (maxVal + minVal) / 2;

	int diff = maxVal - minVal;
	if (diff == 0) {
		// achromatic: saturation = 0 and hue is undefined (set to 0)
		s = 0;
		h = 0;
		return;
	}

	// Saturation:
	if (l < 128)
		s = (u8)((diff * 255 + (maxVal + minVal) / 2) / (maxVal + minVal));
	else
		s = (u8)((diff * 255 + (510 - (maxVal + minVal)) / 2) / (510 - (maxVal + minVal)));

	// Hue (in degrees):
	int hueDeg = 0;
	// Using the standard formula:
	//   if r is max: hue = (g - b) / diff
	//   if g is max: hue = 2 + (b - r) / diff
	//   if b is max: hue = 4 + (r - g) / diff,
	// then multiplied by 60.
	if (r == maxVal)
		hueDeg = ((int)(g - b) * 60 + diff / 2) / diff;
	else if (g == maxVal)
		hueDeg = 120 + (((int)(b - r) * 60 + diff / 2) / diff);
	else // b == maxVal
		hueDeg = 240 + (((int)(r - g) * 60 + diff / 2) / diff);

	if (hueDeg < 0)
		hueDeg += 360;

	// Scale hue from degrees (0–360) to 0–255:
	h = (u8)((hueDeg * 255 + 180) / 360);
}

/**
 * @brief HSL to RGB conversion
 *
 * @param h input Hue (0-255)
 * @param s input Saturation (0-255)
 * @param l input Luminance (0-255)
 * @param r output Red (0-255)
 * @param g output Green (0-255)
 * @param b output Blue (0-255)
 */
void hslToRgb(u8 h, u8 s, u8 l, u8 &r, u8 &g, u8 &b) {
	if (s == 0) {
		// achromatic color (gray)
		r = g = b = l;
		return;
	}

	// First compute “q” and “p” (using the standard formulas, scaled to 0–255):
	int q = (l < 128)
				? (l * (255 + s) + 127) / 255
				: (l + s - (l * s + 127) / 255);
	int p = 2 * l - q;

	// Now compute each color channel.
	// In the p–q method the red channel uses hue shifted by +1/3 (≈85),
	// green uses the original hue and blue uses hue shifted by –1/3.
	int r_val = hue2rgb(p, q, h + 85);
	int g_val = hue2rgb(p, q, h);
	int b_val = hue2rgb(p, q, h - 85);

	// Clamp just in case:
	if (r_val < 0)
		r_val = 0;
	else if (r_val > 255)
		r_val = 255;
	if (g_val < 0)
		g_val = 0;
	else if (g_val > 255)
		g_val = 255;
	if (b_val < 0)
		b_val = 0;
	else if (b_val > 255)
		b_val = 255;

	r = (u8)r_val;
	g = (u8)g_val;
	b = (u8)b_val;
}

void applyBrightness(u8 &r, u8 &g, u8 &b, u8 brightness) {
	r = (r * (i32)brightness) / 255;
	g = (g * (i32)brightness) / 255;
	b = (b * (i32)brightness) / 255;
}

void updateLed(u8 r, u8 g, u8 b) {
	static u8 lastBrightness = 255;
	if (r == ledRed && g == ledGreen && b == ledBlue && lastBrightness == masterBrightness) {
		return;
	}
	ledRed = r;
	ledGreen = g;
	ledBlue = b;
	lastBrightness = masterBrightness;
	applyBrightness(r, g, b, masterBrightness);
	leds.neoPixelFill(r, g, b, false);
	ledUpdated = true;
}

void setMasterBrightness(u8 brightness) {
	masterBrightness = brightness;
	updateLed(ledRed, ledGreen, ledBlue);
}

bool onBrightnessEnter(MenuItem *_item) {
	u8 random = rand() % 256;
	u8 random2 = rand() % 64; // always at least some saturation left
	u8 r, g, b, h = random, s = 255, l = 128 + random2;
	hslToRgb(h, s, l, r, g, b);
	ledSetMode(LED_MODE::STATIC, LIGHT_ID::MENU, 0, r, g, b);
	return true;
}
bool onBrightnessExit(MenuItem *_item) {
	ledSetMode(LED_MODE::OFF, LIGHT_ID::MENU);
	return true;
}
void onBrightnessChange(MenuItem *_item) {
	setMasterBrightness(brightnessMenu);
}

void writeLed() {
	static elapsedMicros lastUpdate;
	if (ledUpdated && lastUpdate > 2000) {
		ledUpdated = false;
		leds.neoPixelShow();
	}
}

void ledInit() {
	leds.neoPixelClear(true);
}

void ledLoop() {
	if (ledDuration > 0 && ledTimer > ledDuration) {
		ledSetMode(LED_MODE::OFF, (LIGHT_ID)255);
	}
	struct LedProgram *program;
	if (rp2040.fifo.pop_nb((uint32_t *)&program)) {
		ledSetMode(program->mode, program->id, program->duration, program->r, program->g, program->b);
		free(program);
	}
	switch (ledMode) {
	case LED_MODE::OFF:
	case LED_MODE::STATIC:
		break;
	case LED_MODE::BLINK:
		if (ledTimer % 500 < 250) {
			updateLed(progRed, progGreen, progBlue);
		} else {
			updateLed(0, 0, 0);
		}
		break;
	case LED_MODE::BLINK_FAST:
		if (ledTimer % 250 < 125) {
			updateLed(progRed, progGreen, progBlue);
		} else {
			updateLed(0, 0, 0);
		}
		break;
	case LED_MODE::FADE: {
		// 1s black, 1s fade in, 1s fade out
		u16 progress = (ledTimer % 3000) * 255 / 1000;
		if (progress < 255) {
			updateLed(progRed * progress / 255, progGreen * progress / 255, progBlue * progress / 255);
		} else if (progress < 510) {
			u16 p = 510 - progress;
			updateLed(progRed * p / 255, progGreen * p / 255, progBlue * p / 255);
		} else {
			updateLed(0, 0, 0);
		}
	} break;
	case LED_MODE::RAINBOW: {
		u8 progress = (ledTimer % 5000) * 255 / 5000;
		u8 h = progress, s = 255, l = 128, r, g, b;
		hslToRgb(h, s, l, r, g, b);
		updateLed(r, g, b);
	} break;
	}
	writeLed();
}

void ledSetMode2(LED_MODE mode, LIGHT_ID id, u16 duration, u8 r, u8 g, u8 b, bool overrideSame) {
	struct LedProgram *program = (struct LedProgram *)malloc(sizeof(struct LedProgram));
	program->mode = mode;
	program->id = id;
	program->duration = duration;
	program->r = r;
	program->g = g;
	program->b = b;
	rp2040.fifo.push_nb((uint32_t)program);
}

void ledSetMode(LED_MODE mode, LIGHT_ID id, u16 duration, u8 r, u8 g, u8 b, bool overrideSame) {
	if (mode >= LED_MODE::MODE_LENGTH) {
		return;
	}
	if (rp2040.cpuid() == 1) {
		ledSetMode2(mode, id, duration, r, g, b, overrideSame);
		return;
	}
	if (overrideSame ? id < lightId : id <= lightId) {
		return;
	}
	lightId = id;
	if (lightId == (LIGHT_ID)255) {
		lightId = (LIGHT_ID)0;
	}
	ledDuration = duration;
	ledTimer = 0;
	ledMode = mode;
	progRed = r;
	progGreen = g;
	progBlue = b;
	switch (mode) {
	case LED_MODE::OFF:
		updateLed(0, 0, 0);
		break;
	case LED_MODE::STATIC:
		updateLed(r, g, b);
		break;
	case LED_MODE::BLINK:
		updateLed(r, g, b);
		break;
	case LED_MODE::BLINK_FAST:
		updateLed(r, g, b);
		break;
	case LED_MODE::FADE:
		updateLed(0, 0, 0);
		break;
	case LED_MODE::RAINBOW:
		updateLed(255, 0, 0);
		break;
	}
}

void releaseLightId(LIGHT_ID id) {
	if (id == lightId) {
		ledSetMode(LED_MODE::OFF, (LIGHT_ID)255);
	}
}

void ledNewClock() {
	leds.recalculateClock();
}

#endif