#include "Arduino.h"
#include "menu/menuItem.h"
#include "utils/filters.h"
#include "utils/fixedPointInt.h"
#include "utils/typedefs.h"

extern u8 joystickCalState;

#define GESTURE_INIT_WAIT 700
#define GESTURE_REPEAT_WAIT 80
#define GESTURE_CENTER_LARGE_PCT 70
#define GESTURE_CENTER_SMALL_PCT 40
#if HW_VERSION == 1
#define JOYSTICK_LPF_CUTOFF_HZ 7
#elif HW_VERSION == 2
#define JOYSTICK_LPF_CUTOFF_HZ 30
#endif
#define JOYSTICK_SCHMITT_BARRIER_DEG 7

enum JoystickCalState {
	JOYSTICK_CAL_START = 0,
	JOYSTICK_CAL_CENTER_TOP = 0,
	JOYSTICK_CAL_CENTER_BOTTOM,
	JOYSTICK_CAL_CENTER_LEFT,
	JOYSTICK_CAL_CENTER_RIGHT,
	JOYSTICK_CAL_LIMITS,
	JOYSTICK_CAL_CONFIRM,
	JOYSTICK_CAL_DONE,
	JOYSTICK_CAL_ABORT,
};

enum class Direction : u8 {
	LEFT = 0,
	DOWN_LEFT,
	DOWN,
	DOWN_RIGHT,
	RIGHT,
	UP_RIGHT,
	UP,
	UP_LEFT,
	NONE = 255,
};

enum GestureType : u8 {
	GESTURE_PRESS = 0,
	GESTURE_RELEASE,
	GESTURE_HOLD,
};

typedef struct gesture {
	i16 angle; // 0 to 360, 0 = left, 90 = down, 180 = right, 270 = up
	u32 duration; // in ms
	u32 count; // number of times gesture was detected
	Direction direction;
	u8 type; // 0 = press, 1 = release, 2 = hold
	u8 prevType;
	bool operator==(const gesture &other) const {
		return memcmp(this, &other, sizeof(gesture)) == 0;
	}
	bool operator!=(const gesture &other) const {
		return memcmp(this, &other, sizeof(gesture)) != 0;
	}
} Gesture;

extern Gesture lastGesture;
extern volatile bool gestureUpdated;
extern i32 joystickXPos;
extern i32 joystickYPos;
extern fix32 joystickAngle;
extern i16 joystickAngleDeg;
extern i32 joystickMagnitude;
extern PT1 joystickXPosRaw;
extern PT1 joystickYPosRaw;
extern i16 joystickTravelAngleDeg;
extern fix32 joystickTravelAngle;
extern i16 joystickRotationTicks;
extern u8 lastJoystickCalDrawState;

bool calJoystick(MenuItem *item = nullptr);
bool startJcal(MenuItem *item = nullptr);

void joystickInit(bool fromReboot);
void joystickLoop();
void putJoystickValsInEeprom();
void drawJoystickCalibration(MenuItem *item = nullptr);