#include "Arduino.h"
#include "utils/typedefs.h"

extern volatile u32 operationState;
extern volatile u32 forceNewOpState;
extern u8 selectedProfile;
extern bool joystickLockout;
extern bool idleOnlyWithMag;
#if HW_VERSION == 1
extern bool idleEnabled;
#elif HW_VERSION == 2
extern u8 idleEnabled;
extern u8 mlIdleMode;
extern u8 mlThresholdPct;
extern u8 maxFireAngleSetting;
extern fix32 actualDps;

// UI helper: returns whether an ML confidence value is available and writes 0–100.
bool mlIdleGetConfidencePct(u8 *outPct);
#endif
extern bool bootUnlockNeeded;
extern u16 rampdownTime, rampupTimeout;
extern u8 timeoutMode;
extern const char timeoutModeStrings[2][6];
extern u8 rpmInRangeTime;
extern volatile u32 motorDisableFlags;
extern elapsedMillis menuOverrideTimer;
extern elapsedMillis menuOverrideRpmTimer;
extern i32 menuOverrideRpm;
extern bool previewIdlingInMenu;
extern i16 menuOverrideEsc[4];
extern u8 enabledProfiles;
extern u8 bootProgress;
extern bool newProfileFlag;
extern i16 profileSelectionJoystickAngle;
extern bool blasterHasFired;
extern elapsedMillis inactivityTimer;
extern u8 inactivityTimeout;
extern u32 thisRampupDuration;
extern u16 pusherDecay;
extern bool updatingDartCount;
extern bool stallDetectionEnabled;
extern u16 revAfterFire;

// motor disabled flags
#define MD_ESC_OVERTEMP (1UL << 0)
#define MD_NO_TELEMETRY (1UL << 1)
#define MD_BATTERY_EMPTY (1UL << 2)
#define MD_MOTORS_BLOCKED (1UL << 3)

#define PROFILE_SELECTION_ANGLE_LOW -10
#define PROFILE_SELECTION_ANGLE_HIGH 190
#define PROFILE_SELECTION_RANGE (PROFILE_SELECTION_ANGLE_HIGH - PROFILE_SELECTION_ANGLE_LOW)

enum OperationState {
	STATE_SETUP = 0, // waiting for first ESC RPMs etc.
	STATE_SAFE, // after boot, require some action to unlock blaster
	STATE_MENU, // menu mode: motors off
	STATE_OPEN_MENU, // menu mode: motors off
	STATE_PROFILE_SELECT, // profile selection: motors remain off or at idle
	STATE_OFF, // waiting for trigger
	STATE_RAMPUP, // motors on, ramping up
	STATE_PUSH, // motors on, solenoid push
	STATE_RETRACT, // motors on, solenoid retract
	STATE_RAMPDOWN, // motors on, ramping down
	STATE_JOYSTICK_CAL, // calibrate joystick
	STATE_FALL_DETECTED, // fall detected
	STATE_BOOT_SELECT, // select the boot mode: normal, joystick calibration, update, ...
	STATE_USB // USB connected: safe service mode (motors/pusher off)
};

void runOperationSm();

void setIdleState(MenuItem *_item);
