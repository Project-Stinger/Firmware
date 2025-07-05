#include "Arduino.h"
#include "utils/typedefs.h"

enum FireMode : u8 {
	FIRE_SINGLE = 0,
	FIRE_BURST,
	FIRE_CONTINUOUS,
};
extern u8 fireMode;
extern bool burstKeepFiring;
extern u8 burstCount, pushCount;
extern u32 pushDuration, retractDuration;
extern u8 minPushDuration, minRetractDuration;
extern u8 dartsPerSecond;
#define FIRE_MODE_STRING_LENGTH 11
extern const char fireModeNames[3][FIRE_MODE_STRING_LENGTH];

void initPusher();
void deinitPusher();
#if HW_VERSION == 2
extern bool pusherEnabled;
void enablePusher();
constexpr fix32 SOLENOID_CURR_CONV = 3.3 / 4096 * 3075 / 700;
extern fix32 solenoidCurrent; // A
extern bool pusherFullyExtended;
extern bool pusherFullyRetracted;
extern bool autoPusherTiming;
extern elapsedMicros sinceExtend;
extern u8 dpsLimit;
void showDpsOrLimit(MenuItem *_item);
#endif
void pusherLoop();
void onFireModeChange(MenuItem *_item);
void calcPushDurations(MenuItem *_item);
void updateMaxDps(MenuItem *_item);

void extendPusher();
void dischargePusher();
void retractPusher();
