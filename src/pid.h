#include "Arduino.h"
#include "utils/typedefs.h"

#define setAllThrottles(x) \
	{                      \
		throttles[0] = x;  \
		throttles[1] = x;  \
		throttles[2] = x;  \
		throttles[3] = x;  \
	}

#define RPM_D_FILTER_CUTOFF 100
#define RPM_FILTER_CUTOFF 50
#define RPM_SHIFT 13 // stay within 32 bit for the gain
#define PID_P_SHIFT 8
#define PID_I_SHIFT 6
#define PID_D_SHIFT 13

extern u16 motorKv;
extern i16 throttles[4];
extern i32 targetRpm;
extern i32 idleRpm;
extern i32 rearRpm;
extern u8 frontRearRatio; // 0 = front == rear, 100 = front == 2x rear, 0 ... 150
extern i32 targetFrontLowThres, targetFrontHighThres, targetRearLowThres, targetRearHighThres;
extern i16 pGainNice, iGainNice, dGainNice;
extern u8 minThrottle;
extern u8 rpmThresPct;
#ifdef USE_BLACKBOX
extern i32 rpmIndex;
void prepareBlackbox();
void checkPrintRpm();
void blSetFired();
#endif

void pidLoop(i32 rpmFront, i32 rpmRear);
void resetPid();
void applyPidSettings(MenuItem *_item);

void calcTargetRpms(MenuItem *_item);
void resetITermSet();
void calcITermRpms();
