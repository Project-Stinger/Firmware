#include "global.h"

volatile u32 operationState = STATE_SETUP;
volatile u32 forceNewOpState = 0xFFFFFFFFUL;
u8 selectedProfile = 0;
u8 enabledProfiles = 5;
bool idleOnlyWithMag = true;
bool joystickLockout = false;
#if HW_VERSION == 1
bool idleEnabled = false;
#define CHECK_IDLE_EN (idleEnabled && (!idleOnlyWithMag || (magPresent || !foundTof)))
#define CHECK_ANGLE_OK (true)
#elif HW_VERSION == 2
u8 idleEnabled = 0;
const fix32 IDLE_5_DEG = fix32(5.0 * PI / 180.0);
bool checkIdle() {
    static bool lastIdleEnabled = false;
    
    // Check if ML predict mode (mode 8)
    if (idleEnabled == 8) {
        // ML mode: only idle when in SPINNING state
        extern PreFireState preFireState;
        lastIdleEnabled = (preFireState == PREFIRE_STATE_SPINNING) && 
                          (!idleOnlyWithMag || (magPresent || !foundTof));
        return lastIdleEnabled;
    }
    
    // Original idle logic for other modes
    fix32 threshold = IDLE_5_DEG * idleEnabled + (lastIdleEnabled ? (IDLE_5_DEG / 3) : (-IDLE_5_DEG / 3));
    lastIdleEnabled = idleEnabled && (idleEnabled == 1 || pitch.abs() < threshold) && (!idleOnlyWithMag || (magPresent || !foundTof));
    return lastIdleEnabled;
}
#define CHECK_IDLE_EN (checkIdle())
u8 maxFireAngleSetting = 0;
#define CHECK_ANGLE_OK ((maxFireAngleSetting == 5) || pitch.abs() < fix32(10 * PI / 180.0) * (4 + maxFireAngleSetting))
fix32 actualDps = 0;
elapsedMillis sinceFirstPush = 0;
elapsedMillis safeFlipTimer = 4000;
#endif
u16 rampdownTime = 500;
u16 rampupTimeout = 1000;
u8 timeoutMode = 0; // 0 = just fire, 1 = rampdown
const char timeoutModeStrings[2][6] = {
	"Fire",
	"Abort",
};
u8 rpmInRangeTime = 10;
i16 profileSelectionJoystickAngle = 0;
elapsedMicros opStateTimer;
u32 lastState = -1;
volatile u32 motorDisableFlags = 0;
elapsedMillis edtEnableTimer = 0;
elapsedMillis menuOverrideTimer = 0;
bool previewIdlingInMenu = false;
elapsedMillis menuOverrideRpmTimer = 0;
i32 menuOverrideRpm = 0;
u8 inactivityTimeout = 10;
elapsedMillis inactivityTimer = 0;
i16 menuOverrideEsc[4] = {0};
bool blasterHasFired = false;
bool firstMenuRun = true;
u32 thisRampupDuration = 0;
u16 pusherDecay = 0;
bool updatingDartCount = false;
bool stallDetectionEnabled = true;
i32 stallDetectionCounter[4] = {0};

u8 bootProgress = 0;
bool bootUnlockNeeded = true;
bool newProfileFlag = false;

u16 revAfterFire = 0;

#if HW_VERSION == 1
#define BOOT_SEQUENCE_DIV 10
#elif HW_VERSION == 2
#define BOOT_SEQUENCE_DIV 2
#endif

#define CHECK_NO_EDT ((escStatusTimer[0] > 2500 || escStatusTimer[1] > 2500 || escStatusTimer[2] > 2500 || escStatusTimer[3] > 2500) && edtEnableTimer > 200)
#ifdef USE_TOF
#define CHECK_MAG_SETTINGS (!foundTof || dartCount || (fireWoDarts && (magPresent || fireWoMag)))
#else
#define CHECK_MAG_SETTINGS 1
#endif

// every time the motors are stopped, the EDT has to be reenabled
void sendEdtStart() {
	edtEnableTimer = 0;
	// docs are wrong, 8 are required, not 6. 10 are sent, because that aligns with other DSHOT commands
	for (u8 i = 0; i < 10; i++)
		pushToAllCommandBufs(DSHOT_CMD_EXTENDED_TELEMETRY_ENABLE);
}

void setIdleState(MenuItem *_item) {
	mainMenu->search("idleRpm")->setVisible(idleEnabled);
	mainMenu->search("previewIdlingInMenu")->setVisible(idleEnabled);
	DEBUG_PRINTF("idle: %d\n", idleEnabled);
}

void __not_in_flash_func(runOperationSm)() {
	static bool firstRun = true;
	if (forceNewOpState != 0xFFFFFFFFUL) {
		operationState = forceNewOpState;
		lastState = forceNewOpState;
		forceNewOpState = 0xFFFFFFFFUL;
		firstRun = true;
		opStateTimer = 0;
	}
	u32 opStateTime = opStateTimer;
	switch (operationState) {
	case STATE_SETUP: {
		static i32 goodSequences = -8;
		static u8 goodSequenceProgress = 0;
		static u8 edtProgress = 0;
		static u8 bootGraceTimeProgress = 0;
		static elapsedMillis escCheckTimer = 0;
#if HW_VERSION == 2
		if (firstRun) {
			ledSetMode(LED_MODE::RAINBOW, LIGHT_ID::BOOT, 10000);
		}
#endif
		// After 3 good sequences (total 1.5s), we can move on. A good sequence has at least one successful bidir dshot response like status
		bool isChecked = false;
		if (escCheckTimer > 500) {
			isChecked = true;
			escCheckTimer = 0;
			if (telemCounter[0] && telemCounter[1] && telemCounter[2] && telemCounter[3]) {
				telemCounter[0] = telemCounter[1] = telemCounter[2] = telemCounter[3] = 0;
				if (goodSequences < 0) {
					if (goodSequences <= -8)
						addBootMsg("Found ESCs");
					goodSequences = 0;
				}
				goodSequences++;
				goodSequenceProgress = goodSequences * 13;
				if (goodSequences == 3)
					addBootMsg("ESCs Ready");
				if (goodSequenceProgress > 3 * 13) {
					goodSequenceProgress = 3 * 13;
				}
			} else {
				if (goodSequences > 0)
					goodSequences = 0;
				goodSequences--;
				if (goodSequences == -8)
					addBootMsg("ESC Lost Comm");
			}
			gestureUpdated = false;
			edtProgress = 0;
			bool escStatusOk = true;
			static bool printedEdtMsg = false;
			for (u8 i = 0; i < 4; i++) {
				if (escStatusCount[i] >= 2)
					edtProgress += 8;
				else
					escStatusOk = false; // require at least 2 status frames from each ESC
			}
			if (escStatusOk && !printedEdtMsg) {
				addBootMsg("EDT found");
				printedEdtMsg = true;
			}
			static bool printedTrigMsg = false;
			if (
				goodSequences >= 6 && bootTimer > 3000 && !triggerState && escStatusOk) {
				operationState = bootUnlockNeeded ? STATE_SAFE : STATE_OFF;
#if HW_VERSION == 2
				releaseLightId(LIGHT_ID::BOOT);
#endif
				triggerUpdateFlag = false;
			} else if (
				goodSequences >= 6 && bootTimer > 3000 && triggerState && escStatusOk && !printedTrigMsg) {
				addBootMsg("Release Trigger to continue");
				printedTrigMsg = true;
			}
		}
		bootGraceTimeProgress = bootTimer * 29 / 3000;
		if (bootGraceTimeProgress > 29)
			bootGraceTimeProgress = 29;
		static bool printedTimeoutMsg = false;
		if (!printedTimeoutMsg && bootTimer > 7000) {
			printedTimeoutMsg = true;
			addBootMsg("Timeout in 3s. Check Battery and ESCs.");
		}
		if (bootTimer > 10000 && !triggerState) {
			operationState = bootUnlockNeeded ? STATE_SAFE : STATE_OFF;
#if HW_VERSION == 2
			releaseLightId(LIGHT_ID::BOOT);
#endif
			triggerUpdateFlag = false;
		}
		i32 timeoutProgress = ((i32)bootTimer - 6000) / 40;
		if (timeoutProgress < 0)
			timeoutProgress = 0;
		if (edtEnableTimer >= 500) {
			sendEdtStart();
		}
		//  39 + 32 + 29 = 100
		u8 bootProgressX = goodSequenceProgress + edtProgress + bootGraceTimeProgress;
		bootProgress = MAX(timeoutProgress, bootProgressX);
		if (isChecked) {
			DEBUG_PRINTF("Total: %d, X: %d, Sequences: %d, EDT: %d, Grace: %d, timeout: %d\n", bootProgress, bootProgressX, goodSequenceProgress, edtProgress, bootGraceTimeProgress, timeoutProgress);
		}
		setAllThrottles(0);
	} break;
	case STATE_MENU: {
#if HW_VERSION == 2
		static u8 pusherCycles = 50;

		if (firstRun) {
			ledSetMode(LED_MODE::OFF, LIGHT_ID::MENU);
		}

		if (pusherCycles <= 50)
			pusherCycles++;
		if (pusherCycles == 10) {
			dischargePusher();
		}
		if (pusherCycles == 15) {
			retractPusher();
		}
		if (MenuItem::settingsSolenoidClickFlag) {
			MenuItem::settingsSolenoidClickFlag = false;
			pusherCycles = 0;
			extendPusher();
		}
#endif
		if (openedMenu == nullptr && !firstBoot) {
			operationState = STATE_OFF;
#if HW_VERSION == 2
			releaseLightId(LIGHT_ID::MENU);
#endif
			triggerUpdateFlag = false;
			firstMenuRun = true;
			resetPid();
			sendEdtStart();
			DEBUG_PRINTSLN("STATE_OFF");
		}
		if (menuOverrideTimer <= 100) {
			memcpy(throttles, menuOverrideEsc, sizeof(menuOverrideEsc));
		} else if (menuOverrideRpmTimer <= 100) {
			pidLoop(menuOverrideRpm, menuOverrideRpm);
		} else {
			setAllThrottles(0);
		}
	} break;
	case STATE_OPEN_MENU: {
		if (firstMenuRun || (CHECK_NO_EDT && menuOverrideTimer < 100)) {
			firstMenuRun = false;
			sendEdtStart();
		}
		if (openedMenu != nullptr || opStateTime > 100000) {
			mainMenu->init();
			operationState = STATE_MENU;
		}
		setAllThrottles(0);
	} break;
	case STATE_PROFILE_SELECT: {
		// keep idle RPM or off motors, waiting for profile selection
		// go to off or idle state when profile is selected
		// show profile selection on the screen
		if (CHECK_IDLE_EN) {
			pidLoop(idleRpm, idleRpm);
		} else {
			resetPid();
			setAllThrottles(0);
		}
		if (gestureUpdated && lastGesture.type == GESTURE_RELEASE) {
			// load profile from last run, if the joystick was returned to the center
			gestureUpdated = false;
			operationState = STATE_OFF;
			triggerUpdateFlag = false;
			if (!CHECK_IDLE_EN)
				sendEdtStart();
			DEBUG_PRINTSLN("STATE_OFF");
			break;
		}
		// reference joystickAngle to PROFILE_SELECTION_ANGLE_LOW
		i16 joystickAngleNew = (joystickAngleDeg + (360 - PROFILE_SELECTION_ANGLE_LOW)) % 360;
		// autocorrect joystick angle to the nearest profile selection angle
		if (joystickAngleNew >= PROFILE_SELECTION_RANGE) {
			// calculate profile selection angle range and center of off-limits area
			if (joystickAngleNew > PROFILE_SELECTION_RANGE / 2 + 180)
				joystickAngleNew = 0;
			else
				joystickAngleNew = PROFILE_SELECTION_RANGE - 1;
		}
		profileSelectionJoystickAngle = joystickAngleNew;
		u8 newProfile = joystickAngleNew * enabledProfiles / PROFILE_SELECTION_RANGE;
		if (newProfile != selectedProfile && (!joystickLockout || triggerState)) {
			bool wasIdling = CHECK_IDLE_EN;
			selectedProfile = newProfile;
			loadSettings();
			newProfileFlag = true;
			if (wasIdling && !CHECK_IDLE_EN) {
				sendEdtStart();
			}
#if HW_VERSION == 2
			ledSetMode(LED_MODE::STATIC, LIGHT_ID::HOMESCREEN, 0, profileColor[0], profileColor[1], profileColor[2]);
#endif
			DEBUG_PRINTF("Profile %d\n", selectedProfile);
		}
	} break;
	case STATE_OFF: {
		static u8 pusherCycles = 50;
		static u8 dcUpdateStart = 0;
		static elapsedMillis joystickLockTimer = 0;
		static bool enteringJoystickLock = false;
		if (firstRun) {
			updatingDartCount = false;
#if HW_VERSION == 2
			ledSetMode(LED_MODE::STATIC, LIGHT_ID::HOMESCREEN, 0, profileColor[0], profileColor[1], profileColor[2]);
#endif
		}
		if (gestureUpdated) {
			gestureUpdated = false;
			if (lastGesture.type == GESTURE_PRESS && lastGesture.prevType == GESTURE_RELEASE && !updatingDartCount) {
				if (joystickAngleDeg <= PROFILE_SELECTION_ANGLE_HIGH || joystickAngleDeg >= 360 + PROFILE_SELECTION_ANGLE_LOW) {
					operationState = STATE_PROFILE_SELECT;
					DEBUG_PRINTSLN("STATE_PROFILE_SELECT");
				} else if (lastGesture.angle < 260 && lastGesture.angle > PROFILE_SELECTION_ANGLE_HIGH + 20) {
					// top right: menu
					if ((!tournamentMode || !tournamentBlockMenu) && !joystickLockout) {
						operationState = STATE_OPEN_MENU;
						DEBUG_PRINTSLN("STATE_OPEN_MENU");
					}
				} else if (lastGesture.angle >= 260 && lastGesture.angle <= 280) {
					joystickLockTimer = 0;
					enteringJoystickLock = true;
#if HW_VERSION == 2
					if (joystickLockout && MenuItem::settingsBeep) {
						makeRtttlSound("unlock:d=4,o=5,b=60:12f,12f#,12g,12g#,12a,15a#,32p,44a#,44f");
						speakerLoopOnFastCore2 = true;
					}
#endif
				} else if (lastGesture.angle > 280 && lastGesture.angle < 360 + PROFILE_SELECTION_ANGLE_LOW - 20) {
					// top left: adjust dartCount
					if (!joystickLockout) {
						updatingDartCount = true;
						dcUpdateStart = dartCount;
					}
				}
			} else if (lastGesture.type == GESTURE_HOLD && lastGesture.angle >= 260 && lastGesture.angle <= 280 && enteringJoystickLock && joystickLockTimer > 2000) {
				joystickLockout = !joystickLockout;
				enteringJoystickLock = false;
#if HW_VERSION == 2
				if (joystickLockout && MenuItem::settingsBeep) {
					makeRtttlSound("lock:d=4,o=5,b=650:4f,4a#");
					speakerLoopOnFastCore2 = true;
				}
#endif
			} else if (lastGesture.type == GESTURE_RELEASE && enteringJoystickLock) {
				enteringJoystickLock = false;
#if HW_VERSION == 2
				if (joystickLockout && MenuItem::settingsBeep) {
					stopSound();
					speakerLoopOnFastCore2 = false;
				}
#endif
			} else if (updatingDartCount && lastGesture.type == GESTURE_RELEASE) {
				updatingDartCount = false;
			}
		}
		if (pusherCycles <= 50)
			pusherCycles++;
		if (pusherCycles == 10) {
			dischargePusher();
		}
		if (pusherCycles == 15) {
			retractPusher();
		}
		if (updatingDartCount) {
#if HW_VERSION == 2
			i32 lastDc = dartCount;
#endif
			dartCount = dcUpdateStart - joystickRotationTicks;
			dartCount = constrain(dartCount, 0, 99);
			if (dcUpdateStart - joystickRotationTicks < -3 * 360 / (32 - rotationTickSensitivity * 8)) {
// 5 full rotations below 0 darts
#if HW_VERSION == 1
				// V1 does not have a gyro, and therefore no safe flip detection
				operationState = STATE_SAFE;
#elif HW_VERSION == 2
				updatingDartCount = false;
				enableStandbyFlag = true;
#endif
			}
#if HW_VERSION == 2
			if (lastDc != dartCount) {
				makeSound(1500, 8);
				extendPusher();
				pusherCycles = 0;
			}
#endif
		}

#if HW_VERSION == 2
		if (pitch > FIX_DEG_TO_RAD * 65) {
			safeFlipTimer = 0;
		}
		if (safeFlipTimer < 4000 && pitch.abs() < FIX_DEG_TO_RAD * 15 && roll.abs() > FIX_DEG_TO_RAD * 155) {
			operationState = STATE_SAFE;
			makeRtttlSound("safe:d=4,o=6,b=150:8d7,8d6,8g6");
			safeFlipTimer = 4000;
			DEBUG_PRINTSLN("STATE_SAFE");
		}
#endif

		if (CHECK_IDLE_EN) {
			pidLoop(idleRpm, idleRpm);
			if (inactivityTimer > 1000 * 60 * inactivityTimeout && inactivityTimeout) setAllThrottles(0);
		} else {
			setAllThrottles(0);
			if (CHECK_NO_EDT || firstRun) {
				sendEdtStart();
			}
		}
#if HW_VERSION == 2
		if (freeFallDetected) {
			operationState = STATE_FALL_DETECTED;
			break;
		}
#endif
		if (triggerUpdateFlag) {
			triggerUpdateFlag = false;
			blasterHasFired = true;
			if (triggerState && CHECK_MAG_SETTINGS && !updatingDartCount && !motorDisableFlags && CHECK_ANGLE_OK) {
				resetPid();
#ifdef USE_BLACKBOX
				rpmIndex = -1;
				prepareBlackbox();
#endif
				resetITermSet();
				calcITermRpms();
				operationState = STATE_RAMPUP;
				DEBUG_PRINTSLN("STATE_RAMPUP");
			}
		}
	} break;
	case STATE_RAMPUP: {
		// run PID loop to ramp up motors to target RPM
		static elapsedMillis goodTimer;
		bool rpmGood = true;
		pidLoop(targetRpm, rearRpm);
		if (escRpm[0] < targetFrontLowThres || escRpm[0] > targetFrontHighThres // comment to prevent formatting
			|| escRpm[1] < targetFrontLowThres || escRpm[1] > targetFrontHighThres // comment to prevent formatting
			|| escRpm[2] < targetRearLowThres || escRpm[2] > targetRearHighThres // comment to prevent formatting
			|| escRpm[3] < targetRearLowThres || escRpm[3] > targetRearHighThres) {
			goodTimer = 0;
			rpmGood = false;
		}
		if (triggerUpdateFlag && opStateTime < 3000) {
			// minimum trigger pull duration of 3ms
			// if shorter, send back to STATE_OFF
			triggerUpdateFlag = false;
			if (!triggerState) {
				operationState = STATE_OFF;
				DEBUG_PRINTSLN("STATE_OFF");
			}
		}
		bool timeout = opStateTime >= rampupTimeout * 1000;
		if ((rpmGood && goodTimer >= rpmInRangeTime && !escErpmFail) || (!timeoutMode && timeout)) {
			operationState = STATE_PUSH;
			DEBUG_PRINTSLN("STATE_PUSH");
#if HW_VERSION == 2
			sinceFirstPush = 0;
#endif
			extendPusher();
			thisRampupDuration = (opStateTime + 500) / 1000;
			pushCount = 0;
		} else if (timeout) {
			thisRampupDuration = 0xFFFFFFFFUL;
			pushCount = 0;
			operationState = STATE_RAMPDOWN;
			DEBUG_PRINTSLN("STATE_RAMPDOWN");
		}
	} break;
	case STATE_PUSH: {
		// run PID loop to keep motors at target RPM, turn on solenoid
		// go to retract state when trigger is released
		pidLoop(targetRpm, rearRpm);
#if HW_VERSION == 1
		if (opStateTime >= pushDuration)
#elif HW_VERSION == 2
		if (autoPusherTiming ? pusherFullyExtended : (opStateTime >= pushDuration))
#endif
		{
			if (dartCount) dartCount--;
			operationState = STATE_RETRACT;
			dischargePusher();
			DEBUG_PRINTSLN("STATE_RETRACT");
		}
	} break;
	case STATE_RETRACT: {
		// run PID loop to keep motors at target RPM, turn off solenoid
		// if continuous mode or burst is enabled, go to push state once retracted
		// go to rampdown state when trigger is released or in single shot mode
		pidLoop(targetRpm, rearRpm);
		if (opStateTime >= 3100) retractPusher();
#if HW_VERSION == 1
		if (opStateTime >= retractDuration)
#elif HW_VERSION == 2
		if (autoPusherTiming ? (pusherFullyRetracted && sinceExtend > 1000000 / dpsLimit) : (opStateTime >= retractDuration))
#endif
		{
			pushCount++;
			triggerUpdateFlag = false; // clear update flag to prevent immediate refire
			switch (fireMode) {
			case FIRE_CONTINUOUS:
				if (triggerState) {
					operationState = STATE_PUSH;
					extendPusher();
					DEBUG_PRINTSLN("STATE_PUSH");
				} else {
#if HW_VERSION == 2
					actualDps = fix32(pushCount) / (i32)sinceFirstPush * 1000;
#endif
					operationState = STATE_RAMPDOWN;
					DEBUG_PRINTSLN("STATE_RAMPDOWN");
				}
				break;
			case FIRE_BURST:
				if (pushCount < burstCount || (triggerState && burstKeepFiring)) {
					operationState = STATE_PUSH;
					extendPusher();
					DEBUG_PRINTSLN("STATE_PUSH");
				} else {
#if HW_VERSION == 2
					actualDps = fix32(pushCount) / (i32)sinceFirstPush * 1000;
#endif
					operationState = STATE_RAMPDOWN;
					DEBUG_PRINTSLN("STATE_RAMPDOWN");
				}
				break;
			case FIRE_SINGLE:
			default:
				operationState = STATE_RAMPDOWN;
				DEBUG_PRINTSLN("STATE_RAMPDOWN");
				break;
			}
		}
	} break;
	case STATE_RAMPDOWN: {
		// slowly decrease target RPM to idle RPM
		// go to idle state when idle RPM is reached, or to rampup state if trigger is pulled
		i32 progress = opStateTime / 1000;
		if (progress < revAfterFire)
			progress = 0;
		else
			progress -= revAfterFire;
		i32 finalRpm = idleEnabled ? idleRpm : 0;
		i32 targetFront = targetRpm + (finalRpm - targetRpm) * progress / rampdownTime;
		i32 targetRear = rearRpm + (finalRpm - rearRpm) * progress / rampdownTime;
		pidLoop(targetFront, targetRear);
		if (triggerState && triggerUpdateFlag) {
			triggerUpdateFlag = false;
			if (opStateTime > pusherDecay * 1000 && CHECK_ANGLE_OK && CHECK_MAG_SETTINGS && !motorDisableFlags) {
				operationState = STATE_RAMPUP;
				DEBUG_PRINTSLN("STATE_RAMPUP");
			}
		}
		if (progress >= rampdownTime) {
			operationState = STATE_OFF;
#ifdef USE_BLACKBOX
			rpmIndex = 10000;
#endif
			DEBUG_PRINTSLN("STATE_OFF");
		}
	} break;
	case STATE_JOYSTICK_CAL: {
		if (calJoystick()) {
			operationState = STATE_OFF;
			DEBUG_PRINTSLN("STATE_OFF");
			resetPid();
		}
		if (CHECK_NO_EDT) sendEdtStart();
		setAllThrottles(0);
	} break;
	case STATE_FALL_DETECTED:
	case STATE_SAFE: {
		static bool wasReleased = false;
		if (firstRun) {
			wasReleased = false;
#if HW_VERSION == 2
			ledSetMode(LED_MODE::FADE, LIGHT_ID::HOMESCREEN, 0, 255, 0, 0);
#endif
		}
		if (lastGesture.type == GESTURE_RELEASE) {
			wasReleased = true;
		}
		setAllThrottles(0);
		retractPusher();
		if (wasReleased && lastGesture.type != GESTURE_RELEASE && (joystickTravelAngleDeg >= 180 || joystickTravelAngleDeg <= -180)) {
			if (operationState == STATE_FALL_DETECTED) {
				operationState = STATE_OFF;
				triggerUpdateFlag = false;
#if HW_VERSION == 2
				freeFallDetected = false;
				makeRtttlSound("fallReset:d=4,o=5,b=1000:d,d#,e,f,f#,g");
#endif
			} else if (operationState == STATE_SAFE) {
				operationState = STATE_OFF;
				triggerUpdateFlag = false;
#if HW_VERSION == 2
				makeRtttlSound("unlock:d=4,o=5,b=650:4f,4a#");
#endif
			}
		}
	} break;
	case STATE_BOOT_SELECT: {
		runBootSelect();
	} break;
	}

	u8 unlock = 0;
	for (i32 i = 0; i < 4; i++) {
		if (!throttles[i] || motorDisableFlags & ~MD_MOTORS_BLOCKED) {
			stallDetectionCounter[i] = -PID_RATE / 2; // allowance of 100ms+100ms=200ms
			unlock |= 1 << i;
		} else {
			if (!escRpm[i]) {
				stallDetectionCounter[i] += 5;
				if (stallDetectionCounter[i] >= PID_RATE / 2) { // 100ms
					if (stallDetectionEnabled && !(motorDisableFlags & MD_MOTORS_BLOCKED)) {
						DEBUG_PRINTSLN("stall");
						motorDisableFlags |= MD_MOTORS_BLOCKED;
					}
					stallDetectionCounter[i] = PID_RATE / 2;
				}
			} else {
				if (stallDetectionCounter[i] <= 0) {
					unlock |= 1 << i;
					if (stallDetectionCounter[i] < 0) stallDetectionCounter[i]++; // allowance fully gone after 500ms
				} else {
					stallDetectionCounter[i]--; // falloff takes 0.5s
				}
			}
		}
	}
	if (unlock == 0xF && motorDisableFlags & MD_MOTORS_BLOCKED) {
		DEBUG_PRINTSLN("unlock");
		motorDisableFlags &= ~MD_MOTORS_BLOCKED;
	}

#ifdef PRINT_DEBUG
	static u32 debugCounter = 0;
	if ((++debugCounter) == PID_RATE * 5) {
		debugCounter = 0;
		if (motorDisableFlags) {
			DEBUG_PRINTF("motor disabled: x%X\n", motorDisableFlags);
			extern elapsedMillis lastTelemetryFrame[4];
			if (motorDisableFlags & MD_NO_TELEMETRY) {
				DEBUG_PRINTF("Last Telemetry: %d %d %d %d\n", (i32)lastTelemetryFrame[0], (i32)lastTelemetryFrame[1], (i32)lastTelemetryFrame[2], (i32)lastTelemetryFrame[3]);
			}
		}
	}
#endif

	static bool firstMotorDisable = true;
	if (motorDisableFlags) {
		if ((CHECK_NO_EDT || firstMotorDisable) && !(throttles[0] || throttles[1] || throttles[2] || throttles[3])) {
			firstMotorDisable = false;
			sendEdtStart();
		}
		setAllThrottles(0);
	} else {
		firstMotorDisable = true;
	}
#if HW_VERSION == 1
	// beep when inactive, V2 has standby for this
	static elapsedMillis lastBeacon = 0;
	if (inactivityTimer > 1000 * 60 * inactivityTimeout && lastBeacon > 5000 && inactivityTimeout) {
		lastBeacon = 0;
		pushToAllCommandBufs(DSHOT_CMD_BEACON4);
	}
#endif
	sendThrottles(throttles);
	firstRun = false;
	if (lastState != operationState) {
		lastState = operationState;
		opStateTimer = 0;
		firstRun = true;
	}
#ifdef USE_BLACKBOX
	checkPrintRpm();
#endif
}
