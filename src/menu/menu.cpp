#include "git_version.h"
#include "global.h"

MenuItem *mainMenu = new MenuItem(MenuItemType::SUBMENU, "main", "Main Menu");
MenuItem *firstBootMenu = new MenuItem(MenuItemType::SUBMENU, "firstBoot", "First Boot");
MenuItem *onboardingMenu = new MenuItem(MenuItemType::SUBMENU, "onboarding", "Onboarding");

MenuItem *openedMenu = nullptr;
u8 copyToProfile = 0;

char profileName[16] = "Profile";
u8 profileColor[3] = {255, 255, 255};
u16 profileColor565 = 0xFFFF;
bool extendedRpmRange = false;

u8 rotationTickSensitivity = 0;
const char rotationSensitivityStrings[3][10] = {"Slow", "Medium", "Fast"};

#if HW_VERSION == 1
#define DEFAULT_PID_P 50
#define DEFAULT_PID_I 30
#define DEFAULT_PID_D 60
#define DEFAULT_IN_RANGE_MS 10
#define DEFAULT_PRECISION 10
#elif HW_VERSION == 2
#define DEFAULT_PID_P 60
#define DEFAULT_PID_I 12
#define DEFAULT_PID_D 18
#define DEFAULT_IN_RANGE_MS 1
#define DEFAULT_PRECISION 5
#endif

void onExtRpmChange(MenuItem *_item) {
	mainMenu->search("rpm")->setMax(extendedRpmRange ? 80000 : DEFAULT_MAX_RPM);
}

void loadSettings() {
#if HW_VERSION == 2
	speakerLoopOnFastCore = true;
#endif
	mainMenu->init();
	applyTournamentLimits();
#if HW_VERSION == 2
	speakerLoopOnFastCore = false;
#endif
}

void updateProfileColor(MenuItem *_item) {
	profileColor565 = tft.color565(profileColor[0], profileColor[1], profileColor[2]);
}

bool firstBootLoop(MenuItem *item) {
	if (triggerUpdateFlag) {
		triggerUpdateFlag = false;
		if (triggerState && bootTimer >= 1000) {
			item->onExit();
			openedMenu = nullptr;
			startJcal(nullptr);
		}
	}
	return true;
}
bool firstBootEnter(MenuItem *_item) {
#if HW_VERSION == 2
	ledSetMode(LED_MODE::RAINBOW, LIGHT_ID::FIRSTBOOT);
#endif
	return true;
}
bool firstBootExit(MenuItem *_item) {
#if HW_VERSION == 2
	releaseLightId(LIGHT_ID::FIRSTBOOT);
#endif
	return true;
}

bool saveAndClose(MenuItem *item) {
#if HW_VERSION == 1
	if (!idleEnabled && MenuItem::settingsBeep)
		MenuItem::makeSettingsBeep(1);
#elif HW_VERSION == 2
	if (MenuItem::settingsBeep)
		makeRtttlSound("save:d=4,o=5,b=650:4f5,4a#5");
	speakerLoopOnFastCore = true;
#endif
	item->parent->save();
	putJoystickValsInEeprom();
	EEPROM.commit();
	item->parent->onExit();
	applyTournamentLimits();
	openedMenu = nullptr;
	if (item->parent->isRebootRequired()) {
		rebootReason = BootReason::MENU;
		rp2040.reboot();
	}
#if HW_VERSION == 2
	speakerLoopOnFastCore = false;
#endif
	return false;
}

bool discardAndClose(MenuItem *item) {
#if HW_VERSION == 1
	if (!idleEnabled && MenuItem::settingsBeep)
		MenuItem::makeSettingsBeep(1);
#elif HW_VERSION == 2
	if (MenuItem::settingsBeep)
		makeRtttlSound("discard:d=4,o=5,b=650:4f5,4d5,4a#4");
	speakerLoopOnFastCore = true;
#endif
	joystickInit(false);
	loadSettings();
	item->parent->onExit();
	openedMenu = nullptr;
	if (item->parent->isRebootRequired()) {
		rebootReason = BootReason::MENU;
		rp2040.reboot();
	}
#if HW_VERSION == 2
	speakerLoopOnFastCore = false;
#endif
	return false;
}

bool jumpToSaveOption(MenuItem *item) {
	if (lastGesture.type == GESTURE_PRESS)
		item->setFocusedChild("save");
#if HW_VERSION == 1
	MenuItem::scheduleBeep(SETTINGS_BEEP_PERIOD, 1);
#elif HW_VERSION == 2
	MenuItem::beep(SETTINGS_BEEP_MIN_FREQ + SETTINGS_BEEP_FREQ_RANGE / 3);
#endif
	return false;
}

bool exitParent(MenuItem *item) {
	if (item->parent == nullptr) return true;
	item->parent->onExit();
	if (item->parent == openedMenu) {
		openedMenu = nullptr;
		bootTimer = 0;
		operationState = STATE_SETUP;
	}
	return true;
}

void copySwapProfileCheck(MenuItem *_item) {
	// copyToProfile is 1 indexed
	mainMenu->search("copyProfileAction")->setVisible((copyToProfile - 1) != selectedProfile);
	mainMenu->search("swapProfileAction")->setVisible((copyToProfile - 1) != selectedProfile);
}
bool copyProfile(MenuItem *item) {
#if HW_VERSION == 2
	speakerLoopOnFastCore = true;
#endif
	mainMenu->save();
	EEPROM.commit();
	for (u16 i = EEPROM_PROFILE_SETTINGS_START; i < EEPROM_PROFILE_SETTINGS_START + PROFILE_EEPROM_SIZE; i++) {
		u8 data;
		EEPROM.get(selectedProfile * PROFILE_EEPROM_SIZE + i, data);
		// copyToProfile is 1 indexed
		EEPROM.put((copyToProfile - 1) * PROFILE_EEPROM_SIZE + i, data);
	}
	EEPROM.commit();
	exitParent(item);
#if HW_VERSION == 2
	speakerLoopOnFastCore = false;
#endif
	return false;
}
bool swapProfiles(MenuItem *item) {
#if HW_VERSION == 2
	speakerLoopOnFastCore = true;
#endif
	mainMenu->save();
	EEPROM.commit();
	u8 tempProfile[PROFILE_EEPROM_SIZE];
	u8 tempProfile2[PROFILE_EEPROM_SIZE];
	EEPROM.get(selectedProfile * PROFILE_EEPROM_SIZE, tempProfile);
	EEPROM.get((copyToProfile - 1) * PROFILE_EEPROM_SIZE, tempProfile2);
	EEPROM.put(selectedProfile * PROFILE_EEPROM_SIZE, tempProfile2);
	EEPROM.put((copyToProfile - 1) * PROFILE_EEPROM_SIZE, tempProfile);
	EEPROM.commit();
	loadSettings();
	exitParent(item);
#if HW_VERSION == 2
	speakerLoopOnFastCore = false;
#endif
	return false;
}

bool preventDefault(MenuItem *_item) {
	return false;
}

void initMenu() {
	// ======================== Motor Menu ========================
	MenuItem *motorMenu = new MenuItem(MenuItemType::SUBMENU, "motor", "Motors");
	MenuItem *motorGoToExpertMenu = new MenuItem(MenuItemType::SUBMENU, "motorGoToExpert", "Expert");
	motorMenu
		->addChild(new MenuItem(VariableType::I32, &targetRpm, 30000, 500, 10000, DEFAULT_MAX_RPM, 1, 0, EEPROM_POS_TARGET_RPM_FRONT, true, "rpm", "Target RPM Front", "RPM Target for front wheels during firing operation"))
#if HW_VERSION == 1
		->addChild(new MenuItem(&idleEnabled, false, EEPROM_POS_IDLE_ENABLED, true, "idleEn", "Idling", "Leave motors running during idle state, decreases rampup time"))
#elif HW_VERSION == 2
		->addChild(new MenuItem(&idleEnabled, 9, EEPROM_POS_IDLE_ENABLED, 9, (const char *)idleStrings, 9, true, "idleEn", "Idling", "Leave motors running during idle state, decreases rampup time"))
		->addChild(new MenuItem(&mlIdleMode, 1, EEPROM_POS_ML_IDLE_MODE, 1, (const char *)mlIdleModeStrings, 9, true, "mlIdleMode", "ML Idle Mode", "Binary = idle on/off. Dynamic = scale RPM with model probability."))
		->addChild(new MenuItem(VariableType::U8, &mlThresholdPct, 50, 1, 5, 95, 1, 0, EEPROM_POS_ML_THRESHOLD_PCT, true, "mlThresh", "ML Threshold %", "Probability threshold. Lower = earlier/more aggressive idling."))
#endif
#ifdef USE_TOF
		->addChild(new MenuItem(&idleOnlyWithMag, true, EEPROM_POS_IDLE_ONLY_WITH_MAG, false, "idleOnlyWithMag", HW_VERSION == 2 ? "Idle only with mag" : "No mag no idle", "Preview the idling RPM in the menu"))
#endif
		->addChild(new MenuItem(&previewIdlingInMenu, false, EEPROM_RUNTIME_OPTION, false, "previewIdlingInMenu", "Preview Idle RPM", "Preview the idling RPM in the menu"))
		->addChild(new MenuItem(VariableType::I32, &idleRpm, 5000, 100, 1000, 20000, 1, 0, EEPROM_POS_IDLE_RPM, true, "idleRpm", "Idle RPM", "PID Target during idle operation", 0, false, false))
		->addChild(new MenuItem(VariableType::U8, &frontRearRatio, 20, 5, 0, 150, 100, 2, EEPROM_POS_FRONT_REAR_RATIO, true, "frRatio", "F/R Ratio", "Ratio by which the rear RPM is lower than the front RPM", 100))
		->addChild(motorGoToExpertMenu);
	MenuItem *motorExpertTakeMeBack = new MenuItem(MenuItemType::ACTION, "motorNoExpert", "Take me back");
	motorExpertTakeMeBack->setOnEnterFunction(exitParent);
	MenuItem *motorExpertMenu = new MenuItem(MenuItemType::SUBMENU, "motorExpert", "I know what I am doing.");
	motorGoToExpertMenu
		->addChild(new MenuItem(MenuItemType::INFO, "motorExpertInfo", "Warning: Changing these settings can lead to damage to the motors, ESCs and the battery. Proceed at your own risk."))
		->addChild(motorExpertTakeMeBack)
		->addChild(motorExpertMenu);
	MenuItem *motorExpertPage2 = new MenuItem(MenuItemType::SUBMENU, "motorExpertPage2", "Next Page");
	motorExpertMenu
		->addChild(new MenuItem(VariableType::U8, &minThrottle, 14, 2, 0, 200, 20, 1, EEPROM_POS_MIN_THROTTLE, false, "minThrottle", "Min Throttle %", "Minimum throttle for the motors to run at (unless they are off), e.g. to overcome glitching at low throttle. Overrides whatever the PID says."))
		->addChild(new MenuItem(VariableType::I16, &pGainNice, DEFAULT_PID_P, 1, 0, 500, 1, 0, EEPROM_POS_PID_P, false, "pGain", "P Gain", "Proportional Gain for the PID controller, higher values make the system react faster but can lead to oscillations and overshoot"))
		->addChild(new MenuItem(VariableType::I16, &iGainNice, DEFAULT_PID_I, 1, 0, 500, 1, 0, EEPROM_POS_PID_I, false, "iGain", "I Gain", "Integral Gain for the PID controller, higher values make the system react faster to long term errors and new setpoints but can lead to overshoot and low frequency oscillations"))
		->addChild(new MenuItem(VariableType::I16, &dGainNice, DEFAULT_PID_D, 1, 0, 500, 1, 0, EEPROM_POS_PID_D, false, "dGain", "D Gain", "Derivative Gain for the PID controller, higher values can reduce overshoot and oscillations but can lead to noise amplification, i.e. hot motors, and instability"))
		->addChild(new MenuItem(VariableType::U8, &rpmInRangeTime, DEFAULT_IN_RANGE_MS, 1, 0, 100, 1, 0, EEPROM_POS_RPM_IN_RANGE_TIME, true, "rpmInRangeTime", "RPM in Range ms", "Time for which the motors have to be within the target RPM range before firing"))
		->addChild(new MenuItem(VariableType::I8, &rpmThresPct, DEFAULT_PRECISION, 1, 3, 20, 1, 0, EEPROM_POS_RPM_THRES_PCT, true, "rpmThresPct", "RPM Tolerance %", "Percentage of the target RPM within which the motors are considered to be in range"))
		->addChild(new MenuItem(VariableType::U16, &rampupTimeout, 300, 50, 200, 2000, 1, 0, EEPROM_POS_RAMPUP_TIMEOUT, true, "rampupTimeout", "Rampup Timeout", "Timeout in ms after which the setting in Timeout Mode takes place, regardless of the RPM being in range, e.g. stuck motor"))
		->addChild(new MenuItem(&timeoutMode, 0, EEPROM_POS_TIMEOUT_MODE, 1, (const char *)timeoutModeStrings, 6, false, "timeoutMode", "Timeout Mode", "Action to take when the rampup timeout is reached"))
		->addChild(new MenuItem(VariableType::U16, &motorKv, 3750, 50, 2000, 6000, 1, 0, EEPROM_POS_MOTOR_KV, false, "motorKv", "Motor KV", "Motor KV rating, used to calculate the maximum RPM for estimated final throttle."))
		->addChild(motorExpertPage2)
		->setOnExitFunction(exitParent);
	motorExpertPage2
		->addChild(new MenuItem(VariableType::U16, &rampdownTime, 1000, 50, 200, 3000, 1, 0, EEPROM_POS_RAMPDOWN_SPEED, true, "rampdownTime", "Rampdown ms", "Time to ramp down the motors to idle/off after firing"))
		->addChild(new MenuItem(VariableType::U16, &revAfterFire, 0, 50, 0, 2000, 1, 0, EEPROM_POS_REV_AFTER_FIRE, true, "revAfterFire", "Rev Hold Time", "Time in ms that the blaster will keep revving for after firing, useful for quick single trigger pulls"))
		->addChild(new MenuItem(VariableType::U8, &escMaxTemp, 100, 1, 60, 255, 1, 0, EEPROM_POS_ESC_MAX_TEMP, false, "maxEscTemp", "Max. ESC Temp", "In degrees Celsius, temperature above which the ESCs will be disabled. The ESC may have its own thermal protection, this can be used to lower that limit, but not extend it."))
		->addChild(new MenuItem(MenuItemType::ACTION, "motorExpertBackTo1", "Previous Page"));
	motorExpertPage2->search("motorExpertBackTo1")->setOnEnterFunction(exitParent);

	// ======================== Firing Menu ========================
	MenuItem *firingMenu = new MenuItem(MenuItemType::SUBMENU, "firing", "Firing");
	MenuItem *firingExpertMenu = new MenuItem(MenuItemType::SUBMENU, "firingExpert", "Expert");
	firingMenu
		->addChild(new MenuItem(&fireMode, FIRE_CONTINUOUS, EEPROM_POS_FIRE_MODE, 2, (const char *)fireModeNames, FIRE_MODE_STRING_LENGTH, true, "fireMode", "Fire Mode", "Mode of operation for the pusher, semi-auto = fixed number of darts, auto = until trigger is released"))
#if HW_VERSION == 2
		->addChild(new MenuItem(VariableType::U8, &dpsLimit, 40, 1, 2, 40, 1, 0, EEPROM_POS_DPS_LIMIT, true, "dpsLimit", "Maximum DPS", "Maximum darts per second the pusher will fire, used for auto timing"))
#endif
		->addChild(new MenuItem(VariableType::U8, &dartsPerSecond, HW_VERSION == 2 ? 30 : 18, 1, 2, 20, 1, 0, EEPROM_POS_DARTS_PER_SECOND, true, "dartsPerSecond", "Darts/s", "Number of darts to fire per second in continuous or burst mode"))
		->addChild(new MenuItem(VariableType::U8, &burstCount, 3, 1, 2, 6, 1, 0, EEPROM_POS_BURST_COUNT, true, "burstCount", "Burst Count", "Number of darts to fire in burst mode"))
		->addChild(new MenuItem(&burstKeepFiring, false, EEPROM_POS_BURST_KEEP_FIRE, true, "burstKeepFiring", "Keep Firing", "If enabled, the blaster will keep firing after the burst count is reached until the trigger is released. If disabled, it will stop after the burst count."))
#ifdef USE_TOF
		->addChild(new MenuItem(VariableType::U8, &magSize, 15, 1, 7, 30, 1, 0, EEPROM_POS_MAG_SIZE, false, "magSize", "Mag. Capacity", "Number of darts in a magazine"))
#endif
		->addChild(firingExpertMenu);
	firingExpertMenu
#ifdef USE_TOF
		->addChild(new MenuItem(&fireWoMag, true, EEPROM_POS_FIRE_WO_MAG, true, "fireWoMag", "Fire w/o mag", "Allow firing without a magazine inserted"))
		->addChild(new MenuItem(&fireWoDarts, true, EEPROM_POS_FIRE_EMPTY, true, "fireEmpty", "Fire w/o darts", "Allow firing when the magazine is empty"))
#endif
		->addChild(new MenuItem(VariableType::U16, &pusherDecay, 50, 10, 0, 500, 1, 0, EEPROM_POS_PUSHER_DECAY, false, "pusherDecay", "Pusher Decay ms", "Time in ms during rampdown, where the pusher will not activate again, even if the trigger is pulled another time"))
#if HW_VERSION == 2
		->addChild(new MenuItem(&autoPusherTiming, true, EEPROM_POS_AUTO_TIMING, true, "autoTiming", "Auto Timing", "Automatically adjust the pusher timing based on the darts per second setting"))
		->addChild(new MenuItem(MenuItemType::INFO, "manualTimingInfo", "Minimal times for manual timing:"))
#endif
		->addChild(new MenuItem(VariableType::U8, &minPushDuration, 15, 1, 10, 200, 1, 0, EEPROM_POS_PUSH_DURATION, false, "pushDuration", "Push ms", "Duration in ms the pusher is active for each dart"))
		->addChild(new MenuItem(VariableType::U8, &minRetractDuration, HW_VERSION == 2 ? 18 : 40, 1, 10, 200, 1, 0, EEPROM_POS_RETRACT_DURATION, false, "retractDuration", "Min. Retract ms", "Minimum duration in ms the pusher is inactive after each dart"));

	// ======================== Battery Menu ========================
	MenuItem *batteryMenu = new MenuItem(MenuItemType::SUBMENU, "battery", HW_VERSION == 2 ? "Battery & Standby" : "Battery");
	batteryMenu
		->addChild(new MenuItem(&batCellsSettings, 0, EEPROM_POS_BAT_CELLS, 4, (const char *)cellSettings, 9, false, "batCells", "Cell Count", "Number of cells in the battery pack"))
		->addChild(new MenuItem(VariableType::U8, &batWarnVoltage, 65, 1, 50, 80, 100, 2, EEPROM_POS_BAT_WARN, false, "batWarn", "Warning Voltage", "Voltage below which a warning is displayed", 300))
		->addChild(new MenuItem(VariableType::U8, &batShutdownVoltage, 50, 1, 40, 60, 100, 2, EEPROM_POS_BAT_SHUTDOWN, false, "batShutdown", "Shutdown Voltage", "Voltage at which the blaster will stop spinning temporarily", 300))
		->addChild(new MenuItem(VariableType::U8, &inactivityTimeout, 10, 1, 0, 30, 1, 0, EEPROM_POS_INACTIVITY_TIMEOUT, false, "inactivityTimeout", HW_VERSION == 2 ? "Standby timeout" : "Inactivity Mins", "0 = Off. If the user does not interact for set time, the blaster will stop idling and beep."))
#if HW_VERSION == 2
		->addChild(new MenuItem(&fastStandbyEnabled, true, EEPROM_POS_FAST_STANDBY, false, "fastStandby", "Fast Standby", "Blaster will go to standby after 1 minute without motion"))
#endif
		->addChild(new MenuItem(VariableType::I8, &batCalibrationOffset, 0, 1, -100, 100, 100, 2, EEPROM_POS_BAT_OFFSET, false, "batVoltOffset", HW_VERSION == 2 ? "Voltage Calibration" : "Voltage Calib.", "Offset for the battery management. Positive values will make the blaster measure higher voltages."))
		->addChild(new MenuItem(MenuItemType::CUSTOM, "storageMode", "Storage Mode", "Spin the motors down to storage voltage"));

	// ======================== User Interface Settings ========================
	MenuItem *interfaceMenu = new MenuItem(MenuItemType::SUBMENU, "interface", "User Interface");
	interfaceMenu
		->addChild(new MenuItem(&MenuItem::settingsBeep, HW_VERSION == 2 ? true : false, EEPROM_POS_SETTINGS_BEEP, false, "settingsBeep", "Settings Beep", "Beep when navigating through the menu"))
#if defined(USE_TOF) && HW_VERSION == 2
		->addChild(new MenuItem(&beepOnMagChange, true, EEPROM_POS_BEEP_ON_MAG_CHANGE, false, "beepOnMagChange", "Mag. insert sound", "Beep when the magazine is inserted or removed"))
#endif
		->addChild(new MenuItem(&rotationTickSensitivity, 1, EEPROM_POS_ROTATION_SENSITIVITY, 2, (const char *)rotationSensitivityStrings, 10, false, "rotationTickSensitivity", HW_VERSION == 2 ? "Joystick dial speed" : "Joystick dial", "Sensitivity of dial, when loading up the dart counter or changing menu values via the joystick"))
#if HW_VERSION == 2
		->addChild(new MenuItem(VariableType::U8, &brightnessMenu, 255, 5, 0, 255, 1, 0, EEPROM_POS_LED_BRIGHTNESS, false, "ledBrightness", "LED Brightness", "Brightness of the LED"))
#endif
		;

	// ======================== Profile Settings ========================
	MenuItem *profileMenu = new MenuItem(MenuItemType::SUBMENU, "profile", "Profile");
	MenuItem *copyProfileMenu = new MenuItem(MenuItemType::SUBMENU, "copyProfileMenu", "Copy Profile");
	MenuItem *swapProfileMenu = new MenuItem(MenuItemType::SUBMENU, "swapProfileMenu", "Swap Profiles");
	profileMenu
		->addChild(new MenuItem(VariableType::U8, &enabledProfiles, 3, 1, 2, MAX_PROFILE_COUNT, 1, 0, EEPROM_POS_PROFILE_COUNT, false, "enabledProfiles", "# of profiles", "Number of enabled profiles to choose from on the main screen"))
		->addChild(new MenuItem(profileName, 16, "Profile", EEPROM_POS_PROFILE_NAME, true, "profileName", "Name"))
		->addChild(new MenuItem(VariableType::U8, &profileColor[0], 180, 5, 0, 255, 1, 0, EEPROM_POS_PROFILE_COLOR_RED, true, "profileColorR", "Red"))
		->addChild(new MenuItem(VariableType::U8, &profileColor[1], 255, 5, 0, 255, 1, 0, EEPROM_POS_PROFILE_COLOR_GREEN, true, "profileColorG", "Green"))
		->addChild(new MenuItem(VariableType::U8, &profileColor[2], 180, 5, 0, 255, 1, 0, EEPROM_POS_PROFILE_COLOR_BLUE, true, "profileColorB", "Blue"))
		->addChild(copyProfileMenu)
		->addChild(swapProfileMenu);
	copyProfileMenu
		->addChild(new MenuItem(MenuItemType::INFO, "copyProfileInfo", "Copy the current profile to another profile, including all expert settings, while saving all the current changes."))
		->addChild(new MenuItem(VariableType::U8, &copyToProfile, 1, 1, 1, MAX_PROFILE_COUNT, 1, 0, EEPROM_RUNTIME_OPTION, false, "copyToProfile", "Copy to "))
		->addChild(new MenuItem(MenuItemType::ACTION, "copyProfileAction", "Copy"));
	swapProfileMenu
		->addChild(new MenuItem(MenuItemType::INFO, "swapProfileInfo", "Swap the current profile with another profile, including all expert settings, while saving all the current changes."))
		->addChild(new MenuItem(VariableType::U8, &copyToProfile, 1, 1, 1, MAX_PROFILE_COUNT, 1, 0, EEPROM_RUNTIME_OPTION, false, "swapProfile", "Swap with "))
		->addChild(new MenuItem(MenuItemType::ACTION, "swapProfileAction", "Swap"));

	// ======================== Device Menu ========================
	MenuItem *deviceMenu = new MenuItem(MenuItemType::SUBMENU, "device", "Device");
	MenuItem *bootScreen = new MenuItem(MenuItemType::SUBMENU, "bootScreen", "Boot Screen");
	MenuItem *safetyMenu = new MenuItem(MenuItemType::SUBMENU, "safetyMenu", "Safety");
	MenuItem *hardwareMenu = new MenuItem(MenuItemType::SUBMENU, "hardwareMenu", "Hardware Setup");
	MenuItem *flywheelBalancing = new MenuItem(MenuItemType::CUSTOM, "flywheelBalancing", "Flywheel Tester");
	MenuItem *remapMotors = new MenuItem(MenuItemType::CUSTOM, "remapMotors", "Remap Motors");
	MenuItem *escConfigMenu = new MenuItem(MenuItemType::SUBMENU, "escConfigMenu", "ESC Config");
	MenuItem *escTempCalibration = new MenuItem(MenuItemType::SUBMENU, "escTempCalibration", "ESC Temp Calibration");
	MenuItem *firmwareInfoMenu = new MenuItem(MenuItemType::SUBMENU, "firmwareInfoMenu", "Firmware Info");
	MenuItem *resetMenu = new MenuItem(MenuItemType::SUBMENU, "resetMenu", "Factory Reset");
	deviceMenu
		->addChild(bootScreen)
		->addChild(safetyMenu)
		->addChild(hardwareMenu)
		->addChild(new MenuItem(MenuItemType::CUSTOM, "docs", "Documentation"))
		->addChild(firmwareInfoMenu)
		->addChild(resetMenu)
#if HW_VERSION == 2
		->addChild(new MenuItem(MenuItemType::ACTION, "mlRecording", "ML Recording", "Start recording IMU data to flash for ML training"))
#endif
		;
	bootScreen
		->addChild(new MenuItem(deviceName, 16, "Stinger", EEPROM_POS_DEVICE_NAME, false, "deviceName", "Device Name"))
		->addChild(new MenuItem(ownerName, 32, "John Doe", EEPROM_POS_OWNER_NAME, false, "ownerName", "Owner"))
		->addChild(new MenuItem(ownerContact, 32, "john.doe@example.com", EEPROM_POS_OWNER_CONTACT, false, "ownerContact", "Contact"))
#if HW_VERSION == 2
		->addChild(new MenuItem(&startSoundId, 1, EEPROM_POS_STARTUP_SOUND, 5, (const char *)soundNames, 20, false, "startupSound", "Startup Sound", "Sound played at boot"))
#endif
		;
	safetyMenu
		->addChild(new MenuItem(&bootUnlockNeeded, true, EEPROM_POS_BOOT_SAFE, false, "bootUnlockNeeded", "SAFE after boot", "Locks the blaster before finishing boot"))
#if HW_VERSION == 2
		->addChild(new MenuItem(&freeFallDetectionEnabled, true, EEPROM_POS_FALL_DETECTION, false, "fallDetection", "Fall Detection", "Enable fall detection, the blaster will stop idling and prevent firing if it detects a fall"))
		->addChild(new MenuItem(&maxFireAngleSetting, 4, EEPROM_POS_FIRE_ANGLE_LIMIT, 5, (const char *)fireAngleStrings, 9, false, "fireAngle", "Max Fire Angle", "Maximum angle at which the blaster will fire, to prevent darts from falling out"))
#endif
		->addChild(new MenuItem(&extendedRpmRange, false, EEPROM_POS_EXTENDED_RPM_RANGE, false, "extendedRpmRange", "Allow higher RPM", "Enable up to 80k RPM. Internal testing has shown that the firing speed is less consistent at these speeds, as the darts enter dynamic friction. Wear on darts is increased however."))
		->addChild(new MenuItem(&stallDetectionEnabled, true, EEPROM_POS_STALL_DETECTION, false, "stallDetection", "Stall Detection", "Enable stall detection, which will stop the motors if they are blocked or stalled."));
	hardwareMenu
		->addChild(new MenuItem(MenuItemType::CUSTOM, "jcal", "Joystick Calibration"))
		->addChild(flywheelBalancing)
		->addChild(remapMotors)
		->addChild(escConfigMenu)
		->addChild(new MenuItem(MenuItemType::CUSTOM, "calibrateTof", "Magazine Detection"))
		->addChild(escTempCalibration)
		->addChild(new MenuItem(MenuItemType::CUSTOM, "inputDiagnostics", "Input Diagnostics"));
	resetMenu
#if HW_VERSION == 2
		->setOnExitFunction(rickroll)
#endif
		->addChild(new MenuItem(MenuItemType::INFO, "resetInfo", "This will reset the whole blaster to the factory default. Continue?"))
		->addChild(new MenuItem(MenuItemType::ACTION, "cancelReset", "No, cancel"))
		->addChild(new MenuItem(MenuItemType::ACTION, "resetAction", "Yes, reset"));
	remapMotors
		->setCustomLoop(remapLoop)
		->setOnEnterFunction(startRemap)
		->setOnExitFunction(stopRemap);
	flywheelBalancing
		->setOnEnterFunction(enterBalancing)
		->setCustomLoop(balancingLoop);
	escConfigMenu
#if HW_VERSION == 1
		->addChild(new MenuItem(MenuItemType::INFO, "escConfigInfo", "You are about to reboot into ESC configuration mode. In this mode, you can use software like BLHeliSuite32 to configure the ESCs. All current changes will be saved."))
#elif HW_VERSION == 2
		->addChild(new MenuItem(MenuItemType::INFO, "escConfigInfo", "You are about to reboot into ESC configuration mode. In this mode, you can use software like AM32 Configurator to configure the ESCs. All current changes will be saved."))
#endif
		->addChild(new MenuItem(MenuItemType::ACTION, "startEscConfig", "Ok"))
		->addChild(new MenuItem(MenuItemType::ACTION, "exitEscConfig", "Back"));
	escTempCalibration
		->addChild(new MenuItem(MenuItemType::INFO, "escTempCalibrationInfo", "Only run this function when the blaster has fully cooled down to a known temperature. Then run this function quickly after turning on the blaster."))
		->addChild(new MenuItem(VariableType::U8, &knownEscTemp, 23, 1, 10, 35, 1, 0, EEPROM_RUNTIME_OPTION, false, "knownEscTemp", "Temperature °C", "The temperature of the ESCs at the time of calibration."))
		->addChild(new MenuItem(MenuItemType::ACTION, "startEscTempCalibration", "Start Calibration"))
		->addChild(new MenuItem(MenuItemType::INFO, "escTempCalSuccess", "Calibration successful!"))
		->addChild(new MenuItem(MenuItemType::INFO, "escTempCalError1", "Calibration failed! Check that the blaster is cool."))
		->addChild(new MenuItem(MenuItemType::INFO, "escTempCalError2", "Calibration failed! Blaster needs LiPo power."))
		->addChild(new MenuItem(MenuItemType::INFO, "escTempCalError3", "No temp sensor found!"));
	firmwareInfoMenu
		->addChild(new MenuItem(MenuItemType::INFO, "hardwareVersion", "Stinger V" HW_VERSION_STRING))
		->addChild(new MenuItem(MenuItemType::INFO, "firmwareVersion", "Firmware Version: " FIRMWARE_VERSION_STRING))
		->addChild(new MenuItem(MenuItemType::INFO, "firmwareDate", __DATE__ " " __TIME__))
		->addChild(new MenuItem(MenuItemType::INFO, "gitHash", "Git #: " GIT_HASH))
		->addChild(new MenuItem(MenuItemType::ACTION, "updateFirmwareAction", "Enter FW Update Mode"));

	// ======================== Tournament Mode ========================
	MenuItem *tournamentMenu = new MenuItem(MenuItemType::SUBMENU, "tournament", "Tournament Mode", "");
	MenuItem *tournamentPage2 = new MenuItem(MenuItemType::SUBMENU, "tournamentPage2", "Next Page");
	tournamentMenu
		->addChild(new MenuItem(VariableType::U32, &tournamentMaxRpm, DEFAULT_MAX_RPM, 1000, 20000, 80000, 1, 0, EEPROM_POS_TOURNAMENT_MAXRPM, false, "tournamentMaxRpm", "Max RPM", "Maximum RPM allowed in tournament mode"))
		->addChild(new MenuItem(VariableType::U8, &tournamentMaxDps, HW_VERSION == 2 ? 30 : 18, 1, 2, 40, 1, 0, EEPROM_POS_TOURNAMENT_MAXDPS, false, "tournamentMaxDps", "Max DPS", "Maximum darts per second allowed in tournament mode"))
		->addChild(new MenuItem(&allowSemiAuto, true, EEPROM_POS_TOURNAMENT_ALLOW_SEMI_AUTO, false, "allowSemiAuto", "Allow Semi-Auto", "Allow semi-auto in tournament mode"))
		->addChild(new MenuItem(&allowFullAuto, true, EEPROM_POS_TOURNAMENT_ALLOW_FULL_AUTO, false, "allowFullAuto", "Allow Full-Auto", "Allow full-auto in tournament mode"))
		->addChild(new MenuItem(&tournamentInvertScreen, false, EEPROM_POS_TOURNAMENT_INVERT_SCREEN, false, "tournamentInvertScreen", "Invert Screen", "Invert the screen in tournament mode"))
		->addChild(new MenuItem(&tournamentBlockMenu, false, EEPROM_POS_TOURNAMENT_BLOCK_MENU, false, "tournamentBlockMenu", HW_VERSION == 2 ? "Block Menu Access" : "Block Menu", "Menu access can be blocked in tournament mode"))
		->addChild(tournamentPage2);
	tournamentPage2
		->addChild(new MenuItem(MenuItemType::INFO, "tournamentInfo", "This will enable Tournament Mode. To exit tournament mode, power the blaster via USB ONLY. Factory reset is disabled."))
		->addChild(new MenuItem(MenuItemType::INFO, "tournamentInfo2", "Plug in a battery to enter tournament mode."))
		->addChild(new MenuItem(MenuItemType::ACTION, "enterTournamentMode", "Enter Tournament Mode"));

	// ======================== Main Menu ========================
	mainMenu
		->addChild(motorMenu)
		->addChild(firingMenu)
		->addChild(batteryMenu)
		->addChild(interfaceMenu)
		->addChild(profileMenu)
		->addChild(deviceMenu)
		->addChild(tournamentMenu)
		->addChild(new MenuItem(MenuItemType::ACTION, "save", "Save", "Save settings"))
		->addChild(new MenuItem(MenuItemType::ACTION, "discard", "Discard", "Discard changes"));

	// ======================== Special actions ========================
	mainMenu->search("save")->setOnEnterFunction(saveAndClose);
	mainMenu->search("discard")->setOnEnterFunction(discardAndClose);
	mainMenu->search("rpm")->setOnChangeFunction(calcTargetRpms);
	mainMenu->search("frRatio")->setOnChangeFunction(calcTargetRpms);
	mainMenu->search("idleEn")->setOnChangeFunction(setIdleState);
	mainMenu->setOnLeftFunction(jumpToSaveOption);
	mainMenu->search("fireMode")->setOnChangeFunction(onFireModeChange);
	mainMenu->search("resetAction")->setOnEnterFunction(clearEeprom);
	mainMenu->search("cancelReset")->setOnEnterFunction(exitParent);
	mainMenu->search("copyProfileAction")->setOnEnterFunction(copyProfile);
	mainMenu->search("copyToProfile")->setOnChangeFunction(copySwapProfileCheck);
	mainMenu->search("swapProfileAction")->setOnEnterFunction(swapProfiles);
	mainMenu->search("swapProfile")->setOnChangeFunction(copySwapProfileCheck);
	mainMenu->search("pGain")->setOnChangeFunction(applyPidSettings);
	mainMenu->search("iGain")->setOnChangeFunction(applyPidSettings);
	mainMenu->search("dGain")->setOnChangeFunction(applyPidSettings);
	mainMenu->search("batCells")->setOnChangeFunction(onBatSettingChange);
	mainMenu->search("batWarn")->setOnChangeFunction(onBatSettingChange);
	mainMenu->search("batShutdown")->setOnChangeFunction(onBatSettingChange);
	mainMenu->search("startEscConfig")->setOnEnterFunction(startEscPassthrough);
	mainMenu->search("exitEscConfig")->setOnEnterFunction(exitParent);
	mainMenu->search("calibrateTof")
		->setOnRightFunction(onTofCalibrationRight)
		->setOnLeftFunction(onTofCalibrationLeft)
		->setOnUpFunction(preventDefault)
		->setOnDownFunction(preventDefault)
		->setCustomDrawFull(drawTofCalibration)
		->setOnEnterFunction(startTofCalibration)
		->setOnExitFunction(onTofCalibrationExit);
	mainMenu->search("escTempCalibration")
		->setOnEnterFunction([](MenuItem *_item) {
			showTempCalResult(-1);
			return true;
		});
	mainMenu->search("startEscTempCalibration")->setOnEnterFunction(calibrateEscTemp);
	mainMenu->search("inputDiagnostics")
		->setOnRightFunction(onInputDiagRight)
		->setOnLeftFunction(onInputDiagLeft)
		->setOnUpFunction(preventDefault)
		->setOnDownFunction(preventDefault)
		->setCustomDrawFull(drawInputDiagnostics);
	mainMenu->search("updateFirmwareAction")
		->setOnEnterFunction([](MenuItem *_item) {
			rp2040.rebootToBootloader();
			return false;
		});
	mainMenu->search("storageMode")
		->setOnEnterFunction(storageModeStart)
		->setOnLeftFunction(onStorageLeft)
		->setOnRightFunction(onStorageRight)
		->setOnUpFunction(onStorageUp)
		->setOnDownFunction(onStorageDown)
		->setCustomLoop(storageModeLoop);
	mainMenu->search("profileColorR")->setOnChangeFunction(updateProfileColor);
	mainMenu->search("profileColorG")->setOnChangeFunction(updateProfileColor);
	mainMenu->search("profileColorB")->setOnChangeFunction(updateProfileColor);
	mainMenu->search("jcal")->setOnEnterFunction(startJcal)->setCustomLoop(calJoystick)->setCustomDrawFull(drawJoystickCalibration);
	mainMenu->search("extendedRpmRange")->setOnChangeFunction(onExtRpmChange);
	mainMenu->search("dartsPerSecond")->setOnChangeFunction(calcPushDurations);
	mainMenu->search("pushDuration")->setOnChangeFunction(updateMaxDps);
	mainMenu->search("retractDuration")->setOnChangeFunction(updateMaxDps);
#if HW_VERSION == 2
	mainMenu->search("autoTiming")->setOnChangeFunction(showDpsOrLimit);
#endif
	mainMenu->search("docs")->setCustomDrawFull(onQrFullDraw);
	mainMenu->search("idleRpm")->setCustomLoop([](MenuItem *_item) {
		if (previewIdlingInMenu) {
			menuOverrideRpm = idleRpm;
			menuOverrideRpmTimer = 0;
		}
		return true;
	});
	mainMenu->search("rotationTickSensitivity")->setOnEnterFunction([](MenuItem *_item) {
		// only enter when the dial is not being used for rotation navigation
		if (MenuItem::enteredRotationNavigation) {
			MenuItem::enteredRotationNavigation = false;
			return false;
		}
		return true;
	});
	mainMenu->search("enterTournamentMode")
		->setOnEnterFunction(enableTournamentMode)
		->setVisible(false);
#if HW_VERSION == 2
	mainMenu->search("mlRecording")->setOnEnterFunction([](MenuItem *_item) -> bool {
		if (!mlLogIsActive()) {
			mlLogStartRecording();
		}
		return false;
	});
	mainMenu->search("ledBrightness")
		->setOnEnterFunction(onBrightnessEnter)
		->setOnExitFunction(onBrightnessExit)
		->setOnChangeFunction(onBrightnessChange);
#endif
	loadSettings();
	MenuItem::settingsAreInEeprom = true;

	firstBootMenu
		->setOnLeftFunction(preventDefault)
		->setOnRightFunction(preventDefault)
		->setOnUpFunction(preventDefault)
		->setOnDownFunction(preventDefault)
		->addChild(new MenuItem(MenuItemType::INFO, "firstBootInfo", "Welcome to Stinger!"))
		->addChild(new MenuItem(MenuItemType::INFO, "firstBootDesc", "You will now be guided through the setup process. Pull the trigger to start."))
		->setCustomLoop(firstBootLoop)
		->setOnEnterFunction(firstBootEnter)
		->setOnExitFunction(firstBootExit);
	onboardingMenu
		->setOnLeftFunction(preventDefault)
		->addChild(new MenuItem(MenuItemType::INFO, "onboardingInfo", "Initial Setup"))
		->addChild(mainMenu->search("deviceName"))
		->addChild(mainMenu->search("ownerName"))
		->addChild(mainMenu->search("ownerContact"))
		->addChild(mainMenu->search("enabledProfiles"))
		->addChild(new MenuItem(MenuItemType::CUSTOM, "docsOnboarding", "Documentation"))
		->addChild(new MenuItem(MenuItemType::ACTION, "saveOnboarding", "Save", "Save settings"));
	onboardingMenu->search("docsOnboarding")->setCustomDrawFull(onQrFullDraw);
	onboardingMenu->search("saveOnboarding")->setOnEnterFunction(saveAndClose);
}
