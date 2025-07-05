#include "Fonts/FreeSansBold12pt7b.h"
#include "global.h"

u8 eepromMajorVersion = 0;
u8 eepromMinorVersion = 0;
u8 eepromPatchVersion = 0;
bool firstBoot = false;

void adjustProfileDefaults() {
	MenuItem *save = mainMenu->search("save");

	// ================== Profile 1 ==================
	// Kinda slow and unsurprising, especially no idle enabled (it's the first when you start the device)
	selectedProfile = 0;
	loadSettings();
	strcpy(profileName, "Sting");
	profileColor[0] = 255;
	profileColor[1] = 95;
	profileColor[2] = 0;
	idleEnabled = 0;
	targetRpm = 25000;
	fireMode = FIRE_SINGLE;
	saveAndClose(save);

	// ================== Profile 2 ==================
	// Just shoot it
	selectedProfile = 1;
	loadSettings();
	strcpy(profileName, "Scorpion");
	profileColor[0] = 0;
	profileColor[1] = 255;
	profileColor[2] = 0;
	idleEnabled = HW_VERSION == 2 ? 5 : 1; // idle permanent on V1, +- 25° on V2
	targetRpm = 40000;
	fireMode = FIRE_CONTINUOUS;
	saveAndClose(save);

	// ================== Profile 3 ==================
	// Burst mode
	selectedProfile = 2;
	loadSettings();
	strcpy(profileName, "Wasp");
	profileColor[0] = 255;
	profileColor[1] = 255;
	profileColor[2] = 0;
	idleEnabled = HW_VERSION == 2 ? 5 : 1; // idle permanent on V1, +- 25° on V2
	fireMode = FIRE_BURST;
	targetRpm = 40000;
	burstCount = 3;
	saveAndClose(save);

	// ================== Profile 4 ==================
	// slow but continuous
	selectedProfile = 3;
	loadSettings();
	strcpy(profileName, "Bee");
	profileColor[0] = 255;
	profileColor[1] = 0;
	profileColor[2] = 255;
	idleEnabled = 0;
	fireMode = FIRE_CONTINUOUS;
	dartsPerSecond = 7;
#if HW_VERSION == 2
	dpsLimit = 7;
#endif
	saveAndClose(save);

	// ================== Profile 5 ==================
	// Single shot, super fast
	selectedProfile = 4;
	loadSettings();
	strcpy(profileName, "Hornet");
	profileColor[0] = 0;
	profileColor[1] = 0;
	profileColor[2] = 255;
	idleEnabled = HW_VERSION == 2 ? 4 : 1; // idle permanent on V1, +- 20° on V2
	fireMode = FIRE_SINGLE;
	targetRpm = DEFAULT_MAX_RPM;
	saveAndClose(save);
}

// only works for types that have one default value for all profiles,
template <typename T>
void eepromSetInAllProfiles(u32 eepromPos, T defaultValue) {
	for (u8 i = 0; i < MAX_PROFILE_COUNT; i++) {
		u32 addr = eepromPos + (i * PROFILE_EEPROM_SIZE);
		EEPROM.put(addr, defaultValue);
	}
}

void eepromMigrate(u8 major, u8 minor, u8 patch) {
	if (major == 1 && minor == 0 && patch == 0) {
		// version 2 does not have 1S and 2S anymore, so that now 0 = auto, 1 = 3S, 2 = 4S, 3 = 5S, 4 = 6S -> migrate 1S and 2S to 3S
		u8 cells;
		EEPROM.get(EEPROM_POS_BAT_CELLS, cells);
		if (cells == 1 || cells == 2) {
			cells = 3;
		}
		if (cells > 0) {
			cells -= 2;
		}
		EEPROM.put(EEPROM_POS_BAT_CELLS, cells);

		// new option: SAFE after boot, default is on
		bool safe = true;
		EEPROM.put(EEPROM_POS_BOOT_SAFE, safe);

		// new option: stall detection, default is on
		bool stall = true;
		EEPROM.put(EEPROM_POS_STALL_DETECTION, stall);

		// new option: battery calibration offset, default is 0
		i8 offset = 0;
		EEPROM.put(EEPROM_POS_BAT_OFFSET, offset);

		// new option: TOF calibration, set default values
		u8 tofLow = 55;
		u8 tofHigh = 65;
		EEPROM.put(EEPROM_POS_TOF_LOW, tofLow);
		EEPROM.put(EEPROM_POS_TOF_HIGH, tofHigh);

		// V2 has tournament mode
		bool tournament = false;
		u32 tournamentMaxRpm = DEFAULT_MAX_RPM;
		u8 tournamentMaxDps = 18;
		bool allowSemiAuto = true;
		bool allowFullAuto = true;
		bool tournamentInvertScreen = false;
		bool tournamentBlockMenu = false;
		EEPROM.put(EEPROM_POS_TOURNAMENT_ENABLED, tournament);
		EEPROM.put(EEPROM_POS_TOURNAMENT_MAXRPM, tournamentMaxRpm);
		EEPROM.put(EEPROM_POS_TOURNAMENT_MAXDPS, tournamentMaxDps);
		EEPROM.put(EEPROM_POS_TOURNAMENT_ALLOW_SEMI_AUTO, allowSemiAuto);
		EEPROM.put(EEPROM_POS_TOURNAMENT_ALLOW_FULL_AUTO, allowFullAuto);
		EEPROM.put(EEPROM_POS_TOURNAMENT_INVERT_SCREEN, tournamentInvertScreen);
		EEPROM.put(EEPROM_POS_TOURNAMENT_BLOCK_MENU, tournamentBlockMenu);

		// new option: timeout mode, default is 0 (just fire)
		u8 timeoutMode = 0;
		EEPROM.put(EEPROM_POS_TIMEOUT_MODE, timeoutMode);

		// migration to 2.0.0 complete
		major = 2;
		minor = 0;
		patch = 0;
	}
	if (major == 2 && minor == 0 && patch == 0) {
		// no changes in EEPROM contents from 2.0.0 to 2.0.1
		major = 2;
		minor = 0;
		patch = 1;
	}
	if (major == 2 && minor == 0 && patch == 1) {
		bool b = true;
		EEPROM.put(EEPROM_POS_IDLE_ONLY_WITH_MAG, b);
#if HW_VERSION == 2
		u8 maxTemp;
		EEPROM.get(EEPROM_POS_ESC_MAX_TEMP, maxTemp);
		if (maxTemp == 90) {
			maxTemp = 100;
			EEPROM.put(EEPROM_POS_ESC_MAX_TEMP, maxTemp);
		}
#endif
		major = 2;
		minor = 1;
		patch = 0;
	}
	if (major == 2 && minor == 1 && patch == 0) {
		eepromSetInAllProfiles(EEPROM_POS_REV_AFTER_FIRE, (u16)0);
		eepromSetInAllProfiles(EEPROM_POS_BURST_KEEP_FIRE, false);
		for (int i = 0; i < 4; i++) {
			EEPROM.put(EEPROM_POS_ESC_TEMP_OFFSETS + i, (i8)0);
		}

		major = 2;
		minor = 2;
		patch = 0;
	}
	if (major == 2 && minor == 2 && patch == 0) {
		// no changes in EEPROM contents from 2.2.0 to 2.2.1
		major = 2;
		minor = 2;
		patch = 1;
	}
	if (major == 2 && minor == 2 && patch == 1) {
		// no changes in EEPROM contents from 2.2.1 to 2.2.2
		major = 2;
		minor = 2;
		patch = 2;
	}

	eepromMajorVersion = FIRMWARE_VERSION_MAJOR;
	eepromMinorVersion = FIRMWARE_VERSION_MINOR;
	eepromPatchVersion = FIRMWARE_VERSION_PATCH;

	EEPROM.put(EEPROM_POS_VERSION_MAJOR, (u8)FIRMWARE_VERSION_MAJOR);
	EEPROM.put(EEPROM_POS_VERSION_MINOR, (u8)FIRMWARE_VERSION_MINOR);
	EEPROM.put(EEPROM_POS_VERSION_PATCH, (u8)FIRMWARE_VERSION_PATCH);
	EEPROM.commit();
}

void requestClearEeprom(u8 major, u8 minor, u8 patch) {
	// halt program indefinitely here, wait for trigger input, then clear EEPROM
	initDisplay();
	tft.fillScreen(ST77XX_BLACK);
#if HW_VERSION == 1
	const u8 y1 = 0;
	const u8 y2 = 16;
	const u8 y3 = 64;
	const u8 y4 = 72;
	SET_DEFAULT_FONT;
#elif HW_VERSION == 2
	const u8 y1 = 20;
	const u8 y2 = 40;
	const u8 y3 = 105;
	const u8 y4 = 120;
	tft.setFont(&FreeSansBold12pt7b);
#endif
	tft.setTextColor(ST77XX_WHITE);
	printCentered("Reset Stinger", SCREEN_WIDTH / 2, y1, SCREEN_WIDTH, 1, 22, ClipBehavior::PRINT_LAST_LINE_CENTERED);
	SET_DEFAULT_FONT;
	printCentered("You are coming from a newer firmware version. Please flash a newer version, or perform a firmware reset by pulling the trigger.", SCREEN_WIDTH / 2, y2, SCREEN_WIDTH, 6, YADVANCE, ClipBehavior::PRINT_LAST_LINE_CENTERED);
	char buf[64];
	snprintf(buf, 64, "Previously flashed: %d.%d.%d", major, minor, patch);
	printCentered(buf, SCREEN_WIDTH / 2, y3, SCREEN_WIDTH, 1, YADVANCE, ClipBehavior::PRINT_LAST_LINE_CENTERED);
	snprintf(buf, 64, "Currently flashed: " FIRMWARE_VERSION_STRING);
	printCentered(buf, SCREEN_WIDTH / 2, y4, SCREEN_WIDTH, 1, YADVANCE, ClipBehavior::PRINT_LAST_LINE_CENTERED);
	triggerInit();
	sleep_ms(1000);
	triggerUpdateFlag = false;
	while (true) {
		triggerLoop();
		if (triggerState && triggerUpdateFlag) {
			break;
		}
		sleep_us(1000000 / PID_RATE);
	}
	clearEeprom(nullptr);
}

void eepromInit() {
	EEPROM.begin(EEPROM_SIZE);

	u64 magic = EEPROM_INIT_MAGIC_NUMBER;
	u64 readMagic;
	EEPROM.get(EEPROM_POS_MAGIC_NUMBER, readMagic);
	if (readMagic != magic) {
		firstBoot = true;
		EEPROM.put(EEPROM_POS_VERSION_MAJOR, (u8)FIRMWARE_VERSION_MAJOR);
		EEPROM.put(EEPROM_POS_VERSION_MINOR, (u8)FIRMWARE_VERSION_MINOR);
		EEPROM.put(EEPROM_POS_VERSION_PATCH, (u8)FIRMWARE_VERSION_PATCH);
		eepromMajorVersion = FIRMWARE_VERSION_MAJOR;
		eepromMinorVersion = FIRMWARE_VERSION_MINOR;
		eepromPatchVersion = FIRMWARE_VERSION_PATCH;
	} else {
		MenuItem::settingsAreInEeprom = true;
		EEPROM.get(EEPROM_POS_VERSION_MAJOR, eepromMajorVersion);
		EEPROM.get(EEPROM_POS_VERSION_MINOR, eepromMinorVersion);
		EEPROM.get(EEPROM_POS_VERSION_PATCH, eepromPatchVersion);
	}
	// initialize all EEPROM dependent settings to default values or read from EEPROM
	const u32 thisVersion = FIRMWARE_VERSION_MAJOR << 16 | FIRMWARE_VERSION_MINOR << 8 | FIRMWARE_VERSION_PATCH;
	const u32 eepromVersion = eepromMajorVersion << 16 | eepromMinorVersion << 8 | eepromPatchVersion;
	if (thisVersion > eepromVersion) {
		// firmware update: migrate EEPROM
		eepromMigrate(eepromMajorVersion, eepromMinorVersion, eepromPatchVersion);
	} else if (thisVersion < eepromVersion) {
		// firmware downgrade: clear EEPROM
		requestClearEeprom(eepromMajorVersion, eepromMinorVersion, eepromPatchVersion);
	}
	initMenu();
	joystickInit(true);
	initEscLayout();
	if (firstBoot) {
		adjustProfileDefaults();
		selectedProfile = 0;
		loadSettings();
		EEPROM.commit();
		EEPROM.put(EEPROM_POS_MAGIC_NUMBER, magic);
		firstBootMenu->onEnter();
		openedMenu = firstBootMenu;
		operationState = STATE_MENU;
	}
}

bool clearEeprom(MenuItem *_item) {
	if (tournamentMode) {
		return false;
	}
	for (u16 i = 0; i < EEPROM_SIZE; i++) {
		EEPROM.write(i, 255);
	}
	EEPROM.commit();
	rebootReason = BootReason::CLEAR_EEPROM;
	rp2040.reboot();
	return true; // never reached
}
