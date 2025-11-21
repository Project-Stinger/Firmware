#include "global.h"

#define USB_VOLT_THRES 2.0f

bool tournamentMode = false;
u32 tournamentMaxRpm = 50000;
u8 tournamentMaxDps = 40;
bool allowFullAuto = true;
bool allowSemiAuto = true;
bool tournamentBlockMenu = false;
bool tournamentInvertScreen = false;

void applyTournamentLimits() {
	if (tournamentMode) {
#if HW_VERSION == 2
		if (dpsLimit > tournamentMaxDps) dpsLimit = tournamentMaxDps;
#endif
		if (dartsPerSecond > tournamentMaxDps) dartsPerSecond = tournamentMaxDps;
		calcPushDurations(nullptr);
		if (targetRpm > tournamentMaxRpm) targetRpm = tournamentMaxRpm;
		calcTargetRpms(nullptr);
		if (fireMode == FIRE_BURST && !allowSemiAuto) fireMode = FIRE_SINGLE;
		if (fireMode == FIRE_CONTINUOUS && !allowFullAuto) fireMode = FIRE_SINGLE;
	}
}

void tournamentInit() {
	if (firstBoot) {
		EEPROM.put(EEPROM_POS_TOURNAMENT_ENABLED, false);
		tournamentMode = false;
	} else {
		bool b = false;
		EEPROM.get(EEPROM_POS_TOURNAMENT_ENABLED, b);
		if (b) {
			enableTournamentMode();
		}
	}
}

void tournamentLoop() {
	if (tournamentMode && powerSource == 2) {
		disableTournamentMode();
	}
	static u8 vis = 0; // 0 = warning visible, 1 = tournament enter visible
	if (vis == 0 && powerSource == 1) {
		mainMenu->search("enterTournamentMode")->setVisible(true);
		mainMenu->search("tournamentInfo2")->setVisible(false);
		vis = 1;
	} else if (vis == 1 && powerSource != 1) {
		mainMenu->search("enterTournamentMode")->setVisible(false);
		mainMenu->search("tournamentInfo2")->setVisible(true);
		vis = 0;
	}
}

bool enableTournamentMode(MenuItem *_item) {
	if (tournamentMode || powerSource == 2) {
		return false;
	}
	tournamentMode = true;
	EEPROM.put(EEPROM_POS_TOURNAMENT_ENABLED, tournamentMode);
	if (tournamentInvertScreen)
		tft.invertDisplay(HW_VERSION == 2 ? false : true);
	mainMenu->search("tournament")->setVisible(false);
	saveAndClose(mainMenu->search("save"));
#if HW_VERSION == 2
	makeRtttlSound("bossmusic:d=4,o=6,b=300:c6,g6,g#6,f6,g6,d#6,f6,d6,1c6");
#endif
	return false;
}

void disableTournamentMode() {
	if (!tournamentMode || operationState != STATE_OFF) {
		return;
	}
	tournamentMode = false;
	EEPROM.put(EEPROM_POS_TOURNAMENT_ENABLED, tournamentMode);
	EEPROM.commit();
	tft.invertDisplay(HW_VERSION == 2 ? true : false);
	mainMenu->search("tournament")->setVisible(true);
	loadSettings();
	triggerFullRedraw();
#if HW_VERSION == 2
	makeRtttlSound("bossmusicoff:d=4,o=6,b=300:d6,a6,b6,g6,a6,f#6,g6,e6,1d6");
#endif
}