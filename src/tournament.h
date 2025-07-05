#include "Arduino.h"
#include "menu/menuItem.h"
#include "utils/typedefs.h"

extern bool tournamentMode;
extern u32 tournamentMaxRpm;
extern u8 tournamentMaxDps;
extern bool allowFullAuto;
extern bool allowSemiAuto;
extern bool tournamentInvertScreen;
extern bool tournamentBlockMenu;

void tournamentInit();
void tournamentLoop();
bool enableTournamentMode(MenuItem *_item = nullptr);
void disableTournamentMode();
void applyTournamentLimits();
