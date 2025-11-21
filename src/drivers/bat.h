#include "Arduino.h"
#include "menu/menuItem.h"
#include "utils/filters.h"
#include "utils/typedefs.h"

extern PT1 batVoltage;
extern char cellSettings[5][9];
extern u8 batCellCount;
extern u8 batCellsSettings;
extern u8 batWarnVoltage;
extern u8 batShutdownVoltage;
extern u8 batState;
extern bool batWarning;
extern i8 batCalibrationOffset;
extern u8 powerSource;

#if HW_VERSION == 2
constexpr fix32 ESC_CURR_CONV = 3.3 / 4096 * 150;
extern fix32 batCurrent;
extern fix32 escCurrentAdc;
extern fix64 mahUsed;
extern volatile bool forceInitBat;

void batCurrLoop();
#endif

enum BatState {
	BAT_SETUP,
	BAT_RUNNING
};

enum StorageModeState {
	STORAGE_START,
	STORAGE_SPIN,
	STORAGE_FINISH,
};

void onBatSettingChange(MenuItem *_item);

void initBat();

void batLoop();

bool storageModeStart(MenuItem *_item);
bool onStorageUp(MenuItem *_item);
bool onStorageDown(MenuItem *_item);
bool onStorageRight(MenuItem *item);
bool onStorageLeft(MenuItem *_item);
bool storageModeLoop(MenuItem *item);