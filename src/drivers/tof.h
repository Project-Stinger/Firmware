#pragma once

extern i32 dartCount;

#ifdef USE_TOF
#include "Arduino.h"
#include "menu/menuItem.h"
#include "utils/typedefs.h"

extern bool magPresent;
extern u8 magSize;
extern bool fireWoMag;
extern bool fireWoDarts;
extern PT1 tofDistance;
#if HW_VERSION == 2
extern bool beepOnMagChange;
#endif
extern bool foundTof;
void initTof();
void tofLoop();
void disableTof();
bool startTofCalibration(MenuItem *item);
bool onTofCalibrationRight(MenuItem *item);
bool onTofCalibrationLeft(MenuItem *item);
void drawTofCalibration(MenuItem *item);
bool onTofCalibrationExit(MenuItem *item);
#endif // USE_TOF