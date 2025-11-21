#include "global.h"

elapsedMillis bootTimer;

char deviceName[16] = "Stinger";
char ownerName[32] = "John Doe";
char ownerContact[32] = "john.doe@example.com";

BootReason bootReason;
BootReason __uninitialized_ram(rebootReason);
u64 __uninitialized_ram(powerOnResetMagicNumber);