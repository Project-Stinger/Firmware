#include "Arduino.h"
#include "utils/typedefs.h"

extern u8 triggerState;
extern volatile bool triggerUpdateFlag;

void triggerInit();
void triggerLoop();