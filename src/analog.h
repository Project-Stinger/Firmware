#include "Arduino.h"
#include "utils/typedefs.h"

#define CONV_RESULT_VBAT 0
#define CONV_RESULT_JOYSTICK_X 1
#define CONV_RESULT_JOYSTICK_Y 2
#if HW_VERSION == 1
extern volatile u32 adcConversions[3];
#elif HW_VERSION == 2
#define CONV_RESULT_ISOLENOID 3
#define CONV_RESULT_IBAT 4
extern volatile u32 adcConversions[5];
#endif

void analogLoop();
void initAnalog();
