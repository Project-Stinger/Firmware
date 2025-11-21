
#if HW_VERSION == 2
void initStandbySwitch();
void standbyOnLoop();
void standbyOffLoop();
void enableStandby();
void disableStandby();

extern bool standbyOn;
extern bool fastStandbyEnabled;
extern bool enableStandbyFlag;
#endif