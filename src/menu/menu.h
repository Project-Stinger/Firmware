#include "Arduino.h"
#include "menuItem.h"
#include "utils/typedefs.h"

#define DEFAULT_MAX_RPM (HW_VERSION == 2 ? 53000 : 60000)

extern MenuItem *mainMenu;
extern MenuItem *firstBootMenu;
extern MenuItem *onboardingMenu;
extern MenuItem *openedMenu;
extern u8 profileColor[3];
extern char profileName[16];
extern u16 profileColor565;

extern u8 rotationTickSensitivity;
extern const char rotationSensitivityStrings[3][10];

void initMenu();
bool saveAndClose(MenuItem *item);
void loadSettings();