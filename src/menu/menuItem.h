#pragma once
#include "Arduino.h"
#include "elapsedMillis.h"
#include "utils/typedefs.h"
#include <vector>
using std::vector;

#define MENU_MAX_DEFAULT_DATA_SIZE 64

#if HW_VERSION == 1
#define SETTINGS_BEEP_PERIOD 250
#define STRING_EDIT_VIEW_LENGTH 7
#define STRING_EDIT_CHAR_WIDTH 6
#elif HW_VERSION == 2
#define SETTINGS_BEEP_MIN_FREQ 500
#define SETTINGS_BEEP_MAX_FREQ 1000
#define SETTINGS_BEEP_FREQ_RANGE (SETTINGS_BEEP_MAX_FREQ - SETTINGS_BEEP_MIN_FREQ)
#define STRING_EDIT_VIEW_LENGTH 12
#define STRING_EDIT_CHAR_WIDTH 8
#endif

enum class MenuItemType {
	INFO,
	ACTION,
	SUBMENU,
	VARIABLE,
	CUSTOM
};

enum class VariableType {
	NONE,
	I32,
	U32,
	I16,
	U16,
	I8,
	U8,
	U8_LUT, // look-up table for index to string conversion
	BOOL,
	FLOAT,
	STRING,
};

enum EntryDrawType {
	DRAW_FOCUSED,
	DRAW_FOCUSED_DESC,
	DRAW_UNFOCUSED,
	DRAW_ENTERED
};

class MenuItem {
public:
	/**
	 * @brief Construct a new int Menu Item
	 *
	 * @param varType VariableType:: e.g. U32, I16
	 * @param data pointer to the variable
	 * @param defaultVal default value if it cannot be found in the EEPROM
	 * @param stepSize step size for the value (based on the value stored in the EEPROM)
	 * @param min minimum value (based on the value stored in the EEPROM)
	 * @param max maximum value (based on the value stored in the EEPROM)
	 * @param displayDivider divider to display the value (e.g. 1000 for 0.001)
	 * @param displayDecimals number of decimals to display
	 * @param eepromPos position in the EEPROM (for first profile, if applicable)
	 * @param isProfileDependent true if the value is different for each profile, offset for each profile: 512
	 * @param identifier string to identify the variable, can be used to search menu items
	 * @param displayName string to display in the menu
	 * @param description appears if the item is focused long enough
	 * @param offset offset to add to the raw value (e.g. set to 100 if a value of 0 in the EEPROM should be displayed as 100 in the menu)
	 * @param rebootOnChange true if a reboot is required after changing the value
	 * @param rollover true if the value should roll over at the min/max values
	 */
	MenuItem(const VariableType varType, void *data, const i32 defaultVal, const i32 stepSize, const i32 min, const i32 max, const i32 displayDivider, const u8 displayDecimals, const u32 eepromPos, const bool isProfileDependent, const char *identifier, const char *displayName, const char *description = nullptr, const i32 offset = 0, const bool rebootOnChange = false, bool rollover = true);
	/**
	 * @brief Construct a new checkbox Menu Item
	 *
	 * @param data pointer to the variable
	 * @param defaultVal default value if it cannot be found in the EEPROM
	 * @param eepromPos position in the EEPROM (for first profile, if applicable)
	 * @param isProfileDependent true if the value is different for each profile, offset for each profile: 512
	 * @param identifier string to identify the variable, can be used to search menu items
	 * @param displayName string to display in the menu
	 * @param description appears if the item is focused long enough
	 * @param rebootOnChange true if a reboot is required after changing the value
	 */
	MenuItem(bool *data, const bool defaultVal, const u32 eepromPos, const bool isProfileDependent, const char *identifier, const char *displayName, const char *description = nullptr, const bool rebootOnChange = false);
	/**
	 * @brief Construct a new LUT Menu Item
	 *
	 * @param data pointer to the variable
	 * @param defaultVal default value if it cannot be found in the EEPROM
	 * @param eepromPos position in the EEPROM (for first profile, if applicable)
	 * @param max maximum value that can be retreived from the LUT
	 * @param lut look-up table for index to string conversion
	 * @param strSize size of the strings in the LUT
	 * @param isProfileDependent true if the value is different for each profile, offset for each profile: 512
	 * @param identifier string to identify the variable, can be used to search menu items
	 * @param displayName string to display in the menu
	 * @param description appears if the item is focused long enough
	 * @param rebootOnChange true if a reboot is required after changing the value
	 */
	MenuItem(u8 *data, const u8 defaultVal, const u32 eepromPos, const u8 max, const char *lut, const u8 strSize, const bool isProfileDependent, const char *identifier, const char *displayName, const char *description = nullptr, const bool rebootOnChange = false);
	/**
	 * @brief Construct a new string Menu Item
	 *
	 * @param data pointer to the string
	 * @param maxStringLength maximum length of the string, including the null terminator
	 * @param defaultVal default value if it cannot be found in the EEPROM
	 * @param eepromPos position in the EEPROM (for first profile, if applicable)
	 * @param isProfileDependent true if the value is different for each profile, offset for each profile: 512
	 * @param identifier string to identify the variable, can be used to search menu items
	 * @param displayName string to display in the menu
	 * @param description appears if the item is focused long enough
	 * @param rebootOnChange true if a reboot is required after changing the value
	 */
	MenuItem(char *data, const u8 maxStringLength, const char *defaultVal, const u32 eepromPos, const bool isProfileDependent, const char *identifier, const char *displayName, const char *description = nullptr, const bool rebootOnChange = false);
	/**
	 * @brief Construct a new float Menu Item
	 *
	 * @param varType e.g. VariableType::FLOAT
	 * @param data pointer to the variable
	 * @param defaultVal default value if it cannot be found in the EEPROM
	 * @param stepSize step size for the value
	 * @param min minimum value
	 * @param max maximum value
	 * @param displayDecimals number of decimals to display
	 * @param eepromPos position in the EEPROM (for first profile, if applicable)
	 * @param isProfileDependent true if the value is different for each profile, offset for each profile: 512
	 * @param identifier string to identify the variable, can be used to search menu items
	 * @param displayName string to display in the menu
	 * @param description appears if the item is focused long enough
	 * @param rebootOnChange true if a reboot is required after changing the value
	 * @param rollover true if the value should roll over at the min/max values
	 */
	MenuItem(const VariableType varType, float *data, const float defaultVal, const float stepSize, const float min, const float max, const u8 displayDecimals, const u32 eepromPos, const bool isProfileDependent, const char *identifier, const char *displayName, const char *description = nullptr, const bool rebootOnChange = false, bool rollover = true);
	/**
	 * @brief Construct a new Submenu, Action or Info Menu Item
	 *
	 * @param itemType MenuItemType:: e.g. info or variable
	 * @param identifier string to identify the variable, can be used to search menu items
	 * @param displayName string to display in the menu
	 * @param description appears if the item is focused long enough
	 */
	MenuItem(const MenuItemType itemType, const char *identifier, const char *displayName, const char *description = nullptr, const bool rebootOnChange = false);
	/**
	 * @brief Add a child to this menu item (only for submenus)
	 *
	 * @param child MenuItem to add
	 * @return MenuItem* this
	 */
	MenuItem *addChild(MenuItem *child);
	/**
	 * @brief Set the parent of this menu item
	 *
	 * @param parent MenuItem to set as parent
	 * @return MenuItem* this
	 */
	MenuItem *setParent(MenuItem *parent);
	/**
	 * @brief Search for a menu item by its identifier
	 *
	 * @param identifier string to search for
	 * @return MenuItem* the item, or nullptr if not found
	 */
	MenuItem *search(const char *identifier);
	MenuItem *parent = nullptr;
	const MenuItemType itemType;
	bool entered = false;
	bool focused = false;
	bool fullRedraw = true;
	void setVisible(bool visible);
	void init();
	void loop();
	void onFocus();
	void onEnter();
	void setFocusedChild(const char *identifier);
	void onExit();
	void drawEntry(bool fullRedraw);
	void save();
	void triggerFullRedraw();
	void triggerRedrawValue();
	bool isRebootRequired();
	/**
	 * @brief set a custom function to call when the item is entered
	 *
	 * @param onEnterFunction function to call, has to return true if the normal enter function (e.g. setting the entered state) should be executed, nullptr to unset
	 * @return MenuItem* this
	 */
	MenuItem *setOnEnterFunction(bool (*onEnterFunction)(MenuItem *i));
	/**
	 * @brief set a custom function to call when the item is exited
	 *
	 * @param onExitFunction function to call, has to return true if the normal exit function should be executed, nullptr to unset
	 * @return MenuItem* this
	 */
	MenuItem *setOnExitFunction(bool (*onExitFunction)(MenuItem *i));
	/**
	 * @brief set a custom loop function
	 *
	 * @param customLoop function to call, has to return true if the normal loop function (e.g. checking for user input) should be executed, nullptr to unset
	 * @return MenuItem* this
	 */
	MenuItem *setCustomLoop(bool (*customLoop)(MenuItem *i));
	/**
	 * @brief set a custom function to call on the joystick up event
	 *
	 * @param onUpFunction function to call, has to return true if the normal up function (e.g. value change) should be executed, nullptr to unset
	 * @return MenuItem* this
	 */
	MenuItem *setOnUpFunction(bool (*onUpFunction)(MenuItem *i));
	/**
	 * @brief set a custom function to call on the joystick down event
	 *
	 * @param onDownFunction function to call, has to return true if the normal down function (e.g. value change) should be executed, nullptr to unset
	 * @return MenuItem* this
	 */
	MenuItem *setOnDownFunction(bool (*onDownFunction)(MenuItem *i));
	/**
	 * @brief set a custom function to call on the joystick left event
	 *
	 * @param onLeftFunction function to call, has to return true if the normal left function (e.g. exiting this menu/entered state) should be executed, nullptr to unset
	 * @return MenuItem* this
	 */
	MenuItem *setOnLeftFunction(bool (*onLeftFunction)(MenuItem *i));
	/**
	 * @brief set a custom function to call on the joystick right event
	 *
	 * @param onRightFunction function to call, has to return true if the normal right function (e.g. entering this menu/entered state) should be executed, nullptr to unset
	 * @return MenuItem* this
	 */
	MenuItem *setOnRightFunction(bool (*onRightFunction)(MenuItem *i));
	/**
	 * @brief set a custom function to draw the full menu item, e.g. for custom graphics
	 *
	 * @param customDrawFull function to call, nullptr to unset
	 * @return MenuItem* this
	 */
	MenuItem *setCustomDrawFull(void (*customDrawFull)(MenuItem *i));
	/**
	 * @brief set a function to call when the value of the item changes, e.g. for computed values that depend this value
	 *
	 * @param onChangeFunction function to call, nullptr to unset
	 * @return MenuItem* this
	 */
	MenuItem *setOnChangeFunction(void (*onChangeFunction)(MenuItem *i));

	void setRange(i32 min, i32 max);
	void setRange(float min, float max);
	void setMin(i32 min);
	void setMax(i32 max);
	void setMin(float min);
	void setMax(float max);
	void checkFocus();

	static bool settingsBeep;
	static bool settingsAreInEeprom;
	static bool enteredRotationNavigation;
#if HW_VERSION == 1
	static void scheduleBeep(i16 msSinceLast, u8 tone);
	static void makeSettingsBeep(u8 tone);
#elif HW_VERSION == 2
	static void beep(u16 freq);
	u16 getValueBeepFreq();
	static bool settingsSolenoidClickFlag;
#endif

private:
	static u8 lastSettingsBeepMotor;
	static u32 scheduledSettingsBeep;
	static elapsedMillis lastSettingsBeepTimer;
	static u8 scheduledBeepTone;
	static i16 lastTickCount;
	vector<MenuItem *> children;
	bool focusable = true;
	bool visible = true;
	i8 charPos = 0;
	i8 charDisplayStart = 0;
	const char *lut = nullptr;
	const u8 lutStringSize = 0;
	void onUp();
	void onDown();
	void onLeft();
	void onRight();
	void drawFull();
	bool (*onEnterFunction)(MenuItem *i) = nullptr;
	bool (*onExitFunction)(MenuItem *i) = nullptr;
	bool (*customLoop)(MenuItem *i) = nullptr;
	bool (*onUpFunction)(MenuItem *i) = nullptr;
	bool (*onDownFunction)(MenuItem *i) = nullptr;
	bool (*onLeftFunction)(MenuItem *i) = nullptr;
	bool (*onRightFunction)(MenuItem *i) = nullptr;
	void (*customDrawFull)(MenuItem *i) = nullptr;
	void (*onChangeFunction)(MenuItem *i) = nullptr;
	const VariableType varType = VariableType::NONE;
	const i32 stepSizeI = 0;
	i32 minI = 0;
	i32 maxI = 0;
	const i32 offsetI = 0;
	void *data;
	u8 defaultVal[MENU_MAX_DEFAULT_DATA_SIZE];
	const i32 displayDivider = 0;
	const u8 displayDecimals = 0;
	const float stepSizeF = 0;
	float minF = 0;
	float maxF = 0;
	const bool isProfileDependent;
	const char *identifier;
	const char *displayName;
	const char *description;
	const u32 startEepromPos;
	const u8 maxStringLength = 0;
	const bool rebootOnChange;
	bool rebootRequired = false;
	const bool rollover = true;
	elapsedMillis focusTimer;
	u16 scrollTop = 0;
	u8 lastEntryDrawType = DRAW_UNFOCUSED;
	void getNumberValueString(char buf[8]);
	bool redrawValue = true;
	u16 lastProfileColor565 = 0;
	u8 entryHeight = 0;

	void drawValueNotEntered(const i16 cY);
	void drawValueEntered(const i16 cY);
	void drawNumberValue(const i16 cY, u16 colorBg, u16 colorFg, u8 drawBg);
	void drawBoolValue(const i16 cY, u16 colorBg, u16 colorFg, u8 drawBg);
	void drawStringValue(const i16 cY, u16 colorBg, u16 colorFg, u8 drawBg);
	void drawLutValue(const i16 cY, u16 colorBg, u16 colorFg, u8 drawBg);
	void drawEditableString(const i16 cY);
};