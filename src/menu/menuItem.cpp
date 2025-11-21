#include "../global.h"

bool MenuItem::settingsBeep = false;
bool MenuItem::settingsAreInEeprom = false;
u8 MenuItem::lastSettingsBeepMotor = 0;
u32 MenuItem::scheduledSettingsBeep = 0;
elapsedMillis MenuItem::lastSettingsBeepTimer = 0;
u8 MenuItem::scheduledBeepTone = 0;
bool MenuItem::enteredRotationNavigation = false;
i16 MenuItem::lastTickCount = 0;
#if HW_VERSION == 2
bool MenuItem::settingsSolenoidClickFlag = false;
#endif

MenuItem::MenuItem(const VariableType varType, void *data, const i32 defaultVal, const i32 stepSize, const i32 min, const i32 max, const i32 displayDivider, const u8 displayDecimals, const u32 eepromPos, const bool isProfileDependent, const char *identifier, const char *displayName, const char *description, const i32 offset, const bool rebootOnChange, bool rollover)
	: varType(varType),
	  stepSizeI(stepSize),
	  minI(min),
	  maxI(max),
	  displayDivider(displayDivider),
	  displayDecimals(displayDecimals),
	  startEepromPos(eepromPos),
	  isProfileDependent(isProfileDependent),
	  identifier(identifier),
	  displayName(displayName),
	  description(description),
	  rebootOnChange(rebootOnChange),
	  itemType(MenuItemType::VARIABLE),
	  offsetI(offset),
	  data(data),
	  rollover(rollover) {
	switch (varType) {
	case VariableType::I32:
	case VariableType::U32:
	case VariableType::FLOAT:
		memcpy(this->defaultVal, &defaultVal, 4);
		break;
	case VariableType::U16:
	case VariableType::I16:
		memcpy(this->defaultVal, &defaultVal, 2);
		break;
	case VariableType::U8:
	case VariableType::I8:
	case VariableType::BOOL:
		memcpy(this->defaultVal, &defaultVal, 1);
		break;
	default:
		break;
	}
}

MenuItem::MenuItem(const VariableType varType, float *data, const float defaultVal, const float stepSize, const float min, const float max, const u8 displayDecimals, const u32 eepromPos, const bool isProfileDependent, const char *identifier, const char *displayName, const char *description, const bool rebootOnChange, bool rollover)
	: varType(varType),
	  stepSizeF(stepSize),
	  minF(min),
	  maxF(max),
	  displayDecimals(displayDecimals),
	  startEepromPos(eepromPos),
	  isProfileDependent(isProfileDependent),
	  identifier(identifier),
	  displayName(displayName),
	  description(description),
	  rebootOnChange(rebootOnChange),
	  itemType(MenuItemType::VARIABLE),
	  data(data),
	  rollover(rollover) {
	memcpy(this->defaultVal, &defaultVal, 4);
}

MenuItem::MenuItem(const MenuItemType itemType, const char *identifier, const char *displayName, const char *description, const bool rebootOnChange)
	: itemType(itemType),
	  identifier(identifier),
	  displayName(displayName),
	  description(description),
	  startEepromPos(0),
	  isProfileDependent(false),
	  rebootOnChange(rebootOnChange) {
	if (itemType == MenuItemType::INFO) {
		this->focusable = false;
	} else {
		this->focusable = true;
	}
}

MenuItem::MenuItem(char *data, const u8 maxStringLength, const char *defaultVal, const u32 eepromPos, const bool isProfileDependent, const char *identifier, const char *displayName, const char *description, const bool rebootOnChange)
	: varType(VariableType::STRING),
	  startEepromPos(eepromPos),
	  isProfileDependent(isProfileDependent),
	  identifier(identifier),
	  displayName(displayName),
	  description(description),
	  rebootOnChange(rebootOnChange),
	  itemType(MenuItemType::VARIABLE),
	  data(data),
	  maxStringLength(MIN(maxStringLength, MENU_MAX_DEFAULT_DATA_SIZE)) {
	memcpy(this->defaultVal, defaultVal, this->maxStringLength);
	((char *)this->defaultVal)[this->maxStringLength - 1] = '\0';
	u8 len = strlen((char *)this->defaultVal);
	for (u8 i = len; i < this->maxStringLength; i++) {
		this->defaultVal[i] = '\0';
	}
}

MenuItem::MenuItem(bool *data, const bool defaultVal, const u32 eepromPos, const bool isProfileDependent, const char *identifier, const char *displayName, const char *description, const bool rebootOnChange)
	: varType(VariableType::BOOL),
	  startEepromPos(eepromPos),
	  isProfileDependent(isProfileDependent),
	  identifier(identifier),
	  displayName(displayName),
	  description(description),
	  rebootOnChange(rebootOnChange),
	  itemType(MenuItemType::VARIABLE),
	  data(data) {
	((bool *)this->defaultVal)[0] = defaultVal;
}

MenuItem::MenuItem(u8 *data, const u8 defaultVal, const u32 eepromPos, const u8 max, const char *lut, const u8 strSize, const bool isProfileDependent, const char *identifier, const char *displayName, const char *description, const bool rebootOnChange)
	: varType(VariableType::U8_LUT),
	  stepSizeI(1),
	  maxI(max),
	  startEepromPos(eepromPos),
	  isProfileDependent(isProfileDependent),
	  identifier(identifier),
	  displayName(displayName),
	  description(description),
	  rebootOnChange(rebootOnChange),
	  itemType(MenuItemType::VARIABLE),
	  data(data),
	  lut(lut),
	  lutStringSize(strSize) {
	this->defaultVal[0] = defaultVal;
}

MenuItem *MenuItem::addChild(MenuItem *child) {
	if (child == nullptr) return this;
	if (child->parent == nullptr)
		child->setParent(this);
	this->children.push_back(child);
	return this;
}

MenuItem *MenuItem::setParent(MenuItem *parent) {
	this->parent = parent;
	return this;
}

MenuItem *MenuItem::search(const char *id) {
	if (strcmp(this->identifier, id) == 0) return this;
	for (u8 i = 0; i < this->children.size(); i++) {
		MenuItem *result = this->children[i]->search(id);
		if (result != nullptr) return result;
	}
	return nullptr;
}

void MenuItem::setVisible(bool visible) {
	if (this->parent != nullptr && this->visible != visible) {
		this->parent->triggerFullRedraw();
		this->visible = visible;
		this->parent->checkFocus();
	}
}

void MenuItem::init() {
	switch (this->itemType) {
	case MenuItemType::VARIABLE:
		if (MenuItem::settingsAreInEeprom && this->startEepromPos < EEPROM_SIZE) {
			switch (this->varType) {
			case VariableType::I32:
			case VariableType::U32:
			case VariableType::FLOAT: {
				u32 addr = this->startEepromPos;
				if (this->isProfileDependent) {
					addr += selectedProfile * PROFILE_EEPROM_SIZE;
				}
				i32 val;
				EEPROM.get(addr, val);
				memcpy(this->data, &val, 4);
			} break;
			case VariableType::U16:
			case VariableType::I16: {
				u32 addr = this->startEepromPos;
				if (this->isProfileDependent) {
					addr += selectedProfile * PROFILE_EEPROM_SIZE;
				}
				i16 val;
				EEPROM.get(addr, val);
				memcpy(this->data, &val, 2);
			} break;
			case VariableType::U8:
			case VariableType::U8_LUT:
			case VariableType::I8:
			case VariableType::BOOL: {
				u32 addr = this->startEepromPos;
				if (this->isProfileDependent) {
					addr += selectedProfile * PROFILE_EEPROM_SIZE;
				}
				i8 val;
				EEPROM.get(addr, val);
				memcpy(this->data, &val, 1);
			} break;
			case VariableType::STRING: {
				u32 addr = this->startEepromPos;
				if (this->isProfileDependent) {
					addr += selectedProfile * PROFILE_EEPROM_SIZE;
				}
				for (u8 i = 0; i < this->maxStringLength; i++) {
					EEPROM.get(addr + i, ((char *)this->data)[i]);
				}
				((char *)this->data)[this->maxStringLength - 1] = '\0';
				u8 len = strlen((char *)this->data);
				for (u8 i = len; i < this->maxStringLength; i++) {
					((char *)this->data)[i] = '\0';
				}
			} break;
			}
		} else {
			switch (this->varType) {
			case VariableType::I32:
			case VariableType::U32:
			case VariableType::FLOAT: {
				memcpy(this->data, this->defaultVal, 4);
				u32 addr = this->startEepromPos;
				if (this->startEepromPos >= EEPROM_SIZE) break;
				if (this->isProfileDependent) {
					for (u8 i = 0; i < MAX_PROFILE_COUNT; i++) {
						EEPROM.put(addr + i * PROFILE_EEPROM_SIZE, *(i32 *)this->data);
					}
				} else {
					EEPROM.put(addr, *(i32 *)this->data);
				}
			} break;
			case VariableType::U16:
			case VariableType::I16: {
				memcpy(this->data, this->defaultVal, 2);
				u32 addr = this->startEepromPos;
				if (this->startEepromPos >= EEPROM_SIZE) break;
				if (this->isProfileDependent) {
					for (u8 i = 0; i < MAX_PROFILE_COUNT; i++) {
						EEPROM.put(addr + i * PROFILE_EEPROM_SIZE, *(i16 *)this->data);
					}
				} else {
					EEPROM.put(addr, *(i16 *)this->data);
				}
			} break;
			case VariableType::U8:
			case VariableType::U8_LUT:
			case VariableType::I8:
			case VariableType::BOOL: {
				memcpy(this->data, this->defaultVal, 1);
				u32 addr = this->startEepromPos;
				if (this->startEepromPos >= EEPROM_SIZE) break;
				if (this->isProfileDependent) {
					for (u8 i = 0; i < MAX_PROFILE_COUNT; i++) {
						EEPROM.put(addr + i * PROFILE_EEPROM_SIZE, *(i8 *)this->data);
					}
				} else {
					EEPROM.put(addr, *(i8 *)this->data);
				}
			} break;
			case VariableType::STRING: {
				u32 defaultStringLength = min(this->maxStringLength, strlen((char *)this->defaultVal) + 1); // +1 for null terminator
				memcpy(this->data, this->defaultVal, defaultStringLength);
				((char *)this->data)[this->maxStringLength - 1] = '\0';
				for (u8 i = defaultStringLength; i < this->maxStringLength; i++) {
					((char *)this->data)[i] = '\0';
				}
				u32 addr = this->startEepromPos;
				if (this->startEepromPos >= EEPROM_SIZE) break;
				if (this->isProfileDependent) {
					for (u8 i = 0; i < MAX_PROFILE_COUNT; i++) {
						for (u8 j = 0; j < this->maxStringLength; j++) {
							EEPROM.put(addr + i * PROFILE_EEPROM_SIZE + j, ((char *)this->data)[j]);
						}
					}
				} else {
					for (u8 i = 0; i < this->maxStringLength; i++) {
						EEPROM.put(addr + i, ((char *)this->data)[i]);
					}
				}
			} break;
			}
		}
		if (this->onChangeFunction != nullptr) {
			this->onChangeFunction(this);
		}
		this->redrawValue = true;
		break;
	case MenuItemType::SUBMENU: {
		for (u8 i = 0; i < this->children.size(); i++) {
			this->children[i]->init();
		}
	}
	}
}

void MenuItem::getNumberValueString(char buf[8]) {
	if (this->itemType != MenuItemType::VARIABLE || this->varType == VariableType::STRING || this->varType == VariableType::BOOL || this->varType == VariableType::U8_LUT) {
		buf[0] = '\0';
		return;
	}
	float valF = 0;
	switch (this->varType) {
	case VariableType::I32:
		valF = *(i32 *)this->data + this->offsetI;
		break;
	case VariableType::U32:
		valF = *(u32 *)this->data + this->offsetI;
		break;
	case VariableType::FLOAT:
		valF = *(float *)this->data;
		break;
	case VariableType::U16:
		valF = *(u16 *)this->data + this->offsetI;
		break;
	case VariableType::I16:
		valF = *(i16 *)this->data + this->offsetI;
		break;
	case VariableType::U8:
		valF = *(u8 *)this->data + this->offsetI;
		break;
	case VariableType::I8:
		valF = *(i8 *)this->data + this->offsetI;
		break;
	};
	if (this->varType != VariableType::FLOAT)
		valF /= this->displayDivider;
	switch (this->varType) {
	case VariableType::I32:
	case VariableType::U32:
	case VariableType::U16:
	case VariableType::I16:
	case VariableType::U8:
	case VariableType::I8:
	case VariableType::FLOAT: {
		snprintf(buf, 8, "%.*f", this->displayDecimals, valF);
	} break;
	}
}

void MenuItem::loop() {
	if (this->customLoop != nullptr) {
		if (!this->customLoop(this)) return;
	}
	bool deepestMenuItem = true;
	bool deepestFullDraw = true;
#if HW_VERSION == 1
	if (scheduledSettingsBeep && millis() >= scheduledSettingsBeep) {
		if (scheduledSettingsBeep + 300 > millis())
			makeSettingsBeep(scheduledBeepTone);
		scheduledSettingsBeep = 0;
	}
#elif HW_VERSION == 2
#endif
	switch (this->itemType) {
	case MenuItemType::SUBMENU: {
		if (!this->entered) return;
		for (auto c : this->children) {
			if (c->entered) {
				c->loop(); // loop nests down to the deepest entered item
				deepestMenuItem = false;
				if (c->itemType == MenuItemType::SUBMENU || c->itemType == MenuItemType::CUSTOM) {
					deepestFullDraw = false;
				}
				break;
			}
		}
	} break;
	case MenuItemType::VARIABLE:
	case MenuItemType::ACTION:
	case MenuItemType::INFO:
		deepestFullDraw = false;
		break;
	}

	if (deepestMenuItem) {
		if (enteredRotationNavigation) {
			if (lastTickCount != joystickRotationTicks && joystickMagnitude >= 85) {
// joystickMagnitude needed to prevent updates while letting the joystick snap back to center
#if HW_VERSION == 2
				if (settingsBeep) settingsSolenoidClickFlag = true;
#endif
				if (lastTickCount > joystickRotationTicks) {
					for (i32 i = lastTickCount - joystickRotationTicks; i > 0; i--) {
						this->onUp();
					}
				} else {
					for (i32 i = joystickRotationTicks - lastTickCount; i > 0; i--) {
						this->onDown();
					}
				}
				lastTickCount = joystickRotationTicks;
			}
			if (gestureUpdated && lastGesture.type == GESTURE_RELEASE) {
				enteredRotationNavigation = false;
				this->onExit();
			}
		} else if (gestureUpdated && (lastGesture.type == GESTURE_PRESS || lastGesture.type == GESTURE_HOLD)) {
			gestureUpdated = false;
			switch (lastGesture.direction) {
			case Direction::UP:
				this->onUp();
				break;
			case Direction::DOWN:
				this->onDown();
				break;
			case Direction::LEFT:
				this->onLeft();
				break;
			case Direction::RIGHT:
				this->onRight();
				break;
			case Direction::UP_LEFT:
				if (lastGesture.type != GESTURE_PRESS) break;
				if (this->itemType == MenuItemType::SUBMENU) {
					MenuItem *focusedChild = nullptr;
					for (u8 i = 0; i < this->children.size(); i++) {
						if (this->children[i]->focused) {
							focusedChild = this->children[i];
							break;
						}
					}
					if (focusedChild && focusedChild->itemType == MenuItemType::VARIABLE && focusedChild->varType != VariableType::BOOL && focusedChild->varType != VariableType::STRING && focusedChild->varType != VariableType::NONE) {
#if HW_VERSION == 1
						scheduleBeep(SETTINGS_BEEP_PERIOD, 1);
#elif HW_VERSION == 2
						beep(SETTINGS_BEEP_MIN_FREQ + SETTINGS_BEEP_FREQ_RANGE / 3);
#endif
						enteredRotationNavigation = true;
						lastTickCount = 0;
						focusedChild->onEnter();
					}
				} else if (this->itemType == MenuItemType::VARIABLE && this->varType != VariableType::BOOL && this->varType != VariableType::STRING && this->varType != VariableType::NONE) {
#if HW_VERSION == 1
					scheduleBeep(SETTINGS_BEEP_PERIOD, 1);
#elif HW_VERSION == 2
					beep(SETTINGS_BEEP_MIN_FREQ + SETTINGS_BEEP_FREQ_RANGE / 3);
#endif
					enteredRotationNavigation = true;
					lastTickCount = 0;
				}
				break;
			}
		}
	}
	if (deepestFullDraw && this->entered) { // in case onLeft etc. called for exit, entered is checked again
		SET_DEFAULT_FONT;
		tft.setTextColor(ST77XX_WHITE);
		this->drawFull();
	}
}

void MenuItem::onFocus() {
	DEBUG_PRINTF("Focusing %s\n", this->identifier);
	this->focusTimer = 0;
}

void MenuItem::onEnter() {
	DEBUG_PRINTF("Entering %s\n", this->identifier);
	if (this->onEnterFunction != nullptr) {
		if (!this->onEnterFunction(this)) return;
	}
	if (this->itemType == MenuItemType::VARIABLE && this->varType == VariableType::BOOL) {
		// confuse the compiler enough so it doesn't do bullshit
		// with *(bool *)this->data = !*(bool *)this->data; it was always changing between 255 and 254 if you had not initialized the bool, i.e. it only worked when the variable was 0 or 1, but not when it was anything else
		bool pre = *(bool *)this->data;
		u8 countOnes = 0;
		for (u8 i = 0; i < 8; i++) {
			if (pre & (1 << i)) countOnes++;
		}
		bool newVal = countOnes == 0;
		*(bool *)this->data = newVal; // don't enter into bools, just change them
		if (this->onChangeFunction != nullptr) {
			this->onChangeFunction(this);
		}
		this->redrawValue = true;
	} else {
		this->entered = true;
		this->charPos = 0;
		this->charDisplayStart = 0;
		this->fullRedraw = true;
		this->scrollTop = 0;
		if (this->itemType == MenuItemType::SUBMENU) {
			u8 size = this->children.size();
			bool focusedChild = false;
			for (u8 i = 0; i < size; i++) {
				MenuItem *c = this->children[i];
				if (c->focusable && c->visible && !focusedChild) {
					c->focused = true;
					focusedChild = true;
					c->onFocus();
				} else {
					c->focused = false;
				}
				if (c->entered) {
					c->entered = false;
				}
			}
		}
	}
}

void MenuItem::setFocusedChild(const char *identifier) {
	for (u8 i = 0; i < this->children.size(); i++) {
		if (strcmp(this->children[i]->identifier, identifier) == 0) {
			this->children[i]->focused = true;
			this->children[i]->onFocus();
		} else {
			this->children[i]->focused = false;
		}
	}
}

void MenuItem::triggerFullRedraw() {
	this->fullRedraw = true;
	for (auto child : this->children) {
		if (child->entered) {
			child->triggerFullRedraw();
		}
	}
}
void MenuItem::triggerRedrawValue() {
	this->redrawValue = true;
}

void MenuItem::drawNumberValue(const i16 cY, u16 colorBg, u16 colorFg, u8 drawBg) {
	if (drawBg) {
#if HW_VERSION == 1
		tft.fillRect(MENU_START_VALUE_X, cY, 42, YADVANCE, colorBg);
#elif HW_VERSION == 2
		tft.fillRect(MENU_START_VALUE_X, cY, 80, YADVANCE, colorBg);
#endif
	}
	tft.setCursor(MENU_START_VALUE_X, cY);
	char buf[8];
	this->getNumberValueString(buf);
	tft.setTextColor(colorFg);
	tft.print(buf);
	tft.setTextColor(ST77XX_WHITE);
}
void MenuItem::drawBoolValue(const i16 cY, u16 colorBg, u16 colorFg, u8 drawBg) {
	tft.setCursor(MENU_START_VALUE_X, cY);
	if (drawBg)
#if HW_VERSION == 1
		tft.fillRect(MENU_START_VALUE_X, cY, 18, YADVANCE, colorBg);
#elif HW_VERSION == 2
		tft.fillRect(MENU_START_VALUE_X, cY, 19, YADVANCE, colorBg);
#endif
	tft.setTextColor(colorFg);
	tft.print(*(bool *)this->data ? "ON" : "OFF");
	tft.setTextColor(ST77XX_WHITE);
}
void MenuItem::drawStringValue(const i16 cY, u16 colorBg, u16 colorFg, u8 drawBg) {
	if (drawBg) {
#if HW_VERSION == 1
		tft.fillRect(109, cY, 42, 8, colorBg);
#elif HW_VERSION == 2
		tft.fillRect(MENU_START_VALUE_X, cY, 100, YADVANCE, colorBg);
#endif
	}
	tft.setCursor(MENU_START_VALUE_X, cY);
#if HW_VERSION == 1
	char buf[8];
	memcpy(buf, this->data, MIN(this->maxStringLength, 8));
	buf[7] = '\0';
#elif HW_VERSION == 2
	char buf[32] = {0};
	u8 len = 0;
	i16 x = 0, y = 0;
	u16 width = 0, height = 0;
	tft.setTextWrap(false);
	while (width <= 100 && len < this->maxStringLength && len < 32) {
		buf[len] = ((char *)this->data)[len];
		len++;
		tft.getTextBounds(buf, 0, 0, &x, &y, &width, &height);
	}
	buf[len - 1] = '\0';
#endif
	tft.setTextColor(colorFg);
	tft.print(buf);
	tft.setTextColor(ST77XX_WHITE);
}
void MenuItem::drawLutValue(const i16 cY, u16 colorBg, u16 colorFg, u8 drawBg) {
	if (drawBg) {
#if HW_VERSION == 1
		u8 maxLen = 0;
		for (u8 i = 0; i <= this->maxI; i++) {
			maxLen = MAX(maxLen, strlen(&this->lut[i * this->lutStringSize]));
		}
		if (maxLen < 8) {
			tft.fillRect(MENU_START_VALUE_X, cY, 42, YADVANCE, colorBg);
		} else {
			tft.fillRect(MENU_START_VALUE_X + 6 * 8 - maxLen * 6, cY, maxLen * 6, YADVANCE, colorBg);
		}
#elif HW_VERSION == 2
		tft.fillRect(MENU_START_VALUE_X, cY, 90, YADVANCE, colorBg);
#endif
	}
#if HW_VERSION == 1
	u8 len = strlen(&this->lut[*(u8 *)this->data * this->lutStringSize]);
	if (len < 8) {
		tft.setCursor(MENU_START_VALUE_X, cY);
	} else {
		tft.setCursor(MENU_START_VALUE_X + 6 * 8 - len * 6, cY); // Fire Mode: Continuous
	}
#elif HW_VERSION == 2
	tft.setCursor(MENU_START_VALUE_X, cY);
#endif
	tft.setTextColor(colorFg);
	tft.print(&this->lut[*(u8 *)this->data * this->lutStringSize]);
	tft.setTextColor(ST77XX_WHITE);
}
void MenuItem::drawEditableString(const i16 cY) {
	u8 len = strlen((char *)this->data);
	if (this->charDisplayStart > this->charPos - 1) this->charDisplayStart = this->charPos - 1; // scroll left if needed
	if (this->charDisplayStart < 0) this->charDisplayStart = 0;
	u8 lengthPotential = this->maxStringLength - 2; // maximum selectable character (0-indexed)
	if (lengthPotential > len) lengthPotential = len;
	if (lengthPotential >= STRING_EDIT_VIEW_LENGTH) {
		if (this->charPos == lengthPotential) {
			this->charDisplayStart = lengthPotential - (STRING_EDIT_VIEW_LENGTH - 1);
		} else if (this->charPos > this->charDisplayStart + (STRING_EDIT_VIEW_LENGTH - 2)) {
			this->charDisplayStart = this->charPos - (STRING_EDIT_VIEW_LENGTH - 2);
		}
	}
	char buf[STRING_EDIT_VIEW_LENGTH + 1];
	memcpy(buf, ((char *)this->data) + this->charDisplayStart, MIN(this->maxStringLength, STRING_EDIT_VIEW_LENGTH));
	buf[STRING_EDIT_VIEW_LENGTH] = '\0';
	tft.setTextColor(tft.color565(150, 150, 150));
#if HW_VERSION == 1
	tft.fillRect(MENU_START_VALUE_X, cY, 42, 8, ST77XX_BLACK);
	tft.setCursor(MENU_START_VALUE_X, cY);
	tft.print(buf);
#elif HW_VERSION == 2
	tft.fillRect(MENU_START_VALUE_X, cY, 100, YADVANCE, ST77XX_BLACK);
	int cX = MENU_START_VALUE_X;
	// Bass11px is not monospace, so we have to print char by char
	for (u8 i = 0; buf[i] != '\0'; i++) {
		tft.setCursor(cX, cY);
		tft.print(buf[i]);
		cX += STRING_EDIT_CHAR_WIDTH;
	}
#endif
	u8 nullSearchBoundHigh = STRING_EDIT_VIEW_LENGTH - 1;
	u8 nullSearchBoundLow = 0;
	if (this->charDisplayStart != 0) {
		nullSearchBoundLow = 1;
		tft.fillRect(MENU_START_VALUE_X, cY, STRING_EDIT_CHAR_WIDTH, YADVANCE, ST77XX_BLACK);
		tft.setCursor(MENU_START_VALUE_X, cY);
		tft.print("<");
	}
	if (lengthPotential > this->charDisplayStart + 6) {
		nullSearchBoundHigh = STRING_EDIT_VIEW_LENGTH - 2;
		tft.fillRect(MENU_START_VALUE_X + STRING_EDIT_CHAR_WIDTH * (STRING_EDIT_VIEW_LENGTH - 1), cY, STRING_EDIT_CHAR_WIDTH, YADVANCE, ST77XX_BLACK);
		tft.setCursor(MENU_START_VALUE_X + STRING_EDIT_CHAR_WIDTH * (STRING_EDIT_VIEW_LENGTH - 1), cY);
		tft.print(">");
	}
	u8 posX = MENU_START_VALUE_X + (this->charPos - this->charDisplayStart) * STRING_EDIT_CHAR_WIDTH;
	tft.setTextColor(ST77XX_WHITE);
	char c = ((char *)this->data)[this->charPos];
	if (c) {
		tft.setCursor(posX, cY);
		tft.print(c);
	} else {
#if HW_VERSION == 1
		tft.fillRect(posX + 1, cY + 2, 4, 4, tft.color565(255, 0, 0));
#elif HW_VERSION == 2
		tft.fillRect(posX + 2, cY + 2, 6, 6, tft.color565(255, 0, 0));
#endif
	}
	for (u8 i = nullSearchBoundLow; i <= nullSearchBoundHigh; i++) {
		char c = ((char *)this->data)[this->charDisplayStart + i];
		if (!c) {
			if (this->charPos == this->charDisplayStart + i)
				break;
#if HW_VERSION == 1
			tft.fillRect(MENU_START_VALUE_X + i * STRING_EDIT_CHAR_WIDTH + 2, cY + 3, 2, 2, tft.color565(150, 0, 0));
#elif HW_VERSION == 2
			tft.fillRect(MENU_START_VALUE_X + i * STRING_EDIT_CHAR_WIDTH + 3, cY + 3, 4, 4, tft.color565(150, 0, 0));
#endif
			break;
		}
	}
}

void MenuItem::drawValueNotEntered(const i16 cY) {
	if (this->itemType != MenuItemType::VARIABLE) return;
	switch (varType) {
	case VariableType::BOOL: {
		this->drawBoolValue(cY, ST77XX_BLACK, ST77XX_WHITE, redrawValue || lastEntryDrawType == DRAW_ENTERED);
	} break;
	case VariableType::U8_LUT: {
		this->drawLutValue(cY, ST77XX_BLACK, ST77XX_WHITE, redrawValue || lastEntryDrawType == DRAW_ENTERED);
	} break;
	case VariableType::STRING: {
		this->drawStringValue(cY, ST77XX_BLACK, ST77XX_WHITE, redrawValue || lastEntryDrawType == DRAW_ENTERED);
	} break;
	default: {
		this->drawNumberValue(cY, ST77XX_BLACK, ST77XX_WHITE, redrawValue || lastEntryDrawType == DRAW_ENTERED);
	} break;
	}
}
void MenuItem::drawValueEntered(const i16 cY) {
	if (this->itemType != MenuItemType::VARIABLE) return;
	switch (varType) {
	case VariableType::BOOL: {
		this->drawBoolValue(cY, ST77XX_WHITE, ST77XX_BLACK, true);
	} break;
	case VariableType::U8_LUT: {
		this->drawLutValue(cY, ST77XX_WHITE, ST77XX_BLACK, true);
	} break;
	case VariableType::STRING: {
		this->drawEditableString(cY);
	} break;
	default: {
		this->drawNumberValue(cY, ST77XX_WHITE, ST77XX_BLACK, true);
	} break;
	}
}

void MenuItem::drawEntry(bool fullRedraw) {
	if (!this->visible) return;
	const i16 cY = tft.getCursorY();
	if (cY >= SCREEN_HEIGHT) return;

	// Draw profile / standard color indicator
	u16 thisColor = this->isProfileDependent ? profileColor565 : ST77XX_WHITE;
	if ((this->lastProfileColor565 != thisColor || fullRedraw) && this->itemType == MenuItemType::VARIABLE) {
#if HW_VERSION == 1
		tft.fillRect(154, cY + 1, 6, 6, thisColor);
#elif HW_VERSION == 2
		tft.fillRect(234, cY + 3, 6, 6, thisColor);
#endif
		this->lastProfileColor565 = thisColor;
	}

	u8 thisDrawType = DRAW_UNFOCUSED;

	if (this->entered)
		thisDrawType = DRAW_ENTERED;
	else if (this->focused)
		thisDrawType = DRAW_FOCUSED;

	if (thisDrawType == DRAW_UNFOCUSED && lastEntryDrawType != DRAW_UNFOCUSED) {
		// remove arrow
#if HW_VERSION == 1
		tft.fillRect(0, cY, 6, YADVANCE, ST77XX_BLACK);
#elif HW_VERSION == 2
		tft.fillRect(0, cY, 8, YADVANCE, ST77XX_BLACK);
#endif
	}
	if (thisDrawType != DRAW_UNFOCUSED && (lastEntryDrawType == DRAW_UNFOCUSED || fullRedraw)) {
		// add arrow to show focus
		tft.setCursor(0, cY);
		tft.print('>');
	}
	if (thisDrawType != DRAW_ENTERED && (lastEntryDrawType == DRAW_ENTERED || redrawValue || fullRedraw)) {
		this->drawValueNotEntered(cY);
	}
	if (thisDrawType == DRAW_ENTERED && (lastEntryDrawType != DRAW_ENTERED || redrawValue || fullRedraw)) {
		this->drawValueEntered(cY);
	}

	if (fullRedraw) {
		switch (this->itemType) {
		case MenuItemType::INFO: {
			u8 usedLines = 0;
			printCentered(this->displayName, SCREEN_WIDTH / 2, cY, SCREEN_WIDTH, MAX_SCREEN_LINES, YADVANCE, ClipBehavior::PRINT_LAST_LINE_DOTS, &usedLines);
			this->entryHeight = usedLines * YADVANCE;
			tft.setCursor(0, cY + entryHeight);
			lastEntryDrawType = DRAW_UNFOCUSED;
			return;
		} break;
		default: {
			this->entryHeight = YADVANCE;
			// one-liner, draw the setting's name
			tft.setCursor(10, cY);
			tft.print(this->displayName);
		} break;
		}
	}
	lastEntryDrawType = thisDrawType;
	redrawValue = false;
	tft.setCursor(0, cY + entryHeight);
}

void MenuItem::save() {
	if (this->itemType == MenuItemType::VARIABLE && this->startEepromPos < EEPROM_SIZE) {
		u32 addr = this->startEepromPos;
		if (this->isProfileDependent) {
			addr += selectedProfile * PROFILE_EEPROM_SIZE;
		}
		switch (this->varType) {
		case VariableType::I32:
		case VariableType::U32:
		case VariableType::FLOAT: {
			i32 val = *(i32 *)this->data;
			EEPROM.put(addr, val);
		} break;
		case VariableType::U16:
		case VariableType::I16: {
			i16 val = *(i16 *)this->data;
			EEPROM.put(addr, val);
		} break;
		case VariableType::U8:
		case VariableType::U8_LUT:
		case VariableType::I8:
		case VariableType::BOOL: {
			i8 val = *(i8 *)this->data;
			EEPROM.put(addr, val);
		} break;
		case VariableType::STRING: {
			for (u8 i = 0; i < this->maxStringLength; i++) {
				char c = ((char *)this->data)[i];
				EEPROM.put(addr + i, c);
			}
		} break;
		}
	}
	for (u8 i = 0; i < this->children.size(); i++) {
		this->children[i]->save();
	}
}

bool MenuItem::isRebootRequired() {
	for (u8 i = 0; i < this->children.size(); i++) {
		if (this->children[i]->isRebootRequired()) return true;
	}
	return rebootRequired;
}

MenuItem *MenuItem::setOnEnterFunction(bool (*onEnterFunction)(MenuItem *i)) {
	this->onEnterFunction = onEnterFunction;
	return this;
}
MenuItem *MenuItem::setOnExitFunction(bool (*onExitFunction)(MenuItem *i)) {
	this->onExitFunction = onExitFunction;
	return this;
}
MenuItem *MenuItem::setCustomLoop(bool (*customLoop)(MenuItem *i)) {
	this->customLoop = customLoop;
	return this;
}
MenuItem *MenuItem::setOnUpFunction(bool (*onUpFunction)(MenuItem *i)) {
	this->onUpFunction = onUpFunction;
	return this;
}
MenuItem *MenuItem::setOnDownFunction(bool (*onDownFunction)(MenuItem *i)) {
	this->onDownFunction = onDownFunction;
	return this;
}
MenuItem *MenuItem::setOnLeftFunction(bool (*onLeftFunction)(MenuItem *i)) {
	this->onLeftFunction = onLeftFunction;
	return this;
}
MenuItem *MenuItem::setOnRightFunction(bool (*onRightFunction)(MenuItem *i)) {
	this->onRightFunction = onRightFunction;
	return this;
}
MenuItem *MenuItem::setCustomDrawFull(void (*customDrawFull)(MenuItem *i)) {
	this->customDrawFull = customDrawFull;
	return this;
}
MenuItem *MenuItem::setOnChangeFunction(void (*onChangeFunction)(MenuItem *i)) {
	this->onChangeFunction = onChangeFunction;
	return this;
}

void MenuItem::setRange(i32 min, i32 max) {
	setMin(min);
	setMax(max);
}
void MenuItem::setRange(float min, float max) {
	setMin(min);
	setMax(max);
}
void MenuItem::setMin(i32 min) {
	this->minI = min;
	this->minF = min;
}
void MenuItem::setMin(float min) {
	this->minI = min;
	this->minF = min;
}
void MenuItem::setMax(i32 max) {
	this->maxI = max;
	this->maxF = max;
}
void MenuItem::setMax(float max) {
	this->maxI = max;
	this->maxF = max;
}
void MenuItem::checkFocus() {
	if (this->itemType != MenuItemType::SUBMENU) return;
	// check if exactly one child has focus. If not, focus the first.
	u8 focusedCount = 0;
	for (u8 i = 0; i < this->children.size(); i++) {
		if ((!this->children[i]->visible || !this->children[i]->focusable) && this->children[i]->focused) {
			this->children[i]->focused = false;
		}
		if (this->children[i]->focused) {
			focusedCount++;
		}
	}
	if (focusedCount == 1) return;
	for (u8 i = 0; i < this->children.size(); i++) {
		if (this->children[i]->focusable && this->children[i]->visible) {
			this->children[i]->focused = true;
			this->children[i]->onFocus();
			return;
		}
	}
}

#if HW_VERSION == 1
void MenuItem::scheduleBeep(i16 msSinceLast, u8 tone) {
	if (!settingsBeep) return;
	u32 settingsBeepAt = millis();
	if (msSinceLast <= 0 || msSinceLast < lastSettingsBeepTimer) {
		scheduledSettingsBeep = settingsBeepAt;
		scheduledBeepTone = tone;
		return;
	}
	settingsBeepAt += msSinceLast - lastSettingsBeepTimer;
	if (settingsBeepAt < scheduledSettingsBeep || !scheduledSettingsBeep) {
		scheduledSettingsBeep = settingsBeepAt;
		scheduledBeepTone = tone;
	}
}

void MenuItem::makeSettingsBeep(u8 tone) {
	lastSettingsBeepMotor++;
	lastSettingsBeepMotor %= 4;
	escCommandBuffer[lastSettingsBeepMotor].push(DSHOT_CMD_BEACON1 + tone);
	lastSettingsBeepTimer = 0;
}
#elif HW_VERSION == 2
void MenuItem::beep(u16 freq) {
	if (!settingsBeep) return;
	makeSound(freq, enteredRotationNavigation ? 25 : GESTURE_REPEAT_WAIT / 2);
}

u16 MenuItem::getValueBeepFreq() {
	if (this->itemType != MenuItemType::VARIABLE) return 0;
	if (!settingsBeep) return 0;
	switch (this->varType) {
	case VariableType::BOOL:
		return SETTINGS_BEEP_MIN_FREQ + ((*(bool *)this->data) != 0) * SETTINGS_BEEP_FREQ_RANGE;
	case VariableType::U8_LUT:
		return SETTINGS_BEEP_MIN_FREQ + (*(u8 *)this->data) * SETTINGS_BEEP_FREQ_RANGE / this->maxI;
	case VariableType::U8:
		return SETTINGS_BEEP_MIN_FREQ + ((*(u8 *)this->data) - this->minI) * SETTINGS_BEEP_FREQ_RANGE / (this->maxI - this->minI);
	case VariableType::I8:
		return SETTINGS_BEEP_MIN_FREQ + ((*(i8 *)this->data) - this->minI) * SETTINGS_BEEP_FREQ_RANGE / (this->maxI - this->minI);
	case VariableType::U16:
		return SETTINGS_BEEP_MIN_FREQ + ((*(u16 *)this->data) - this->minI) * SETTINGS_BEEP_FREQ_RANGE / (this->maxI - this->minI);
	case VariableType::I16:
		return SETTINGS_BEEP_MIN_FREQ + ((*(i16 *)this->data) - this->minI) * SETTINGS_BEEP_FREQ_RANGE / (this->maxI - this->minI);
	case VariableType::U32:
		return SETTINGS_BEEP_MIN_FREQ + ((*(u32 *)this->data) - this->minI) * SETTINGS_BEEP_FREQ_RANGE / (this->maxI - this->minI);
	case VariableType::I32:
		return SETTINGS_BEEP_MIN_FREQ + ((*(i32 *)this->data) - this->minI) * SETTINGS_BEEP_FREQ_RANGE / (this->maxI - this->minI);
	case VariableType::FLOAT:
		return SETTINGS_BEEP_MIN_FREQ + ((*(float *)this->data) - this->minF) * SETTINGS_BEEP_FREQ_RANGE / (this->maxF - this->minF);
	default:
		return 0;
	}
	return 0;
}
#endif

void MenuItem::onExit() {
	DEBUG_PRINTF("Exiting %s\n", this->identifier);
	if (this->itemType != MenuItemType::VARIABLE) {
		if (this->parent != nullptr) {
			this->parent->triggerFullRedraw();
		}
	}
	if (this->onExitFunction != nullptr) {
		if (!this->onExitFunction(this)) return;
	}
	this->entered = false;
	this->focusTimer = 0;
	// clear last bytes to 0
	if (this->itemType == MenuItemType::VARIABLE && this->varType == VariableType::STRING)
		for (u8 i = strlen((char *)this->data); i < this->maxStringLength; i++)
			((char *)this->data)[i] = '\0';
	if (this->itemType == MenuItemType::SUBMENU) {
		for (u8 i = 0; i < this->children.size(); i++)
			this->children[i]->focused = false;
	}
}

void MenuItem::onUp() {
	if (this->onUpFunction != nullptr) {
		if (!this->onUpFunction(this)) return;
	}
#if HW_VERSION == 1
	scheduleBeep(SETTINGS_BEEP_PERIOD, 3);
#endif
	switch (this->itemType) {
	case MenuItemType::VARIABLE: {
		this->rebootRequired = this->rebootOnChange;
		switch (this->varType) {
		case VariableType::I32: {
			i32 &val = *(i32 *)this->data;
			i32 pVal = val;
			val += this->stepSizeI;
			if (val > this->maxI || val < this->minI || (this->stepSizeI > 0 ? pVal > val : pVal < val)) {
				if (this->rollover) {
					if (this->stepSizeI > 0) {
						val = this->minI;
					} else {
						val = this->maxI;
					}
				} else {
					if (this->stepSizeI > 0) {
						val = this->maxI;
					} else {
						val = this->minI;
					}
				}
			}
#if HW_VERSION == 2
			beep(getValueBeepFreq());
#endif
		} break;
		case VariableType::U32: {
			u32 &val = *(u32 *)this->data;
			u32 pVal = val;
			val += this->stepSizeI;
			if (val > this->maxI || val < this->minI || (this->stepSizeI > 0 ? pVal > val : pVal < val)) {
				if (this->rollover) {
					if (this->stepSizeI > 0) {
						val = this->minI;
					} else {
						val = this->maxI;
					}
				} else {
					if (this->stepSizeI > 0) {
						val = this->maxI;
					} else {
						val = this->minI;
					}
				}
			}
#if HW_VERSION == 2
			beep(getValueBeepFreq());
#endif
		} break;
		case VariableType::FLOAT: {
			float &val = *(float *)this->data;
			val += this->stepSizeF;
			if (val > this->maxF || val < this->minF) {
				if (this->rollover) {
					if (this->stepSizeF > 0) {
						val = this->minF;
					} else {
						val = this->maxF;
					}
				} else {
					if (this->stepSizeF > 0) {
						val = this->maxF;
					} else {
						val = this->minF;
					}
				}
			}
#if HW_VERSION == 2
			beep(getValueBeepFreq());
#endif
		} break;
		case VariableType::U16: {
			u16 &val = *(u16 *)this->data;
			u16 pVal = val;
			val += this->stepSizeI;
			if (val > this->maxI || val < this->minI || (this->stepSizeI > 0 ? pVal > val : pVal < val)) {
				if (this->rollover) {
					if (this->stepSizeI > 0) {
						val = this->minI;
					} else {
						val = this->maxI;
					}
				} else {
					if (this->stepSizeI > 0) {
						val = this->maxI;
					} else {
						val = this->minI;
					}
				}
			}
#if HW_VERSION == 2
			beep(getValueBeepFreq());
#endif
		} break;
		case VariableType::I16: {
			i16 &val = *(i16 *)this->data;
			i16 pVal = val;
			val += this->stepSizeI;
			if (val > this->maxI || val < this->minI || (this->stepSizeI > 0 ? pVal > val : pVal < val)) {
				if (this->rollover) {
					if (this->stepSizeI > 0) {
						val = this->minI;
					} else {
						val = this->maxI;
					}
				} else {
					if (this->stepSizeI > 0) {
						val = this->maxI;
					} else {
						val = this->minI;
					}
				}
			}
#if HW_VERSION == 2
			beep(getValueBeepFreq());
#endif
		} break;
		case VariableType::U8:
		case VariableType::U8_LUT: {
			u8 &val = *(u8 *)this->data;
			u8 pVal = val;
			val += this->stepSizeI;
			if (val > this->maxI || val < this->minI || (this->stepSizeI > 0 ? pVal > val : pVal < val)) {
				if (this->rollover) {
					if (this->stepSizeI > 0) {
						val = this->minI;
					} else {
						val = this->maxI;
					}
				} else {
					if (this->stepSizeI > 0) {
						val = this->maxI;
					} else {
						val = this->minI;
					}
				}
			}
#if HW_VERSION == 2
			beep(getValueBeepFreq());
#endif
		} break;
		case VariableType::I8: {
			i8 &val = *(i8 *)this->data;
			i8 pVal = val;
			val += this->stepSizeI;
			if (val > this->maxI || val < this->minI || (this->stepSizeI > 0 ? pVal > val : pVal < val)) {
				if (this->rollover) {
					if (this->stepSizeI > 0) {
						val = this->minI;
					} else {
						val = this->maxI;
					}
				} else {
					if (this->stepSizeI > 0) {
						val = this->maxI;
					} else {
						val = this->minI;
					}
				}
			}
#if HW_VERSION == 2
			beep(getValueBeepFreq());
#endif
		} break;
		case VariableType::STRING: {
			char &c = ((char *)this->data)[charPos];
			switch (c) {
			case 'Z':
				c = 'a';
				break;
			case 'z':
				c = '.';
				break;
			case '.':
				c = ',';
				break;
			case ',':
				c = '-';
				break;
			case '-':
				c = '/';
				break;
			case '/':
				c = '+';
				break;
			case '+':
				c = '&';
				break;
			case '&':
				c = '!';
				break;
			case '!':
				c = '?';
				break;
			case '?':
				c = '#';
				break;
			case '#':
				c = '<';
				break;
			case '<':
				c = '>';
				break;
			case '>':
				c = '(';
				break;
			case '(':
				c = ')';
				break;
			case ')':
				c = '@';
				break;
			case '@':
				c = ' ';
				break;
			case ' ':
				c = '\0';
				break;
			case '\0':
				c = '0';
				break;
			case '9':
				c = 'A';
				break;
			default:
				c++;
			}
#if HW_VERSION == 2
			beep(SETTINGS_BEEP_MAX_FREQ);
#endif
		} break;
		}
		if (this->onChangeFunction != nullptr) {
			this->onChangeFunction(this);
		}
		this->redrawValue = true;
	} break;
	case MenuItemType::SUBMENU: {
		u8 size = this->children.size();
		for (u8 i = 0; i < size; i++) {
			if (this->children[i]->focused) {
				for (u8 j = i + size - 1; j != i; j--) {
					MenuItem *c = this->children[j % size];
					if (c->focusable && c->visible) {
						c->focused = true;
						c->onFocus();
						this->children[i]->focused = false;
						break;
					}
				}
				break;
			}
		}
#if HW_VERSION == 2
		beep(SETTINGS_BEEP_MAX_FREQ);
#endif
	} break;
#if HW_VERSION == 2
	default:
		beep(SETTINGS_BEEP_MAX_FREQ);
		break;
#endif
	}
}

void MenuItem::onDown() {
	if (this->onDownFunction != nullptr) {
		if (!this->onDownFunction(this)) return;
	}
#if HW_VERSION == 1
	scheduleBeep(SETTINGS_BEEP_PERIOD, 0);
#endif
	switch (this->itemType) {
	case MenuItemType::VARIABLE: {
		this->rebootRequired = this->rebootOnChange;
		switch (this->varType) {
		case VariableType::I32: {
			i32 &val = *(i32 *)this->data;
			i32 pVal = val;
			val -= this->stepSizeI;
			if (val < this->minI || val > this->maxI || (this->stepSizeI > 0 ? pVal < val : pVal > val)) {
				if (this->rollover) {
					if (this->stepSizeI > 0) {
						val = this->maxI;
					} else {
						val = this->minI;
					}
				} else {
					if (this->stepSizeI > 0) {
						val = this->minI;
					} else {
						val = this->maxI;
					}
				}
			}
#if HW_VERSION == 2
			beep(getValueBeepFreq());
#endif
		} break;
		case VariableType::U32: {
			u32 &val = *(u32 *)this->data;
			u32 pVal = val;
			val -= this->stepSizeI;
			if (val < this->minI || val > this->maxI || (this->stepSizeI > 0 ? pVal < val : pVal > val)) {
				if (this->rollover) {
					if (this->stepSizeI > 0) {
						val = this->maxI;
					} else {
						val = this->minI;
					}
				} else {
					if (this->stepSizeI > 0) {
						val = this->minI;
					} else {
						val = this->maxI;
					}
				}
			}
#if HW_VERSION == 2
			beep(getValueBeepFreq());
#endif
		} break;
		case VariableType::FLOAT: {
			float &val = *(float *)this->data;
			val -= this->stepSizeF;
			if (val < this->minF || val > this->maxF) {
				if (this->rollover) {
					if (this->stepSizeF > 0) {
						val = this->maxF;
					} else {
						val = this->minF;
					}
				} else {
					if (this->stepSizeF > 0) {
						val = this->minF;
					} else {
						val = this->maxF;
					}
				}
			}
#if HW_VERSION == 2
			beep(getValueBeepFreq());
#endif
		} break;
		case VariableType::U16: {
			u16 &val = *(u16 *)this->data;
			val -= this->stepSizeI;
			if (val < this->minI || val > this->maxI) {
				if (this->rollover) {
					if (this->stepSizeI > 0) {
						val = this->maxI;
					} else {
						val = this->minI;
					}
				} else {
					if (this->stepSizeI > 0) {
						val = this->minI;
					} else {
						val = this->maxI;
					}
				}
			}
#if HW_VERSION == 2
			beep(getValueBeepFreq());
#endif
		} break;
		case VariableType::I16: {
			i16 &val = *(i16 *)this->data;
			i16 pVal = val;
			val -= this->stepSizeI;
			if (val < this->minI || val > this->maxI || (this->stepSizeI > 0 ? pVal < val : pVal > val)) {
				if (this->rollover) {
					if (this->stepSizeI > 0) {
						val = this->maxI;
					} else {
						val = this->minI;
					}
				} else {
					if (this->stepSizeI > 0) {
						val = this->minI;
					} else {
						val = this->maxI;
					}
				}
			}
#if HW_VERSION == 2
			beep(getValueBeepFreq());
#endif
		} break;
		case VariableType::U8:
		case VariableType::U8_LUT: {
			u8 &val = *(u8 *)this->data;
			u8 pVal = val;
			val -= this->stepSizeI;
			if (val < this->minI || val > this->maxI || (this->stepSizeI > 0 ? pVal < val : pVal > val)) {
				if (this->rollover) {
					if (this->stepSizeI > 0) {
						val = this->maxI;
					} else {
						val = this->minI;
					}
				} else {
					if (this->stepSizeI > 0) {
						val = this->minI;
					} else {
						val = this->maxI;
					}
				}
			}
#if HW_VERSION == 2
			beep(getValueBeepFreq());
#endif
		} break;
		case VariableType::I8: {
			i8 &val = *(i8 *)this->data;
			i8 pVal = val;
			val -= this->stepSizeI;
			if (val < this->minI || val > this->maxI || (this->stepSizeI > 0 ? pVal < val : pVal > val)) {
				if (this->rollover) {
					if (this->stepSizeI > 0) {
						val = this->maxI;
					} else {
						val = this->minI;
					}
				} else {
					if (this->stepSizeI > 0) {
						val = this->minI;
					} else {
						val = this->maxI;
					}
				}
			}
#if HW_VERSION == 2
			beep(getValueBeepFreq());
#endif
		} break;
		case VariableType::STRING: {
			char &c = ((char *)this->data)[charPos];
			switch (c) {
			case '.':
				c = 'z';
				break;
			case 'a':
				c = 'Z';
				break;
			case ',':
				c = '.';
				break;
			case '-':
				c = ',';
				break;
			case '/':
				c = '-';
				break;
			case '+':
				c = '/';
				break;
			case '&':
				c = '+';
				break;
			case '!':
				c = '&';
				break;
			case '?':
				c = '!';
				break;
			case '#':
				c = '?';
				break;
			case '<':
				c = '#';
				break;
			case '>':
				c = '<';
				break;
			case '(':
				c = '>';
				break;
			case ')':
				c = '(';
				break;
			case '@':
				c = ')';
				break;
			case ' ':
				c = '@';
				break;
			case '\0':
				c = ' ';
				break;
			case '0':
				c = '\0';
				break;
			case 'A':
				c = '9';
				break;
			default:
				c--;
			}
#if HW_VERSION == 2
			beep(SETTINGS_BEEP_MIN_FREQ);
#endif
		} break;
		}
		if (this->onChangeFunction != nullptr) {
			this->onChangeFunction(this);
		}
		this->redrawValue = true;
	} break;
	case MenuItemType::SUBMENU: {
		u8 size = this->children.size();
		for (u8 i = 0; i < size; i++) {
			if (this->children[i]->focused) {
				for (u8 j = i + 1; j != i + size; j++) {
					MenuItem *c = this->children[j % size];
					if (c->focusable && c->visible) {
						c->focused = true;
						c->onFocus();
						this->children[i]->focused = false;
						break;
					}
				}
				break;
			}
		}
#if HW_VERSION == 2
		beep(SETTINGS_BEEP_MIN_FREQ);
#endif
	} break;
#if HW_VERSION == 2
	default:
		beep(SETTINGS_BEEP_MIN_FREQ);
		break;
#endif
	}
}

void MenuItem::onLeft() {
	if (this->onLeftFunction != nullptr) {
		if (!this->onLeftFunction(this)) return;
	}
	if (this->itemType == MenuItemType::VARIABLE && this->varType == VariableType::STRING) redrawValue = true;
	if (this->itemType == MenuItemType::VARIABLE && this->varType == VariableType::STRING && this->charPos > 0) {
		this->charPos--;
#if HW_VERSION == 1
		scheduleBeep(0, 1);
#elif HW_VERSION == 2
		beep(SETTINGS_BEEP_MIN_FREQ + SETTINGS_BEEP_FREQ_RANGE / 3);
#endif
	} else if (lastGesture.type == GESTURE_PRESS) {
#if HW_VERSION == 1
		scheduleBeep(0, 1);
#elif HW_VERSION == 2
		beep(SETTINGS_BEEP_MIN_FREQ + SETTINGS_BEEP_FREQ_RANGE / 3);
#endif
		this->onExit();
	}
}

void MenuItem::onRight() {
	if (this->onRightFunction != nullptr) {
		if (!this->onRightFunction(this)) return;
	}
	switch (this->itemType) {
	case MenuItemType::VARIABLE:
		if (this->varType == VariableType::STRING) {
			redrawValue = true;
#if HW_VERSION == 1
			scheduleBeep(SETTINGS_BEEP_PERIOD, 1);
#elif HW_VERSION == 2
			beep(SETTINGS_BEEP_MIN_FREQ + SETTINGS_BEEP_FREQ_RANGE / 3);
#endif
		}
		if (this->varType == VariableType::STRING && this->charPos < strlen((char *)this->data) && this->charPos < this->maxStringLength - 2)
			this->charPos++;
		else
			this->charPos = 0;
		break;
	case MenuItemType::SUBMENU:
		if (lastGesture.type == GESTURE_PRESS) {
#if HW_VERSION == 1
			scheduleBeep(SETTINGS_BEEP_PERIOD, 1);
#elif HW_VERSION == 2
			beep(SETTINGS_BEEP_MIN_FREQ + SETTINGS_BEEP_FREQ_RANGE / 3);
#endif
			for (u8 i = 0; i < this->children.size(); i++) {
				if (this->children[i]->focused) {
					this->children[i]->onEnter();
					break;
				}
			}
		}
		break;
	default:
#if HW_VERSION == 1
		scheduleBeep(SETTINGS_BEEP_PERIOD, 1);
#elif HW_VERSION == 2
		beep(SETTINGS_BEEP_MIN_FREQ + SETTINGS_BEEP_FREQ_RANGE / 3);
#endif
	}
}

void MenuItem::drawFull() {
#if HW_VERSION == 2
	speakerLoopOnFastCore = true;
#endif
	if (this->customDrawFull != nullptr) {
		this->customDrawFull(this);
#if HW_VERSION == 2
		speakerLoopOnFastCore = false;
#endif
		return;
	}
	if (this->itemType != MenuItemType::SUBMENU) return;
	if (fullRedraw) tft.fillScreen(ST77XX_BLACK);
	tft.setCursor(0, -scrollTop);
	tft.setTextWrap(false);
	for (u8 i = 0; i < this->children.size(); i++) {
		this->children[i]->drawEntry(fullRedraw);
	}
	fullRedraw = false;
#if HW_VERSION == 2
	speakerLoopOnFastCore = false;
#endif
}