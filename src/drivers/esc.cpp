#include "../global.h"
#include "utils/ringbuffer.h"

BidirDShotX1 *escs[4] = {nullptr, nullptr, nullptr, nullptr};
volatile u32 escRpm[4] = {0};
u8 escTemp[4] = {0};
u8 escTempRaw[4] = {0};
fix32 escVoltage[4] = {0};
u8 escCurrent[4] = {0};
u8 escStatus[4] = {0};
u32 escStatusCount[4] = {0};
elapsedMillis escStatusTimer[4] = {0};
elapsedMillis escErrorTimer[4] = {0};
elapsedMillis lastTelemetryFrame[4] = {0};
u8 escErpmFail = 0;
RingBuffer<u16> escCommandBuffer[4] = {RingBuffer<u16>(64), RingBuffer<u16>(64), RingBuffer<u16>(64), RingBuffer<u16>(64)};
u8 escMaxTemp = 115;
volatile u8 escPins[4] = {255, 255, 255, 255};
mutex_t escMutex;
i8 escTempOffset[4] = {0, 0, 0, 0}; // offset for ESC temperature calibration
u8 knownEscTemp = 23;

void initEscLayout() {
	u8 layout;
	if (firstBoot) {
		layout = 0b11100100;
		EEPROM.put(EEPROM_POS_MOTOR_LAYOUT, layout);
	} else {
		EEPROM.get(EEPROM_POS_MOTOR_LAYOUT, layout);
	}
	for (int i = 0; i < 4; i++) {
		escPins[i] = PIN_MOTOR_BASE + (layout & 0x03);
		layout >>= 2;
	}
}

void showTempCalResult(int i) {
	mainMenu->search("escTempCalSuccess")->setVisible(i == 0);
	mainMenu->search("escTempCalError1")->setVisible(i == 1);
	mainMenu->search("escTempCalError2")->setVisible(i == 2);
	mainMenu->search("escTempCalError3")->setVisible(i == 3);
}

bool calibrateEscTemp(MenuItem *_item) {
	if (batState != BAT_RUNNING) {
		showTempCalResult(2);
		return false;
	}
	mutex_enter_blocking(&escMutex);
	for (int i = 0; i < 4; i++) {
		if (escTempRaw[i] > 80) {
			showTempCalResult(1);
			return false;
		}
		if (!escTempRaw[i]) {
			showTempCalResult(3);
			return false;
		}
	}
	for (int i = 0; i < 4; i++) {
		escTempOffset[i] = knownEscTemp - escTempRaw[i];
		EEPROM.put(EEPROM_POS_ESC_TEMP_OFFSETS + i, escTempOffset[i]);
	}
	mutex_exit(&escMutex);
	showTempCalResult(0);
	DEBUG_PRINTSLN("ESC temperature calibration done");
	return false;
}

void pushToAllCommandBufs(u16 command) {
	mutex_enter_blocking(&escMutex);
	for (int i = 0; i < 4; i++) {
		escCommandBuffer[i].push(command);
	}
	mutex_exit(&escMutex);
}

void saveEscLayout() {
	u8 layout = 0;
	for (int i = 0; i < 4; i++) {
		layout |= (escPins[i] - PIN_MOTOR_BASE) << (i * 2);
	}
	EEPROM.put(EEPROM_POS_MOTOR_LAYOUT, layout);
	if (!firstBoot)
		EEPROM.commit();
}

void updateEscPins(volatile u8 pins[4]) {
	for (int i = 0; i < 4; i++) {
		escPins[i] = pins[i];
	}
	// apply mapping to the PIO, not saving the settings yet
	mutex_enter_blocking(&escMutex);
	for (int i = 0; i < 4; i++) {
		delete escs[i];
	}
	for (int i = 0; i < 4; i++) {
		escs[i] = new BidirDShotX1(escPins[i], 600, ESC_PIO, i);
	}
	mutex_exit(&escMutex);
}

void initESCs() {
	mutex_init(&escMutex);
	mutex_enter_blocking(&escMutex);
	for (int i = 0; i < 4; i++) {
		escs[i] = new BidirDShotX1(escPins[i], 600, ESC_PIO, i);
		if (firstBoot)
			EEPROM.put(EEPROM_POS_ESC_TEMP_OFFSETS + i, (i8)0);
		else
			EEPROM.get(EEPROM_POS_ESC_TEMP_OFFSETS + i, escTempOffset[i]);
	}
	mutex_exit(&escMutex);
}

void sendRaw12Bit(const u16 raw[4]) {
	mutex_enter_blocking(&escMutex);
	for (int i = 0; i < 4; i++) {
		escs[i]->sendRaw12Bit(raw[i]);
	}
	mutex_exit(&escMutex);
}

void sendRaw11Bit(const u16 raw[4]) {
	mutex_enter_blocking(&escMutex);
	for (int i = 0; i < 4; i++) {
		escs[i]->sendRaw11Bit(raw[i]);
	}
	mutex_exit(&escMutex);
}

void sendThrottles(const i16 throttles[4]) {
	mutex_enter_blocking(&escMutex);
	for (int i = 0; i < 4; i++) {
		if (!escCommandBuffer[i].isEmpty()) {
			escs[i]->sendRaw11Bit(escCommandBuffer[i].pop());
		} else {
			escs[i]->sendThrottle(throttles[i]);
		}
	}
	mutex_exit(&escMutex);
}

u32 telemCounter[4] = {0};
void decodeErpm() {
	mutex_enter_blocking(&escMutex);
	for (int m = 0; m < 4; m++) {
		u32 packet = 0;
		bool fail = false;
		BidirDshotTelemetryType type = escs[m]->getTelemetryPacket((uint32_t *)&packet);
		switch (type) {
		case BidirDshotTelemetryType::NO_PACKET:
		case BidirDshotTelemetryType::CHECKSUM_ERROR:
			fail = true;
			break;
		case BidirDshotTelemetryType::ERPM:
			escRpm[m] = packet / (MOTOR_POLES / 2);
			break;
		case BidirDshotTelemetryType::TEMPERATURE:
			escTemp[m] = packet + escTempOffset[m]; // temperature in °C
			escTempRaw[m] = packet;
			break;
		case BidirDshotTelemetryType::VOLTAGE:
			escVoltage[m].setRaw(packet << 14); // steps of 0.25V
			break;
		case BidirDshotTelemetryType::CURRENT:
			escCurrent[m] = packet;
			break;
		case BidirDshotTelemetryType::STATUS:
			escStatusTimer[m] = 0;
			escStatusCount[m]++;
			escStatus[m] = packet;
			break;
		}
		if (fail) {
			escErpmFail |= 1 << m;
		} else {
			lastTelemetryFrame[m] = 0;
			escErpmFail &= ~(1 << m);
			telemCounter[m]++;
		}
	}
	mutex_exit(&escMutex);
}

void checkTelemetry() {
	bool overtemp = false;
	for (int i = 0; i < 4; i++) {
		if (escTemp[i] > escMaxTemp)
			overtemp = true;
		if (escStatus[i] & ESC_STATUS_ERROR_MASK) {
			escErrorTimer[i] = 0;
			DEBUG_PRINTSLN("ESC error");
		}
	}
	if (lastTelemetryFrame[0] > 1500 || lastTelemetryFrame[1] > 1500 || lastTelemetryFrame[2] > 1500 || lastTelemetryFrame[3] > 1500)
		motorDisableFlags |= MD_NO_TELEMETRY;
	else {
		motorDisableFlags &= ~MD_NO_TELEMETRY;
	}
	if (overtemp)
		motorDisableFlags |= MD_ESC_OVERTEMP;
	else
		motorDisableFlags &= ~MD_ESC_OVERTEMP;
}

bool startEscPassthrough(MenuItem *_item) {
	rebootReason = BootReason::TO_ESC_PASSTHROUGH;
	rp2040.reboot();
	return false;
}
