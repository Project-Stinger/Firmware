#pragma once
#include "elapsedMillis.h"
#include "utils/fixedPointInt.h"
#include "utils/ringbuffer.h"
#include "utils/typedefs.h"
#include <Arduino.h>

#define MOTOR_POLES 12

enum DshotTelemetryType : u8 {
	TELEMETRY_ERPM = 0,
	TELEMETRY_TEMP,
	TELEMETRY_VOLTAGE,
	TELEMETRY_CURRENT,
	TELEMETRY_DBG1,
	TELEMETRY_DBG2,
	TELEMETRY_STRESS_LEVEL,
	TELEMETRY_STATUS,
	TELEMETRY_MAX,
};

#define ESC_STATUS_MAX_STRESS_MASK 0b00001111
#define ESC_STATUS_ERROR_MASK 0b00100000
#define ESC_STATUS_WARNING_MASK 0b01000000
#define ESC_STATUS_ALERT_MASK 0b10000000

#define ESC_PIO pio0
extern volatile u32 erpmEdges[4][32]; // edge detection durations for each motor
extern volatile u32 escRpm[4]; // decoded RPM values
extern const u32 escDecodeLut[32]; // lookup table for GCR decoding
extern u8 escErpmFail; // flags for failed RPM decoding
extern u8 escTemp[4]; // ESC temps in degrees C
extern fix32 escVoltage[4];
extern u8 escCurrent[4]; // ESC current in 1A
extern u8 escStatus[4]; // ESC status byte
extern u32 escStatusCount[4]; // number of status updates received
extern elapsedMillis escStatusTimer[4]; // time since last status update
extern RingBuffer<u16> escCommandBuffer[4]; // buffer for DShot commands (11 bit raw)
extern u8 escMaxTemp; // maximum temperature for the ESCs (°C)
extern volatile u8 escPins[4]; // pins used for the ESC communication
extern u32 telemCounter[4];
extern u8 knownEscTemp;

/// @brief Initializes the ESC communication
void initESCs();

void pushToAllCommandBufs(u16 command);

void initEscLayout();
void saveEscLayout();
void updateEscPins(volatile u8 pins[4]);

/**
 * @brief Sends throttles to all four ESCs
 *
 * @details Telemetry bit is not set
 *
 * @param throttles Array of four throttle values (0-2000)
 */
void sendThrottles(const i16 throttles[4]);

/**
 * @brief Sends raw values to all four ESCs (useful for special commands)
 *
 * @details Telemetry bit always set
 *
 * @param raw Array of four raw values (0-2047, with 1-47 being the special commands, and the others being the throttle values)
 */
void sendRaw11Bit(const u16 raw[4]);

/**
 * @brief Sends raw values to all four ESCs (useful for special commands)
 *
 * @param raw Array of four raw values, including the telemetry bits
 */
void sendRaw12Bit(const u16 raw[4]);

void decodeErpm();

void checkTelemetry();

bool startEscPassthrough(MenuItem *_item);

bool calibrateEscTemp(MenuItem *_item);

// 0 = success, 1-3 = error codes, others: disable all
void showTempCalResult(int i);
