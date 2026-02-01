#pragma once

#ifndef HW_VERSION
#error "HW_VERSION must be defined"
#endif

#if HW_VERSION == 1
#include "Adafruit_ST7735.h"
#elif HW_VERSION == 2
#include "Adafruit_ST7789.h"
#include "drivers/ledStrip.h"
#include <NeoPixelConnect.h>
#else
#error "Unsupported HW_VERSION"
#endif
#ifdef USE_TINYUSB
#include <Adafruit_TinyUSB.h>
#endif

#include "Bass11px.h"
#include "Bass12px.h"
#include "SPI.h"
#include "Wire.h"
#include "analog.h"
#include "bootSelect.h"
#include "drivers/bat.h"
#include "drivers/display.h"
#include "drivers/esc.h"
#include "drivers/gyro.h"
#include "drivers/joystick.h"
#include "drivers/ledStrip.h"
#include "drivers/speaker.h"
#include "drivers/spi.h"
#include "drivers/tof.h"
#include "drivers/trigger.h"
#include "eepromImpl.h"
#include "elapsedMillis.h"
#include "esc_passthrough.h"
#include "hardware/adc.h"
#include "hardware/pwm.h"
#include "hardware/timer.h"
#include "imu.h"
#include "menu/flywheelBalancing.h"
#include "menu/inputDiagnostics.h"
#include "menu/menu.h"
#include "menu/menuItem.h"
#include "menu/motorRemap.h"
#include "menu/qr.h"
#include "operationSm.h"
#include "pid.h"
#include "pins.h"
#include "pusher.h"
#include "standby.h"
#include "tournament.h"
#include "mlLog.h"
#include "mlInfer.h"
#include "utils/filters.h"
#include "utils/fixedPointInt.h"
#include "utils/quaternion.h"
#include "utils/typedefs.h"
#include <Arduino.h>
#include <EEPROM.h>
#include <PIO_DShot.h>

#define PID_RATE 3200

#if HW_VERSION == 1
#define SPI_DISPLAY spi0 // SPI for display
#define SPI_DISPLAY_ARDUINO SPI
#define WIRE_TOF Wire
#elif HW_VERSION == 2
#define SPI_GYRO spi0 // SPI for gyro
#define SPI_DISPLAY spi1 // SPI for display
#define SPI_DISPLAY_ARDUINO SPI1
#define WIRE_TOF Wire1
#endif

extern elapsedMillis bootTimer; // time since setup is done, prevents artifacts from delays in setup

extern char deviceName[16];
extern char ownerName[32];
extern char ownerContact[32];

#define FIRMWARE_NAME "Stinger"
#define FIRMWARE_VERSION_MAJOR 2
#define FIRMWARE_VERSION_MINOR 2
#define FIRMWARE_VERSION_PATCH 2
#define RELEASE_SUFFIX ""
#define xstr(a) str(a)

#define str(a) #a
#define HW_VERSION_STRING xstr(HW_VERSION)
#define FIRMWARE_VERSION_STRING xstr(FIRMWARE_VERSION_MAJOR) "." xstr(FIRMWARE_VERSION_MINOR) "." xstr(FIRMWARE_VERSION_PATCH) RELEASE_SUFFIX
#ifdef PRINT_DEBUG
#define DEBUG_PRINT(x) Serial.print(x)
#define DEBUG_PRINTSLN(x) \
	Serial.printf("%15s:%3d: %s\n", __FILE__, __LINE__, x);
#define DEBUG_PRINTLN(x)                             \
	Serial.printf("%15s:%3d: ", __FILE__, __LINE__); \
	Serial.println(x);
#define DEBUG_PRINTF(x, ...)                         \
	Serial.printf("%15s:%3d: ", __FILE__, __LINE__); \
	Serial.printf(x, __VA_ARGS__)
#endif
#ifndef PRINT_DEBUG
#define DEBUG_PRINT(x)
#define DEBUG_PRINTSLN(x)
#define DEBUG_PRINTF(x, ...)
#endif

enum class BootReason {
	POR, // Power-on reset
	WATCHDOG,
	CLEAR_EEPROM,
	MENU,
	TO_ESC_PASSTHROUGH,
	FROM_ESC_PASSTHROUGH,
	FROM_BOOT_SELECTION
};

extern BootReason bootReason; // Reason for booting
extern BootReason rebootReason; // Reason for rebooting (can be set right before an intentional reboot, WATCHDOG otherwise)
extern u64 powerOnResetMagicNumber; // Magic number to detect power-on reset (0xdeadbeefdeadbeef)
