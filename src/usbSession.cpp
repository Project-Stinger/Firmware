#include "global.h"

#if HW_VERSION == 2

static volatile bool gUsbMounted = false;
static volatile bool gUsbCdcActive = false;

void usbSessionInit() {
	__atomic_store_n(&gUsbMounted, false, __ATOMIC_RELAXED);
	__atomic_store_n(&gUsbCdcActive, false, __ATOMIC_RELAXED);
}

void usbSessionLoop0() {
#ifdef USE_TINYUSB
	// Keep TinyUSB state machine ticking so mounted()/DTR reflects real cable/host state.
	TinyUSBDevice.task();
	const bool mounted = TinyUSBDevice.mounted();
	const bool cdcActive = mounted && (Serial.dtr() != 0);
	__atomic_store_n(&gUsbMounted, mounted, __ATOMIC_RELEASE);
	__atomic_store_n(&gUsbCdcActive, cdcActive, __ATOMIC_RELEASE);
#else
	__atomic_store_n(&gUsbMounted, false, __ATOMIC_RELEASE);
	__atomic_store_n(&gUsbCdcActive, false, __ATOMIC_RELEASE);
#endif
}

bool usbSessionActive() {
	return __atomic_load_n(&gUsbMounted, __ATOMIC_ACQUIRE);
}

bool usbCdcActive() {
	return __atomic_load_n(&gUsbCdcActive, __ATOMIC_ACQUIRE);
}

#else

void usbSessionInit() {}
void usbSessionLoop0() {}
bool usbSessionActive() { return false; }
bool usbCdcActive() { return false; }

#endif
