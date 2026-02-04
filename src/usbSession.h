#pragma once

#include "utils/typedefs.h"

// We expose two related flags:
// - usbSessionActive(): USB cable is connected / TinyUSB is mounted.
// - usbCdcActive(): USB is mounted AND a host has opened the CDC port (DTR asserted).
//
// IMPORTANT: TinyUSB APIs are not guaranteed to be thread-safe across both cores.
// We therefore sample the USB session state on core 0 and publish a simple flag that
// core 1 can read without touching TinyUSB/Serial internals.

void usbSessionInit();
void usbSessionLoop0(); // call from loop() (core 0)
bool usbSessionActive(); // safe from both cores
bool usbCdcActive(); // safe from both cores
