	#include "global.h"

	#if HW_VERSION == 2

	bool standbyOn = false;
	bool enableStandbyFlag = false;
	bool fastStandbyEnabled = true;

	static inline bool usbMounted() {
	#ifdef USE_TINYUSB
		// Core 1 must not call TinyUSB/Serial internals directly (thread-safety).
		return usbSessionActive();
	#else
		return false;
	#endif
	}

elapsedMillis noMotionTimer = 0;
elapsedMillis standbyBeepTimer = 0;
elapsedMillis standbyOnTimer = 0;
u32 standbyBeepTimeout = 600000; // 10 minutes
u8 standbyBeepTimes = 0;

void initStandbySwitch() {
	gpio_init(PIN_STANDBY);
	gpio_set_dir(PIN_STANDBY, GPIO_OUT);
	gpio_put(PIN_STANDBY, 0);
}

	void standbyOnLoop() {
		// If the device is connected over USB, keep it awake.
		if (usbMounted()) {
			disableStandby();
			return;
		}
		if (triggerState && triggerUpdateFlag) {
			triggerUpdateFlag = false;
			disableStandby();
		}
	if (gpio_get(PIN_GYRO_INT)) {
		if (noMotionTimer > 30000) // prevent turning back while it is still moving, only enable motion detection after 30s
			disableStandby();
		noMotionTimer = 0;
	}
	if (!standbyOn) return;
	static bool lightState = false;
	if (standbyOnTimer % 30000 > 27000) {
		if (!lightState) {
			lightState = true;
			ledSetMode(LED_MODE::FADE, LIGHT_ID::STANDBY, 0, 0, 160, 0);
		}
	} else if (lightState) {
		lightState = false;
		ledSetMode(LED_MODE::OFF, LIGHT_ID::STANDBY, 0);
	}
	if (standbyBeepTimer >= standbyBeepTimeout) {
		standbyBeepTimes++;
		standbyBeepTimer = 0;
		standbyBeepTimeout = 600000; // beep every 10 minutes
		if (standbyBeepTimes < 4) { // up to 1h
			makeRtttlSound("standby2:d=4,o=7,b=240:5c,1p,1p,1p,1p,1p,5c");
		} else if (standbyBeepTimes < 10) { // up to 2h
			makeRtttlSound("standby2:d=4,o=7,b=240:5c,2p,5c,1p,1p,1p,1p,1p,5c,2p,5c");
		} else { // > 2h
			makeSweepSound(1000, 3000, 3700, 900, 500);
		}
	}
}

	void standbyOffLoop() {
		// If the device is connected over USB, keep it awake.
		if (usbMounted()) {
			inactivityTimer = 0;
			noMotionTimer = 0;
			return;
		}
		if (gpio_get(PIN_GYRO_INT)) {
			// if blaster is moving, reset timer
			noMotionTimer = 0;
		}
	if (inactivityTimeout && inactivityTimer > 1000 * 60 * inactivityTimeout) {
		enableStandby();
	}
	if (inactivityTimeout && fastStandbyEnabled && noMotionTimer > 60000 && inactivityTimer > 60000) {
		enableStandby();
	}
	if (enableStandbyFlag) {
		enableStandbyFlag = false;
		enableStandby();
	}
}

void enableStandby() {
	if (standbyOn) return;
	if (operationState == STATE_BOOT_SELECT) return;
	if (operationState == STATE_PROFILE_SELECT) return;
	if (operationState >= STATE_RAMPUP && operationState <= STATE_RAMPDOWN) return;
	for (int i = 0; i < 4; i++) {
		gpio_init(PIN_MOTOR_BASE + i);
		gpio_set_dir(PIN_MOTOR_BASE + i, GPIO_IN);
		gpio_set_pulls(PIN_MOTOR_BASE + i, false, true);
	}
	tft.fillScreen(ST77XX_BLACK);
	if (escRpm[0] > 0 || escRpm[1] > 0 || escRpm[2] > 0 || escRpm[3] > 0) {
		for (u16 c = 0; c < PID_RATE / 2; c++) {
			// if any motor is still spinning, send 0 throttle for half a second
			sleep_us(1000000 / PID_RATE);
			setAllThrottles(0);
			sendThrottles(throttles);
		}
	}
#ifdef USE_TOF
	disableTof();
#endif
	gpio_put(PIN_STANDBY, 1);
	deinitPusher();
	initBat();
	u16 data = 0xBB00;
	regWrite(SPI_GYRO, PIN_GYRO_CS, (u8)GyroReg::ANYMO_2, (u8 *)&data, 2, 500);
	set_sys_clock_hz(18000000, true);
	ledNewClock();
	pwm_set_clkdiv_int_frac(speakerPwmSlice, 18, 0);
	ledSetMode(LED_MODE::OFF, LIGHT_ID::STANDBY);
	standbyOn = true;
	standbyOnTimer = 0;
	standbyBeepTimer = 0;
	standbyBeepTimeout = 1800000; // first beep after 30 minutes
	standbyBeepTimes = 0;
}

void disableStandby() {
	if (!standbyOn) return;
	set_sys_clock_hz(F_CPU, true);
	ledNewClock();
	pwm_set_clkdiv_int_frac(speakerPwmSlice, F_CPU / MHZ, 0);
	u16 data = 0xB820;
	regWrite(SPI_GYRO, PIN_GYRO_CS, (u8)GyroReg::ANYMO_2, (u8 *)&data, 2, 500);
	gpio_put(PIN_STANDBY, 0);
	initDisplay();
	initPusher();
#ifdef USE_TOF
	initTof();
#endif
	// display takes pin 12 as default MISO pin#
	for (int i = 0; i < 4; i++) {
		gpio_set_dir(PIN_MOTOR_BASE + i, GPIO_OUT);
		gpio_set_pulls(PIN_MOTOR_BASE + i, true, false);
		pio_gpio_init(ESC_PIO, PIN_MOTOR_BASE + i);
	}
	stopSound();
	joystickLockout = false;
	triggerFullRedraw();
	forceInitBat = true;
#if HW_VERSION == 2
	releaseLightId(LIGHT_ID::STANDBY);
#endif
	inactivityTimer = 0;
	noMotionTimer = 0;
	bootTimer = 0;
	if (operationState == STATE_OFF) forceNewOpState = STATE_SAFE;
	standbyOn = false;
}
#endif
