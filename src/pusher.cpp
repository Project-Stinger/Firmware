#include "global.h"
#include "pio/pulse.pio.h"
u8 fireMode = FIRE_SINGLE;
bool burstKeepFiring = false;
u8 burstCount = 3, pushCount = 0;
u32 pushDuration = 20000, retractDuration = 50000;
u8 minPushDuration = 15, minRetractDuration = 40;
u8 dartsPerSecond = 10;
u8 pusherState = 0; // 0 = passive, 1 = extending (forward voltage), 2 = discharging (reverse voltage)
const char fireModeNames[3][FIRE_MODE_STRING_LENGTH] = {"Semi", "Burst", "Auto"};

#if HW_VERSION == 2
fix32 solenoidCurrent = 0;
elapsedMicros sinceExtend = 0;
elapsedMicros sinceRetract = 0;
bool pusherFullyExtended = false;
bool pusherFullyRetracted = true;
fix32 solenoidCurrentLast[2] = {0, 0};
u32 mic[2] = {0, 0};
bool autoPusherTiming = true;
u8 dpsLimit = 40;
bool pusherEnabled = false;
#endif

void initPusher() {
#if HW_VERSION == 1
	gpio_init(PIN_SOLENOID);
	gpio_set_dir(PIN_SOLENOID, GPIO_OUT);
	gpio_put(PIN_SOLENOID, 0);
#elif HW_VERSION == 2
	gpio_init(PIN_SOLENOID_NSLEEP);
	gpio_init(PIN_SOLENOID_IN1);
	gpio_init(PIN_SOLENOID_IN2);
	gpio_set_dir(PIN_SOLENOID_NSLEEP, GPIO_OUT);
	gpio_set_dir(PIN_SOLENOID_IN1, GPIO_OUT);
	gpio_set_dir(PIN_SOLENOID_IN2, GPIO_OUT);
	gpio_put(PIN_SOLENOID_NSLEEP, 1);
	gpio_put(PIN_SOLENOID_IN1, 0);
	gpio_put(PIN_SOLENOID_IN2, 0);
#endif
	pusherState = 0;
}

void deinitPusher() {
#if HW_VERSION == 2
	gpio_put(PIN_SOLENOID_NSLEEP, 0);
	gpio_put(PIN_SOLENOID_IN1, 0);
	gpio_put(PIN_SOLENOID_IN2, 0);
	gpio_set_dir(PIN_SOLENOID_NSLEEP, GPIO_IN);
	gpio_set_dir(PIN_SOLENOID_IN1, GPIO_IN);
	gpio_set_dir(PIN_SOLENOID_IN2, GPIO_IN);
	pusherEnabled = false;
#endif
}

#if HW_VERSION == 2
void enablePusher() {
#ifndef DISABLE_PUSHER
	PIO p = pio1;
	u32 sm = 1;
	u32 offset = pio_add_program(p, &pulse_program);
	pio_sm_config c = pulse_program_get_default_config(offset);
	gpio_set_function(PIN_SOLENOID_NSLEEP, GPIO_FUNC_PIO1);
	sm_config_set_out_pins(&c, PIN_SOLENOID_NSLEEP, 1);
	float div = F_CPU / pulse_CLOCK;
	sm_config_set_clkdiv(&c, div);
	pio_sm_init(p, sm, offset, &c);
	pio_sm_set_enabled(p, sm, true);
	while (pio_sm_get_pc(p, sm) != offset + 3) {
		tight_loop_contents();
	}
	gpio_set_function(PIN_SOLENOID_NSLEEP, GPIO_FUNC_SIO); // return control back to SIO without changing the pin state
	pio_remove_program(p, &pulse_program, offset);
#endif
	pusherEnabled = true;
}
#endif

void pusherLoop() {
#if HW_VERSION == 2
	u32 absTime = micros();
	solenoidCurrent = SOLENOID_CURR_CONV * adcConversions[CONV_RESULT_ISOLENOID];
	static i32 lastAccel = 0;
	u32 maxExtendTime = 460000 / (fix32(batVoltage) + fix32(0.5f)).geti32(); // e.g. 18.4 ms for 25V 6S, 28.8 ms for 16V 4S
	if (pusherState == 1) {
		u32 time = sinceExtend; // reduce micros() calls
		if (time >= maxExtendTime) {
			pusherFullyExtended = true;
		} else if (time >= maxExtendTime * 2 / 3) { // e.g. 12.3 ms for 25V 6S
			// detect extended solenoid by having 2 consecutive frames with high current
			// spikes OR one accelerometer spike frame. Ane accel spike frame has to be enough due to
			// vibrations, which cause the spikes to go negative too. And with bad luck, the
			// spikes might cancel each other out.

			// mA/µs = A/ms over the last two frames
			i32 deltaT = absTime - mic[1];
			if (deltaT < 200) return;
			fix32 dIdT = (solenoidCurrent - solenoidCurrentLast[1]) * 1000 / deltaT;

			// more than 0.8A/ms @ 6S increase, or accel spike
			if ((accelDataRaw[1] > 20000 && accelDataRaw[1] - lastAccel > 15000) || (dIdT > fix32(batVoltage) / 29)) {
				pusherFullyExtended = true;
			}
		}
	} else if (pusherState == 0) {
		u32 time = sinceRetract;
		if (time >= 20000) {
			pusherFullyRetracted = true;
		} else if (time >= 10000) {
			if (accelDataRaw[1] < -15000 && accelDataRaw[1] - lastAccel < -10000) {
				pusherFullyRetracted = true;
			}
		}
	}
	solenoidCurrentLast[1] = solenoidCurrentLast[0];
	solenoidCurrentLast[0] = solenoidCurrent;
	mic[1] = mic[0];
	mic[0] = absTime;
	lastAccel = accelDataRaw[1];
#endif
}

void onFireModeChange(MenuItem *_item) {
	mainMenu->search("burstCount")->setVisible(fireMode == FIRE_BURST);
	mainMenu->search("burstKeepFiring")->setVisible(fireMode == FIRE_BURST);
#if HW_VERSION == 1
	mainMenu->search("dartsPerSecond")->setVisible(fireMode != FIRE_SINGLE);
#elif HW_VERSION == 2
	showDpsOrLimit(nullptr);
#endif
	calcPushDurations(nullptr);
}

void calcPushDurations(MenuItem *_item) {
	i32 totalMicros = 1000000 / dartsPerSecond;
	pushDuration = minPushDuration * 1000;
	retractDuration = totalMicros - minPushDuration * 1000;
	if (retractDuration < minRetractDuration * 1000 || fireMode == FIRE_SINGLE)
		retractDuration = minRetractDuration * 1000;
}
void updateMaxDps(MenuItem *_item) {
	mainMenu->search("dartsPerSecond")->setMax(1000 / (minRetractDuration + minPushDuration));
	calcPushDurations(nullptr);
}
#if HW_VERSION == 2
void showDpsOrLimit(MenuItem *_item) {
	if (fireMode != FIRE_SINGLE && autoPusherTiming) {
		mainMenu->search("dpsLimit")->setVisible(true);
		mainMenu->search("dartsPerSecond")->setVisible(false);
	} else if (fireMode != FIRE_SINGLE) {
		mainMenu->search("dpsLimit")->setVisible(false);
		mainMenu->search("dartsPerSecond")->setVisible(true);
	} else {
		mainMenu->search("dpsLimit")->setVisible(false);
		mainMenu->search("dartsPerSecond")->setVisible(false);
	}
}
#endif

void extendPusher() {
#ifndef DISABLE_PUSHER
#if HW_VERSION == 1
	gpio_put(PIN_SOLENOID, 1);
#elif HW_VERSION == 2
	if (!pusherEnabled) return;
	gpio_put(PIN_SOLENOID_IN1, 0);
	gpio_put(PIN_SOLENOID_IN2, 1);
#endif
#endif
#if HW_VERSION == 2
	pusherFullyRetracted = false;
	if (pusherState != 1)
		sinceExtend = 0;
#endif
#ifdef USE_BLACKBOX
	if (pusherState != 1)
		blSetFired();
#endif
	pusherState = 1;
}

void dischargePusher() {
#ifndef DISABLE_PUSHER
#if HW_VERSION == 1
	gpio_put(PIN_SOLENOID, 0);
#elif HW_VERSION == 2
	if (!pusherEnabled) return;
	gpio_put(PIN_SOLENOID_IN2, 0);
	gpio_put(PIN_SOLENOID_IN1, 1);
	pusherFullyExtended = false;
	if (pusherState != 2)
		sinceRetract = 0;
#endif
#endif
	pusherState = 2;
}

void retractPusher() {
#ifndef DISABLE_PUSHER
#if HW_VERSION == 1
	gpio_put(PIN_SOLENOID, 0);
#elif HW_VERSION == 2
	if (!pusherEnabled) return;
	gpio_put(PIN_SOLENOID_IN1, 0);
	gpio_put(PIN_SOLENOID_IN2, 0);
	pusherFullyExtended = false;
	if (pusherState != 0 && pusherState != 2)
		sinceRetract = 0;
#endif
#endif
	pusherState = 0;
}
