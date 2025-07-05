#include "global.h"

u8 triggerState = 0;
volatile bool triggerUpdateFlag = false;
#if HW_VERSION == 1
volatile u32 triggerCounter = 0;
volatile u8 triggerAlarm = 0;
volatile u32 triggerAlarmTime = 0;
PT1 triggerFilter(100, 10000);
const fix32 TRIGGER_THRESHOLD_HIGH = 0.8;
const fix32 TRIGGER_THRESHOLD_LOW = 0.1;

void triggerIsr() {
	timer_hw->intr = 1 << triggerAlarm;
	triggerCounter++;
	triggerFilter.update(gpio_get(PIN_TRIGGER));
	triggerAlarmTime += 100;
	if (triggerAlarmTime < timer_hw->timerawl + 50) {
		triggerAlarmTime = timer_hw->timerawl + 100;
	}
	timer_hw->alarm[triggerAlarm] = triggerAlarmTime;
}
#endif

void triggerInit() {
	gpio_init(PIN_TRIGGER);
	gpio_set_dir(PIN_TRIGGER, GPIO_IN);
	gpio_set_pulls(PIN_TRIGGER, true, false);
	sleep_us(10); // wait for settling after applying pullup
	u8 read = gpio_get(PIN_TRIGGER);
	triggerState = !read;

#if HW_VERSION == 1
	triggerFilter.set(read);

	// set up timer to notify every 100us
	triggerAlarm = hardware_alarm_claim_unused(true);
	irq_set_exclusive_handler(triggerAlarm + TIMER_IRQ_0, triggerIsr);
	irq_set_enabled(triggerAlarm + TIMER_IRQ_0, true);
	timer_hw->inte |= 1 << triggerAlarm;
	triggerAlarmTime = timer_hw->timerawl + 100;
	timer_hw->alarm[triggerAlarm] = triggerAlarmTime;
#endif

	// go to STATE_BOOT_MENU to select boot mode
	if (triggerState && bootReason == BootReason::POR)
		operationState = STATE_BOOT_SELECT;
}

void triggerLoop() {
#if HW_VERSION == 1
	static elapsedMillis x = 0;
	if (x >= 2) {
		x = 0;
		if (!triggerCounter) {
			// interrupt missed, disabled or similar, trigger manually
			triggerIsr();
			DEBUG_PRINTSLN("trigger missed");
		}
		triggerCounter = 0;
	}

	if (!triggerState && (fix32)triggerFilter < TRIGGER_THRESHOLD_LOW)
#elif HW_VERSION == 2
	if (!triggerState && !gpio_get(PIN_TRIGGER))
#endif
	{
		triggerUpdateFlag = false;
		triggerState = 1;
		triggerUpdateFlag = true;
		Serial.println("trigger pressed");
		inactivityTimer = 0;
#if HW_VERSION == 1
	} else if (triggerState && (fix32)triggerFilter > TRIGGER_THRESHOLD_HIGH) {
#elif HW_VERSION == 2
	} else if (triggerState && gpio_get(PIN_TRIGGER)) {
#endif
		triggerUpdateFlag = false;
		triggerState = 0;
		triggerUpdateFlag = true;
		Serial.println("trigger released");
		inactivityTimer = 0;
	}
}