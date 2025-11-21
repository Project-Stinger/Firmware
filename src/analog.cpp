#include "global.h"
#if HW_VERSION == 1
volatile u32 adcConversions[3];
#elif HW_VERSION == 2
volatile u32 adcConversions[5];
#endif

void initAnalog() {
	adc_init();
#if HW_VERSION == 1
	adc_gpio_init(PIN_JOYSTICK_Y);
	adc_gpio_init(PIN_JOYSTICK_X);
	adc_gpio_init(PIN_VBAT);
#elif HW_VERSION == 2
	adc_gpio_init(PIN_SOLENOID_CURR);
	adc_gpio_init(PIN_VBAT);
	adc_gpio_init(PIN_IBAT);
	adc_gpio_init(PIN_JOYSTICK_COMBINED);
	gpio_init(PIN_ANALOG_SELECT);
	gpio_set_dir(PIN_ANALOG_SELECT, GPIO_OUT);
#endif

	adc_fifo_setup(true, false, 0, false, false);
	adc_set_round_robin(0b1111);
	adc_select_input(0);
	adc_run(true);
}

void analogLoop() {
	// the following lines have been superseded by adc_run(false) in loop1()
	// adc_run(false);
	// sleep_us(3);
#if HW_VERSION == 1
	if (adc_fifo_get_level() >= 3) {
		adcConversions[CONV_RESULT_JOYSTICK_Y] = adc_fifo_get();
		adcConversions[CONV_RESULT_JOYSTICK_X] = adc_fifo_get();
		adcConversions[CONV_RESULT_VBAT] = adc_fifo_get();
	}
#elif HW_VERSION == 2
	static bool select = false;
	if (adc_fifo_get_level() >= 4) {
		adcConversions[CONV_RESULT_ISOLENOID] = adc_fifo_get();
		adcConversions[CONV_RESULT_VBAT] = adc_fifo_get();
		adcConversions[CONV_RESULT_IBAT] = adc_fifo_get();
		if (select)
			adcConversions[CONV_RESULT_JOYSTICK_Y] = adc_fifo_get();
		else
			adcConversions[CONV_RESULT_JOYSTICK_X] = adc_fifo_get();
	}
	select = !select;
	gpio_put(PIN_ANALOG_SELECT, select);
#endif
	adc_fifo_drain();
	adc_select_input(0);
	adc_run(true);
}