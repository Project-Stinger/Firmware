#if HW_VERSION == 1
// extern u8 escPins[4]; // esc.cpp, esc.h
#define PIN_SOLENOID 15

// ST7735 display SPI pins
#define PIN_TFT_RST 1
#define PIN_TFT_SCK 2
#define PIN_TFT_MOSI 3
#define PIN_TFT_CS 6
#define PIN_TFT_DC 7

#define PIN_TRIGGER 0

#define PIN_SDA 4
#define PIN_SCL 5

// Analog joystick inputs and battery voltage
#define PIN_JOYSTICK_X 27 // Analog input pin for joystick 1
#define PIN_JOYSTICK_Y 26 // Analog input pin for joystick 2
#define PIN_VBAT 28 // Analog input pin for battery voltage

#define PIN_MOTOR_BASE 9

#elif HW_VERSION == 2
// Solenoid
#define PIN_SOLENOID_NSLEEP 23
#define PIN_SOLENOID_IN1 22
#define PIN_SOLENOID_IN2 20
#define PIN_SOLENOID_CURR 26

// Display
#define PIN_TFT_RST 16
#define PIN_TFT_SCK 10 // SPI1
#define PIN_TFT_MOSI 11 // SPI1
#define PIN_TFT_CS 9
#define PIN_TFT_DC 8

// Gyro
#define PIN_GYRO_INT 17
#define PIN_GYRO_CS 5
#define PIN_GYRO_SCK 6 // SPI0
#define PIN_GYRO_MISO 4 // SPI0
#define PIN_GYRO_MOSI 7 // SPI0

// TOF
#define PIN_SDA 2 // I2C1
#define PIN_SCL 3 // I2C1

// Trigger
#define PIN_TRIGGER 25

// joystick and battery
#define PIN_ANALOG_SELECT 21
#define PIN_JOYSTICK_COMBINED 29 // select X/Y with PIN_ANALOG_SELECT
#define PIN_VBAT 27
#define PIN_IBAT 28

// motor pins 12-15
#define PIN_MOTOR_BASE 12

// misc
#define PIN_EXPANSION_TX_SDA 0 // I2C0, UART0
#define PIN_EXPANSION_RX_SCL 1 // I2C0, UART0
#define PIN_STANDBY 24
#define PIN_LED_STRIP 19
#define PIN_SPEAKER 18

#endif