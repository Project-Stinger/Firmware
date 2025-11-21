# Stinger Firmware

This repository contains the firmware for the Stinger foam dart blaster. Core features include:

-   PID controller for flywheel RPM
-   Bidirectional DShot using the RP2040's PIO
-   Closed loop solenoid control
-   Multiple firing modes
-   Menu system controlled by a joystick and a screen
-   IMU for motion sensing (safety features, automatic idle, ...)
-   Speaker and LED output
-   Tournament mode

## Documentation

Documentation for using the firmware is provided in the [Stinger-Docs wiki](https://github.com/bastian2001/Stinger-Docs/wiki).

## Contributing

There are several ways in which you can contribute to this firmware

-   **Bug reports**: If you find a bug, please [open an issue](https://github.com/The-Stinger/stinger-firmware/issues/new) on GitHub.
-   **Feature requests**: If you have an idea for a new feature, please [open an issue](https://github.com/The-Stinger/stinger-firmware/issues/new)
-   **Pull requests**: If you want to contribute code, please fork the repository and submit a pull request. Please file an issue first to discuss the changes you want to make. This helps to avoid duplicate work and ensures that your changes are in line with the project's goals.

## Building the firmware

The firmware is written in C++ and uses the [PlatformIO](https://platformio.org/) build system. It is recommended to use the [VSCode extension](https://marketplace.visualstudio.com/items?itemName=platformio.platformio-ide) for PlatformIO, but you can also use the command line interface. To select your target (V1 or V2), press `Ctrl+Shift+P` (View -> Command Palette) in VSCode, search for "PlatformIO: Pick project environment" and select either v1 or v2. You can then build and upload the firmware using the buttons in the bottom bar of VSCode (you can also assign shortcuts).

You may find some options from the `platformio.ini` file useful for debugging and development, such as the blackbox or debug print statements.

## Missing something?

This project is still in the early stages of making the source available. If you feel like something is missing in order to make this repo accessible, please open an issue or ask in the discussion forum.

## License

This project is licensed under the Polyform non-commercial license. See the [LICENSE](https://github.com/The-Stinger/stinger-firmware?tab=License-1-ov-file) file for details.
