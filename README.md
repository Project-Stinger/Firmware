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

## ML IMU Data Logging (V2 only)

A compile-time feature for recording raw IMU and trigger data to onboard flash, intended for collecting ML training datasets during normal blaster use.

### Overview

When built with the `v2_ml` environment, the firmware continuously captures 6-axis IMU data (accelerometer + gyroscope) and trigger state at 100 Hz, storing it in a binary log on the Pico's onboard flash via LittleFS. Recording starts automatically on boot and runs in the background while the blaster operates normally.

### How it works

- **Core 1** (real-time, 3200 Hz): decimates the 1600 Hz gyro data to 100 Hz and pushes samples into a lock-free SPSC ring buffer in RAM (~4096 samples, ~41 seconds)
- **Core 0** (non-critical): flushes the RAM buffer to flash only when the blaster is not actively firing (not in RAMPUP, PUSH, RETRACT, or RAMPDOWN states). This avoids flash write stalls that would disrupt the PID control loop
- Flushing is triggered by a shot ending or the buffer reaching 75% capacity
- If flash fills up, recording stops automatically with an audible warning
- If the RAM buffer fills during a long continuous firing burst (>41s), excess samples are silently dropped

### Data format

Each sample is a 17-byte packed struct:

| Field          | Type | Description                              |
|----------------|------|------------------------------------------|
| `timestamp_ms` | u32  | `millis()` at capture time               |
| `ax, ay, az`   | i16  | Raw accelerometer values (post-calibration) |
| `gx, gy, gz`   | i16  | Raw gyroscope values (post-calibration)  |
| `trigger`      | u8   | Trigger state (0 or 1)                   |

### Building

```bash
pio run -e v2_ml
```

This uses a dedicated PlatformIO environment (`env:v2_ml`) that extends `env:v2` with `-DUSE_ML_LOG` and allocates 1.5 MB of flash for the LittleFS filesystem. The normal `env:v2` build is unaffected.

### Retrieving data

After a recording session, connect the Pico to a PC via USB. The firmware exposes the log file as a USB mass storage device (using `SingleFileDrive`). Copy `ml_log.bin` from the drive, then convert it:

```bash
python python/ml_log_convert.py ml_log.bin
# produces ml_log.csv with columns: timestamp_ms,ax,ay,az,gx,gy,gz,trigger
```

The CSV can be loaded directly with pandas:

```python
import pandas as pd
df = pd.read_csv("ml_log.csv")
```

### Capacity

At 17 bytes/sample and 100 Hz (~1.7 KB/s), 1.5 MB of flash stores approximately 13-15 minutes of recording, accounting for LittleFS overhead. Each session overwrites the previous one.

### Code changes

All ML logging code is behind the `USE_ML_LOG` compile-time flag and only applies to V2 hardware.

**New files:**

| File | Purpose |
|------|---------|
| `src/mlLog.h` | `MlSample` struct definition, function declarations |
| `src/mlLog.cpp` | SPSC ring buffer, LittleFS logging, flush-on-idle logic, SingleFileDrive USB exposure |
| `python/ml_log_convert.py` | Converts binary log to CSV |

**Modified files:**

| File | Change |
|------|--------|
| `platformio.ini` | Added `[env:v2_ml]` build environment with `-DUSE_ML_LOG` and `board_build.filesystem_size = 1.5m` |
| `src/global.h` | Added `#include "mlLog.h"` |
| `src/main.cpp` | Added `mlLogInit()` in `setup()`, `mlLogSlowLoop()` in `loop()` (Core 0), `mlLogLoop()` in `loop1()` gyro cycle (Core 1) — all behind `#ifdef USE_ML_LOG` |

### Design decisions

- **RAM buffering with deferred flush**: flash writes stall the RP2040's XIP (execute-in-place) and can pause the other core for 3-5ms. By buffering in RAM and only flushing when motors are off, the 3200 Hz PID control loop is never disrupted during active firing
- **SPSC ring buffer**: uses GCC `__atomic` builtins with acquire/release semantics for safe cross-core communication without mutexes. Power-of-two capacity allows bitmask indexing
- **Separate build environment**: the 80 KB static RAM buffer and 1.5 MB filesystem reservation only apply when `USE_ML_LOG` is defined, keeping the standard firmware build unaffected
- **SingleFileDrive**: exposes the binary log as a USB mass storage device for drag-and-drop file retrieval without any special software on the host PC

## Missing something?

This project is still in the early stages of making the source available. If you feel like something is missing in order to make this repo accessible, please open an issue or ask in the discussion forum.

## License

This project is licensed under the Polyform non-commercial license. See the [LICENSE](https://github.com/The-Stinger/stinger-firmware?tab=License-1-ov-file) file for details.
