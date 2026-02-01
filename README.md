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

The V2 firmware can capture 6-axis IMU data (accelerometer + gyroscope) and trigger state at 100 Hz, storing it in a binary log on the Pico's onboard flash via LittleFS. Recording is activated from the menu: Menu → Device → ML Recording. A red `R0%`–`R100%` indicator on the home screen shows recording status and flash usage. The blaster operates normally while recording runs in the background.

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

The ML logging code is included in the standard `env:v2` build. The `v2` environment allocates 1.5 MB of flash for the LittleFS filesystem used by the logger.

### Retrieving data

After a recording session, connect the Pico to a PC via USB and pull the log over USB serial using the helper script:

```bash
python python/ml_log_pull.py -o ml_log.bin
python python/ml_log_convert.py ml_log.bin -o ml_log.csv
```

The CSV can be loaded directly with pandas:

```python
import pandas as pd
df = pd.read_csv("ml_log.csv")
```

For quick exploration (plots + shot crops; requires `matplotlib`):

```bash
python python/ml_log_explore.py ml_log.csv --outdir ml_out --clean
```

This writes:
- `ml_out/dataset.png` (all 6 axes over time + trigger pulls marked; shots within 1s of a previous shot are marked as rejected)
- `ml_out/shots/` (per-shot cropped plots with more pre-shot context and only 200ms after the shot)

To export a ready-to-train labeled window dataset (raw sequences):

```bash
python python/ml_log_explore.py ml_log.csv --outdir ml_out --clean --export-dataset
```

This writes `ml_out/train/` containing:
- `windows_i16le.bin` (shape `[N, window_samples, 6]`, int16, channels `[ax,ay,az,gx,gy,gz]`)
- `labels_u8.bin` (shape `[N]`, 0/1)
- `index.csv` and `meta.json`

### Capacity

At 17 bytes/sample and 100 Hz (~1.7 KB/s), 1.5 MB of flash stores approximately 13-15 minutes of recording, accounting for LittleFS overhead. Each session overwrites the previous one.

### Code changes

All ML logging code is V2-only (guarded by `#if HW_VERSION == 2`).

**New files:**

| File | Purpose |
|------|---------|
| `src/mlLog.h` | `MlSample` struct definition, function declarations |
| `src/mlLog.cpp` | SPSC ring buffer, LittleFS logging, flush-on-idle logic, USB serial export (`MLDUMP`) |
| `src/mlInfer.h` | Inference API: predict, cached prob, sliding window push |
| `src/mlInfer.cpp` | Seqlock window, featurization, LogReg and MLP forward pass |
| `src/mlWeights.h` | Auto-generated model weights (scaler params, coefficients, MLP layers) |
| `python/ml_log_convert.py` | Converts binary log to CSV |
| `python/ml_log_pull.py` | Pulls the binary log over USB serial |
| `python/ml_log_explore.py` | Exploration plots + per-shot crops + dataset export |

**Modified files:**

| File | Change |
|------|--------|
| `platformio.ini` | Added `-DUSE_ML_LOG` and `board_build.filesystem_size = 1.5m` to `env:v2` |
| `src/global.h` | Added `#include "mlLog.h"` and `#include "mlInfer.h"` |
| `src/main.cpp` | Added init/loop calls for both mlLog and mlInfer on the appropriate cores |
| `src/menu/menu.cpp` | Added ML Recording action, ML idle mode and threshold settings |
| `src/drivers/display.cpp` | Added recording (red R%) and inference confidence (cyan %) HUD indicators |
| `src/operationSm.cpp/.h` | ML idle logic: envelope follower, binary/dynamic modes, pitch cancel, hysteresis |
| `src/eepromImpl.h` | EEPROM slots for ML idle mode and threshold |
| `src/pid.cpp` | Min throttle default lowered from 40 to 14 |

### Design decisions

- **RAM buffering with deferred flush**: flash writes stall the RP2040's XIP (execute-in-place) and can pause the other core for 3-5ms. By buffering in RAM and only flushing when motors are off, the 3200 Hz PID control loop is never disrupted during active firing
- **SPSC ring buffer**: uses GCC `__atomic` builtins with acquire/release semantics for safe cross-core communication without mutexes. Power-of-two capacity allows bitmask indexing
- **Menu activation**: recording only starts when explicitly selected from Device → ML Recording; the 80 KB RAM buffer and filesystem are allocated but idle until activated

## ML Inference (Idle Pre-Spin)

### Overview

The firmware includes on-device ML inference to predict when a shot is about to happen. When the model is confident a shot is imminent, the flywheels pre-spin to reduce ramp-up latency. Two models are available: a logistic regression (fast, lightweight) and a 2-layer MLP (more accurate, heavier).

### Menu options

Under **Menu → Motor → Idling**, the idle mode selector now includes two ML options alongside the existing angle-based modes:

| Setting | Description |
|---------|-------------|
| `ML:LR` | Logistic regression on 18 summary features (mean, std, absmax per axis) |
| `ML:MLP` | 2-layer MLP (30→64→32→1) on 30 rich features including magnitude stats |

When an ML idle mode is selected, two additional settings appear:

| Setting | Description |
|---------|-------------|
| **ML Idle Mode** | `Binary` (on/off with hysteresis) or `Dynamic` (RPM scales with probability) |
| **ML Threshold %** | Probability threshold for spin-up (5–95%, default 50%). Lower = more aggressive |

A cyan percentage indicator on the home screen shows the model's current confidence in real-time.

### Architecture

- **Core 1** (3200 Hz PID loop): pushes decimated IMU samples (100 Hz) into a 50-sample sliding window via `mlInferLoop()`. Reads the cached probability via `mlInferGetCachedProb()` — a single atomic load, no float math.
- **Core 0** (slow loop): runs the full inference pipeline in `mlInferSlowLoop()` at up to 100 Hz — copies the window (seqlock-protected to prevent torn reads), computes features, runs the forward pass, and caches the result via atomic store.
- **Mutual exclusion with logging**: when ML Recording is active, inference is automatically disabled to keep flash flush windows predictable and avoid competing for CPU time on Core 0.

### Model weights

Weights are stored in `src/mlWeights.h` (auto-generated by `ml_training/export_weights.py`). The logistic regression uses 18 floats for coefficients + scaler params. The MLP uses ~7K floats total (~28 KB in flash). All weights are `const` and live in flash, not RAM.

## Missing something?

This project is still in the early stages of making the source available. If you feel like something is missing in order to make this repo accessible, please open an issue or ask in the discussion forum.

## License

This project is licensed under the Polyform non-commercial license. See the [LICENSE](https://github.com/The-Stinger/stinger-firmware?tab=License-1-ov-file) file for details.
