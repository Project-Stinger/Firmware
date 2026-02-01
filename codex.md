# Codex Notes — Stinger Firmware (RP2040 / PlatformIO)

This repo contains the firmware for the **Stinger** foam dart blaster. The code targets the **Raspberry Pi Pico (RP2040)** using **Arduino-Pico (earlephilhower core)** via PlatformIO.

This file is intentionally “internal-facing”: it’s meant to help you (or future contributors) quickly re-orient in the codebase and understand the main runtime architecture.

## Build + environments

- Build system: PlatformIO (`platformio.ini`).
- Two hardware variants:
  - `env:v1` ⇒ `-DHW_VERSION=1`
  - `env:v2` ⇒ `-DHW_VERSION=2` (adds NeoPixel lib and some extra peripherals)
- Board / framework:
  - `board = pico`
  - `framework = arduino`
  - `board_build.core = earlephilhower`
  - `board_build.f_cpu = 132000000L`
- A pre-build script writes the current git hash to `include/git_version.h`:
  - `extra_scripts = pre:python/gitVersion.py`

### Useful compile-time flags

These are set via `platformio.ini` `build_flags` (some are currently commented out there):

- `PRINT_DEBUG`: enables `DEBUG_PRINT*` macros (adds `__FILE__/__LINE__` prefixes).
- `USE_BLACKBOX`: records an in-memory “blackbox” of RPM + PID terms; dump over Serial by sending any byte.
- `DISABLE_PUSHER`: disables solenoid/pusher actuation (useful to avoid noise while debugging).
- `ENABLE_DEBUG_GYRO` / `ENABLE_GYRO_BB`: extra IMU debug modes (V2).
- `USE_TOF`: enables time-of-flight magazine detection code path (enabled in `platformio.ini`).

## Directory map (what lives where)

- `src/main.cpp`: entrypoint; dual-core scheduling and high/low priority loops.
- `src/operationSm.cpp` / `src/operationSm.h`: the **operation state machine** (rev/fire/ramp logic).
- (V2) `src/mlLog.cpp` / `src/mlLog.h`: IMU + trigger logging to LittleFS + serial export + model upload protocol.
- (V2) `src/mlInfer.cpp` / `src/mlInfer.h`: on-device inference (LR + MLP), cached probability, user model load.
- (V2) `src/usbSession.cpp` / `src/usbSession.h`: core-0 USB state sampler (mounted/DTR) exposed as atomics for core 1.
- `src/pid.cpp` / `src/pid.h`: flywheel RPM PID loop and (optional) blackbox capture.
- `src/drivers/*`: hardware drivers (ESC, display, joystick, trigger, battery, ToF, gyro, speaker, LEDs, SPI helpers).
- `src/menu/*`: menu tree + `MenuItem` system; settings persistence (EEPROM-backed).
- `src/eepromImpl.*`: EEPROM layout, first-boot defaults, migrations.
- `src/analog.*`: ADC sampling, round-robin conversion results.
- `src/pusher.*`: pusher/solenoid control (closed-loop on V2 using current/accel cues).
- `src/standby.*`: V2 standby mode (clock scaling + wake sources).
- `src/tournament.*`: tournament mode rules and UI behavior.
- `src/utils/*`: fixed-point math, PT filters, ring buffer, quaternion helpers.
- `src/pio/*`: PIO programs used by the firmware (currently a solenoid “pulse” program for V2).

## High-level runtime architecture

The firmware is split into:

- **Core 0 (slow / non-critical)**: UI and periodic peripherals.
- **Core 1 (fast / time-critical)**: control loop at a fixed rate; ESC comm, trigger, state machine.

This separation is visible in `src/main.cpp`:

- `setup()` runs on core 0
- `loop()` runs on core 0 (non time critical)
- `setup1()` runs on core 1
- `loop1()` runs on core 1 (time critical)

### Loop rates and why they matter

`PID_RATE` is defined in `src/global.h` as:

- `#define PID_RATE 3200`

Core 1 runs a “tick” roughly every `1e6 / PID_RATE` microseconds and calls the time-critical subsystem functions. If you add work to `loop1()` (or to anything it calls), you are trading away real-time margin.

### V2 ML cross-core split (why it exists)

RP2040 is Cortex-M0+ (no HW float), and inference/featurization is expensive. The ML design keeps timing deterministic:

- **Core 1**: samples IMU, does cheap decimation, pushes into ring buffers / windows.
- **Core 0**: performs flash I/O (LittleFS flush/export) and runs the heavy inference (`mlInferSlowLoop`) and caches a probability value.

Core 1 should never call TinyUSB internals directly; `usbSessionLoop0()` samples USB mounted/DTR on core 0 and publishes simple flags.

## Boot + initialization flow

### Boot reason tracking

`src/main.cpp` uses `powerOnResetMagicNumber` + `rebootReason` in uninitialized RAM to distinguish:

- Power-on reset (`BootReason::POR`)
- Watchdog / intentional reboots (`BootReason::*`)

Key cases:

- `TO_ESC_PASSTHROUGH`: boot into an ESC configuration passthrough mode (see “ESC passthrough” below).
- Holding trigger on POR enters the boot selection UI:
  - `triggerInit()` sets `operationState = STATE_BOOT_SELECT` if trigger is pressed and boot reason is POR.

### Core 0 `setup()` (simplified)

1. Serial init, debug banner.
2. V2-only early init: standby switch, speaker, LEDs.
3. ADC init (`initAnalog()`).
4. Optional ESC passthrough mode (blocks here until exit).
5. EEPROM init / migration (`eepromInit()`).
6. Optional ToF init (`initTof()` when `USE_TOF`).
7. `initDisplay()` then `initESCs()` (ordering matters due to SPI pin reservation).
8. Battery + tournament + (V2) IMU/gyro init.
9. Wait for core 1 to finish its setup phase (`setupDone` barrier).

### Core 1 `setup1()` (simplified)

1. Waits until core 0 signals it’s safe to proceed (`setup1CanStart`).
2. Initializes trigger (`triggerInit()`).
3. Initializes pusher hardware (`initPusher()`).
4. Signals setup completion (`setupDone` barrier).

## The “fast loop” (core 1) dataflow

`loop1()` does:

1. `triggerLoop()` — edge detection / debouncing + `triggerUpdateFlag`.
2. On each PID tick:
   - (V2 standby guard) `standbyOnLoop()` if in standby mode.
   - `decodeErpm()` — read bidirectional DShot telemetry (RPM/temp/voltage/current/status).
   - `checkTelemetry()` — set/clear disable flags based on health.
   - `runOperationSm()` — **main state machine** (rev/fire/ramp/menu gating).
   - `analogLoop()` — fetch ADC conversion results (VBAT, joystick, currents).
   - `joystickLoop()` — gesture detection and rotation tick tracking.
   - (V2) `pusherLoop()`, `batCurrLoop()`, `gyroLoop()/freeFallDetection()/updateAtti*()`, standby off logic, optional speaker loop.

### The “slow loop” (core 0)

`loop()` is intentionally less time-sensitive and handles:

- ToF updates (`tofLoop()` when `USE_TOF`)
- battery UI logic (`batLoop()`)
- display redraws (`displayLoop()`)
- menu updates (`openedMenu->loop()` when in `STATE_MENU`)
- tournament UI logic (`tournamentLoop()`)
- (V2) LED + speaker loop if not running on fast core

On V2, core 0 also runs:

- `usbSessionLoop0()` (samples USB mounted/DTR and publishes `usbCdcActive()`)
- `mlLogSlowLoop()` (serial protocol + deferred LittleFS flush)
- `mlInferSlowLoop()` (ML inference, cached probability updates)

## Operation state machine (`runOperationSm`)

The core behavior is governed by `operationState` (`src/operationSm.h`):

- `STATE_SETUP`: ESC discovery + telemetry readiness gating; enables EDT repeatedly.
- `STATE_SAFE`: locked state; requires an “unlock” gesture.
- `STATE_OFF`: idle state; watches trigger and joystick gestures.
- `STATE_PROFILE_SELECT`: joystick-based profile selection while keeping motors idle/off.
- `STATE_RAMPUP`: PID to target RPM; waits for RPM-in-range or timeout.
- `STATE_PUSH`: pusher extend; decrements dart count when complete.
- `STATE_RETRACT`: retract timing; implements semi/burst/auto refire logic.
- `STATE_RAMPDOWN`: ramps target RPM toward idle/off; supports “rev-after-fire”.
- `STATE_MENU` / `STATE_OPEN_MENU`: menu mode; motors off or overridden for testing.
- `STATE_JOYSTICK_CAL`: interactive joystick calibration.
- (V2) `STATE_FALL_DETECTED`: safety lockout after free-fall detection.
- `STATE_BOOT_SELECT`: boot selection carousel UI (trigger to select).
- (V2) `STATE_USB`: USB service mode (safe state for MLDUMP / model upload). Entered when `usbCdcActive()` is true.

### Safety and “disable” flags

`motorDisableFlags` can force motor outputs to zero regardless of state:

- `MD_ESC_OVERTEMP`: ESC temp exceeds threshold.
- `MD_NO_TELEMETRY`: no bidir telemetry frames for too long.
- `MD_BATTERY_EMPTY`: below shutdown threshold.
- `MD_MOTORS_BLOCKED`: stall detection (RPM stuck at 0 with throttle applied).

When any disable flag is active, `runOperationSm()` will:

- re-send “EDT enable” when needed
- clamp outputs to 0 and still call `sendThrottles()`

## ESC control (bidirectional DShot)

The ESC path lives in `src/drivers/esc.cpp`:

- 4 ESC instances created as `BidirDShotX1(pin, 600, pio0, index)`.
- All access is guarded by a `mutex_t escMutex` (core 0 and core 1 both touch ESCs).
- Telemetry decoding (`decodeErpm()`) reads one packet per motor per tick.
- “Extended telemetry” enable is sent as a DShot command periodically during setup and whenever motors stop.
- A per-motor ringbuffer `escCommandBuffer[]` allows queued DShot commands to be injected instead of throttle.

### ESC passthrough mode

If `rebootReason` is set to `BootReason::TO_ESC_PASSTHROUGH`, the firmware boots into a blocking loop that:

- initializes display + trigger
- runs `beginPassthrough(...)` / `processPassthrough()`
- exits on disconnect or a 3s trigger hold
- reboots back to normal mode

## Inputs and sensors

### Trigger

`src/drivers/trigger.cpp`:

- V1: uses a hardware timer alarm at 100µs to sample and low-pass filter the GPIO, with hysteresis thresholds.
- V2: reads the GPIO directly.
- Updates:
  - `triggerState` (pressed/not pressed)
  - `triggerUpdateFlag` (edge event happened this tick)

### Joystick gestures

`src/drivers/joystick.cpp`:

- Converts ADC readings into a calibrated `(x,y)` in `[-100,100]`.
- Emits `Gesture` events: `GESTURE_PRESS`, `GESTURE_HOLD`, `GESTURE_RELEASE`.
- Tracks rotation via `joystickRotationTicks` (used e.g. for adjusting dart count).
- Used for:
  - profile selection / menu navigation
  - safe/lock actions in `STATE_OFF`

### ToF magazine detection (`USE_TOF`)

`src/drivers/tof.cpp` (VL53L0X):

- Maintains `magPresent` and `dartCount`.
- Thresholds are EEPROM-backed and calibratable via a menu flow.
- When no sensor is detected at boot, `foundTof=false` and the feature silently disables.

### IMU / gyro (V2)

V2 adds a BMI270 IMU (`src/drivers/gyro.cpp`, `src/imu.cpp`):

- calibration runs early and expects quiet motion for good samples
- used for:
  - free-fall detection
  - orientation gating (idle enable threshold, max fire angle limit)
  - standby wake behavior (motion interrupt pin)

## Pusher / solenoid

`src/pusher.cpp`:

- V1: simple solenoid GPIO on/off timing.
- V2: H-bridge-style driver with:
  - `NSLEEP`, `IN1`, `IN2`
  - current sensing via ADC (`CONV_RESULT_ISOLENOID`)
  - closed-loop “auto timing”:
    - detects full extension via current slope and/or accelerometer impulse
    - detects retraction via accelerometer impulse / time

This logic enables “darts per second limit” behavior in V2 when `autoPusherTiming` is enabled.

## Battery / power model

`src/drivers/bat.cpp`:

- VBAT is sampled via ADC, median-ish filtered, then smoothed.
- Determines `powerSource`:
  - `1` = battery
  - `2` = USB (low VBAT)
- Provides warning + shutdown thresholds (per-cell), EEPROM-backed.
- Enforces motor cutoff by setting `MD_BATTERY_EMPTY` below the shutdown threshold.
- (V2) estimates current draw and mAh used in `batCurrLoop()`.

## Display and menu system

`src/drivers/display.cpp`:

- V1 uses ST7735 (160×80); V2 uses ST7789 (240×135).
- Provides text layout helpers like `printCentered(...)` and various UI screens.

`src/menu/*`:

- The menu is a tree of `MenuItem` objects.
- Menu items are EEPROM-backed variables or actions with callbacks:
  - `onEnter`, `onExit`, `customLoop`, and direction handlers.
- `saveAndClose(...)` writes menu values to EEPROM and can reboot if a setting requires it.

## Settings persistence (EEPROM)

`src/eepromImpl.h` defines:

- profile slots (up to `MAX_PROFILE_COUNT`, 400 bytes each)
- “general” settings at fixed offsets (names, thresholds, calibration values)
- versioned migrations so new firmware can upgrade old EEPROM layouts

On first boot, default profiles are written (see `adjustProfileDefaults()` in `src/eepromImpl.cpp`).

## Concurrency notes (things to be careful about)

- Core 0 and core 1 run simultaneously; treat globals as shared state.
- ESC objects and command buffers are protected by `escMutex`.
- Some UI updates are offloaded across cores using `rp2040.fifo` (e.g. LED programs in V2).
- Many variables are `volatile` to avoid optimization issues, but `volatile` is not a full concurrency primitive.

## “Where do I start?” (reading order)

If you want to understand behavior end-to-end:

1. `src/main.cpp` — what runs on which core, and at what cadence.
2. `src/operationSm.cpp` — state machine logic and transitions.
3. `src/pid.cpp` — how throttles are computed from RPM telemetry.
4. `src/drivers/esc.cpp` — how DShot + telemetry are sent/received.
5. `src/pusher.cpp` + `src/drivers/tof.cpp` — firing mechanics and dart accounting.
6. `src/menu/menu.cpp` + `src/eepromImpl.cpp` — settings model and persistence.
