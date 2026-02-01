# ML Features (V2 only)

This document describes the **V2-only** ML stack:

- **ML Logging**: record 6-axis IMU + trigger at **100 Hz** to LittleFS on flash.
- **ML Inference**: run **Logistic Regression (LR)** and **MLP (64,32)** on-device to predict “about to shoot”.
- **Personalization workflow**: user records a single session, plugs in USB, browser pulls the log, trains LR+MLP, and uploads weights back **without reflashing UF2**.

All ML code is guarded by `#if HW_VERSION == 2` and built in `env:v2`.

## Quick start (recommended flow)

1) Flash the latest V2 firmware.
2) On the blaster: **Menu → Device → ML Recording → Start**.
3) Play normally for a few minutes. Include some **negative time**: aim/move without shooting.
4) Stop recording: **Menu → Device → ML Recording → Stop**.
5) Plug in USB, run the local web UI, click **I’m ready**, and follow the steps to train + upload models.

See **Web UI (local MVP)** below.

## Build / flash (V2)

Build:
```sh
pio run -e v2
```

UF2 output:
- `.pio/build/v2/firmware.uf2`
- (convenience copy) `stinger-v2-latest.uf2` at repo root

If BOOTSEL mass storage doesn’t mount on macOS, you can still flash UF2 with `picotool` (PlatformIO installs it):
```sh
/Users/stan/.platformio/packages/tool-picotool-rp2040-earlephilhower/picotool load -t uf2 stinger-v2-latest.uf2
/Users/stan/.platformio/packages/tool-picotool-rp2040-earlephilhower/picotool reboot
```

## ML Logging

### What is recorded

- Sampling rate: **100 Hz**
- Channels: `ax, ay, az, gx, gy, gz` (raw i16)
- Trigger: `trigger` (0/1)
- Timestamp: `timestamp_ms` (`millis()`)

Binary record (`MlSample`) is **17 bytes**:

| Field | Type |
|---|---|
| `timestamp_ms` | `u32` |
| `ax, ay, az` | `i16` |
| `gx, gy, gz` | `i16` |
| `trigger` | `u8` |

### Where it is stored / capacity

- Stored in LittleFS: `/ml_log.bin`
- LittleFS size (V2): `board_build.filesystem_size = 1.5m`
- Capacity: about **13–15 minutes** (LittleFS overhead + metadata).

### Runtime behavior (important details)

The logger is built to avoid hurting motor control timing:

- **Core 1 (fast loop)** produces samples at 100 Hz and pushes them into a lock-free SPSC ring buffer in RAM.
- **Core 0 (slow loop)** flushes from RAM → LittleFS only when it is safe (not during firing/ramp states).

Pausing:
- **Pauses while still** to reduce useless “dead time” in datasets.
- **Pauses while a USB serial session is active** (CDC open / DTR asserted) so MLDUMP + model upload is reliable.

HUD indicator on the home screen:
- `Rxx%` = recording, flash fullness estimate.
- `Pxx%` = recording is active but paused (stillness or USB session).

Recording session semantics:
- The log is **one session per boot**. It is cleared on reboot / reflash.

### Starting / stopping recording

From the blaster menu:
- **Menu → Device → ML Recording → Start**
- **Menu → Device → ML Recording → Stop**

## USB behavior (V2)

There are two “USB-related” behaviors:

1) **USB power / cable connected**: the firmware avoids going to standby (so the device stays awake).
2) **USB service mode** (safe mode): activated only when a host opens the CDC serial port (**DTR asserted**).

### USB service mode (STATE_USB)

When a host opens the serial port (for example, the web app presses “I’m ready”), the firmware transitions into `STATE_USB`:

- Motors are forced off (no spin).
- Firing is blocked.
- Intended to make MLDUMP + model upload deterministic and safe.

Exiting:
- When the host closes serial / drops DTR (web app Disconnect), firmware returns via `STATE_SETUP` to re-establish ESC comm/EDT.

Implementation note:
- TinyUSB APIs are sampled on **core 0** in `usbSessionLoop0()` and published as atomics (`usbCdcActive()` / `usbSessionActive()`). Core 1 must not call TinyUSB internals directly.

## Getting data off the blaster

### Serial command: MLDUMP

The firmware supports:
- `MLDUMP` → streams the full binary log.

Protocol:
- Device prints `MLDUMP1 <size>`
- Device streams `<size>` raw bytes
- Device prints `MLDUMP_DONE`

### Python pull + convert

```sh
source .venv/bin/activate
python python/ml_log_pull.py -o ml_log.bin
python python/ml_log_convert.py ml_log.bin -o ml_log.csv
```

### Explore + export training dataset

```sh
source .venv/bin/activate
python python/ml_log_explore.py ml_log.csv --outdir ml_out --clean
python python/ml_log_explore.py ml_log.csv --outdir ml_out --clean --export-dataset
```

Outputs:
- `ml_out/dataset.png` (whole session plot)
- `ml_out/shots/` (per-shot crops)
- `ml_out/train/` (binary windows + labels + metadata)

## Inference (Idle pre-spin)

### Objective

The model predicts the probability that a shot is imminent:

- Window length: `window_ms = 500`
- Lead time: `lead_ms = 100`
- At time `t`, the model sees `[t-(lead+window), t-lead]` and outputs `p(about_to_shoot_at_t)`.

### Available models

- **LR**: 18 “summary” features (mean/std/absmax per axis).
- **MLP**: 30 “rich” features (summary + absmean + magnitude stats), network `30 → 64 → 32 → 1`.

### Where inference runs

- Core 1 only pushes samples into the sliding window (`mlInferLoop()`).
- Core 0 computes inference in `mlInferSlowLoop()` (~100 Hz max) and caches the probability for core 1 to read.

### Motor → Idling integration

Under **Menu → Motor → Idling**:
- `ML:LR`
- `ML:MLP`

Extra settings (visible when ML idling is selected):
- **ML Idle Mode**: `Binary` or `Dynamic`
- **ML Threshold %**: maps probability to “on” threshold (and dynamic scaling).

HUD:
- When not recording, the top-center can show model confidence as a percentage while ML idling is selected.

## Uploading personalized models (no UF2)

### Model file format (MLMD)

On-device models are stored as `MLMD` files:
- A small header (`magic`, model type, feature dims, layer dims)
- Followed by float payload (scaler + weights)
- Payload CRC32 checked on load; failures revert that model back to factory defaults

Files:
- `/ml_model_lr.bin`
- `/ml_model_mlp.bin`

Persistence:
- Uploaded models persist in LittleFS and are **auto-loaded on boot**.

### Serial commands

Query:
- `MLMODEL_INFO` → prints:
  - `model_source=factory|user`
  - `user_model_present=0|1`
  - `user_has_lr=0|1`
  - `user_has_mlp=0|1`

Upload:
- `MLMODEL_PUT_LR <size> <crc32hex>` → device replies `MLMODEL_READY`, then you stream bytes, then device replies `MLMODEL_OK`
- `MLMODEL_PUT_MLP <size> <crc32hex>` → same

Load into RAM:
- `MLMODEL_LOAD_LR` → `MLMODEL_LOADED`
- `MLMODEL_LOAD_MLP` → `MLMODEL_LOADED`

Delete user models:
- `MLMODEL_DELETE` → deletes user model files and reverts to factory weights

### Python export + upload (manual)

```sh
source .venv/bin/activate
python python/ml_model_export.py --train-dir ml_out/train --model both
python python/ml_model_push.py --port /dev/cu.usbmodemXXXX
python python/ml_model_push.py --port /dev/cu.usbmodemXXXX --info
```

## Web UI (local MVP)

This repo includes `ml_web/`:

- Browser (Web Serial) connects to the blaster, pulls `MLDUMP` with progress.
- Backend (FastAPI) runs the dataset pipeline + trains LR+MLP.
- Browser shows per-shot plots and then uploads both model files back to the blaster.

Run:
```sh
cd /Users/stan/Documents/GitHub/Firmware
source .venv/bin/activate
python -m pip install -r python/requirements.txt
python -m pip install -r ml_web/server/requirements.txt
python -m ml_web.server.app
```

Open:
- `http://127.0.0.1:8000/` (Chrome)

Notes:
- Don’t open the HTML as `file://` (fetches will fail).
- The web UI has a **downloadable debug trace** for diagnosing WebSerial issues.

