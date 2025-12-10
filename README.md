# Stinger Firmware

Firmware for the Stinger foam dart blaster with ML-based pre-fire prediction.

## Features

- PID controller for flywheel RPM
- Bidirectional DShot using RP2040's PIO
- Closed loop solenoid control
- Multiple firing modes
- Menu system (joystick + display)
- IMU for motion sensing
- ML pre-fire prediction (predicts trigger pulls 100-400ms in advance)
- Speaker and LED output
- Tournament mode

## Building

Built with [PlatformIO](https://platformio.org/). Use the VSCode extension or CLI.

1. Select target: `Ctrl+Shift+P` then "PlatformIO: Pick project environment" then v1 or v2
2. Build and upload using the bottom bar buttons

See `platformio.ini` for debug options.

## ML Prediction

The firmware uses a Random Forest model to predict trigger pulls before they happen, enabling flywheel pre-spin for faster shots.

### How It Works

- Input: IMU data (accelerometer + gyroscope) at 1600 Hz
- Features: 42 features extracted from 50-sample sliding window
- Model: Random Forest (10 trees, ~104 KB)
- Inference: Runs at ~167 Hz (every 6ms)

### Tunable Parameters (via menu)

| Setting | Default | Range | Effect |
|---------|---------|-------|--------|
| Sensitivity | 20 | 5-50 | Consecutive predictions required (lower = faster, more false positives) |
| Confidence | 0.35 | 0.1-0.9 | Prediction threshold (higher = fewer false positives) |

### Data Collection

Use `capture_data.py` to record IMU data for training:

```bash
python capture_data.py
```

Data is saved to `nerf_imu_data.csv`.

### Retraining the Model

```bash
cd MLmodel
pip install -r requirements.txt
python run_full_pipeline.py
cp outputs/deployment/rf_model.h ../src/
```

Key files:
- `capture_data.py` - Data collection script
- `MLmodel/*.py` - Training pipeline
- `src/ml_predictor.cpp` - Firmware inference
- `src/rf_model.h` - Exported model weights

## Documentation

See the [Stinger-Docs wiki](https://github.com/bastian2001/Stinger-Docs/wiki).

## Contributing

- Bug reports: [Open an issue](https://github.com/The-Stinger/stinger-firmware/issues/new)
- Feature requests: [Open an issue](https://github.com/The-Stinger/stinger-firmware/issues/new)
- Pull requests: Fork, file an issue first, then submit PR

## License

[Polyform Non-Commercial License](LICENSE)
