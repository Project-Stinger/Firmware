# Nerf Gun Pre-Fire Prediction ML Pipeline

## Overview
This ML pipeline predicts trigger pulls 100-400ms in advance based on IMU sensor data to enable pre-spinning of flywheels for faster nerf dart firing.

## Problem Statement
- **Task**: Binary classification (predict trigger pull 100-400ms before it happens)
- **Input**: IMU data (3-axis accelerometer, 3-axis gyroscope) at 1600 Hz
- **Output**: Trigger pull prediction (0 = no action, 1 = pre-spin flywheels)
- **Key Constraint**: LOW false positive rate (to avoid unnecessary flywheel spinning/twitching)

## Dataset
- **File**: `nerf_imu_data.csv`
- **Size**: ~559K samples
- **Format**: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, trigger_state
- **Sampling Rate**: 1600 Hz (625 µs per sample)

## Models

### 1. Logistic Regression (Baseline)
- Simple linear model for baseline comparison
- Fast inference, low memory footprint
- Good for understanding feature importance

### 2. Random Forest (MCU Deployable)
- Target: Small enough to run on RP2040 MCU
- Optimized for low false positive rate
- Balance between accuracy and model size

### 3. Neural Network (Best Performance)
- PyTorch implementation
- Not deployable on RP2040 (for comparison only)
- Aimed at achieving best possible performance

## Feature Engineering
Matching the C++ implementation (42 features total):
- **Basic statistics** (24 features): mean, std, min, max for each of 6 axes
- **Derivative features** (12 features): mean and std of velocity for each axis
- **Magnitude features** (6 features): accel_magnitude (mean, std, max), gyro_magnitude (mean, std, max)

## Performance Metrics
1. **Classification Metrics**:
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - ROC Curve and AUC

2. **False Positive Analysis** (CRITICAL):
   - False Positive Rate (FPR)
   - False Positives Per Second (FP/s)
   - Precision-Recall trade-off

3. **Temporal Analysis**:
   - Prediction lead time distribution
   - Early prediction accuracy (100-400ms window)

4. **Deployment Metrics**:
   - Model size (bytes)
   - Inference time
   - Memory footprint

## File Structure
```
MLmodel/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── 1_data_exploration.py              # Data analysis and visualization
├── 2_preprocessing.py                 # Temporal labeling and train/val/test split
├── 3_feature_engineering.py           # Extract 42 features from raw IMU data
├── 4_train_logistic_regression.py     # Train LR model
├── 5_train_random_forest.py           # Train RF model with size optimization
├── 6_train_neural_network.py          # Train NN model with PyTorch
├── 7_evaluate_models.py               # Comprehensive model comparison
├── 8_export_for_deployment.py         # Export RF model for C++ deployment
└── utils/
    ├── __init__.py
    ├── data_loader.py                 # Data loading utilities
    ├── feature_extractor.py           # Feature extraction matching C++ code
    ├── metrics.py                     # Custom metrics (FP/s, etc.)
    └── visualization.py               # Plotting utilities
```

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
python 1_data_exploration.py          # Explore data distribution
python 2_preprocessing.py             # Create temporal labels
python 3_feature_engineering.py       # Extract features
python 4_train_logistic_regression.py # Train LR
python 5_train_random_forest.py       # Train RF
python 6_train_neural_network.py      # Train NN
python 7_evaluate_models.py           # Compare all models
python 8_export_for_deployment.py     # Export RF to C++
```

## Key Findings (To Be Filled After Training)
- Best model for RP2040 deployment: TBD
- Achieved FP/s rate: TBD
- Prediction lead time: TBD
- Trade-offs: TBD

## References
- C++ implementation: `src/ml_predictor.cpp`
- Feature extraction: 50-sample sliding window, 1600 Hz sampling
- Target prediction window: 100-400ms before trigger pull
