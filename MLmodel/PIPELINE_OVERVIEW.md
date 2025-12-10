# Nerf Gun Pre-Fire Prediction - ML Pipeline Overview

## Executive Summary

This ML pipeline predicts trigger pulls **100-400ms in advance** based on IMU sensor data, enabling pre-spinning of flywheels for faster nerf dart firing. The system addresses the key challenge of **minimizing false positives** to avoid unnecessary flywheel activation while maintaining good recall for actual trigger pulls.

## Problem Statement

### Objective
Predict when a user is about to pull the trigger on a nerf gun, 100-400ms before the actual trigger pull, based on motion patterns detected by a 6-axis IMU (accelerometer + gyroscope).

### Key Constraints
1. **Low False Positive Rate**: Flywheels shouldn't spin up unnecessarily (wastes battery, creates noise/vibration)
2. **MCU Deployment**: Model must fit on RP2040 microcontroller (264KB RAM, limited flash)
3. **Real-time Inference**: Fast enough to run at 1600Hz sampling rate
4. **100-400ms Lead Time**: Enough time to spin up flywheels (need ~100ms)

## Dataset

- **File**: `nerf_imu_data.csv`
- **Size**: ~559,000 samples
- **Duration**: ~349 seconds of data
- **Sampling Rate**: 1600 Hz (625μs period)
- **Features**: 6 raw IMU channels (accel_x/y/z, gyro_x/y/z) + trigger_state
- **Collection**: Real trigger pull events during gameplay

## ML Pipeline Architecture

```
Raw IMU Data (1600 Hz)
         ↓
[1] Data Exploration
    - Visualize IMU patterns
    - Analyze trigger events
    - Temporal characteristics
         ↓
[2] Preprocessing
    - Create temporal labels (100-400ms pre-fire window)
    - Temporal train/val/test split (70/15/15)
         ↓
[3] Feature Engineering
    - 50-sample sliding window (31.25ms)
    - Extract 42 features (matching C++ implementation)
         ├── 24 basic stats (mean/std/min/max per axis)
         ├── 12 derivative features (velocity patterns)
         └── 6 magnitude features (accel/gyro magnitude stats)
    - StandardScaler normalization
         ↓
[4-6] Model Training
    ├── [4] Logistic Regression (baseline, simple)
    ├── [5] Random Forest (MCU-deployable, optimized size)
    └── [6] Neural Network (PyTorch, best performance)
         ↓
[7] Model Evaluation
    - Comprehensive metrics (accuracy, precision, recall, F1, ROC-AUC)
    - **False Positives per Second** (critical metric)
    - Model comparison and analysis
         ↓
[8] Deployment Export
    - Export Random Forest to C++ header file
    - Integration guide for firmware
    - Optimization recommendations
```

## Feature Engineering Details

### 42 Features (matching C++ implementation)

**Basic Statistics (24 features)**:
- For each of 6 axes (accel_x/y/z, gyro_x/y/z):
  - Mean, Std Dev, Min, Max

**Derivative Features (12 features)**:
- For each of 6 axes:
  - Mean of differences (velocity)
  - Std dev of differences (acceleration variability)

**Magnitude Features (6 features)**:
- Accelerometer magnitude: mean, std, max
- Gyroscope magnitude: mean, std, max

**Window**: 50 samples (31.25ms at 1600Hz)
**Stride**: Configurable (10 for training to reduce dataset size)

## Models

### 1. Logistic Regression
**Purpose**: Baseline model
- **Pros**: Fast, interpretable, tiny memory footprint
- **Cons**: Limited capacity for complex patterns
- **Deployment**: Easily deployable on MCU
- **Use Case**: Baseline comparison

### 2. Random Forest
**Purpose**: Main deployable model
- **Configuration**: 10 trees, max depth 8 (tunable)
- **Size**: ~50-100 KB (fits on RP2040)
- **Pros**: Good performance, deployable, robust
- **Cons**: Larger than LR, more complex
- **Deployment**: **RECOMMENDED for RP2040**
- **Use Case**: Production deployment

### 3. Neural Network (PyTorch)
**Purpose**: Performance benchmark
- **Architecture**: [128, 64, 32] hidden layers + dropout + batch norm
- **Size**: ~50,000+ parameters (~200KB)
- **Pros**: Best possible performance
- **Cons**: Too large/slow for RP2040
- **Deployment**: **NOT deployable on MCU**
- **Use Case**: Understand performance ceiling

## Performance Metrics

### Primary Metrics
1. **False Positives per Second (FP/s)** ⭐ CRITICAL
   - Target: < 1-2 FP/s
   - Measures unnecessary flywheel activations

2. **Recall (Sensitivity)**
   - Percentage of actual trigger pulls detected
   - Target: > 80%

3. **Precision**
   - Of all pre-fire predictions, how many were correct
   - Target: > 50%

4. **F1-Score**
   - Harmonic mean of precision and recall
   - Balances both metrics

### Secondary Metrics
- Accuracy (less important due to class imbalance)
- ROC-AUC (model discrimination ability)
- Precision-Recall AUC (better for imbalanced data)
- Confusion Matrix (detailed breakdown)

## Evaluation Strategy

### Data Split
- **Temporal split** (not random) to avoid leakage
- 70% train, 15% validation, 15% test
- Maintains temporal ordering

### Threshold Optimization
- Default threshold: 0.5
- **Optimal threshold**: Found on validation set to target 1.0 FP/s
- Can be adjusted based on use case

### Consecutive Prediction Filtering
- C++ implementation can require N consecutive positive predictions
- Significantly reduces false positives
- Recommended: 10-20 consecutive predictions at 1600Hz

## Deployment

### Selected Model: Random Forest

**Integration Steps**:
1. Run full pipeline: `python run_full_pipeline.py`
2. Find exported model: `outputs/deployment/rf_model.h`
3. Copy to firmware: `cp outputs/deployment/rf_model.h ../src/`
4. Update `ml_predictor.cpp`:
   ```cpp
   #include "rf_model.h"

   bool prediction = predict_prefire(features);  // Replace model.predict()
   ```
5. Add consecutive filtering (optional but recommended)
6. Test and iterate

**Memory Usage**:
- Model size: ~50-100 KB (in flash, not RAM)
- Runtime stack: ~200 bytes
- No dynamic allocation

**Performance on RP2040**:
- Inference time: < 1ms
- Runs easily at 1600Hz
- Low power consumption

## File Structure

```
MLmodel/
├── README.md                          # Quick start guide
├── PIPELINE_OVERVIEW.md               # This file (detailed overview)
├── requirements.txt                   # Python dependencies
├── run_full_pipeline.py               # Master script (runs all steps)
│
├── 1_data_exploration.py              # Visualize and analyze data
├── 2_preprocessing.py                 # Create labels and splits
├── 3_feature_engineering.py           # Extract 42 features
├── 4_train_logistic_regression.py     # Train LR model
├── 5_train_random_forest.py           # Train RF model (MCU-optimized)
├── 6_train_neural_network.py          # Train NN model (PyTorch)
├── 7_evaluate_models.py               # Compare all models
├── 8_export_for_deployment.py         # Export RF to C++
│
├── utils/
│   ├── __init__.py
│   ├── data_loader.py                 # Data loading and temporal labeling
│   ├── feature_extractor.py           # Feature extraction (42 features)
│   ├── metrics.py                     # Performance metrics (FP/s, etc.)
│   └── visualization.py               # Plotting utilities
│
└── outputs/                           # Generated outputs
    ├── exploration/                   # Data visualizations
    ├── preprocessed/                  # Train/val/test splits
    ├── features/                      # Extracted features
    ├── models/                        # Trained models
    │   ├── logistic_regression/
    │   ├── random_forest/
    │   └── neural_network/
    ├── evaluation/                    # Model comparisons
    └── deployment/                    # C++ export
        ├── rf_model.h                 # ⭐ Main deployment file
        ├── usage_example.cpp
        └── README.md
```

## Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline (10-30 minutes)
python run_full_pipeline.py

# Or run individual steps:
python 1_data_exploration.py
python 2_preprocessing.py
python 3_feature_engineering.py
# ... etc
```

### Individual Steps
Each script is self-contained and can be run independently (assuming previous steps are complete):

```bash
# Explore data
python 1_data_exploration.py

# Process data
python 2_preprocessing.py

# Extract features
python 3_feature_engineering.py

# Train models
python 4_train_logistic_regression.py
python 5_train_random_forest.py
python 6_train_neural_network.py

# Compare models
python 7_evaluate_models.py

# Export for deployment
python 8_export_for_deployment.py
```

## Customization and Tuning

### Adjust Prediction Window
In `2_preprocessing.py`:
```python
PREDICTION_WINDOW_MS = (100, 400)  # Change to (150, 300) for example
```

### Random Forest Size Optimization
In `5_train_random_forest.py`:
```python
RF_CONFIG = {
    'n_estimators': 10,    # Try 5, 10, 15, 20
    'max_depth': 8,        # Try 5, 8, 10, 12
    # ...
}
```

Smaller values = smaller model size, potentially lower performance
Larger values = better performance, larger model size

### Threshold Tuning
In `7_evaluate_models.py` or after deployment:
- Higher threshold → fewer false positives, lower recall
- Lower threshold → more catches, more false positives
- Start with optimal threshold from validation set
- Adjust based on real-world testing

### Feature Engineering
In `3_feature_engineering.py`:
```python
WINDOW_SIZE = 50  # Adjust window size (matches C++)
STRIDE = 10       # Adjust stride (1 = every sample, slower training)
```

## Research Context

This pipeline applies machine learning concepts from the course:

1. **Supervised Learning**: Binary classification (pre-fire vs. no pre-fire)
2. **Feature Engineering**: Transform raw sensor data into meaningful features
3. **Model Selection**: Compare multiple algorithms (LR, RF, NN)
4. **Regularization**: Prevent overfitting (depth limits, dropout, class weights)
5. **Evaluation**: Comprehensive metrics, cross-validation strategy
6. **Deployment**: Model export and optimization for embedded systems

### Related Research Areas
- **Human Activity Recognition** (HAR) from IMU data
- **Gesture Recognition** for wearable devices
- **Predictive Motion Analysis** in robotics
- **Time-Series Classification** with limited compute
- **Edge ML** and model compression

## Known Limitations and Future Work

### Current Limitations
1. **Single user training**: Model may not generalize to different grip styles
2. **Limited scenarios**: Trained on specific shooting scenarios
3. **Static model**: No online learning or adaptation
4. **Window size**: Fixed 50-sample window may not be optimal

### Future Improvements
1. **Multi-user data**: Collect data from diverse users
2. **Data augmentation**: Synthetic motion patterns
3. **Model quantization**: Further reduce model size
4. **Online learning**: Adapt to user over time
5. **Feature selection**: Automatic feature importance analysis
6. **Hyperparameter optimization**: Grid search or Bayesian optimization
7. **Ensemble methods**: Combine multiple models
8. **Recurrent models**: LSTM/GRU for temporal dependencies (if compute allows)

## Troubleshooting

### High False Positive Rate
- Increase threshold in `rf_model.h`
- Implement consecutive prediction filtering (see usage_example.cpp)
- Retrain with more diverse negative samples

### Low Recall
- Decrease threshold
- Check feature extraction matches training
- Verify 1600Hz sampling rate
- Ensure 100-400ms prediction window

### Model Too Large
- Reduce `n_estimators` or `max_depth` in RF config
- Retrain and re-export
- Consider using Logistic Regression instead

### Poor Generalization
- Collect more diverse training data
- Check for data leakage (temporal split is correct)
- Validate feature extraction implementation

## References

### ML Course Concepts
- Linear/Logistic Regression
- Decision Trees and Random Forests
- Neural Networks and Backpropagation
- Model Evaluation and Validation
- Regularization Techniques
- Feature Engineering

### Implementation Resources
- scikit-learn documentation
- PyTorch documentation
- IMU sensor data processing
- Embedded ML deployment

## Contact and Contribution

This pipeline was developed as part of the Machine Learning course project for predictive trigger control in nerf guns. The goal is to demonstrate end-to-end ML deployment on resource-constrained hardware.

**Key Achievement**: Successfully deploy a machine learning model on a microcontroller with <100KB memory that can predict user intent 100-400ms in advance from high-frequency sensor data.

---

**Generated**: 2025
**Version**: 1.0
**Status**: Production Ready
