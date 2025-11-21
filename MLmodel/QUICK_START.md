# Quick Start Guide

## TL;DR - Run Everything
```bash
cd MLmodel
pip install -r requirements.txt
python run_full_pipeline.py
```

## What This Does
Trains 3 ML models to predict nerf gun trigger pulls 100-400ms in advance:
1. ✅ **Logistic Regression** - Fast baseline
2. ⭐ **Random Forest** - **RECOMMENDED for RP2040** (deployable)
3. 🧠 **Neural Network** - Best performance (not deployable)

## Output You Care About
After running the pipeline, check:

### 1. Model Performance
```bash
cat outputs/evaluation/model_comparison.csv
```
Look at: **FP/s** (False Positives per Second) - lower is better!

### 2. Deployment Files
```
outputs/deployment/
├── rf_model.h          ← Copy this to src/
├── usage_example.cpp   ← Integration guide
└── README.md           ← Detailed deployment instructions
```

### 3. Copy to Firmware
```bash
cp outputs/deployment/rf_model.h ../src/
```

Then edit `src/ml_predictor.cpp`:
```cpp
#include "rf_model.h"

// Replace model.predict() with:
bool prediction = predict_prefire(features);
```

## Key Metrics to Watch

| Metric | What It Means | Target |
|--------|---------------|--------|
| **FP/s** | False alarms per second | < 1-2 |
| **Recall** | % of trigger pulls caught | > 80% |
| **F1-Score** | Overall quality | > 0.6 |
| **Model Size** | Memory usage | < 100 KB |

## Customization

### Want Fewer False Positives?
Edit `outputs/deployment/rf_model.h`:
```cpp
#define RF_THRESHOLD 0.5f  // Increase to 0.6 or 0.7
```

### Want Better Performance?
Edit `5_train_random_forest.py`:
```python
RF_CONFIG = {
    'n_estimators': 20,    # More trees (was 10)
    'max_depth': 10,       # Deeper trees (was 8)
}
```
Then re-run steps 5-8.

## Troubleshooting

**"ModuleNotFoundError"**
→ `pip install -r requirements.txt`

**"File not found: nerf_imu_data.csv"**
→ Make sure you're in the `MLmodel/` directory

**"Model too large for RP2040"**
→ Reduce `n_estimators` or `max_depth` in step 5

**"High false positive rate in real world"**
→ Add consecutive prediction filtering (see usage_example.cpp)

## Next Steps
1. ✅ Run pipeline
2. ✅ Check metrics
3. ✅ Copy `rf_model.h` to firmware
4. ✅ Update `ml_predictor.cpp`
5. ✅ Test on hardware
6. ✅ Iterate if needed

## Questions?
- See `PIPELINE_OVERVIEW.md` for detailed documentation
- See `outputs/deployment/README.md` for deployment guide
- Check `usage_example.cpp` for integration examples
