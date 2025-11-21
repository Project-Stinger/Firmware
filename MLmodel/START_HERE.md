# 🚀 START HERE - Quick Guide

## What You Have

A complete ML pipeline to predict nerf gun trigger pulls 100-400ms in advance! ✨

## Quick Start (5 minutes)

### 1. Verify Your Setup
```bash
cd MLmodel
python verify_setup.py
```

This will check if everything is ready. If not, it will tell you what to install.

### 2. Install Packages (if needed)
```bash
pip install -r requirements.txt
```

⏱️ Takes 5-10 minutes (PyTorch is large)

### 3. Run the Pipeline
```bash
python run_full_pipeline.py
```

⏱️ Takes 15-20 minutes total

That's it! ✅

## What It Does

Trains 3 models on your IMU data:
1. **Logistic Regression** - Fast baseline
2. **Random Forest** ⭐ - Deployable on RP2040 
3. **Neural Network** - Best performance (optional)

## After It Runs

### 1. Check Performance
```bash
cat outputs/evaluation/model_comparison.csv
```

Look for:
- **FP/s** (False Positives per Second) - Lower is better
- **ROC-AUC** - Higher is better (0.8+ is good)

### 2. Get Deployment File
```bash
ls outputs/deployment/rf_model.h
```

This is your C++ header file for the RP2040!

### 3. Deploy to Firmware
```bash
cp outputs/deployment/rf_model.h ../src/
```

Then edit `src/ml_predictor.cpp`:
```cpp
#include "rf_model.h"
bool prediction = predict_prefire(features);
```

See `outputs/deployment/README.md` for details.

## Files & Docs

- **LOCAL_SETUP.md** - Detailed installation guide
- **QUICK_START.md** - TL;DR version
- **PIPELINE_OVERVIEW.md** - Technical deep dive
- **requirements.txt** - Python packages needed

## Troubleshooting

### Packages won't install?
See `LOCAL_SETUP.md` - Section "Troubleshooting"

### Out of memory?
Edit `3_feature_engineering.py`, line 16:
```python
STRIDE = 20  # Change from 10 to 20
```

### Don't want Neural Network?
Skip step 6:
```bash
python 1_data_exploration.py
python 2_preprocessing.py
python 3_feature_engineering.py
python 4_train_logistic_regression.py
python 5_train_random_forest.py
# Skip: python 6_train_neural_network.py
python 7_evaluate_models.py
python 8_export_for_deployment.py
```

The Random Forest (step 5) is what you'll deploy anyway!

## Expected Results

Your Random Forest model should achieve:
- ✅ False Positives: < 1 per second
- ✅ ROC-AUC: > 0.85
- ✅ Model Size: ~100 KB (fits on RP2040)

## Need Help?

1. Run `python verify_setup.py` first
2. Check `LOCAL_SETUP.md` for detailed troubleshooting
3. Check individual script headers for documentation

## Next Steps

After deployment:
1. Test on hardware
2. Adjust threshold if needed (in `rf_model.h`)
3. Collect more data to improve model
4. Iterate!

---

**Made by Claude** 🤖 | Ready to predict your trigger pulls! 🎯
