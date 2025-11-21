# Local Setup Instructions

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- ~500MB free disk space (for packages)

## Installation

### 1. Clone and Navigate
```bash
cd MLmodel
```

### 2. Install Dependencies
```bash
# Option 1: Install all at once (recommended)
pip install -r requirements.txt

# Option 2: Install individually (if Option 1 fails)
pip install numpy pandas scikit-learn
pip install torch torchvision  # This may take a while
pip install matplotlib seaborn
pip install joblib tqdm scipy
```

**Note**: PyTorch installation may take 5-10 minutes depending on your internet connection.

### 3. Verify Installation
```bash
python -c "import numpy, pandas, sklearn, torch, matplotlib; print('✓ All packages installed!')"
```

## Running the Pipeline

### Full Pipeline (Recommended)
```bash
python run_full_pipeline.py
```
This will run all 8 steps automatically (takes 10-30 minutes total).

### Individual Steps
If you want to run steps individually or if something fails:

```bash
# Step 1: Data Exploration (2-3 min)
python 1_data_exploration.py

# Step 2: Preprocessing (1-2 min)
python 2_preprocessing.py

# Step 3: Feature Engineering (1-2 min)
python 3_feature_engineering.py

# Step 4: Train Logistic Regression (< 1 min)
python 4_train_logistic_regression.py

# Step 5: Train Random Forest (< 1 min)
python 5_train_random_forest.py

# Step 6: Train Neural Network (3-5 min)
python 6_train_neural_network.py

# Step 7: Evaluate Models (< 1 min)
python 7_evaluate_models.py

# Step 8: Export for Deployment (< 1 min)
python 8_export_for_deployment.py
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'X'"
→ Run `pip install -r requirements.txt` again
→ Or install the specific package: `pip install X`

### PyTorch Installation Fails
Try CPU-only version:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Out of Memory Error
The dataset is large (~559K samples). If you get memory errors:
1. Edit `3_feature_engineering.py`
2. Change `STRIDE = 10` to `STRIDE = 20` (line 16)
3. This will use fewer samples but still work fine

### Neural Network Training Too Slow
You can skip the neural network:
```bash
# Run steps 1-5 and 7-8 only
python 1_data_exploration.py
python 2_preprocessing.py
python 3_feature_engineering.py
python 4_train_logistic_regression.py
python 5_train_random_forest.py
# Skip step 6
python 7_evaluate_models.py  # Works without NN
python 8_export_for_deployment.py
```

The Random Forest is the deployable model anyway!

### Permission Errors on Windows
Run your terminal/command prompt as Administrator.

### Path Issues
Make sure you're in the `MLmodel/` directory when running scripts.

## Expected Runtime

On a typical laptop:
- **Steps 1-5**: ~10 minutes
- **Step 6 (Neural Network)**: 3-5 minutes (depends on CPU)
- **Steps 7-8**: ~2 minutes
- **Total**: 15-20 minutes

## What You'll Get

After completion, check these directories:

```
MLmodel/outputs/
├── exploration/              # Data visualizations
│   ├── imu_data_sample.png
│   ├── temporal_analysis.png
│   └── ...
├── models/                   # Trained models
│   ├── logistic_regression/
│   ├── random_forest/       # ⭐ Main deployment model
│   └── neural_network/      # (optional)
├── evaluation/              # Model comparison
│   ├── model_comparison.csv
│   └── roc_curves_comparison.png
└── deployment/              # ⭐ C++ export
    ├── rf_model.h           # Copy this to firmware!
    ├── usage_example.cpp
    └── README.md
```

## Quick Test

To quickly verify everything works:
```bash
# Quick test (uses small subset of data)
python -c "
from utils.data_loader import load_imu_data
df = load_imu_data('../nerf_imu_data.csv')
print(f'✓ Loaded {len(df)} samples')
"
```

## Next Steps After Pipeline Runs

1. **Check results**: `cat outputs/evaluation/model_comparison.csv`
2. **Get deployment file**: `outputs/deployment/rf_model.h`
3. **Copy to firmware**: `cp outputs/deployment/rf_model.h ../src/`
4. **Integrate**: See `outputs/deployment/README.md`

## Need Help?

- Check `QUICK_START.md` for common issues
- Check `PIPELINE_OVERVIEW.md` for technical details
- Check individual script headers for documentation

## Platform-Specific Notes

### macOS / Linux
Should work out of the box. If you use conda:
```bash
conda create -n nerf python=3.9
conda activate nerf
pip install -r requirements.txt
```

### Windows
Use Command Prompt or PowerShell (not Git Bash for Python).
If matplotlib doesn't show plots, install: `pip install PyQt5`

## Minimal Working Example

If you just want to test the core functionality:

```python
# test_pipeline.py
import numpy as np
from utils.data_loader import load_imu_data, create_temporal_labels
from utils.feature_extractor import FeatureExtractor

# Load small sample
df = load_imu_data('../nerf_imu_data.csv')
df_sample = df.head(5000)  # Just 5000 samples

# Create labels
df_labeled = create_temporal_labels(df_sample)

# Extract features
extractor = FeatureExtractor(window_size=50)
features, labels = extractor.extract_features_from_dataframe(df_labeled[:500])

print(f"✓ Extracted {features.shape[0]} feature vectors")
print(f"✓ Each vector has {features.shape[1]} features")
```

Run: `python test_pipeline.py`
