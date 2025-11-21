#!/usr/bin/env python3
"""
Verify that your environment is set up correctly to run the ML pipeline.
"""

import sys

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ✗ Python 3.8+ required")
        return False
    else:
        print("  ✓ Python version OK")
        return True


def check_packages():
    """Check if required packages are installed."""
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'scikit-learn',
        'torch': 'PyTorch',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'joblib': 'Joblib',
        'tqdm': 'tqdm',
        'scipy': 'SciPy'
    }

    print("\nChecking packages:")
    all_ok = True

    for module, name in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - Install with: pip install {name.lower()}")
            all_ok = False

    return all_ok


def check_data_file():
    """Check if data file exists."""
    import os
    print("\nChecking data file:")

    if os.path.exists('../nerf_imu_data.csv'):
        size_mb = os.path.getsize('../nerf_imu_data.csv') / (1024 * 1024)
        print(f"  ✓ nerf_imu_data.csv found ({size_mb:.1f} MB)")
        return True
    else:
        print("  ✗ nerf_imu_data.csv not found")
        print("     Make sure you're in the MLmodel/ directory")
        return False


def check_gpu():
    """Check if GPU is available (optional)."""
    print("\nChecking GPU (optional):")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA GPU available: {torch.cuda.get_device_name(0)}")
            print("     Neural network training will be faster!")
        else:
            print("  ⚠ No GPU detected (will use CPU)")
            print("     This is fine, training will just take a bit longer")
    except:
        print("  ⚠ Cannot check GPU")


def check_disk_space():
    """Check available disk space."""
    import os
    print("\nChecking disk space:")

    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)

        if free_gb < 0.5:
            print(f"  ⚠ Only {free_gb:.1f} GB free (need ~0.5 GB)")
        else:
            print(f"  ✓ {free_gb:.1f} GB available")
    except:
        print("  ? Cannot check disk space")


def quick_test():
    """Run a quick functionality test."""
    print("\nRunning quick functionality test:")

    try:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier

        # Quick ML test
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, 100)

        model = RandomForestClassifier(n_estimators=5, max_depth=3)
        model.fit(X, y)
        pred = model.predict(X[:5])

        print("  ✓ Can create and train models")

        # Test feature extraction
        sys.path.insert(0, '.')
        from utils.feature_extractor import FeatureExtractor

        extractor = FeatureExtractor(window_size=50)
        print("  ✓ Can import utilities")

        return True

    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        return False


def main():
    print("="*60)
    print("ML Pipeline Setup Verification")
    print("="*60)

    results = {
        'Python version': check_python_version(),
        'Required packages': check_packages(),
        'Data file': check_data_file(),
        'Functionality': quick_test()
    }

    check_gpu()
    check_disk_space()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = all(results.values())

    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test:.<40} {status}")

    print("="*60)

    if all_passed:
        print("\n🎉 All checks passed! You're ready to run the pipeline.")
        print("\nNext step:")
        print("  python run_full_pipeline.py")
        print("\nOr run individual steps:")
        print("  python 1_data_exploration.py")
        return 0
    else:
        print("\n⚠ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install packages: pip install -r requirements.txt")
        print("  - Make sure you're in the MLmodel/ directory")
        print("  - Check that nerf_imu_data.csv is in parent directory")
        return 1


if __name__ == '__main__':
    sys.exit(main())
