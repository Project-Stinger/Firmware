#!/usr/bin/env python3
"""
Step 3: Feature Engineering

Extract 42 features from IMU data using sliding window (matching C++ implementation).
"""

import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Add utils to path
sys.path.insert(0, '.')

from utils.feature_extractor import extract_features, analyze_feature_importance

# Configuration
INPUT_DIR = 'outputs/preprocessed'
OUTPUT_DIR = 'outputs/features'
WINDOW_SIZE = 50  # Matching C++ implementation
STRIDE = 10  # Use stride to reduce dataset size (every 10th sample)


def main():
    print("="*80)
    print("STEP 3: FEATURE ENGINEERING")
    print("="*80)

    # Create output directory
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load preprocessed data
    print("\n[1] Loading preprocessed data...")
    train_df = pd.read_csv(f'{INPUT_DIR}/train.csv')
    val_df = pd.read_csv(f'{INPUT_DIR}/val.csv')
    test_df = pd.read_csv(f'{INPUT_DIR}/test.csv')

    print(f"   Train: {len(train_df):,} samples")
    print(f"   Val:   {len(val_df):,} samples")
    print(f"   Test:  {len(test_df):,} samples")

    # Extract features
    print(f"\n[2] Extracting features (window_size={WINDOW_SIZE}, stride={STRIDE})...")
    print("   This may take a few minutes...")

    print("\n   Extracting train features...")
    X_train, y_train = extract_features(train_df, window_size=WINDOW_SIZE, stride=STRIDE)

    print("\n   Extracting validation features...")
    X_val, y_val = extract_features(val_df, window_size=WINDOW_SIZE, stride=STRIDE)

    print("\n   Extracting test features...")
    X_test, y_test = extract_features(test_df, window_size=WINDOW_SIZE, stride=STRIDE)

    # Feature analysis
    print("\n[3] Analyzing features...")
    analyze_feature_importance(X_train, y_train)

    # Normalize features
    print("\n[4] Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("   Features normalized using StandardScaler")
    print(f"   Feature mean: {X_train_scaled.mean():.6f}")
    print(f"   Feature std:  {X_train_scaled.std():.6f}")

    # Save features
    print("\n[5] Saving features...")

    np.save(f'{OUTPUT_DIR}/X_train.npy', X_train_scaled)
    np.save(f'{OUTPUT_DIR}/y_train.npy', y_train)
    print(f"   Saved: {OUTPUT_DIR}/X_train.npy, y_train.npy ({len(X_train_scaled):,} samples)")

    np.save(f'{OUTPUT_DIR}/X_val.npy', X_val_scaled)
    np.save(f'{OUTPUT_DIR}/y_val.npy', y_val)
    print(f"   Saved: {OUTPUT_DIR}/X_val.npy, y_val.npy ({len(X_val_scaled):,} samples)")

    np.save(f'{OUTPUT_DIR}/X_test.npy', X_test_scaled)
    np.save(f'{OUTPUT_DIR}/y_test.npy', y_test)
    print(f"   Saved: {OUTPUT_DIR}/X_test.npy, y_test.npy ({len(X_test_scaled):,} samples)")

    # Save scaler and feature names
    joblib.dump(scaler, f'{OUTPUT_DIR}/scaler.pkl')
    print(f"   Saved: {OUTPUT_DIR}/scaler.pkl")

    from utils.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor(window_size=WINDOW_SIZE)
    feature_names = extractor.feature_names
    joblib.dump(feature_names, f'{OUTPUT_DIR}/feature_names.pkl')
    print(f"   Saved: {OUTPUT_DIR}/feature_names.pkl")

    # Save metadata
    metadata = {
        'window_size': WINDOW_SIZE,
        'stride': STRIDE,
        'n_features': X_train.shape[1],
        'feature_names': feature_names,
        'train_samples': len(X_train_scaled),
        'val_samples': len(X_val_scaled),
        'test_samples': len(X_test_scaled),
        'train_positive_ratio': y_train.sum() / len(y_train),
        'val_positive_ratio': y_val.sum() / len(y_val),
        'test_positive_ratio': y_test.sum() / len(y_test),
    }

    joblib.dump(metadata, f'{OUTPUT_DIR}/feature_metadata.pkl')
    print(f"   Saved: {OUTPUT_DIR}/feature_metadata.pkl")

    # Print summary
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"\nFeatures extracted:")
    print(f"  Window size: {WINDOW_SIZE} samples")
    print(f"  Stride: {STRIDE} samples")
    print(f"  Number of features: {X_train.shape[1]}")
    print(f"\nDataset sizes:")
    print(f"  Train: {len(X_train_scaled):,} samples ({y_train.sum():,} positive, {y_train.sum()/len(y_train)*100:.2f}%)")
    print(f"  Val:   {len(X_val_scaled):,} samples ({y_val.sum():,} positive, {y_val.sum()/len(y_val)*100:.2f}%)")
    print(f"  Test:  {len(X_test_scaled):,} samples ({y_test.sum():,} positive, {y_test.sum()/len(y_test)*100:.2f}%)")

    print("\nNext steps:")
    print("  - Run 4_train_logistic_regression.py")
    print("  - Run 5_train_random_forest.py")
    print("  - Run 6_train_neural_network.py")


if __name__ == '__main__':
    main()
