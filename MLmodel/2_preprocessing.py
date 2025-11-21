#!/usr/bin/env python3
"""
Step 2: Data Preprocessing

Create temporal labels for pre-fire prediction and split data.
"""

import sys
import numpy as np
import pandas as pd
import joblib

# Add utils to path
sys.path.insert(0, '.')

from utils.data_loader import (
    load_imu_data,
    create_temporal_labels,
    split_data_temporal
)
from utils.visualization import plot_imu_data

# Configuration
DATA_PATH = '../nerf_imu_data.csv'
OUTPUT_DIR = 'outputs/preprocessed'
SAMPLING_RATE_HZ = 1600
PREDICTION_WINDOW_MS = (100, 400)  # Predict 100-400ms before trigger pull
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def main():
    print("="*80)
    print("STEP 2: DATA PREPROCESSING")
    print("="*80)

    # Create output directory
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("\n[1] Loading data...")
    df = load_imu_data(DATA_PATH)

    # Create temporal labels
    print("\n[2] Creating temporal labels...")
    df_labeled = create_temporal_labels(
        df,
        prediction_window_ms=PREDICTION_WINDOW_MS,
        sampling_rate_hz=SAMPLING_RATE_HZ
    )

    # Visualize labeled data
    print("\n[3] Visualizing labeled data...")

    # Find a trigger pull event
    trigger_pull_idx = None
    for i in range(1, len(df_labeled)):
        if df_labeled.iloc[i]['trigger_state'] == 1 and df_labeled.iloc[i-1]['trigger_state'] == 0:
            trigger_pull_idx = i
            break

    if trigger_pull_idx:
        # Plot 1 second around trigger pull showing pre-fire window
        start_idx = max(0, trigger_pull_idx - 1200)  # 750ms before
        plot_imu_data(df_labeled, start_idx=start_idx, n_samples=2400,  # 1.5 seconds total
                     save_path=f'{OUTPUT_DIR}/labeled_data_example.png')
        print(f"   Saved: {OUTPUT_DIR}/labeled_data_example.png")

    # Split data temporally
    print("\n[4] Splitting data into train/val/test...")
    train_df, val_df, test_df = split_data_temporal(
        df_labeled,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )

    # Save splits
    print("\n[5] Saving preprocessed data...")
    train_df.to_csv(f'{OUTPUT_DIR}/train.csv', index=False)
    print(f"   Saved: {OUTPUT_DIR}/train.csv ({len(train_df):,} samples)")

    val_df.to_csv(f'{OUTPUT_DIR}/val.csv', index=False)
    print(f"   Saved: {OUTPUT_DIR}/val.csv ({len(val_df):,} samples)")

    test_df.to_csv(f'{OUTPUT_DIR}/test.csv', index=False)
    print(f"   Saved: {OUTPUT_DIR}/test.csv ({len(test_df):,} samples)")

    # Save metadata
    metadata = {
        'sampling_rate_hz': SAMPLING_RATE_HZ,
        'prediction_window_ms': PREDICTION_WINDOW_MS,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'train_pre_fire_samples': int(train_df['pre_fire_label'].sum()),
        'val_pre_fire_samples': int(val_df['pre_fire_label'].sum()),
        'test_pre_fire_samples': int(test_df['pre_fire_label'].sum()),
    }

    joblib.dump(metadata, f'{OUTPUT_DIR}/metadata.pkl')
    print(f"   Saved: {OUTPUT_DIR}/metadata.pkl")

    # Print summary
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\nDataset split:")
    print(f"  Train: {len(train_df):,} samples ({len(train_df)/len(df_labeled)*100:.1f}%)")
    print(f"    - Pre-fire: {train_df['pre_fire_label'].sum():,} ({train_df['pre_fire_label'].sum()/len(train_df)*100:.2f}%)")
    print(f"  Val:   {len(val_df):,} samples ({len(val_df)/len(df_labeled)*100:.1f}%)")
    print(f"    - Pre-fire: {val_df['pre_fire_label'].sum():,} ({val_df['pre_fire_label'].sum()/len(val_df)*100:.2f}%)")
    print(f"  Test:  {len(test_df):,} samples ({len(test_df)/len(df_labeled)*100:.1f}%)")
    print(f"    - Pre-fire: {test_df['pre_fire_label'].sum():,} ({test_df['pre_fire_label'].sum()/len(test_df)*100:.2f}%)")

    print(f"\nPrediction window: {PREDICTION_WINDOW_MS[0]}-{PREDICTION_WINDOW_MS[1]}ms before trigger pull")
    print(f"Sampling rate: {SAMPLING_RATE_HZ} Hz")

    print("\nNext step: Run 3_feature_engineering.py to extract features")


if __name__ == '__main__':
    main()
