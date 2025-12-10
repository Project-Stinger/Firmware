#!/usr/bin/env python3
"""
Step 1: Data Exploration

Analyze the IMU dataset to understand:
- Data distribution and statistics
- Trigger pull patterns
- Class imbalance
- Temporal characteristics
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add utils to path
sys.path.insert(0, '.')

from utils.data_loader import load_imu_data, calculate_sampling_statistics
from utils.visualization import plot_imu_data, plot_temporal_analysis

# Configuration
DATA_PATH = '../nerf_imu_data.csv'
OUTPUT_DIR = 'outputs/exploration'
SAMPLING_RATE_HZ = 1600

def main():
    print("="*80)
    print("STEP 1: DATA EXPLORATION")
    print("="*80)

    # Create output directory
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("\n[1] Loading data...")
    df = load_imu_data(DATA_PATH)

    # Calculate statistics
    print("\n[2] Dataset statistics:")
    calculate_sampling_statistics(df, SAMPLING_RATE_HZ)

    # Basic statistics
    print("\n[3] IMU Data Statistics:")
    print(df.describe())

    # Check for missing values
    print("\n[4] Missing values:")
    print(df.isnull().sum())

    # Check data types
    print("\n[5] Data types:")
    print(df.dtypes)

    # Plot IMU data samples
    print("\n[6] Plotting IMU data samples...")

    # Find a trigger pull event for visualization
    trigger_pull_idx = None
    for i in range(1, len(df)):
        if df.iloc[i]['trigger_state'] == 1 and df.iloc[i-1]['trigger_state'] == 0:
            trigger_pull_idx = i
            break

    if trigger_pull_idx:
        # Plot 2 seconds around trigger pull (3200 samples = 2 seconds at 1600Hz)
        start_idx = max(0, trigger_pull_idx - 1600)
        plot_imu_data(df, start_idx=start_idx, n_samples=3200,
                     save_path=f'{OUTPUT_DIR}/imu_data_sample.png')
        print(f"   Saved: {OUTPUT_DIR}/imu_data_sample.png")

    # Plot random idle period
    idle_idx = 10000  # Some arbitrary index
    plot_imu_data(df, start_idx=idle_idx, n_samples=3200,
                 save_path=f'{OUTPUT_DIR}/imu_data_idle.png')
    print(f"   Saved: {OUTPUT_DIR}/imu_data_idle.png")

    # Temporal analysis
    print("\n[7] Temporal analysis of trigger pulls...")
    plot_temporal_analysis(df, save_path=f'{OUTPUT_DIR}/temporal_analysis.png')
    print(f"   Saved: {OUTPUT_DIR}/temporal_analysis.png")

    # Distribution plots
    print("\n[8] Creating distribution plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Accelerometer distributions
    axes[0, 0].hist(df['accel_x'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Accel X Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')

    axes[0, 1].hist(df['accel_y'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Accel Y Distribution')
    axes[0, 1].set_xlabel('Value')

    axes[0, 2].hist(df['accel_z'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 2].set_title('Accel Z Distribution')
    axes[0, 2].set_xlabel('Value')

    # Gyroscope distributions
    axes[1, 0].hist(df['gyro_x'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 0].set_title('Gyro X Distribution')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')

    axes[1, 1].hist(df['gyro_y'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 1].set_title('Gyro Y Distribution')
    axes[1, 1].set_xlabel('Value')

    axes[1, 2].hist(df['gyro_z'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1, 2].set_title('Gyro Z Distribution')
    axes[1, 2].set_xlabel('Value')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/imu_distributions.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/imu_distributions.png")
    plt.close()

    # Correlation matrix
    print("\n[9] Computing correlation matrix...")
    corr_matrix = df[['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('IMU Sensor Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/correlation_matrix.png")
    plt.close()

    # Magnitude analysis
    print("\n[10] Analyzing sensor magnitudes...")
    accel_mag = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
    gyro_mag = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(accel_mag, bins=50, edgecolor='black', alpha=0.7, color='blue')
    axes[0].set_title(f'Accelerometer Magnitude Distribution\nMean: {accel_mag.mean():.1f}, Std: {accel_mag.std():.1f}')
    axes[0].set_xlabel('Magnitude')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(gyro_mag, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_title(f'Gyroscope Magnitude Distribution\nMean: {gyro_mag.mean():.1f}, Std: {gyro_mag.std():.1f}')
    axes[1].set_xlabel('Magnitude')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/magnitude_distributions.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/magnitude_distributions.png")
    plt.close()

    # Trigger vs non-trigger comparison
    print("\n[11] Comparing trigger vs non-trigger samples...")
    trigger_samples = df[df['trigger_state'] == 1]
    non_trigger_samples = df[df['trigger_state'] == 0].sample(n=min(len(trigger_samples), 10000))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, col in enumerate(['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']):
        row = idx // 3
        col_idx = idx % 3

        axes[row, col_idx].hist(non_trigger_samples[col], bins=50, alpha=0.5,
                               label='Non-Trigger', color='blue', density=True)
        axes[row, col_idx].hist(trigger_samples[col], bins=50, alpha=0.5,
                               label='Trigger', color='red', density=True)
        axes[row, col_idx].set_title(f'{col} Distribution')
        axes[row, col_idx].set_xlabel('Value')
        axes[row, col_idx].set_ylabel('Density')
        axes[row, col_idx].legend()
        axes[row, col_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/trigger_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/trigger_comparison.png")
    plt.close()

    print("\n" + "="*80)
    print("DATA EXPLORATION COMPLETE")
    print("="*80)
    print(f"\nAll plots saved to: {OUTPUT_DIR}/")
    print("\nKey findings:")
    print(f"  - Total samples: {len(df):,}")
    print(f"  - Trigger pulls: {df['trigger_state'].sum():,} ({df['trigger_state'].sum()/len(df)*100:.2f}%)")
    print(f"  - Sampling rate: {SAMPLING_RATE_HZ} Hz")
    print(f"  - Duration: {len(df)/SAMPLING_RATE_HZ:.1f} seconds")
    print("\nNext step: Run 2_preprocessing.py to create temporal labels")


if __name__ == '__main__':
    main()
