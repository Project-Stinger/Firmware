"""
Data loading and temporal labeling utilities
"""

import numpy as np
import pandas as pd
from typing import Tuple


def load_imu_data(filepath: str) -> pd.DataFrame:
    """
    Load IMU data from CSV file.

    Args:
        filepath: Path to CSV file

    Returns:
        DataFrame with columns: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, trigger_state
    """
    df = pd.read_csv(filepath)

    # Verify required columns
    required_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'trigger_state']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print(f"Loaded {len(df)} samples from {filepath}")
    print(f"Trigger pulls: {df['trigger_state'].sum()} samples ({df['trigger_state'].sum() / len(df) * 100:.2f}%)")

    return df


def create_temporal_labels(df: pd.DataFrame,
                           prediction_window_ms: Tuple[int, int] = (100, 400),
                           sampling_rate_hz: int = 1600) -> pd.DataFrame:
    """
    Create temporal labels for pre-fire prediction.

    Labels samples as 1 if they occur 100-400ms BEFORE a trigger pull (transition 0->1).
    This creates the "pre-fire window" that the model should learn to detect.

    Args:
        df: DataFrame with trigger_state column
        prediction_window_ms: Tuple of (min_ms, max_ms) for prediction window
        sampling_rate_hz: IMU sampling rate in Hz

    Returns:
        DataFrame with new 'pre_fire_label' column
    """
    df = df.copy()

    # Calculate samples corresponding to time windows
    min_ms, max_ms = prediction_window_ms
    samples_per_ms = sampling_rate_hz / 1000.0
    min_samples = int(min_ms * samples_per_ms)  # 160 samples for 100ms
    max_samples = int(max_ms * samples_per_ms)  # 640 samples for 400ms

    print(f"\nCreating temporal labels:")
    print(f"  Prediction window: {min_ms}-{max_ms}ms")
    print(f"  Sample window: {min_samples}-{max_samples} samples")
    print(f"  Sampling rate: {sampling_rate_hz}Hz")

    # Initialize pre-fire labels to 0
    df['pre_fire_label'] = 0

    # Find trigger pull events (transition from 0 to 1)
    trigger_pulls = []
    for i in range(1, len(df)):
        if df.iloc[i]['trigger_state'] == 1 and df.iloc[i-1]['trigger_state'] == 0:
            trigger_pulls.append(i)

    print(f"  Found {len(trigger_pulls)} trigger pull events")

    # Label the pre-fire window before each trigger pull
    pre_fire_samples = 0
    for pull_idx in trigger_pulls:
        # Label samples in the window [pull_idx - max_samples, pull_idx - min_samples]
        start_idx = max(0, pull_idx - max_samples)
        end_idx = max(0, pull_idx - min_samples)

        if end_idx > start_idx:
            df.loc[start_idx:end_idx, 'pre_fire_label'] = 1
            pre_fire_samples += (end_idx - start_idx + 1)

    print(f"  Labeled {pre_fire_samples} samples as pre-fire ({pre_fire_samples / len(df) * 100:.2f}%)")
    print(f"  Class distribution: {df['pre_fire_label'].value_counts().to_dict()}")
    print(f"  Class imbalance ratio: {(df['pre_fire_label']==0).sum() / (df['pre_fire_label']==1).sum():.2f}:1")

    return df


def split_data_temporal(df: pd.DataFrame,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test sets using temporal splitting.

    IMPORTANT: We use temporal splitting (not random) to avoid data leakage,
    since consecutive samples are highly correlated in time-series data.

    Args:
        df: DataFrame to split
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    print(f"\nTemporal split:")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/n*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/n*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/n*100:.1f}%)")

    print(f"\n  Train pre-fire: {train_df['pre_fire_label'].sum()} ({train_df['pre_fire_label'].sum()/len(train_df)*100:.2f}%)")
    print(f"  Val pre-fire:   {val_df['pre_fire_label'].sum()} ({val_df['pre_fire_label'].sum()/len(val_df)*100:.2f}%)")
    print(f"  Test pre-fire:  {test_df['pre_fire_label'].sum()} ({test_df['pre_fire_label'].sum()/len(test_df)*100:.2f}%)")

    return train_df, val_df, test_df


def calculate_sampling_statistics(df: pd.DataFrame, sampling_rate_hz: int = 1600):
    """
    Calculate statistics about the data collection.

    Args:
        df: DataFrame with IMU data
        sampling_rate_hz: Sampling rate
    """
    total_samples = len(df)
    total_time_s = total_samples / sampling_rate_hz

    print(f"\nDataset statistics:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total time: {total_time_s:.2f}s ({total_time_s/60:.2f}min)")
    print(f"  Sampling rate: {sampling_rate_hz}Hz")
    print(f"  Sample period: {1000/sampling_rate_hz:.3f}ms")

    # Analyze trigger state
    trigger_samples = (df['trigger_state'] == 1).sum()
    trigger_time_s = trigger_samples / sampling_rate_hz

    print(f"\n  Trigger pulled: {trigger_samples:,} samples ({trigger_samples/total_samples*100:.2f}%)")
    print(f"  Trigger time: {trigger_time_s:.2f}s")

    # Count trigger pull events
    trigger_events = 0
    for i in range(1, len(df)):
        if df.iloc[i]['trigger_state'] == 1 and df.iloc[i-1]['trigger_state'] == 0:
            trigger_events += 1

    print(f"  Trigger events: {trigger_events}")
    if trigger_events > 0:
        avg_event_duration_ms = (trigger_time_s / trigger_events) * 1000
        print(f"  Avg trigger duration: {avg_event_duration_ms:.1f}ms")
