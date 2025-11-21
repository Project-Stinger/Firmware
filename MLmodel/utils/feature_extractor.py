"""
Feature extraction matching the C++ implementation in ml_predictor.cpp

Extracts 42 features from IMU data using a sliding window approach.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm


class FeatureExtractor:
    """
    Extract features from IMU data matching the C++ implementation.

    Features (42 total):
    - Basic statistics (24): mean, std, min, max for each of 6 axes
    - Derivative features (12): mean and std of velocity for each of 6 axes
    - Magnitude features (6): mean, std, max for accel and gyro magnitudes
    """

    def __init__(self, window_size: int = 50):
        """
        Args:
            window_size: Number of samples in sliding window (default 50, matching C++)
        """
        self.window_size = window_size
        self.feature_names = self._generate_feature_names()

    def _generate_feature_names(self) -> List[str]:
        """Generate feature names matching the order in C++ code."""
        names = []

        # Basic statistics (24 features)
        axes = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        stats = ['mean', 'std', 'min', 'max']
        for axis in axes:
            for stat in stats:
                names.append(f'{axis}_{stat}')

        # Derivative features (12 features)
        for axis in axes:
            names.append(f'{axis}_diff_mean')
            names.append(f'{axis}_diff_std')

        # Magnitude features (6 features)
        names.extend(['accel_mag_mean', 'accel_mag_std', 'accel_mag_max'])
        names.extend(['gyro_mag_mean', 'gyro_mag_std', 'gyro_mag_max'])

        assert len(names) == 42, f"Expected 42 features, got {len(names)}"
        return names

    def extract_features_single_window(self, window_data: pd.DataFrame) -> np.ndarray:
        """
        Extract 42 features from a single window of IMU data.

        Args:
            window_data: DataFrame with window_size rows containing IMU data

        Returns:
            Array of 42 features
        """
        if len(window_data) != self.window_size:
            raise ValueError(f"Window must have {self.window_size} samples, got {len(window_data)}")

        features = np.zeros(42)
        idx = 0

        # Basic statistics (24 features)
        axes = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        for axis in axes:
            data = window_data[axis].values
            features[idx] = np.mean(data)      # mean
            features[idx+1] = np.std(data)     # std
            features[idx+2] = np.min(data)     # min
            features[idx+3] = np.max(data)     # max
            idx += 4

        # Derivative features (12 features)
        # Using last 10 samples for velocity calculation (matching C++)
        derivative_window = 10
        for axis in axes:
            data = window_data[axis].values
            if len(data) >= derivative_window:
                diffs = np.diff(data[-derivative_window:])
                features[idx] = np.mean(diffs)     # diff_mean
                features[idx+1] = np.std(diffs)    # diff_std
            else:
                features[idx] = 0.0
                features[idx+1] = 0.0
            idx += 2

        # Magnitude features (6 features)
        # Accelerometer magnitude
        accel_x = window_data['accel_x'].values
        accel_y = window_data['accel_y'].values
        accel_z = window_data['accel_z'].values
        accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

        features[idx] = np.mean(accel_mag)
        features[idx+1] = np.std(accel_mag)
        features[idx+2] = np.max(accel_mag)
        idx += 3

        # Gyroscope magnitude
        gyro_x = window_data['gyro_x'].values
        gyro_y = window_data['gyro_y'].values
        gyro_z = window_data['gyro_z'].values
        gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

        features[idx] = np.mean(gyro_mag)
        features[idx+1] = np.std(gyro_mag)
        features[idx+2] = np.max(gyro_mag)

        return features

    def extract_features_from_dataframe(self, df: pd.DataFrame,
                                       stride: int = 1,
                                       show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from entire DataFrame using sliding window.

        Args:
            df: DataFrame with IMU data and 'pre_fire_label' column
            stride: Sliding window stride (1 = every sample)
            show_progress: Show progress bar

        Returns:
            Tuple of (features_array, labels_array)
        """
        n_samples = len(df) - self.window_size + 1
        n_samples = (n_samples + stride - 1) // stride  # Account for stride

        features = np.zeros((n_samples, 42))
        labels = np.zeros(n_samples)

        sample_idx = 0
        iterator = range(0, len(df) - self.window_size + 1, stride)

        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features")

        for i in iterator:
            window = df.iloc[i:i+self.window_size]
            features[sample_idx] = self.extract_features_single_window(window)

            # Label is the label of the LAST sample in the window
            # (this matches the C++ implementation where features are calculated
            # from the past 50 samples and used to predict the current state)
            labels[sample_idx] = window['pre_fire_label'].iloc[-1]
            sample_idx += 1

        return features[:sample_idx], labels[:sample_idx]


def extract_features(df: pd.DataFrame,
                    window_size: int = 50,
                    stride: int = 1) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Convenience function to extract features and return as DataFrame.

    Args:
        df: DataFrame with IMU data and labels
        window_size: Window size for feature extraction
        stride: Sliding window stride

    Returns:
        Tuple of (features_df, labels_array)
    """
    extractor = FeatureExtractor(window_size=window_size)
    features, labels = extractor.extract_features_from_dataframe(df, stride=stride)

    features_df = pd.DataFrame(features, columns=extractor.feature_names)

    print(f"\nFeature extraction complete:")
    print(f"  Features shape: {features.shape}")
    print(f"  Feature names: {len(extractor.feature_names)}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Positive samples: {labels.sum()} ({labels.sum()/len(labels)*100:.2f}%)")

    return features_df, labels


def analyze_feature_importance(features_df: pd.DataFrame, labels: np.ndarray):
    """
    Analyze feature statistics and correlations.

    Args:
        features_df: DataFrame with extracted features
        labels: Array of labels
    """
    print("\nFeature Statistics:")
    print("=" * 80)

    # Basic statistics
    print(features_df.describe())

    # Check for NaN or Inf values
    nan_count = features_df.isna().sum().sum()
    inf_count = np.isinf(features_df.values).sum()

    if nan_count > 0:
        print(f"\nWARNING: {nan_count} NaN values found in features!")
    if inf_count > 0:
        print(f"\nWARNING: {inf_count} Inf values found in features!")

    # Feature ranges
    print("\nFeature Ranges:")
    for col in features_df.columns:
        min_val = features_df[col].min()
        max_val = features_df[col].max()
        print(f"  {col:30s}: [{min_val:12.2f}, {max_val:12.2f}]")

    # Correlation with label
    print("\nTop 10 Features by Correlation with Label:")
    correlations = []
    for col in features_df.columns:
        corr = np.corrcoef(features_df[col], labels)[0, 1]
        correlations.append((col, abs(corr)))

    correlations.sort(key=lambda x: x[1], reverse=True)
    for feature, corr in correlations[:10]:
        print(f"  {feature:30s}: {corr:.4f}")
