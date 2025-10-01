import pandas as pd
import numpy as np
import os

def prepare_data():
    """
    Loads the raw IMU data, creates labels for pre-fire events,
    engineers features using a sliding window, and saves the result.
    """
    print("--- ML-Stinger: Part 1 - Data Preparation (100ms Pre-Fire Window) ---")
    
    # --- 1. Load Data ---
    print("\n[Step 1/4] Loading and cleaning raw data...")
    input_file = 'nerf_imu_data.csv'
    
    if not os.path.exists(input_file):
        print(f"--- ERROR ---")
        print(f"Input file '{input_file}' not found. Please make sure you have captured the data and it's in the correct folder.")
        return

    try:
        column_names = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'trigger_state']
        df = pd.read_csv(input_file, header=0, names=column_names, on_bad_lines='skip', low_memory=False)

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        print(f"Successfully loaded {len(df)} lines of data.")

    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return

    # --- 2. Labeling ---
    print("\n[Step 2/4] Labeling pre-fire signals...")
    # *** THIS IS THE VALUE YOU ASKED TO CHANGE ***
    PREDICTION_WINDOW_MS = 500
    SENSOR_HZ = 1000 # This should match the firmware's loop rate
    window_samples = int(SENSOR_HZ * (PREDICTION_WINDOW_MS / 1000.0))
    print(f"Using a pre-fire prediction window of {PREDICTION_WINDOW_MS} ms ({window_samples} samples).")

    df['pre_fire_signal'] = 0
    trigger_pull_indices = df.index[(df['trigger_state'] == 1) & (df['trigger_state'].shift(1) == 0)]
    print(f"Found {len(trigger_pull_indices)} trigger pull events.")

    for idx in trigger_pull_indices:
        start_index = max(0, idx - window_samples)
        df.loc[start_index:idx, 'pre_fire_signal'] = 1

    print(f"Labeled {df['pre_fire_signal'].sum()} total samples as 'pre_fire_signal'.")

    # --- 3. Feature Engineering ---
    print("\n[Step 3/4] Engineering features with a sliding window...")
    FEATURE_WINDOW_MS = 100
    feature_window_samples = int(SENSOR_HZ * (FEATURE_WINDOW_MS / 1000.0))
    print(f"Using a feature window of {FEATURE_WINDOW_MS} ms ({feature_window_samples} samples).")

    sensor_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    rolling_features = df[sensor_cols].rolling(window=feature_window_samples)
    df_features = pd.DataFrame()

    print("Calculating rolling mean, std, min, and max for each sensor axis...")
    for col in sensor_cols:
        df_features[f'{col}_mean'] = rolling_features[col].mean()
        df_features[f'{col}_std'] = rolling_features[col].std()
        df_features[f'{col}_min'] = rolling_features[col].min()
        df_features[f'{col}_max'] = rolling_features[col].max()

    df_features['pre_fire_signal'] = df['pre_fire_signal']
    df_features.dropna(inplace=True)
    print(f"Created features DataFrame with shape: {df_features.shape}")

    # --- 4. Save Data ---
    print("\n[Step 4/4] Saving processed data...")
    output_file = 'processed_features.csv'
    df_features.to_csv(output_file, index=False)
    
    print("-" * 50)
    print(f"Success! Processed data saved to '{output_file}'.")
    print("\nThis file is now ready for training your model.")
    print("-" * 50)

if __name__ == '__main__':
    prepare_data()

