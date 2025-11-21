import pandas as pd
import numpy as np

# Load the raw data
df = pd.read_csv('nerf_imu_data.csv')

# --- Feature Engineering ---
# Create a rolling window
window_size = 50
imu_columns = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
rolling_window = df[imu_columns].rolling(window=window_size)

# Calculate rolling statistics
rolling_mean = rolling_window.mean().add_suffix('_mean')
rolling_std = rolling_window.std().add_suffix('_std')
rolling_min = rolling_window.min().add_suffix('_min')
rolling_max = rolling_window.max().add_suffix('_max')
# Calculate rate of change (derivative)
rolling_deriv = df[imu_columns].diff().rolling(window=window_size).mean().add_suffix('_deriv')

# Combine the features into a new DataFrame
features_df = pd.concat([rolling_mean, rolling_std, rolling_min, rolling_max, rolling_deriv], axis=1)
df = pd.concat([df, features_df], axis=1)

# Drop rows with NaN values created by the rolling window and diff
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# --- Labeling (Full Window Strategy) ---
df['pre_fire'] = 0

# Find where the trigger is pulled (0 to 1 transition)
trigger_pull_indices = df.index[ (df['trigger_state'].shift(1) == 0) & (df['trigger_state'] == 1) ].tolist()

# Define the pre-fire window in samples (1200Hz sample rate)
pre_fire_window_start = 360  # 300ms
pre_fire_window_end = 120    # 100ms

# Label the entire pre-fire window
for idx in trigger_pull_indices:
    start_label_idx = max(0, idx - pre_fire_window_start)
    end_label_idx = idx - pre_fire_window_end
    if start_label_idx < end_label_idx:
        df.loc[start_label_idx:end_label_idx, 'pre_fire'] = 1

# --- Save Processed Data ---
feature_cols = [col for col in df.columns if any(suffix in col for suffix in ['_mean', '_std', '_min', '_max', '_deriv'])]
output_columns = ['pre_fire'] + feature_cols
processed_df = df[output_columns]

processed_df.to_csv('processed_nerf_imu_data.csv', index=False)

print("Data preparation complete.")
print(f"Total samples: {len(processed_df)}")