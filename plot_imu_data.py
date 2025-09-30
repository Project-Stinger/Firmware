import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def analyze_and_plot_data():
    """
    Loads, analyzes, and plots the captured IMU and trigger data.
    Highlights the regions where the trigger is active.
    """
    input_file = 'nerf_imu_data.csv'
    
    try:
        # Define the column names based on our new firmware function
        column_names = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'trigger_state']
        
        # Load the data, using a comma as the separator.
        df = pd.read_csv(input_file, header=0, names=column_names, on_bad_lines='skip', low_memory=False)

        # Convert all data to numeric, coercing any errors
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        print(f"Successfully loaded {len(df)} lines of data from '{input_file}'.")

        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
        fig.suptitle('Full IMU Data Stream with Trigger Events', fontsize=18, fontweight='bold')
        fig.patch.set_facecolor('#f0f0f0')

        # Plot 1: Accelerometer Data
        ax1.plot(df.index, df['accel_x'], label='Accel X', alpha=0.9)
        ax1.plot(df.index, df['accel_y'], label='Accel Y', alpha=0.9)
        ax1.plot(df.index, df['accel_z'], label='Accel Z', alpha=0.9)
        ax1.set_ylabel('Accelerometer Reading', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_title('Accelerometer', fontsize=14)
        ax1.set_facecolor('#ffffff')

        # Plot 2: Gyroscope Data
        ax2.plot(df.index, df['gyro_x'], label='Gyro X', alpha=0.9)
        ax2.plot(df.index, df['gyro_y'], label='Gyro Y', alpha=0.9)
        ax2.plot(df.index, df['gyro_z'], label='Gyro Z', alpha=0.9)
        ax2.set_xlabel('Sample Number (Time)', fontsize=12)
        ax2.set_ylabel('Gyroscope Reading', fontsize=12)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_title('Gyroscope', fontsize=14)
        ax2.set_facecolor('#ffffff')
        
        # --- Highlight Trigger Pulls ---
        # Find where the trigger state is 1
        trigger_active = df[df['trigger_state'] == 1]
        
        # Iterate through the axes and add shaded regions for trigger pulls
        for ax in [ax1, ax2]:
            for index in trigger_active.index:
                # Add a vertical shaded bar for each trigger press.
                ax.axvspan(index, index + 1, color='red', alpha=0.3, lw=0)
        
        # *** FIX: Correct way to add a custom legend entry ***
        # Get the existing handles and labels from the first plot
        handles, labels = ax1.get_legend_handles_labels()
        # Create a new patch for our custom legend entry
        trigger_patch = Patch(facecolor='red', alpha=0.3, label='Trigger Pressed')
        # Add the new patch to the list of handles
        handles.append(trigger_patch)
        # Recreate the legend with the combined list
        ax1.legend(handles=handles)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    except FileNotFoundError:
        print(f"Error: '{input_file}' not found.")
        print("Please make sure the data file is in the same directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    analyze_and_plot_data()

