import serial
import serial.tools.list_ports
import csv
import time

def find_serial_port():
    """Finds the serial port for the Raspberry Pi Pico."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # The Pico often shows up with this VID/PID
        if port.vid == 0x2E8A and port.pid == 0x000A:
            print(f"Found Raspberry Pi Pico at: {port.device}")
            return port.device
        # A common fallback for macOS/Linux
        if "usbmodem" in port.device:
            print(f"Found a potential device at: {port.device}")
            return port.device
    return None

def capture_data():
    """Captures IMU and trigger data from the serial port and saves it to a CSV file."""
    
    port = find_serial_port()
    if not port:
        print("\n--- ERROR ---")
        print("Could not automatically find the serial port for the Nerf blaster.")
        print("Please make sure it's plugged in.")
        print("You may need to manually enter the port name below.")
        port = input("Enter the serial port (e.g., COM3 or /dev/cu.usbmodem1234): ")

    # --- Configuration ---
    BAUD_RATE = 115200
    OUTPUT_FILE = 'nerf_imu_data.csv'

    try:
        # Open the serial port
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print(f"Listening on port {port} at {BAUD_RATE} baud...")
        
        # Give the board a moment to initialize
        time.sleep(2)

        # Clear any pending data
        ser.reset_input_buffer()

        # Send command to start ML logging
        print("Activating ML data logging...")
        ser.write(b'1')
        ser.flush()

        # Wait for acknowledgment
        print("Waiting for acknowledgment...")
        ack_received = False
        for _ in range(10):  # Try for up to 5 seconds
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"  Received: {line}")
            if "STARTED" in line:
                ack_received = True
                break
            time.sleep(0.5)

        if not ack_received:
            print("WARNING: Did not receive start acknowledgment. Proceeding anyway...")

        # Open the CSV file for writing
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write the header row
            header = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'trigger_state']
            writer.writerow(header)

            print(f"\nSaving data to {OUTPUT_FILE}")
            print("Press Ctrl+C to stop logging.")

            sample_count = 0  # Track number of samples
            start_time = time.time()  # Track start time for rate calculation
            last_milestone_time = start_time  # Track time of last 1000-sample milestone

            while True:
                # Read a line from the serial port
                line = ser.readline().decode('utf-8', errors='ignore').strip()

                # If the line is not empty, process it
                if line:
                    # Print status messages to console
                    if "---" in line:
                        print(f"\n{line}")
                        continue

                    # Check if it's a data line (comma-separated numbers)
                    if ',' in line and not line.startswith('accel'):
                        # The firmware sends comma-separated values
                        data_points = line.split(',')
                        if len(data_points) == 7:
                            try:
                                # Validate all values are integers
                                [int(x) for x in data_points]
                                writer.writerow(data_points)
                                sample_count += 1

                                # Print a dot every 100 samples
                                if sample_count % 100 == 0:
                                    print(".", end="", flush=True)

                                # Print sample count and rate every 1000 samples
                                if sample_count % 1000 == 0:
                                    current_time = time.time()
                                    elapsed_since_last = current_time - last_milestone_time
                                    instantaneous_rate = 1000 / elapsed_since_last if elapsed_since_last > 0 else 0

                                    total_elapsed = current_time - start_time
                                    average_rate = sample_count / total_elapsed if total_elapsed > 0 else 0

                                    print(f" {sample_count} samples (rate: {instantaneous_rate:.0f} Hz, avg: {average_rate:.0f} Hz)", flush=True)
                                    last_milestone_time = current_time
                            except ValueError:
                                # Not all numeric, skip
                                pass

    except serial.SerialException as e:
        print(f"\n--- SERIAL ERROR ---")
        print(f"Error opening or reading from port {port}: {e}")
        print("Is the blaster connected? Is the port correct? Is another program (like PlatformIO's monitor) using it?")
    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        average_rate = sample_count / elapsed_time if elapsed_time > 0 else 0
        print(f"\n\nLogging stopped by user.")

        # Send stop command to firmware
        if 'ser' in locals() and ser.is_open:
            print("Sending stop command to firmware...")
            ser.write(b'0')
            ser.flush()
            time.sleep(0.5)

            # Read any final messages
            while ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    print(f"  {line}")

        print(f"Captured {sample_count} samples in {elapsed_time:.1f} seconds")
        print(f"Average rate: {average_rate:.1f} samples/sec (target: 1600 Hz)")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")
        print(f"Data saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    capture_data()
