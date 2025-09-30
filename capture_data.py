import serial
import serial.tools.list_ports
import csv

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
        print("Waiting for data stream to start... (Use your other terminal to send a character now)")

        # Open the CSV file for writing
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write the header row
            header = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'trigger_state']
            writer.writerow(header)
            
            print(f"Saving data to {OUTPUT_FILE}")
            print("Press Ctrl+C to stop logging.")

            while True:
                # Read a line from the serial port
                line = ser.readline().decode('utf-8').strip()

                # If the line is not empty, process it
                if line:
                    # Print to console for real-time feedback, but don't overwhelm it
                    if "---" in line:
                         print(line) # Print the "Started" message
                    
                    # Check if it's a data line (contains a semicolon) and write to file
                    if ';' in line:
                        # The firmware sends semicolon-separated values. We split them and write.
                        data_points = line.split(';')
                        if len(data_points) == 7: # Ensure it's a valid data line
                            writer.writerow(data_points)
                        else:
                            print(f"Skipping malformed line: {line}")


    except serial.SerialException as e:
        print(f"\n--- SERIAL ERROR ---")
        print(f"Error opening or reading from port {port}: {e}")
        print("Is the blaster connected? Is the port correct? Is another program (like PlatformIO's monitor) using it?")
    except KeyboardInterrupt:
        print("\n\nLogging stopped by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed.")
        print(f"Data saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    capture_data()
