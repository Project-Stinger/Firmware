import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from micromlgen import port
import os

def export_model():
    """
    Loads the full processed dataset, retrains the Random Forest model,
    and then converts it into a C++ header file.
    """
    print("--- ML-Stinger: Part 3 - Model Export to C++ ---")
    
    # --- 1. Load Data ---
    print("\n[Step 1/3] Loading full processed feature data...")
    input_file = 'processed_features.csv'

    if not os.path.exists(input_file):
        print(f"--- ERROR ---")
        print(f"Input file '{input_file}' not found. Please run the data preparation script first.")
        return
    
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} rows of feature data.")
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return

    # --- 2. Retrain Model on ALL Data ---
    print("\n[Step 2/3] Retraining Random Forest model on 100% of the data...")
    X = df.drop('pre_fire_signal', axis=1)
    y = df['pre_fire_signal']
    
    # We use the same settings as before, but now we train on the entire dataset.
    # n_estimators is a key parameter; 30 is a good balance of performance and size.
    final_model = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42)
    final_model.fit(X, y)
    
    print("Final model has been successfully trained.")

    # --- 3. Convert Model to C++ ---
    print("\n[Step 3/3] Converting model to C++ header file...")
    
    try:
        # Use micromlgen to port the trained model to a C++ class
        cpp_code = port(final_model)
        
        # Save the generated code to a header file
        output_file = 'model.h'
        with open(output_file, 'w') as f:
            f.write(cpp_code)
            
        print("-" * 50)
        print(f"Success! Model exported to '{output_file}'.")
        print("\nNext steps:")
        print("1. Copy the 'model.h' file into your firmware's 'src' folder.")
        print("2. Update your 'main.cpp' file with the logic from the implementation plan.")
        print("-" * 50)
        
    except Exception as e:
        print(f"An error occurred during model conversion: {e}")
        print("You may need to install the conversion library with: pip3 install micromlgen")

if __name__ == '__main__':
    export_model()
