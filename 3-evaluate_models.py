import pandas as pd
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

# --- Configuration ---
SAMPLE_RATE_HZ = 1200  # Your IMU sample rate

# --- Load Data ---
print("Loading test data...")
# RF uses Raw data
X_test_raw = pd.read_csv('X_test_raw.csv')
# MLP uses Scaled data
X_test_scaled = pd.read_csv('X_test_scaled.csv')
y_test = pd.read_csv('y_test.csv').squeeze()

# --- Define MLP Class (Must match training exactly) ---
class SimpleMLP(nn.Module):
    def __init__(self, num_features):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(num_features, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.bn1(self.layer1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.layer2(x))))
        x = self.layer3(x)
        return x

# --- Load Models ---
print("Loading models...")
rf_model = joblib.load('random_forest_model.joblib')

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
num_features = X_test_scaled.shape[1]
mlp_model = SimpleMLP(num_features=num_features).to(device)
mlp_model.load_state_dict(torch.load('pytorch_mlp_model.pth', map_location=device))
mlp_model.eval()

# --- Generate Predictions ---
print("Running Inference...")
rf_probs = rf_model.predict_proba(X_test_raw)[:, 1]
rf_raw_pred = (rf_probs > 0.5).astype(int)

X_tensor = torch.tensor(X_test_scaled.values.astype(np.float32)).to(device)
with torch.no_grad():
    mlp_logits = mlp_model(X_tensor)
    mlp_probs = torch.sigmoid(mlp_logits).cpu().numpy().flatten()
mlp_raw_pred = (mlp_probs > 0.5).astype(int)

# --- Debounce Analysis Function ---
def analyze_debounce(model_name, raw_predictions, y_true):
    print(f"\n====== {model_name} Performance by Debounce Window ======")
    print(f"{'Window':<10} | {'Recall':<10} | {'Precision':<10} | {'FP / sec':<10} | {'Est. Feel':<20}")
    print("-" * 80)

    windows = [1, 5, 20, 50, 100, 200, 300, 400, 500]
    
    for win in windows:
        # Simulate C++ Debounce Logic
        debounced_pred = np.zeros_like(raw_predictions)
        consecutive_count = 0
        
        for i in range(len(raw_predictions)):
            if raw_predictions[i] == 1:
                consecutive_count += 1
            else:
                consecutive_count = 0
            
            # If we hit the threshold, output 1
            if consecutive_count >= win:
                debounced_pred[i] = 1
        
        # Calculate Metrics
        # FP / sec = Total False Positives / Total Seconds in Test Set
        tn, fp, fn, tp = confusion_matrix(y_true, debounced_pred).ravel()
        recall = recall_score(y_true, debounced_pred, zero_division=0)
        precision = precision_score(y_true, debounced_pred, zero_division=0)
        
        total_seconds = len(y_true) / SAMPLE_RATE_HZ
        fp_per_sec = fp / total_seconds
        
        feel = "Glitchy"
        if fp_per_sec < 50: feel = "Stuttery"
        if fp_per_sec < 10: feel = "Occasional Chirp"
        if fp_per_sec < 1:  feel = "Solid"
        if fp_per_sec == 0: feel = "Perfect Silence"

        print(f"{win:<10} | {recall:<10.4f} | {precision:<10.4f} | {fp_per_sec:<10.2f} | {feel:<20}")

# --- Run Analysis ---
analyze_debounce("Random Forest", rf_raw_pred, y_test)
analyze_debounce("PyTorch MLP", mlp_raw_pred, y_test)