import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import joblib
import numpy as np

# Import PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Load Data ---
print("Loading data...")
df = pd.read_csv('processed_nerf_imu_data.csv')
X = df.drop('pre_fire', axis=1).values
y = df['pre_fire'].values

print(f"Total Features: {X.shape[1]}") 
# NOTE: Ensure this number matches your C++ extractFeatures() count exactly!

# --- Chronological Split (Block Split) ---
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
safety_gap = 50 

# Raw splits (for Random Forest)
X_train_raw = X[:split_index]
y_train = y[:split_index]
X_test_raw = X[split_index + safety_gap:]
y_test = y[split_index + safety_gap:]

print(f"Training Samples: {len(X_train_raw)}")
print(f"Testing Samples:  {len(X_test_raw)}")

# --- Feature Scaling (Crucial for MLP, Optional for RF) ---
# We fit the scaler ONLY on training data to avoid leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
# We apply the same scaling to the test set
X_test_scaled = scaler.transform(X_test_raw)

# Save the scaler (You might need its mean/var values for C++ if you deploy MLP)
joblib.dump(scaler, 'scaler.joblib')

# --- Handle Class Imbalance ---
class_weights_sklearn = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_sklearn = {i : class_weights_sklearn[i] for i in range(len(class_weights_sklearn))}

# --- Train Random Forest (Unscaled) ---
print("\nTraining Random Forest (Unscaled)...")
rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=15,        # Memory constraint for RP2040
    min_samples_leaf=4,  # Noise reduction
    random_state=42, 
    class_weight=class_weights_sklearn, 
    n_jobs=-1
)
rf_model.fit(X_train_raw, y_train)
joblib.dump(rf_model, 'random_forest_model.joblib')
print("Random Forest saved.")

# --- Train Neural Network (PyTorch MLP - Scaled) ---
print("\nTraining PyTorch MLP (Scaled)...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Prepare DataLoaders
X_tensor = torch.tensor(X_train_scaled.astype(np.float32)).to(device)
y_tensor = torch.tensor(y_train.astype(np.float32)).to(device)
train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

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

num_features = X.shape[1]
model = SimpleMLP(num_features=num_features).to(device)

pos_weight_val = np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-6)
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        labels = labels.view(-1, 1)
        outputs = model(features)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss: {running_loss/len(train_loader):.4f}')

torch.save(model.state_dict(), 'pytorch_mlp_model.pth')
print("PyTorch MLP saved.")

# --- Save Test Data (Both Raw and Scaled) ---
# We need RAW for RF evaluation and SCALED for MLP evaluation
pd.DataFrame(X_test_raw, columns=df.drop('pre_fire', axis=1).columns).to_csv('X_test_raw.csv', index=False)
pd.DataFrame(X_test_scaled, columns=df.drop('pre_fire', axis=1).columns).to_csv('X_test_scaled.csv', index=False)
pd.DataFrame(y_test, columns=['pre_fire']).to_csv('y_test.csv', index=False)
print("Test data saved.")