#!/usr/bin/env python3
"""
Step 6: Train Neural Network Model (PyTorch)

Best performance model (not deployable on RP2040, for comparison only).
"""

import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

# Add utils to path
sys.path.insert(0, '.')

from utils.metrics import (
    calculate_comprehensive_metrics,
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    find_optimal_threshold
)

# Configuration
FEATURES_DIR = 'outputs/features'
OUTPUT_DIR = 'outputs/models/neural_network'
SAMPLING_RATE_HZ = 1600

# Neural Network configuration
NN_CONFIG = {
    'hidden_layers': [128, 64, 32],  # 3 hidden layers
    'dropout': 0.3,
    'batch_size': 256,
    'learning_rate': 0.001,
    'epochs': 50,
    'early_stopping_patience': 10,
    'weight_decay': 1e-4
}

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class NeuralNetworkClassifier(nn.Module):
    """
    Feed-forward neural network for binary classification.
    """

    def __init__(self, input_size, hidden_layers, dropout=0.3):
        super(NeuralNetworkClassifier, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.unsqueeze(1).float()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        predicted = (outputs >= 0.5).float()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.unsqueeze(1).float()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def predict(model, X, device, batch_size=256):
    """Make predictions on data."""
    model.eval()
    predictions = []

    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())

    return np.concatenate(predictions).squeeze()


def main():
    print("="*80)
    print("STEP 6: TRAIN NEURAL NETWORK MODEL (PyTorch)")
    print("="*80)
    print(f"Using device: {device}")

    # Create output directory
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load features
    print("\n[1] Loading features...")
    X_train = np.load(f'{FEATURES_DIR}/X_train.npy')
    y_train = np.load(f'{FEATURES_DIR}/y_train.npy')
    X_val = np.load(f'{FEATURES_DIR}/X_val.npy')
    y_val = np.load(f'{FEATURES_DIR}/y_val.npy')
    X_test = np.load(f'{FEATURES_DIR}/X_test.npy')
    y_test = np.load(f'{FEATURES_DIR}/y_test.npy')

    print(f"   Train: {X_train.shape}")
    print(f"   Val:   {X_val.shape}")
    print(f"   Test:  {X_test.shape}")

    # Handle class imbalance with weighted loss
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\n   Class imbalance ratio: {pos_weight:.2f}:1")
    print(f"   Using BCELoss with pos_weight={pos_weight:.2f}")

    # Create data loaders
    print("\n[2] Creating data loaders...")
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=NN_CONFIG['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=NN_CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Create model
    print("\n[3] Creating neural network...")
    input_size = X_train.shape[1]
    model = NeuralNetworkClassifier(
        input_size=input_size,
        hidden_layers=NN_CONFIG['hidden_layers'],
        dropout=NN_CONFIG['dropout']
    ).to(device)

    print(f"   Architecture:")
    print(f"      Input:  {input_size} features")
    for i, hidden_size in enumerate(NN_CONFIG['hidden_layers']):
        print(f"      Hidden {i+1}: {hidden_size} units (ReLU + BatchNorm + Dropout)")
    print(f"      Output: 1 unit (Sigmoid)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Estimated model size: {total_params * 4 / 1024:.2f} KB (float32)")

    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = optim.Adam(
        model.parameters(),
        lr=NN_CONFIG['learning_rate'],
        weight_decay=NN_CONFIG['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print(f"\n[4] Training for {NN_CONFIG['epochs']} epochs...")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(NN_CONFIG['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        print(f"   Epoch {epoch+1:3d}/{NN_CONFIG['epochs']}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'{OUTPUT_DIR}/best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= NN_CONFIG['early_stopping_patience']:
            print(f"\n   Early stopping at epoch {epoch+1}")
            break

    # Load best model
    print("\n[5] Loading best model...")
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/best_model.pth'))

    # Plot training history
    print("\n[6] Plotting training history...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_accs, label='Train Accuracy')
    axes[1].plot(val_accs, label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/training_history.png', dpi=300, bbox_inches='tight')
    print(f"   Saved: {OUTPUT_DIR}/training_history.png")
    plt.close()

    # Evaluate on validation set
    print("\n[7] Evaluating on validation set...")
    y_val_prob = predict(model, X_val, device)
    y_val_pred = (y_val_prob >= 0.5).astype(int)

    val_metrics = calculate_comprehensive_metrics(
        y_val, y_val_pred, y_val_prob, sampling_rate_hz=SAMPLING_RATE_HZ
    )
    print_metrics(val_metrics, "Neural Network (Validation)")

    # Find optimal threshold
    optimal_threshold, threshold_metrics = find_optimal_threshold(
        y_val, y_val_prob, target_fp_per_second=1.0, sampling_rate_hz=SAMPLING_RATE_HZ
    )

    # Evaluate on test set
    print("\n[8] Evaluating on test set...")
    y_test_prob = predict(model, X_test, device)
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    # Default threshold
    test_metrics = calculate_comprehensive_metrics(
        y_test, y_test_pred, y_test_prob, sampling_rate_hz=SAMPLING_RATE_HZ
    )
    print_metrics(test_metrics, "Neural Network (Test, threshold=0.5)")

    # Optimal threshold
    y_test_pred_opt = (y_test_prob >= optimal_threshold).astype(int)

    test_metrics_opt = calculate_comprehensive_metrics(
        y_test, y_test_pred_opt, y_test_prob, sampling_rate_hz=SAMPLING_RATE_HZ
    )
    print_metrics(test_metrics_opt, f"Neural Network (Test, threshold={optimal_threshold:.3f})")

    # Classification report
    print("\n[9] Detailed classification report:")
    print(classification_report(y_test, y_test_pred_opt, target_names=['No Pre-Fire', 'Pre-Fire']))

    # Visualizations
    print("\n[10] Creating visualizations...")

    plot_confusion_matrix(y_test, y_test_pred_opt,
                         title=f'Neural Network - Confusion Matrix (threshold={optimal_threshold:.3f})',
                         save_path=f'{OUTPUT_DIR}/confusion_matrix.png')

    plot_roc_curve(y_test, y_test_prob,
                  title='Neural Network - ROC Curve',
                  save_path=f'{OUTPUT_DIR}/roc_curve.png')

    plot_precision_recall_curve(y_test, y_test_prob,
                                title='Neural Network - Precision-Recall Curve',
                                save_path=f'{OUTPUT_DIR}/pr_curve.png')

    # Save model and results
    print("\n[11] Saving model and results...")

    torch.save({
        'model_state_dict': model.state_dict(),
        'config': NN_CONFIG,
        'input_size': input_size
    }, f'{OUTPUT_DIR}/model.pth')
    print(f"   Saved: {OUTPUT_DIR}/model.pth")

    results = {
        'config': NN_CONFIG,
        'optimal_threshold': optimal_threshold,
        'total_parameters': total_params,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        },
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'test_metrics_optimal': test_metrics_opt,
        'predictions': {
            'y_test': y_test,
            'y_test_pred': y_test_pred_opt,
            'y_test_prob': y_test_prob
        }
    }

    joblib.dump(results, f'{OUTPUT_DIR}/results.pkl')
    print(f"   Saved: {OUTPUT_DIR}/results.pkl")

    # Export to ONNX (for potential optimization)
    print("\n[12] Exporting to ONNX format...")
    try:
        dummy_input = torch.randn(1, input_size).to(device)
        torch.onnx.export(
            model,
            dummy_input,
            f'{OUTPUT_DIR}/model.onnx',
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
        print(f"   Saved: {OUTPUT_DIR}/model.onnx")
    except Exception as e:
        print(f"   Warning: Failed to export ONNX: {e}")

    # Print summary
    print("\n" + "="*80)
    print("NEURAL NETWORK TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel Architecture:")
    print(f"  Hidden layers: {NN_CONFIG['hidden_layers']}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Training epochs: {len(train_losses)}")

    print(f"\nTest Set Performance (with optimal threshold={optimal_threshold:.3f}):")
    print(f"  Accuracy:  {test_metrics_opt['accuracy']:.4f}")
    print(f"  Precision: {test_metrics_opt['precision']:.4f}")
    print(f"  Recall:    {test_metrics_opt['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics_opt['f1']:.4f}")
    print(f"  ROC-AUC:   {test_metrics_opt['roc_auc']:.4f}")
    print(f"\n  FALSE POSITIVES PER SECOND: {test_metrics_opt['fp_per_second']:.2f} FP/s")

    print(f"\nModel saved to: {OUTPUT_DIR}/")
    print("\nNote: This model is NOT deployable on RP2040 due to size and complexity.")
    print("      It serves as a performance benchmark for the deployable Random Forest model.")
    print("\nNext step: Run 7_evaluate_models.py to compare all models")


if __name__ == '__main__':
    main()
