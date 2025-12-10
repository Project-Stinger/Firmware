#!/usr/bin/env python3
"""
Step 4: Train Logistic Regression Model

Baseline model for comparison.
"""

import sys
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
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
OUTPUT_DIR = 'outputs/models/logistic_regression'
SAMPLING_RATE_HZ = 1600


def main():
    print("="*80)
    print("STEP 4: TRAIN LOGISTIC REGRESSION MODEL")
    print("="*80)

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

    # Train model
    print("\n[2] Training Logistic Regression model...")
    print("   Using class_weight='balanced' to handle class imbalance")

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        solver='lbfgs',
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("   Training complete!")

    # Validate on validation set
    print("\n[3] Evaluating on validation set...")
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    val_metrics = calculate_comprehensive_metrics(
        y_val, y_val_pred, y_val_prob, sampling_rate_hz=SAMPLING_RATE_HZ
    )
    print_metrics(val_metrics, "Logistic Regression (Validation)")

    # Find optimal threshold
    optimal_threshold, threshold_metrics = find_optimal_threshold(
        y_val, y_val_prob, target_fp_per_second=1.0, sampling_rate_hz=SAMPLING_RATE_HZ
    )

    # Test on test set with default threshold
    print("\n[4] Evaluating on test set (default threshold=0.5)...")
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    test_metrics = calculate_comprehensive_metrics(
        y_test, y_test_pred, y_test_prob, sampling_rate_hz=SAMPLING_RATE_HZ
    )
    print_metrics(test_metrics, "Logistic Regression (Test)")

    # Test with optimal threshold
    print(f"\n[5] Evaluating on test set with optimal threshold={optimal_threshold:.3f}...")
    y_test_pred_opt = (y_test_prob >= optimal_threshold).astype(int)

    test_metrics_opt = calculate_comprehensive_metrics(
        y_test, y_test_pred_opt, y_test_prob, sampling_rate_hz=SAMPLING_RATE_HZ
    )
    print_metrics(test_metrics_opt, f"Logistic Regression (Test, threshold={optimal_threshold:.3f})")

    # Classification report
    print("\n[6] Detailed classification report:")
    print(classification_report(y_test, y_test_pred_opt, target_names=['No Pre-Fire', 'Pre-Fire']))

    # Visualizations
    print("\n[7] Creating visualizations...")

    plot_confusion_matrix(y_test, y_test_pred_opt,
                         title=f'Logistic Regression - Confusion Matrix (threshold={optimal_threshold:.3f})',
                         save_path=f'{OUTPUT_DIR}/confusion_matrix.png')

    plot_roc_curve(y_test, y_test_prob,
                  title='Logistic Regression - ROC Curve',
                  save_path=f'{OUTPUT_DIR}/roc_curve.png')

    plot_precision_recall_curve(y_test, y_test_prob,
                                title='Logistic Regression - Precision-Recall Curve',
                                save_path=f'{OUTPUT_DIR}/pr_curve.png')

    # Feature importance (coefficients)
    print("\n[8] Analyzing feature importance...")
    feature_names = joblib.load(f'{FEATURES_DIR}/feature_names.pkl')
    coefficients = model.coef_[0]

    # Sort by absolute value
    importance_indices = np.argsort(np.abs(coefficients))[::-1]

    print("   Top 15 most important features:")
    for i, idx in enumerate(importance_indices[:15]):
        print(f"      {i+1:2d}. {feature_names[idx]:30s}: {coefficients[idx]:8.4f}")

    # Save model and results
    print("\n[9] Saving model and results...")

    joblib.dump(model, f'{OUTPUT_DIR}/model.pkl')
    print(f"   Saved: {OUTPUT_DIR}/model.pkl")

    results = {
        'model': model,
        'optimal_threshold': optimal_threshold,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'test_metrics_optimal': test_metrics_opt,
        'feature_importance': {
            'feature_names': feature_names,
            'coefficients': coefficients
        },
        'predictions': {
            'y_test': y_test,
            'y_test_pred': y_test_pred_opt,
            'y_test_prob': y_test_prob
        }
    }

    joblib.dump(results, f'{OUTPUT_DIR}/results.pkl')
    print(f"   Saved: {OUTPUT_DIR}/results.pkl")

    # Print summary
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION TRAINING COMPLETE")
    print("="*80)
    print(f"\nTest Set Performance (with optimal threshold={optimal_threshold:.3f}):")
    print(f"  Accuracy:  {test_metrics_opt['accuracy']:.4f}")
    print(f"  Precision: {test_metrics_opt['precision']:.4f}")
    print(f"  Recall:    {test_metrics_opt['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics_opt['f1']:.4f}")
    print(f"  ROC-AUC:   {test_metrics_opt['roc_auc']:.4f}")
    print(f"\n  FALSE POSITIVES PER SECOND: {test_metrics_opt['fp_per_second']:.2f} FP/s")

    print(f"\nModel saved to: {OUTPUT_DIR}/")
    print("\nNext step: Run 5_train_random_forest.py")


if __name__ == '__main__':
    main()
