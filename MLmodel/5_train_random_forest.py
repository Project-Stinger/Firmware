#!/usr/bin/env python3
"""
Step 5: Train Random Forest Model

Optimized for deployment on RP2040 MCU with size constraints.
"""

import sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score, precision_score
import matplotlib.pyplot as plt

# Add utils to path
sys.path.insert(0, '.')

from utils.metrics import (
    calculate_comprehensive_metrics,
    calculate_false_positives_per_second,
    print_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    find_optimal_threshold,
    analyze_consecutive_predictions
)
from utils.visualization import plot_feature_importance

# Configuration
FEATURES_DIR = 'outputs/features'
OUTPUT_DIR = 'outputs/models/random_forest'
SAMPLING_RATE_HZ = 1600

# Random Forest hyperparameters optimized for MCU deployment
# Goal: Keep model small enough for RP2040 (264KB RAM)
RF_CONFIG = {
    'n_estimators': 10,        # Small number of trees (can tune: try 5, 10, 15, 20)
    'max_depth': 8,            # Limit tree depth (can tune: try 5, 8, 10)
    'min_samples_split': 20,   # Prevent overfitting
    'min_samples_leaf': 10,    # Prevent overfitting
    'max_features': 'sqrt',    # Reduce tree complexity
    'class_weight': 'balanced', # Handle class imbalance
    'random_state': 42,
    'n_jobs': -1
}


def estimate_model_size(model):
    """
    Estimate memory footprint of Random Forest model.

    Args:
        model: Trained RandomForestClassifier

    Returns:
        Dictionary with size estimates
    """
    total_nodes = 0
    total_leaves = 0

    for tree in model.estimators_:
        tree_structure = tree.tree_
        total_nodes += tree_structure.node_count
        total_leaves += tree_structure.n_leaves

    # Rough estimate: each node needs ~32 bytes (feature_idx, threshold, left_child, right_child, etc.)
    # Each leaf needs ~16 bytes (class probabilities)
    estimated_bytes = (total_nodes * 32) + (total_leaves * 16)

    return {
        'n_trees': len(model.estimators_),
        'total_nodes': total_nodes,
        'total_leaves': total_leaves,
        'avg_nodes_per_tree': total_nodes / len(model.estimators_),
        'avg_leaves_per_tree': total_leaves / len(model.estimators_),
        'estimated_size_kb': estimated_bytes / 1024,
        'estimated_size_bytes': estimated_bytes
    }


def main():
    print("="*80)
    print("STEP 5: TRAIN RANDOM FOREST MODEL (MCU-Optimized)")
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
    print("\n[2] Training Random Forest model...")
    print(f"   Configuration:")
    for key, value in RF_CONFIG.items():
        if key != 'n_jobs':
            print(f"      {key}: {value}")

    model = RandomForestClassifier(**RF_CONFIG)

    model.fit(X_train, y_train)
    print("   Training complete!")

    # Estimate model size
    print("\n[3] Estimating model size for MCU deployment...")
    size_info = estimate_model_size(model)

    print(f"   Model Size Analysis:")
    print(f"      Number of trees:        {size_info['n_trees']}")
    print(f"      Total nodes:            {size_info['total_nodes']:,}")
    print(f"      Total leaves:           {size_info['total_leaves']:,}")
    print(f"      Avg nodes per tree:     {size_info['avg_nodes_per_tree']:.1f}")
    print(f"      Avg leaves per tree:    {size_info['avg_leaves_per_tree']:.1f}")
    print(f"      Estimated size:         {size_info['estimated_size_kb']:.2f} KB")

    if size_info['estimated_size_kb'] < 100:
        print(f"   ✓ Model size is GOOD for RP2040 deployment (< 100 KB)")
    elif size_info['estimated_size_kb'] < 200:
        print(f"   ⚠ Model size is ACCEPTABLE for RP2040 (< 200 KB)")
    else:
        print(f"   ✗ Model size may be TOO LARGE for RP2040 (> 200 KB)")
        print(f"   Consider reducing n_estimators or max_depth")

    # Validate on validation set
    print("\n[4] Evaluating on validation set...")
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]

    val_metrics = calculate_comprehensive_metrics(
        y_val, y_val_pred, y_val_prob, sampling_rate_hz=SAMPLING_RATE_HZ
    )
    print_metrics(val_metrics, "Random Forest (Validation)")

    # Find optimal threshold using F-beta (favors recall) + consecutive filtering
    print("\n[5] Finding optimal threshold with EVENT-BASED optimization...")
    print("   Strategy: Maximize F2-score (weights recall 2x) with 20 consecutive predictions")
    print("   Constraint: Catch 80% of trigger EVENTS (not samples!)")
    print("   Constraint: Maximum 5 false alarm EVENTS per second (not FP samples!)")

    from utils.metrics import find_optimal_threshold_fbeta

    optimal_threshold, threshold_metrics = find_optimal_threshold_fbeta(
        y_val, y_val_prob,
        min_consecutive=20,  # Simulate C++ filtering
        sampling_rate_hz=SAMPLING_RATE_HZ,
        beta=2.0,  # Favor recall 2x over precision
        min_event_recall=0.8,  # Must catch at least 80% of trigger EVENTS
        max_false_alarms_per_second=5.0  # Maximum 5 false alarm EVENTS per second
    )

    print(f"\n   Optimal Threshold (with consecutive filtering):")
    print(f"      Threshold:         {optimal_threshold:.3f}")
    print(f"      F2-Score:          {threshold_metrics['fbeta']:.3f}")
    print(f"      Event Recall:      {threshold_metrics['event_recall']:.3f} ({threshold_metrics['detected_events']}/{threshold_metrics['total_events']} events)")
    print(f"      Sample Recall:     {threshold_metrics['recall']:.3f}")
    print(f"      Precision:         {threshold_metrics['precision']:.3f}")
    print(f"      False Alarms/s:    {threshold_metrics['false_alarms_per_second']:.2f} events/s ({threshold_metrics['false_alarm_events']} events)")
    print(f"      FP Samples/s:      {threshold_metrics['fp_per_second']:.2f}")

    # Test on test set
    print("\n[6] Evaluating on test set...")
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Default threshold
    test_metrics = calculate_comprehensive_metrics(
        y_test, y_test_pred, y_test_prob, sampling_rate_hz=SAMPLING_RATE_HZ
    )
    print_metrics(test_metrics, "Random Forest (Test, threshold=0.5, no filtering)")

    # Optimal threshold WITH consecutive filtering
    y_test_pred_raw = (y_test_prob >= optimal_threshold).astype(int)

    # Apply consecutive filtering (matching validation)
    y_test_pred_opt = np.zeros_like(y_test_pred_raw)
    consecutive_count = 0
    for i in range(len(y_test_pred_raw)):
        if y_test_pred_raw[i] == 1:
            consecutive_count += 1
            if consecutive_count >= 20:
                y_test_pred_opt[i] = 1
        else:
            consecutive_count = 0

    test_metrics_opt = calculate_comprehensive_metrics(
        y_test, y_test_pred_opt, y_test_prob, sampling_rate_hz=SAMPLING_RATE_HZ
    )
    print_metrics(test_metrics_opt, f"Random Forest (Test, threshold={optimal_threshold:.3f}, 20 consecutive)")

    # Add event-based metrics display
    from utils.metrics import calculate_event_based_metrics
    test_event_metrics = calculate_event_based_metrics(y_test, y_test_pred_opt, SAMPLING_RATE_HZ)
    print(f"\n   EVENT-BASED METRICS:")
    print(f"      Events Detected:   {test_event_metrics['detected_events']}/{test_event_metrics['total_events']} ({test_event_metrics['event_recall']:.1%})")
    print(f"      False Alarms:      {test_event_metrics['false_alarm_events']} events ({test_event_metrics['false_alarms_per_second']:.2f} events/s)")

    # Try different consecutive counts
    print("\n[7] Testing different consecutive prediction requirements...")
    for n_consecutive in [10, 15, 20, 25]:
        y_pred_filtered = np.zeros_like(y_test_pred_raw)
        consecutive_count = 0
        for i in range(len(y_test_pred_raw)):
            if y_test_pred_raw[i] == 1:
                consecutive_count += 1
                if consecutive_count >= n_consecutive:
                    y_pred_filtered[i] = 1
            else:
                consecutive_count = 0

        recall = recall_score(y_test, y_pred_filtered, zero_division=0)
        precision = precision_score(y_test, y_pred_filtered, zero_division=0)
        fp_per_s = calculate_false_positives_per_second(y_test, y_pred_filtered, SAMPLING_RATE_HZ)

        print(f"   Consecutive={n_consecutive:2d}: Recall={recall:.3f}, Precision={precision:.3f}, FP/s={fp_per_s:.2f}")

    # Classification report
    print("\n[8] Detailed classification report:")
    print(classification_report(y_test, y_test_pred_opt, target_names=['No Pre-Fire', 'Pre-Fire']))

    # Visualizations
    print("\n[9] Creating visualizations...")

    plot_confusion_matrix(y_test, y_test_pred_opt,
                         title=f'Random Forest - Confusion Matrix (threshold={optimal_threshold:.3f}, 20 consecutive)',
                         save_path=f'{OUTPUT_DIR}/confusion_matrix.png')

    plot_roc_curve(y_test, y_test_prob,
                  title='Random Forest - ROC Curve',
                  save_path=f'{OUTPUT_DIR}/roc_curve.png')

    plot_precision_recall_curve(y_test, y_test_prob,
                                title='Random Forest - Precision-Recall Curve',
                                save_path=f'{OUTPUT_DIR}/pr_curve.png')

    # Feature importance
    print("\n[10] Analyzing feature importance...")
    feature_names = joblib.load(f'{FEATURES_DIR}/feature_names.pkl')
    importances = model.feature_importances_

    # Plot feature importance
    plot_feature_importance(feature_names, importances, top_n=20,
                          title='Random Forest - Feature Importance (Top 20)',
                          save_path=f'{OUTPUT_DIR}/feature_importance.png')

    # Print top features
    importance_indices = np.argsort(importances)[::-1]
    print("   Top 15 most important features:")
    for i, idx in enumerate(importance_indices[:15]):
        print(f"      {i+1:2d}. {feature_names[idx]:30s}: {importances[idx]:.4f}")

    # Save model and results
    print("\n[11] Saving model and results...")

    joblib.dump(model, f'{OUTPUT_DIR}/model.pkl')
    print(f"   Saved: {OUTPUT_DIR}/model.pkl")

    results = {
        'model': model,
        'config': RF_CONFIG,
        'optimal_threshold': optimal_threshold,
        'size_info': size_info,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'test_metrics_optimal': test_metrics_opt,
        'feature_importance': {
            'feature_names': feature_names,
            'importances': importances
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
    print("RANDOM FOREST TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel Configuration:")
    print(f"  Trees: {RF_CONFIG['n_estimators']}, Max Depth: {RF_CONFIG['max_depth']}")
    print(f"  Estimated Size: {size_info['estimated_size_kb']:.2f} KB")

    print(f"\nTest Set Performance (with optimal threshold={optimal_threshold:.3f}):")
    print(f"  Accuracy:  {test_metrics_opt['accuracy']:.4f}")
    print(f"  Precision: {test_metrics_opt['precision']:.4f}")
    print(f"  Recall:    {test_metrics_opt['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics_opt['f1']:.4f}")
    print(f"  ROC-AUC:   {test_metrics_opt['roc_auc']:.4f}")
    print(f"\n  FALSE POSITIVES PER SECOND: {test_metrics_opt['fp_per_second']:.2f} FP/s")

    print(f"\nModel saved to: {OUTPUT_DIR}/")
    print("\nNext step: Run 6_train_neural_network.py")


if __name__ == '__main__':
    main()
