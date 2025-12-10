#!/usr/bin/env python3
"""
Step 7: Comprehensive Model Evaluation and Comparison

Compare all three models and generate final analysis.
"""

import sys
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Add utils to path
sys.path.insert(0, '.')

from utils.visualization import plot_roc_curves, plot_model_comparison
from utils.metrics import print_metrics

# Configuration
MODELS_DIR = 'outputs/models'
OUTPUT_DIR = 'outputs/evaluation'


def load_model_results(model_name):
    """Load results for a model."""
    results_path = f'{MODELS_DIR}/{model_name}/results.pkl'
    return joblib.load(results_path)


def create_comparison_table(models_results):
    """Create comparison table of all models."""
    rows = []

    for model_name, results in models_results.items():
        metrics = results['test_metrics_optimal']
        row = {
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}",
            'ROC-AUC': f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] else 'N/A',
            'FP/s': f"{metrics['fp_per_second']:.2f}",
            'FPR': f"{metrics['false_positive_rate']:.4f}",
            'Threshold': f"{results['optimal_threshold']:.3f}"
        }
        rows.append(row)

    return pd.DataFrame(rows)


def create_deployment_comparison(models_results):
    """Create deployment-focused comparison."""
    rows = []

    # Logistic Regression
    if 'Logistic Regression' in models_results:
        lr_results = models_results['Logistic Regression']
        lr_model = lr_results['model']
        lr_params = sum(1 for coef in lr_model.coef_[0]) + 1  # weights + bias
        lr_size_kb = lr_params * 4 / 1024  # float32

        rows.append({
            'Model': 'Logistic Regression',
            'Parameters': f'{lr_params:,}',
            'Size (KB)': f'{lr_size_kb:.2f}',
            'RP2040 Deployable': 'Yes',
            'Inference Speed': 'Very Fast',
            'Complexity': 'Low'
        })

    # Random Forest
    if 'Random Forest' in models_results:
        rf_results = models_results['Random Forest']
        rf_size = rf_results['size_info']

        rows.append({
            'Model': 'Random Forest',
            'Parameters': f'{rf_size["total_nodes"]:,} nodes',
            'Size (KB)': f'{rf_size["estimated_size_kb"]:.2f}',
            'RP2040 Deployable': 'Yes' if rf_size['estimated_size_kb'] < 200 else 'Maybe',
            'Inference Speed': 'Fast',
            'Complexity': 'Medium'
        })

    # Neural Network
    if 'Neural Network' in models_results:
        nn_results = models_results['Neural Network']
        nn_params = nn_results['total_parameters']
        nn_size_kb = nn_params * 4 / 1024

        rows.append({
            'Model': 'Neural Network',
            'Parameters': f'{nn_params:,}',
            'Size (KB)': f'{nn_size_kb:.2f}',
            'RP2040 Deployable': 'No',
            'Inference Speed': 'Slow',
            'Complexity': 'High'
        })

    return pd.DataFrame(rows)


def main():
    print("="*80)
    print("STEP 7: COMPREHENSIVE MODEL EVALUATION")
    print("="*80)

    # Create output directory
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load all model results
    print("\n[1] Loading model results...")
    models_results = {}

    model_names = ['logistic_regression', 'random_forest', 'neural_network']
    display_names = ['Logistic Regression', 'Random Forest', 'Neural Network']

    for model_name, display_name in zip(model_names, display_names):
        try:
            results = load_model_results(model_name)
            models_results[display_name] = results
            print(f"   ✓ Loaded {display_name}")
        except Exception as e:
            print(f"   ✗ Failed to load {display_name}: {e}")

    if len(models_results) < 2:
        print("\nError: Need at least 2 models trained to compare.")
        print("Please run the training scripts first (4, 5, 6)")
        return

    # Print individual metrics
    print("\n[2] Individual Model Performance:")
    print("="*80)

    for model_name, results in models_results.items():
        print_metrics(results['test_metrics_optimal'], f"{model_name} (Test Set)")

    # Create comparison table
    print("\n[3] Creating comparison table...")
    comparison_table = create_comparison_table(models_results)

    print("\n" + "="*80)
    print("MODEL COMPARISON TABLE")
    print("="*80)
    print(comparison_table.to_string(index=False))
    print("="*80)

    # Save comparison table
    comparison_table.to_csv(f'{OUTPUT_DIR}/model_comparison.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR}/model_comparison.csv")

    # Deployment comparison
    print("\n[4] Creating deployment comparison...")
    deployment_table = create_deployment_comparison(models_results)

    print("\n" + "="*80)
    print("DEPLOYMENT COMPARISON")
    print("="*80)
    print(deployment_table.to_string(index=False))
    print("="*80)

    deployment_table.to_csv(f'{OUTPUT_DIR}/deployment_comparison.csv', index=False)
    print(f"\nSaved: {OUTPUT_DIR}/deployment_comparison.csv")

    # ROC curve comparison
    print("\n[5] Creating ROC curves comparison...")
    roc_data = {}
    for model_name, results in models_results.items():
        y_test = results['predictions']['y_test']
        y_prob = results['predictions']['y_test_prob']
        roc_data[model_name] = (y_test, y_prob)

    plot_roc_curves(roc_data, save_path=f'{OUTPUT_DIR}/roc_curves_comparison.png')

    # Model comparison visualization
    print("\n[6] Creating comprehensive comparison visualization...")
    models_metrics = {}
    for model_name, results in models_results.items():
        models_metrics[model_name] = results['test_metrics_optimal']

    plot_model_comparison(models_metrics, save_path=f'{OUTPUT_DIR}/models_comparison.png')

    # False positive analysis
    print("\n[7] False Positive Rate Analysis...")
    print("="*80)
    print(f"{'Model':<25} {'FP/s':<10} {'FPR':<10} {'Precision':<10} {'Recall':<10}")
    print("="*80)

    for model_name, results in models_results.items():
        metrics = results['test_metrics_optimal']
        print(f"{model_name:<25} "
              f"{metrics['fp_per_second']:<10.2f} "
              f"{metrics['false_positive_rate']:<10.4f} "
              f"{metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f}")

    print("="*80)

    # Recommendations
    print("\n[8] Generating recommendations...")
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR RP2040 DEPLOYMENT")
    print("="*80)

    # Find best deployable model
    deployable_models = {}
    for model_name, results in models_results.items():
        if model_name == 'Random Forest':
            if results['size_info']['estimated_size_kb'] < 200:
                deployable_models[model_name] = results
        elif model_name == 'Logistic Regression':
            deployable_models[model_name] = results

    if deployable_models:
        # Sort by F1-score
        best_model = max(deployable_models.items(),
                        key=lambda x: x[1]['test_metrics_optimal']['f1'])

        best_name = best_model[0]
        best_metrics = best_model[1]['test_metrics_optimal']

        print(f"\nRECOMMENDED MODEL: {best_name}")
        print(f"  ✓ Deployable on RP2040")
        print(f"  ✓ F1-Score: {best_metrics['f1']:.4f}")
        print(f"  ✓ Recall: {best_metrics['recall']:.4f}")
        print(f"  ✓ False Positives: {best_metrics['fp_per_second']:.2f} per second")

        if best_name == 'Random Forest':
            print(f"  ✓ Model Size: {best_model[1]['size_info']['estimated_size_kb']:.2f} KB")
            print(f"  ✓ Trees: {best_model[1]['config']['n_estimators']}")

        print("\nADDITIONAL RECOMMENDATIONS:")
        print("  1. Test with consecutive prediction filtering (20+ samples)")
        print("     to further reduce false positives")
        print("  2. Consider adjusting threshold based on use case:")
        print(f"     - Current optimal: {best_model[1]['optimal_threshold']:.3f}")
        print("     - Higher threshold = fewer false positives, lower recall")
        print("     - Lower threshold = more triggers caught, more false positives")
        print("  3. Monitor performance in real-world conditions")
        print("  4. Collect more diverse training data if needed")

    # Neural network as baseline
    if 'Neural Network' in models_results:
        nn_metrics = models_results['Neural Network']['test_metrics_optimal']
        print(f"\nNEURAL NETWORK BASELINE (not deployable):")
        print(f"  Best achievable F1-Score: {nn_metrics['f1']:.4f}")
        print(f"  Best achievable Recall: {nn_metrics['recall']:.4f}")
        print(f"  Best achievable FP/s: {nn_metrics['fp_per_second']:.2f}")

        if deployable_models:
            best_metrics = best_model[1]['test_metrics_optimal']
            f1_gap = nn_metrics['f1'] - best_metrics['f1']
            print(f"\n  Performance gap vs {best_name}: {f1_gap:.4f} F1-score")
            if f1_gap < 0.05:
                print(f"  → Deployable model performs similarly to NN!")
            else:
                print(f"  → Trade-off: Accept {f1_gap:.1%} performance loss for deployment")

    # Summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  - model_comparison.csv")
    print("  - deployment_comparison.csv")
    print("  - roc_curves_comparison.png")
    print("  - models_comparison.png")
    print("\nNext step: Run 8_export_for_deployment.py to export the chosen model")


if __name__ == '__main__':
    main()
