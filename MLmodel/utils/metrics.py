"""
Comprehensive metrics for model evaluation with focus on false positive rate
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
from typing import Dict, Tuple


def calculate_false_positives_per_second(y_true: np.ndarray,
                                         y_pred: np.ndarray,
                                         sampling_rate_hz: int = 1600) -> float:
    """
    Calculate false positives per second - critical metric for this application.

    A false positive means the model predicts pre-fire when there isn't one,
    causing unnecessary flywheel spinning.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        sampling_rate_hz: Sampling rate in Hz

    Returns:
        False positives per second
    """
    # Count false positives
    fp = ((y_pred == 1) & (y_true == 0)).sum()

    # Calculate total time
    total_samples = len(y_true)
    total_time_s = total_samples / sampling_rate_hz

    fp_per_second = fp / total_time_s if total_time_s > 0 else 0

    return fp_per_second


def calculate_comprehensive_metrics(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_prob: np.ndarray = None,
                                   sampling_rate_hz: int = 1600) -> Dict:
    """
    Calculate comprehensive metrics for model evaluation.

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_prob: Predicted probabilities (optional)
        sampling_rate_hz: Sampling rate

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    # Rates
    metrics['true_positive_rate'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # 1 - FPR

    # Critical metric: False positives per second
    metrics['fp_per_second'] = calculate_false_positives_per_second(
        y_true, y_pred, sampling_rate_hz
    )

    # ROC-AUC (if probabilities provided)
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics['roc_auc'] = auc(fpr, tpr)

        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        metrics['pr_auc'] = auc(recall, precision)
    else:
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None

    # Calculate total time and false positive density
    total_samples = len(y_true)
    total_time_s = total_samples / sampling_rate_hz
    metrics['total_samples'] = total_samples
    metrics['total_time_seconds'] = total_time_s

    return metrics


def print_metrics(metrics: Dict, model_name: str = "Model"):
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{'='*80}")
    print(f"{model_name} Performance Metrics")
    print(f"{'='*80}")

    print(f"\n[Classification Metrics]")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")

    print(f"\n[Confusion Matrix]")
    print(f"  True Positives:  {metrics['true_positives']:8d}")
    print(f"  False Positives: {metrics['false_positives']:8d}")
    print(f"  True Negatives:  {metrics['true_negatives']:8d}")
    print(f"  False Negatives: {metrics['false_negatives']:8d}")

    print(f"\n[Rates]")
    print(f"  True Positive Rate:  {metrics['true_positive_rate']:.4f} (Sensitivity/Recall)")
    print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
    print(f"  Specificity:         {metrics['specificity']:.4f} (1 - FPR)")

    print(f"\n[CRITICAL: False Positive Analysis]")
    print(f"  False Positives per Second: {metrics['fp_per_second']:.2f} FP/s")
    print(f"  Total False Positives:      {metrics['false_positives']:,}")
    print(f"  Total Time:                 {metrics['total_time_seconds']:.1f}s ({metrics['total_time_seconds']/60:.1f}min)")

    if metrics['roc_auc'] is not None:
        print(f"\n[Area Under Curve]")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:  {metrics['pr_auc']:.4f}")

    print(f"\n{'='*80}\n")


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         title: str = "Confusion Matrix",
                         save_path: str = None):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        save_path: Path to save figure (optional)
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([0.5, 1.5], ['No Pre-Fire (0)', 'Pre-Fire (1)'])
    plt.yticks([0.5, 1.5], ['No Pre-Fire (0)', 'Pre-Fire (1)'])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    plt.tight_layout()
    return plt.gcf()


def plot_roc_curve(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   title: str = "ROC Curve",
                   save_path: str = None):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Path to save figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curve to {save_path}")

    plt.tight_layout()
    return plt.gcf()


def plot_precision_recall_curve(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                title: str = "Precision-Recall Curve",
                                save_path: str = None):
    """
    Plot Precision-Recall curve.

    Important for imbalanced datasets.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Path to save figure
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    # Add baseline (random classifier for imbalanced data)
    baseline = y_true.sum() / len(y_true)
    plt.axhline(y=baseline, color='navy', linestyle='--',
                label=f'Baseline (prevalence = {baseline:.3f})')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curve to {save_path}")

    plt.tight_layout()
    return plt.gcf()


def find_optimal_threshold_fbeta(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                min_consecutive: int = 20,
                                sampling_rate_hz: int = 1600,
                                beta: float = 2.0,
                                min_recall: float = 0.5,
                                max_fp_per_second: float = 20.0) -> Tuple[float, Dict]:
    """
    Find optimal threshold using F-beta score with consecutive filtering.

    Beta=2 weights recall 2x more than precision (we care more about catching triggers).
    Simulates consecutive prediction filtering to get realistic FP/s.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        min_consecutive: Minimum consecutive predictions required
        sampling_rate_hz: Sampling rate
        beta: F-beta parameter (2.0 = favor recall 2x)
        min_recall: Minimum acceptable recall
        max_fp_per_second: Maximum acceptable FP/s (with consecutive filtering)

    Returns:
        Tuple of (optimal_threshold, metrics_dict)
    """
    from sklearn.metrics import fbeta_score

    thresholds = np.linspace(0.1, 0.7, 100)  # Focus on usable range
    results = []

    for threshold in thresholds:
        y_pred_raw = (y_prob >= threshold).astype(int)

        # Apply consecutive filtering
        y_pred_filtered = np.zeros_like(y_pred_raw)
        consecutive_count = 0
        for i in range(len(y_pred_raw)):
            if y_pred_raw[i] == 1:
                consecutive_count += 1
                if consecutive_count >= min_consecutive:
                    y_pred_filtered[i] = 1
            else:
                consecutive_count = 0

        # Calculate metrics with filtering
        recall = recall_score(y_true, y_pred_filtered, zero_division=0)
        precision = precision_score(y_true, y_pred_filtered, zero_division=0)
        fp_per_s = calculate_false_positives_per_second(y_true, y_pred_filtered, sampling_rate_hz)

        # Skip if below minimum recall OR above maximum FP/s
        if recall < min_recall:
            continue
        if fp_per_s > max_fp_per_second:
            continue

        fbeta = fbeta_score(y_true, y_pred_filtered, beta=beta, zero_division=0)

        results.append({
            'threshold': threshold,
            'fbeta': fbeta,
            'recall': recall,
            'precision': precision,
            'fp_per_second': fp_per_s,
            'predictions': y_pred_filtered.sum()
        })

    if not results:
        print(f"  WARNING: No threshold achieves min_recall={min_recall} AND max_fp_per_second={max_fp_per_second}")
        print(f"  Relaxing FP/s constraint to 50...")
        # Retry with relaxed FP/s constraint
        return find_optimal_threshold_fbeta(y_true, y_prob, min_consecutive,
                                           sampling_rate_hz, beta, min_recall, max_fp_per_second=50.0)

    # Find threshold with best F-beta score
    best_idx = max(range(len(results)), key=lambda i: results[i]['fbeta'])
    optimal = results[best_idx]

    return optimal['threshold'], optimal


def find_optimal_threshold(y_true: np.ndarray,
                          y_prob: np.ndarray,
                          target_fp_per_second: float = 1.0,
                          sampling_rate_hz: int = 1600) -> Tuple[float, Dict]:
    """
    DEPRECATED: This function optimizes for FP/s but ignores recall.
    Use find_optimal_threshold_fbeta() instead for better results.

    Find optimal prediction threshold to achieve target false positive rate.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        target_fp_per_second: Target FP/s rate
        sampling_rate_hz: Sampling rate

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    # Try different thresholds
    thresholds = np.linspace(0.0, 1.0, 100)
    results = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        fp_per_s = calculate_false_positives_per_second(y_true, y_pred, sampling_rate_hz)
        recall = recall_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            'threshold': threshold,
            'fp_per_second': fp_per_s,
            'recall': recall,
            'precision': precision,
            'f1': f1
        })

    # Find threshold closest to target FP/s
    results_array = np.array([(r['threshold'], abs(r['fp_per_second'] - target_fp_per_second))
                              for r in results])
    optimal_idx = np.argmin(results_array[:, 1])
    optimal_threshold = results[optimal_idx]['threshold']

    print(f"\nOptimal Threshold Analysis (Target: {target_fp_per_second} FP/s)")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    print(f"  FP/s at threshold: {results[optimal_idx]['fp_per_second']:.3f}")
    print(f"  Recall:            {results[optimal_idx]['recall']:.3f}")
    print(f"  Precision:         {results[optimal_idx]['precision']:.3f}")
    print(f"  F1-Score:          {results[optimal_idx]['f1']:.3f}")

    return optimal_threshold, results[optimal_idx]


def analyze_consecutive_predictions(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   min_consecutive: int = 20) -> Dict:
    """
    Analyze consecutive predictions to simulate C++ implementation strategy.

    The C++ code can require N consecutive positive predictions before
    activating flywheels. This analyzes how that affects performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        min_consecutive: Minimum consecutive predictions required

    Returns:
        Dictionary with analysis results
    """
    # Find runs of consecutive predictions
    filtered_pred = np.zeros_like(y_pred)

    consecutive_count = 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            consecutive_count += 1
            if consecutive_count >= min_consecutive:
                filtered_pred[i] = 1
        else:
            consecutive_count = 0

    # Calculate metrics with filtering
    metrics_filtered = calculate_comprehensive_metrics(y_true, filtered_pred)

    print(f"\nConsecutive Prediction Analysis (min={min_consecutive})")
    print(f"  Original predictions: {y_pred.sum()}")
    print(f"  Filtered predictions: {filtered_pred.sum()}")
    print(f"  Reduction: {(1 - filtered_pred.sum()/max(y_pred.sum(), 1)) * 100:.1f}%")
    print(f"  FP/s (original):  {calculate_false_positives_per_second(y_true, y_pred):.3f}")
    print(f"  FP/s (filtered):  {metrics_filtered['fp_per_second']:.3f}")
    print(f"  Recall (original): {recall_score(y_true, y_pred):.3f}")
    print(f"  Recall (filtered): {metrics_filtered['recall']:.3f}")

    return {
        'filtered_predictions': filtered_pred,
        'metrics': metrics_filtered
    }
