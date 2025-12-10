"""
Visualization utilities for IMU data and model analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict


def plot_imu_data(df: pd.DataFrame,
                  start_idx: int = 0,
                  n_samples: int = 3200,
                  save_path: str = None):
    """
    Plot IMU data with trigger state.

    Args:
        df: DataFrame with IMU data
        start_idx: Starting index
        n_samples: Number of samples to plot
        save_path: Path to save figure
    """
    end_idx = min(start_idx + n_samples, len(df))
    data = df.iloc[start_idx:end_idx]

    # Create time axis (assuming 1600 Hz sampling)
    time_ms = np.arange(len(data)) * (1000 / 1600)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Accelerometer
    axes[0].plot(time_ms, data['accel_x'], label='Accel X', alpha=0.7)
    axes[0].plot(time_ms, data['accel_y'], label='Accel Y', alpha=0.7)
    axes[0].plot(time_ms, data['accel_z'], label='Accel Z', alpha=0.7)
    axes[0].set_ylabel('Acceleration')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('IMU Data Visualization')

    # Gyroscope
    axes[1].plot(time_ms, data['gyro_x'], label='Gyro X', alpha=0.7)
    axes[1].plot(time_ms, data['gyro_y'], label='Gyro Y', alpha=0.7)
    axes[1].plot(time_ms, data['gyro_z'], label='Gyro Z', alpha=0.7)
    axes[1].set_ylabel('Angular Velocity')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # Trigger state
    axes[2].fill_between(time_ms, 0, data['trigger_state'],
                         alpha=0.5, color='red', label='Trigger Pulled')
    axes[2].set_ylabel('Trigger State')
    axes[2].set_ylim([-0.1, 1.1])
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    # Pre-fire label (if exists)
    if 'pre_fire_label' in data.columns:
        axes[3].fill_between(time_ms, 0, data['pre_fire_label'],
                            alpha=0.5, color='orange', label='Pre-Fire Window')
        axes[3].set_ylabel('Pre-Fire Label')
        axes[3].set_ylim([-0.1, 1.1])
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (ms)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved IMU plot to {save_path}")

    return fig


def plot_roc_curves(models_results: Dict,
                   save_path: str = None):
    """
    Plot ROC curves for multiple models on the same plot.

    Args:
        models_results: Dict mapping model_name -> (y_true, y_prob)
        save_path: Path to save figure
    """
    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(10, 8))

    colors = ['darkorange', 'green', 'purple', 'red', 'blue']

    for idx, (model_name, (y_true, y_prob)) in enumerate(models_results.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                label=f'{model_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves to {save_path}")

    plt.tight_layout()
    return plt.gcf()


def plot_temporal_analysis(df: pd.DataFrame,
                          prediction_window_ms: tuple = (100, 400),
                          sampling_rate_hz: int = 1600,
                          save_path: str = None):
    """
    Analyze and plot temporal characteristics of trigger pulls.

    Args:
        df: DataFrame with trigger_state column
        prediction_window_ms: Prediction window in ms
        sampling_rate_hz: Sampling rate
        save_path: Path to save figure
    """
    # Find trigger pull events
    trigger_pulls = []
    for i in range(1, len(df)):
        if df.iloc[i]['trigger_state'] == 1 and df.iloc[i-1]['trigger_state'] == 0:
            trigger_pulls.append(i)

    # Analyze pre-trigger motion patterns
    min_ms, max_ms = prediction_window_ms
    window_samples = int(max_ms * sampling_rate_hz / 1000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Average IMU signals around trigger pulls
    pre_trigger_accels = []
    pre_trigger_gyros = []

    for pull_idx in trigger_pulls[:100]:  # Use first 100 events
        start_idx = max(0, pull_idx - window_samples)
        if start_idx < pull_idx:
            window = df.iloc[start_idx:pull_idx]
            if len(window) == window_samples:
                accel_mag = np.sqrt(window['accel_x']**2 + window['accel_y']**2 + window['accel_z']**2)
                gyro_mag = np.sqrt(window['gyro_x']**2 + window['gyro_y']**2 + window['gyro_z']**2)
                pre_trigger_accels.append(accel_mag.values)
                pre_trigger_gyros.append(gyro_mag.values)

    if pre_trigger_accels:
        time_before_ms = np.linspace(-max_ms, 0, window_samples)

        avg_accel = np.mean(pre_trigger_accels, axis=0)
        std_accel = np.std(pre_trigger_accels, axis=0)

        axes[0, 0].plot(time_before_ms, avg_accel, 'b-', linewidth=2, label='Mean')
        axes[0, 0].fill_between(time_before_ms,
                                avg_accel - std_accel,
                                avg_accel + std_accel,
                                alpha=0.3, label='±1 STD')
        axes[0, 0].axvspan(-max_ms, -min_ms, alpha=0.2, color='orange',
                          label=f'Prediction Window ({min_ms}-{max_ms}ms)')
        axes[0, 0].axvline(0, color='red', linestyle='--', label='Trigger Pull')
        axes[0, 0].set_xlabel('Time Before Trigger (ms)')
        axes[0, 0].set_ylabel('Accelerometer Magnitude')
        axes[0, 0].set_title('Average Accelerometer Magnitude Before Trigger Pull')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        avg_gyro = np.mean(pre_trigger_gyros, axis=0)
        std_gyro = np.std(pre_trigger_gyros, axis=0)

        axes[0, 1].plot(time_before_ms, avg_gyro, 'g-', linewidth=2, label='Mean')
        axes[0, 1].fill_between(time_before_ms,
                                avg_gyro - std_gyro,
                                avg_gyro + std_gyro,
                                alpha=0.3, label='±1 STD')
        axes[0, 1].axvspan(-max_ms, -min_ms, alpha=0.2, color='orange',
                          label=f'Prediction Window ({min_ms}-{max_ms}ms)')
        axes[0, 1].axvline(0, color='red', linestyle='--', label='Trigger Pull')
        axes[0, 1].set_xlabel('Time Before Trigger (ms)')
        axes[0, 1].set_ylabel('Gyroscope Magnitude')
        axes[0, 1].set_title('Average Gyroscope Magnitude Before Trigger Pull')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # Plot 2: Distribution of trigger pull intervals
    if len(trigger_pulls) > 1:
        intervals_samples = np.diff(trigger_pulls)
        intervals_ms = intervals_samples * (1000 / sampling_rate_hz)

        axes[1, 0].hist(intervals_ms, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Time Between Trigger Pulls (ms)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Trigger Pull Interval Distribution\n(Mean: {np.mean(intervals_ms):.1f}ms, Median: {np.median(intervals_ms):.1f}ms)')
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 3: Trigger duration distribution
    trigger_durations = []
    in_trigger = False
    trigger_start = 0

    for i in range(len(df)):
        if df.iloc[i]['trigger_state'] == 1 and not in_trigger:
            in_trigger = True
            trigger_start = i
        elif df.iloc[i]['trigger_state'] == 0 and in_trigger:
            in_trigger = False
            duration_samples = i - trigger_start
            duration_ms = duration_samples * (1000 / sampling_rate_hz)
            trigger_durations.append(duration_ms)

    if trigger_durations:
        axes[1, 1].hist(trigger_durations, bins=30, edgecolor='black', alpha=0.7, color='red')
        axes[1, 1].set_xlabel('Trigger Pull Duration (ms)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title(f'Trigger Pull Duration Distribution\n(Mean: {np.mean(trigger_durations):.1f}ms, Median: {np.median(trigger_durations):.1f}ms)')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved temporal analysis to {save_path}")

    return fig


def plot_feature_importance(feature_names: List[str],
                           importances: np.ndarray,
                           top_n: int = 20,
                           title: str = "Feature Importance",
                           save_path: str = None):
    """
    Plot feature importance.

    Args:
        feature_names: List of feature names
        importances: Array of importance scores
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save figure
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:top_n]
    top_names = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_importances, align='center')
    plt.yticks(range(top_n), top_names)
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature importance to {save_path}")

    plt.tight_layout()
    return plt.gcf()


def plot_model_comparison(models_metrics: Dict,
                         save_path: str = None):
    """
    Create comprehensive model comparison visualization.

    Args:
        models_metrics: Dict mapping model_name -> metrics_dict
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    model_names = list(models_metrics.keys())
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot 1: Classification Metrics
    x = np.arange(len(model_names))
    width = 0.2

    for i, metric in enumerate(metrics_to_plot):
        values = [models_metrics[name][metric] for name in model_names]
        axes[0, 0].bar(x + i*width, values, width, label=metric.capitalize())

    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Classification Metrics Comparison')
    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticks(x + width * 1.5)
    axes[0, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim([0, 1.1])

    # Plot 2: False Positives per Second (LOG SCALE)
    fp_rates = [models_metrics[name]['fp_per_second'] for name in model_names]
    axes[0, 1].bar(model_names, fp_rates, color=colors[:len(model_names)])
    axes[0, 1].set_ylabel('False Positives per Second')
    axes[0, 1].set_title('False Positive Rate Comparison (CRITICAL)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].tick_params(axis='x', rotation=15)
    axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Add values on top of bars
    for i, v in enumerate(fp_rates):
        axes[0, 1].text(i, v, f'{v:.2f}', ha='center', va='bottom')

    # Plot 3: Confusion Matrix Comparison (stacked bar)
    tp_values = [models_metrics[name]['true_positives'] for name in model_names]
    fp_values = [models_metrics[name]['false_positives'] for name in model_names]
    tn_values = [models_metrics[name]['true_negatives'] for name in model_names]
    fn_values = [models_metrics[name]['false_negatives'] for name in model_names]

    x = np.arange(len(model_names))
    axes[1, 0].bar(x, tp_values, label='True Positives', color='green', alpha=0.7)
    axes[1, 0].bar(x, fp_values, bottom=tp_values, label='False Positives', color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Count (log scale)')
    axes[1, 0].set_title('Prediction Distribution')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=15, ha='right')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Plot 4: ROC-AUC Comparison
    if all('roc_auc' in models_metrics[name] and models_metrics[name]['roc_auc'] is not None
           for name in model_names):
        roc_aucs = [models_metrics[name]['roc_auc'] for name in model_names]
        axes[1, 1].bar(model_names, roc_aucs, color=colors[:len(model_names)])
        axes[1, 1].set_ylabel('ROC-AUC')
        axes[1, 1].set_title('ROC-AUC Comparison')
        axes[1, 1].tick_params(axis='x', rotation=15)
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # Add values on top of bars
        for i, v in enumerate(roc_aucs):
            axes[1, 1].text(i, v, f'{v:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved model comparison to {save_path}")

    return fig
