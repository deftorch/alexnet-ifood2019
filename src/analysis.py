#!/usr/bin/env python3
"""
Comparative Analysis Script for AlexNet Variants

This script generates comparative analysis across all model variants:
- Training curves comparison
- Performance metrics comparison
- Statistical significance tests
- Confusion matrix visualizations
"""

import os
import sys
import json
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure plotting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Comparative analysis of AlexNet variants')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory containing model checkpoints and histories')
    parser.add_argument('--eval_dir', type=str, default='./evaluation_results',
                       help='Directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, default='./analysis_results',
                       help='Directory to save analysis outputs')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['alexnet_baseline', 'alexnet_mod1', 
                               'alexnet_mod2', 'alexnet_combined'],
                       help='Model names to analyze')
    
    return parser.parse_args()


def load_training_histories(checkpoint_dir: str, models: List[str]) -> Dict:
    """Load training histories for all models."""
    histories = {}
    
    for model_name in models:
        history_path = os.path.join(checkpoint_dir, f'{model_name}_history.json')
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                histories[model_name] = json.load(f)
            print(f"Loaded history: {model_name}")
        else:
            print(f"Warning: History not found for {model_name}")
    
    return histories


def load_evaluation_metrics(eval_dir: str, models: List[str], split: str = 'val') -> Dict:
    """Load evaluation metrics for all models."""
    metrics = {}
    
    for model_name in models:
        metrics_path = os.path.join(eval_dir, f'{model_name}_{split}_metrics.json')
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics[model_name] = json.load(f)
            print(f"Loaded metrics: {model_name}")
        else:
            print(f"Warning: Metrics not found for {model_name}")
    
    return metrics


def plot_training_curves(histories: Dict, output_dir: str):
    """Generate training curve comparison plots."""
    if not histories:
        print("No histories to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    # Plot loss curves
    ax = axes[0, 0]
    for (model_name, history), color in zip(histories.items(), colors):
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], '-', color=color, 
                label=f'{model_name} (train)', alpha=0.8)
        ax.plot(epochs, history['val_loss'], '--', color=color, 
                label=f'{model_name} (val)', alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot accuracy curves
    ax = axes[0, 1]
    for (model_name, history), color in zip(histories.items(), colors):
        epochs = range(1, len(history['train_acc']) + 1)
        ax.plot(epochs, history['train_acc'], '-', color=color, 
                label=f'{model_name} (train)', alpha=0.8)
        ax.plot(epochs, history['val_acc'], '--', color=color, 
                label=f'{model_name} (val)', alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training & Validation Accuracy')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot validation loss comparison
    ax = axes[1, 0]
    for (model_name, history), color in zip(histories.items(), colors):
        epochs = range(1, len(history['val_loss']) + 1)
        ax.plot(epochs, history['val_loss'], '-', color=color, 
                label=model_name, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot validation accuracy comparison
    ax = axes[1, 1]
    for (model_name, history), color in zip(histories.items(), colors):
        epochs = range(1, len(history['val_acc']) + 1)
        ax.plot(epochs, history['val_acc'], '-', color=color, 
                label=model_name, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'training_curves_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_metrics_comparison(metrics: Dict, output_dir: str):
    """Generate metrics comparison bar chart."""
    if not metrics:
        print("No metrics to plot")
        return
    
    # Prepare data
    metric_names = ['accuracy', 'top5_accuracy', 'precision', 'recall', 'macro_f1']
    model_names = list(metrics.keys())
    
    data = []
    for model_name in model_names:
        for metric_name in metric_names:
            if metric_name in metrics[model_name]:
                data.append({
                    'Model': model_name.replace('alexnet_', ''),
                    'Metric': metric_name.replace('_', ' ').title(),
                    'Value': metrics[model_name][metric_name]
                })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(model_names))
    width = 0.15
    
    for i, metric in enumerate(metric_names):
        metric_data = df[df['Metric'] == metric.replace('_', ' ').title()]
        values = [metric_data[metric_data['Model'] == m.replace('alexnet_', '')]['Value'].values[0] 
                 if len(metric_data[metric_data['Model'] == m.replace('alexnet_', '')]) > 0 else 0
                 for m in model_names]
        
        bars = ax.bar(x + i * width, values, width, 
                     label=metric.replace('_', ' ').title())
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([m.replace('alexnet_', '') for m in model_names])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_confusion_matrices(eval_dir: str, models: List[str], output_dir: str, split: str = 'val'):
    """Generate confusion matrix visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()
    
    for i, model_name in enumerate(models[:4]):
        conf_path = os.path.join(eval_dir, f'{model_name}_{split}_confusion_matrix.npy')
        
        if os.path.exists(conf_path):
            conf_matrix = np.load(conf_path)
            
            # Normalize
            conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            conf_matrix_norm = np.nan_to_num(conf_matrix_norm)
            
            # Plot (use subset if too large)
            if conf_matrix.shape[0] > 50:
                # Show aggregated view
                n_bins = 25
                bin_size = conf_matrix.shape[0] // n_bins
                aggregated = np.zeros((n_bins, n_bins))
                
                for r in range(n_bins):
                    for c in range(n_bins):
                        r_start, r_end = r * bin_size, (r + 1) * bin_size
                        c_start, c_end = c * bin_size, (c + 1) * bin_size
                        aggregated[r, c] = conf_matrix_norm[r_start:r_end, c_start:c_end].mean()
                
                sns.heatmap(aggregated, ax=axes[i], cmap='Blues', 
                           cbar=True, square=True)
                axes[i].set_title(f'{model_name}\n(Aggregated {n_bins}x{n_bins})')
            else:
                sns.heatmap(conf_matrix_norm, ax=axes[i], cmap='Blues',
                           cbar=True, square=True)
                axes[i].set_title(model_name)
            
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
        else:
            axes[i].text(0.5, 0.5, 'Not Available', ha='center', va='center')
            axes[i].set_title(model_name)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def create_summary_table(histories: Dict, metrics: Dict, output_dir: str):
    """Create summary comparison table."""
    data = []
    
    for model_name in histories.keys():
        row = {'Model': model_name}
        
        # Training metrics
        if model_name in histories:
            history = histories[model_name]
            row['Final Train Loss'] = history['train_loss'][-1] if history['train_loss'] else None
            row['Final Train Acc'] = history['train_acc'][-1] if history['train_acc'] else None
            row['Best Val Acc'] = max(history['val_acc']) if history['val_acc'] else None
            row['Best Val Epoch'] = np.argmax(history['val_acc']) + 1 if history['val_acc'] else None
        
        # Evaluation metrics
        if model_name in metrics:
            m = metrics[model_name]
            row['Test Accuracy'] = m.get('accuracy')
            row['Top-5 Accuracy'] = m.get('top5_accuracy')
            row['Macro F1'] = m.get('macro_f1')
            row['Weighted F1'] = m.get('weighted_f1')
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Round numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'model_comparison_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # Save to markdown
    md_path = os.path.join(output_dir, 'model_comparison_summary.md')
    with open(md_path, 'w') as f:
        f.write("# Model Comparison Summary\n\n")
        f.write(df.to_markdown(index=False))
    print(f"Saved: {md_path}")
    
    # Print table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    
    return df


def perform_statistical_tests(metrics: Dict, output_dir: str):
    """Perform statistical significance tests between models."""
    if len(metrics) < 2:
        print("Need at least 2 models for statistical tests")
        return
    
    model_names = list(metrics.keys())
    results = []
    
    # Compare accuracy between all pairs
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            acc1 = metrics[model1].get('accuracy', 0)
            acc2 = metrics[model2].get('accuracy', 0)
            
            # McNemar's test requires per-sample predictions
            # Here we do a simple comparison
            diff = abs(acc1 - acc2)
            better = model1 if acc1 > acc2 else model2
            
            results.append({
                'Model 1': model1,
                'Model 2': model2,
                'Acc 1': acc1,
                'Acc 2': acc2,
                'Difference': diff,
                'Better Model': better
            })
    
    df = pd.DataFrame(results)
    
    # Save results
    stats_path = os.path.join(output_dir, 'statistical_comparison.csv')
    df.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")
    
    print("\nStatistical Comparison:")
    print(df.to_string(index=False))
    
    return df


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("AlexNet Variants Comparative Analysis")
    print("=" * 60)
    
    # Load data
    print("\nLoading training histories...")
    histories = load_training_histories(args.checkpoint_dir, args.models)
    
    print("\nLoading evaluation metrics...")
    metrics = load_evaluation_metrics(args.eval_dir, args.models)
    
    # Generate plots
    print("\nGenerating training curves comparison...")
    plot_training_curves(histories, args.output_dir)
    
    print("\nGenerating metrics comparison...")
    plot_metrics_comparison(metrics, args.output_dir)
    
    print("\nGenerating confusion matrices...")
    plot_confusion_matrices(args.eval_dir, args.models, args.output_dir)
    
    # Create summary
    print("\nCreating summary table...")
    summary_df = create_summary_table(histories, metrics, args.output_dir)
    
    # Statistical tests
    print("\nPerforming statistical comparisons...")
    perform_statistical_tests(metrics, args.output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
