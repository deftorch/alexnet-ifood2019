"""
Evaluation Script for AlexNet on iFood 2019

Computes:
- Accuracy (Top-1, Top-5)
- Per-class accuracy
- Confusion matrix
- Precision, Recall, F1-score

Author: deftorch
Repository: https://github.com/deftorch/alexnet-ifood2019
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.alexnet import get_model
from data_loader import get_dataloaders


def evaluate_model(model, dataloader, device, num_classes=251):
    """
    Evaluate model on a dataset
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Compute metrics
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(all_labels, all_preds) * 100
    
    # Top-5 accuracy
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_correct = np.array([label in top5_preds[i] 
                             for i, label in enumerate(all_labels)])
    metrics['top5_accuracy'] = top5_correct.mean() * 100
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    metrics['per_class'] = {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1_score': f1.tolist(),
        'support': support.tolist()
    }
    
    # Macro/Micro averages
    for avg in ['macro', 'micro', 'weighted']:
        p, r, f, _ = precision_recall_fscore_support(
            all_labels, all_preds, average=avg, zero_division=0
        )
        metrics[f'{avg}_precision'] = p * 100
        metrics[f'{avg}_recall'] = r * 100
        metrics[f'{avg}_f1'] = f * 100
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(
        all_labels, all_preds
    ).tolist()
    
    # Store predictions
    metrics['predictions'] = all_preds.tolist()
    metrics['labels'] = all_labels.tolist()
    metrics['probabilities'] = all_probs.tolist()
    
    return metrics


def print_metrics(metrics, class_names=None):
    """Print evaluation metrics in a readable format"""
    
    print(f"\n{'='*70}")
    print("Evaluation Results")
    print(f"{'='*70}\n")
    
    # Overall metrics
    print("Overall Metrics:")
    print(f"  Accuracy (Top-1): {metrics['accuracy']:.2f}%")
    print(f"  Accuracy (Top-5): {metrics['top5_accuracy']:.2f}%")
    print()
    
    # Average metrics
    print("Average Metrics:")
    for avg in ['macro', 'micro', 'weighted']:
        print(f"  {avg.capitalize()}:")
        print(f"    Precision: {metrics[f'{avg}_precision']:.2f}%")
        print(f"    Recall: {metrics[f'{avg}_recall']:.2f}%")
        print(f"    F1-Score: {metrics[f'{avg}_f1']:.2f}%")
    print()
    
    # Per-class summary
    per_class = metrics['per_class']
    precision = np.array(per_class['precision'])
    recall = np.array(per_class['recall'])
    f1 = np.array(per_class['f1_score'])
    support = np.array(per_class['support'])
    
    print("Per-Class Statistics:")
    print(f"  Number of classes: {len(precision)}")
    print(f"  Precision - Mean: {precision.mean()*100:.2f}% | "
          f"Std: {precision.std()*100:.2f}% | "
          f"Min: {precision.min()*100:.2f}% | "
          f"Max: {precision.max()*100:.2f}%")
    print(f"  Recall - Mean: {recall.mean()*100:.2f}% | "
          f"Std: {recall.std()*100:.2f}% | "
          f"Min: {recall.min()*100:.2f}% | "
          f"Max: {recall.max()*100:.2f}%")
    print(f"  F1-Score - Mean: {f1.mean()*100:.2f}% | "
          f"Std: {f1.std()*100:.2f}% | "
          f"Min: {f1.min()*100:.2f}% | "
          f"Max: {f1.max()*100:.2f}%")
    print()
    
    # Best and worst performing classes
    top_k = 5
    best_idx = np.argsort(f1)[-top_k:][::-1]
    worst_idx = np.argsort(f1)[:top_k]
    
    print(f"Top {top_k} Best Performing Classes (by F1-score):")
    for i, idx in enumerate(best_idx, 1):
        class_name = class_names[idx] if class_names else f"Class {idx}"
        print(f"  {i}. {class_name}: F1={f1[idx]*100:.2f}%, "
              f"Precision={precision[idx]*100:.2f}%, "
              f"Recall={recall[idx]*100:.2f}% "
              f"(n={support[idx]})")
    print()
    
    print(f"Top {top_k} Worst Performing Classes (by F1-score):")
    for i, idx in enumerate(worst_idx, 1):
        class_name = class_names[idx] if class_names else f"Class {idx}"
        print(f"  {i}. {class_name}: F1={f1[idx]*100:.2f}%, "
              f"Precision={precision[idx]*100:.2f}%, "
              f"Recall={recall[idx]*100:.2f}% "
              f"(n={support[idx]})")
    
    print(f"\n{'='*70}\n")


def save_results(metrics, output_dir, model_name, split='test'):
    """Save evaluation results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full metrics as JSON
    json_path = os.path.join(output_dir, f'{model_name}_{split}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved: {json_path}")
    
    # Save confusion matrix as CSV
    cm = np.array(metrics['confusion_matrix'])
    cm_path = os.path.join(output_dir, f'{model_name}_{split}_confusion_matrix.csv')
    pd.DataFrame(cm).to_csv(cm_path, index=False)
    print(f"✅ Confusion matrix saved: {cm_path}")
    
    # Save per-class metrics as CSV
    per_class_df = pd.DataFrame({
        'class_id': range(len(metrics['per_class']['precision'])),
        'precision': metrics['per_class']['precision'],
        'recall': metrics['per_class']['recall'],
        'f1_score': metrics['per_class']['f1_score'],
        'support': metrics['per_class']['support']
    })
    per_class_path = os.path.join(output_dir, f'{model_name}_{split}_per_class.csv')
    per_class_df.to_csv(per_class_path, index=False)
    print(f"✅ Per-class metrics saved: {per_class_path}")
    
    # Save summary
    summary = {
        'model': model_name,
        'split': split,
        'accuracy': metrics['accuracy'],
        'top5_accuracy': metrics['top5_accuracy'],
        'macro_f1': metrics['macro_f1'],
        'weighted_f1': metrics['weighted_f1'],
        'num_samples': len(metrics['labels'])
    }
    summary_path = os.path.join(output_dir, f'{model_name}_{split}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate AlexNet on iFood 2019')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['alexnet_baseline', 'alexnet_mod1',
                                'alexnet_mod2', 'alexnet_combined'],
                        help='Model variant to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=251,
                        help='Number of classes')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save results')
    parser.add_argument('--class_list', type=str, default='',
                        help='Path to class names file (optional)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"Evaluating {args.model_name}")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}\n")
    
    # Load class names (if available)
    class_names = None
    if args.class_list and os.path.exists(args.class_list):
        with open(args.class_list, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(class_names)} class names\n")
    
    # Load data
    print("Loading data...")
    dataloaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    dataloader = dataloaders[args.split]
    
    # Create model
    print(f"\nCreating model: {args.model_name}")
    model = get_model(args.model_name, num_classes=args.num_classes)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'val_acc1' in checkpoint:
        print(f"   Validation accuracy: {checkpoint['val_acc1']:.2f}%")
    print()
    
    # Evaluate
    print(f"{'='*70}")
    print(f"Starting evaluation on {args.split} set...")
    print(f"{'='*70}\n")
    
    metrics = evaluate_model(model, dataloader, device, args.num_classes)
    
    # Print results
    print_metrics(metrics, class_names)
    
    # Save results
    save_results(metrics, args.output_dir, args.model_name, args.split)
    
    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
