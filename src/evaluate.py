#!/usr/bin/env python3
"""
Evaluation Script for AlexNet on iFood 2019

This script evaluates trained models and generates:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix
- Per-class performance analysis
- Top-5 accuracy
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.alexnet import get_model
from src.data_loader import get_dataloaders


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate AlexNet on iFood 2019')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_name', type=str, required=True,
                       choices=['alexnet_baseline', 'alexnet_mod1', 
                               'alexnet_mod2', 'alexnet_combined'],
                       help='Model variant')
    parser.add_argument('--num_classes', type=int, default=251,
                       help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--split', type=str, default='val',
                       choices=['val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    
    return parser.parse_args()


def load_model(model_path: str, model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    """Load trained model from checkpoint."""
    model = get_model(model_name, num_classes=num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 251
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on dataset.
    
    Returns:
        Tuple of (all_predictions, all_labels, all_probs)
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)
    
    return all_predictions, all_labels, all_probs


def calculate_topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    """Calculate top-k accuracy."""
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = np.any(top_k_preds == labels.reshape(-1, 1), axis=1)
    return np.mean(correct)


def calculate_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    class_names: List[str] = None
) -> Dict:
    """Calculate comprehensive evaluation metrics."""
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Top-5 accuracy
    top5_accuracy = calculate_topk_accuracy(probs, labels, k=5)
    
    # Precision, Recall, F1 (macro-averaged)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    
    # Weighted F1 (accounts for class imbalance)
    weighted_f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(labels, predictions)
    
    # Per-class metrics
    class_report = classification_report(
        labels, predictions,
        target_names=class_names if class_names else None,
        output_dict=True,
        zero_division=0
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'top5_accuracy': float(top5_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'num_samples': len(labels),
        'num_classes': len(np.unique(labels))
    }
    
    return metrics


def save_results(
    metrics: Dict,
    predictions: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    output_dir: str,
    model_name: str,
    split: str
):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f'{model_name}_{split}_metrics.json')
    
    # Create serializable metrics (exclude confusion matrix for main JSON)
    metrics_to_save = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"Metrics saved to: {metrics_path}")
    
    # Save confusion matrix separately
    conf_matrix_path = os.path.join(output_dir, f'{model_name}_{split}_confusion_matrix.npy')
    np.save(conf_matrix_path, np.array(metrics['confusion_matrix']))
    print(f"Confusion matrix saved to: {conf_matrix_path}")
    
    # Save predictions
    predictions_path = os.path.join(output_dir, f'{model_name}_{split}_predictions.csv')
    df = pd.DataFrame({
        'true_label': labels,
        'predicted_label': predictions,
        'correct': labels == predictions
    })
    df.to_csv(predictions_path, index=True)
    print(f"Predictions saved to: {predictions_path}")
    
    # Save probabilities
    probs_path = os.path.join(output_dir, f'{model_name}_{split}_probabilities.npy')
    np.save(probs_path, probs)
    print(f"Probabilities saved to: {probs_path}")


def print_results(metrics: Dict, model_name: str, split: str):
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print(f"Evaluation Results: {model_name} on {split}")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:      {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f} ({metrics['top5_accuracy']*100:.2f}%)")
    print(f"  Precision:     {metrics['precision']:.4f}")
    print(f"  Recall:        {metrics['recall']:.4f}")
    print(f"  Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:   {metrics['weighted_f1']:.4f}")
    
    print(f"\nDataset Info:")
    print(f"  Samples: {metrics['num_samples']:,}")
    print(f"  Classes: {metrics['num_classes']}")
    
    # Per-class summary
    if 'classification_report' in metrics:
        report = metrics['classification_report']
        
        # Find best and worst classes
        class_f1s = [(k, v['f1-score']) for k, v in report.items() 
                     if isinstance(v, dict) and 'f1-score' in v]
        
        if class_f1s:
            class_f1s.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\nTop 5 Classes (by F1):")
            for name, f1 in class_f1s[:5]:
                print(f"  {name}: {f1:.4f}")
            
            print(f"\nBottom 5 Classes (by F1):")
            for name, f1 in class_f1s[-5:]:
                print(f"  {name}: {f1:.4f}")
    
    print("\n" + "=" * 60)


def main():
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    print(f"Checkpoint: {args.model_path}")
    model = load_model(args.model_path, args.model_name, args.num_classes, device)
    
    # Load data
    print(f"\nLoading {args.split} data...")
    dataloaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if args.split not in dataloaders:
        raise ValueError(f"Split '{args.split}' not found in dataloaders")
    
    dataloader = dataloaders[args.split]
    
    # Load class names if available
    class_file = os.path.join(args.data_dir, 'class_list.txt')
    if os.path.exists(class_file):
        with open(class_file, 'r') as f:
            class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]
    else:
        class_names = None
    
    # Evaluate
    print("\nEvaluating model...")
    predictions, labels, probs = evaluate(model, dataloader, device, args.num_classes)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, labels, probs, class_names)
    
    # Save results
    save_results(metrics, predictions, labels, probs, 
                args.output_dir, args.model_name, args.split)
    
    # Print results
    print_results(metrics, args.model_name, args.split)
    
    return metrics


if __name__ == "__main__":
    main()
