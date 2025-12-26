#!/usr/bin/env python3
"""
Training Script for AlexNet on iFood 2019

This script handles training of AlexNet variants with:
- Checkpoint saving/resuming
- WandB logging (optional)
- Learning rate scheduling
- Training history export
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.alexnet import get_model, count_parameters
from src.data_loader import get_dataloaders, get_class_weights


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train AlexNet on iFood 2019')
    
    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to dataset directory')
    
    # Model
    parser.add_argument('--model_name', type=str, default='alexnet_baseline',
                       choices=['alexnet_baseline', 'alexnet_mod1', 
                               'alexnet_mod2', 'alexnet_combined'],
                       help='Model variant to train')
    parser.add_argument('--num_classes', type=int, default=251,
                       help='Number of output classes')
    
    # Training
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--scheduler', type=str, default='step',
                       choices=['step', 'cosine'],
                       help='Learning rate scheduler')
    
    # Data loading
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    
    # Saving
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='ifood-alexnet',
                       help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='WandB run name')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='Use class weights for imbalanced data')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Validate model.
    
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    best_acc: float,
    history: Dict,
    save_path: str
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
        'history': history
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer = None,
    scheduler = None
) -> Tuple[int, float, Dict]:
    """
    Load training checkpoint.
    
    Returns:
        Tuple of (start_epoch, best_acc, history)
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return (
        checkpoint['epoch'] + 1,
        checkpoint['best_acc'],
        checkpoint.get('history', {'train_loss': [], 'train_acc': [], 
                                   'val_loss': [], 'val_acc': []})
    )


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize WandB
    if args.use_wandb:
        try:
            import wandb
            run_name = args.wandb_run_name or f"{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args)
            )
        except ImportError:
            print("Warning: wandb not installed, disabling logging")
            args.use_wandb = False
    
    # Create dataloaders
    print("\nLoading data...")
    dataloaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    
    if 'train' not in dataloaders or 'val' not in dataloaders:
        raise ValueError("Train and val dataloaders required")
    
    # Create model
    print(f"\nCreating model: {args.model_name}")
    model = get_model(args.model_name, num_classes=args.num_classes)
    model = model.to(device)
    
    params = count_parameters(model)
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    
    # Loss function
    if args.use_class_weights:
        class_weights = get_class_weights(args.data_dir, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        start_epoch, best_acc, history = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        print(f"Resumed from epoch {start_epoch}, best_acc: {best_acc:.4f}")
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print("=" * 60)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, epoch + 1
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, dataloaders['val'], criterion, device, epoch + 1
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update scheduler
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print summary
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Log to WandB
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': current_lr
            })
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.save_dir, f'{args.model_name}_best.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, history, save_path)
            print(f"  ✓ New best model saved (acc: {best_acc:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f'{args.model_name}_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, history, save_path)
            print(f"  ✓ Checkpoint saved: {save_path}")
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Best validation accuracy: {best_acc:.4f}")
    
    # Save final model
    final_path = os.path.join(args.save_dir, f'{args.model_name}_final.pth')
    save_checkpoint(model, optimizer, scheduler, args.num_epochs - 1, best_acc, history, final_path)
    
    # Save training history
    history_path = os.path.join(args.save_dir, f'{args.model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Finish WandB
    if args.use_wandb:
        wandb.finish()
    
    return best_acc


if __name__ == "__main__":
    main()
