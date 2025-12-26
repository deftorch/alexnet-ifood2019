"""
Training Script for AlexNet on iFood 2019

Features:
- Multi-model training support
- Checkpoint saving/loading
- Learning rate scheduling
- Training progress logging
- Early stopping (optional)

Author: deftorch
Repository: https://github.com/deftorch/alexnet-ifood2019
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.alexnet import get_model
from data_loader import get_dataloaders


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, args):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.num_epochs}')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        
        # Update metrics
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'acc1': f'{top1.avg:.2f}%',
            'acc5': f'{top5.avg:.2f}%'
        })
        
        # Log to file every N batches
        if args.log_freq > 0 and (batch_idx + 1) % args.log_freq == 0:
            log_msg = (f'Epoch [{epoch}/{args.num_epochs}] '
                      f'Batch [{batch_idx+1}/{len(dataloader)}] '
                      f'Loss: {losses.avg:.4f} '
                      f'Acc@1: {top1.avg:.2f}% '
                      f'Acc@5: {top5.avg:.2f}%')
            log_to_file(args.log_file, log_msg)
    
    return losses.avg, top1.avg, top5.avg


def validate(model, dataloader, criterion, device, epoch, args):
    """Validate the model"""
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Validation')
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Compute accuracy
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            
            # Update metrics
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'acc1': f'{top1.avg:.2f}%',
                'acc5': f'{top5.avg:.2f}%'
            })
    
    return losses.avg, top1.avg, top5.avg


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    """Save checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_path)
        print(f'✅ Best model saved: {best_path}')


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load checkpoint"""
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0)
        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"   Best accuracy: {best_acc:.2f}%")
        
        return start_epoch, best_acc
    else:
        print(f"❌ No checkpoint found at {checkpoint_path}")
        return 0, 0


def log_to_file(filepath, message):
    """Append message to log file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'[{timestamp}] {message}\n')


def main():
    parser = argparse.ArgumentParser(description='Train AlexNet on iFood 2019')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='alexnet_baseline',
                        choices=['alexnet_baseline', 'alexnet_mod1', 
                                'alexnet_mod2', 'alexnet_combined'],
                        help='Model variant to train')
    parser.add_argument('--num_classes', type=int, default=251,
                        help='Number of classes')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer')
    
    # Learning rate scheduler
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['plateau', 'step', 'none'],
                        help='LR scheduler')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='Patience for ReduceLROnPlateau')
    parser.add_argument('--lr_factor', type=float, default=0.1,
                        help='Factor for LR reduction')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Step size for StepLR')
    
    # Checkpoint & logging
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='Log every N batches (0 to disable)')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"Training {args.model_name}")
    print(f"{'='*70}\n")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Epochs: {args.num_epochs}\n")
    
    # Create model-specific save directory
    model_save_dir = os.path.join(args.save_dir, args.model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Setup logging
    log_dir = os.path.join('logs', args.model_name)
    os.makedirs(log_dir, exist_ok=True)
    args.log_file = os.path.join(log_dir, 'train.log')
    
    # Log start
    log_to_file(args.log_file, f"{'='*70}")
    log_to_file(args.log_file, f"Training started: {args.model_name}")
    log_to_file(args.log_file, f"{'='*70}")
    log_to_file(args.log_file, f"Arguments: {json.dumps(vars(args), indent=2)}")
    
    # Load data
    print("Loading data...")
    dataloaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print(f"\nCreating model: {args.model_name}")
    model = get_model(args.model_name, num_classes=args.num_classes)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    
    # Learning rate scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=args.lr_factor,
            patience=args.lr_patience,
            verbose=True
        )
    elif args.scheduler == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.lr_factor
        )
    else:
        scheduler = None
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(model, optimizer, args.resume)
    
    # Training loop
    print(f"\n{'='*70}")
    print("Starting training...")
    print(f"{'='*70}\n")
    
    training_start = time.time()
    
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, device, epoch+1, args
        )
        
        # Validate
        val_loss, val_acc1, val_acc5 = validate(
            model, dataloaders['val'], criterion, device, epoch+1, args
        )
        
        # Update learning rate
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_acc1)
            else:
                scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{args.num_epochs} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Acc@1: {train_acc1:.2f}% | Acc@5: {train_acc5:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Acc@1: {val_acc1:.2f}% | Acc@5: {val_acc5:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Log to file
        log_msg = (f"Epoch {epoch+1}/{args.num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc1:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc1:.2f}% - "
                  f"Time: {epoch_time:.2f}s")
        log_to_file(args.log_file, log_msg)
        
        # Save checkpoint
        is_best = val_acc1 > best_acc
        best_acc = max(val_acc1, best_acc)
        
        if (epoch + 1) % args.save_freq == 0 or is_best:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_name': args.model_name,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'val_acc1': val_acc1,
                    'val_acc5': val_acc5,
                    'train_acc1': train_acc1,
                    'args': vars(args)
                },
                is_best,
                model_save_dir,
                filename=f'checkpoint_epoch_{epoch+1}.pth'
            )
    
    # Training complete
    total_time = time.time() - training_start
    
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved to: {model_save_dir}")
    print(f"Logs saved to: {args.log_file}")
    print(f"{'='*70}\n")
    
    log_to_file(args.log_file, f"Training completed in {total_time/3600:.2f} hours")
    log_to_file(args.log_file, f"Best validation accuracy: {best_acc:.2f}%")
    log_to_file(args.log_file, f"{'='*70}")


if __name__ == '__main__':
    main()
