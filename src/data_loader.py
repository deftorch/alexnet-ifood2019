"""
Data Loader for iFood 2019 Dataset

Handles:
- Loading images and labels from CSV
- Data augmentation (training)
- Normalization
- Batch loading with DataLoader

Author: deftorch
Repository: https://github.com/deftorch/alexnet-ifood2019
"""

import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class iFoodDataset(Dataset):
    """
    Custom Dataset for iFood 2019
    
    Args:
        csv_file (str): Path to CSV file (train_info.csv, val_info.csv, test_info.csv)
        img_dir (str): Directory with images
        transform (callable, optional): Transform to apply to images
    """
    
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Get column names (flexible for different CSV formats)
        self.img_col = self.data.columns[0]  # First column = image path
        self.label_col = self.data.columns[1] if len(self.data.columns) > 1 else None
        
        print(f"Loaded {len(self.data)} samples from {os.path.basename(csv_file)}")
        if self.label_col:
            num_classes = self.data[self.label_col].nunique()
            print(f"  Classes: {num_classes}")
            print(f"  Image column: '{self.img_col}'")
            print(f"  Label column: '{self.label_col}'")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get image path
        img_name = self.data.iloc[idx][self.img_col]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Get label (if available)
        if self.label_col:
            label = int(self.data.iloc[idx][self.label_col])
        else:
            label = -1  # For test set without labels
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split='train', img_size=224):
    """
    Get transforms for each split
    
    Args:
        split (str): 'train', 'val', or 'test'
        img_size (int): Target image size
    
    Returns:
        torchvision.transforms.Compose: Transform pipeline
    """
    
    # ImageNet normalization (standard for transfer learning)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == 'train':
        # Training augmentation
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # Validation/Test (no augmentation)
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloaders(data_dir, batch_size=64, num_workers=4, img_size=224):
    """
    Create DataLoaders for train/val/test splits
    
    Args:
        data_dir (str): Path to data directory containing:
                        - train_info.csv, train_images/
                        - val_info.csv, val_images/
                        - test_info.csv, test_images/
        batch_size (int): Batch size for training
        num_workers (int): Number of data loading workers
        img_size (int): Image size (default 224 for AlexNet)
    
    Returns:
        dict: Dictionary with 'train', 'val', 'test' DataLoaders
    
    Example:
        >>> dataloaders = get_dataloaders('data/', batch_size=64)
        >>> for images, labels in dataloaders['train']:
        >>>     # Training loop
    """
    
    print("\n" + "="*70)
    print("Creating DataLoaders")
    print("="*70 + "\n")
    
    # Paths
    paths = {
        'train': {
            'csv': os.path.join(data_dir, 'train_info.csv'),
            'img_dir': os.path.join(data_dir, 'train_images')
        },
        'val': {
            'csv': os.path.join(data_dir, 'val_info.csv'),
            'img_dir': os.path.join(data_dir, 'val_images')
        },
        'test': {
            'csv': os.path.join(data_dir, 'test_info.csv'),
            'img_dir': os.path.join(data_dir, 'test_images')
        }
    }
    
    # Verify all paths exist
    for split, p in paths.items():
        if not os.path.exists(p['csv']):
            raise FileNotFoundError(f"{split} CSV not found: {p['csv']}")
        if not os.path.exists(p['img_dir']):
            raise FileNotFoundError(f"{split} image dir not found: {p['img_dir']}")
    
    # Create datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        transform = get_transforms(split, img_size)
        datasets[split] = iFoodDataset(
            csv_file=paths[split]['csv'],
            img_dir=paths[split]['img_dir'],
            transform=transform
        )
        print()
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    # Summary
    print("="*70)
    print("DataLoader Summary")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")
    print(f"  Image size: {img_size}x{img_size}")
    
    print(f"\nDataset Sizes:")
    for split in ['train', 'val', 'test']:
        n_samples = len(datasets[split])
        n_batches = len(dataloaders[split])
        print(f"  {split:5s}: {n_samples:6,} samples | {n_batches:4,} batches")
    
    print(f"\nMemory per batch (approx):")
    mem_per_img = img_size * img_size * 3 * 4 / (1024**2)  # MB (float32)
    mem_per_batch = mem_per_img * batch_size
    print(f"  Per image: {mem_per_img:.2f} MB")
    print(f"  Per batch: {mem_per_batch:.2f} MB")
    
    print("\n" + "="*70 + "\n")
    
    return dataloaders


def verify_dataloader(dataloader, split_name='train', num_batches=1):
    """
    Verify dataloader by loading a few batches
    
    Args:
        dataloader: PyTorch DataLoader
        split_name (str): Name for printing
        num_batches (int): Number of batches to test
    """
    print(f"\n{'='*70}")
    print(f"Verifying {split_name} DataLoader")
    print(f"{'='*70}\n")
    
    for i, (images, labels) in enumerate(dataloader):
        if i >= num_batches:
            break
        
        print(f"Batch {i+1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Label range: [{labels.min()}, {labels.max()}]")
        print(f"  Memory: {images.element_size() * images.nelement() / 1024**2:.2f} MB")
        
        # Check for NaN or Inf
        if torch.isnan(images).any():
            print("  ⚠️  WARNING: NaN values detected!")
        if torch.isinf(images).any():
            print("  ⚠️  WARNING: Inf values detected!")
        
        print()
    
    print("✅ DataLoader verification complete!\n")


if __name__ == "__main__":
    # Test data loader
    import argparse
    
    parser = argparse.ArgumentParser(description='Test iFood Data Loader')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers')
    
    args = parser.parse_args()
    
    # Create dataloaders
    try:
        dataloaders = get_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Verify each split
        for split in ['train', 'val', 'test']:
            verify_dataloader(dataloaders[split], split_name=split, num_batches=2)
        
        print("="*70)
        print("✅ All DataLoaders working correctly!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
