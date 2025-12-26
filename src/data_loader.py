#!/usr/bin/env python3
"""
Data Loader for iFood 2019 Dataset

This module provides data loading utilities for training AlexNet
on the iFood 2019 food classification dataset.
"""

import os
from typing import Dict, Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class IFood2019Dataset(Dataset):
    """
    PyTorch Dataset for iFood 2019.
    
    Args:
        data_dir: Root directory containing dataset
        split: One of 'train', 'val', 'test'
        transform: Optional transforms to apply
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None
    ):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load annotations
        if split == 'test':
            # Test set has no labels
            self.annotations = pd.read_csv(
                os.path.join(data_dir, 'test_info.csv'),
                header=None,
                names=['image_name']
            )
            self.has_labels = False
        else:
            self.annotations = pd.read_csv(
                os.path.join(data_dir, f'{split}_info.csv'),
                header=None,
                names=['image_name', 'label']
            )
            self.has_labels = True
        
        # Image directory
        self.image_dir = os.path.join(data_dir, f'{split}_images')
        
        # Load class names if available
        class_file = os.path.join(data_dir, 'class_list.txt')
        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                self.classes = [line.strip().split(' ', 1)[1] for line in f.readlines()]
        else:
            self.classes = [f'class_{i}' for i in range(251)]
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Get image path
        img_name = self.annotations.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        if self.has_labels:
            label = int(self.annotations.iloc[idx, 1])
        else:
            label = -1  # No label for test set
        
        return image, label


def get_transforms(split: str = 'train', img_size: int = 224) -> transforms.Compose:
    """
    Get transforms for each split.
    
    Args:
        split: One of 'train', 'val', 'test'
        img_size: Target image size
    
    Returns:
        torchvision transforms composition
    """
    # ImageNet normalization values
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
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
        # Validation and test transforms
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),  # 256 for 224
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, val, and test sets.
    
    Args:
        data_dir: Root directory containing dataset
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        img_size: Target image size
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        # Check if split exists
        split_dir = os.path.join(data_dir, f'{split}_images')
        split_csv = os.path.join(data_dir, f'{split}_info.csv')
        
        if not os.path.exists(split_dir) or not os.path.exists(split_csv):
            print(f"Warning: {split} split not found, skipping...")
            continue
        
        # Create dataset
        dataset = IFood2019Dataset(
            data_dir=data_dir,
            split=split,
            transform=get_transforms(split, img_size)
        )
        
        # Create dataloader
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == 'train')
        )
        
        print(f"Created {split} dataloader: {len(dataset)} samples, "
              f"{len(dataloaders[split])} batches")
    
    return dataloaders


def get_class_weights(data_dir: str, device: torch.device) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset.
    
    Uses inverse frequency weighting.
    
    Args:
        data_dir: Root directory containing dataset
        device: Device to put weights on
    
    Returns:
        Tensor of class weights
    """
    # Load training annotations
    train_csv = os.path.join(data_dir, 'train_info.csv')
    df = pd.read_csv(train_csv, header=None, names=['image_name', 'label'])
    
    # Count samples per class
    class_counts = df['label'].value_counts().sort_index()
    
    # Calculate weights (inverse frequency)
    total_samples = len(df)
    num_classes = len(class_counts)
    
    weights = total_samples / (num_classes * class_counts.values)
    weights = torch.FloatTensor(weights).to(device)
    
    return weights


if __name__ == "__main__":
    # Test data loader
    print("Testing iFood 2019 Data Loader")
    print("=" * 60)
    
    # Create mock data for testing
    data_dir = './data_mock'
    os.makedirs(os.path.join(data_dir, 'train_images'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val_images'), exist_ok=True)
    
    # Create mock CSV files
    import numpy as np
    
    for split in ['train', 'val']:
        n_samples = 100 if split == 'train' else 20
        images = [f'img_{i:05d}.jpg' for i in range(n_samples)]
        labels = np.random.randint(0, 251, n_samples)
        
        df = pd.DataFrame({'image': images, 'label': labels})
        df.to_csv(os.path.join(data_dir, f'{split}_info.csv'), 
                  header=False, index=False)
        
        # Create mock images
        for img_name in images:
            img = Image.new('RGB', (224, 224), color='red')
            img.save(os.path.join(data_dir, f'{split}_images', img_name))
    
    # Test transform
    transform = get_transforms('train')
    print(f"Train transform: {transform}")
    
    # Test dataloader
    dataloaders = get_dataloaders(data_dir, batch_size=16, num_workers=0)
    
    for split, loader in dataloaders.items():
        batch = next(iter(loader))
        images, labels = batch
        print(f"\n{split}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels sample: {labels[:5]}")
    
    print("\n" + "=" * 60)
    print("Data loader test complete!")
