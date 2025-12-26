#!/usr/bin/env python3
"""
Mock Data Generator for Testing

Creates a small mock dataset with the same structure as iFood 2019
for testing the training pipeline without the full dataset.
"""

import os
import argparse
import random
from typing import List

import numpy as np
from PIL import Image
import pandas as pd


def create_mock_images(
    output_dir: str,
    num_images: int,
    img_size: int = 224,
    prefix: str = 'img'
) -> List[str]:
    """
    Create mock food images with random colors.
    
    Args:
        output_dir: Directory to save images
        num_images: Number of images to create
        img_size: Image size (square)
        prefix: Filename prefix
    
    Returns:
        List of created image filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    
    filenames = []
    
    for i in range(num_images):
        # Create random colored image
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        
        # Add some variation (simulate food texture)
        img_array = np.random.randint(
            max(0, color[0] - 30), min(255, color[0] + 30),
            (img_size, img_size), dtype=np.uint8
        )
        img_r = img_array
        
        img_array = np.random.randint(
            max(0, color[1] - 30), min(255, color[1] + 30),
            (img_size, img_size), dtype=np.uint8
        )
        img_g = img_array
        
        img_array = np.random.randint(
            max(0, color[2] - 30), min(255, color[2] + 30),
            (img_size, img_size), dtype=np.uint8
        )
        img_b = img_array
        
        img = Image.fromarray(np.stack([img_r, img_g, img_b], axis=2), mode='RGB')
        
        filename = f'{prefix}_{i:06d}.jpg'
        img.save(os.path.join(output_dir, filename), quality=85)
        filenames.append(filename)
    
    return filenames


def create_class_list(output_dir: str, num_classes: int = 251) -> List[str]:
    """
    Create mock class list file.
    
    Args:
        output_dir: Directory to save file
        num_classes: Number of classes
    
    Returns:
        List of class names
    """
    # Sample food categories
    food_categories = [
        'pizza', 'burger', 'sushi', 'pasta', 'salad', 'soup', 'sandwich',
        'steak', 'chicken', 'fish', 'rice', 'noodles', 'bread', 'cake',
        'ice_cream', 'donut', 'cookie', 'pie', 'pancake', 'waffle',
        'taco', 'burrito', 'curry', 'ramen', 'pho', 'dim_sum', 'dumpling',
        'spring_roll', 'tempura', 'sashimi', 'maki', 'nigiri', 'pad_thai',
        'fried_rice', 'lo_mein', 'chow_mein', 'orange_chicken', 'beef_broccoli',
        'kung_pao', 'sweet_sour', 'general_tso', 'mongolian_beef', 'teriyaki'
    ]
    
    class_names = []
    for i in range(num_classes):
        if i < len(food_categories):
            name = food_categories[i]
        else:
            name = f'food_class_{i}'
        class_names.append(name)
    
    # Save class list
    class_file = os.path.join(output_dir, 'class_list.txt')
    with open(class_file, 'w') as f:
        for i, name in enumerate(class_names):
            f.write(f'{i} {name}\n')
    
    return class_names


def create_mock_dataset(
    output_dir: str,
    num_classes: int = 251,
    train_per_class: int = 10,
    val_per_class: int = 2,
    test_per_class: int = 3,
    img_size: int = 224
):
    """
    Create complete mock dataset.
    
    Args:
        output_dir: Root directory for dataset
        num_classes: Number of classes
        train_per_class: Training samples per class
        val_per_class: Validation samples per class
        test_per_class: Test samples per class
        img_size: Image size
    """
    print(f"Creating mock dataset in: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create class list
    print("Creating class list...")
    class_names = create_class_list(output_dir, num_classes)
    
    # Create splits
    splits = {
        'train': train_per_class * num_classes,
        'val': val_per_class * num_classes,
        'test': test_per_class * num_classes
    }
    
    for split, num_images in splits.items():
        print(f"\nCreating {split} split ({num_images} images)...")
        
        # Create images
        img_dir = os.path.join(output_dir, f'{split}_images')
        filenames = create_mock_images(img_dir, num_images, img_size, f'{split}')
        
        # Create annotations
        if split == 'test':
            # Test set has no labels
            df = pd.DataFrame({'image_name': filenames})
        else:
            # Distribute images across classes
            labels = []
            for i in range(num_images):
                labels.append(i % num_classes)
            
            df = pd.DataFrame({
                'image_name': filenames,
                'label': labels
            })
        
        # Save annotations
        csv_path = os.path.join(output_dir, f'{split}_info.csv')
        df.to_csv(csv_path, header=False, index=False)
        print(f"  Created: {len(filenames)} images, saved to {csv_path}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Mock Dataset Created!")
    print("=" * 50)
    print(f"Location: {output_dir}")
    print(f"Classes: {num_classes}")
    print(f"Train images: {splits['train']}")
    print(f"Val images: {splits['val']}")
    print(f"Test images: {splits['test']}")
    print("=" * 50)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create mock iFood 2019 dataset')
    
    parser.add_argument('--output_dir', type=str, default='./data_mock',
                       help='Output directory for mock dataset')
    parser.add_argument('--num_classes', type=int, default=251,
                       help='Number of classes')
    parser.add_argument('--train_per_class', type=int, default=10,
                       help='Training samples per class')
    parser.add_argument('--val_per_class', type=int, default=2,
                       help='Validation samples per class')
    parser.add_argument('--test_per_class', type=int, default=3,
                       help='Test samples per class')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Image size')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    create_mock_dataset(
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        img_size=args.img_size
    )
