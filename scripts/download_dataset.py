#!/usr/bin/env python3
"""
Script untuk download dan extract iFood 2019 dataset

The iFood 2019 dataset is large (~100GB). This script helps verify
and organize the dataset after manual download.
"""

import os
import sys
import tarfile
from typing import Optional

try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False


def verify_dataset_structure(data_dir: str) -> bool:
    """
    Verify that dataset has correct structure.
    
    Args:
        data_dir: Root dataset directory
    
    Returns:
        True if structure is valid, False otherwise
    """
    required_items = [
        ('train_images', 'dir'),
        ('val_images', 'dir'),
        ('test_images', 'dir'),
        ('train_info.csv', 'file'),
        ('val_info.csv', 'file'),
        ('test_info.csv', 'file'),
        ('class_list.txt', 'file')
    ]
    
    missing = []
    
    for item, item_type in required_items:
        path = os.path.join(data_dir, item)
        
        if item_type == 'dir':
            if not os.path.isdir(path):
                missing.append(f"Directory: {item}")
        else:
            if not os.path.isfile(path):
                missing.append(f"File: {item}")
    
    if missing:
        print("Missing items:")
        for item in missing:
            print(f"  ✗ {item}")
        return False
    
    return True


def count_images(data_dir: str) -> dict:
    """Count images in each split."""
    counts = {}
    
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(data_dir, f'{split}_images')
        if os.path.exists(img_dir):
            counts[split] = len([f for f in os.listdir(img_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            counts[split] = 0
    
    return counts


def extract_tar_gz(tar_path: str, extract_to: str):
    """Extract tar.gz file."""
    print(f"Extracting: {tar_path}")
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    
    print(f"✓ Extracted to: {extract_to}")


def download_from_gdrive(file_id: str, output_path: str):
    """Download file from Google Drive."""
    if not HAS_GDOWN:
        print("Error: gdown not installed. Install with: pip install gdown")
        return False
    
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)
    return True


def setup_dataset(data_dir: str):
    """
    Interactive dataset setup.
    
    Args:
        data_dir: Target directory for dataset
    """
    print("=" * 70)
    print("iFood 2019 Dataset Setup")
    print("=" * 70)
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if already exists
    if verify_dataset_structure(data_dir):
        print("\n✓ Dataset structure verified!")
        counts = count_images(data_dir)
        print("\nImage counts:")
        for split, count in counts.items():
            print(f"  {split}: {count:,} images")
        return True
    
    # Dataset not found - provide instructions
    print("\n" + "-" * 70)
    print("Dataset not found or incomplete.")
    print("-" * 70)
    
    print("""
The iFood 2019 dataset is approximately 100GB and requires manual download.

OPTION 1: Download from Official Source
----------------------------------------
1. Visit: https://github.com/karansikka1/iFood_2019
2. Follow the download instructions
3. Extract files to: {data_dir}

OPTION 2: Download from Kaggle
------------------------------
1. Visit: https://www.kaggle.com/c/ifood-2019-fgvc6/data
2. Download all files
3. Extract to: {data_dir}

OPTION 3: Use Google Drive (if you have a copy)
------------------------------------------------
1. Upload dataset to your Google Drive
2. Update the file IDs in this script
3. Run download function

Required Structure:
-------------------
{data_dir}/
├── train_images/      (training images)
├── val_images/        (validation images)
├── test_images/       (test images)
├── train_info.csv     (training labels)
├── val_info.csv       (validation labels)
├── test_info.csv      (test image names)
└── class_list.txt     (class names)
""".format(data_dir=data_dir))
    
    print("-" * 70)
    
    # Offer to create mock data
    print("\nFor testing, you can create a mock dataset:")
    print("  python src/create_mock_data.py --output_dir", data_dir)
    
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup iFood 2019 dataset')
    parser.add_argument('--data_dir', type=str, 
                       default='./data',
                       help='Dataset directory')
    parser.add_argument('--verify_only', action='store_true',
                       help='Only verify existing dataset')
    
    args = parser.parse_args()
    
    if args.verify_only:
        if verify_dataset_structure(args.data_dir):
            print("✓ Dataset structure is valid")
            counts = count_images(args.data_dir)
            print("\nImage counts:")
            for split, count in counts.items():
                print(f"  {split}: {count:,} images")
        else:
            print("✗ Dataset structure is invalid")
            sys.exit(1)
    else:
        setup_dataset(args.data_dir)


if __name__ == "__main__":
    main()
