#!/usr/bin/env python3
"""
Script untuk sync results ke Google Drive

Provides utilities for syncing training results,
checkpoints, and outputs to Google Drive for persistence.
"""

import os
import shutil
from datetime import datetime
from typing import List, Optional


def sync_directory(
    source_dir: str,
    dest_dir: str,
    extensions: Optional[List[str]] = None,
    overwrite: bool = True,
    verbose: bool = True
) -> int:
    """
    Sync directory contents to destination.
    
    Args:
        source_dir: Source directory
        dest_dir: Destination directory
        extensions: Optional list of file extensions to sync (e.g., ['.pth', '.json'])
        overwrite: Whether to overwrite existing files
        verbose: Print progress messages
    
    Returns:
        Number of files synced
    """
    if not os.path.exists(source_dir):
        if verbose:
            print(f"Warning: Source directory does not exist: {source_dir}")
        return 0
    
    os.makedirs(dest_dir, exist_ok=True)
    
    synced_count = 0
    
    for root, dirs, files in os.walk(source_dir):
        # Calculate relative path
        rel_path = os.path.relpath(root, source_dir)
        dest_root = os.path.join(dest_dir, rel_path) if rel_path != '.' else dest_dir
        
        # Create subdirectories
        os.makedirs(dest_root, exist_ok=True)
        
        for filename in files:
            # Filter by extension if specified
            if extensions:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in extensions:
                    continue
            
            src_file = os.path.join(root, filename)
            dest_file = os.path.join(dest_root, filename)
            
            # Check if should copy
            should_copy = overwrite or not os.path.exists(dest_file)
            
            if should_copy:
                shutil.copy2(src_file, dest_file)
                synced_count += 1
                if verbose:
                    print(f"  Synced: {filename}")
    
    return synced_count


def sync_checkpoints(
    checkpoint_dir: str,
    drive_checkpoint_dir: str,
    sync_best_only: bool = False,
    verbose: bool = True
):
    """
    Sync model checkpoints to Google Drive.
    
    Args:
        checkpoint_dir: Local checkpoint directory
        drive_checkpoint_dir: Google Drive checkpoint directory
        sync_best_only: Only sync *_best.pth files
        verbose: Print progress
    """
    if verbose:
        print(f"\nSyncing checkpoints to Drive...")
    
    if sync_best_only:
        extensions = None  # We'll filter manually
        
        # Only copy *_best.pth files
        synced = 0
        for filename in os.listdir(checkpoint_dir):
            if filename.endswith('_best.pth') or filename.endswith('_history.json'):
                src = os.path.join(checkpoint_dir, filename)
                dest = os.path.join(drive_checkpoint_dir, filename)
                shutil.copy2(src, dest)
                synced += 1
                if verbose:
                    print(f"  Synced: {filename}")
    else:
        synced = sync_directory(
            checkpoint_dir,
            drive_checkpoint_dir,
            extensions=['.pth', '.pt', '.json'],
            verbose=verbose
        )
    
    if verbose:
        print(f"✓ Synced {synced} checkpoint files")


def sync_results(
    eval_dir: str,
    analysis_dir: str,
    drive_eval_dir: str,
    drive_analysis_dir: str,
    verbose: bool = True
):
    """
    Sync evaluation and analysis results to Drive.
    
    Args:
        eval_dir: Local evaluation results directory
        analysis_dir: Local analysis results directory
        drive_eval_dir: Drive evaluation directory
        drive_analysis_dir: Drive analysis directory
        verbose: Print progress
    """
    if verbose:
        print(f"\nSyncing results to Drive...")
    
    # Sync evaluation results
    eval_synced = sync_directory(
        eval_dir,
        drive_eval_dir,
        extensions=['.json', '.csv', '.npy'],
        verbose=verbose
    )
    
    # Sync analysis results
    analysis_synced = sync_directory(
        analysis_dir,
        drive_analysis_dir,
        extensions=['.png', '.jpg', '.csv', '.md', '.json'],
        verbose=verbose
    )
    
    if verbose:
        print(f"✓ Synced {eval_synced} evaluation files")
        print(f"✓ Synced {analysis_synced} analysis files")


def create_backup(
    project_path: str,
    backup_dir: str,
    include_checkpoints: bool = True,
    verbose: bool = True
) -> str:
    """
    Create timestamped backup of project.
    
    Args:
        project_path: Source project directory
        backup_dir: Backup destination directory
        include_checkpoints: Whether to include model checkpoints
        verbose: Print progress
    
    Returns:
        Path to created backup
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f'backup_{timestamp}'
    backup_path = os.path.join(backup_dir, backup_name)
    
    if verbose:
        print(f"\nCreating backup: {backup_path}")
    
    os.makedirs(backup_path, exist_ok=True)
    
    # Directories to backup
    dirs_to_backup = ['evaluation_results', 'analysis_results']
    
    if include_checkpoints:
        dirs_to_backup.append('checkpoints')
    
    for dir_name in dirs_to_backup:
        src = os.path.join(project_path, dir_name)
        dest = os.path.join(backup_path, dir_name)
        
        if os.path.exists(src):
            shutil.copytree(src, dest)
            if verbose:
                print(f"  Backed up: {dir_name}")
    
    if verbose:
        print(f"✓ Backup created: {backup_path}")
    
    return backup_path


def full_sync(
    repo_path: str = '/content/alexnet-ifood2019',
    drive_path: str = '/content/drive/MyDrive/AlexNet_iFood2019',
    verbose: bool = True
):
    """
    Full sync of all results to Google Drive.
    
    Args:
        repo_path: Path to local repository
        drive_path: Path to project folder in Drive
        verbose: Print progress
    """
    if verbose:
        print("=" * 60)
        print("Syncing to Google Drive")
        print("=" * 60)
    
    # Sync checkpoints
    sync_checkpoints(
        os.path.join(repo_path, 'checkpoints'),
        os.path.join(drive_path, 'checkpoints'),
        verbose=verbose
    )
    
    # Sync results
    sync_results(
        os.path.join(repo_path, 'evaluation_results'),
        os.path.join(repo_path, 'analysis_results'),
        os.path.join(drive_path, 'evaluation_results'),
        os.path.join(drive_path, 'analysis_results'),
        verbose=verbose
    )
    
    if verbose:
        print("\n" + "=" * 60)
        print("✓ Sync Complete!")
        print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Sync results to Google Drive')
    parser.add_argument('--repo_path', type=str, 
                       default='/content/alexnet-ifood2019',
                       help='Path to repository')
    parser.add_argument('--drive_path', type=str,
                       default='/content/drive/MyDrive/AlexNet_iFood2019',
                       help='Path to Drive project folder')
    
    args = parser.parse_args()
    
    full_sync(args.repo_path, args.drive_path)
