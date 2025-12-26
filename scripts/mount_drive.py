#!/usr/bin/env python3
"""
Script untuk mount Google Drive di Colab

Provides utility functions for mounting Google Drive
and setting up the project environment in Colab.
"""

import os
import sys


def mount_google_drive(mount_path: str = '/content/drive') -> str:
    """
    Mount Google Drive to Colab.
    
    Args:
        mount_path: Path where drive will be mounted
        
    Returns:
        project_path: Path to project folder in drive
    """
    try:
        from google.colab import drive
    except ImportError:
        print("Warning: Not running in Google Colab")
        print("This script is designed for Colab environment")
        return None
    
    print("Mounting Google Drive...")
    drive.mount(mount_path, force_remount=True)
    
    # Project folder path
    project_path = os.path.join(mount_path, 'MyDrive', 'AlexNet_iFood2019')
    
    # Create folder structure if not exists
    folders = [
        'dataset',
        'checkpoints',
        'evaluation_results',
        'analysis_results',
        'logs',
        'notebooks'
    ]
    
    for folder in folders:
        folder_path = os.path.join(project_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"✓ Created/verified: {folder}")
    
    print(f"\n✓ Drive mounted successfully!")
    print(f"Project path: {project_path}")
    
    return project_path


def setup_environment(project_path: str, repo_path: str = None):
    """
    Setup Python path and working directory.
    
    Args:
        project_path: Path to project in Google Drive
        repo_path: Path to cloned repository (optional)
    """
    # Add src to Python path
    if repo_path:
        src_path = os.path.join(repo_path, 'src')
    else:
        src_path = os.path.join(project_path, 'src')
    
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Also add repo root
    if repo_path and repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    
    print(f"✓ Python path updated")
    print(f"  Added: {src_path}")


def create_symlinks(project_path: str, repo_path: str):
    """
    Create symbolic links from repo to Drive folders.
    
    This allows data to persist across Colab sessions.
    
    Args:
        project_path: Path to project in Google Drive
        repo_path: Path to cloned repository
    """
    links = {
        'data': 'dataset',
        'checkpoints': 'checkpoints',
        'evaluation_results': 'evaluation_results',
        'analysis_results': 'analysis_results'
    }
    
    for local_name, drive_name in links.items():
        local_path = os.path.join(repo_path, local_name)
        drive_path = os.path.join(project_path, drive_name)
        
        # Remove existing local folder/link
        if os.path.islink(local_path):
            os.unlink(local_path)
        elif os.path.exists(local_path):
            import shutil
            shutil.rmtree(local_path)
        
        # Create symlink
        os.symlink(drive_path, local_path)
        print(f"✓ Linked: {local_name} -> {drive_path}")


def verify_gpu():
    """Verify GPU availability and print info."""
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"\n✓ GPU Available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("\n⚠ No GPU available - training will be slow")
            return False
    except ImportError:
        print("\n⚠ PyTorch not installed")
        return False


def full_setup():
    """
    Complete setup for Colab environment.
    
    Call this function at the start of Colab notebooks.
    """
    print("=" * 60)
    print("AlexNet iFood2019 - Colab Setup")
    print("=" * 60)
    
    # Mount Drive
    project_path = mount_google_drive()
    
    if project_path:
        # Clone repo if needed
        repo_path = '/content/alexnet-ifood2019'
        
        if not os.path.exists(repo_path):
            print("\nℹ Repository not found")
            print("Please clone with:")
            print("  !git clone https://github.com/YOUR_USERNAME/alexnet-ifood2019.git")
        else:
            # Setup environment
            setup_environment(project_path, repo_path)
            
            # Create symlinks
            create_symlinks(project_path, repo_path)
            
            # Change to repo directory
            os.chdir(repo_path)
            print(f"\n✓ Working directory: {os.getcwd()}")
        
        # Verify GPU
        verify_gpu()
        
        print("\n" + "=" * 60)
        print("Setup Complete!")
        print("=" * 60)
        
        return project_path, repo_path
    
    return None, None


if __name__ == "__main__":
    full_setup()
