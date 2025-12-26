#!/bin/bash
# Setup script for AlexNet iFood2019 project

echo "=============================================="
echo "AlexNet iFood2019 - Setup"
echo "=============================================="

# Create virtual environment (optional)
if [ "$1" == "--venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo "Creating directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p evaluation_results
mkdir -p analysis_results

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import torch
import torchvision
print(f'PyTorch: {torch.__version__}')
print(f'TorchVision: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# Test model import
from src.models.alexnet import get_model
model = get_model('alexnet_baseline', num_classes=251)
print(f'Model loaded successfully')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Download the iFood 2019 dataset to ./data/"
echo "2. Run: python src/create_mock_data.py --output_dir data_mock (for testing)"
echo "3. Run: python src/train.py --data_dir data --model_name alexnet_baseline"
