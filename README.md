# AlexNet iFood 2019 Classification

Implementation of AlexNet variants for the iFood 2019 food classification challenge.

## 📋 Project Overview

This project implements and compares 4 variants of AlexNet architecture:
1. **Baseline AlexNet** - Original architecture adapted for 251 classes
2. **Modified AlexNet 1** - Enhanced with Batch Normalization
3. **Modified AlexNet 2** - With Dropout regularization
4. **Combined AlexNet** - BatchNorm + Dropout + improved architecture

## 🗂️ Project Structure

```
alexnet-ifood2019/
├── src/
│   ├── models/
│   │   └── alexnet.py      # AlexNet model variants
│   ├── data_loader.py      # Data loading utilities
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── analysis.py         # Comparative analysis
├── notebooks/
│   ├── 00_setup_and_verification.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_baseline.ipynb
│   ├── 03_train_all_models.ipynb
│   └── 04_analysis_and_visualization.ipynb
├── scripts/
│   ├── download_dataset.py
│   ├── mount_drive.py
│   └── sync_to_drive.py
└── docs/
    ├── PAPER_SUMMARY.md
    ├── FINAL_REPORT.md
    └── REPORT_FILLING_GUIDE.md
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. Upload this repository to GitHub
2. Open `notebooks/00_setup_and_verification.ipynb` in Colab
3. Follow the step-by-step instructions

### Option 2: Local Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/alexnet-ifood2019.git
cd alexnet-ifood2019

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py --data_dir data --model_name alexnet_baseline --num_epochs 50
```

## 📊 Dataset

This project uses the [iFood 2019 Dataset](https://github.com/karansikka1/iFood_2019):
- **251 food categories**
- **~120,000 training images**
- **~12,000 validation images**
- **~28,000 test images**

## 🏋️ Training

```bash
# Train baseline model
python src/train.py \
    --data_dir data \
    --model_name alexnet_baseline \
    --num_epochs 50 \
    --batch_size 128 \
    --lr 0.01

# Train all models
./run_experiments.sh
```

## 📈 Results

| Model | Val Accuracy | Top-5 Accuracy | Macro F1 |
|-------|-------------|----------------|----------|
| Baseline | [TBD] | [TBD] | [TBD] |
| Mod 1 (BatchNorm) | [TBD] | [TBD] | [TBD] |
| Mod 2 (Dropout) | [TBD] | [TBD] | [TBD] |
| Combined | [TBD] | [TBD] | [TBD] |

## 📝 License

MIT License

## 🙏 Acknowledgments

- [iFood 2019 Challenge](https://www.kaggle.com/c/ifood-2019-fgvc6)
- [AlexNet Paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
