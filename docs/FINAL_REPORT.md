# AlexNet iFood 2019 - Final Report

## 1. Introduction

### 1.1 Background
[ISI: Jelaskan latar belakang masalah food classification dan pentingnya]

### 1.2 Dataset
- **Dataset**: iFood 2019 Fine-Grained Visual Categorization
- **Classes**: 251 food categories
- **Training samples**: [ISI jumlah]
- **Validation samples**: [ISI jumlah]
- **Test samples**: [ISI jumlah]

### 1.3 Objectives
1. Implement AlexNet baseline for food classification
2. Develop modified architectures with improvements
3. Compare performance across all variants
4. Analyze factors affecting model performance

---

## 2. Methodology

### 2.1 Model Architectures

#### 2.1.1 AlexNet Baseline
- Original AlexNet architecture adapted for 251 classes
- Uses Local Response Normalization (LRN)
- Dropout (0.5) in fully connected layers

#### 2.1.2 AlexNet Mod1 (Batch Normalization)
- Replaces LRN with Batch Normalization
- Improves training stability and convergence

#### 2.1.3 AlexNet Mod2 (Enhanced Dropout)
- Adds Dropout2D in convolutional layers
- Increases dropout rates in FC layers (0.6)

#### 2.1.4 AlexNet Combined
- Combines BatchNorm + Dropout
- Uses Leaky ReLU activation
- Additional FC layer for better feature extraction

### 2.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD |
| Learning Rate | 0.01 |
| Momentum | 0.9 |
| Weight Decay | 0.0005 |
| Batch Size | 128 |
| Epochs | 50 |
| LR Schedule | StepLR (γ=0.1 every 10 epochs) |

### 2.3 Data Augmentation
- Random Resized Crop (224×224)
- Random Horizontal Flip
- Random Rotation (±15°)
- Color Jitter

---

## 3. Results

### 3.1 Training Curves

[ISI: Masukkan gambar training curves dari analysis_results/]

### 3.2 Performance Comparison

| Model | Val Accuracy | Top-5 Accuracy | Macro F1 |
|-------|-------------|----------------|----------|
| Baseline | [ISI] | [ISI] | [ISI] |
| Mod1 (BN) | [ISI] | [ISI] | [ISI] |
| Mod2 (Dropout) | [ISI] | [ISI] | [ISI] |
| Combined | [ISI] | [ISI] | [ISI] |

### 3.3 Best Model
- **Model**: [ISI nama model terbaik]
- **Accuracy**: [ISI]
- **Improvement over baseline**: [ISI]

### 3.4 Confusion Matrix

[ISI: Masukkan gambar confusion matrix]

---

## 4. Analysis

### 4.1 Effect of Batch Normalization
[ISI: Analisis pengaruh BatchNorm terhadap konvergensi dan performa]

### 4.2 Effect of Dropout
[ISI: Analisis pengaruh Dropout terhadap overfitting]

### 4.3 Combined Improvements
[ISI: Analisis mengapa kombinasi memberikan hasil terbaik/terburuk]

### 4.4 Challenging Classes
[ISI: Kelas-kelas yang sulit dikenali dan analisis penyebabnya]

---

## 5. Conclusion

### 5.1 Summary
[ISI: Ringkasan hasil eksperimen]

### 5.2 Key Findings
1. [ISI: Temuan 1]
2. [ISI: Temuan 2]
3. [ISI: Temuan 3]

### 5.3 Future Work
- [ISI: Saran pengembangan lebih lanjut]

---

## References

1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. NIPS.
2. iFood 2019 Challenge: https://github.com/karansikka1/iFood_2019

---

## Appendix

### A. Hardware Specifications
- GPU: [ISI]
- Training Time: [ISI]

### B. Code Repository
- GitHub: [ISI URL]
- Colab Notebooks: [ISI link]
