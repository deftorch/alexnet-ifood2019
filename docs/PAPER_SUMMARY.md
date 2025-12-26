# AlexNet Paper Summary

## Paper: ImageNet Classification with Deep Convolutional Neural Networks

**Authors**: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton  
**Year**: 2012  
**Conference**: NIPS  

## Key Contributions

1. **Deep CNN Architecture**: First large-scale CNN to win ImageNet with significant margin
2. **GPU Training**: Efficient implementation on GPUs
3. **ReLU Activation**: Popularized ReLU for faster training
4. **Dropout Regularization**: Introduced dropout to reduce overfitting
5. **Data Augmentation**: Extensive augmentation techniques

## Architecture Details

| Layer | Type | Filters | Size | Stride | Output |
|-------|------|---------|------|--------|--------|
| Input | - | - | - | - | 224×224×3 |
| Conv1 | Conv | 96 | 11×11 | 4 | 55×55×96 |
| Pool1 | MaxPool | - | 3×3 | 2 | 27×27×96 |
| Conv2 | Conv | 256 | 5×5 | 1 | 27×27×256 |
| Pool2 | MaxPool | - | 3×3 | 2 | 13×13×256 |
| Conv3 | Conv | 384 | 3×3 | 1 | 13×13×384 |
| Conv4 | Conv | 384 | 3×3 | 1 | 13×13×384 |
| Conv5 | Conv | 256 | 3×3 | 1 | 13×13×256 |
| Pool5 | MaxPool | - | 3×3 | 2 | 6×6×256 |
| FC6 | Dense | 4096 | - | - | 4096 |
| FC7 | Dense | 4096 | - | - | 4096 |
| FC8 | Dense | 1000 | - | - | 1000 |

## Key Techniques

### ReLU Activation
- Formula: `f(x) = max(0, x)`
- Faster training compared to tanh/sigmoid
- Solves vanishing gradient problem

### Local Response Normalization (LRN)
- Normalizes across adjacent feature maps
- Encourages competition between neurons

### Overlapping Pooling
- Pool size 3×3 with stride 2
- Reduces overfitting compared to non-overlapping

### Dropout
- Rate: 0.5 in FC layers
- Prevents co-adaptation of neurons

### Data Augmentation
- Random crops (224×224 from 256×256)
- Horizontal flipping
- PCA color augmentation

## Training Details

- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 0.01, divided by 10 when validation error plateaus
- **Weight Decay**: 0.0005
- **Batch Size**: 128
- **Epochs**: ~90
- **GPUs**: 2× NVIDIA GTX 580 (3GB each)

## Results on ImageNet

| Metric | Top-1 Error | Top-5 Error |
|--------|-------------|-------------|
| AlexNet | 37.5% | 17.0% |
| Previous SOTA | 45.7% | 25.7% |

## Relevance to iFood 2019

- **Transfer Learning**: AlexNet features transfer well
- **Fine-tuning**: FC layers adapted for 251 classes
- **Modifications**: BatchNorm can replace LRN for better convergence

## References

```bibtex
@inproceedings{krizhevsky2012imagenet,
  title={ImageNet classification with deep convolutional neural networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle={Advances in neural information processing systems},
  pages={1097--1105},
  year={2012}
}
```
