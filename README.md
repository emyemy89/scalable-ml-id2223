# ID2223 – Lab 1: CNNs on CIFAR-10

## Overview

Implementation and evaluation of AlexNet and ResNet-18 on the CIFAR-10 image classification dataset using PyTorch.

The assignment investigates model architecture, capacity, regularization, robustness, and computational cost.

## Dataset we used:

- **Dataset:** CIFAR-10
- **Training set:** 40,000 images
- **Validation set:** 10,000 images
- **Test set:** 10,000 images

The training set is split into 40,000 training images and 10,000 validation images using `random_split` with seed 42. The test set is reserved for final evaluation.

Images are converted to tensors and normalized using the CIFAR-10 channel-wise mean and standard deviation.

## Models

Both architectures are implemented from scratch in PyTorch, following the CIFAR-10 adaptations described in section A9.

- **AlexNet:** Five convolutional layers followed by three fully connected layers.
- **ResNet-18:** Residual network with four stages of basic blocks and a global average pooling classification head.

## Dependencies

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install torch torchvision numpy matplotlib
```

## Reproducibility

The dataset split uses a fixed random seed. 
Model configurations, training settings, and evaluation procedures are documented in the accompanying code and report.