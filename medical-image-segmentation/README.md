# Medical Image Segmentation with U-Net

## 1. Dataset and Preprocessing

This project implements a U-Net architecture for segmentation of breast ultrasound images using the BUSI dataset.

### Dataset

The BUSI (Breast Ultrasound Images) dataset contains ultrasound images of breast cancer cases with corresponding ground truth segmentation masks. The dataset includes:
- Benign cases: 437 images
- Malignant cases: 210 images
- Normal cases: 133 images

For this implementation, we focus on the benign ultrasound images located in `Dataset_BUSI_with_GT/benign/`.

### Preprocessing Pipeline

The preprocessing pipeline includes the following techniques:
1. **Grayscale Conversion**: Ensuring all images are in grayscale format
2. **Denoising**: Using *either* Gaussian filtering *or* Bilateral filtering (configurable)
3. **Contrast Enhancement**: Applying CLAHE (Contrast Limited Adaptive Histogram Equalization)
4. **Normalization**: Scaling pixel values to [0, 1] range
5. **Resizing**: Standardizing all images to 256×256 pixels

### Denoising Methods Comparison

We've implemented two alternative denoising techniques:

1. **Gaussian Filtering**: A traditional approach that applies uniform smoothing
2. **Bilateral Filtering**: An edge-preserving technique that reduces noise while maintaining important boundaries

Our experiments show that bilateral filtering is more selective about edge preservation, with metrics revealing:
- Lower average edge strength (more selective noise reduction)
- Similar overall brightness preservation
- ~2-3% mean absolute difference in pixel values compared to Gaussian filtering

Bilateral filtering appears particularly suitable for ultrasound images as it better preserves anatomical boundaries while reducing speckle noise.

## 2. U-Net Implementation with Residual Connections

We've implemented a U-Net architecture enhanced with residual connections to improve gradient flow and feature learning. The model is designed for efficient training and potential deployment on edge devices.

### Architecture Details

The U-Net architecture follows the classic encoder-decoder structure with the following enhancements:

1. **Residual Connections**: Added to both encoder and decoder blocks to:
   - Improve gradient flow during training
   - Mitigate the vanishing gradient problem
   - Enable more effective feature learning
   - Preserve information throughout the network

2. **Model Configuration**:
   - Input: Single-channel grayscale images (256×256)
   - Output: Single-channel binary segmentation mask
   - Feature dimensions: [64, 128, 256, 512]
   - Skip connections between encoder and decoder at each level

### Training Results

Our model achieved significant improvements across all segmentation metrics:

| Metric    | Before Training | After Training | Improvement Factor |
|-----------|----------------|----------------|-------------------|
| Dice      | 0.0803         | 0.7210         | 9.0× increase     |
| IoU       | 0.0418         | 0.5702         | 13.6× increase    |
| Precision | 0.0704         | 0.7926         | 11.3× increase    |
| Recall    | 0.0936         | 0.6709         | 7.2× increase     |

These results demonstrate strong segmentation performance, with:
- Dice coefficient >0.72 (excellent for medical ultrasound)
- High precision (79%) reducing false positives
- Good recall (67%) capturing most abnormal regions
- IoU >0.57 indicating majority overlap with ground truth

### Loss Functions

The implementation includes specialized loss functions for segmentation:

1. **Dice Loss**: Optimizes for region overlap, suitable for imbalanced segmentation tasks
2. **Combined Loss**: Weighted combination of Binary Cross-Entropy and Dice loss for balanced optimization

## Requirements

See requirements.txt for the complete list of dependencies.
