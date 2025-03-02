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


### Requirements

See requirements.txt for the complete list of dependencies.

### Usage

To use the dataset with either denoising method:

```python
from src.dataset import BUSIDataset, create_data_loaders

# Using Gaussian filtering
dataset_gaussian = BUSIDataset(
    data_dir="Dataset_BUSI_with_GT/benign",
    denoising_method="gaussian"
)

# Using Bilateral filtering
dataset_bilateral = BUSIDataset(
    data_dir="Dataset_BUSI_with_GT/benign",
    denoising_method="bilateral"
)

# Create data loaders
train_loader, val_loader = create_data_loaders(
    data_dir="Dataset_BUSI_with_GT/benign",
    batch_size=8,
    denoising_method="bilateral"  # or "gaussian"
)
