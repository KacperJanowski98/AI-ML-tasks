# Medical Image Segmentation with U-Net

## Overview

Medical image segmentation is a critical component in computer-aided diagnosis systems, enabling the automatic identification of anatomical structures or abnormalities. This task focuses on implementing and optimizing a U-Net architecture for segmenting structures in ultrasound images. You'll explore preprocessing techniques to handle the unique challenges of ultrasound images (e.g., speckle noise, low contrast) and implement optimization strategies for deployment on resource-constrained devices.

## Objectives

1. Implement a U-Net architecture for medical image segmentation
2. Develop effective preprocessing techniques for ultrasound images
3. Apply at least one optimization technique to improve inference speed on edge devices
4. Evaluate the model's performance using appropriate metrics for segmentation tasks

## Technical Requirements

### 1. Dataset and Preprocessing

You may use a small medical dataset from one of these sources:
- A subset of the BUSI (Breast Ultrasound Images) dataset
- Sample ultrasound images from The Cancer Imaging Archive (TCIA)
- Any publicly available ultrasound dataset with segmentation masks

The focus is on implementing proper preprocessing techniques, such as:

```python
import cv2
import numpy as np
import pydicom  # For DICOM files, if applicable

def preprocess_ultrasound_image(image, target_size=(256, 256)):
    """
    Preprocess ultrasound image for segmentation
    
    Parameters:
        image: Input image (numpy array)
        target_size: Output image size
        
    Returns:
        Preprocessed image ready for model input
    """
    # Ensure grayscale
    if len(image.shape) > 2 and image.shape[2] > 1:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Step 1: Denoise (speckle noise reduction)
    # Option 1: Simple Gaussian filter
    denoised = cv2.GaussianBlur(image, (3, 3), 0)
    
    # Option 2: Non-Local Means denoising (better but slower)
    # denoised = cv2.fastNlMeansDenoising(image, None, h=10, searchWindowSize=21, templateWindowSize=7)
    
    # Step 2: Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised.astype(np.uint8))
    
    # Step 3: Normalize pixel values
    normalized = enhanced / 255.0
    
    # Step 4: Resize to target dimensions
    resized = cv2.resize(normalized, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Add channel dimension if needed
    if len(resized.shape) == 2:
        resized = np.expand_dims(resized, axis=0)  # Add channel dimension
    
    return resized
```

Implement and experiment with at least one additional preprocessing technique beyond the basic ones shown above. Options include:
- Specialized speckle noise filters (e.g., Lee, Frost, or Kuan filters)
- Adaptive histogram equalization variants
- Edge-preserving smoothing techniques

### 2. U-Net Implementation

Implement a U-Net architecture for segmentation. Your implementation should include:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # First encoder block doesn't follow a pooling
        self.encoder.append(DoubleConv(in_channels, features[0]))
        
        # Rest of encoder blocks
        for i in range(1, len(features)):
            self.encoder.append(DoubleConv(features[i-1], features[i]))
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        self.upconv = nn.ModuleList()
        
        for i in range(len(features)-1, 0, -1):
            self.upconv.append(
                nn.ConvTranspose2d(features[i], features[i-1], kernel_size=2, stride=2)
            )
            self.decoder.append(
                DoubleConv(features[i], features[i-1])
            )
        
        # Final 1x1 convolution to produce segmentation map
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Store encoder outputs for skip connections
        skip_connections = []
        
        # Encoder path
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Remove the last skip connection (the one before bottleneck)
        skip_connections = skip_connections[:-1]
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        # Decoder path with skip connections
        for i, decoder_block in enumerate(self.decoder):
            x = self.upconv[i](x)
            skip = skip_connections[i]
            
            # Handle case when dimensions don't match exactly
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                
            x = torch.cat((skip, x), dim=1)  # Skip connection
            x = decoder_block(x)
        
        # Final 1x1 convolution
        return self.final_conv(x)
```

Extend or modify the U-Net architecture in at least one of these ways:
- Add attention mechanisms to improve feature focusing
- Implement residual connections in encoder/decoder blocks
- Test different activation functions
- Experiment with different normalization techniques

### 3. Training and Loss Functions

Implement appropriate loss functions for segmentation tasks:

```python
def dice_loss(pred, target, smooth=1.0):
    """
    Dice loss for segmentation tasks
    """
    pred = torch.sigmoid(pred)  # For binary segmentation
    
    # Flatten predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return 1 - dice

def combined_loss(pred, target, alpha=0.5, smooth=1.0):
    """
    Combination of BCE and Dice loss
    """
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target, smooth)
    
    return alpha * bce + (1 - alpha) * dice
```

### 4. Model Optimization for Edge Deployment

Implement at least one model optimization technique for deployment on edge devices:

```python
def optimize_unet_for_edge(model, example_input, optimization_type="quantization"):
    """
    Optimize U-Net model for edge deployment
    
    Parameters:
        model: Trained PyTorch U-Net model
        example_input: Example input tensor for tracing
        optimization_type: Type of optimization to apply
        
    Returns:
        Optimized model
    """
    import torch.quantization
    
    if optimization_type == "quantization":
        # Post-training static quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d}, dtype=torch.qint8
        )
        return quantized_model
        
    elif optimization_type == "pruning":
        # Simple magnitude-based pruning
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))
        
        from torch.nn.utils import prune
        
        # Prune 20% of connections
        for module, name in parameters_to_prune:
            prune.l1_unstructured(module, name=name, amount=0.2)
            
        return model
        
    elif optimization_type == "onnx_export":
        # Convert to ONNX format for deployment
        torch.onnx.export(
            model, 
            example_input,
            "unet_model.onnx",
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        return "Model exported to ONNX format: unet_model.onnx"
```

In addition to the example code, consider implementing:
- Model pruning with fine-tuning to recover accuracy
- Knowledge distillation to a smaller model
- More advanced quantization techniques (e.g., quantization-aware training)
- Optimized inference using ONNX Runtime, TensorRT, or OpenVINO

### 5. Evaluation Metrics and Visualization

Implement functions to evaluate your segmentation model:

```python
def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculate segmentation metrics
    
    Parameters:
        pred: Predicted probabilities
        target: Ground truth binary mask
        threshold: Threshold for binary prediction
        
    Returns:
        Dictionary of metrics
    """
    # Convert predictions to binary
    pred_binary = (pred > threshold).float()
    
    # Flatten tensors
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target.view(-1).cpu().numpy()
    
    # Calculate metrics
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    # Dice coefficient (F1)
    dice = (2. * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-6)
    
    # IoU (Jaccard index)
    iou = intersection / (union + 1e-6)
    
    # Precision & Recall
    precision = intersection / (pred_flat.sum() + 1e-6)
    recall = intersection / (target_flat.sum() + 1e-6)
    
    return {
        'dice': dice,
        'iou': iou,
        'precision': precision,
        'recall': recall
    }

def visualize_results(image, true_mask, pred_mask, threshold=0.5):
    """
    Visualize segmentation results
    
    Parameters:
        image: Original image
        true_mask: Ground truth mask
        pred_mask: Predicted mask probabilities
        threshold: Threshold for binary prediction
    """
    import matplotlib.pyplot as plt
    
    # Convert to numpy and normalize if needed
    if torch.is_tensor(image):
        image = image.cpu().numpy().squeeze()
    if torch.is_tensor(true_mask):
        true_mask = true_mask.cpu().numpy().squeeze()
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.cpu().numpy().squeeze()
    
    # Binarize prediction
    pred_binary = (pred_mask > threshold).astype(np.float32)
    
    # Create RGB versions for overlay
    true_rgb = np.zeros((*true_mask.shape, 3), dtype=np.float32)
    true_rgb[..., 0] = true_mask  # Red channel for true mask
    
    pred_rgb = np.zeros((*pred_binary.shape, 3), dtype=np.float32)
    pred_rgb[..., 1] = pred_binary  # Green channel for prediction
    
    # Combine for overlay
    overlay = np.zeros((*true_mask.shape, 3), dtype=np.float32)
    overlay[..., 0] = true_mask  # Red channel for true mask
    overlay[..., 1] = pred_binary  # Green channel for prediction
    # Yellow indicates overlap
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[2].imshow(pred_binary, cmap='gray')
    axes[2].set_title('Prediction')
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (Yellow=Overlap)')
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
```

## Expected Deliverables

1. **Code Repository**:
   - Complete implementation of U-Net architecture
   - Data loading and preprocessing pipeline
   - Training and evaluation scripts
   - Model optimization for edge deployment
   - README with instructions to run your code

2. **Technical Report** (markdown or PDF):
   - Description of the dataset and preprocessing techniques
   - Explanation of your architectural choices and implementations
   - Analysis of model performance (including segmentation metrics)
   - Comparison of optimization techniques and their impact on performance
   - Discussion of challenges and potential improvements

3. **Visualization**:
   - Sample images showing original, ground truth, and predicted segmentations
   - Learning curves (loss and metrics over epochs)
   - Visualization of the optimization impact on memory/speed

## Evaluation Criteria

Your solution will be evaluated based on:

1. **Technical Implementation**: Quality of code, architecture design, and proper implementation of U-Net
2. **Preprocessing Effectiveness**: Appropriate handling of ultrasound image challenges
3. **Model Performance**: Segmentation accuracy measured by appropriate metrics (Dice, IoU)
4. **Optimization Success**: Effectiveness of the optimization techniques for edge deployment
5. **Documentation Quality**: Clarity of explanation and analysis in the technical report

## Time Expectation

- This task should take approximately 4-6 hours for an experienced ML engineer
- Focus on implementing the core architecture and one optimization technique thoroughly rather than attempting all suggested extensions if time is limited

## Bonus Challenges

If you complete the main tasks and want to push further:

1. Implement data augmentation techniques specific to ultrasound images
2. Compare standard U-Net with a variant like Attention U-Net or U-Net++
3. Design a simple interface to visualize the segmentation results in real-time
4. Benchmark your optimized model on an actual edge device or simulate resource constraints
