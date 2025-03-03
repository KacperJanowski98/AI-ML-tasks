import torch
import numpy as np
import matplotlib.pyplot as plt


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
