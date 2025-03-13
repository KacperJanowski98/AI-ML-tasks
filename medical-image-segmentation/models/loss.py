import torch
import torch.nn.functional as F


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
