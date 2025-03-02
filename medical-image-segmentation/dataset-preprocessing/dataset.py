"""
Dataset module for loading and preprocessing ultrasound images with segmentation masks.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional, Callable

from preprocessing import preprocess_ultrasound_image, preprocess_mask


class BUSIDataset(Dataset):
    """
    Dataset class for the BUSI (Breast Ultrasound Images) dataset.
    """
    
    def __init__(self, 
                 data_dir: str,
                 transform: Optional[Callable] = None,
                 target_size: Tuple[int, int] = (256, 256),
                 denoising_method: str = "gaussian",
                 limit_samples: Optional[int] = None):
        """
        Initialize the BUSI dataset.
        
        Args:
            data_dir: Directory containing the BUSI dataset
            transform: Optional additional transformations
            target_size: Size to resize images and masks to
            denoising_method: Method to use for denoising ("gaussian" or "bilateral")
            limit_samples: Limit dataset to first N samples (for debugging)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.denoising_method = denoising_method
        
        # Get all image file names
        self.image_files = []
        self.mask_files = []
        
        # Look for pairs of images and masks
        all_files = sorted(os.listdir(data_dir))
        
        # Find all image files that have corresponding masks
        for file_name in all_files:
            if "_mask" not in file_name and file_name.endswith((".png", ".jpg", ".jpeg")):
                # Construct the expected mask filename
                mask_name = os.path.splitext(file_name)[0] + "_mask" + os.path.splitext(file_name)[1]
                
                # Check if mask exists
                if mask_name in all_files:
                    self.image_files.append(os.path.join(data_dir, file_name))
                    self.mask_files.append(os.path.join(data_dir, mask_name))
        
        # Limit samples if specified
        if limit_samples is not None:
            self.image_files = self.image_files[:limit_samples]
            self.mask_files = self.mask_files[:limit_samples]
            
        print(f"Found {len(self.image_files)} image-mask pairs in {data_dir}")
        print(f"Using {denoising_method} denoising method")
            
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the image and mask tensors
        """
        # Load image and mask
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        # Read image and mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Preprocess image and mask
        processed_image = preprocess_ultrasound_image(
            image, 
            target_size=self.target_size,
            denoising_method=self.denoising_method
        )
        
        processed_mask = preprocess_mask(
            mask, 
            target_size=self.target_size
        )
        
        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(processed_image)
        mask_tensor = torch.from_numpy(processed_mask)
        
        # Apply additional transforms if specified
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'image_path': image_path,
            'mask_path': mask_path
        }
    
    def visualize_sample(self, idx: int, figsize: Tuple[int, int] = (12, 4)) -> None:
        """
        Visualize a sample from the dataset.
        
        Args:
            idx: Index of the sample to visualize
            figsize: Figure size
        """
        sample = self[idx]
        image = sample['image'].numpy()
        mask = sample['mask'].numpy()
        
        # Convert from CHW to HWC if needed
        if image.shape[0] == 1:  # If single channel
            image = image[0]  # Remove channel dimension for visualization
        elif image.shape[0] == 3:  # If RGB
            image = np.transpose(image, (1, 2, 0))
            
        if mask.shape[0] == 1:  # If single channel
            mask = mask[0]  # Remove channel dimension for visualization
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f'Preprocessed Image ({self.denoising_method})')
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Mask')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(image, cmap='gray')
        axes[2].imshow(mask, alpha=0.3, cmap='Reds')
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
        print(f"Mask unique values: {np.unique(mask)}")


def create_data_loaders(
    data_dir: str,
    batch_size: int = 8,
    train_val_split: float = 0.8,
    target_size: Tuple[int, int] = (256, 256),
    denoising_method: str = "gaussian",
    num_workers: int = 4,
    limit_samples: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for the data loaders
        train_val_split: Fraction of data to use for training
        target_size: Size to resize images to
        denoising_method: Method to use for denoising ("gaussian" or "bilateral")
        num_workers: Number of workers for data loading
        limit_samples: Limit dataset to first N samples (for debugging)
        
    Returns:
        Training and validation data loaders
    """
    # Create the dataset
    dataset = BUSIDataset(
        data_dir=data_dir,
        target_size=target_size,
        denoising_method=denoising_method,
        limit_samples=limit_samples
    )
    
    # Split into training and validation sets
    dataset_size = len(dataset)
    train_size = int(train_val_split * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"Created data loaders with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    return train_loader, val_loader
