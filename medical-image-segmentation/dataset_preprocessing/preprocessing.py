"""
Preprocessing module for medical image segmentation.
Implements various preprocessing techniques for ultrasound images.
"""

import cv2
import numpy as np
from typing import Tuple


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if it has multiple channels.
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        Grayscale image
    """
    if len(image.shape) > 2 and image.shape[2] > 1:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image


def apply_gaussian_filter(image: np.ndarray, kernel_size: int = 3, sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian filter for speckle noise reduction.
    
    Args:
        image: Input image (numpy array)
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation of Gaussian kernel
        
    Returns:
        Denoised image
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def apply_bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filter to reduce noise while preserving edges.
    
    Bilateral filtering smooths images while preserving edges by combining
    domain and range filtering. It considers both spatial proximity and 
    intensity similarity of pixels.
    
    Args:
        image: Input image (numpy array)
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        Edge-preserving denoised image
    """
    # Ensure image is in uint8 format as bilateral filter requires it
    if image.dtype != np.uint8:
        # Scale to [0, 255] if it's float
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
    
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Args:
        image: Input image (numpy array)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Contrast enhanced image
    """
    # Ensure image is in uint8 format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
        
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def normalize(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to range [0, 1].
    
    Args:
        image: Input image (numpy array)
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (256, 256), 
                interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize image to target dimensions.
    
    Args:
        image: Input image (numpy array)
        target_size: Output image dimensions (width, height)
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    return cv2.resize(image, target_size, interpolation=interpolation)


def preprocess_ultrasound_image(image: np.ndarray, target_size: Tuple[int, int] = (256, 256),
                              denoising_method: str = "gaussian") -> np.ndarray:
    """
    Complete preprocessing pipeline for ultrasound images.
    
    Args:
        image: Input image (numpy array)
        target_size: Output image dimensions (width, height)
        denoising_method: Method to use for denoising ("gaussian" or "bilateral")
        
    Returns:
        Preprocessed image ready for model input
    """
    # Ensure grayscale
    image = convert_to_grayscale(image)
    
    # Step 1: Apply denoising based on the selected method
    if denoising_method.lower() == "gaussian":
        denoised = apply_gaussian_filter(image, kernel_size=3)
    elif denoising_method.lower() == "bilateral":
        denoised = apply_bilateral_filter(image)
    else:
        raise ValueError(f"Unsupported denoising method: {denoising_method}. Choose 'gaussian' or 'bilateral'.")
    
    # Step 2: Contrast enhancement using CLAHE
    enhanced = apply_clahe(denoised)
    
    # Step 3: Normalize pixel values
    normalized = normalize(enhanced)
    
    # Step 4: Resize to target dimensions
    resized = resize_image(normalized, target_size)
    
    # Add channel dimension for PyTorch if needed
    if len(resized.shape) == 2:
        resized = np.expand_dims(resized, axis=0)  # Add channel dimension
    
    return resized


def preprocess_mask(mask: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """
    Preprocess segmentation mask.
    
    Args:
        mask: Input mask (numpy array)
        target_size: Output mask dimensions (width, height)
        
    Returns:
        Preprocessed binary mask
    """
    # Ensure grayscale
    mask = convert_to_grayscale(mask)
    
    # Resize mask
    resized = resize_image(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Binarize the mask (threshold)
    _, binary_mask = cv2.threshold(resized, 127, 1, cv2.THRESH_BINARY)
    
    # Add channel dimension for PyTorch
    if len(binary_mask.shape) == 2:
        binary_mask = np.expand_dims(binary_mask, axis=0)
    
    return binary_mask.astype(np.float32)
