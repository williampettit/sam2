"""
Test-Time Augmentation manager for SAM2.
"""
from typing import List, Tuple, Callable
import numpy as np
import torch
from PIL import ImageOps, ImageEnhance
from sam2.utils.transforms import (
    horizontal_flip_image,
    horizontal_flip_mask,
    vertical_flip_image,
    vertical_flip_mask,
    rotate_image,
    rotate_mask,
    pil_adjust_brightness,
    pil_adjust_contrast,
    pil_adjust_saturation,
    pil_adjust_hue,
    pil_grayscale,
)

class TTAManager:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        # list of (image_fn, mask_fn)
        self.augmentations: List[Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]] = [
            (lambda x: x, lambda m: m),  # original
            # (horizontal_flip_image, horizontal_flip_mask),  # horizontal flip
            # (vertical_flip_image, vertical_flip_mask),  # vertical flip
            # (lambda x: rotate_image(x, 90), lambda m: rotate_mask(m, 90)),  # rotate 90
            # (lambda x: rotate_image(x, -90), lambda m: rotate_mask(m, -90)),  # rotate 270/rotate -90
        ]
        # PIL-based TTA ops for image predictor
        self.pil_augmentations: List[Tuple[Callable, Callable]] = [
            (lambda img: img, lambda m: m),  # original
            # (ImageOps.mirror, lambda m: np.flip(m, axis=-1)),  # horizontal flip
            # (ImageOps.flip, lambda m: np.flip(m, axis=-2)),  # vertical flip
            # (lambda img: img.rotate(90, expand=True), lambda m: np.rot90(m, k=3, axes=(1,2))),  # rotate 90
            # (lambda img: img.rotate(270, expand=True), lambda m: np.rot90(m, k=1, axes=(1,2))),  # rotate 270
            (lambda img: pil_grayscale(img), lambda m: m),  # grayscale
            
            (lambda img: pil_adjust_brightness(img, factor=1.2), lambda m: m),  # brightness
            (lambda img: pil_adjust_contrast(img, factor=1.2), lambda m: m),  # contrast
            # (lambda img: pil_adjust_saturation(img, factor=1.2), lambda m: m),  # saturation
            
            (lambda img: pil_adjust_brightness(img, factor=0.8), lambda m: m),  # brightness
            (lambda img: pil_adjust_contrast(img, factor=0.8), lambda m: m),  # contrast
            # (lambda img: pil_adjust_saturation(img, factor=0.8), lambda m: m),  # saturation

            # (lambda img: pil_adjust_hue(img, factor=0.25), lambda m: m),  # hue
        ]

    def apply_augmentations(self, image: torch.Tensor) -> List[Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]]:
        """
        Apply each augmentation to the image.
        Returns list of (augmented image, mask inverse function).
        """
        results = []
        for img_fn, mask_fn in self.augmentations:
            aug_img = img_fn(image)
            results.append((aug_img, mask_fn))
        return results

    def aggregate_masks_max(self, masks: List[np.ndarray], apply_threshold: bool = True) -> np.ndarray:
        """
        Aggregate a list of mask arrays via pixel-wise max and optionally apply threshold.
        
        Args:
            masks: List of mask arrays in float32 format (logits).
            apply_threshold: Whether to apply threshold to create binary mask (default: True).
                             For testing purposes, this can be set to False to return raw max values.
            
        Returns:
            Float array mask:
            - If apply_threshold is True: Values are 1.0 where mask is above threshold, 0.0 elsewhere
            - If apply_threshold is False: Raw max mask values (float array).
        """
        # Stack masks along a new axis and calculate max value per pixel
        # This should create more expansive masks compared to mean aggregation
        max_mask = np.max(np.stack(masks, axis=0), axis=0)
        
        # Apply threshold to get binary mask if requested
        if apply_threshold:
            # Convert boolean mask to float (1.0 and 0.0) to avoid issues with tensor operations
            return (max_mask > self.threshold).astype(np.float32)
        else:
            return max_mask

    def aggregate_masks(self, masks: List[np.ndarray], apply_threshold: bool = True) -> np.ndarray:
        """
        Aggregate a list of mask arrays via pixel-wise mean and optionally apply threshold.
        
        Args:
            masks: List of mask arrays in float32 format (logits).
            apply_threshold: Whether to apply threshold to create binary mask (default: True).
                             For testing purposes, this can be set to False to return raw mean values.
            
        Returns:
            Float array mask:
            - If apply_threshold is True: Values are 1.0 where mask is above threshold, 0.0 elsewhere
            - If apply_threshold is False: Raw mean mask values (float array).
        """
        # Stack masks along a new axis and calculate mean
        mean_mask = np.mean(np.stack(masks, axis=0), axis=0)
        
        # Apply threshold to get binary mask if requested
        if apply_threshold:
            # Convert boolean mask to float (1.0 and 0.0) to avoid issues with tensor operations
            return (mean_mask > self.threshold).astype(np.float32)
        else:
            return mean_mask
