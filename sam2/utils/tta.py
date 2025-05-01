"""
Test-Time Augmentation manager for SAM2.
"""
from typing import List, Literal, Tuple, Callable, Dict, Optional
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


TTAAugmentationName = Literal[
    "horizontal_flip",
    "vertical_flip",
    "rotate_90",
    "rotate_270",
    "grayscale",
    "increase_brightness",
    "increase_contrast",
    "increase_saturation",
    "increase_hue",
    "decrease_brightness",
    "decrease_contrast",
    "decrease_saturation",
    "decrease_hue",
]


DEFAULT_ENABLED_TTA_AUGMENTATIONS: List[TTAAugmentationName] = [
    "horizontal_flip",
    "vertical_flip",
    "rotate_90",
    "rotate_270",
    "grayscale",
    "increase_brightness",
    "increase_contrast",
    "increase_saturation",
    "increase_hue",
    "decrease_brightness",
    "decrease_contrast",
    "decrease_saturation",
    "decrease_hue",
]


type TTAAugmentImageFn = Callable[[Image], Image]
type TTAAugmentMaskFn = Callable[[np.ndarray], np.ndarray]
type TTAAugmentationPair = Tuple[TTAAugmentImageFn, TTAAugmentMaskFn]


TTA_AUGMENTATION_MAP: Dict[TTAAugmentationName, TTAAugmentationPair] = {
    #
    # Geometric transforms
    #

    "horizontal_flip": (ImageOps.mirror, lambda m: np.flip(m, axis=-1)),
    "vertical_flip": (ImageOps.flip, lambda m: np.flip(m, axis=-2)),
    "rotate_90": (lambda img: img.rotate(90, expand=True), lambda m: np.rot90(m, k=3, axes=(1,2))),
    "rotate_270": (lambda img: img.rotate(270, expand=True), lambda m: np.rot90(m, k=1, axes=(1,2))),
    
    #
    # Photometric/color transforms
    #

    "grayscale": (pil_grayscale, lambda m: m),

    "increase_brightness": (lambda img: pil_adjust_brightness(img, factor=1.2), lambda m: m),
    "increase_contrast": (lambda img: pil_adjust_contrast(img, factor=1.2), lambda m: m),
    "increase_saturation": (lambda img: pil_adjust_saturation(img, factor=1.2), lambda m: m),
    "increase_hue": (lambda img: pil_adjust_hue(img, factor=0.1), lambda m: m),
    
    "decrease_brightness": (lambda img: pil_adjust_brightness(img, factor=0.8), lambda m: m),
    "decrease_contrast": (lambda img: pil_adjust_contrast(img, factor=0.8), lambda m: m),
    "decrease_saturation": (lambda img: pil_adjust_saturation(img, factor=0.8), lambda m: m),
    "decrease_hue": (lambda img: pil_adjust_hue(img, factor=-0.1), lambda m: m),
}


TTAAggregationMethod = Literal["max", "mean"]


class TTAManager:
    def __init__(
        self,
        threshold: float = 0.5,
        enabled_augmentations: Optional[List[TTAAugmentationName]] = None,
        agg_method: Optional[TTAAggregationMethod] = None,
    ):
        self.threshold = threshold

        if enabled_augmentations is None:
            self.enabled_augmentations = DEFAULT_ENABLED_TTA_AUGMENTATIONS
        else:
            self.enabled_augmentations = enabled_augmentations

        if agg_method is None:
            self.agg_method = "max"
        else:
            self.agg_method = agg_method
        
        # PIL-based TTA ops: list of (augment_image_fn, deaugment_mask_fn)
        self.pil_augmentations: List[TTAAugmentationPair] = [
            (lambda img: img, lambda m: m),  # original
        ]

        # Add enabled augmentations
        if self.enabled_augmentations is not None:
            for augmentation in self.enabled_augmentations:
                self.pil_augmentations.append(TTA_AUGMENTATION_MAP[augmentation])

    def aggregate_masks(self, masks: List[np.ndarray], apply_threshold: bool = True) -> np.ndarray:
        if self.agg_method == "max":
            return self.aggregate_masks_max(masks, apply_threshold)
        elif self.agg_method == "mean":
            return self.aggregate_masks_mean(masks, apply_threshold)
        else:
            raise ValueError(f"Unknown aggregation method: {self.agg_method}")

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

    def aggregate_masks_mean(self, masks: List[np.ndarray], apply_threshold: bool = True) -> np.ndarray:
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
