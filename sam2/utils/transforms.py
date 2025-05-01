# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, Resize, ToTensor


class SAM2Transforms(nn.Module):
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = torch.jit.script(
            nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            )
        )

    def __call__(self, x):
        x = self.to_tensor(x)
        return self.transforms(x)

    def forward_batch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch

    def transform_coords(
        self, coords: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. The coordinates can be in absolute image or normalized coordinates,
        If the coords are in absolute image coordinates, normalize should be set to True and original image size is required.

        Returns
            Un-normalized coordinates in the range of [0, 1] which is expected by the SAM2 model.
        """
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h

        coords = coords * self.resolution  # unnormalize coords
        return coords

    def transform_boxes(
        self, boxes: torch.Tensor, normalize=False, orig_hw=None
    ) -> torch.Tensor:
        """
        Expects a tensor of shape Bx4. The coordinates can be in absolute image or normalized coordinates,
        if the coords are in absolute image coordinates, normalize should be set to True and original image size is required.
        """
        boxes = self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)
        return boxes

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """
        Perform PostProcessing on output masks.
        """
        from sam2.utils.misc import get_connected_components

        masks = masks.float()
        input_masks = masks
        mask_flat = masks.flatten(0, 1).unsqueeze(1)  # flatten as 1-channel image
        try:
            if self.max_hole_area > 0:
                # Holes are those connected components in background with area <= self.fill_hole_area
                # (background regions are those with mask scores <= self.mask_threshold)
                labels, areas = get_connected_components(
                    mask_flat <= self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with a small positive mask score (10.0) to change them to foreground.
                masks = torch.where(is_hole, self.mask_threshold + 10.0, masks)

            if self.max_sprinkle_area > 0:
                labels, areas = get_connected_components(
                    mask_flat > self.mask_threshold
                )
                is_hole = (labels > 0) & (areas <= self.max_sprinkle_area)
                is_hole = is_hole.reshape_as(masks)
                # We fill holes with negative mask score (-10.0) to change them to background.
                masks = torch.where(is_hole, self.mask_threshold - 10.0, masks)
        except Exception as e:
            # Skip the post-processing step if the CUDA kernel fails
            warnings.warn(
                f"{e}\n\nSkipping the post-processing step due to the error above. You can "
                "still use SAM 2 and it's OK to ignore the error above, although some post-processing "
                "functionality may be limited (which doesn't affect the results in most cases; see "
                "https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).",
                category=UserWarning,
                stacklevel=2,
            )
            masks = input_masks

        masks = F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
        return masks


# TTA augmentation ops and inverses
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

def horizontal_flip_image(image: torch.Tensor) -> torch.Tensor:
    """Flip image horizontally."""
    return torch.flip(image, dims=[-1])


def horizontal_flip_mask(mask: torch.Tensor) -> torch.Tensor:
    """Flip mask horizontally."""
    return torch.flip(mask, dims=[-1])


def vertical_flip_image(image: torch.Tensor) -> torch.Tensor:
    """Flip image vertically."""
    return torch.flip(image, dims=[-2])


def vertical_flip_mask(mask: torch.Tensor) -> torch.Tensor:
    """Flip mask vertically."""
    return torch.flip(mask, dims=[-2])


def rotate_image(image: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate image by angle (degrees) with bilinear interpolation."""
    return TF.rotate(image, angle=angle, interpolation=InterpolationMode.BILINEAR)


def rotate_mask(mask: torch.Tensor, angle: float) -> torch.Tensor:
    """Inverse rotate mask by angle with nearest neighbor to avoid artifacts."""
    return TF.rotate(mask, angle=-angle, interpolation=InterpolationMode.NEAREST)


# PIL-based color ops

from PIL import Image, ImageEnhance
import colorsys
import numpy as np

def pil_adjust_brightness(image: Image, factor: float) -> Image:
    """Adjust brightness of an image.
    
    Args:
        image: PIL Image to adjust
        factor: Brightness adjustment factor. 0 gives black image, 
               1 gives original image, values > 1 increase brightness
    
    Returns:
        Brightness adjusted PIL Image
    """
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def pil_adjust_contrast(image: Image, factor: float) -> Image:
    """Adjust contrast of an image.
    
    Args:
        image: PIL Image to adjust
        factor: Contrast adjustment factor. 0 gives solid gray image, 
               1 gives original image, values > 1 increase contrast
    
    Returns:
        Contrast adjusted PIL Image
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def pil_adjust_saturation(image: Image, factor: float) -> Image:
    """Adjust saturation of an image.
    
    Args:
        image: PIL Image to adjust
        factor: Saturation adjustment factor. 0 gives grayscale image, 
               1 gives original image, values > 1 increase saturation
    
    Returns:
        Saturation adjusted PIL Image
    """
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(factor)


def pil_adjust_hue(image: Image, factor: float) -> Image:
    """Adjust hue of an image.
    
    Args:
        image: PIL Image to adjust
        factor: Hue adjustment factor in range [-0.5, 0.5].
               0 gives original image, negative values decrease hue,
               positive values increase hue.
    
    Returns:
        Hue adjusted PIL Image
    """
    # Convert to HSV for hue adjustment
    # Convert PIL image to numpy array
    arr = np.array(image.convert('RGB'))
    
    # Convert RGB to HSV
    h, s, v = [], [], []
    for r, g, b in arr.reshape(-1, 3):
        hsv = colorsys.rgb_to_hsv(r/255., g/255., b/255.)
        h.append((hsv[0] + factor) % 1.0)  # Adjust hue and wrap around to [0,1]
        s.append(hsv[1])
        v.append(hsv[2])
    
    # Convert back to RGB
    rgb = []
    for i in range(len(h)):
        r, g, b = colorsys.hsv_to_rgb(h[i], s[i], v[i])
        rgb.append((int(r*255), int(g*255), int(b*255)))
    
    # Reshape and convert back to PIL image
    rgb_arr = np.array(rgb, dtype=np.uint8).reshape(arr.shape)
    return Image.fromarray(rgb_arr)


def pil_grayscale(image: Image) -> Image:
    """
    Convert image to grayscale.
    
    Args:
        image: PIL Image to convert to grayscale
        
    Returns:
        Grayscale PIL Image (still in RGB mode for compatibility)
    """
    # Convert to grayscale and back to RGB mode to keep 3 channels
    grayscale = image.convert('L')
    return grayscale.convert('RGB')
