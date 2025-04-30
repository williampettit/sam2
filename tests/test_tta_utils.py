import torch
import numpy as np
import pytest
from sam2.utils.transforms import (
    horizontal_flip_image, horizontal_flip_mask,
    vertical_flip_image, vertical_flip_mask,
    rotate_image, rotate_mask,
    adjust_brightness_contrast
)
from sam2.utils.tta import TTAManager


def test_horizontal_flip_inversion():
    x = torch.arange(12).view(1,1,3,4)
    flipped = horizontal_flip_image(x)
    inv = horizontal_flip_image(flipped)
    assert torch.equal(inv, x)
    mask_flipped = horizontal_flip_mask(x)
    assert torch.equal(mask_flipped, flipped)


def test_vertical_flip_inversion():
    x = torch.arange(12).view(1,1,3,4)
    flipped = vertical_flip_image(x)
    inv = vertical_flip_image(flipped)
    assert torch.equal(inv, x)
    mask_flipped = vertical_flip_mask(x)
    assert torch.equal(mask_flipped, flipped)


def test_rotate_mask_inversion_identity():
    x = torch.randint(0, 2, (1,1,5,5)) * 1.0  # binary mask
    rotated = rotate_image(x, 90)
    inv = rotate_mask(rotated, 90)
    # For binary mask with nearest neighbor, invert should match original
    assert torch.equal(inv, x)


def test_adjust_brightness_contrast_change():
    x = torch.ones(1,3,10,10)
    out = adjust_brightness_contrast(x, brightness=1.5, contrast=0.5)
    assert not torch.equal(out, x)


def test_tta_manager_application_and_aggregation():
    mgr = TTAManager(threshold=0.5)
    x = torch.randn(1,3,16,16)
    results = mgr.apply_augmentations(x)
    assert len(results) == len(mgr.augmentations)
    masks = [np.full((1,16,16), fill_value=i) for i in range(len(results))]
    agg = mgr.aggregate_masks(masks)
    expected = np.mean(np.stack(masks, axis=0), axis=0)
    np.testing.assert_allclose(agg, expected)
