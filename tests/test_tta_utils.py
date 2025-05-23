import torch
import numpy as np
import pytest
from sam2.utils.transforms import (
    horizontal_flip_image, horizontal_flip_mask,
    vertical_flip_image, vertical_flip_mask,
    rotate_image, rotate_mask,
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


def test_tta_manager_application_and_aggregation():
    mgr = TTAManager(threshold=0.5)
    x = torch.randn(1,3,16,16)
    results = mgr.apply_augmentations(x)
    assert len(results) == len(mgr.augmentations)
    masks = [np.full((1,16,16), fill_value=i) for i in range(len(results))]
    
    # Test raw aggregation (without threshold)
    agg_raw = mgr.aggregate_masks(masks, apply_threshold=False)
    expected_raw = np.mean(np.stack(masks, axis=0), axis=0)
    np.testing.assert_allclose(agg_raw, expected_raw)
    
    # Test thresholded aggregation
    agg_thresholded = mgr.aggregate_masks(masks, apply_threshold=True)
    expected_thresholded = expected_raw > mgr.threshold
    np.testing.assert_equal(agg_thresholded, expected_thresholded)
