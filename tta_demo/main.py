"""
Demo for Test-Time Augmentation (TTA): compare original, baseline, and TTA masks overlaid.
"""

import os
import sys
import argparse
import random
import time

import numpy as np
import cv2
import torch
from PIL import Image
from typing import get_args

# allow importing sam2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.utils.tta import TTAAugmentationName, TTAAggregationMethod


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple = (0, 0, 255), alpha: float = 0.5) -> np.ndarray:
    """Overlay a binary mask on the image with given color and alpha."""
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TTA Demo: compare original vs baseline vs TTA masks"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Directory containing input images",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base-plus", "large"],
        help="SAM2 model size"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--tta-enabled-augmentations",
        type=lambda s: s.split(" ") if s else None,
        default=None,
        help=(
            "Space-separated list of TTA augmentations to enable. "
            f"Valid: {', '.join(list(get_args(TTAAugmentationName)))}"
        ),
    )
    parser.add_argument(
        "--tta-agg-method",
        type=str,
        default="max",
        choices=list(get_args(TTAAggregationMethod)),
        help="Aggregation method for TTA masks"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Path to save the resulting image grid"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    model_id = f"facebook/sam2-hiera-{args.model_size}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building SAM2 model {model_id} on {device}...")
    sam_model = build_sam2_hf(model_id=model_id, device=device)
    predictor = SAM2ImagePredictor(sam_model)
    predictor_tta = SAM2ImagePredictor(
        sam_model,
        tta_enabled_augmentations=args.tta_enabled_augmentations,
        tta_agg_method=args.tta_agg_method,
    )

    # collect images
    img_files = sorted(
        f for f in os.listdir(args.images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not img_files:
        print(f"No images found in {args.images_dir}")
        sys.exit(1)

    rows = []
    for fname in img_files:
        path = os.path.join(args.images_dir, fname)
        img = Image.open(path).convert("RGB")
        image = np.array(img)
        h, w = image.shape[:2]
        box = np.array([0, 0, w, h])

        # baseline prediction
        predictor.set_image(image)
        masks_b, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )
        mask_base = masks_b[0]

        # TTA prediction
        predictor_tta.set_image(image)
        masks_t, _, _ = predictor_tta.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )
        mask_tta = masks_t[0]

        orig_col = image
        base_col = overlay_mask(image, mask_base, color=(0, 0, 255), alpha=0.5)
        tta_col = overlay_mask(image, mask_tta, color=(0, 255, 0), alpha=0.5)

        row = np.concatenate([orig_col, base_col, tta_col], axis=1)
        rows.append(row)

    # stack rows vertically into a grid
    grid = np.vstack(rows)
    cv2.imwrite(args.output, grid)
    print(f"Saved demo grid to {args.output}")


if __name__ == "__main__":
    main()
