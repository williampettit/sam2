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


# images to use for demo, along with the xy point our object is at
DEMO_IMAGES = [
  ("./tta_demo/images/IMG_1511.jpg", (1971, 2110)),
  ("./tta_demo/images/IMG_1513.jpg", (1641, 1430)),
  ("./tta_demo/images/IMG_1518.jpg", (1752, 1500)),
  ("./tta_demo/images/IMG_1519.jpg", (1252, 3180)),
  ("./tta_demo/images/IMG_1521.jpg", (1402, 2410)),
  ("./tta_demo/images/IMG_1522.jpg", (2122, 3350)),
  ("./tta_demo/images/IMG_1523.jpg", (1992, 1960)),
  ("./tta_demo/images/IMG_3013.jpg", (2052, 2921)),
]


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

    # --- Find max dimensions from DEMO_IMAGES --- START
    max_h, max_w = 0, 0
    valid_demo_images = []
    print(f"Scanning {len(DEMO_IMAGES)} demo images for dimensions...")
    for img_path, point_xy in DEMO_IMAGES:
        if not os.path.exists(img_path):
            print(f"Warning: Demo image not found, skipping: {img_path}")
            continue
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                max_h = max(max_h, h)
                max_w = max(max_w, w)
                valid_demo_images.append((img_path, point_xy))
        except Exception as e:
            print(f"Warning: Could not read demo image {img_path}: {e}")

    if not valid_demo_images:
        print("Error: No valid demo images found or readable.")
        sys.exit(1)

    print(f"Max dimensions found: H={max_h}, W={max_w}")
    # --- Find max dimensions --- END

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

    rows = []
    num_images = len(valid_demo_images)
    for i, (path, point_xy) in enumerate(valid_demo_images):
        print(f"Processing image {i+1}/{num_images}: {os.path.basename(path)}...")
        try:
            img = Image.open(path).convert("RGB")
            image = np.array(img)
            h, w = image.shape[:2]
            point_coords_np = np.array([point_xy])
            point_labels_np = np.array([1])

            # baseline prediction
            predictor.set_image(image)
            masks_b, _, _ = predictor.predict(
                point_coords=point_coords_np,
                point_labels=point_labels_np,
                multimask_output=False,
            )
            mask_base = masks_b[0]

            # TTA prediction
            predictor_tta.set_image(image)
            masks_t, _, _ = predictor_tta.predict(
                point_coords=point_coords_np,
                point_labels=point_labels_np,
                multimask_output=False,
            )
            mask_tta = masks_t[0]

            orig_col = image
            base_col = overlay_mask(image, mask_base, color=(0, 0, 255), alpha=0.5)
            tta_col = overlay_mask(image, mask_tta, color=(0, 255, 0), alpha=0.5)

            # --- Pad columns to max dimensions --- START
            pad_h = max_h - h
            pad_w = max_w - w
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            border_args = {
                "borderType": cv2.BORDER_CONSTANT,
                "value": [255, 255, 255] # White padding
            }

            orig_col_padded = cv2.copyMakeBorder(orig_col, pad_top, pad_bottom, pad_left, pad_right, **border_args)
            base_col_padded = cv2.copyMakeBorder(base_col, pad_top, pad_bottom, pad_left, pad_right, **border_args)
            tta_col_padded = cv2.copyMakeBorder(tta_col, pad_top, pad_bottom, pad_left, pad_right, **border_args)
            # --- Pad columns to max dimensions --- END

            row = np.concatenate([orig_col_padded, base_col_padded, tta_col_padded], axis=1)
            rows.append(row)

        except Exception as e:
            print(f"\nError processing image {path}: {e}")
            print("Skipping this image.\n")

    # stack rows vertically into a grid
    if not rows:
        print("No images were successfully processed.")
        sys.exit(1)

    grid = np.vstack(rows)
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cv2.imwrite(args.output, grid)
    print(f"Saved demo grid to {args.output}")


if __name__ == "__main__":
    main()
