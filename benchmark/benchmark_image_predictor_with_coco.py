#!/usr/bin/env python3
"""
Benchmark SAM2 Image Predictor with and without TTA using COCO validation dataset.
This script:
1. Downloads MS COCO 2017 validation dataset and annotations
2. Runs SAM2 Image Predictor with and without TTA
3. Computes evaluation metrics and saves results
4. Saves visualization images of mask predictions with and without TTA
"""

import os
import sys
import json
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import urllib.request
import zipfile
import argparse
from pathlib import Path
import cv2
from pycocotools import mask as mask_utils
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

# Add parent directory to path for importing sam2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SAM2 components
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_hf

# Constants
DATA_DIR = os.path.expanduser("~/data/coco_benchmark")
COCO_DIR = os.path.join(DATA_DIR, "coco2017")
RESULTS_DIR = os.path.join(DATA_DIR, "results", SAM2_MODEL_ID.split("/")[-1])
VIS_DIR = os.path.join(RESULTS_DIR, "visualizations")  # Directory for mask visualizations
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Default CLI argument values
MAX_IMAGES = 50  # Limit for quick testing, set to None for full dataset

# Model IDs (for Huggingface)
SAM2_MODEL_ID = "facebook/sam2-hiera-tiny"
# SAM2_MODEL_ID = "facebook/sam2-hiera-base-plus"


def setup_directories():
    """Create necessary directories for data and results."""
    os.makedirs(COCO_DIR, exist_ok=True)
    os.makedirs(os.path.join(COCO_DIR, "annotations"), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)


def download_file(url, destination):
    """Download a file if it doesn't exist."""
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return

    print(f"Downloading {url} to {destination}")
    urllib.request.urlretrieve(url, destination)
    print(f"Downloaded {destination}")


def download_datasets():
    """Download COCO val2017 images and annotations if needed."""
    # Download COCO val2017 images
    coco_val_images_zip = os.path.join(DATA_DIR, "val2017.zip")
    if not os.path.exists(os.path.join(COCO_DIR, "val2017")):
        download_file(COCO_VAL_IMAGES_URL, coco_val_images_zip)
        
        # Extract COCO val2017 images
        print(f"Extracting {coco_val_images_zip} to {COCO_DIR}")
        with zipfile.ZipFile(coco_val_images_zip, 'r') as zip_ref:
            zip_ref.extractall(COCO_DIR)
        print(f"Extracted {coco_val_images_zip}")
        
        # Optionally remove the zip file to save space
        os.remove(coco_val_images_zip)
        print(f"Removed {coco_val_images_zip}")
    
    # Download COCO annotations
    coco_annotations_zip = os.path.join(DATA_DIR, "annotations_trainval2017.zip")
    if not os.path.exists(os.path.join(COCO_DIR, "annotations", "instances_val2017.json")):
        download_file(COCO_VAL_ANNOTATIONS_URL, coco_annotations_zip)
        
        # Extract COCO annotations
        print(f"Extracting {coco_annotations_zip} to {COCO_DIR}")
        with zipfile.ZipFile(coco_annotations_zip, 'r') as zip_ref:
            zip_ref.extractall(COCO_DIR)
        print(f"Extracted {coco_annotations_zip}")
        
        # Optionally remove the zip file to save space
        os.remove(coco_annotations_zip)
        print(f"Removed {coco_annotations_zip}")


def load_coco_dataset():
    """Load COCO validation set annotations."""
    coco_annotations_path = os.path.join(COCO_DIR, "annotations", "instances_val2017.json")
    with open(coco_annotations_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data


def initialize_predictors(device="cuda" if torch.cuda.is_available() else "cpu"):
    """Initialize SAM2 Image Predictors with and without TTA."""
    print(f"Using device: {device}")
    
    # Build the model
    sam2_model = build_sam2_hf(
        model_id=SAM2_MODEL_ID,
        device=device
    )
    
    # Create two predictors - one without TTA, one with TTA
    predictor = SAM2ImagePredictor(sam2_model)
    predictor_tta = SAM2ImagePredictor(sam2_model)
    
    return predictor, predictor_tta


def calculate_iou(mask1, mask2):
    """Calculate IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union


def calculate_boundary_f1(mask1, mask2, tolerance=2):
    """Calculate boundary F1 score with tolerance."""
    
    # Get boundaries (simple approach: difference between dilated and eroded)
    boundary1 = binary_dilation(mask1) & ~binary_erosion(mask1)
    boundary2 = binary_dilation(mask2) & ~binary_erosion(mask2)
    
    # Calculate distance transforms
    dist1 = distance_transform_edt(~boundary1)
    dist2 = distance_transform_edt(~boundary2)
    
    # Calculate precision and recall
    boundary_precision = (dist2[boundary1] <= tolerance).sum() / (boundary1.sum() + 1e-10)
    boundary_recall = (dist1[boundary2] <= tolerance).sum() / (boundary2.sum() + 1e-10)
    
    if boundary_precision + boundary_recall > 0:
        boundary_f1 = 2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
    else:
        boundary_f1 = 0
    
    return boundary_f1


def create_side_by_side_visualization(image, gt_mask, mask_baseline, mask_tta):
    """Create a side-by-side visualization of ground truth and predicted masks."""
    # Convert masks to RGB visualizations
    h, w = image.shape[:2]
    
    # Create colored overlays for masks
    gt_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    gt_overlay[gt_mask > 0] = [0, 255, 0]  # Green for ground truth
    
    baseline_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    baseline_overlay[mask_baseline > 0] = [0, 0, 255]  # Blue for baseline
    
    tta_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    tta_overlay[mask_tta > 0] = [255, 0, 0]  # Red for TTA
    
    # Blend the overlays with the original image
    alpha = 0.5
    gt_blend = cv2.addWeighted(image.copy(), 1, gt_overlay, alpha, 0)
    baseline_blend = cv2.addWeighted(image.copy(), 1, baseline_overlay, alpha, 0)
    tta_blend = cv2.addWeighted(image.copy(), 1, tta_overlay, alpha, 0)
    
    # Draw contours for better visualization
    gt_contours, _ = cv2.findContours((gt_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    baseline_contours, _ = cv2.findContours((mask_baseline > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tta_contours, _ = cv2.findContours((mask_tta > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(gt_blend, gt_contours, -1, (0, 255, 0), 2)
    cv2.drawContours(baseline_blend, baseline_contours, -1, (0, 0, 255), 2)
    cv2.drawContours(tta_blend, tta_contours, -1, (255, 0, 0), 2)
    
    # Create the side-by-side comparison
    # Format: [Original | GT | Baseline | TTA]
    combined_width = image.shape[1] * 4
    combined_height = image.shape[0]
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    # Original image
    combined_image[:, :w] = image
    
    # GT mask
    combined_image[:, w:2*w] = gt_blend
    
    # Baseline mask
    combined_image[:, 2*w:3*w] = baseline_blend
    
    # TTA mask
    combined_image[:, 3*w:] = tta_blend
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined_image, "Original", (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined_image, "Ground Truth", (w + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined_image, "Baseline", (2*w + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(combined_image, "TTA", (3*w + 10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    return combined_image


def process_image(image_info, coco_dir, predictor, predictor_tta, coco_data):
    """Process a single image with both predictors and return metrics."""
    # Get image path - COCO uses 12-digit zero-padded image IDs
    file_name = f"{image_info['id']:012d}.jpg"
    image_path = os.path.join(coco_dir, "val2017", file_name)
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return None
    
    # Load image
    image = np.array(Image.open(image_path).convert("RGB"))
    if image is None:
        print(f"Warning: Failed to load image: {image_path}")
        return None
    
    # Get ground truth mask from annotations
    # For simplicity, we'll use the first annotation for this image
    annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_info["id"]]
    if not annotations:
        return None
    
    # For simplicity, pick the first annotation
    annotation = annotations[0]
    
    # Decode segmentation mask
    if isinstance(annotation["segmentation"], dict):  # RLE format
        gt_mask = mask_utils.decode(annotation["segmentation"])
    else:  # Polygon format
        h, w = image_info["height"], image_info["width"]
        rles = mask_utils.frPyObjects(annotation["segmentation"], h, w)
        gt_mask = mask_utils.decode(mask_utils.merge(rles))
    
    # Extract bounding box from annotation
    if "bbox" in annotation:
        # LVIS uses XYWH format, convert to XYXY for SAM2
        x, y, width, height = annotation["bbox"]
        box = np.array([x, y, x + width, y + height])
    else:
        # If no bbox, create one from the mask
        bbox = mask_utils.toBbox(annotation["segmentation"])
        x, y, width, height = bbox
        box = np.array([x, y, x + width, y + height])
    
    # Run prediction with and without TTA
    start_time = time.time()
    predictor.set_image(image)
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=False
    )
    baseline_time = time.time() - start_time
    
    start_time = time.time()
    predictor_tta.set_image(image)
    masks_tta = predictor_tta.predict_with_tta(
        image,
        point_coords=None,
        point_labels=None,
        box=box,
        multimask_output=False
    )
    tta_time = time.time() - start_time
    
    # Convert masks to binary
    mask = masks[0] > 0.5
    mask_tta = masks_tta[0] > 0.5
    
    # Calculate metrics
    iou_baseline = calculate_iou(gt_mask, mask)
    iou_tta = calculate_iou(gt_mask, mask_tta)
    
    boundary_f1_baseline = calculate_boundary_f1(gt_mask, mask)
    boundary_f1_tta = calculate_boundary_f1(gt_mask, mask_tta)
    
    # Create and save visualization
    # Make sure image is in uint8 RGB format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.shape[2] == 4:  # If RGBA, convert to RGB
        image = image[:, :, :3]
        
    # Create side-by-side visualization
    vis_image = create_side_by_side_visualization(image, gt_mask, mask, mask_tta)
    
    # Save visualization
    vis_path = os.path.join(VIS_DIR, f"coco_{image_info['id']:012d}.jpg")
    cv2.imwrite(vis_path, vis_image)
    
    # Return metrics
    return {
        "model_id": SAM2_MODEL_ID,
        "image_id": image_info["id"],
        "file_name": file_name,
        "vis_path": vis_path,
        "metrics": {
            "baseline": {
                "iou": iou_baseline,
                "boundary_f1": boundary_f1_baseline,
                "time": baseline_time
            },
            "tta": {
                "iou": iou_tta,
                "boundary_f1": boundary_f1_tta,
                "time": tta_time
            }
        }
    }


def visualize_results(results, output_path):
    """Visualize the benchmark results."""
    # Extract metrics
    iou_baseline = [r["metrics"]["baseline"]["iou"] for r in results]
    iou_tta = [r["metrics"]["tta"]["iou"] for r in results]
    boundary_f1_baseline = [r["metrics"]["baseline"]["boundary_f1"] for r in results]
    boundary_f1_tta = [r["metrics"]["tta"]["boundary_f1"] for r in results]
    time_baseline = [r["metrics"]["baseline"]["time"] for r in results]
    time_tta = [r["metrics"]["tta"]["time"] for r in results]
    
    # Calculate averages
    avg_iou_baseline = np.mean(iou_baseline)
    avg_iou_tta = np.mean(iou_tta)
    avg_boundary_f1_baseline = np.mean(boundary_f1_baseline)
    avg_boundary_f1_tta = np.mean(boundary_f1_tta)
    avg_time_baseline = np.mean(time_baseline)
    avg_time_tta = np.mean(time_tta)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Set model ID
    model_id = SAM2_MODEL_ID.split("/")[-1]
    
    # IoU plot
    axes[0].bar(["Baseline", "TTA"], [avg_iou_baseline, avg_iou_tta])
    axes[0].set_title(f"Mean IoU - {model_id}")
    axes[0].set_ylim(0, 1)
    
    # Boundary F1 plot
    axes[1].bar(["Baseline", "TTA"], [avg_boundary_f1_baseline, avg_boundary_f1_tta])
    axes[1].set_title(f"Mean Boundary F1 - {model_id}")
    axes[1].set_ylim(0, 1)
    
    # Time plot
    axes[2].bar(["Baseline", "TTA"], [avg_time_baseline, avg_time_tta])
    axes[2].set_title(f"Mean Inference Time (s) - {model_id}")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    # Print summary
    print("\n===== Benchmark Results =====")
    print(f"Model ID: {model_id}")
    print(f"Number of images: {len(results)}")
    print(f"Mean IoU - Baseline: {avg_iou_baseline:.4f}, TTA: {avg_iou_tta:.4f}")
    print(f"Mean Boundary F1 - Baseline: {avg_boundary_f1_baseline:.4f}, TTA: {avg_boundary_f1_tta:.4f}")
    print(f"Mean Inference Time - Baseline: {avg_time_baseline:.4f}s, TTA: {avg_time_tta:.4f}s")
    print(f"IoU Improvement: {(avg_iou_tta - avg_iou_baseline) / avg_iou_baseline * 100:.2f}%")
    print(f"Boundary F1 Improvement: {(avg_boundary_f1_tta - avg_boundary_f1_baseline) / avg_boundary_f1_baseline * 100:.2f}%")
    print(f"Time Increase: {(avg_time_tta - avg_time_baseline) / avg_time_baseline * 100:.2f}%")
    
    # Find images where TTA outperformed baseline
    tta_wins = []
    for r in results:
        # Check if TTA outperformed baseline on IoU or boundary F1
        if (r["metrics"]["tta"]["iou"] > r["metrics"]["baseline"]["iou"] or 
            r["metrics"]["tta"]["boundary_f1"] > r["metrics"]["baseline"]["boundary_f1"]):
            tta_wins.append({
                "image_id": r["image_id"],
                "vis_path": r["vis_path"],
                "iou_diff": r["metrics"]["tta"]["iou"] - r["metrics"]["baseline"]["iou"],
                "boundary_f1_diff": r["metrics"]["tta"]["boundary_f1"] - r["metrics"]["baseline"]["boundary_f1"],
                "metrics": r["metrics"]
            })
    
    # Sort by IoU improvement (descending)
    tta_wins.sort(key=lambda x: x["iou_diff"], reverse=True)
    
    # Print list of images where TTA outperformed baseline
    print("\n===== Images Where TTA Outperformed Baseline =====")
    print(f"Found {len(tta_wins)} images where TTA performed better ({len(tta_wins)/len(results)*100:.1f}% of total)")
    
    if len(tta_wins) > 0:
        top_n = 25
        top_n_wins = tta_wins[:top_n]
        
        print(f"\nTop {len(top_n_wins)} improvements (sorted by IoU difference):")
        for i, win in enumerate(top_n_wins):
            print(f"\n{i+1}. Image ID: {win['image_id']}")
            print(f"   IoU: {win['metrics']['baseline']['iou']:.4f} → {win['metrics']['tta']['iou']:.4f} (+{win['iou_diff']:.4f})")
            print(f"   Boundary F1: {win['metrics']['baseline']['boundary_f1']:.4f} → {win['metrics']['tta']['boundary_f1']:.4f} (+{win['boundary_f1_diff']:.4f})")
            print(f"   Visualization: {win['vis_path']}")
    
    return {
        "model_id": model_id,
        "number_of_images": len(results),
        "iou_baseline": avg_iou_baseline,
        "iou_tta": avg_iou_tta,
        "boundary_f1_baseline": avg_boundary_f1_baseline,
        "boundary_f1_tta": avg_boundary_f1_tta,
        "time_baseline": avg_time_baseline,
        "time_tta": avg_time_tta,
        "iou_improvement": (avg_iou_tta - avg_iou_baseline) / avg_iou_baseline * 100,
        "boundary_f1_improvement": (avg_boundary_f1_tta - avg_boundary_f1_baseline) / avg_boundary_f1_baseline * 100,
        "time_increase": (avg_time_tta - avg_time_baseline) / avg_time_baseline * 100,
        "tta_wins": tta_wins
    }


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Benchmark SAM2 Image Predictor with and without TTA using COCO dataset")
    parser.add_argument("--max_images", type=int, default=MAX_IMAGES, help="Maximum number of images to process")
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Download datasets
    download_datasets()
    
    # Load COCO validation data
    coco_data = load_coco_dataset()
    print(f"Loaded COCO validation set with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    
    # Initialize predictors
    predictor, predictor_tta = initialize_predictors()
    
    # Sample some images for evaluation
    images = coco_data["images"]
    if args.max_images is not None and args.max_images < len(images):
        images = random.sample(images, args.max_images)
        print(f"Using {args.max_images} random images for evaluation")
    
    # Process images
    results = []
    
    for image_info in tqdm(images, desc="Processing images"):
        result = process_image(image_info, COCO_DIR, predictor, predictor_tta, coco_data)
        if result is not None:
            results.append(result)
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, "benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f)
    print(f"Saved results to {results_path}")
    print(f"Saved {len(results)} visualization images to {VIS_DIR}/")
    
    # Visualize results
    plot_path = os.path.join(RESULTS_DIR, "benchmark_plot.png")
    summary = visualize_results(results, plot_path)
    print(f"Saved plot to {plot_path}")
    
    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "benchmark_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()