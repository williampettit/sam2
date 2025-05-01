"""
Quick demo script for testing SAM2 with Test-Time Augmentation (TTA).
This script can process both images and videos, and saves overlay output for visual comparison.
"""

import os
import argparse
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.tta import TTAManager  # Explicit import for clarity


def overlay_masks_on_image(image, masks, obj_ids=None, alpha=0.5):
    """Overlay segmentation masks on an image.
    
    Args:
        image: RGB image as numpy array of shape (H, W, 3)
        masks: Dictionary mapping obj_id -> binary mask of shape (H, W)
        obj_ids: List of object IDs to overlay. If None, use all keys in masks.
        alpha: Transparency of the overlay
    
    Returns:
        Image with masks overlaid
    """
    if obj_ids is None:
        obj_ids = list(masks.keys())
    
    # Make a copy of the image
    overlay = image.copy()
    
    # Define colors for different objects (using tab10 colormap)
    cmap = plt.get_cmap("tab10")
    
    # Overlay each mask
    for obj_id in obj_ids:
        if obj_id not in masks:
            continue
        
        mask = masks[obj_id]
        if mask.sum() == 0:  # Skip empty masks
            continue
        
        # Get color for this object
        color_idx = (obj_id - 1) % 10  # Map to 0-9 range for tab10
        color = np.array(cmap(color_idx)[:3]) * 255
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color
        
        # Overlay the mask
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
        
        # Draw contour around the mask for better visibility
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)
        
        # Add object ID label
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.putText(overlay, f"ID: {obj_id}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Calculate and display mask coverage percentage
        frame_area = mask.shape[0] * mask.shape[1]
        positive_pixels = np.sum(mask > 0)
        mask_coverage = positive_pixels / frame_area
        cv2.putText(overlay, f"Coverage: {mask_coverage:.1%}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return overlay


def process_image(image_path, output_dir, use_tta=False, box=None):
    """Process a single image with SAM2 and save the results."""
    print(f"Processing image: {image_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    image = np.array(Image.open(image_path))
    
    # Initialize the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2_hf(
        model_id="facebook/sam2-hiera-base-plus",
        device=device
    )
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Run prediction with and without TTA for comparison
    # With TTA
    if use_tta:
        predictor.set_image(image)
        with torch.amp.autocast('cuda', enabled=True):
            if box is not None:
                box_coords = np.array(box)
                tta_masks, *_ = predictor.predict_with_tta(
                    image, box=box_coords, normalize_coords=False
                )
            else:
                # Use center point if no box provided
                h, w = image.shape[:2]
                point_coords = np.array([[w//2, h//2]])
                point_labels = np.array([1])
                tta_masks, *_ = predictor.predict_with_tta(
                    image, point_coords=point_coords, point_labels=point_labels
                )
        tta_overlay = overlay_masks_on_image(image, tta_masks)
        tta_output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_tta.png")
        cv2.imwrite(tta_output_path, cv2.cvtColor(tta_overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved TTA result to: {tta_output_path}")
    
    # Without TTA (baseline)
    predictor.set_image(image)
    with torch.amp.autocast('cuda', enabled=True):
        if box is not None:
            box_coords = np.array(box)
            baseline_masks, *_ = predictor.predict(
                box=box_coords, normalize_coords=False
            )
        else:
            # Use center point if no box provided
            h, w = image.shape[:2]
            point_coords = np.array([[w//2, h//2]])
            point_labels = np.array([1])
            baseline_masks, *_ = predictor.predict(
                point_coords=point_coords, point_labels=point_labels
            )
    baseline_overlay = overlay_masks_on_image(image, baseline_masks)
    baseline_output_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_baseline.png")
    cv2.imwrite(baseline_output_path, cv2.cvtColor(baseline_overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved baseline result to: {baseline_output_path}")
    
    # Create side-by-side comparison if both are available
    if use_tta:
        side_by_side = np.hstack([baseline_overlay, tta_overlay])
        comparison_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_comparison.png")
        cv2.imwrite(comparison_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
        print(f"Saved side-by-side comparison to: {comparison_path}")


def process_video(video_path, output_dir, use_tta=False, box=None):
    """Process a video with SAM2 and save the results.
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save the output
        use_tta: Whether to use test-time augmentation
        box: Optional bounding box in format [x1, y1, x2, y2]
    """
    print(f"Processing video: {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize with mask_threshold for TTA consistency
    predictor = SAM2VideoPredictor.from_pretrained(
        "facebook/sam2-hiera-base-plus", 
        mask_threshold=0.5,  # Same threshold used by TTAManager by default
        non_overlap_masks=True  # Enable non-overlapping constraints for better results
    )
    
    # Generate output paths
    video_name = os.path.basename(video_path).split('.')[0]
    baseline_output_path = os.path.join(output_dir, f"{video_name}_baseline.mp4")
    tta_output_path = os.path.join(output_dir, f"{video_name}_tta.mp4")
    
    # Process baseline (no TTA)
    print("\n=== Processing baseline (no TTA) ===")
    # Set a lower threshold for baseline to ensure we get a mask
    predictor.mask_threshold = 0.3  # Lower threshold for baseline
    process_result = process_single_video(predictor, video_path, baseline_output_path, use_tta=False, box=box)
    print(f"Saved baseline video to: {baseline_output_path}")
    
    # Process with TTA if requested
    if use_tta:
        print("\n=== Processing with TTA ===")
        # Reset threshold for TTA
        predictor.mask_threshold = 0.3  # Same lower threshold for TTA
        process_result = process_single_video(predictor, video_path, tta_output_path, use_tta=True, box=box)
        print(f"Saved TTA video to: {tta_output_path}")
        
        # Create side-by-side comparison
        create_side_by_side_video(
            baseline_output_path,
            tta_output_path,
            os.path.join(output_dir, f"{video_name}_comparison.mp4")
        )
        print(f"Saved side-by-side comparison to: {os.path.join(output_dir, f'{video_name}_comparison.mp4')}")


def process_single_video(predictor, video_path, output_path, use_tta=False, box=None):
    """Process a single video with or without TTA and save the results.
    
    Args:
        predictor: SAM2VideoPredictor instance
        video_path: Path to input video
        output_path: Path to save output video
        use_tta: Whether to use test-time augmentation
        box: Optional bounding box in format [x1, y1, x2, y2]
    """
    # Initialize inference state
    inference_state = predictor.init_state(
        video_path=video_path,
        offload_video_to_cpu=True
    )
    
    # Get the first frame for visualization and point selection
    first_frame = inference_state["images"][0].cpu().numpy().transpose(1, 2, 0)
    h, w = first_frame.shape[:2]
    
    # Create debug frame for visualization
    debug_frame = (first_frame * 255).astype(np.uint8).copy()
    
    # Determine whether to use box or point
    if box is not None:
        # Use the provided bounding box
        x1, y1, x2, y2 = box
        
        # Draw the box on the debug frame
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(debug_frame, "Bounding Box", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        print(f"Using bounding box: {box}")
    else:
        # Fall back to a single point at the center of the frame
        point = [[w//2, h//2]]
        label = [1]  # foreground
        
        # Draw the point on the debug frame
        cv2.circle(debug_frame, (point[0][0], point[0][1]), 5, (0, 255, 0), -1)
        cv2.putText(debug_frame, "Center Point", (point[0][0] + 10, point[0][1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        print(f"Using center point: {point}")
    
    # Save the debug frame
    debug_dir = os.path.join(os.path.dirname(output_path), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, f"{os.path.basename(output_path).split('.')[0]}_prompt.jpg")
    cv2.imwrite(debug_path, cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR))
    print(f"Saved debug frame with prompt to: {debug_path}")
    
    # Add prompt to first frame
    if box is not None:
        # Use bounding box
        print(f"Adding bounding box {box} to frame 0")
        frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
            inference_state,
            frame_idx=0,
            obj_id=1,
            box=box
        )
    else:
        # Use center point
        point = [[w//2, h//2]]
        label = [1]  # foreground
        print(f"Adding point {point} with label {label} to frame 0")
        frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
            inference_state,
            frame_idx=0,
            obj_id=1,
            points=point,
            labels=label,
        )
    
    # Print information about the mask logits
    if isinstance(mask_logits, torch.Tensor):
        print(f"Mask logits shape: {mask_logits.shape}, type: {type(mask_logits)}")
        print(f"Mask logits min: {mask_logits.min().item():.2f}, max: {mask_logits.max().item():.2f}")
    else:
        print(f"Mask logits type: {type(mask_logits)}")
    
    # Store segmentation results for all frames
    video_segments = {}
    
    # Propagate through the video
    if use_tta:
        print("Propagating masks through video with TTA...")
        generator = predictor.propagate_in_video_with_tta(inference_state)
    else:
        print("Propagating masks through video...")
        generator = predictor.propagate_in_video(inference_state)
    
    # Process all frames and store results
    num_processed = 0
    for frame_idx, obj_ids, mask_logits in generator:
        # Store binary masks for each object
        video_segments[frame_idx] = {}
        for i, obj_id in enumerate(obj_ids):
            # Extract raw mask logits for better debugging
            raw_mask = mask_logits[i].cpu().numpy()
            
            # Print some statistics about the raw mask for debugging
            if frame_idx % 50 == 0:
                print(f"Frame {frame_idx}, Object {obj_id}: Raw mask shape={raw_mask.shape}, min={raw_mask.min():.2f}, max={raw_mask.max():.2f}")
            
            # Apply threshold to get binary mask
            # For logits, we need to check if they're already in probability space
            if raw_mask.max() > 1.0 or raw_mask.min() < 0.0:
                # These are logits, apply sigmoid first
                mask_probs = 1 / (1 + np.exp(-raw_mask))
                mask = (mask_probs > predictor.mask_threshold).astype(np.uint8)
            else:
                # These are already probabilities
                mask_probs = raw_mask
                mask = (mask_probs > predictor.mask_threshold).astype(np.uint8)
            
            # Remove any extra dimensions (e.g., from batch or channel dimensions)
            if len(mask.shape) == 3 and mask.shape[0] == 1:  # If shape is (1, H, W)
                mask = mask[0]  # Convert to (H, W)
            elif len(mask.shape) > 2:  # Handle any other unexpected dimensions
                mask = mask.squeeze()  # Remove all singleton dimensions
                if len(mask.shape) > 2:  # If still more than 2D, take the first slice
                    print(f"Warning: Unexpected mask shape {mask.shape}, taking first slice")
                    mask = mask[0]
            
            # Calculate mask coverage (percentage of frame covered)
            frame_area = mask.shape[0] * mask.shape[1]
            positive_pixels = np.sum(mask > 0)
            mask_coverage = positive_pixels / frame_area
            
            if frame_idx % 50 == 0:
                print(f"Frame {frame_idx}, Object {obj_id}: Mask coverage: {mask_coverage:.2%} of frame")
                
            # Add a sanity check for masks that cover too much of the frame
            # If more than 70% of the frame is covered, this is likely a background mask
            if mask_coverage > 0.7 and frame_idx % 50 == 0:
                print(f"WARNING: Frame {frame_idx}, Object {obj_id}: Mask covers {mask_coverage:.2%} of frame - likely a background mask!")
            
            # Check if mask contains any positive pixels
            positive_pixels = np.sum(mask > 0)
            if frame_idx % 50 == 0 or positive_pixels == 0:
                print(f"Frame {frame_idx}, Object {obj_id}: Mask contains {positive_pixels} positive pixels")
                
            # Store the properly dimensioned mask
            video_segments[frame_idx][obj_id] = mask
        
        num_processed += 1
        if num_processed % 10 == 0:
            print(f"  Processed frame {frame_idx} with {len(obj_ids)} objects")
    
    print(f"Processed {num_processed} frames")
    
    # Save video with masks
    save_video_with_masks(video_path, video_segments, output_path)
    
    return video_segments


def save_video_with_masks(video_path, video_segments, output_path):
    """Save a video with masks overlaid.
    
    Args:
        video_path: Path to the input video
        video_segments: Dictionary mapping frame_idx -> {obj_id -> binary_mask}
        output_path: Path to save the output video with masks
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Define colors for different objects (using tab10 colormap)
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Dark red
        (0, 128, 0),    # Dark green
        (0, 0, 128),    # Dark blue
        (128, 128, 0),  # Olive
    ]
    
    print(f"Processing video with {total_frames} frames, {len(video_segments)} frames have masks")
    
    # Process each frame
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if we have masks for this frame
        if frame_idx in video_segments:
            # Get masks for all objects in this frame
            frame_masks = video_segments[frame_idx]
            
            # Create a mask overlay for each object
            for obj_id, binary_mask in frame_masks.items():
                # Get color for this object
                color_idx = (obj_id % len(colors))
                color = colors[color_idx]
                
                # Ensure mask has correct dimensions
                # At this point, binary_mask should already be 2D with shape (H, W)
                if binary_mask.shape != (frame_height, frame_width):
                    # Only need to resize if dimensions don't match
                    binary_mask = cv2.resize(
                        binary_mask.astype(np.uint8), 
                        (frame_width, frame_height), 
                        interpolation=cv2.INTER_NEAREST
                    )
                
                # Check if mask contains any positive pixels
                positive_pixels = np.sum(binary_mask > 0)
                
                if positive_pixels > 0:
                    # Create colored mask overlay
                    mask_overlay = np.zeros_like(frame)
                    mask_overlay[binary_mask > 0] = color
                    
                    # Overlay mask on frame
                    alpha = 0.5
                    frame = cv2.addWeighted(frame, 1.0, mask_overlay, alpha, 0)
                    
                    # Draw a border around the mask for better visibility
                    # Find contours of the mask
                    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), 
                                                  cv2.RETR_EXTERNAL, 
                                                  cv2.CHAIN_APPROX_SIMPLE)
                    # Draw contours on the frame
                    cv2.drawContours(frame, contours, -1, color, 2)
                else:
                    # If no positive pixels, print a warning
                    if frame_idx % 20 == 0:
                        print(f"Frame {frame_idx}, Object {obj_id}: No positive pixels in mask")
                
                # Add a small label with object ID
                cv2.putText(
                    frame, f"ID: {obj_id}", 
                    (10, 30 + 30 * color_idx), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )
        
        # Write frame to output video
        out.write(frame)
        frame_idx += 1
        
        # Print progress
        if frame_idx % 20 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Saved video with masks to: {output_path}")


def create_side_by_side_video(video1, video2, output_path):
    """Create side-by-side comparison video using ffmpeg."""
    print(f"Creating side-by-side comparison of {video1} and {video2}...")
    # Use -loglevel error to reduce ffmpeg output verbosity
    os.system(f"ffmpeg -loglevel error -i {video1} -i {video2} -filter_complex \"[0:v][1:v]hstack=inputs=2\" {output_path} -y")
    if os.path.exists(output_path):
        print(f"Successfully created comparison video: {output_path}")
    else:
        print(f"Failed to create comparison video: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM2 Test-Time Augmentation demo")
    parser.add_argument("input", help="Input image or video file path")
    parser.add_argument("--output_dir", default="./output", help="Output directory for results")
    parser.add_argument("--use_tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--box", nargs=4, type=int, help="Bounding box coordinates (x1 y1 x2 y2)")
    args = parser.parse_args()
    
    # Determine if input is image or video
    _, ext = os.path.splitext(args.input.lower())
    is_video = ext in ['.mp4', '.avi', '.mov', '.mkv']
    
    if is_video:
        process_video(args.input, args.output_dir, args.use_tta, args.box)
    else:
        process_image(args.input, args.output_dir, args.use_tta, args.box)


if __name__ == "__main__":
    main()
