"""
Quick demo script for testing SAM2 with Test-Time Augmentation (TTA).
This script can process both images and videos, and saves overlay output for visual comparison.
"""

import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image

from sam2.build_sam import build_sam2_hf
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.tta import TTAManager  # Explicit import for clarity


def overlay_masks_on_image(image, masks, alpha=0.5, colors=None):
    """Overlay masks on an image with different colors.
    
    Args:
        image: Input image (HxWx3 numpy array)
        masks: List of masks or single mask (HxW numpy array)
        alpha: Transparency of the mask overlay (0-1)
        colors: List of RGB tuples for mask colors
        
    Returns:
        Image with masks overlaid
    """
    # Ensure masks is a batch
    if len(np.array(masks).shape) == 2:
        masks = [masks]
    
    if colors is None:
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
    
    overlay = image.copy()
    h, w = image.shape[:2]
    
    for i, mask in enumerate(masks):
        # Convert mask to boolean if it's a float array
        if isinstance(mask, np.ndarray) and mask.dtype != np.bool_:
            if mask.max() <= 1.0:
                # Mask is in range [0, 1]
                binary_mask = mask > 0.5
            else:
                # Mask is in range [0, 255]
                binary_mask = mask > 127
        else:
            binary_mask = mask
        
        # Ensure mask has correct dimensions
        if binary_mask.shape[:2] != (h, w):
            binary_mask = cv2.resize(
                binary_mask.astype(np.uint8), 
                (w, h), 
                interpolation=cv2.INTER_NEAREST
            )
            binary_mask = binary_mask > 0
            
        color = colors[i % len(colors)]
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        colored_mask[binary_mask] = color
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
        
        # Add a small label with mask index
        cv2.putText(
            overlay, f"Mask {i+1}", 
            (10, 30 + 30 * i), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
    
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


def process_video(video_path, output_dir, use_tta=False):
    """Process a video with SAM2 and save the results."""
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
    process_result = process_single_video(predictor, video_path, baseline_output_path, use_tta=False)
    print(f"Saved baseline video to: {baseline_output_path}")
    
    # Process with TTA if requested
    if use_tta:
        print("\n=== Processing with TTA ===")
        process_result = process_single_video(predictor, video_path, tta_output_path, use_tta=True)
        print(f"Saved TTA video to: {tta_output_path}")
        
        # Create side-by-side comparison
        create_side_by_side_video(
            baseline_output_path,
            tta_output_path,
            os.path.join(output_dir, f"{video_name}_comparison.mp4")
        )
        print(f"Saved side-by-side comparison to: {os.path.join(output_dir, f'{video_name}_comparison.mp4')}")


def process_single_video(predictor, video_path, output_path, use_tta=False):
    """Process a single video with or without TTA and save the results."""
    # Initialize inference state
    inference_state = predictor.init_state(
        video_path=video_path,
        offload_video_to_cpu=True
    )
    
    # Add a click at the center of the first frame
    first_frame = inference_state["images"][0].cpu().numpy().transpose(1, 2, 0)
    h, w = first_frame.shape[:2]
    point = [[w//2, h//2]]
    label = [1]  # foreground
    
    print(f"Adding point at {point} with label {label} to frame 0")
    
    # Add point to first frame
    frame_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
        inference_state,
        frame_idx=0,
        obj_id=1,
        points=point,
        labels=label,
    )
    
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
        video_segments[frame_idx] = {
            obj_id: (mask_logits[i] > predictor.mask_threshold).cpu().numpy()
            for i, obj_id in enumerate(obj_ids)
        }
        
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
                if binary_mask.shape != (frame_height, frame_width):
                    binary_mask = cv2.resize(
                        binary_mask.astype(np.uint8), 
                        (frame_width, frame_height), 
                        interpolation=cv2.INTER_NEAREST
                    )
                
                # Create colored mask overlay
                mask_overlay = np.zeros_like(frame)
                mask_overlay[binary_mask > 0] = color
                
                # Overlay mask on frame
                alpha = 0.5
                frame = cv2.addWeighted(frame, 1.0, mask_overlay, alpha, 0)
                
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
        process_video(args.input, args.output_dir, args.use_tta)
    else:
        process_image(args.input, args.output_dir, args.use_tta, args.box)


if __name__ == "__main__":
    main()
