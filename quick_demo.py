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


def overlay_masks_on_image(image, masks, alpha=0.5, colors=None):
    """Overlay masks on an image with different colors."""
    # Ensure masks is a batch
    if len(np.array(masks).shape) == 2:
        masks = [masks]
    
    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                   (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
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
            
        color = colors[i % len(colors)]
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        colored_mask[binary_mask] = color
        overlay = cv2.addWeighted(overlay, 1, colored_mask, alpha, 0)
    
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
    sam2_model = build_sam2_hf(
        model_id="facebook/sam2-hiera-base-plus",
        device=device
    )
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-base-plus")
    
    # Create output paths
    video_name = os.path.basename(video_path).split('.')[0]
    baseline_output_path = os.path.join(output_dir, f"{video_name}_baseline.mp4")
    tta_output_path = os.path.join(output_dir, f"{video_name}_tta.mp4") if use_tta else None
    
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
    
    # Process with baseline
    predictor.add_new_points_or_box(
        inference_state,
        frame_idx=0,
        obj_id=1,
        points=point,
        labels=label,
        use_tta=False
    )
    
    # Propagate through the video
    predictor.propagate_in_video(inference_state, use_tta=False)
    
    # Save baseline video
    save_video_with_masks(video_path, inference_state, baseline_output_path, obj_id=1)
    print(f"Saved baseline video to: {baseline_output_path}")
    
    if use_tta:
        # Reset state for TTA processing
        inference_state = predictor.init_state(
            video_path=video_path,
            offload_video_to_cpu=True
        )
        
        # Process with TTA
        predictor.add_new_points_or_box(
            inference_state,
            frame_idx=0,
            obj_id=1,
            points=point,
            labels=label,
            use_tta=True
        )
        
        # Propagate through the video with TTA
        predictor.propagate_in_video(inference_state, use_tta=True)
        
        # Save TTA video
        save_video_with_masks(video_path, inference_state, tta_output_path, obj_id=1)
        print(f"Saved TTA video to: {tta_output_path}")
        
        # Create side-by-side comparison
        create_side_by_side_video(
            baseline_output_path,
            tta_output_path,
            os.path.join(output_dir, f"{video_name}_comparison.mp4")
        )
        print(f"Saved side-by-side comparison to: {os.path.join(output_dir, f'{video_name}_comparison.mp4')}")


def save_video_with_masks(video_path, inference_state, output_path, obj_id=1):
    """Save a video with masks overlaid."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_idx = 0
    obj_idx = inference_state["obj_id_to_idx"][obj_id]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get the mask for this frame
        output_dict = inference_state["output_dict_per_obj"][obj_idx]
        mask = None
        
        # Check in both cond_frame_outputs and non_cond_frame_outputs
        if frame_idx in output_dict["cond_frame_outputs"]:
            mask = output_dict["cond_frame_outputs"][frame_idx]["pred_masks"]
        elif frame_idx in output_dict["non_cond_frame_outputs"]:
            mask = output_dict["non_cond_frame_outputs"][frame_idx]["pred_masks"]
        
        if mask is not None:
            # Resize the mask to match the frame size
            mask = mask.cpu().squeeze().numpy() > 0.5
            mask = cv2.resize(mask.astype(np.uint8), (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
            
            # Create a colored mask overlay
            mask_overlay = np.zeros_like(frame)
            mask_overlay[mask == 1] = [0, 0, 255]  # Red color for the mask
            
            # Add the mask overlay to the frame
            alpha = 0.5
            frame = cv2.addWeighted(frame, 1, mask_overlay, alpha, 0)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()


def create_side_by_side_video(video1, video2, output_path):
    """Create side-by-side comparison video using ffmpeg."""
    os.system(f"ffmpeg -i {video1} -i {video2} -filter_complex \"[0:v][1:v]hstack=inputs=2\" {output_path} -y")


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
