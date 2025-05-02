# SAM2 Test-Time Augmentation (TTA) Enhancements

This document summarizes the TTA features we added to the SAM2 predictors, and also provides usage instructions for our benchmark and demo scripts.

## Summary of Changes

- **TTA Integration (Image)**: The `SAM2ImagePredictor` class in `sam2/sam2_image_predictor.py` now supports TTA. This allows applying various image augmentations (e.g., grayscale, contrast changes, flips) at inference time and aggregating the resulting masks to potentially improve segmentation quality, especially for challenging images or distribution shifts.
- **TTA Integration (Video)**: Similarly, the `SAM2VideoPredictor` class in `sam2/sam2_video_predictor.py` has been enhanced to support TTA for video frame predictions, utilizing the same set of augmentations and aggregation methods.
- **Configurable Augmentations**: Users can specify which TTA augmentations to apply via the `tta_enabled_augmentations` parameter during predictor initialization. Supported augmentations are defined in `sam2/utils/tta.py` (`TTAAugmentationName`).
- **Aggregation Methods**: Multiple mask aggregation strategies (e.g., `max`, `mean`) are available via the `tta_agg_method` parameter (`TTAAggregationMethod` in `sam2/utils/tta.py`).
- **Benchmark Script Update**: The `benchmark/benchmark_image_predictor_with_coco.py` script was updated to accept space-separated TTA augmentation names, allowing for benchmarking TTA performance against the baseline on the COCO dataset.
- **TTA Combination Generation**: A new script `benchmark/compare_tta_combinations.py` was added to generate combinations of TTA augmentations for systematic evaluation.
- **TTA Demo Script**: A new script `tta_demo/main.py` was created to visually compare the original image, the baseline SAM2 prediction, and the TTA-enhanced prediction side-by-side in a grid format.

## Usage

### 1. Benchmarking TTA on COCO (`benchmark/benchmark_image_predictor_with_coco.py`)

This script compares the performance of the baseline SAM2 predictor against a TTA-enabled predictor on the COCO dataset.

**Example:**

```bash
cd benchmark
python benchmark_image_predictor_with_coco.py \
    --coco-path /path/to/your/coco/dataset \
    --model-size tiny \
    --tta-enabled-augmentations "grayscale decrease_contrast" \
    --tta-agg-method max \
    --output-dir ./benchmark_results \
    --limit 100
```

**Arguments:**

- `--coco-path`: Path to the COCO dataset (it will download `val2017` and `annotations` if not found).
- `--model-size`: SAM2 model size (`tiny`, `small`, `base-plus`, `large`).
- `--tta-enabled-augmentations`: Space-separated list of augmentations to enable (e.g., `"grayscale flip_horizontal"`). See `sam2/utils/tta.py` for valid names.
- `--tta-agg-method`: Aggregation method (`max`, `mean`).
- `--output-dir`: Directory to save results (metrics CSV, example images).
- `--limit`: (Optional) Maximum number of images to process.

### 2. Demo (`tta_demo/main.py`)

This generates a visual comparison grid showing the original image, the baseline mask overlay (blue), and the TTA mask overlay (green) for images in the directory.

**Example:**

```bash
cd tta_demo
python main.py \
    --images-dir ./images # Directory with your input images \
    --model-size tiny \
    --tta-enabled-augmentations "grayscale decrease_contrast" \
    --tta-agg-method max \
    --output ./tta_comparison_grid.png
```

**Arguments:**

- `--images-dir`: Folder containing demo images.
- `--model-size`: model size to use
- `--tta-enabled-augmentations`: Space-separated list of augmentations
- `--tta-agg-method`: Aggregation method to use.
- `--output`: Path to save the output grid image.