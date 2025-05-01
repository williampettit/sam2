import subprocess
import os
import sys
import json
import time
from typing import List, TypedDict

# Add parent directory to path for importing sam2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SAM2 components
from sam2.utils.tta import TTAAggregationMethod, TTAAugmentationName


class TTACombination(TypedDict):
    name: str
    enabled_augmentations: List[TTAAugmentationName]
    agg_method: TTAAggregationMethod


# Run benchmark_image_predictor_with_coco.py for each combination
def main():
    # Base arguments
    model_size = "tiny"
    max_images = 25
    seed = 123

    # TTA combinations to try
    combinations_to_try: List[TTACombination] = [
        {
            "name": "All augmentations with max aggregation",
            "enabled_augmentations": [
                "grayscale",
                "increase_brightness",
                "increase_contrast",
                "increase_saturation",
                "decrease_brightness",
                "decrease_contrast",
                "decrease_saturation",
            ],
            "agg_method": "max",
        },

        {
            "name": "All augmentations with mean aggregation",
            "enabled_augmentations": [
                "grayscale",
                "increase_brightness",
                "increase_contrast",
                "increase_saturation",
                "decrease_brightness",
                "decrease_contrast",
                "decrease_saturation",
            ],
            "agg_method": "mean",
        },
    ]

    # Try all combinations
    results = []
    for combination in combinations_to_try:
        print("=" * 80)
        print("Trying combination:")
        print(combination["name"])
        print("=" * 80)

        result = subprocess.run(
            [
                "python",
                os.path.join(os.path.dirname(__file__), "benchmark_image_predictor_with_coco.py"),
                "--max_images", str(max_images),
                "--model_size", model_size,
                "--seed", str(seed),
                "--tta_enabled_augmentations", " ".join(combination["enabled_augmentations"]),
                "--tta_agg_method", combination["agg_method"],
            ]
        )

        print(result)
        print("=" * 80)

        results.append({
            "combination_data": combination,
            "raw_result_output": result.stdout,
            "raw_result_returncode": result.returncode,
        })

    # Save results
    with open("tta_combinations_results.json", "w") as f:
        json.dump(
            {
                "results": results,
                "model_size": model_size,
                "max_images": max_images,
                "seed": seed,
                "timestamp": str(int(time.time())),
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    main()
