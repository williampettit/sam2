import subprocess
import os
import sys
import json
import time
import itertools
from typing import List, TypedDict

# Add parent directory to path for importing sam2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import SAM2 components
from sam2.utils.tta import TTAAggregationMethod, TTAAugmentationName


class TTACombination(TypedDict):
    name: str
    skip: bool
    enabled_augmentations: List[TTAAugmentationName]
    agg_method: TTAAggregationMethod


def generate_combinations_for_experiment_3(skip: bool) -> List[TTACombination]:
    """Generate all combinations of 'decrease' augmentations with 'grayscale' always enabled."""
    decrease_augs = [
        "decrease_brightness",
        "decrease_contrast",
        "decrease_saturation",
    ]
    
    combos: List[TTACombination] = []

    # iterate over all subsets of decrease augmentations
    for r in range(len(decrease_augs) + 1):
        for subset in itertools.combinations(decrease_augs, r):
            enabled = ["grayscale", *list(subset)]
            
            if subset:
                name = f"Grayscale + {', '.join(subset)}"
            else:
                name = "Grayscale only"
            
            combos.append({
                "skip": skip,
                "name": f"{name}, max aggregation",
                "enabled_augmentations": enabled,
                "agg_method": "max",
            })
    
    return combos


# Run benchmark_image_predictor_with_coco.py for each combination
def main():
    # Base arguments
    model_size = "tiny"
    max_images = 25
    seed = 123

    # TTA combinations to try
    combinations_to_try: List[TTACombination] = [
        #
        # Experiment 1
        # We try all augmentations with max aggregation and mean aggregation
        #
        # Results:
        #   Max Aggregation Results:
        #     IoU Improvement: 3.45%
        #     Boundary F1 Improvement: 2.82%
        #     Time Increase: 635.44%
        #
        #   Mean Aggregation Results:
        #     IoU Improvement: 0.24%
        #     Boundary F1 Improvement: 0.28%
        #     Time Increase: 673.08%
        #
        # Conclusion:
        #   We found that max aggregation is much better than mean aggregation.
        #   We will focus on max aggregation in our experiments from now on.
        #
        {
            "skip": True,
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
            "skip": True,
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

        #
        # Experiment 2
        # We will try remove some augmentations to decrease inference cost, while maintaining performance.
        #
        # Results:
        #   Removing 'decrease' augmentations:
        #     IoU Improvement: 3.16%
        #     Boundary F1 Improvement: 2.52%
        #     Time Increase: 409.96%
        #  
        #   Removing 'increase' augmentations:
        #     IoU Improvement: 3.45%
        #     Boundary F1 Improvement: 2.55%
        #     Time Increase: 380.84%
        #  
        #   Removing 'decrease' and 'increase' augmentations (leaving only 'grayscale' enabled):
        #     IoU Improvement: 3.10%
        #     Boundary F1 Improvement: 2.08%
        #     Time Increase: 125.54%
        #
        # Conclusion:
        #   Trivially, all combinations where we removed augmentations resulted in less inference cost.
        #   Also, 'grayscale' is a cheap way to get good metrics.
        #   Removing 'increase' augmentations gave us better metrics than removing 'decrease' augmentations.
        #   We will now stick with just 'decrease' and 'grayscale' augmentations.
        #   We will do some experiments to see how many of the 'decrease' ops we can disable without hurting metrics.
        #
        {
            "skip": True,
            "name": "Remove 'decrease' augmentations, use max aggregation",
            "enabled_augmentations": [
                "grayscale",
                "increase_brightness",
                "increase_contrast",
                "increase_saturation",
                # "decrease_brightness",
                # "decrease_contrast",
                # "decrease_saturation",
            ],
            "agg_method": "max",
        },
        {
            "skip": True,
            "name": "Remove 'increase' augmentations, use max aggregation",
            "enabled_augmentations": [
                "grayscale",
                # "increase_brightness",
                # "increase_contrast",
                # "increase_saturation",
                "decrease_brightness",
                "decrease_contrast",
                "decrease_saturation",
            ],
            "agg_method": "max",
        },
        {
            "skip": True,
            "name": "Remove 'increase' and 'decrease' augmentations, use max aggregation",
            "enabled_augmentations": [
                "grayscale",
                # "increase_brightness",
                # "increase_contrast",
                # "increase_saturation",
                # "decrease_brightness",
                # "decrease_contrast",
                # "decrease_saturation",
            ],
            "agg_method": "max",
        },

        #
        # Experiment 3
        # Following our conclusion from Experiment 2, we will try to disable some of the 'decrease' augmentations.
        # We basically want to see which of the 'decrease' augmentations are better than others.
        # Depending on our results, we want to be able to conclude that we only need 1-2 of them in order to achieve good metrics.
        # In turn, this will allow us to save on inference time.
        #
        # Results:
        #   Grayscale only:
        #     IoU Improvement: 3.10%
        #     Boundary F1 Improvement: 2.08%
        #     Time Increase: 121.51%
        #
        #   Grayscale + decrease_brightness
        #     IoU Improvement: 3.32%
        #     Boundary F1 Improvement: 2.43%
        #     Time Increase: 228.22%
        #
        #   Grayscale + decrease_contrast
        #     IoU Improvement: 3.42%
        #     Boundary F1 Improvement: 2.44%
        #     Time Increase: 215.22%
        #
        #   Grayscale + decrease_saturation
        #     IoU Improvement: 3.31%
        #     Boundary F1 Improvement: 2.31%
        #     Time Increase: 229.74%
        #
        #   ... All other combinations are omitted because their performance was equal or worse, while having much higher inference times.
        #
        #  Conclusion:
        #    The metrics above show that 'decrease_contrast' yields the best improvement to IoU and F1, and also happens to have the lowest inference time.
        #    We will overall conclude that 'grayscale' + 'decrease_contrast' is a good final combination of augmentation ops.
        #
        *generate_combinations_for_experiment_3(skip=False),
    ]

    # Try all combinations
    results = []
    for combination in combinations_to_try:
        if combination["skip"]:
            print("=" * 80)
            print("Skipping combination:")
            print(combination["name"])
            print("=" * 80)
            continue

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
