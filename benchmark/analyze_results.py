"""
Analyze benchmark results from benchmark_image_predictor_with_coco.py.

This module generates summary metrics overall, and metrics excluding TTA-worse cases.
"""

import os
import sys
from pathlib import Path
import argparse

# Ensure shared module can be imported
sys.path.append(os.path.dirname(__file__))

from shared import load_results, calculate_summary, print_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SAM2 TTA benchmark results")
    parser.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Path to benchmark_results.json (default from shared.py)"
    )
    args = parser.parse_args()

    if args.results_json:
        results = load_results(Path(args.results_json))
    else:
        results = load_results()

    # Overall summary (all images)
    summary_all = calculate_summary(results, exclude_tta_worse=False)
    print_summary(summary_all, title="Overall Summary (all images)")

    # Summary excluding cases where TTA IoU <= baseline IoU
    summary_excl = calculate_summary(results, exclude_tta_worse=True)
    print_summary(summary_excl, title="Summary (TTA IoU > baseline IoU)")


if __name__ == "__main__":
    main()
