"""
Shared constants and functions for benchmarking SAM2 Image Predictor with and without TTA.

This module provides utility functions for loading and analyzing benchmark results.
"""

import os
import json
import numpy as np
from pathlib import Path

# Shared paths
DATA_DIR = Path(os.path.expanduser("~/data/coco_benchmark"))
RESULTS_DIR = DATA_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"


def load_results(results_path: Path = RESULTS_JSON) -> list:
    """Load benchmark results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def calculate_summary(results: list, exclude_tta_worse: bool = False) -> dict:
    """Calculate summary metrics. If exclude_tta_worse, exclude entries where TTA IoU <= baseline IoU."""
    filtered = [
        r for r in results
        if not exclude_tta_worse
        or r['metrics']['tta']['iou'] > r['metrics']['baseline']['iou']
    ]
    count = len(filtered)
    if count == 0:
        return {}
    iou_baseline = np.array([r['metrics']['baseline']['iou'] for r in filtered])
    iou_tta = np.array([r['metrics']['tta']['iou'] for r in filtered])
    f1_baseline = np.array([r['metrics']['baseline']['boundary_f1'] for r in filtered])
    f1_tta = np.array([r['metrics']['tta']['boundary_f1'] for r in filtered])
    time_baseline = np.array([r['metrics']['baseline']['time'] for r in filtered])
    time_tta = np.array([r['metrics']['tta']['time'] for r in filtered])
    return {
        'count': count,
        'avg_iou_baseline': float(iou_baseline.mean()),
        'avg_iou_tta': float(iou_tta.mean()),
        'avg_boundary_f1_baseline': float(f1_baseline.mean()),
        'avg_boundary_f1_tta': float(f1_tta.mean()),
        'avg_time_baseline': float(time_baseline.mean()),
        'avg_time_tta': float(time_tta.mean()),
        'iou_improvement': float((iou_tta.mean() - iou_baseline.mean()) / iou_baseline.mean() * 100),
        'boundary_f1_improvement': float((f1_tta.mean() - f1_baseline.mean()) / f1_baseline.mean() * 100),
        'time_increase': float((time_tta.mean() - time_baseline.mean()) / time_baseline.mean() * 100),
    }


def print_summary(summary: dict, title: str = "Summary") -> None:
    """Print summary metrics."""
    print(f"\n===== {title} =====")
    if not summary:
        print("No data to display.")
        return
    print(f"Number of images: {summary['count']}")
    print(f"Mean IoU - Baseline: {summary['avg_iou_baseline']:.4f}, TTA: {summary['avg_iou_tta']:.4f}")
    print(f"Mean Boundary F1 - Baseline: {summary['avg_boundary_f1_baseline']:.4f}, TTA: {summary['avg_boundary_f1_tta']:.4f}")
    print(f"Mean Inference Time - Baseline: {summary['avg_time_baseline']:.4f}s, TTA: {summary['avg_time_tta']:.4f}s")
    print(f"IoU Improvement: {summary['iou_improvement']:.2f}%")
    print(f"Boundary F1 Improvement: {summary['boundary_f1_improvement']:.2f}%")
    print(f"Time Increase: {summary['time_increase']:.2f}%")
