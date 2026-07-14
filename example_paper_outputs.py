#!/usr/bin/env python3
"""
Example script demonstrating programmatic usage of the paper_outputs mode.
This shows how to use the paper_outputs function directly from Python.
"""

from pathlib import Path
from app_dep import paper_outputs

# Define your model paths
MODEL_PATHS = [
    "/path/to/model_v1.pt",
    "/path/to/model_v2.pt",
    "/path/to/model_v3.pt",
    "/path/to/model_v4.pt",
    "/path/to/model_v5.pt",
    "/path/to/model_v6.pt",
    "/path/to/model_v7.pt",
    "/path/to/model_v8.pt",
    "/path/to/model_v9.pt",
]

# Define display names for each model
MODEL_NAMES = [
    "V1-Baseline",
    "V2-Augmented",
    "V3-DataAug",
    "V4-Finetune",
    "V5-LowRank",
    "V6-Ensemble",
    "V7-Distilled",
    "V8-Optimized",
    "V9-Final",
]

# Test dataset path
TEST_DATA = "/home/mekala/Desktop/evals/real_test"

# Output directory
OUTPUT_ROOT = "./paper_comparison_results"

def main():
    """Run paper outputs comparison."""
    
    # Call the paper_outputs function
    result = paper_outputs(
        model_paths=MODEL_PATHS,
        test_dataset=TEST_DATA,
        model_names=MODEL_NAMES,
        output_root=OUTPUT_ROOT,
        imgsz=640,
        conf=0.25,
        iou_threshold=0.5,
    )
    
    # Print summary
    print("\n" + "="*70)
    print("PAPER OUTPUTS GENERATION COMPLETE")
    print("="*70)
    
    print(f"\nOutput Directory: {result.output_root}")
    print(f"Metrics Table: {result.metrics_table}")
    print(f"Combined PR Curve: {result.combined_pr_curve}")
    
    print("\nComparison Summary:")
    print(f"  Total Models: {result.comparison_summary['num_models']}")
    print(f"  Total Test Images: {result.comparison_summary['total_test_images']}")
    print(f"  Top Model: {result.comparison_summary['top_model']}")
    print(f"  Top AP: {result.comparison_summary['top_ap']:.4f}")
    print(f"  Mean AP: {result.comparison_summary['mean_ap']:.4f} ± {result.comparison_summary['std_ap']:.4f}")
    
    print("\nPer-Model Results:")
    print("-" * 70)
    for model_output in result.model_outputs:
        print(f"\n{model_output['name']}:")
        print(f"  AP: {model_output['ap']:.4f}")
        print(f"  Predictions: {model_output['predictions']}")
        print(f"  PR Curve: {model_output['pr_curve']}")
        print(f"  Confidence Plot: {model_output['confidence_plot']}")
    
    print("\n" + "="*70)
    print("Files ready for paper/presentation:")
    print(f"  1. {result.combined_pr_curve}")
    print(f"  2. {result.metrics_table}")
    print("="*70)


if __name__ == "__main__":
    main()
