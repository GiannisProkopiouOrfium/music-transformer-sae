#!/usr/bin/env python3
"""Helper script to find optimal thresholds for concepts.

This script analyzes the distribution of metrics in your dataset
and suggests balanced thresholds.
"""

import argparse
import json
import logging
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import config
from data_curator import (
    load_and_segment_json,
    calculate_metric,
)


def analyze_metric_distribution(
    concept: str = "velocity",
    n_beats: int = 16,
    sample_size: int = 1000,
):
    """Analyze the distribution of a metric in the dataset.

    Args:
        concept: Concept to analyze
        n_beats: Segment length in beats
        sample_size: Number of files to sample (None = all)
    """
    print("=" * 60)
    print(f"Analyzing {concept} distribution")
    print("=" * 60)

    # Get JSON files
    json_files = sorted(config.JSON_DIR.rglob("*.json"))

    # Filter by train split
    train_names_file = config.DATA_DIR / "train-names.txt"
    if train_names_file.exists():
        with open(train_names_file, "r") as f:
            train_names = {line.strip() for line in f if line.strip()}

        json_files = [
            jf
            for jf in json_files
            if jf.relative_to(config.JSON_DIR).with_suffix("").as_posix() in train_names
        ]

    print(f"\nFound {len(json_files)} files in train split")

    # Sample if requested
    if sample_size and sample_size < len(json_files):
        np.random.seed(42)
        json_files = list(np.random.choice(json_files, sample_size, replace=False))
        print(f"Sampling {sample_size} files for faster analysis")

    # Collect metrics
    print(f"\nProcessing files...")
    all_metrics = []

    for json_path in json_files[:sample_size] if sample_size else json_files:
        try:
            segments = load_and_segment_json(json_path, n_beats, resolution=12)

            for segment in segments:
                metric_value = calculate_metric(segment, concept)
                if metric_value > 0:  # Skip empty segments
                    all_metrics.append(metric_value)

        except Exception as e:
            continue

    all_metrics = np.array(all_metrics)

    print(f"\nCollected {len(all_metrics)} segments")

    # Calculate statistics
    print("\n" + "=" * 60)
    print("DISTRIBUTION STATISTICS")
    print("=" * 60)
    print(f"Mean:     {np.mean(all_metrics):.2f}")
    print(f"Median:   {np.median(all_metrics):.2f}")
    print(f"Std Dev:  {np.std(all_metrics):.2f}")
    print(f"Min:      {np.min(all_metrics):.2f}")
    print(f"Max:      {np.max(all_metrics):.2f}")

    # Percentiles
    print("\n" + "=" * 60)
    print("PERCENTILES")
    print("=" * 60)
    percentiles = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95]
    for p in percentiles:
        value = np.percentile(all_metrics, p)
        print(f"{p:3d}th percentile: {value:6.2f}")

    # Suggest thresholds
    print("\n" + "=" * 60)
    print("SUGGESTED THRESHOLDS")
    print("=" * 60)

    # Option 1: 20/80 split (extreme contrast)
    p20 = np.percentile(all_metrics, 20)
    p80 = np.percentile(all_metrics, 80)
    print(f"\nOption 1 (Extreme Contrast - 20th/80th percentile):")
    print(f"  low_threshold:  {p20:.1f}")
    print(f"  high_threshold: {p80:.1f}")
    count_low_1 = np.sum(all_metrics <= p20)
    count_high_1 = np.sum(all_metrics >= p80)
    print(
        f"  → Low samples:  {count_low_1:,} ({100*count_low_1/len(all_metrics):.1f}%)"
    )
    print(
        f"  → High samples: {count_high_1:,} ({100*count_high_1/len(all_metrics):.1f}%)"
    )

    # Option 2: 30/70 split (moderate contrast)
    p30 = np.percentile(all_metrics, 30)
    p70 = np.percentile(all_metrics, 70)
    print(f"\nOption 2 (Moderate Contrast - 30th/70th percentile):")
    print(f"  low_threshold:  {p30:.1f}")
    print(f"  high_threshold: {p70:.1f}")
    count_low_2 = np.sum(all_metrics <= p30)
    count_high_2 = np.sum(all_metrics >= p70)
    print(
        f"  → Low samples:  {count_low_2:,} ({100*count_low_2/len(all_metrics):.1f}%)"
    )
    print(
        f"  → High samples: {count_high_2:,} ({100*count_high_2/len(all_metrics):.1f}%)"
    )

    # Option 3: 40/60 split (wide coverage)
    p40 = np.percentile(all_metrics, 40)
    p60 = np.percentile(all_metrics, 60)
    print(f"\nOption 3 (Wide Coverage - 40th/60th percentile):")
    print(f"  low_threshold:  {p40:.1f}")
    print(f"  high_threshold: {p60:.1f}")
    count_low_3 = np.sum(all_metrics <= p40)
    count_high_3 = np.sum(all_metrics >= p60)
    print(
        f"  → Low samples:  {count_low_3:,} ({100*count_low_3/len(all_metrics):.1f}%)"
    )
    print(
        f"  → High samples: {count_high_3:,} ({100*count_high_3/len(all_metrics):.1f}%)"
    )

    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(f"\nFor best results, use Option 2 (Moderate Contrast):")
    print(f"")
    print(f'    "velocity": {{')
    print(f'        "high_threshold": {p70:.0f},')
    print(f'        "low_threshold": {p30:.0f},')
    print(f"    }},")
    print(f"")
    print(
        f"This gives you ~{count_high_2:,} high samples and ~{count_low_2:,} low samples."
    )
    print("=" * 60)

    # Plot histogram
    try:
        plt.figure(figsize=(12, 6))

        plt.hist(all_metrics, bins=50, alpha=0.7, edgecolor="black")
        plt.axvline(
            p30,
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"Low threshold (30th: {p30:.1f})",
        )
        plt.axvline(
            p70,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"High threshold (70th: {p70:.1f})",
        )
        plt.axvline(
            np.mean(all_metrics),
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"Mean: {np.mean(all_metrics):.1f}",
        )
        plt.axvline(
            np.median(all_metrics),
            color="orange",
            linestyle="-",
            linewidth=2,
            label=f"Median: {np.median(all_metrics):.1f}",
        )

        plt.xlabel(f"{concept.capitalize()}", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.title(
            f"Distribution of {concept.capitalize()} in Dataset",
            fontsize=14,
            fontweight="bold",
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = config.OUTPUT_DIR / "datasets" / f"{concept}_distribution.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Saved distribution plot to: {output_path}")
        plt.close()
    except Exception as e:
        print(f"\nCouldn't create plot: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Find optimal thresholds for concepts")
    parser.add_argument(
        "--concept",
        type=str,
        default="velocity",
        choices=list(config.CONCEPTS.keys()),
        help="Concept to analyze",
    )
    parser.add_argument(
        "--n_beats", type=int, default=16, help="Segment length in beats"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of files to sample (None = all)",
    )

    args = parser.parse_args()

    analyze_metric_distribution(
        concept=args.concept,
        n_beats=args.n_beats,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
