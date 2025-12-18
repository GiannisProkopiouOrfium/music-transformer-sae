"""Visualize Layer-Alpha Optimization Results.

This script creates visualizations for the layer-alpha grid search results.

Usage:
    python experiments/visualize_results.py \\
        --results_file steering_interventions/experiments/results/layer_alpha_grid_average_pitch.json \\
        --output_dir steering_interventions/experiments/results/figures
"""

import argparse
import json
import logging
import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def setup_logging(output_dir: pathlib.Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "visualize_results.log"),
            logging.StreamHandler(),
        ],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize optimization results")
    parser.add_argument(
        "--results_file",
        type=pathlib.Path,
        required=True,
        help="Path to layer_alpha_grid results JSON",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("steering_interventions/experiments/results/figures"),
        help="Output directory for figures",
    )
    return parser.parse_args()


def plot_heatmap_pitch_shift(
    results: Dict, layer_groups: Dict, alpha_values: list, output_path: pathlib.Path
):
    """Plot heatmap of pitch shift by layer and alpha."""
    # Prepare data matrix
    layer_names = list(layer_groups.keys())
    data = np.zeros((len(layer_names), len(alpha_values)))

    for i, layer_name in enumerate(layer_names):
        for j, alpha in enumerate(alpha_values):
            if str(alpha) in results[layer_name]:
                data[i, j] = results[layer_name][str(alpha)].get(
                    "pitch_shift_from_baseline", 0.0
                )

    # Create heatmap
    _, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-30, vmax=30)

    # Set ticks
    ax.set_xticks(np.arange(len(alpha_values)))
    ax.set_yticks(np.arange(len(layer_names)))
    ax.set_xticklabels([f"{a:+.1f}" for a in alpha_values])
    ax.set_yticklabels(layer_names)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Pitch Shift from Baseline (MIDI notes)", fontsize=12)

    # Add values as text
    for i in range(len(layer_names)):
        for j in range(len(alpha_values)):
            text = ax.text(
                j,
                i,
                f"{data[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    ax.set_xlabel("Alpha (Steering Strength)", fontsize=12)
    ax.set_ylabel("Layer Group", fontsize=12)
    ax.set_title("Pitch Shift by Layer and Alpha", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap_degradation(
    results: Dict, layer_groups: Dict, alpha_values: list, output_path: pathlib.Path
):
    """Plot heatmap of quality degradation by layer and alpha."""
    layer_names = list(layer_groups.keys())
    data = np.zeros((len(layer_names), len(alpha_values)))

    for i, layer_name in enumerate(layer_names):
        for j, alpha in enumerate(alpha_values):
            if str(alpha) in results[layer_name]:
                degradation = results[layer_name][str(alpha)].get("degradation", {})
                data[i, j] = degradation.get("total_degradation", 0.0)

    # Create heatmap
    _, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=5)

    # Set ticks
    ax.set_xticks(np.arange(len(alpha_values)))
    ax.set_yticks(np.arange(len(layer_names)))
    ax.set_xticklabels([f"{a:+.1f}" for a in alpha_values])
    ax.set_yticklabels(layer_names)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Quality Degradation Score (Lower = Better)", fontsize=12)

    # Add values as text
    for i in range(len(layer_names)):
        for j in range(len(alpha_values)):
            text = ax.text(
                j,
                i,
                f"{data[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if data[i, j] > 2.5 else "black",
                fontsize=8,
            )

    ax.set_xlabel("Alpha (Steering Strength)", fontsize=12)
    ax.set_ylabel("Layer Group", fontsize=12)
    ax.set_title(
        "Quality Degradation by Layer and Alpha", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_line_per_layer(
    results: Dict, layer_groups: Dict, alpha_values: list, output_path: pathlib.Path
):
    """Plot line plots of mean pitch per layer group."""
    _, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Mean pitch
    ax = axes[0]
    for layer_name in layer_groups.keys():
        pitches = []
        for alpha in alpha_values:
            if str(alpha) in results[layer_name]:
                pitches.append(results[layer_name][str(alpha)]["mean_pitch"])
            else:
                pitches.append(np.nan)
        ax.plot(
            alpha_values, pitches, "o-", label=layer_name, linewidth=2, markersize=6
        )

    ax.set_xlabel("Alpha (Steering Strength)", fontsize=12)
    ax.set_ylabel("Mean Pitch (MIDI)", fontsize=12)
    ax.set_title("Mean Pitch by Layer Group", fontsize=12, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=60, color="gray", linestyle="--", alpha=0.5, label="Middle C")

    # Right: Pitch class entropy
    ax = axes[1]
    for layer_name in layer_groups.keys():
        entropies = []
        for alpha in alpha_values:
            if str(alpha) in results[layer_name]:
                entropies.append(results[layer_name][str(alpha)]["pitch_class_entropy"])
            else:
                entropies.append(np.nan)
        ax.plot(
            alpha_values, entropies, "o-", label=layer_name, linewidth=2, markersize=6
        )

    ax.set_xlabel("Alpha (Steering Strength)", fontsize=12)
    ax.set_ylabel("Pitch Class Entropy", fontsize=12)
    ax.set_title("Pitch Class Entropy by Layer Group", fontsize=12, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=2.802, color="red", linestyle="--", alpha=0.5, label="Baseline")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_pareto_frontier(
    results: Dict, layer_groups: Dict, alpha_values: list, output_path: pathlib.Path
):
    """Plot Pareto frontier of steering effectiveness vs quality."""
    _, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(layer_groups)))

    for idx, layer_name in enumerate(layer_groups.keys()):
        x_vals = []  # Steering effectiveness (absolute pitch shift)
        y_vals = []  # Quality preservation (negative degradation)

        for alpha in alpha_values:
            if alpha == 0.0:  # Skip baseline
                continue
            if str(alpha) in results[layer_name]:
                r = results[layer_name][str(alpha)]
                if "pitch_shift_from_baseline" in r and "degradation" in r:
                    x_vals.append(abs(r["pitch_shift_from_baseline"]))
                    y_vals.append(-r["degradation"]["total_degradation"])

        ax.scatter(
            x_vals, y_vals, label=layer_name, s=100, alpha=0.7, color=colors[idx]
        )

    ax.set_xlabel("Steering Effectiveness (|Pitch Shift| from Baseline)", fontsize=12)
    ax.set_ylabel("Quality Preservation (-Degradation Score)", fontsize=12)
    ax.set_title(
        "Pareto Frontier: Steering vs Quality Trade-off", fontsize=14, fontweight="bold"
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    # Add quadrant lines
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_quality_metrics_boxplot(
    results: Dict, layer_groups: Dict, alpha_values: list, output_path: pathlib.Path
):
    """Plot box plots of quality metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ["pitch_class_entropy", "scale_consistency", "groove_consistency"]
    titles = ["Pitch Class Entropy", "Scale Consistency (%)", "Groove Consistency (%)"]

    for ax, metric, title in zip(axes, metrics, titles):
        data_by_layer = []
        labels = []

        for layer_name in layer_groups.keys():
            values = []
            for alpha in alpha_values:
                if str(alpha) in results[layer_name]:
                    val = results[layer_name][str(alpha)].get(metric, np.nan)
                    if not np.isnan(val):
                        values.append(val)
            if values:
                data_by_layer.append(values)
                labels.append(layer_name)

        bp = ax.boxplot(data_by_layer, labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")

        ax.set_xlabel("Layer Group", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_strategy_comparison(comparison_data: Dict, output_path: pathlib.Path):
    """Plot comparison between All-to-All and One-to-All strategies."""
    alphas = comparison_data["alphas"]
    results = comparison_data["results"]
    best_layers = comparison_data["best_layers"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("pitch", "mean", "Mean Pitch (MIDI)"),
        ("pitch", "diversity", "Pitch Diversity (Unique Pitches)"),
        ("quality", "pitch_class_entropy", "Pitch Class Entropy"),
        ("quality", "scale_consistency", "Scale Consistency (%)"),
    ]

    for ax, (category, metric, label) in zip(axes.flat, metrics):
        # All-to-All
        ata_vals = []
        for alpha in alphas:
            if str(alpha) in results["all_to_all"]:
                ata_vals.append(results["all_to_all"][str(alpha)][category][metric])
            else:
                ata_vals.append(np.nan)
        ax.plot(alphas, ata_vals, "o-", linewidth=3, markersize=8, label="All-to-All")

        # One-to-All for each best layer
        for best_layer in best_layers:
            ota_vals = []
            for alpha in alphas:
                if str(alpha) in results["one_to_all"][str(best_layer)]:
                    ota_vals.append(
                        results["one_to_all"][str(best_layer)][str(alpha)][category][
                            metric
                        ]
                    )
                else:
                    ota_vals.append(np.nan)
            ax.plot(
                alphas,
                ota_vals,
                "s--",
                linewidth=2,
                markersize=6,
                alpha=0.7,
                label=f"One-to-All (L{best_layer})",
            )

        ax.set_xlabel("Alpha", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)

    logging.info("=" * 80)
    logging.info("VISUALIZATION OF OPTIMIZATION RESULTS")
    logging.info("=" * 80)
    logging.info(f"Results file: {args.results_file}")

    # Load results
    with open(args.results_file) as f:
        data = json.load(f)

    results = data["results"]
    layer_groups = data["layer_groups"]
    alpha_values = data["alpha_values"]
    concept = data["concept"]

    logging.info(f"Concept: {concept}")
    logging.info(f"Layer groups: {len(layer_groups)}")
    logging.info(f"Alpha values: {len(alpha_values)}")

    # Generate visualizations
    logging.info("\nGenerating visualizations...")

    # 1. Heatmap: Pitch shift
    logging.info("  1. Pitch shift heatmap...")
    plot_heatmap_pitch_shift(
        results,
        layer_groups,
        alpha_values,
        args.output_dir / f"{concept}_pitch_shift_heatmap.png",
    )

    # 2. Heatmap: Degradation
    logging.info("  2. Quality degradation heatmap...")
    plot_heatmap_degradation(
        results,
        layer_groups,
        alpha_values,
        args.output_dir / f"{concept}_degradation_heatmap.png",
    )

    # 3. Line plots
    logging.info("  3. Line plots per layer...")
    plot_line_per_layer(
        results,
        layer_groups,
        alpha_values,
        args.output_dir / f"{concept}_line_plots.png",
    )

    # 4. Pareto frontier
    logging.info("  4. Pareto frontier...")
    plot_pareto_frontier(
        results,
        layer_groups,
        alpha_values,
        args.output_dir / f"{concept}_pareto_frontier.png",
    )

    # 5. Quality metrics boxplot
    logging.info("  5. Quality metrics boxplot...")
    plot_quality_metrics_boxplot(
        results,
        layer_groups,
        alpha_values,
        args.output_dir / f"{concept}_quality_boxplot.png",
    )

    # Check for strategy comparison results
    strategy_file = args.results_file.parent / f"strategy_comparison_{concept}.json"
    if strategy_file.exists():
        logging.info("  6. Strategy comparison...")
        with open(strategy_file) as f:
            comparison_data = json.load(f)
        plot_strategy_comparison(
            comparison_data,
            args.output_dir / f"{concept}_strategy_comparison.png",
        )

    logging.info(f"\nAll figures saved to {args.output_dir}")
    logging.info("Visualization complete!")


if __name__ == "__main__":
    main()
