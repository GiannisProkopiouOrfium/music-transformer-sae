#!/usr/bin/env python3
"""Analyze dual-steering grid search results with comprehensive visualizations.

Compares direct vs gram_schmidt strategies across extended parameter grid
including negative alphas, and provides detailed analysis of:

1. Heatmaps: Pitch control, modality control, quality degradation
2. Pareto frontiers: Control strength vs quality trade-offs
3. Interaction analysis: Cross-influence between pitch and modality
4. Baseline comparison: Delta from baseline (α=0, α=0) for all configs

Metrics compared:
- Success rate (% of valid generations)
- Pitch control effectiveness (mean pitch shift per alpha)
- Modality control effectiveness (major percentage shift)
- Quality degradation (distance from ground truth metrics)
- Stability and consistency

Output:
- Strategy comparison visualizations
- Heatmap grids for each strategy
- Pareto frontier plots
- Interaction analysis plots
- Baseline delta comparisons
- Statistical analysis JSON
- Clear recommendation for next phase

Usage:
    python dual_steering/analyze_strategy_results.py \\
        --results_file steering_interventions/dual_steering/outputs/phase3_grid_search/phase3_results.json \\
        --output_dir steering_interventions/dual_steering/outputs/phase3_analysis
"""

import argparse
import json
import logging
import pathlib
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_file: pathlib.Path) -> Dict:
    """Load Phase 2 results from JSON.

    Args:
        results_file: Path to phase2_results.json

    Returns:
        Dictionary with config and results
    """
    with open(results_file) as f:
        return json.load(f)


def group_by_strategy(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by composition strategy.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary mapping strategy name to list of results
    """
    grouped = {}
    for result in results:
        strategy = result["strategy"]
        if strategy not in grouped:
            grouped[strategy] = []
        grouped[strategy].append(result)

    return grouped


def analyze_pitch_control(results: List[Dict]) -> Dict:
    """Analyze pitch control effectiveness.

    Args:
        results: List of results for one strategy

    Returns:
        Dictionary with pitch control metrics
    """
    # Group by alpha_pitch
    pitch_alphas = {}
    for r in results:
        alpha = r["alpha_pitch"]
        if alpha not in pitch_alphas:
            pitch_alphas[alpha] = []
        pitch_alphas[alpha].append(r["pitch_control"]["mean"])

    # Compute mean pitch for each alpha
    alpha_means = {
        alpha: float(np.mean(pitches)) for alpha, pitches in pitch_alphas.items()
    }

    # Measure control strength: correlation between alpha and pitch
    alphas_sorted = sorted(alpha_means.keys())
    means_sorted = [alpha_means[a] for a in alphas_sorted]

    if len(alphas_sorted) > 1:
        correlation = float(np.corrcoef(alphas_sorted, means_sorted)[0, 1])
    else:
        correlation = 0.0

    # Measure range of control
    pitch_range = max(means_sorted) - min(means_sorted) if means_sorted else 0.0

    return {
        "alpha_to_pitch": alpha_means,
        "correlation": correlation,
        "pitch_range": pitch_range,
        "control_strength": abs(correlation) * pitch_range,  # Combined metric
    }


def analyze_modality_control(results: List[Dict]) -> Dict:
    """Analyze modality control effectiveness.

    Args:
        results: List of results for one strategy

    Returns:
        Dictionary with modality control metrics
    """
    # Group by alpha_modality
    modality_alphas = {}
    for r in results:
        alpha = r["alpha_modality"]
        if alpha not in modality_alphas:
            modality_alphas[alpha] = []
        # Use major_percentage from new structure
        modality_alphas[alpha].append(r["modality_control"]["major_percentage"])

    # Compute mean major percentage for each alpha
    alpha_means = {
        alpha: float(np.mean(percentages))
        for alpha, percentages in modality_alphas.items()
    }

    # Measure control strength
    alphas_sorted = sorted(alpha_means.keys())
    means_sorted = [alpha_means[a] for a in alphas_sorted]

    if len(alphas_sorted) > 1:
        correlation = float(np.corrcoef(alphas_sorted, means_sorted)[0, 1])
    else:
        correlation = 0.0

    modality_range = max(means_sorted) - min(means_sorted) if means_sorted else 0.0

    return {
        "alpha_to_modality": alpha_means,
        "correlation": correlation,
        "modality_range": modality_range,
        "control_strength": abs(correlation) * modality_range,
    }


def analyze_stability(results: List[Dict]) -> Dict:
    """Analyze generation stability (consistency).

    Args:
        results: List of results for one strategy

    Returns:
        Dictionary with stability metrics
    """
    success_rates = [r["success_rate"] for r in results]
    pitch_stds = [r["pitch_control"]["std"] for r in results if r["valid_samples"] > 0]

    return {
        "mean_success_rate": float(np.mean(success_rates)),
        "std_success_rate": float(np.std(success_rates)),
        "mean_pitch_variability": float(np.mean(pitch_stds)) if pitch_stds else 0.0,
        "stability_score": float(np.mean(success_rates))
        * (1.0 - float(np.std(success_rates))),
    }


def analyze_quality_metrics(results: List[Dict]) -> Dict:
    """Analyze music quality metrics across results.

    Args:
        results: List of results for one strategy

    Returns:
        Dictionary with quality metric statistics
    """
    quality_summary = {}

    for metric in ["pitch_class_entropy", "scale_consistency", "groove_consistency"]:
        means = []
        for r in results:
            if r["valid_samples"] > 0:
                mean_val = r["quality_metrics"][metric]["mean"]
                if not np.isnan(mean_val):
                    means.append(mean_val)

        quality_summary[metric] = {
            "overall_mean": float(np.mean(means)) if means else np.nan,
            "overall_std": float(np.std(means)) if means else np.nan,
        }

    return quality_summary


def analyze_degradation(results: List[Dict]) -> Dict:
    """Analyze quality degradation across results.

    Args:
        results: List of results for one strategy

    Returns:
        Dictionary with degradation statistics
    """
    degradation_summary = {}

    for metric in ["entropy_diff", "scale_diff", "groove_diff", "total_degradation"]:
        means = []
        for r in results:
            if r["valid_samples"] > 0:
                mean_val = r["degradation"][metric]["mean"]
                if not np.isnan(mean_val):
                    means.append(mean_val)

        degradation_summary[metric] = {
            "overall_mean": float(np.mean(means)) if means else np.nan,
            "overall_std": float(np.std(means)) if means else np.nan,
        }

    return degradation_summary


def compare_strategies(grouped_results: Dict[str, List[Dict]]) -> Dict:
    """Compare all strategies across all metrics.

    Args:
        grouped_results: Results grouped by strategy

    Returns:
        Dictionary with comparison results
    """
    comparison = {}

    for strategy, results in grouped_results.items():
        pitch_analysis = analyze_pitch_control(results)
        modality_analysis = analyze_modality_control(results)
        stability_analysis = analyze_stability(results)
        quality_analysis = analyze_quality_metrics(results)
        degradation_analysis = analyze_degradation(results)

        # Calculate degradation penalty (lower is better)
        total_deg = degradation_analysis["total_degradation"]["overall_mean"]
        degradation_penalty = 0.0 if np.isnan(total_deg) else min(total_deg / 10.0, 1.0)

        comparison[strategy] = {
            "pitch_control": pitch_analysis,
            "modality_control": modality_analysis,
            "stability": stability_analysis,
            "quality_metrics": quality_analysis,
            "degradation": degradation_analysis,
            # Overall score (weighted average with degradation penalty)
            "overall_score": (
                0.30 * pitch_analysis["control_strength"]
                + 0.30 * modality_analysis["control_strength"]
                + 0.25 * stability_analysis["stability_score"]
                - 0.15 * degradation_penalty  # Penalize quality loss
            ),
        }

    return comparison


def visualize_comparison(
    grouped_results: Dict[str, List[Dict]],
    comparison: Dict,
    output_dir: pathlib.Path,
):
    """Create visualization comparing strategies.

    Args:
        grouped_results: Results grouped by strategy
        comparison: Strategy comparison results
        output_dir: Directory to save figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(
        "Phase 2: Dual-Steering Strategy Comparison", fontsize=16, fontweight="bold"
    )

    strategies = sorted(grouped_results.keys())
    colors = {"direct": "#2E86AB", "gram_schmidt": "#A23B72"}

    # 1. Success rate comparison
    ax = axes[0, 0]
    success_rates = [
        comparison[s]["stability"]["mean_success_rate"] for s in strategies
    ]
    bars = ax.bar(
        strategies, success_rates, color=[colors.get(s, "gray") for s in strategies]
    )
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Generation Success Rate", fontsize=13)
    ax.set_ylim(0, 1.0)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2%}",
            ha="center",
            va="bottom",
        )

    # 2. Pitch control strength
    ax = axes[0, 1]
    pitch_strengths = [
        comparison[s]["pitch_control"]["control_strength"] for s in strategies
    ]
    bars = ax.bar(
        strategies, pitch_strengths, color=[colors.get(s, "gray") for s in strategies]
    )
    ax.set_ylabel("Control Strength", fontsize=12)
    ax.set_title("Pitch Control Effectiveness", fontsize=13)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    # 3. Modality control strength
    ax = axes[0, 2]
    modality_strengths = [
        comparison[s]["modality_control"]["control_strength"] for s in strategies
    ]
    bars = ax.bar(
        strategies,
        modality_strengths,
        color=[colors.get(s, "gray") for s in strategies],
    )
    ax.set_ylabel("Control Strength", fontsize=12)
    ax.set_title("Modality Control Effectiveness", fontsize=13)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    # 4. Overall score
    ax = axes[1, 0]
    overall_scores = [comparison[s]["overall_score"] for s in strategies]
    bars = ax.bar(
        strategies, overall_scores, color=[colors.get(s, "gray") for s in strategies]
    )
    ax.set_ylabel("Overall Score", fontsize=12)
    ax.set_title("Overall Performance Score", fontsize=13)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
        )

    # 5. Pitch correlation
    ax = axes[1, 1]
    pitch_corrs = [comparison[s]["pitch_control"]["correlation"] for s in strategies]
    bars = ax.bar(
        strategies, pitch_corrs, color=[colors.get(s, "gray") for s in strategies]
    )
    ax.set_ylabel("Correlation", fontsize=12)
    ax.set_title("Pitch Alpha ↔ Pitch Mean Correlation", fontsize=13)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:+.3f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
        )

    # 6. Modality correlation
    ax = axes[1, 2]
    modality_corrs = [
        comparison[s]["modality_control"]["correlation"] for s in strategies
    ]
    bars = ax.bar(
        strategies, modality_corrs, color=[colors.get(s, "gray") for s in strategies]
    )
    ax.set_ylabel("Correlation", fontsize=12)
    ax.set_title("Modality Alpha ↔ Major % Correlation", fontsize=13)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:+.3f}",
            ha="center",
            va="bottom" if height >= 0 else "top",
        )

    # 7. Total degradation
    ax = axes[2, 0]
    total_degs = [
        comparison[s]["degradation"]["total_degradation"]["overall_mean"]
        for s in strategies
    ]
    bars = ax.bar(
        strategies, total_degs, color=[colors.get(s, "gray") for s in strategies]
    )
    ax.set_ylabel("Total Degradation", fontsize=12)
    ax.set_title("Quality Degradation (Lower is Better)", fontsize=13)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
        )

    # 8. Scale consistency (quality metric)
    ax = axes[2, 1]
    scale_cons = [
        comparison[s]["quality_metrics"]["scale_consistency"]["overall_mean"]
        for s in strategies
    ]
    bars = ax.bar(
        strategies, scale_cons, color=[colors.get(s, "gray") for s in strategies]
    )
    ax.set_ylabel("Scale Consistency (%)", fontsize=12)
    ax.set_title("Scale Consistency (Higher is Better)", fontsize=13)
    ax.axhline(y=92.26, color="green", linestyle="--", alpha=0.5, label="Ground Truth")
    ax.legend()
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    # 9. Groove consistency (quality metric)
    ax = axes[2, 2]
    groove_cons = [
        comparison[s]["quality_metrics"]["groove_consistency"]["overall_mean"]
        for s in strategies
    ]
    bars = ax.bar(
        strategies, groove_cons, color=[colors.get(s, "gray") for s in strategies]
    )
    ax.set_ylabel("Groove Consistency (%)", fontsize=12)
    ax.set_title("Groove Consistency (Higher is Better)", fontsize=13)
    ax.axhline(y=93.05, color="green", linestyle="--", alpha=0.5, label="Ground Truth")
    ax.legend()
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    fig_path = output_dir / "strategy_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved figure: {fig_path}")
    plt.show()


def determine_recommendation(comparison: Dict) -> str:
    """Determine which strategy to recommend for Phase 3.

    Args:
        comparison: Strategy comparison results

    Returns:
        Recommendation string
    """
    strategies = sorted(
        comparison.keys(), key=lambda s: comparison[s]["overall_score"], reverse=True
    )
    best = strategies[0]
    best_score = comparison[best]["overall_score"]
    best_deg = comparison[best]["degradation"]["total_degradation"]["overall_mean"]

    if len(strategies) > 1:
        second = strategies[1]
        second_score = comparison[second]["overall_score"]
        second_deg = comparison[second]["degradation"]["total_degradation"][
            "overall_mean"
        ]
        score_diff = best_score - second_score

        # Add degradation context
        deg_comparison = (
            f"\nQuality Degradation:\n"
            f"  {best}: {best_deg:.2f}\n"
            f"  {second}: {second_deg:.2f}"
        )

        if score_diff < 0.05:  # Very close
            return (
                f"🟡 CLOSE COMPETITION\n\n"
                f"Best: {best} (score: {best_score:.3f})\n"
                f"Runner-up: {second} (score: {second_score:.3f})\n"
                f"Difference: {score_diff:.3f}{deg_comparison}\n\n"
                f"Recommendation: Proceed with BOTH strategies in Phase 3 grid search.\n"
                f"The performance difference is too small to confidently eliminate one strategy."
            )
        else:
            return (
                f"🟢 CLEAR WINNER\n\n"
                f"Best: {best} (score: {best_score:.3f})\n"
                f"Runner-up: {second} (score: {second_score:.3f})\n"
                f"Difference: {score_diff:.3f}{deg_comparison}\n\n"
                f"Recommendation: Proceed with {best.upper()} strategy only in Phase 3.\n"
                f"This strategy shows significantly better performance across metrics."
            )
    else:
        return (
            f"🟢 SINGLE STRATEGY\n\n"
            f"Strategy: {best} (score: {best_score:.3f})\n"
            f"Degradation: {best_deg:.2f}\n\n"
            f"Recommendation: Proceed with {best.upper()} strategy in Phase 3."
        )


def create_heatmap_visualizations(
    results: List[Dict], strategy: str, output_dir: pathlib.Path
):
    """Create heatmap visualizations for pitch, modality, and degradation.

    Args:
        results: List of results for one strategy
        strategy: Strategy name
        output_dir: Directory to save figures
    """
    # Extract unique alpha values (sorted)
    alphas_pitch = sorted(set(r["alpha_pitch"] for r in results))
    alphas_modality = sorted(set(r["alpha_modality"] for r in results))

    # Create grids for heatmaps
    n_pitch = len(alphas_pitch)
    n_modality = len(alphas_modality)

    pitch_grid = np.full((n_modality, n_pitch), np.nan)
    modality_grid = np.full((n_modality, n_pitch), np.nan)
    degradation_grid = np.full((n_modality, n_pitch), np.nan)

    # Fill grids
    for r in results:
        i_pitch = alphas_pitch.index(r["alpha_pitch"])
        i_modality = alphas_modality.index(r["alpha_modality"])

        if r["valid_samples"] > 0:
            pitch_grid[i_modality, i_pitch] = r["pitch_control"]["mean"]
            modality_grid[i_modality, i_pitch] = r["modality_control"][
                "major_percentage"
            ]
            degradation_grid[i_modality, i_pitch] = r["degradation"][
                "total_degradation"
            ]["mean"]

    # Create figure with 3 heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Dual-Steering Landscape: {strategy.upper()} Strategy",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Pitch control heatmap
    ax = axes[0]
    im1 = ax.imshow(
        pitch_grid, cmap="RdYlGn", aspect="auto", origin="lower", vmin=50, vmax=90
    )
    ax.set_xlabel("α_pitch", fontsize=12)
    ax.set_ylabel("α_modality", fontsize=12)
    ax.set_title("Mean Pitch (MIDI)", fontsize=13)
    ax.set_xticks(range(n_pitch))
    ax.set_xticklabels([f"{a:+.1f}" for a in alphas_pitch], rotation=45)
    ax.set_yticks(range(n_modality))
    ax.set_yticklabels([f"{a:+.1f}" for a in alphas_modality])
    plt.colorbar(im1, ax=ax, label="MIDI Pitch")

    # Add text annotations
    for i in range(n_modality):
        for j in range(n_pitch):
            if not np.isnan(pitch_grid[i, j]):
                text = ax.text(
                    j,
                    i,
                    f"{pitch_grid[i, j]:.0f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    # 2. Modality control heatmap
    ax = axes[1]
    im2 = ax.imshow(
        modality_grid, cmap="PuOr", aspect="auto", origin="lower", vmin=0, vmax=100
    )
    ax.set_xlabel("α_pitch", fontsize=12)
    ax.set_ylabel("α_modality", fontsize=12)
    ax.set_title("Major Percentage (%)", fontsize=13)
    ax.set_xticks(range(n_pitch))
    ax.set_xticklabels([f"{a:+.1f}" for a in alphas_pitch], rotation=45)
    ax.set_yticks(range(n_modality))
    ax.set_yticklabels([f"{a:+.1f}" for a in alphas_modality])
    plt.colorbar(im2, ax=ax, label="% Major")

    # Add text annotations
    for i in range(n_modality):
        for j in range(n_pitch):
            if not np.isnan(modality_grid[i, j]):
                text = ax.text(
                    j,
                    i,
                    f"{modality_grid[i, j]:.0f}%",
                    ha="center",
                    va="center",
                    color="white" if 30 < modality_grid[i, j] < 70 else "black",
                    fontsize=8,
                )

    # 3. Quality degradation heatmap
    ax = axes[2]
    im3 = ax.imshow(
        degradation_grid, cmap="YlOrRd", aspect="auto", origin="lower", vmin=0, vmax=10
    )
    ax.set_xlabel("α_pitch", fontsize=12)
    ax.set_ylabel("α_modality", fontsize=12)
    ax.set_title("Quality Degradation (Lower is Better)", fontsize=13)
    ax.set_xticks(range(n_pitch))
    ax.set_xticklabels([f"{a:+.1f}" for a in alphas_pitch], rotation=45)
    ax.set_yticks(range(n_modality))
    ax.set_yticklabels([f"{a:+.1f}" for a in alphas_modality])
    plt.colorbar(im3, ax=ax, label="Degradation")

    # Add text annotations
    for i in range(n_modality):
        for j in range(n_pitch):
            if not np.isnan(degradation_grid[i, j]):
                text = ax.text(
                    j,
                    i,
                    f"{degradation_grid[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white" if degradation_grid[i, j] > 5 else "black",
                    fontsize=8,
                )

    plt.tight_layout()
    fig_path = output_dir / f"heatmaps_{strategy}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved heatmaps: {fig_path}")
    plt.close()


def create_pareto_frontier(
    results: List[Dict], strategy: str, output_dir: pathlib.Path
):
    """Create Pareto frontier plot showing trade-offs.

    Args:
        results: List of results for one strategy
        strategy: Strategy name
        output_dir: Directory to save figures
    """
    # Extract metrics
    valid_results = [r for r in results if r["valid_samples"] > 0]

    pitch_control = [abs(r["pitch_control"]["mean"] - 70) for r in valid_results]
    modality_control = [
        abs(r["modality_control"]["major_percentage"] - 50) for r in valid_results
    ]
    degradation = [r["degradation"]["total_degradation"]["mean"] for r in valid_results]
    alpha_pitch = [r["alpha_pitch"] for r in valid_results]
    alpha_modality = [r["alpha_modality"] for r in valid_results]

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        pitch_control,
        modality_control,
        c=degradation,
        s=100,
        cmap="RdYlGn_r",
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
        vmin=0,
        vmax=10,
    )

    ax.set_xlabel("Pitch Control Strength\n(distance from baseline)", fontsize=12)
    ax.set_ylabel("Modality Control Strength\n(distance from 50% major)", fontsize=12)
    ax.set_title(
        f"Control vs Quality Trade-off: {strategy.upper()}\n"
        "Color = Quality Degradation (darker = better)",
        fontsize=14,
        fontweight="bold",
    )

    cbar = plt.colorbar(scatter, ax=ax, label="Quality Degradation")

    # Annotate some key points
    for i, (pc, mc, ap, am, deg) in enumerate(
        zip(pitch_control, modality_control, alpha_pitch, alpha_modality, degradation)
    ):
        # Annotate baseline and extreme points
        if (ap == 0 and am == 0) or abs(ap) >= 1.5 or abs(am) >= 1.5:
            ax.annotate(
                f"({ap:+.1f},{am:+.1f})",
                (pc, mc),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=7,
                alpha=0.7,
            )

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = output_dir / f"pareto_frontier_{strategy}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved Pareto frontier: {fig_path}")
    plt.close()


def create_interaction_analysis(
    results: List[Dict], strategy: str, output_dir: pathlib.Path
):
    """Analyze interaction between pitch and modality steering.

    Args:
        results: List of results for one strategy
        strategy: Strategy name
        output_dir: Directory to save figures
    """
    valid_results = [r for r in results if r["valid_samples"] > 0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Concept Interaction Analysis: {strategy.upper()}",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Pitch vs alpha_pitch (grouped by alpha_modality)
    ax = axes[0, 0]
    modality_groups = {}
    for r in valid_results:
        am = r["alpha_modality"]
        if am not in modality_groups:
            modality_groups[am] = {"pitch": [], "alpha_pitch": []}
        modality_groups[am]["pitch"].append(r["pitch_control"]["mean"])
        modality_groups[am]["alpha_pitch"].append(r["alpha_pitch"])

    colors = plt.cm.coolwarm(np.linspace(0, 1, len(modality_groups)))
    for i, (am, data) in enumerate(sorted(modality_groups.items())):
        ax.plot(
            data["alpha_pitch"],
            data["pitch"],
            marker="o",
            label=f"α_mod={am:+.1f}",
            color=colors[i],
            alpha=0.7,
        )

    ax.set_xlabel("α_pitch", fontsize=11)
    ax.set_ylabel("Mean Pitch (MIDI)", fontsize=11)
    ax.set_title("Pitch Control: Effect of α_modality", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Modality vs alpha_modality (grouped by alpha_pitch)
    ax = axes[0, 1]
    pitch_groups = {}
    for r in valid_results:
        ap = r["alpha_pitch"]
        if ap not in pitch_groups:
            pitch_groups[ap] = {"modality": [], "alpha_modality": []}
        pitch_groups[ap]["modality"].append(r["modality_control"]["major_percentage"])
        pitch_groups[ap]["alpha_modality"].append(r["alpha_modality"])

    colors = plt.cm.viridis(np.linspace(0, 1, len(pitch_groups)))
    for i, (ap, data) in enumerate(sorted(pitch_groups.items())):
        ax.plot(
            data["alpha_modality"],
            data["modality"],
            marker="s",
            label=f"α_pitch={ap:+.1f}",
            color=colors[i],
            alpha=0.7,
        )

    ax.set_xlabel("α_modality", fontsize=11)
    ax.set_ylabel("Major Percentage (%)", fontsize=11)
    ax.set_title("Modality Control: Effect of α_pitch", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Degradation vs |α_pitch|
    ax = axes[1, 0]
    abs_alpha_pitch = [abs(r["alpha_pitch"]) for r in valid_results]
    degradation = [r["degradation"]["total_degradation"]["mean"] for r in valid_results]
    ax.scatter(abs_alpha_pitch, degradation, alpha=0.6, s=50)
    ax.set_xlabel("|α_pitch|", fontsize=11)
    ax.set_ylabel("Total Degradation", fontsize=11)
    ax.set_title("Quality vs Pitch Steering Strength", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(abs_alpha_pitch, degradation, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(abs_alpha_pitch), max(abs_alpha_pitch), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, label="Trend")
    ax.legend()

    # 4. Degradation vs |α_modality|
    ax = axes[1, 1]
    abs_alpha_modality = [abs(r["alpha_modality"]) for r in valid_results]
    ax.scatter(abs_alpha_modality, degradation, alpha=0.6, s=50, color="purple")
    ax.set_xlabel("|α_modality|", fontsize=11)
    ax.set_ylabel("Total Degradation", fontsize=11)
    ax.set_title("Quality vs Modality Steering Strength", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(abs_alpha_modality, degradation, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(abs_alpha_modality), max(abs_alpha_modality), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, label="Trend")
    ax.legend()

    plt.tight_layout()
    fig_path = output_dir / f"interaction_analysis_{strategy}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved interaction analysis: {fig_path}")
    plt.close()


def create_baseline_comparison(results: List[Dict], output_dir: pathlib.Path):
    """Compare all results against baseline (α=0, α=0).

    Args:
        results: All results across strategies
        output_dir: Directory to save figures
    """
    # Find baseline for each strategy
    baselines = {}
    for r in results:
        if r["alpha_pitch"] == 0 and r["alpha_modality"] == 0:
            strategy = r["strategy"]
            baselines[strategy] = {
                "pitch": r["pitch_control"]["mean"],
                "major_pct": r["modality_control"]["major_percentage"],
                "degradation": r["degradation"]["total_degradation"]["mean"],
            }

    # Calculate deltas from baseline
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Performance Relative to Baseline (α_pitch=0, α_modality=0)",
        fontsize=16,
        fontweight="bold",
    )

    strategies = sorted(set(r["strategy"] for r in results))
    colors_map = {"direct": "#2E86AB", "gram_schmidt": "#A23B72"}

    for strategy in strategies:
        strategy_results = [r for r in results if r["strategy"] == strategy]
        valid_results = [r for r in strategy_results if r["valid_samples"] > 0]
        baseline = baselines.get(strategy, {})

        if not baseline:
            continue

        # 1. Pitch delta vs alpha_pitch
        ax = axes[0, 0]
        alpha_pitch = [r["alpha_pitch"] for r in valid_results]
        pitch_delta = [
            r["pitch_control"]["mean"] - baseline["pitch"] for r in valid_results
        ]
        ax.scatter(
            alpha_pitch,
            pitch_delta,
            label=strategy,
            alpha=0.6,
            s=50,
            color=colors_map.get(strategy, "gray"),
        )

        # 2. Modality delta vs alpha_modality
        ax = axes[0, 1]
        alpha_modality = [r["alpha_modality"] for r in valid_results]
        modality_delta = [
            r["modality_control"]["major_percentage"] - baseline["major_pct"]
            for r in valid_results
        ]
        ax.scatter(
            alpha_modality,
            modality_delta,
            label=strategy,
            alpha=0.6,
            s=50,
            color=colors_map.get(strategy, "gray"),
        )

        # 3. Degradation delta vs alpha_pitch
        ax = axes[1, 0]
        degradation_delta = [
            r["degradation"]["total_degradation"]["mean"] - baseline["degradation"]
            for r in valid_results
        ]
        ax.scatter(
            alpha_pitch,
            degradation_delta,
            label=strategy,
            alpha=0.6,
            s=50,
            color=colors_map.get(strategy, "gray"),
        )

        # 4. Degradation delta vs alpha_modality
        ax = axes[1, 1]
        ax.scatter(
            alpha_modality,
            degradation_delta,
            label=strategy,
            alpha=0.6,
            s=50,
            color=colors_map.get(strategy, "gray"),
        )

    # Configure axes
    axes[0, 0].set_xlabel("α_pitch", fontsize=11)
    axes[0, 0].set_ylabel("Δ Mean Pitch (MIDI)", fontsize=11)
    axes[0, 0].set_title("Pitch Change vs α_pitch", fontsize=12)
    axes[0, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 0].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("α_modality", fontsize=11)
    axes[0, 1].set_ylabel("Δ Major % (percentage points)", fontsize=11)
    axes[0, 1].set_title("Major % Change vs α_modality", fontsize=12)
    axes[0, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("α_pitch", fontsize=11)
    axes[1, 0].set_ylabel("Δ Degradation", fontsize=11)
    axes[1, 0].set_title("Quality Change vs α_pitch", fontsize=12)
    axes[1, 0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 0].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("α_modality", fontsize=11)
    axes[1, 1].set_ylabel("Δ Degradation", fontsize=11)
    axes[1, 1].set_title("Quality Change vs α_modality", fontsize=12)
    axes[1, 1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = output_dir / "baseline_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved baseline comparison: {fig_path}")
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze Phase 2 strategy results")
    parser.add_argument(
        "--results_file",
        type=pathlib.Path,
        required=True,
        help="Path to phase2_results.json",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/dual_steering/outputs/phase2_analysis"
        ),
        help="Output directory for analysis",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load results
    logging.info(f"Loading results from: {args.results_file}")
    data = load_results(args.results_file)
    results = data["results"]
    config_info = data["config"]

    logging.info(
        f"Loaded {len(results)} results from {config_info['total_configs']} configs"
    )

    # Group by strategy
    grouped_results = group_by_strategy(results)
    logging.info(f"Strategies tested: {list(grouped_results.keys())}")

    # Compare strategies
    comparison = compare_strategies(grouped_results)

    # Visualize strategy comparison
    logging.info("\nGenerating strategy comparison visualizations...")
    visualize_comparison(grouped_results, comparison, args.output_dir)

    # Create extended visualizations for each strategy
    logging.info("\nGenerating extended visualizations...")
    for strategy, strategy_results in grouped_results.items():
        logging.info(f"  Creating heatmaps for {strategy}...")
        create_heatmap_visualizations(strategy_results, strategy, args.output_dir)

        logging.info(f"  Creating Pareto frontier for {strategy}...")
        create_pareto_frontier(strategy_results, strategy, args.output_dir)

        logging.info(f"  Creating interaction analysis for {strategy}...")
        create_interaction_analysis(strategy_results, strategy, args.output_dir)

    # Create baseline comparison across all results
    logging.info("\nGenerating baseline comparison...")
    create_baseline_comparison(results, args.output_dir)

    # Print detailed comparison
    print("\n" + "=" * 80)
    print("PHASE 2 STRATEGY COMPARISON")
    print("=" * 80)

    for strategy in sorted(grouped_results.keys()):
        comp = comparison[strategy]
        print(f"\n{strategy.upper()}:")
        print(f"  Overall Score:           {comp['overall_score']:.4f}")
        print(
            f"  Success Rate:            {comp['stability']['mean_success_rate']:.2%}"
        )
        print(
            f"  Pitch Control Strength:  {comp['pitch_control']['control_strength']:.4f}"
        )
        print(
            f"  Modality Control Strength: {comp['modality_control']['control_strength']:.4f}"
        )
        print(f"  Pitch Correlation:       {comp['pitch_control']['correlation']:+.4f}")
        print(
            f"  Modality Correlation:    {comp['modality_control']['correlation']:+.4f}"
        )
        print(f"  Stability Score:         {comp['stability']['stability_score']:.4f}")

    # Determine recommendation
    recommendation = determine_recommendation(comparison)

    print("\n" + "=" * 80)
    print("RECOMMENDATION FOR PHASE 3")
    print("=" * 80)
    print(f"\n{recommendation}\n")
    print("=" * 80)

    # Save analysis
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis_file = args.output_dir / "strategy_analysis.json"

    with open(analysis_file, "w") as f:
        json.dump(
            {
                "comparison": comparison,
                "recommendation": recommendation,
                "config": config_info,
            },
            f,
            indent=2,
        )

    logging.info(f"\nSaved analysis to: {analysis_file}")


if __name__ == "__main__":
    main()
