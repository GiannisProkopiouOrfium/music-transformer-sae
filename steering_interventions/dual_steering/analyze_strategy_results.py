#!/usr/bin/env python3
"""Analyze Phase 2 validation results to determine best composition strategy.

Compares direct vs gram_schmidt strategies across all test configurations
and provides recommendation for which strategy to use in full grid search.

Metrics compared:
1. Success rate (% of valid generations)
2. Pitch control effectiveness (mean pitch shift per alpha)
3. Modality control effectiveness (major likelihood shift)
4. Stability (consistency across samples)

Output:
- Comparison visualizations
- Statistical analysis
- Clear recommendation for Phase 3

Usage:
    python dual_steering/analyze_strategy_results.py \\
        --results_file steering_interventions/dual_steering/outputs/phase2_validation/phase2_results.json \\
        --output_dir steering_interventions/dual_steering/outputs/phase2_analysis
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
        modality_alphas[alpha].append(r["modality_control"]["mean_major_likelihood"])

    # Compute mean major likelihood for each alpha
    alpha_means = {
        alpha: float(np.mean(likelihoods))
        for alpha, likelihoods in modality_alphas.items()
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

        comparison[strategy] = {
            "pitch_control": pitch_analysis,
            "modality_control": modality_analysis,
            "stability": stability_analysis,
            # Overall score (weighted average)
            "overall_score": (
                0.35 * pitch_analysis["control_strength"]
                + 0.35 * modality_analysis["control_strength"]
                + 0.30 * stability_analysis["stability_score"]
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

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
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
    ax.set_title("Modality Alpha ↔ Major Likelihood Correlation", fontsize=13)
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

    if len(strategies) > 1:
        second = strategies[1]
        second_score = comparison[second]["overall_score"]
        score_diff = best_score - second_score

        if score_diff < 0.05:  # Very close
            return (
                f"🟡 CLOSE COMPETITION\n\n"
                f"Best: {best} (score: {best_score:.3f})\n"
                f"Runner-up: {second} (score: {second_score:.3f})\n"
                f"Difference: {score_diff:.3f}\n\n"
                f"Recommendation: Proceed with BOTH strategies in Phase 3 grid search.\n"
                f"The performance difference is too small to confidently eliminate one strategy."
            )
        else:
            return (
                f"🟢 CLEAR WINNER\n\n"
                f"Best: {best} (score: {best_score:.3f})\n"
                f"Runner-up: {second} (score: {second_score:.3f})\n"
                f"Difference: {score_diff:.3f}\n\n"
                f"Recommendation: Proceed with {best.upper()} strategy only in Phase 3.\n"
                f"This strategy shows significantly better performance across metrics."
            )
    else:
        return (
            f"🟢 SINGLE STRATEGY\n\n"
            f"Strategy: {best} (score: {best_score:.3f})\n\n"
            f"Recommendation: Proceed with {best.upper()} strategy in Phase 3."
        )


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

    # Visualize
    logging.info("\nGenerating visualizations...")
    visualize_comparison(grouped_results, comparison, args.output_dir)

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
