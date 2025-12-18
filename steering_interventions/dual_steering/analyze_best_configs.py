#!/usr/bin/env python3
"""Analyze best configurations for different steering scenarios.

Identifies optimal (α_pitch, α_duration) pairs for various use cases:
- Low pitch + long duration
- High pitch + short duration
- Low pitch + short duration
- High pitch + long duration
- Pitch-only steering (duration neutral)
- Duration-only steering (pitch neutral)
- Extreme steering (max range)
- Quality-focused (min degradation)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def load_results(
    filepath: str = "steering_interventions/dual_steering_dp/outputs/validation_grid_orthogonal/phase3_results.json",
) -> Tuple[Dict, List[Dict]]:
    """Load phase3 results."""
    with open(filepath) as f:
        data = json.load(f)
    return data["config"], data["results"]


def get_baseline_values(results: List[Dict], strategy: str) -> Tuple[float, float]:
    """Get baseline pitch and duration for α=0,0."""
    baseline = [
        r
        for r in results
        if r["strategy"] == strategy
        and r["alpha_pitch"] == 0.0
        and r["alpha_duration"] == 0.0
        and r["valid_samples"] > 0
    ]

    if baseline:
        return (
            baseline[0]["pitch_control"]["mean"],
            baseline[0]["duration_control"]["mean"],
        )
    return 60.0, 10.0  # Fallback


def classify_config(r: Dict, baseline_pitch: float, baseline_duration: float) -> str:
    """Classify config based on steering direction."""
    pitch = r["pitch_control"]["mean"]
    duration = r["duration_control"]["mean"]

    pitch_change = pitch - baseline_pitch
    duration_change = duration - baseline_duration

    # Thresholds for classification
    PITCH_THRESHOLD = 5  # semitones
    DURATION_THRESHOLD = 3  # ticks

    if (
        abs(pitch_change) < PITCH_THRESHOLD
        and abs(duration_change) < DURATION_THRESHOLD
    ):
        return "baseline"
    elif pitch_change < -PITCH_THRESHOLD and duration_change > DURATION_THRESHOLD:
        return "low_pitch_long_duration"
    elif pitch_change > PITCH_THRESHOLD and duration_change < -DURATION_THRESHOLD:
        return "high_pitch_short_duration"
    elif pitch_change < -PITCH_THRESHOLD and duration_change < -DURATION_THRESHOLD:
        return "low_pitch_short_duration"
    elif pitch_change > PITCH_THRESHOLD and duration_change > DURATION_THRESHOLD:
        return "high_pitch_long_duration"
    elif (
        abs(pitch_change) > PITCH_THRESHOLD
        and abs(duration_change) < DURATION_THRESHOLD
    ):
        return "pitch_only"
    elif (
        abs(pitch_change) < PITCH_THRESHOLD
        and abs(duration_change) > DURATION_THRESHOLD
    ):
        return "duration_only"
    else:
        return "mixed"


def calculate_steering_effectiveness(
    r: Dict, baseline_pitch: float, baseline_duration: float
) -> float:
    """Calculate combined steering effectiveness score."""
    pitch_change = abs(r["pitch_control"]["mean"] - baseline_pitch)
    duration_change = abs(r["duration_control"]["mean"] - baseline_duration)

    # Normalize changes (pitch in semitones, duration in ticks)
    pitch_score = pitch_change / 50.0  # Max expected ~50 semitones
    duration_score = duration_change / 20.0  # Max expected ~20 ticks

    return pitch_score + duration_score


def find_best_configs_by_scenario(
    results: List[Dict], strategy: str, top_n: int = 5
) -> Dict:
    """Find best configs for each steering scenario."""

    # Filter for strategy and valid samples
    valid = [
        r
        for r in results
        if r["strategy"] == strategy
        and r["valid_samples"] > 0
        and not np.isnan(r["degradation"]["total_degradation"]["mean"])
    ]

    baseline_pitch, baseline_duration = get_baseline_values(results, strategy)

    # Classify all configs
    scenarios = {}
    for r in valid:
        category = classify_config(r, baseline_pitch, baseline_duration)
        if category not in scenarios:
            scenarios[category] = []
        scenarios[category].append(r)

    # Find best configs per scenario
    best_configs = {}

    for scenario, configs in scenarios.items():
        if scenario == "baseline":
            continue

        # Score = low degradation + high steering effectiveness
        scored = []
        for r in configs:
            degradation = r["degradation"]["total_degradation"]["mean"]
            effectiveness = calculate_steering_effectiveness(
                r, baseline_pitch, baseline_duration
            )

            # Combined score (lower is better for degradation, higher for effectiveness)
            # Normalize: degradation [0-30] -> [0-1], effectiveness [0-2] -> [0-1]
            score = (degradation / 30.0) - effectiveness

            scored.append((score, r))

        # Sort by score (lower is better)
        scored.sort(key=lambda x: x[0])
        best_configs[scenario] = [r for _, r in scored[:top_n]]

    return best_configs, baseline_pitch, baseline_duration


def print_scenario_analysis(
    best_configs: Dict, baseline_pitch: float, baseline_duration: float
):
    """Print detailed analysis of best configs per scenario."""

    scenario_names = {
        "low_pitch_long_duration": "🔽📏 LOW Pitch + LONG Duration",
        "high_pitch_short_duration": "🔼⚡ HIGH Pitch + SHORT Duration",
        "low_pitch_short_duration": "🔽⚡ LOW Pitch + SHORT Duration",
        "high_pitch_long_duration": "🔼📏 HIGH Pitch + LONG Duration",
        "pitch_only": "🎵 Pitch Control Only (duration neutral)",
        "duration_only": "⏱️ Duration Control Only (pitch neutral)",
        "mixed": "🔀 Mixed Steering",
    }

    print("\n" + "=" * 80)
    print("BEST CONFIGURATIONS BY STEERING SCENARIO")
    print("=" * 80)
    print(
        f"\nBaseline (α=0,0): Pitch={baseline_pitch:.1f} MIDI, Duration={baseline_duration:.1f} ticks"
    )

    for scenario, name in scenario_names.items():
        if scenario not in best_configs:
            continue

        configs = best_configs[scenario]
        if not configs:
            continue

        print(f"\n{name}")
        print("-" * 80)

        for i, r in enumerate(configs, 1):
            pitch = r["pitch_control"]["mean"]
            duration = r["duration_control"]["mean"]
            degradation = r["degradation"]["total_degradation"]["mean"]

            pitch_change = pitch - baseline_pitch
            duration_change = duration - baseline_duration

            print(
                f"\n  {i}. α_pitch={r['alpha_pitch']:+.2f}, α_duration={r['alpha_duration']:+.2f}"
            )
            print(f"     Pitch: {pitch:.1f} MIDI ({pitch_change:+.1f} from baseline)")
            print(
                f"     Duration: {duration:.1f} ticks ({duration_change:+.1f} from baseline)"
            )
            print(f"     Degradation: {degradation:.2f}")
            print(
                f"     Quality: entropy={r['quality_metrics']['pitch_class_entropy']['mean']:.2f}, "
                f"scale={r['quality_metrics']['scale_consistency']['mean']:.1f}%, "
                f"groove={r['quality_metrics']['groove_consistency']['mean']:.1f}%"
            )


def find_pareto_optimal(results: List[Dict], strategy: str) -> List[Dict]:
    """Find Pareto-optimal configs (best tradeoff between steering and quality)."""

    valid = [
        r
        for r in results
        if r["strategy"] == strategy
        and r["valid_samples"] > 0
        and not np.isnan(r["degradation"]["total_degradation"]["mean"])
    ]

    baseline_pitch, baseline_duration = get_baseline_values(results, strategy)

    # Calculate steering magnitude and degradation for each config
    configs_with_scores = []
    for r in valid:
        steering = calculate_steering_effectiveness(
            r, baseline_pitch, baseline_duration
        )
        degradation = r["degradation"]["total_degradation"]["mean"]
        configs_with_scores.append((steering, degradation, r))

    # Find Pareto frontier
    pareto = []
    for steering1, deg1, r1 in configs_with_scores:
        is_dominated = False
        for steering2, deg2, r2 in configs_with_scores:
            # r2 dominates r1 if it has both higher steering AND lower degradation
            if steering2 > steering1 and deg2 < deg1:
                is_dominated = True
                break
        if not is_dominated:
            pareto.append((steering1, deg1, r1))

    # Sort by steering effectiveness
    pareto.sort(key=lambda x: x[0], reverse=True)

    return [(r, steering, deg) for steering, deg, r in pareto]


def print_pareto_analysis(
    pareto_configs: List[Tuple], baseline_pitch: float, baseline_duration: float
):
    """Print Pareto-optimal configurations."""

    print("\n" + "=" * 80)
    print("PARETO-OPTIMAL CONFIGURATIONS (Best Steering vs Quality Tradeoff)")
    print("=" * 80)
    print("\nThese configs are not dominated by any other config")
    print("(no other config has both higher steering AND lower degradation)")

    print(f"\nTop 10 Pareto-optimal configs:")
    print("-" * 80)

    for i, (r, steering, degradation) in enumerate(pareto_configs[:10], 1):
        pitch = r["pitch_control"]["mean"]
        duration = r["duration_control"]["mean"]

        pitch_change = pitch - baseline_pitch
        duration_change = duration - baseline_duration

        print(
            f"\n{i:2d}. α_pitch={r['alpha_pitch']:+.2f}, α_duration={r['alpha_duration']:+.2f}"
        )
        print(f"    Steering: {steering:.3f} | Degradation: {degradation:.2f}")
        print(
            f"    Pitch: {pitch:.1f} ({pitch_change:+.1f}) | Duration: {duration:.1f} ({duration_change:+.1f})"
        )


def create_scenario_visualization(
    best_configs: Dict,
    baseline_pitch: float,
    baseline_duration: float,
    output_path: str = "steering_scenarios.png",
):
    """Create visualization of best configs for each scenario."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    scenarios = [
        ("low_pitch_long_duration", "🔽📏 Low Pitch + Long Duration", axes[0]),
        ("high_pitch_short_duration", "🔼⚡ High Pitch + Short Duration", axes[1]),
        ("low_pitch_short_duration", "🔽⚡ Low Pitch + Short Duration", axes[2]),
        ("high_pitch_long_duration", "🔼📏 High Pitch + Long Duration", axes[3]),
    ]

    for scenario_key, title, ax in scenarios:
        if scenario_key not in best_configs or not best_configs[scenario_key]:
            continue

        configs = best_configs[scenario_key][:10]  # Top 10

        pitches = [r["pitch_control"]["mean"] for r in configs]
        durations = [r["duration_control"]["mean"] for r in configs]
        degradations = [r["degradation"]["total_degradation"]["mean"] for r in configs]

        # Scatter plot with degradation as color
        scatter = ax.scatter(
            pitches,
            durations,
            c=degradations,
            s=200,
            cmap="RdYlGn_r",
            vmin=0,
            vmax=10,
            edgecolors="black",
            linewidths=1.5,
            alpha=0.8,
        )

        # Mark baseline
        ax.scatter(
            [baseline_pitch],
            [baseline_duration],
            c="blue",
            s=300,
            marker="*",
            edgecolors="black",
            linewidths=2,
            label="Baseline",
            zorder=10,
        )

        # Add alpha labels to top 3
        for i, r in enumerate(configs[:3]):
            pitch = r["pitch_control"]["mean"]
            duration = r["duration_control"]["mean"]
            label = f"({r['alpha_pitch']:+.1f}, {r['alpha_duration']:+.1f})"
            ax.annotate(
                label,
                (pitch, duration),
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax.set_xlabel("Pitch (MIDI)", fontsize=12)
        ax.set_ylabel("Duration (ticks)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add colorbar
        plt.colorbar(scatter, ax=ax, label="Degradation")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Saved scenario visualization to: {output_path}")


def create_heatmaps(
    results: List[Dict],
    strategy: str,
    baseline_pitch: float,
    baseline_duration: float,
    output_path: str = "steering_heatmaps.png",
):
    """Create comprehensive heatmaps."""

    valid = [r for r in results if r["strategy"] == strategy and r["valid_samples"] > 0]

    alphas_pitch = sorted(list(set(r["alpha_pitch"] for r in valid)))
    alphas_duration = sorted(list(set(r["alpha_duration"] for r in valid)))

    # Create matrices
    pitch_matrix = np.full((len(alphas_duration), len(alphas_pitch)), np.nan)
    duration_matrix = np.full((len(alphas_duration), len(alphas_pitch)), np.nan)
    quality_matrix = np.full((len(alphas_duration), len(alphas_pitch)), np.nan)
    steering_matrix = np.full((len(alphas_duration), len(alphas_pitch)), np.nan)

    for r in valid:
        i = alphas_duration.index(r["alpha_duration"])
        j = alphas_pitch.index(r["alpha_pitch"])

        pitch_matrix[i, j] = r["pitch_control"]["mean"]
        duration_matrix[i, j] = r["duration_control"]["mean"]

        if not np.isnan(r["degradation"]["total_degradation"]["mean"]):
            quality_matrix[i, j] = r["degradation"]["total_degradation"]["mean"]

        # Combined steering effectiveness
        steering_eff = calculate_steering_effectiveness(
            r, baseline_pitch, baseline_duration
        )
        steering_matrix[i, j] = steering_eff

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    # Pitch control heatmap
    im1 = axes[0, 0].imshow(pitch_matrix, cmap="RdYlBu_r", aspect="auto")
    axes[0, 0].set_title("Pitch Control (MIDI note)", fontsize=14, fontweight="bold")
    axes[0, 0].set_xlabel("Alpha Pitch", fontsize=12)
    axes[0, 0].set_ylabel("Alpha Duration", fontsize=12)
    axes[0, 0].set_xticks(range(len(alphas_pitch)))
    axes[0, 0].set_xticklabels(
        [f"{a:.2f}" for a in alphas_pitch], rotation=45, ha="right"
    )
    axes[0, 0].set_yticks(range(len(alphas_duration)))
    axes[0, 0].set_yticklabels([f"{a:.2f}" for a in alphas_duration])
    plt.colorbar(im1, ax=axes[0, 0], label="Pitch (MIDI)")

    # Duration control heatmap
    im2 = axes[0, 1].imshow(duration_matrix, cmap="viridis", aspect="auto")
    axes[0, 1].set_title("Duration Control (ticks)", fontsize=14, fontweight="bold")
    axes[0, 1].set_xlabel("Alpha Pitch", fontsize=12)
    axes[0, 1].set_ylabel("Alpha Duration", fontsize=12)
    axes[0, 1].set_xticks(range(len(alphas_pitch)))
    axes[0, 1].set_xticklabels(
        [f"{a:.2f}" for a in alphas_pitch], rotation=45, ha="right"
    )
    axes[0, 1].set_yticks(range(len(alphas_duration)))
    axes[0, 1].set_yticklabels([f"{a:.2f}" for a in alphas_duration])
    plt.colorbar(im2, ax=axes[0, 1], label="Duration (ticks)")

    # Quality degradation heatmap
    im3 = axes[1, 0].imshow(
        quality_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=10
    )
    axes[1, 0].set_title("Quality Degradation", fontsize=14, fontweight="bold")
    axes[1, 0].set_xlabel("Alpha Pitch", fontsize=12)
    axes[1, 0].set_ylabel("Alpha Duration", fontsize=12)
    axes[1, 0].set_xticks(range(len(alphas_pitch)))
    axes[1, 0].set_xticklabels(
        [f"{a:.2f}" for a in alphas_pitch], rotation=45, ha="right"
    )
    axes[1, 0].set_yticks(range(len(alphas_duration)))
    axes[1, 0].set_yticklabels([f"{a:.2f}" for a in alphas_duration])
    plt.colorbar(im3, ax=axes[1, 0], label="Degradation")

    # Steering effectiveness heatmap
    im4 = axes[1, 1].imshow(steering_matrix, cmap="plasma", aspect="auto")
    axes[1, 1].set_title(
        "Combined Steering Effectiveness", fontsize=14, fontweight="bold"
    )
    axes[1, 1].set_xlabel("Alpha Pitch", fontsize=12)
    axes[1, 1].set_ylabel("Alpha Duration", fontsize=12)
    axes[1, 1].set_xticks(range(len(alphas_pitch)))
    axes[1, 1].set_xticklabels(
        [f"{a:.2f}" for a in alphas_pitch], rotation=45, ha="right"
    )
    axes[1, 1].set_yticks(range(len(alphas_duration)))
    axes[1, 1].set_yticklabels([f"{a:.2f}" for a in alphas_duration])
    plt.colorbar(im4, ax=axes[1, 1], label="Effectiveness")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"📊 Saved comprehensive heatmaps to: {output_path}")


def export_best_configs_json(
    best_configs: Dict,
    pareto_configs: List,
    baseline_pitch: float,
    baseline_duration: float,
    output_path: str = "best_configs_summary.json",
):
    """Export best configs to JSON for easy reference."""

    export_data = {
        "baseline": {
            "pitch": baseline_pitch,
            "duration": baseline_duration,
        },
        "scenarios": {},
        "pareto_optimal_top10": [],
    }

    # Export scenario-specific best configs
    for scenario, configs in best_configs.items():
        export_data["scenarios"][scenario] = []
        for r in configs:
            export_data["scenarios"][scenario].append(
                {
                    "alpha_pitch": r["alpha_pitch"],
                    "alpha_duration": r["alpha_duration"],
                    "pitch": r["pitch_control"]["mean"],
                    "duration": r["duration_control"]["mean"],
                    "degradation": r["degradation"]["total_degradation"]["mean"],
                    "quality_metrics": {
                        "pitch_class_entropy": r["quality_metrics"][
                            "pitch_class_entropy"
                        ]["mean"],
                        "scale_consistency": r["quality_metrics"]["scale_consistency"][
                            "mean"
                        ],
                        "groove_consistency": r["quality_metrics"][
                            "groove_consistency"
                        ]["mean"],
                    },
                }
            )

    # Export Pareto optimal configs
    for r, steering, degradation in pareto_configs[:10]:
        export_data["pareto_optimal_top10"].append(
            {
                "alpha_pitch": r["alpha_pitch"],
                "alpha_duration": r["alpha_duration"],
                "pitch": r["pitch_control"]["mean"],
                "duration": r["duration_control"]["mean"],
                "steering_effectiveness": steering,
                "degradation": degradation,
            }
        )

    with open(output_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"📄 Exported best configs to: {output_path}")


def main():
    """Main analysis pipeline."""

    print("=" * 80)
    print("DUAL STEERING BEST CONFIGURATIONS ANALYSIS")
    print("=" * 80)

    # Load results
    config, results = load_results()

    # Identify best strategy
    strategies = list(set(r["strategy"] for r in results if r["valid_samples"] > 0))

    print(f"\nStrategies found: {strategies}")

    # Analyze each strategy
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"ANALYZING STRATEGY: {strategy.upper()}")
        print(f"{'='*80}")

        # Find best configs by scenario
        best_configs, baseline_pitch, baseline_duration = find_best_configs_by_scenario(
            results, strategy, top_n=5
        )

        # Print scenario analysis
        print_scenario_analysis(best_configs, baseline_pitch, baseline_duration)

        # Find Pareto-optimal configs
        pareto_configs = find_pareto_optimal(results, strategy)
        print_pareto_analysis(pareto_configs, baseline_pitch, baseline_duration)

        # Create visualizations
        strategy_clean = strategy.replace("_", "-")
        create_scenario_visualization(
            best_configs,
            baseline_pitch,
            baseline_duration,
            output_path=f"steering_scenarios_{strategy_clean}.png",
        )

        create_heatmaps(
            results,
            strategy,
            baseline_pitch,
            baseline_duration,
            output_path=f"steering_heatmaps_{strategy_clean}.png",
        )

        # Export to JSON
        export_best_configs_json(
            best_configs,
            pareto_configs,
            baseline_pitch,
            baseline_duration,
            output_path=f"best_configs_{strategy_clean}.json",
        )

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - steering_scenarios_*.png - Scenario-specific best configs")
    print("  - steering_heatmaps_*.png - Comprehensive heatmaps")
    print("  - best_configs_*.json - Exportable config summaries")
    print("=" * 80)


if __name__ == "__main__":
    main()
