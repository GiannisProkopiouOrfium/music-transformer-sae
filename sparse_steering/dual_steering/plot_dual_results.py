#!/usr/bin/env python3
"""Plotting suite for dual-SAS steering results.

Generates publication-quality figures from unconditioned and conditioned
experiment JSON outputs.

Figures:
  fig1  — Pitch × Duration heatmaps per strategy (unconditioned)
  fig2  — Degradation heatmaps per strategy (unconditioned)
  fig3  — Strategy comparison: Pareto frontier (control vs quality)
  fig4  — Conditioned: success rate summary (4 scenarios × N strategies)
  fig5  — Conditioned: per-scenario pitch & duration delta bar charts

Usage:
    python sparse_steering/dual_steering/plot_dual_results.py \
        --unconditioned_json exp/sod/sparse_steering/dual_steering/unconditioned/unconditioned_results.json \
        --conditioned_json exp/sod/sparse_steering/dual_steering/conditioned/conditioned_results.json \
        --output_dir plots/dual_sas
"""

import argparse
import json
import pathlib
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not available — plotting disabled")


# ── Style config ─────────────────────────────────────────────────────────────
if HAS_MPL:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

STRATEGY_COLORS = {
    "direct": "#1f77b4",
    "cross_concept_masking": "#2ca02c",
    "gram_schmidt_pitch": "#ff7f0e",
    "gram_schmidt_duration": "#d62728",
}

STRATEGY_LABELS = {
    "direct": "Direct Addition",
    "cross_concept_masking": "Cross-Concept Masking",
    "gram_schmidt_pitch": "GS (orth. pitch)",
    "gram_schmidt_duration": "GS (orth. duration)",
}


# ════════════════════════════════════════════════════════════════════════════
# Fig 1 — Pitch × Duration response heatmaps (unconditioned)
# ════════════════════════════════════════════════════════════════════════════


def plot_response_heatmaps(data: dict, output_dir: pathlib.Path):
    """One heatmap per strategy: x=λ_pitch, y=λ_duration, colour=generated metric."""
    results = data["results"]
    strategies = sorted(set(r["strategy"] for r in results if "error" not in r))
    n_strats = len(strategies)
    if n_strats == 0:
        return

    fig, axes = plt.subplots(n_strats, 2, figsize=(12, 5 * n_strats), squeeze=False)

    for row, strat in enumerate(strategies):
        strat_results = [
            r for r in results if r["strategy"] == strat and "error" not in r
        ]
        if not strat_results:
            continue

        lp_vals = sorted(set(r["lambda_pitch"] for r in strat_results))
        ld_vals = sorted(set(r["lambda_duration"] for r in strat_results))

        # Build grids
        pitch_grid = np.full((len(ld_vals), len(lp_vals)), np.nan)
        dur_grid = np.full((len(ld_vals), len(lp_vals)), np.nan)
        lp_idx = {v: i for i, v in enumerate(lp_vals)}
        ld_idx = {v: i for i, v in enumerate(ld_vals)}

        for r in strat_results:
            i = ld_idx[r["lambda_duration"]]
            j = lp_idx[r["lambda_pitch"]]
            if r["pitch_mean"] is not None:
                pitch_grid[i, j] = r["pitch_mean"]
            if r["duration_mean"] is not None:
                dur_grid[i, j] = r["duration_mean"]

        # Pitch heatmap
        ax = axes[row, 0]
        im = ax.imshow(pitch_grid, aspect="auto", origin="lower", cmap="RdYlBu_r")
        ax.set_xticks(range(len(lp_vals)))
        ax.set_xticklabels([f"{v:+.2f}" for v in lp_vals], fontsize=8, rotation=45)
        ax.set_yticks(range(len(ld_vals)))
        ax.set_yticklabels([f"{v:+.2f}" for v in ld_vals], fontsize=8)
        ax.set_xlabel("λ pitch")
        ax.set_ylabel("λ duration")
        ax.set_title(f"{STRATEGY_LABELS.get(strat, strat)} — Mean Pitch")
        plt.colorbar(im, ax=ax, label="MIDI pitch")

        # Duration heatmap
        ax = axes[row, 1]
        im = ax.imshow(dur_grid, aspect="auto", origin="lower", cmap="RdYlBu_r")
        ax.set_xticks(range(len(lp_vals)))
        ax.set_xticklabels([f"{v:+.2f}" for v in lp_vals], fontsize=8, rotation=45)
        ax.set_yticks(range(len(ld_vals)))
        ax.set_yticklabels([f"{v:+.2f}" for v in ld_vals], fontsize=8)
        ax.set_xlabel("λ pitch")
        ax.set_ylabel("λ duration")
        ax.set_title(f"{STRATEGY_LABELS.get(strat, strat)} — Mean Duration")
        plt.colorbar(im, ax=ax, label="ticks")

    fig.suptitle(
        "Dual-SAS Unconditioned: Pitch & Duration Response", fontsize=15, y=1.01
    )
    plt.tight_layout()
    fig.savefig(output_dir / "fig1_response_heatmaps.png")
    fig.savefig(output_dir / "fig1_response_heatmaps.pdf")
    plt.close(fig)
    print("  Saved fig1_response_heatmaps")


# ════════════════════════════════════════════════════════════════════════════
# Fig 2 — Degradation heatmaps (unconditioned)
# ════════════════════════════════════════════════════════════════════════════


def plot_degradation_heatmaps(data: dict, output_dir: pathlib.Path):
    """Degradation heatmap per strategy."""
    results = data["results"]
    strategies = sorted(set(r["strategy"] for r in results if "error" not in r))
    n = len(strategies)
    if n == 0:
        return

    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)

    for idx, strat in enumerate(strategies):
        ax = axes[idx // cols, idx % cols]
        strat_results = [
            r for r in results if r["strategy"] == strat and r.get("degradation")
        ]

        lp_vals = sorted(set(r["lambda_pitch"] for r in strat_results))
        ld_vals = sorted(set(r["lambda_duration"] for r in strat_results))
        grid = np.full((len(ld_vals), len(lp_vals)), np.nan)
        lp_idx = {v: i for i, v in enumerate(lp_vals)}
        ld_idx = {v: i for i, v in enumerate(ld_vals)}

        for r in strat_results:
            grid[ld_idx[r["lambda_duration"]], lp_idx[r["lambda_pitch"]]] = r[
                "degradation"
            ]["total_degradation"]

        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="YlOrRd")
        ax.set_xticks(range(len(lp_vals)))
        ax.set_xticklabels([f"{v:+.1f}" for v in lp_vals], fontsize=8, rotation=45)
        ax.set_yticks(range(len(ld_vals)))
        ax.set_yticklabels([f"{v:+.1f}" for v in ld_vals], fontsize=8)
        ax.set_xlabel("λ pitch")
        ax.set_ylabel("λ duration")
        ax.set_title(STRATEGY_LABELS.get(strat, strat))
        plt.colorbar(im, ax=ax, label="Total Degradation")

    # Hide empty axes
    for idx in range(len(strategies), rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("Dual-SAS: Quality Degradation by Strategy", fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "fig2_degradation_heatmaps.png")
    fig.savefig(output_dir / "fig2_degradation_heatmaps.pdf")
    plt.close(fig)
    print("  Saved fig2_degradation_heatmaps")


# ════════════════════════════════════════════════════════════════════════════
# Fig 3 — Strategy comparison: Pareto (control range vs degradation)
# ════════════════════════════════════════════════════════════════════════════


def plot_pareto_frontier(data: dict, output_dir: pathlib.Path):
    """Scatter: x = pitch range × duration range, y = mean degradation."""
    results = data["results"]
    strategies = sorted(set(r["strategy"] for r in results if "error" not in r))

    fig, ax = plt.subplots(figsize=(10, 7))

    for strat in strategies:
        sr = [
            r
            for r in results
            if r["strategy"] == strat and r.get("pitch_mean") is not None
        ]
        if not sr:
            continue

        pitch_range = max(r["pitch_mean"] for r in sr) - min(
            r["pitch_mean"] for r in sr
        )
        dur_range = max(r["duration_mean"] for r in sr) - min(
            r["duration_mean"] for r in sr
        )
        control_area = pitch_range * dur_range

        degs = [
            r["degradation"]["total_degradation"] for r in sr if r.get("degradation")
        ]
        mean_deg = np.mean(degs) if degs else 0

        color = STRATEGY_COLORS.get(strat, "grey")
        label = STRATEGY_LABELS.get(strat, strat)
        ax.scatter(
            control_area,
            mean_deg,
            s=200,
            color=color,
            label=label,
            edgecolors="black",
            linewidths=1,
            zorder=5,
        )
        ax.annotate(
            f"Δpitch={pitch_range:.0f}\nΔdur={dur_range:.0f}",
            (control_area, mean_deg),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=8,
        )

    ax.set_xlabel("Control Area (pitch range × duration range)")
    ax.set_ylabel("Mean Total Degradation")
    ax.set_title("Dual-SAS Strategy Comparison: Control vs Quality")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "fig3_pareto_frontier.png")
    fig.savefig(output_dir / "fig3_pareto_frontier.pdf")
    plt.close(fig)
    print("  Saved fig3_pareto_frontier")


# ════════════════════════════════════════════════════════════════════════════
# Fig 4 — Conditioned: success rate summary
# ════════════════════════════════════════════════════════════════════════════


def plot_conditioned_success(data: dict, output_dir: pathlib.Path):
    """Grouped bar chart: success rates per scenario × strategy."""
    analysis = data.get("analysis", {})
    by_scenario = analysis.get("by_scenario", {})
    if not by_scenario:
        return

    scenarios = sorted(by_scenario.keys())
    strategies = sorted(set(s for sc in by_scenario.values() for s in sc.keys()))
    n_strats = len(strategies)

    fig, ax = plt.subplots(figsize=(max(12, 3 * len(scenarios)), 6))

    x = np.arange(len(scenarios))
    width = 0.8 / n_strats

    for i, strat in enumerate(strategies):
        rates = []
        for sc in scenarios:
            entry = by_scenario.get(sc, {}).get(strat, {})
            rates.append(entry.get("both_success_rate", 0) * 100)

        color = STRATEGY_COLORS.get(strat, f"C{i}")
        label = STRATEGY_LABELS.get(strat, strat)
        bars = ax.bar(
            x + i * width - 0.4 + width / 2,
            rates,
            width,
            label=label,
            color=color,
            edgecolor="white",
        )

        for bar, rate in zip(bars, rates):
            if rate > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{rate:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("_to_", "\n→ ").replace("_", " ") for s in scenarios], fontsize=9
    )
    ax.set_ylabel("Both-Concept Success Rate (%)")
    ax.set_title("Conditioned Dual-SAS: Success Rate by Scenario & Strategy")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    fig.savefig(output_dir / "fig4_conditioned_success.png")
    fig.savefig(output_dir / "fig4_conditioned_success.pdf")
    plt.close(fig)
    print("  Saved fig4_conditioned_success")


# ════════════════════════════════════════════════════════════════════════════
# Fig 5 — Conditioned: pitch & duration deltas per scenario
# ════════════════════════════════════════════════════════════════════════════


def plot_conditioned_deltas(data: dict, output_dir: pathlib.Path):
    """Per-scenario: mean pitch Δ and duration Δ for each strategy."""
    analysis = data.get("analysis", {})
    by_scenario = analysis.get("by_scenario", {})
    if not by_scenario:
        return

    scenarios = sorted(by_scenario.keys())
    strategies = sorted(set(s for sc in by_scenario.values() for s in sc.keys()))

    fig, axes = plt.subplots(
        len(scenarios), 2, figsize=(12, 4 * len(scenarios)), squeeze=False
    )

    for row, sc in enumerate(scenarios):
        strats_data = by_scenario.get(sc, {})

        # Pitch delta
        ax = axes[row, 0]
        for i, strat in enumerate(strategies):
            entry = strats_data.get(strat, {})
            delta = entry.get("mean_pitch_delta", 0)
            color = STRATEGY_COLORS.get(strat, f"C{i}")
            ax.bar(
                i,
                delta,
                color=color,
                edgecolor="white",
                label=STRATEGY_LABELS.get(strat, strat) if row == 0 else None,
            )
            ax.text(
                i,
                delta + (1 if delta >= 0 else -1),
                f"{delta:+.1f}",
                ha="center",
                fontsize=9,
            )

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(
            [STRATEGY_LABELS.get(s, s) for s in strategies],
            fontsize=8,
            rotation=20,
            ha="right",
        )
        ax.set_ylabel("Mean Δ Pitch")
        ax.set_title(sc.replace("_to_", " → ").replace("_", " "))

        # Duration delta
        ax = axes[row, 1]
        for i, strat in enumerate(strategies):
            entry = strats_data.get(strat, {})
            delta = entry.get("mean_duration_delta", 0)
            color = STRATEGY_COLORS.get(strat, f"C{i}")
            ax.bar(i, delta, color=color, edgecolor="white")
            ax.text(
                i,
                delta + (0.5 if delta >= 0 else -0.5),
                f"{delta:+.1f}",
                ha="center",
                fontsize=9,
            )

        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(
            [STRATEGY_LABELS.get(s, s) for s in strategies],
            fontsize=8,
            rotation=20,
            ha="right",
        )
        ax.set_ylabel("Mean Δ Duration")
        ax.set_title(sc.replace("_to_", " → ").replace("_", " "))

    if strategies:
        fig.legend(
            [
                plt.Rectangle((0, 0), 1, 1, fc=STRATEGY_COLORS.get(s, "grey"))
                for s in strategies
            ],
            [STRATEGY_LABELS.get(s, s) for s in strategies],
            loc="upper center",
            ncol=len(strategies),
            fontsize=9,
            bbox_to_anchor=(0.5, 1.02),
        )

    fig.suptitle("Conditioned Dual-SAS: Pitch & Duration Changes", fontsize=15, y=1.05)
    plt.tight_layout()
    fig.savefig(output_dir / "fig5_conditioned_deltas.png")
    fig.savefig(output_dir / "fig5_conditioned_deltas.pdf")
    plt.close(fig)
    print("  Saved fig5_conditioned_deltas")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Plot dual-SAS steering results")
    parser.add_argument("--unconditioned_json", type=pathlib.Path, default=None)
    parser.add_argument("--conditioned_json", type=pathlib.Path, default=None)
    parser.add_argument(
        "--output_dir", type=pathlib.Path, default=pathlib.Path("plots/dual_sas")
    )
    args = parser.parse_args()

    if not HAS_MPL:
        print("ERROR: matplotlib required. Install with: pip install matplotlib")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Dual-SAS — Plot Generation")
    print("=" * 60)

    if args.unconditioned_json and args.unconditioned_json.exists():
        print(f"\nLoading unconditioned results: {args.unconditioned_json}")
        with open(args.unconditioned_json) as f:
            uncond_data = json.load(f)

        print("\n[1/3] Response heatmaps")
        plot_response_heatmaps(uncond_data, args.output_dir)
        print("[2/3] Degradation heatmaps")
        plot_degradation_heatmaps(uncond_data, args.output_dir)
        print("[3/3] Pareto frontier")
        plot_pareto_frontier(uncond_data, args.output_dir)
    else:
        print("\nNo unconditioned JSON provided — skipping figs 1-3")

    if args.conditioned_json and args.conditioned_json.exists():
        print(f"\nLoading conditioned results: {args.conditioned_json}")
        with open(args.conditioned_json) as f:
            cond_data = json.load(f)

        print("\n[4/5] Conditioned success rates")
        plot_conditioned_success(cond_data, args.output_dir)
        print("[5/5] Conditioned deltas")
        plot_conditioned_deltas(cond_data, args.output_dir)
    else:
        print("\nNo conditioned JSON provided — skipping figs 4-5")

    print(f"\n✓ All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
