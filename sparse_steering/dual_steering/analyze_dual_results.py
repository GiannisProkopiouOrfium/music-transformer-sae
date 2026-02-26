#!/usr/bin/env python3
"""Comprehensive analysis of dual-SAS unconditioned grid-search results.

Computes per-strategy metrics consistent with single-concept SAS evaluation:
  - Pearson r, R², slope (λ_pitch → pitch, λ_duration → duration)
  - Cross-talk metrics (λ_pitch → duration, λ_duration → pitch)
  - Monotonicity score
  - Dual Steering Success Rate
  - Quality degradation profile

Generates publication-quality figures:
  fig1  — Response heatmaps (pitch + duration per strategy)
  fig2  — Degradation heatmaps per strategy
  fig3  — Pareto frontier (control area vs degradation)
  fig4  — Marginal response curves (pitch vs λ_pitch, duration vs λ_duration)
  fig5  — Dual Steering Success Rate bar chart
  fig6  — Cross-talk analysis (off-diagonal effects)

Usage:
    python sparse_steering/dual_steering/analyze_dual_results.py \\
        --json exp/sod/sparse_steering/dual_steering/unconditioned/unconditioned_results.json \\
        --output_dir plots/dual_sas_analysis
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Style ────────────────────────────────────────────────────────────────────
STRATEGY_COLORS = {
    "direct": "#1f77b4",
    "cross_concept_masking": "#2ca02c",
    "gram_schmidt_pitch": "#ff7f0e",
    "gram_schmidt_duration": "#d62728",
    "cross_concept_sas": "#9467bd",
    "expanded_k": "#8c564b",
    "expanded_k_2x": "#e377c2",
    "sequential": "#7f7f7f",
    "topk_budget": "#bcbd22",
    "opposite_sign_masking": "#17becf",
    "opposite_sign_masking_ek2": "#1a9850",
}

STRATEGY_LABELS = {
    "direct": "Direct Addition",
    "cross_concept_masking": "Cross-Concept Masking",
    "gram_schmidt_pitch": "GS (orth. pitch)",
    "gram_schmidt_duration": "GS (orth. duration)",
    "cross_concept_sas": "Cross-Concept SAS",
    "expanded_k": "Expanded K (1.5×)",
    "expanded_k_2x": "Expanded K (2.0×)",
    "sequential": "Sequential",
    "topk_budget": "TopK Budget Alloc.",
    "opposite_sign_masking": "Opp-Sign Masking",
    "opposite_sign_masking_ek2": "Opp-Sign + EK 2×",
}

if HAS_MPL:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


# ════════════════════════════════════════════════════════════════════════════
# Data loading & structuring
# ════════════════════════════════════════════════════════════════════════════


def load_results(json_path: pathlib.Path) -> dict:
    """Load and validate results JSON."""
    with open(json_path) as f:
        data = json.load(f)
    results = [r for r in data["results"] if r.get("pitch_mean") is not None]
    strategies = sorted(set(r["strategy"] for r in results))
    print(f"Loaded {len(results)} valid results across {len(strategies)} strategies")
    return data, results, strategies


def get_strategy_results(results: list, strategy: str) -> list:
    """Filter results for one strategy."""
    return [r for r in results if r["strategy"] == strategy]


def get_baseline(sr: list) -> Tuple[Optional[float], Optional[float]]:
    """Get baseline pitch and duration at (0,0)."""
    bl = [r for r in sr if r["lambda_pitch"] == 0 and r["lambda_duration"] == 0]
    if bl:
        return bl[0]["pitch_mean"], bl[0]["duration_mean"]
    return None, None


# ════════════════════════════════════════════════════════════════════════════
# Metric computation
# ════════════════════════════════════════════════════════════════════════════


def compute_marginal_response(
    sr: list, lambda_key: str, metric_key: str, fixed_key: str, fixed_val: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the marginal response: vary one λ while holding the other at fixed_val.

    Returns (lambda_values, metric_values) sorted by lambda.
    """
    points = [
        (r[lambda_key], r[metric_key])
        for r in sr
        if abs(r[fixed_key] - fixed_val) < 1e-6 and r[metric_key] is not None
    ]
    if not points:
        # If exact fixed_val not found, average across the other axis
        by_lambda = defaultdict(list)
        for r in sr:
            if r[metric_key] is not None:
                by_lambda[r[lambda_key]].append(r[metric_key])
        points = [(k, np.mean(v)) for k, v in by_lambda.items()]

    points.sort(key=lambda x: x[0])
    if not points:
        return np.array([]), np.array([])
    lams, vals = zip(*points)
    return np.array(lams), np.array(vals)


def compute_averaged_marginal(
    sr: list, lambda_key: str, metric_key: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Average the metric across all values of the OTHER lambda for each target lambda."""
    by_lambda = defaultdict(list)
    for r in sr:
        if r[metric_key] is not None:
            by_lambda[r[lambda_key]].append(r[metric_key])
    items = sorted(by_lambda.items())
    if not items:
        return np.array([]), np.array([])
    lams = np.array([k for k, _ in items])
    vals = np.array([np.mean(v) for _, v in items])
    return lams, vals


def monotonicity_score(lambdas: np.ndarray, values: np.ndarray) -> float:
    """Fraction of consecutive pairs that are monotonically ordered.

    1.0 = perfectly monotonic (increasing), 0.0 = fully reversed.
    """
    if len(values) < 2:
        return float("nan")
    n_monotone = sum(1 for i in range(len(values) - 1) if values[i + 1] >= values[i])
    return n_monotone / (len(values) - 1)


def compute_strategy_metrics(sr: list) -> dict:
    """Compute comprehensive metrics for one strategy."""
    bl_pitch, bl_dur = get_baseline(sr)

    metrics = {}

    # ── On-diagonal: λ_pitch → pitch, λ_duration → duration ──
    for concept, lam_key, metric_key, fixed_key in [
        ("pitch", "lambda_pitch", "pitch_mean", "lambda_duration"),
        ("duration", "lambda_duration", "duration_mean", "lambda_pitch"),
    ]:
        # Fixed-other-at-zero marginal
        lams_fixed, vals_fixed = compute_marginal_response(
            sr, lam_key, metric_key, fixed_key, 0.0
        )
        # Averaged marginal
        lams_avg, vals_avg = compute_averaged_marginal(sr, lam_key, metric_key)

        stats = {}
        for suffix, lams, vals in [
            ("fixed0", lams_fixed, vals_fixed),
            ("averaged", lams_avg, vals_avg),
        ]:
            if len(lams) >= 3 and HAS_SCIPY:
                r, p = scipy_stats.pearsonr(lams, vals)
                slope, intercept, r_val, _, std_err = scipy_stats.linregress(lams, vals)
                stats[suffix] = {
                    "pearson_r": float(r),
                    "pearson_p": float(p),
                    "r_squared": float(r_val**2),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "std_err": float(std_err),
                    "monotonicity": float(monotonicity_score(lams, vals)),
                    "effect_range": float(vals.max() - vals.min()),
                    "lambda_values": lams.tolist(),
                    "metric_values": vals.tolist(),
                }
            else:
                stats[suffix] = {"pearson_r": float("nan"), "effect_range": 0.0}

        metrics[concept] = stats

    # ── Off-diagonal (cross-talk): λ_pitch → duration, λ_duration → pitch ──
    for concept, lam_key, metric_key, fixed_key in [
        (
            "crosstalk_pitch_to_duration",
            "lambda_pitch",
            "duration_mean",
            "lambda_duration",
        ),
        (
            "crosstalk_duration_to_pitch",
            "lambda_duration",
            "pitch_mean",
            "lambda_pitch",
        ),
    ]:
        lams_avg, vals_avg = compute_averaged_marginal(sr, lam_key, metric_key)
        if len(lams_avg) >= 3 and HAS_SCIPY:
            r, p = scipy_stats.pearsonr(lams_avg, vals_avg)
            metrics[concept] = {
                "pearson_r": float(r),
                "pearson_p": float(p),
                "effect_range": float(vals_avg.max() - vals_avg.min()),
                "lambda_values": lams_avg.tolist(),
                "metric_values": vals_avg.tolist(),
            }
        else:
            metrics[concept] = {"pearson_r": float("nan"), "effect_range": 0.0}

    # ── Dual Steering Success Rate ──
    if bl_pitch is not None and bl_dur is not None:
        n_total = 0
        n_pitch_success = 0
        n_duration_success = 0
        n_both_success = 0

        for r in sr:
            lp, ld = r["lambda_pitch"], r["lambda_duration"]
            if lp == 0 and ld == 0:
                continue  # skip baseline
            if r["pitch_mean"] is None or r["duration_mean"] is None:
                continue

            n_total += 1
            pitch_desired_dir = np.sign(lp)  # positive λ → higher pitch expected
            dur_desired_dir = np.sign(ld)  # positive λ → longer duration expected

            pitch_actual_dir = np.sign(r["pitch_mean"] - bl_pitch)
            dur_actual_dir = np.sign(r["duration_mean"] - bl_dur)

            # Success: moved in desired direction (or λ=0 for that concept)
            p_ok = (pitch_desired_dir == 0) or (pitch_actual_dir == pitch_desired_dir)
            d_ok = (dur_desired_dir == 0) or (dur_actual_dir == dur_desired_dir)

            if p_ok:
                n_pitch_success += 1
            if d_ok:
                n_duration_success += 1
            if p_ok and d_ok:
                n_both_success += 1

        metrics["steering_success"] = {
            "n_total": n_total,
            "pitch_success_rate": n_pitch_success / max(n_total, 1),
            "duration_success_rate": n_duration_success / max(n_total, 1),
            "both_success_rate": n_both_success / max(n_total, 1),
            "n_pitch_success": n_pitch_success,
            "n_duration_success": n_duration_success,
            "n_both_success": n_both_success,
        }
    else:
        metrics["steering_success"] = {
            "n_total": 0,
            "pitch_success_rate": float("nan"),
            "duration_success_rate": float("nan"),
            "both_success_rate": float("nan"),
        }

    # ── Quality degradation ──
    degs = [r["degradation"]["total_degradation"] for r in sr if r.get("degradation")]
    metrics["quality"] = {
        "mean_degradation": float(np.mean(degs)) if degs else float("nan"),
        "std_degradation": float(np.std(degs)) if degs else float("nan"),
        "min_degradation": float(np.min(degs)) if degs else float("nan"),
        "max_degradation": float(np.max(degs)) if degs else float("nan"),
    }

    # ── Composite score (consistent with single-concept SAS) ──
    # Score = |r_pitch| * range_pitch + |r_duration| * range_duration - degradation_penalty
    p_stats = metrics["pitch"].get("averaged", {})
    d_stats = metrics["duration"].get("averaged", {})
    r_p = abs(p_stats.get("pearson_r", 0) or 0)
    r_d = abs(d_stats.get("pearson_r", 0) or 0)
    range_p = p_stats.get("effect_range", 0) or 0
    range_d = d_stats.get("effect_range", 0) or 0
    deg = (
        metrics["quality"]["mean_degradation"]
        if not np.isnan(metrics["quality"]["mean_degradation"])
        else 0
    )
    metrics["composite_score"] = float(r_p * range_p + r_d * range_d - deg * 0.1)

    # ── Baseline ──
    metrics["baseline_pitch"] = bl_pitch
    metrics["baseline_duration"] = bl_dur

    return metrics


# ════════════════════════════════════════════════════════════════════════════
# Plotting
# ════════════════════════════════════════════════════════════════════════════


def _save(fig, output_dir: pathlib.Path, name: str):
    fig.savefig(output_dir / f"{name}.png")
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"  Saved {name}")


def plot_response_heatmaps(results: list, strategies: list, output_dir: pathlib.Path):
    """Fig 1 — Pitch & Duration heatmaps per strategy."""
    n = len(strategies)
    fig, axes = plt.subplots(n, 2, figsize=(13, 5 * n), squeeze=False)

    for row, strat in enumerate(strategies):
        sr = get_strategy_results(results, strat)
        lp_vals = sorted(set(r["lambda_pitch"] for r in sr))
        ld_vals = sorted(set(r["lambda_duration"] for r in sr))
        lp_idx = {v: i for i, v in enumerate(lp_vals)}
        ld_idx = {v: i for i, v in enumerate(ld_vals)}

        pitch_grid = np.full((len(ld_vals), len(lp_vals)), np.nan)
        dur_grid = np.full((len(ld_vals), len(lp_vals)), np.nan)
        for r in sr:
            i, j = ld_idx[r["lambda_duration"]], lp_idx[r["lambda_pitch"]]
            if r["pitch_mean"] is not None:
                pitch_grid[i, j] = r["pitch_mean"]
            if r["duration_mean"] is not None:
                dur_grid[i, j] = r["duration_mean"]

        for col, (grid, cmap, label, title_suffix) in enumerate(
            [
                (pitch_grid, "RdYlBu_r", "MIDI pitch", "Mean Pitch"),
                (dur_grid, "RdYlBu_r", "ticks", "Mean Duration"),
            ]
        ):
            ax = axes[row, col]
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap)
            ax.set_xticks(range(len(lp_vals)))
            ax.set_xticklabels([f"{v:+.2f}" for v in lp_vals], fontsize=7, rotation=45)
            ax.set_yticks(range(len(ld_vals)))
            ax.set_yticklabels([f"{v:+.2f}" for v in ld_vals], fontsize=7)
            ax.set_xlabel("λ_pitch")
            ax.set_ylabel("λ_duration")
            ax.set_title(f"{STRATEGY_LABELS.get(strat, strat)} — {title_suffix}")
            plt.colorbar(im, ax=ax, label=label)

    fig.suptitle("Dual-SAS: Pitch & Duration Response Heatmaps", fontsize=15, y=1.01)
    plt.tight_layout()
    _save(fig, output_dir, "fig1_response_heatmaps")


def plot_degradation_heatmaps(
    results: list, strategies: list, output_dir: pathlib.Path
):
    """Fig 2 — Degradation heatmaps."""
    n = len(strategies)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)

    for idx, strat in enumerate(strategies):
        ax = axes[idx // cols, idx % cols]
        sr = [r for r in results if r["strategy"] == strat and r.get("degradation")]
        lp_vals = sorted(set(r["lambda_pitch"] for r in sr))
        ld_vals = sorted(set(r["lambda_duration"] for r in sr))
        grid = np.full((len(ld_vals), len(lp_vals)), np.nan)
        lp_idx = {v: i for i, v in enumerate(lp_vals)}
        ld_idx = {v: i for i, v in enumerate(ld_vals)}
        for r in sr:
            grid[ld_idx[r["lambda_duration"]], lp_idx[r["lambda_pitch"]]] = r[
                "degradation"
            ]["total_degradation"]

        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="YlOrRd")
        ax.set_xticks(range(len(lp_vals)))
        ax.set_xticklabels([f"{v:+.1f}" for v in lp_vals], fontsize=7, rotation=45)
        ax.set_yticks(range(len(ld_vals)))
        ax.set_yticklabels([f"{v:+.1f}" for v in ld_vals], fontsize=7)
        ax.set_xlabel("λ_pitch")
        ax.set_ylabel("λ_duration")
        ax.set_title(STRATEGY_LABELS.get(strat, strat))
        plt.colorbar(im, ax=ax, label="Total Degradation")

    for idx in range(len(strategies), rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    fig.suptitle("Dual-SAS: Quality Degradation", fontsize=15, y=1.01)
    plt.tight_layout()
    _save(fig, output_dir, "fig2_degradation_heatmaps")


def plot_pareto(all_metrics: dict, output_dir: pathlib.Path):
    """Fig 3 — Pareto: control area vs degradation."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for strat, m in all_metrics.items():
        p_range = m["pitch"].get("averaged", {}).get("effect_range", 0) or 0
        d_range = m["duration"].get("averaged", {}).get("effect_range", 0) or 0
        control = p_range * d_range
        deg = m["quality"]["mean_degradation"]

        color = STRATEGY_COLORS.get(strat, "grey")
        label = STRATEGY_LABELS.get(strat, strat)
        ax.scatter(
            control,
            deg,
            s=200,
            color=color,
            label=label,
            edgecolors="black",
            linewidths=1,
            zorder=5,
        )
        ax.annotate(
            f"Δp={p_range:.0f} Δd={d_range:.0f}",
            (control, deg),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
        )

    ax.set_xlabel("Control Area (pitch range × duration range)")
    ax.set_ylabel("Mean Total Degradation")
    ax.set_title("Strategy Comparison: Control vs Quality (Pareto)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, output_dir, "fig3_pareto_frontier")


def plot_marginal_curves(all_metrics: dict, output_dir: pathlib.Path):
    """Fig 4 — Marginal response curves for pitch and duration.

    For each concept: averaged response as a function of the corresponding λ,
    with Pearson r and monotonicity annotated.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for col, (concept, lam_label) in enumerate(
        [
            ("pitch", "λ_pitch"),
            ("duration", "λ_duration"),
        ]
    ):
        ax = axes[col]
        for strat, m in all_metrics.items():
            stats = m[concept].get("averaged", {})
            lams = stats.get("lambda_values", [])
            vals = stats.get("metric_values", [])
            if not lams:
                continue
            color = STRATEGY_COLORS.get(strat, "grey")
            label = STRATEGY_LABELS.get(strat, strat)
            r = stats.get("pearson_r", float("nan"))
            mono = stats.get("monotonicity", float("nan"))
            ax.plot(
                lams,
                vals,
                "o-",
                color=color,
                label=f"{label} (r={r:+.2f}, M={mono:.2f})",
                linewidth=2,
                markersize=6,
            )

        metric_label = (
            "Mean MIDI Pitch" if concept == "pitch" else "Mean Duration (ticks)"
        )
        ax.set_xlabel(lam_label, fontsize=13)
        ax.set_ylabel(metric_label, fontsize=13)
        ax.set_title(f"{concept.title()} Response (averaged over other λ)")
        ax.legend(fontsize=8, loc="best")
        ax.axhline(y=0, color="grey", linestyle=":", alpha=0.3)

    fig.suptitle("Dual-SAS: Marginal Response Curves", fontsize=15)
    plt.tight_layout()
    _save(fig, output_dir, "fig4_marginal_response_curves")


def plot_steering_success(all_metrics: dict, output_dir: pathlib.Path):
    """Fig 5 — Dual Steering Success Rate bar chart."""
    strategies = list(all_metrics.keys())
    n = len(strategies)

    pitch_rates = []
    dur_rates = []
    both_rates = []
    for s in strategies:
        ss = all_metrics[s].get("steering_success", {})
        pitch_rates.append(ss.get("pitch_success_rate", 0) * 100)
        dur_rates.append(ss.get("duration_success_rate", 0) * 100)
        both_rates.append(ss.get("both_success_rate", 0) * 100)

    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, 3 * n), 6))
    bars1 = ax.bar(
        x - width,
        pitch_rates,
        width,
        label="Pitch Success",
        color="#4A90D9",
        edgecolor="white",
    )
    bars2 = ax.bar(
        x,
        dur_rates,
        width,
        label="Duration Success",
        color="#E67E22",
        edgecolor="white",
    )
    bars3 = ax.bar(
        x + width,
        both_rates,
        width,
        label="Both Success",
        color="#27AE60",
        edgecolor="white",
    )

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1,
                    f"{h:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [STRATEGY_LABELS.get(s, s) for s in strategies],
        fontsize=9,
        rotation=15,
        ha="right",
    )
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 115)
    ax.set_title(
        "Dual Steering Success Rate\n(% configs where output moved in desired direction vs baseline)"
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    _save(fig, output_dir, "fig5_steering_success_rate")


def plot_crosstalk(all_metrics: dict, output_dir: pathlib.Path):
    """Fig 6 — Cross-talk: off-diagonal correlations."""
    strategies = list(all_metrics.keys())
    n = len(strategies)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, (key, title, xlabel, ylabel) in enumerate(
        [
            (
                "crosstalk_pitch_to_duration",
                "λ_pitch → Duration (Cross-talk)",
                "λ_pitch",
                "Mean Duration (ticks)",
            ),
            (
                "crosstalk_duration_to_pitch",
                "λ_duration → Pitch (Cross-talk)",
                "λ_duration",
                "Mean MIDI Pitch",
            ),
        ]
    ):
        ax = axes[col]
        for strat, m in all_metrics.items():
            ct = m.get(key, {})
            lams = ct.get("lambda_values", [])
            vals = ct.get("metric_values", [])
            if not lams:
                continue
            r = ct.get("pearson_r", float("nan"))
            color = STRATEGY_COLORS.get(strat, "grey")
            label = STRATEGY_LABELS.get(strat, strat)
            ax.plot(
                lams,
                vals,
                "s--",
                color=color,
                alpha=0.8,
                label=f"{label} (r={r:+.2f})",
                linewidth=1.5,
                markersize=5,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Dual-SAS: Cross-Talk Analysis", fontsize=15)
    plt.tight_layout()
    _save(fig, output_dir, "fig6_crosstalk_analysis")


# ════════════════════════════════════════════════════════════════════════════
# Console report
# ════════════════════════════════════════════════════════════════════════════


def print_report(all_metrics: dict):
    """Print a comprehensive console report."""
    print("\n" + "=" * 90)
    print("DUAL-SAS STRATEGY ANALYSIS REPORT")
    print("=" * 90)

    for strat, m in all_metrics.items():
        label = STRATEGY_LABELS.get(strat, strat)
        print(f"\n{'─' * 90}")
        print(f"  {label}  ({strat})")
        print(f"{'─' * 90}")

        bl_p = m.get("baseline_pitch")
        bl_d = m.get("baseline_duration")
        print(
            f"  Baseline (0,0):  pitch = {bl_p:.1f},  duration = {bl_d:.1f}"
            if bl_p
            else "  Baseline: N/A"
        )

        for concept in ["pitch", "duration"]:
            stats = m[concept].get("averaged", {})
            r = stats.get("pearson_r", float("nan"))
            r2 = stats.get("r_squared", float("nan"))
            slope = stats.get("slope", float("nan"))
            mono = stats.get("monotonicity", float("nan"))
            rng = stats.get("effect_range", 0)
            print(
                f"\n  {concept.upper()} (λ_{concept} → {concept}, averaged over other λ):"
            )
            print(f"    Pearson r   = {r:+.4f}")
            print(f"    R²          = {r2:.4f}")
            print(f"    Slope       = {slope:+.2f}")
            print(f"    Monotonicity= {mono:.2f}  {'✓' if mono >= 0.8 else '✗'}")
            print(f"    Effect Δ    = {rng:.1f}")

        # Cross-talk
        ct_pd = m.get("crosstalk_pitch_to_duration", {})
        ct_dp = m.get("crosstalk_duration_to_pitch", {})
        print(f"\n  CROSS-TALK:")
        print(
            f"    λ_pitch → duration:  r = {ct_pd.get('pearson_r', float('nan')):+.4f},"
            f"  Δ = {ct_pd.get('effect_range', 0):.1f}"
        )
        print(
            f"    λ_duration → pitch:  r = {ct_dp.get('pearson_r', float('nan')):+.4f},"
            f"  Δ = {ct_dp.get('effect_range', 0):.1f}"
        )

        # Success
        ss = m.get("steering_success", {})
        print(
            f"\n  DUAL STEERING SUCCESS ({ss.get('n_total', 0)} non-baseline configs):"
        )
        print(
            f"    Pitch only:    {ss.get('pitch_success_rate', 0) * 100:.1f}%"
            f"  ({ss.get('n_pitch_success', 0)}/{ss.get('n_total', 0)})"
        )
        print(
            f"    Duration only: {ss.get('duration_success_rate', 0) * 100:.1f}%"
            f"  ({ss.get('n_duration_success', 0)}/{ss.get('n_total', 0)})"
        )
        print(
            f"    Both:          {ss.get('both_success_rate', 0) * 100:.1f}%"
            f"  ({ss.get('n_both_success', 0)}/{ss.get('n_total', 0)})"
        )

        # Quality
        q = m["quality"]
        print(f"\n  QUALITY:")
        print(
            f"    Mean degradation: {q['mean_degradation']:.2f} ± {q['std_degradation']:.2f}"
        )
        print(f"    Range: [{q['min_degradation']:.2f}, {q['max_degradation']:.2f}]")

        print(f"\n  COMPOSITE SCORE: {m['composite_score']:.2f}")

    # ── Summary table ──
    print("\n\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    header = (
        f"{'Strategy':<24} {'r_pitch':>8} {'r_dur':>8} {'Δ_pitch':>8} {'Δ_dur':>8}"
        f" {'Mono_p':>7} {'Mono_d':>7} {'Both%':>7} {'Degrad':>7} {'Score':>7}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for strat, m in all_metrics.items():
        p = m["pitch"].get("averaged", {})
        d = m["duration"].get("averaged", {})
        ss = m.get("steering_success", {})
        rows.append(
            {
                "strat": strat,
                "r_p": p.get("pearson_r", float("nan")),
                "r_d": d.get("pearson_r", float("nan")),
                "range_p": p.get("effect_range", 0),
                "range_d": d.get("effect_range", 0),
                "mono_p": p.get("monotonicity", float("nan")),
                "mono_d": d.get("monotonicity", float("nan")),
                "both": ss.get("both_success_rate", 0) * 100,
                "deg": m["quality"]["mean_degradation"],
                "score": m["composite_score"],
            }
        )

    rows.sort(key=lambda x: x["score"], reverse=True)
    for r in rows:
        label = STRATEGY_LABELS.get(r["strat"], r["strat"])[:24]
        print(
            f"{label:<24} {r['r_p']:>+8.3f} {r['r_d']:>+8.3f}"
            f" {r['range_p']:>8.1f} {r['range_d']:>8.1f}"
            f" {r['mono_p']:>7.2f} {r['mono_d']:>7.2f}"
            f" {r['both']:>6.1f}% {r['deg']:>7.2f} {r['score']:>7.1f}"
        )


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Analyze dual-SAS unconditioned grid-search results"
    )
    parser.add_argument(
        "--json",
        type=pathlib.Path,
        required=True,
        help="Path to unconditioned_results.json",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("plots/dual_sas_analysis"),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Restrict analysis to these strategies (default: all in JSON)",
    )
    args = parser.parse_args()

    if not args.json.exists():
        print(f"ERROR: {args.json} not found")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    data, results, strategies = load_results(args.json)
    if args.strategies:
        strategies = [s for s in strategies if s in args.strategies]
        results = [r for r in results if r["strategy"] in strategies]
    print(f"Analyzing strategies: {strategies}")

    # Compute metrics
    print("\nComputing metrics...")
    all_metrics = {}
    for strat in strategies:
        sr = get_strategy_results(results, strat)
        all_metrics[strat] = compute_strategy_metrics(sr)

    # Console report
    print_report(all_metrics)

    # Save metrics JSON
    metrics_path = args.output_dir / "analysis_metrics.json"

    # Convert NaN to null for JSON
    def _clean(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(metrics_path, "w") as f:
        json.dump(_clean(all_metrics), f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Plots
    if HAS_MPL:
        print("\nGenerating plots...")
        plot_response_heatmaps(results, strategies, args.output_dir)
        plot_degradation_heatmaps(results, strategies, args.output_dir)
        plot_pareto(all_metrics, args.output_dir)
        plot_marginal_curves(all_metrics, args.output_dir)
        plot_steering_success(all_metrics, args.output_dir)
        plot_crosstalk(all_metrics, args.output_dir)
        print(f"\nAll plots saved to {args.output_dir}")
    else:
        print("\nWARNING: matplotlib not available — skipping plots")

    print("\nDone!")


if __name__ == "__main__":
    main()
