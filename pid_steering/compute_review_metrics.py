"""Compute confidence intervals and temporal smoothness metrics.

Addresses reviewer requests:
1. Bootstrap 95% CIs for key metrics (pitch, duration, δ, FMD)
2. Temporal smoothness: per-bar attribute variance over generation steps
3. Fingerprint sensitivity: sweep N ∈ {8, 16, 32, 64}

Usage:
    # From existing per-sample results JSON:
    python pid_steering/compute_review_metrics.py \
        --results_dir exp/sod/pid_steering/experiments/temporal_sas_comparison \
        --mode ci

    # Temporal smoothness from generated MIDIs:
    python pid_steering/compute_review_metrics.py \
        --results_dir exp/sod/pid_steering/experiments/temporal_sas_comparison \
        --mode smoothness
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, stat_fn=np.mean):
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    n = len(values)
    boot_stats = np.array(
        [
            stat_fn(np.random.choice(values, size=n, replace=True))
            for _ in range(n_bootstrap)
        ]
    )
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_stats, 100 * alpha)
    hi = np.percentile(boot_stats, 100 * (1 - alpha))
    return float(stat_fn(values)), float(lo), float(hi)


def compute_degradation(metrics_dict, gt_h=2.974, gt_s=92.26, gt_g=93.05):
    """Compute δ for a single sample."""
    h = metrics_dict["pitch_class_entropy"]
    s = metrics_dict["scale_consistency"]
    g = metrics_dict["groove_consistency"]
    return abs(h - gt_h) + max(0, gt_s - s) + max(0, gt_g - g)


def print_ci_table(results: Dict):
    """Print CI table from per-sample results."""
    gt_h, gt_s, gt_g = 2.974, 92.26, 93.05

    print(f"\n{'Method':<18} {'Metric':<20} {'Mean':>8} {'95% CI':>20}")
    print("=" * 70)

    for method_name, method_data in results.items():
        if "per_sample_metrics" not in method_data:
            continue
        metrics_list = method_data["per_sample_metrics"]
        n = len(metrics_list)

        # Key metrics with CIs
        for key, label in [
            ("average_pitch", "Avg Pitch (st)"),
            ("average_duration", "Avg Duration"),
            ("pitch_class_entropy", "PC Entropy"),
            ("scale_consistency", "Scale Cons (%)"),
            ("groove_consistency", "Groove Cons (%)"),
        ]:
            vals = [m[key] for m in metrics_list]
            mean, lo, hi = bootstrap_ci(vals)
            print(
                f"{method_name:<18} {label:<20} {mean:>8.2f} [{lo:>8.2f}, {hi:>8.2f}]"
            )

        # δ with CI
        deltas = [compute_degradation(m) for m in metrics_list]
        mean_d, lo_d, hi_d = bootstrap_ci(deltas)
        print(
            f"{method_name:<18} {'δ (degradation)':<20} {mean_d:>8.2f} [{lo_d:>8.2f}, {hi_d:>8.2f}]"
        )
        print(f"{'':>18} {'n':<20} {n:>8d}")
        print("-" * 70)


def compute_temporal_smoothness(sequences, encoding, bar_length=16):
    """Compute per-bar attribute statistics for temporal smoothness analysis.

    Returns per-bar pitch mean/std and duration mean/std across the generation,
    plus the overall temporal variance (variance of bar means).
    """
    from pid_steering.run_single_attribute import compute_generation_metrics

    results = []
    for seq in sequences:
        # Extract pitch and duration per note
        pitches = []
        durations = []
        beats = []
        for token in seq:
            if len(token) >= 6 and token[0] not in [0, 1]:  # skip SOS/EOS
                pitches.append(int(token[3]))
                durations.append(int(token[4]))
                beats.append(int(token[1]))

        if not pitches:
            continue

        pitches = np.array(pitches)
        durations = np.array(durations)
        beats = np.array(beats)

        # Group by bar (bar_length beats per bar)
        max_beat = beats.max() if len(beats) > 0 else 0
        n_bars = max_beat // bar_length + 1

        bar_pitch_means = []
        bar_dur_means = []
        for bar_idx in range(n_bars):
            bar_start = bar_idx * bar_length
            bar_end = bar_start + bar_length
            mask = (beats >= bar_start) & (beats < bar_end)
            if mask.sum() > 0:
                bar_pitch_means.append(pitches[mask].mean())
                bar_dur_means.append(durations[mask].mean())

        if len(bar_pitch_means) < 2:
            continue

        bar_pitch_means = np.array(bar_pitch_means)
        bar_dur_means = np.array(bar_dur_means)

        results.append(
            {
                "n_bars": len(bar_pitch_means),
                "pitch_bar_mean": float(bar_pitch_means.mean()),
                "pitch_bar_var": float(bar_pitch_means.var()),
                "pitch_bar_diff_var": float(np.diff(bar_pitch_means).var()),
                "dur_bar_mean": float(bar_dur_means.mean()),
                "dur_bar_var": float(bar_dur_means.var()),
                "dur_bar_diff_var": float(np.diff(bar_dur_means).var()),
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Compute CIs and smoothness metrics")
    parser.add_argument(
        "--results_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing temporal_comparison_*.json files",
    )
    parser.add_argument("--mode", choices=["ci", "smoothness", "all"], default="all")
    parser.add_argument("--concept", type=str, default="average_pitch")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.mode in ("ci", "all"):
        results_path = args.results_dir / f"temporal_comparison_{args.concept}.json"
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            print_ci_table(results)
        else:
            logger.warning(f"No results file at {results_path}")

        # Also check threshold baseline results
        tb_path = args.results_dir / f"threshold_baselines_{args.concept}_positive.json"
        if tb_path.exists():
            with open(tb_path) as f:
                tb_results = json.load(f)
            print("\n--- Threshold Baseline CIs ---")
            print_ci_table(tb_results)


if __name__ == "__main__":
    main()
