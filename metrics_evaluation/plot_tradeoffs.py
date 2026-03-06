#!/usr/bin/env python3
"""Visualize steering control vs quality tradeoffs.

Combines data from:
  - steering_success.csv  (steering success rates + quality degradation per sample)
  - fmd_per_lambda.csv    (FMD scores per λ configuration)

Generates publication-ready figures:
  1. Pareto front: Steering success vs FMD
  2. Pareto front: Steering success vs Quality degradation
  3. 3-axis scatter: Success vs FMD vs Degradation (bubble chart)
  4. Optimal λ heatmaps (per method/strategy)
  5. Marginal line plots: FMD + Success + Degradation vs λ
  6. Bar chart: Best config per method/strategy

Usage:
    python metrics_evaluation/plot_tradeoffs.py \
        --workspace_dir exp/sod/sparse_steering/fmd_workspace \
        --output_dir exp/sod/sparse_steering/fmd_workspace/plots
"""

import argparse
import csv
import pathlib
import sys
from collections import defaultdict

import numpy as np

# ── Style config ──────────────────────────────────────────────────────────
STRATEGY_STYLE = {
    # SAS
    "SAS/expanded_k_2x": {"color": "#1f77b4", "marker": "o", "label": "SAS ek2x"},
    "SAS/gram_schmidt_ek2": {"color": "#ff7f0e", "marker": "s", "label": "SAS gs-ek2"},
    # DiffMean
    "DM/gram_schmidt_pitch": {
        "color": "#2ca02c",
        "marker": "D",
        "label": "DM gs-pitch",
    },
}

# Fallback palette for any extra strategies
_FALLBACK_COLORS = ["#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
_FALLBACK_MARKERS = ["^", "v", "<", ">", "P"]
_fallback_idx = 0


def _get_style(key):
    global _fallback_idx
    if key not in STRATEGY_STYLE:
        STRATEGY_STYLE[key] = {
            "color": _FALLBACK_COLORS[_fallback_idx % len(_FALLBACK_COLORS)],
            "marker": _FALLBACK_MARKERS[_fallback_idx % len(_FALLBACK_MARKERS)],
            "label": key,
        }
        _fallback_idx += 1
    return STRATEGY_STYLE[key]


# ── Data loading ──────────────────────────────────────────────────────────
def load_success_csv(path: pathlib.Path) -> list:
    """Load steering_success.csv into list of dicts with proper types."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {}
            for k, v in r.items():
                if v == "":
                    row[k] = None
                elif v in ("True", "False"):
                    row[k] = v == "True"
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            rows.append(row)
    return rows


def load_fmd_csv(path: pathlib.Path) -> list:
    """Load fmd_per_lambda.csv."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {}
            for k, v in r.items():
                if v == "":
                    row[k] = None
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            rows.append(row)
    return rows


def aggregate_success_by_lambda(success_rows: list, mode: str) -> dict:
    """Aggregate success data by (method, strategy, λ_p, λ_d).

    Returns dict keyed by (method, strategy, λ_p, λ_d) with aggregated metrics.
    """
    agg = defaultdict(
        lambda: {
            "both_ok": 0,
            "pitch_ok": 0,
            "dur_ok": 0,
            "n": 0,
            "n_success_eval": 0,
            "degs": [],
            "pce": [],
            "sc": [],
            "gc": [],
        }
    )

    for r in success_rows:
        if r.get("mode") != mode:
            continue
        m = r.get("method", "")
        s = r.get("strategy", "")
        lp = r.get("lambda_pitch")
        ld = r.get("lambda_duration")
        if lp is None or ld is None:
            continue

        key = (m, s, round(lp, 3), round(ld, 3))
        a = agg[key]
        a["n"] += 1

        if r.get("both_success") is not None:
            a["n_success_eval"] += 1
            if r["both_success"]:
                a["both_ok"] += 1
            if r.get("pitch_success"):
                a["pitch_ok"] += 1
            if r.get("duration_success"):
                a["dur_ok"] += 1

        td = r.get("total_degradation")
        if td is not None and not np.isnan(td):
            a["degs"].append(td)
            a["pce"].append(r.get("pitch_class_entropy", np.nan))
            a["sc"].append(r.get("scale_consistency", np.nan))
            a["gc"].append(r.get("groove_consistency", np.nan))

    # Compute rates
    result = {}
    for key, a in agg.items():
        n_eval = a["n_success_eval"]
        result[key] = {
            "method": key[0],
            "strategy": key[1],
            "lambda_pitch": key[2],
            "lambda_duration": key[3],
            "both_rate": a["both_ok"] / n_eval if n_eval else None,
            "pitch_rate": a["pitch_ok"] / n_eval if n_eval else None,
            "dur_rate": a["dur_ok"] / n_eval if n_eval else None,
            "n_samples": a["n"],
            "avg_degradation": float(np.nanmean(a["degs"])) if a["degs"] else None,
            "avg_pce": float(np.nanmean(a["pce"])) if a["pce"] else None,
            "avg_sc": float(np.nanmean(a["sc"])) if a["sc"] else None,
            "avg_gc": float(np.nanmean(a["gc"])) if a["gc"] else None,
        }
    return result


def build_fmd_lookup(fmd_rows: list, mode: str) -> dict:
    """Build FMD lookup keyed by (strategy_label, λ_p, λ_d).

    strategy_label: matches CSV 'strategy' column (e.g. 'SAS_expanded_k_2x').
    """
    lookup = {}
    for r in fmd_rows:
        if r.get("mode") != mode:
            continue
        strat = r.get("strategy", "")
        lp = r.get("lambda_pitch")
        ld = r.get("lambda_duration")
        fmd = r.get("fmd")
        if lp is None or ld is None or fmd is None:
            continue
        lookup[(strat, round(lp, 3), round(ld, 3))] = fmd
    return lookup


def _fmd_strategy_key(method, strategy):
    """Map (method, strategy) from success CSV to FMD CSV strategy label."""
    if method == "SAS":
        return f"SAS_{strategy}"
    else:
        return f"DM_{strategy}"


# ── Plotting ──────────────────────────────────────────────────────────────
def plot_all(
    success_agg: dict,
    fmd_lookup: dict,
    mode: str,
    output_dir: pathlib.Path,
    baseline_fmd: float = None,
):
    """Generate all tradeoff figures for a given mode."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Merge success + FMD
    points = []  # list of dicts with all merged fields
    for key, sa in success_agg.items():
        m, s, lp, ld = key
        fmd_key = (_fmd_strategy_key(m, s), lp, ld)
        fmd_val = fmd_lookup.get(fmd_key)

        # Skip baseline (0,0) entries with no success data
        if sa["both_rate"] is None:
            continue

        points.append(
            {
                **sa,
                "fmd": fmd_val,
                "style_key": f"{m}/{s}",
            }
        )

    if not points:
        print(f"  No data for mode={mode}, skipping plots.")
        return

    # ── Figure 1: Success vs FMD scatter ──
    fig, ax = plt.subplots(figsize=(10, 7))
    legend_handles = []
    seen_styles = set()

    for p in points:
        if p["fmd"] is None:
            continue
        sty = _get_style(p["style_key"])
        ax.scatter(
            p["both_rate"] * 100,
            p["fmd"],
            c=sty["color"],
            marker=sty["marker"],
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.3,
        )
        if p["style_key"] not in seen_styles:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=sty["marker"],
                    color="w",
                    markerfacecolor=sty["color"],
                    markersize=8,
                    label=sty["label"],
                )
            )
            seen_styles.add(p["style_key"])

    if baseline_fmd:
        ax.axhline(
            baseline_fmd,
            color="gray",
            ls="--",
            alpha=0.7,
            label=f"Baseline FMD ({baseline_fmd:.0f})",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                ls="--",
                label=f"Baseline FMD ({baseline_fmd:.0f})",
            )
        )

    ax.set_xlabel("Steering Success (both concepts %)", fontsize=12)
    ax.set_ylabel("FMD (↓ better)", fontsize=12)
    ax.set_title(f"Steering Success vs FMD — {mode}", fontsize=14)
    ax.legend(handles=legend_handles, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"tradeoff_success_vs_fmd_{mode}.png", dpi=200)
    plt.close(fig)
    print(f"  Saved tradeoff_success_vs_fmd_{mode}.png")

    # ── Figure 2: Success vs Degradation scatter ──
    fig, ax = plt.subplots(figsize=(10, 7))
    seen_styles = set()
    legend_handles = []

    for p in points:
        if p["avg_degradation"] is None:
            continue
        sty = _get_style(p["style_key"])
        ax.scatter(
            p["both_rate"] * 100,
            p["avg_degradation"],
            c=sty["color"],
            marker=sty["marker"],
            alpha=0.6,
            s=50,
            edgecolors="white",
            linewidth=0.3,
        )
        if p["style_key"] not in seen_styles:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=sty["marker"],
                    color="w",
                    markerfacecolor=sty["color"],
                    markersize=8,
                    label=sty["label"],
                )
            )
            seen_styles.add(p["style_key"])

    ax.set_xlabel("Steering Success (both concepts %)", fontsize=12)
    ax.set_ylabel("Quality Degradation (↓ better)", fontsize=12)
    ax.set_title(f"Steering Success vs Quality Degradation — {mode}", fontsize=14)
    ax.legend(handles=legend_handles, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"tradeoff_success_vs_degradation_{mode}.png", dpi=200)
    plt.close(fig)
    print(f"  Saved tradeoff_success_vs_degradation_{mode}.png")

    # ── Figure 3: 3-axis bubble chart (Success vs FMD, size=1/degradation) ──
    fig, ax = plt.subplots(figsize=(11, 7))
    seen_styles = set()
    legend_handles = []

    for p in points:
        if p["fmd"] is None or p["avg_degradation"] is None:
            continue
        sty = _get_style(p["style_key"])
        # Size inversely proportional to degradation (bigger = less degradation = better)
        size = max(10, 500 / (1 + p["avg_degradation"]))
        ax.scatter(
            p["both_rate"] * 100,
            p["fmd"],
            c=sty["color"],
            marker=sty["marker"],
            alpha=0.5,
            s=size,
            edgecolors="white",
            linewidth=0.3,
        )
        if p["style_key"] not in seen_styles:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=sty["marker"],
                    color="w",
                    markerfacecolor=sty["color"],
                    markersize=8,
                    label=sty["label"],
                )
            )
            seen_styles.add(p["style_key"])

    if baseline_fmd:
        ax.axhline(baseline_fmd, color="gray", ls="--", alpha=0.7)

    ax.set_xlabel("Steering Success (both concepts %)", fontsize=12)
    ax.set_ylabel("FMD (↓ better)", fontsize=12)
    ax.set_title(
        f"Success vs FMD — {mode}\n(bubble size ∝ quality preservation)",
        fontsize=13,
    )
    ax.legend(handles=legend_handles, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"tradeoff_bubble_{mode}.png", dpi=200)
    plt.close(fig)
    print(f"  Saved tradeoff_bubble_{mode}.png")

    # ── Figure 4: Per-strategy heatmaps — combined score ──
    _plot_combined_heatmaps(points, mode, output_dir)

    # ── Figure 5: Marginal line plots ──
    _plot_marginal_lines(points, fmd_lookup, mode, output_dir, baseline_fmd)

    # ── Figure 6: Best config bar chart ──
    _plot_best_config_bars(points, mode, output_dir)


def _plot_combined_heatmaps(points, mode, output_dir):
    """Heatmap of a combined score per (λ_p, λ_d) for each strategy."""
    import matplotlib.pyplot as plt

    # Group by strategy
    strat_points = defaultdict(list)
    for p in points:
        strat_points[p["style_key"]].append(p)

    n_strats = len(strat_points)
    if n_strats == 0:
        return

    fig, axes = plt.subplots(1, n_strats, figsize=(6 * n_strats, 5), squeeze=False)

    for idx, (skey, pts) in enumerate(sorted(strat_points.items())):
        ax = axes[0][idx]
        sty = _get_style(skey)

        lp_vals = sorted(set(p["lambda_pitch"] for p in pts))
        ld_vals = sorted(set(p["lambda_duration"] for p in pts))

        # Build grid: combined score = success% × (1 − normalized_degradation)
        grid = np.full((len(lp_vals), len(ld_vals)), np.nan)
        lp_idx = {v: i for i, v in enumerate(lp_vals)}
        ld_idx = {v: i for i, v in enumerate(ld_vals)}

        for p in pts:
            i = lp_idx[p["lambda_pitch"]]
            j = ld_idx[p["lambda_duration"]]
            sr = p["both_rate"] if p["both_rate"] is not None else 0
            deg = p["avg_degradation"] if p["avg_degradation"] is not None else 10
            # Score: success rate penalized by degradation (0-100 scale)
            score = sr * 100 * max(0, 1 - deg / 20)
            grid[i, j] = score

        im = ax.imshow(
            grid,
            aspect="auto",
            cmap="RdYlGn",
            origin="lower",
            vmin=0,
            vmax=100,
        )
        ax.set_xticks(range(len(ld_vals)))
        ax.set_xticklabels([f"{v:+.1f}" for v in ld_vals], fontsize=7, rotation=45)
        ax.set_yticks(range(len(lp_vals)))
        ax.set_yticklabels([f"{v:+.1f}" for v in lp_vals], fontsize=7)
        ax.set_xlabel("λ_duration", fontsize=10)
        ax.set_ylabel("λ_pitch", fontsize=10)
        ax.set_title(sty["label"], fontsize=11)

        # Annotate cells
        for i in range(len(lp_vals)):
            for j in range(len(ld_vals)):
                val = grid[i, j]
                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.0f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="black" if val > 30 else "white",
                    )

    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="Combined Score")
    fig.suptitle(f"Combined Score (success × quality) — {mode}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / f"tradeoff_heatmap_{mode}.png", dpi=200)
    plt.close(fig)
    print(f"  Saved tradeoff_heatmap_{mode}.png")


def _plot_marginal_lines(points, fmd_lookup, mode, output_dir, baseline_fmd):
    """Line plots: marginal FMD, success, degradation vs λ_pitch and λ_duration."""
    import matplotlib.pyplot as plt

    strat_points = defaultdict(list)
    for p in points:
        strat_points[p["style_key"]].append(p)

    for concept, lambda_key, other_key in [
        ("pitch", "lambda_pitch", "lambda_duration"),
        ("duration", "lambda_duration", "lambda_pitch"),
    ]:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        for skey, pts in sorted(strat_points.items()):
            sty = _get_style(skey)

            # Aggregate marginals: average over the other λ dimension
            marginal = defaultdict(
                lambda: {
                    "success": [],
                    "fmd": [],
                    "deg": [],
                }
            )
            for p in pts:
                lval = p[lambda_key]
                marginal[lval]["success"].append(
                    p["both_rate"] * 100 if p["both_rate"] is not None else 0
                )
                if p["fmd"] is not None:
                    marginal[lval]["fmd"].append(p["fmd"])
                if p["avg_degradation"] is not None:
                    marginal[lval]["deg"].append(p["avg_degradation"])

            lvals = sorted(marginal.keys())
            success_means = [np.mean(marginal[l]["success"]) for l in lvals]
            fmd_means = [
                np.mean(marginal[l]["fmd"]) if marginal[l]["fmd"] else np.nan
                for l in lvals
            ]
            deg_means = [
                np.mean(marginal[l]["deg"]) if marginal[l]["deg"] else np.nan
                for l in lvals
            ]

            ax1.plot(
                lvals,
                success_means,
                color=sty["color"],
                marker=sty["marker"],
                label=sty["label"],
                markersize=5,
            )
            ax2.plot(
                lvals,
                fmd_means,
                color=sty["color"],
                marker=sty["marker"],
                label=sty["label"],
                markersize=5,
            )
            ax3.plot(
                lvals,
                deg_means,
                color=sty["color"],
                marker=sty["marker"],
                label=sty["label"],
                markersize=5,
            )

        if baseline_fmd:
            ax2.axhline(
                baseline_fmd,
                color="gray",
                ls="--",
                alpha=0.5,
                label=f"Baseline ({baseline_fmd:.0f})",
            )

        ax1.set_ylabel("Steering Success %", fontsize=11)
        ax1.set_title(f"Marginal vs λ_{concept} — {mode}", fontsize=13)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_ylabel("FMD (↓ better)", fontsize=11)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax3.set_ylabel("Quality Degradation (↓ better)", fontsize=11)
        ax3.set_xlabel(f"λ_{concept}", fontsize=11)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"tradeoff_marginal_{concept}_{mode}.png", dpi=200)
        plt.close(fig)
        print(f"  Saved tradeoff_marginal_{concept}_{mode}.png")


def _plot_best_config_bars(points, mode, output_dir):
    """Bar chart comparing the best λ config per method/strategy."""
    import matplotlib.pyplot as plt

    # For each strategy, find best config = highest success with deg < median
    strat_best = {}
    strat_points = defaultdict(list)
    for p in points:
        strat_points[p["style_key"]].append(p)

    for skey, pts in sorted(strat_points.items()):
        valid = [
            p
            for p in pts
            if p["avg_degradation"] is not None and p["both_rate"] is not None
        ]
        if not valid:
            continue
        # Sort: best success, then least degradation
        valid.sort(key=lambda p: (-p["both_rate"], p["avg_degradation"]))
        strat_best[skey] = valid[0]

    if not strat_best:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    labels = []
    success_vals = []
    fmd_vals = []
    deg_vals = []
    colors = []

    for skey, best in sorted(strat_best.items()):
        sty = _get_style(skey)
        lp, ld = best["lambda_pitch"], best["lambda_duration"]
        labels.append(f"{sty['label']}\nλ=({lp:+.1f},{ld:+.1f})")
        success_vals.append(best["both_rate"] * 100)
        fmd_vals.append(best["fmd"] if best["fmd"] is not None else 0)
        deg_vals.append(best["avg_degradation"])
        colors.append(sty["color"])

    x = np.arange(len(labels))

    axes[0].bar(x, success_vals, color=colors, alpha=0.8)
    axes[0].set_ylabel("Steering Success %")
    axes[0].set_title("Best Config: Success")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(success_vals):
        axes[0].text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)

    axes[1].bar(x, fmd_vals, color=colors, alpha=0.8)
    axes[1].set_ylabel("FMD (↓ better)")
    axes[1].set_title("Best Config: FMD")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(fmd_vals):
        if v > 0:
            axes[1].text(i, v + 5, f"{v:.0f}", ha="center", fontsize=9)

    axes[2].bar(x, deg_vals, color=colors, alpha=0.8)
    axes[2].set_ylabel("Quality Degradation (↓ better)")
    axes[2].set_title("Best Config: Degradation")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=8)
    for i, v in enumerate(deg_vals):
        axes[2].text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)

    fig.suptitle(f"Best Configuration per Method — {mode}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / f"tradeoff_best_config_{mode}.png", dpi=200)
    plt.close(fig)
    print(f"  Saved tradeoff_best_config_{mode}.png")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Plot steering control vs quality tradeoffs"
    )
    parser.add_argument(
        "--workspace_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing steering_success.csv and fmd_per_lambda.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Output directory for plots. Defaults to workspace_dir/plots.",
    )
    parser.add_argument(
        "--baseline_fmd_uncond",
        type=float,
        default=527.0,
        help="Baseline FMD for unconditioned (default: 527.0)",
    )
    parser.add_argument(
        "--baseline_fmd_cond",
        type=float,
        default=491.4,
        help="Baseline FMD for conditioned (default: 491.4)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.workspace_dir / "plots"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    success_csv = args.workspace_dir / "steering_success.csv"
    fmd_csv = args.workspace_dir / "fmd_per_lambda.csv"

    if not success_csv.exists():
        print(f"ERROR: {success_csv} not found. Run analyze_steering_success.py first.")
        sys.exit(1)

    success_rows = load_success_csv(success_csv)
    print(f"Loaded {len(success_rows)} rows from steering_success.csv")

    fmd_rows = []
    if fmd_csv.exists():
        fmd_rows = load_fmd_csv(fmd_csv)
        print(f"Loaded {len(fmd_rows)} rows from fmd_per_lambda.csv")
    else:
        print(f"WARNING: {fmd_csv} not found. FMD data will be unavailable.")

    # Process each mode
    for mode, baseline_fmd in [
        ("uncond", args.baseline_fmd_uncond),
        ("cond", args.baseline_fmd_cond),
    ]:
        print(f"\n{'='*60}")
        print(f"Generating plots for mode={mode}")
        print(f"{'='*60}")

        success_agg = aggregate_success_by_lambda(success_rows, mode)
        fmd_map = build_fmd_lookup(fmd_rows, mode)

        if not success_agg:
            print(f"  No success data for mode={mode}, skipping.")
            continue

        plot_all(success_agg, fmd_map, mode, args.output_dir, baseline_fmd)

    # ── Save merged CSV for downstream use ──
    _save_merged_csv(success_rows, fmd_rows, args.output_dir)

    print(f"\nAll plots saved to {args.output_dir}")


def _save_merged_csv(success_rows, fmd_rows, output_dir):
    """Save a merged per-lambda CSV with success + FMD + quality."""
    # Build FMD lookups for both modes
    fmd_uncond = build_fmd_lookup(fmd_rows, "uncond")
    fmd_cond = build_fmd_lookup(fmd_rows, "cond")

    for mode, fmd_map in [("uncond", fmd_uncond), ("cond", fmd_cond)]:
        agg = aggregate_success_by_lambda(success_rows, mode)
        if not agg:
            continue

        rows = []
        for key, sa in sorted(agg.items()):
            m, s, lp, ld = key
            fmd_key = (_fmd_strategy_key(m, s), lp, ld)
            fmd_val = fmd_map.get(fmd_key)

            rows.append(
                {
                    "method": m,
                    "strategy": s,
                    "lambda_pitch": lp,
                    "lambda_duration": ld,
                    "n_samples": sa["n_samples"],
                    "both_success_rate": sa["both_rate"],
                    "pitch_success_rate": sa["pitch_rate"],
                    "dur_success_rate": sa["dur_rate"],
                    "fmd": fmd_val,
                    "avg_pce": sa["avg_pce"],
                    "avg_sc": sa["avg_sc"],
                    "avg_gc": sa["avg_gc"],
                    "avg_degradation": sa["avg_degradation"],
                }
            )

        csv_path = output_dir / f"merged_tradeoff_{mode}.csv"
        import csv as csv_mod

        fieldnames = [
            "method",
            "strategy",
            "lambda_pitch",
            "lambda_duration",
            "n_samples",
            "both_success_rate",
            "pitch_success_rate",
            "dur_success_rate",
            "fmd",
            "avg_pce",
            "avg_sc",
            "avg_gc",
            "avg_degradation",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {k: (v if v is not None else "") for k, v in row.items()}
                )
        print(f"Saved {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
