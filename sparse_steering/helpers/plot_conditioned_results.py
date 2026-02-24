"""
Visualization script for SAS Conditioned Evaluation Results.

Generates publication-quality plots for the conditioned steering experiments:
  Fig 8  — Per-song response curves (pitch & duration side by side)
  Fig 9  — Effect sizes (Cohen's d) bar chart for both concepts
  Fig 10 — Conditioned steering summary: Δ from baseline at each λ
  Fig 11 — Control vs degradation trade-off for conditioned experiments

Usage:
    # From JSON result files (on server where experiments ran)
    python sparse_steering/helpers/plot_conditioned_results.py \
        --pitch_json exp/sod/sparse_steering/conditioned_evaluation/conditioned_results_average_pitch.json \
        --duration_json exp/sod/sparse_steering/conditioned_evaluation/conditioned_results_average_duration.json \
        --output_dir plots/conditioned/

    # Using hardcoded data (works anywhere)
    python sparse_steering/helpers/plot_conditioned_results.py \
        --output_dir plots/conditioned/
"""

import argparse
import json
import pathlib
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.gridspec import GridSpec

# ============================================================================
# Style Configuration
# ============================================================================

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
        "legend.fontsize": 9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

# Colorblind-friendly palette
SONG_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

LOW_COLOR = "#1f77b4"   # blue
HIGH_COLOR = "#d62728"  # red
BASELINE_COLOR = "#7f7f7f"


# ============================================================================
# Hardcoded data from experiment outputs
# ============================================================================


def get_conditioned_pitch_data() -> dict:
    """Conditioned pitch evaluation: 5 songs per category, layer 10.

    λ values: LOW gets [0.0, 0.25, 0.5, 0.75], HIGH gets [0.0, -0.25, -0.5, -0.75, -1.0, -1.5]
    """
    return {
        "concept": "average_pitch",
        "layers": [10],
        "conditioning_beats": 16,
        "per_song": {
            # LOW PITCH songs — steered with positive λ to push pitch UP
            "Musicalion-2538": {
                "category": "low",
                "initial": 34.4,
                "lambdas": [0.0, 0.25, 0.50, 0.75],
                "values": [39.1, 52.1, 54.1, 55.3],
                "monotonic": True,
                "r": 0.960,
            },
            "Musicalion-2829": {
                "category": "low",
                "initial": 38.0,
                "lambdas": [0.0, 0.25, 0.50, 0.75],
                "values": [68.3, 71.1, 64.8, 69.7],
                "monotonic": False,
                "r": 0.261,
            },
            "Musicalion-2841": {
                "category": "low",
                "initial": 37.6,
                "lambdas": [0.0, 0.25, 0.50, 0.75],
                "values": [56.2, 62.2, 73.5, 77.2],
                "monotonic": True,
                "r": 0.964,
            },
            "Musicalion-3393": {
                "category": "low",
                "initial": 38.2,
                "lambdas": [0.0, 0.25, 0.50, 0.75],
                "values": [71.4, 64.5, 73.4, 72.1],
                "monotonic": False,
                "r": 0.436,
            },
            "Musicalion-2828": {
                "category": "low",
                "initial": 37.2,
                "lambdas": [0.0, 0.25, 0.50, 0.75],
                "values": [50.2, 50.5, 60.3, 72.7],
                "monotonic": True,
                "r": 0.952,
            },
            # HIGH PITCH songs — steered with negative λ to push pitch DOWN
            "Kunstderfuge-389": {
                "category": "high",
                "initial": 81.0,
                "lambdas": [-1.50, -1.00, -0.75, -0.50, -0.25, 0.0],
                "values": [33.5, 35.1, 39.0, 57.7, 66.2, 81.0],
                "monotonic": True,
                "r": 0.974,
            },
            "Kunstderfuge-521": {
                "category": "high",
                "initial": 80.0,
                "lambdas": [-1.50, -1.00, -0.75, -0.50, -0.25, 0.0],
                "values": [35.2, 37.2, 44.6, 59.2, 73.6, 76.2],
                "monotonic": True,
                "r": 0.977,
            },
            "Musicalion-2841_high": {
                "category": "high",
                "initial": 78.0,
                "lambdas": [-1.50, -1.00, -0.75, -0.50, -0.25, 0.0],
                "values": [34.6, 36.9, 38.1, 62.3, 71.7, 74.6],
                "monotonic": True,
                "r": 0.953,
            },
            "Kunstderfuge-245": {
                "category": "high",
                "initial": 79.0,
                "lambdas": [-1.50, -1.00, -0.75, -0.50, -0.25, 0.0],
                "values": [39.8, 39.2, 46.4, 50.2, 63.4, 69.1],
                "monotonic": False,
                "r": 0.955,
            },
            "Kunstderfuge-115": {
                "category": "high",
                "initial": 80.5,
                "lambdas": [-1.50, -1.00, -0.75, -0.50, -0.25, 0.0],
                "values": [34.8, 37.2, 48.8, 54.4, 74.1, 73.0],
                "monotonic": False,
                "r": 0.824,
            },
        },
        "aggregate": {
            "low": {
                "r": 0.541,
                "p": 0.0008,
                "r2": 0.293,
                "slope": 28.99,
            },
            "high": {
                "r": 0.678,
                "p": 0.0000,
                "r2": 0.460,
                "slope": 28.53,
            },
        },
        "comparisons": {
            "low_pitch": {
                0.25: {"delta": 3.2, "success": True},
                0.50: {"delta": 8.1, "success": True},
                0.75: {"delta": 19.2, "success": True},
            },
            "high_pitch": {
                -0.25: {"delta": -7.2, "success": True},
                -0.50: {"delta": -14.5, "success": True},
                -0.75: {"delta": -25.6, "success": True},
                -1.00: {"delta": -34.5, "success": True},
                -1.50: {"delta": -47.3, "success": True},
            },
        },
        "effect_sizes": {
            "low": {
                0.25: {"d": 0.328, "change": 3.18},
                0.50: {"d": 1.190, "change": 11.61},
                0.75: {"d": 2.310, "change": 19.24},
            },
            "high": {
                -0.25: {"d": -0.767, "change": -7.21},
                -0.50: {"d": -2.427, "change": -14.52},
                -0.75: {"d": -4.264, "change": -25.56},
                -1.00: {"d": -6.485, "change": -34.50},
                -1.50: {"d": -9.102, "change": -47.34},
            },
        },
    }


def get_conditioned_duration_data() -> dict:
    """Conditioned duration evaluation: 5 songs per category, layer 10.

    λ values: LOW gets [0.0, 0.5, 1.0, 1.5], HIGH gets [0.0, -0.5, -1.0, -1.5]
    """
    return {
        "concept": "average_duration",
        "layers": [10],
        "conditioning_beats": 16,
        "per_song": {
            # LOW DURATION songs — steered with positive λ to push duration UP
            "Kunstderfuge-1306": {
                "category": "low",
                "initial": 1.30,
                "lambdas": [0.0, 0.5, 1.0, 1.5],
                "values": [1.48, 3.87, 40.97, 71.90],
                "monotonic": True,
                "r": 0.950,
            },
            "Musicalion-962": {
                "category": "low",
                "initial": 1.86,
                "lambdas": [0.0, 0.5, 1.0, 1.5],
                "values": [3.07, 4.52, 12.42, 43.85],
                "monotonic": True,
                "r": 0.970,
            },
            "Kunstderfuge-963": {
                "category": "low",
                "initial": 1.81,
                "lambdas": [0.0, 0.5, 1.0, 1.5],
                "values": [1.22, 3.35, 67.40, 86.82],
                "monotonic": True,
                "r": 0.943,
            },
            "Musicalion-2882": {
                "category": "low",
                "initial": 1.25,
                "lambdas": [0.0, 0.5, 1.0, 1.5],
                "values": [3.92, 5.30, 50.47, 88.97],
                "monotonic": True,
                "r": 0.939,
            },
            "Kunstderfuge-766": {
                "category": "low",
                "initial": 1.78,
                "lambdas": [0.0, 0.5, 1.0, 1.5],
                "values": [1.87, 4.65, 62.47, 38.53],
                "monotonic": False,
                "r": 0.924,
            },
            # HIGH DURATION songs — steered with negative λ to push duration DOWN
            "Musicalion-4187": {
                "category": "high",
                "initial": 29.23,
                "lambdas": [-1.5, -1.0, -0.5, 0.0],
                "values": [3.25, 88.40, 343.18, 299.11],
                "monotonic": False,
                "r": 0.860,
            },
            "Musicalion-3512": {
                "category": "high",
                "initial": 32.00,
                "lambdas": [-1.5, -1.0, -0.5, 0.0],
                "values": [3.82, 4.55, 10.48, 47.11],
                "monotonic": True,
                "r": 0.803,
            },
            "Musicalion-1416": {
                "category": "high",
                "initial": 31.25,
                "lambdas": [-1.5, -1.0, -0.5, 0.0],
                "values": [3.07, 3.82, 4.21, 338.66],
                "monotonic": True,
                "r": 0.801,
            },
            "Musicalion-1413": {
                "category": "high",
                "initial": 28.67,
                "lambdas": [-1.5, -1.0, -0.5, 0.0],
                "values": [4.20, 4.43, 15.97, 38.61],
                "monotonic": True,
                "r": 0.786,
            },
            "Musicalion-1698": {
                "category": "high",
                "initial": 30.00,
                "lambdas": [-1.5, -1.0, -0.5, 0.0],
                "values": [3.73, 3.76, 11.19, 327.63],
                "monotonic": True,
                "r": 0.784,
            },
        },
        "aggregate": {
            "low": {
                "r": 0.871,
                "p": 0.0000,
                "r2": 0.758,
                "slope": 46.70,
            },
            "high": {
                "r": 0.608,
                "p": 0.0045,
                "r2": 0.369,
                "slope": 138.49,
            },
        },
        "comparisons": {
            "low_duration": {
                0.50: {"delta": 2.03, "success": True},
                1.00: {"delta": 44.44, "success": True},
                1.50: {"delta": 63.70, "success": True},
            },
            "high_duration": {
                -0.50: {"delta": -133.21, "success": True},
                -1.00: {"delta": -205.84, "success": True},
                -1.50: {"delta": -206.61, "success": True},
            },
        },
        "effect_sizes": {
            "low": {
                0.50: {"d": 0.816, "change": 2.03},
                1.00: {"d": 4.820, "change": 44.44},
                1.50: {"d": 3.801, "change": 63.70},
            },
            "high": {
                -0.50: {"d": -0.881, "change": -133.21},
                -1.00: {"d": -1.896, "change": -205.84},
                -1.50: {"d": -1.904, "change": -206.61},
            },
        },
    }


# ============================================================================
# Data loading helpers
# ============================================================================


def load_from_json(json_path: pathlib.Path) -> Optional[dict]:
    """Load conditioned results from a JSON file and reshape for plotting."""
    if not json_path.exists():
        return None

    with open(json_path) as f:
        raw = json.load(f)

    stat = raw.get("statistical_analysis", {})
    per_song = stat.get("per_song", {})
    aggregate = stat.get("aggregate", {})
    results = raw.get("results", [])

    # Determine concept
    concept = "average_pitch"
    for r in results:
        if "duration" in r.get("category", ""):
            concept = "average_duration"
            break

    is_duration = "duration" in concept

    # Build per_song structure
    song_data = {}
    for r in results:
        if r.get("generated_n_notes", 0) == 0:
            continue
        name = r["song_name"]
        cat = "low" if "low" in r["category"] else "high"
        if name not in song_data:
            song_data[name] = {
                "category": cat,
                "initial": r.get(
                    "initial_duration" if is_duration else "initial_pitch", 0
                ),
                "lambdas": [],
                "values": [],
            }
        song_data[name]["lambdas"].append(r["lambda"])
        song_data[name]["values"].append(
            r["generated_mean_duration"] if is_duration else r["generated_mean_pitch"]
        )

    # Sort lambdas for each song
    for name, data in song_data.items():
        idx = np.argsort(data["lambdas"])
        data["lambdas"] = [data["lambdas"][i] for i in idx]
        data["values"] = [data["values"][i] for i in idx]

        # Compute per-song r and monotonicity
        if name in per_song:
            data["r"] = per_song[name].get("pearson_r", 0)
            data["monotonic"] = per_song[name].get("monotonic", False)
        else:
            from scipy import stats as scipy_stats

            lams = np.array(data["lambdas"])
            vals = np.array(data["values"])
            if len(lams) >= 3:
                r_val, _ = scipy_stats.pearsonr(lams, vals)
                data["r"] = float(r_val)
            else:
                data["r"] = 0.0
            data["monotonic"] = all(
                vals[i] <= vals[i + 1] for i in range(len(vals) - 1)
            )

    # Build comparisons from analysis
    analysis = raw.get("analysis", {})
    comparisons = {}
    for comp_key, comp in analysis.get("comparisons", {}).items():
        cat = comp["category"]
        lam = comp["lambda"]
        if is_duration:
            delta = comp.get("duration_difference", 0)
        else:
            delta = comp.get("pitch_difference", 0)
        comparisons.setdefault(cat, {})[lam] = {
            "delta": delta,
            "success": comp.get("success", False),
        }

    # Build effect sizes
    effect_sizes = {}
    for cat_key in ["low", "high"]:
        cat_agg = aggregate.get(cat_key, {})
        es_data = cat_agg.get("effect_sizes", {})
        effect_sizes[cat_key] = {}
        for lam_str, es in es_data.items():
            lam_val = float(lam_str)
            effect_sizes[cat_key][lam_val] = {
                "d": es.get("cohens_d", 0),
                "change": es.get("change_from_baseline", 0),
            }

    # Build aggregate
    agg = {}
    for cat_key in ["low", "high"]:
        cat_agg = aggregate.get(cat_key, {})
        agg[cat_key] = {
            "r": cat_agg.get("pearson_r", 0),
            "p": cat_agg.get("pearson_p", 1),
            "r2": cat_agg.get("linear_r2", 0),
            "slope": cat_agg.get("linear_slope", 0),
        }

    return {
        "concept": concept,
        "layers": [10],
        "conditioning_beats": 16,
        "per_song": song_data,
        "aggregate": agg,
        "comparisons": comparisons,
        "effect_sizes": effect_sizes,
    }


# ============================================================================
# Fig 8: Per-Song Response Curves (Pitch & Duration)
# ============================================================================


def plot_per_song_curves(
    pitch_data: dict, duration_data: dict, output_dir: pathlib.Path
):
    """Per-song steering response curves for conditioned experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    datasets = [
        (pitch_data, "Pitch (MIDI)", "semitones", axes[0]),
        (duration_data, "Duration (ticks)", "ticks", axes[1]),
    ]

    for data, metric_label, unit, (ax_low, ax_high) in datasets:
        per_song = data["per_song"]

        # Separate by category
        low_songs = {
            k: v for k, v in per_song.items() if v["category"] == "low"
        }
        high_songs = {
            k: v for k, v in per_song.items() if v["category"] == "high"
        }

        # --- LOW songs (positive λ) ---
        for i, (name, sdata) in enumerate(
            sorted(low_songs.items(), key=lambda x: abs(x[1]["r"]), reverse=True)
        ):
            color = SONG_COLORS[i % len(SONG_COLORS)]
            mono_marker = "✓" if sdata.get("monotonic") else ""
            lw = 2.2 if sdata.get("monotonic") else 1.5
            ax_low.plot(
                sdata["lambdas"],
                sdata["values"],
                marker="o",
                markersize=6,
                linewidth=lw,
                color=color,
                label=f"{name} (r={sdata['r']:+.2f}) {mono_marker}",
            )
            # Show initial value as dashed horizontal
            ax_low.axhline(
                sdata["initial"], color=color, ls=":", alpha=0.3, linewidth=0.8
            )

        ax_low.axvline(0, color="grey", ls="--", alpha=0.4)
        ax_low.set_xlabel("Steering Strength (λ)")
        ax_low.set_ylabel(f"Generated {metric_label}")
        ax_low.set_title(f"LOW {metric_label} Songs → Push UP")
        ax_low.legend(fontsize=8, loc="upper left", framealpha=0.9)

        # --- HIGH songs (negative λ) ---
        for i, (name, sdata) in enumerate(
            sorted(high_songs.items(), key=lambda x: abs(x[1]["r"]), reverse=True)
        ):
            color = SONG_COLORS[i % len(SONG_COLORS)]
            mono_marker = "✓" if sdata.get("monotonic") else ""
            lw = 2.2 if sdata.get("monotonic") else 1.5
            ax_high.plot(
                sdata["lambdas"],
                sdata["values"],
                marker="s",
                markersize=6,
                linewidth=lw,
                color=color,
                label=f"{name} (r={sdata['r']:+.2f}) {mono_marker}",
            )
            ax_high.axhline(
                sdata["initial"], color=color, ls=":", alpha=0.3, linewidth=0.8
            )

        ax_high.axvline(0, color="grey", ls="--", alpha=0.4)
        ax_high.set_xlabel("Steering Strength (λ)")
        ax_high.set_ylabel(f"Generated {metric_label}")
        ax_high.set_title(f"HIGH {metric_label} Songs → Push DOWN")
        ax_high.legend(fontsize=8, loc="upper right", framealpha=0.9)

    fig.suptitle(
        "Conditioned SAS Steering: Per-Song Response Curves (Layer 10)",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "fig8_conditioned_per_song_curves.png")
    fig.savefig(output_dir / "fig8_conditioned_per_song_curves.pdf")
    plt.close(fig)
    print("  Saved fig8_conditioned_per_song_curves.png/pdf")


# ============================================================================
# Fig 9: Effect Sizes (Cohen's d) Bar Chart
# ============================================================================


def plot_effect_sizes(
    pitch_data: dict, duration_data: dict, output_dir: pathlib.Path
):
    """Cohen's d effect size bar chart for both concepts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    datasets = [
        (pitch_data, "Pitch", axes[0]),
        (duration_data, "Duration", axes[1]),
    ]

    for data, concept_label, ax in datasets:
        all_lambdas = []
        all_d = []
        all_colors = []
        all_labels = []

        for cat in ["low", "high"]:
            es = data.get("effect_sizes", {}).get(cat, {})
            for lam_val in sorted(es.keys()):
                d_val = es[lam_val]["d"]
                all_lambdas.append(lam_val)
                all_d.append(d_val)
                all_colors.append(LOW_COLOR if cat == "low" else HIGH_COLOR)
                cat_label = "LOW→UP" if cat == "low" else "HIGH→DOWN"
                all_labels.append(f"{cat_label}\nλ={lam_val:+.2f}")

        x = np.arange(len(all_d))
        bars = ax.bar(x, all_d, color=all_colors, edgecolor="white", linewidth=0.5)

        # Annotate
        for i, (bar, d_val) in enumerate(zip(bars, all_d)):
            va = "bottom" if d_val >= 0 else "top"
            offset = 0.15 if d_val >= 0 else -0.15
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                f"d={d_val:+.1f}",
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
            )

        # Reference lines for effect size thresholds
        for thresh, label in [(0.8, "Large"), (-0.8, "Large")]:
            ax.axhline(thresh, color="grey", ls=":", alpha=0.4)

        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, fontsize=8)
        ax.set_ylabel("Cohen's d (effect size)")
        ax.set_title(f"{concept_label} — Effect Sizes vs Baseline (λ=0)")
        ax.axhline(0, color="black", linewidth=0.8)

        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=LOW_COLOR, label="Low songs (push UP)"),
            Patch(facecolor=HIGH_COLOR, label="High songs (push DOWN)"),
        ]
        ax.legend(handles=legend_elements, loc="best", fontsize=9)

    fig.suptitle(
        "Conditioned Steering Effect Sizes (Cohen's d, Layer 10)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "fig9_conditioned_effect_sizes.png")
    fig.savefig(output_dir / "fig9_conditioned_effect_sizes.pdf")
    plt.close(fig)
    print("  Saved fig9_conditioned_effect_sizes.png/pdf")


# ============================================================================
# Fig 10: Conditioned Steering Summary — Δ from baseline
# ============================================================================


def plot_conditioned_summary(
    pitch_data: dict, duration_data: dict, output_dir: pathlib.Path
):
    """2x2 summary: mean Δ from baseline at each λ, with success markers."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    datasets = [
        (pitch_data, "Pitch", "semitones", axes[0]),
        (duration_data, "Duration", "ticks", axes[1]),
    ]

    for data, concept_label, unit, (ax_low, ax_high) in datasets:
        comps = data.get("comparisons", {})

        for cat_key, ax, color, direction in [
            (
                [k for k in comps if "low" in k][0] if any("low" in k for k in comps) else None,
                ax_low,
                LOW_COLOR,
                "LOW → Push UP",
            ),
            (
                [k for k in comps if "high" in k][0] if any("high" in k for k in comps) else None,
                ax_high,
                HIGH_COLOR,
                "HIGH → Push DOWN",
            ),
        ]:
            if cat_key is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            cat_comps = comps[cat_key]
            lambdas = sorted(cat_comps.keys(), key=float)
            deltas = [cat_comps[l]["delta"] for l in lambdas]
            successes = [cat_comps[l]["success"] for l in lambdas]

            bars = ax.bar(
                range(len(lambdas)),
                deltas,
                color=[color if s else "#bdbdbd" for s in successes],
                edgecolor="white",
                linewidth=0.5,
            )

            # Annotate with delta values
            for i, (bar, delta, success) in enumerate(
                zip(bars, deltas, successes)
            ):
                va = "bottom" if delta >= 0 else "top"
                offset = abs(delta) * 0.05 if abs(delta) > 5 else 1
                offset = offset if delta >= 0 else -offset
                marker = "✓" if success else "✗"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"Δ={delta:+.1f}\n{marker}",
                    ha="center",
                    va=va,
                    fontsize=9,
                    fontweight="bold",
                )

            ax.set_xticks(range(len(lambdas)))
            ax.set_xticklabels([f"λ={float(l):+.2f}" for l in lambdas], fontsize=10)
            ax.set_ylabel(f"Δ {concept_label} ({unit})")
            ax.set_title(f"{concept_label}: {direction}")
            ax.axhline(0, color="black", linewidth=0.8)

    fig.suptitle(
        "Conditioned SAS Steering: Mean Change from Baseline (All Comparisons = SUCCESS)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "fig10_conditioned_delta_summary.png")
    fig.savefig(output_dir / "fig10_conditioned_delta_summary.pdf")
    plt.close(fig)
    print("  Saved fig10_conditioned_delta_summary.png/pdf")


# ============================================================================
# Fig 11: SAS vs DiffMean comparison (unconditional)
# ============================================================================


def plot_sas_vs_diffmean(output_dir: pathlib.Path):
    """Side-by-side comparison of SAS and DiffMean unconditional steering."""

    # DiffMean data (from user's LaTeX tables)
    diffmean_pitch = {
        "alphas": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        "means": [36.73, 37.44, 38.07, 42.98, 65.73, 76.31, 77.27, 80.29, 81.23],
        "degs": [0.25, 0.66, 1.58, 0.24, 0.06, 1.25, 2.92, 0.19, 2.01],
        "r": 0.903,
        "R2": 0.815,
        "slope": 13.35,
    }
    diffmean_duration = {
        "alphas": [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        "means": [3.08, 3.06, 3.10, 3.42, 7.51, 25.38, 34.09, 36.10, 38.06],
        "degs": [0.90, 3.97, 1.07, 9.46, 1.81, 0.03, 0.39, 1.72, 1.97],
        "r": 0.926,
        "R2": 0.858,
        "slope": 10.77,
    }

    # SAS data (from refined experiments, single_10)
    sas_pitch = {
        "lambdas": [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
        "means": [39.29, 40.05, 43.93, 57.93, 64.00, 67.61, 67.59, 74.78, 74.69],
        "degs": [1.70, 3.50, 1.02, 0.52, 0.24, 0.30, 0.96, 1.31, 1.49],
        "r": 0.958,
        "R2": 0.918,
        "slope": 18.72,
    }
    sas_duration = {
        "lambdas": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
        "means": [3.53, 4.14, 5.14, 9.09, 20.77, 43.57, 56.49],
        "degs": [0.33, 0.19, 3.43, 0.31, 0.33, 1.81, 0.14],
        "r": 0.913,
        "R2": 0.834,
        "slope": 18.10,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Row 1: Response curves ---
    # Pitch
    ax = axes[0, 0]
    ax.plot(
        diffmean_pitch["alphas"],
        diffmean_pitch["means"],
        "s--",
        color="#d62728",
        markersize=6,
        linewidth=1.8,
        label=f"DiffMean (r={diffmean_pitch['r']:+.3f}, R²={diffmean_pitch['R2']:.3f})",
    )
    ax.plot(
        sas_pitch["lambdas"],
        sas_pitch["means"],
        "o-",
        color="#1f77b4",
        markersize=7,
        linewidth=2.2,
        label=f"SAS L10 (r={sas_pitch['r']:+.3f}, R²={sas_pitch['R2']:.3f})",
    )
    ax.set_xlabel("Steering Strength (α / λ)")
    ax.set_ylabel("Mean Pitch (MIDI)")
    ax.set_title("Pitch Steering Response")
    ax.legend(loc="lower right", fontsize=9)

    # Duration
    ax = axes[0, 1]
    ax.plot(
        diffmean_duration["alphas"],
        diffmean_duration["means"],
        "s--",
        color="#d62728",
        markersize=6,
        linewidth=1.8,
        label=f"DiffMean (r={diffmean_duration['r']:+.3f}, R²={diffmean_duration['R2']:.3f})",
    )
    ax.plot(
        sas_duration["lambdas"],
        sas_duration["means"],
        "o-",
        color="#1f77b4",
        markersize=7,
        linewidth=2.2,
        label=f"SAS L10 (r={sas_duration['r']:+.3f}, R²={sas_duration['R2']:.3f})",
    )
    ax.set_xlabel("Steering Strength (α / λ)")
    ax.set_ylabel("Mean Duration (ticks)")
    ax.set_title("Duration Steering Response")
    ax.legend(loc="upper left", fontsize=9)

    # --- Row 2: Degradation ---
    # Pitch degradation
    ax = axes[1, 0]
    x_dm = np.arange(len(diffmean_pitch["alphas"]))
    x_sas = np.arange(len(sas_pitch["lambdas"]))
    width = 0.35
    ax.bar(
        x_dm - width / 2,
        diffmean_pitch["degs"],
        width,
        color="#d62728",
        alpha=0.7,
        label=f"DiffMean (avg={np.mean(diffmean_pitch['degs']):.2f})",
    )
    # Offset SAS bars to align λ=0 with α=0
    dm_zero_idx = diffmean_pitch["alphas"].index(0.0)
    sas_zero_idx = sas_pitch["lambdas"].index(0.0)
    offset = dm_zero_idx - sas_zero_idx
    ax.bar(
        x_sas + offset + width / 2,
        sas_pitch["degs"],
        width,
        color="#1f77b4",
        alpha=0.7,
        label=f"SAS L10 (avg={np.mean(sas_pitch['degs']):.2f})",
    )
    ax.set_xticks(x_dm)
    ax.set_xticklabels(
        [f"{a:+.1f}" for a in diffmean_pitch["alphas"]], fontsize=9
    )
    ax.set_xlabel("Steering Strength")
    ax.set_ylabel("Total Degradation")
    ax.set_title("Pitch Quality Degradation")
    ax.legend(fontsize=9)
    ax.axhline(2.0, color="grey", ls=":", alpha=0.5, label="deg=2.0")

    # Duration degradation
    ax = axes[1, 1]
    x_dm = np.arange(len(diffmean_duration["alphas"]))
    x_sas = np.arange(len(sas_duration["lambdas"]))
    dm_zero_idx = diffmean_duration["alphas"].index(0.0)
    sas_zero_idx = sas_duration["lambdas"].index(0.0)
    offset = dm_zero_idx - sas_zero_idx
    ax.bar(
        x_dm - width / 2,
        diffmean_duration["degs"],
        width,
        color="#d62728",
        alpha=0.7,
        label=f"DiffMean (avg={np.mean(diffmean_duration['degs']):.2f})",
    )
    ax.bar(
        x_sas + offset + width / 2,
        sas_duration["degs"],
        width,
        color="#1f77b4",
        alpha=0.7,
        label=f"SAS L10 (avg={np.mean(sas_duration['degs']):.2f})",
    )
    ax.set_xticks(x_dm)
    ax.set_xticklabels(
        [f"{a:+.1f}" for a in diffmean_duration["alphas"]], fontsize=9
    )
    ax.set_xlabel("Steering Strength")
    ax.set_ylabel("Total Degradation")
    ax.set_title("Duration Quality Degradation")
    ax.legend(fontsize=9)
    ax.axhline(2.0, color="grey", ls=":", alpha=0.5)

    fig.suptitle(
        "SAS (Layer 10) vs DiffMean: Steering Effectiveness & Quality",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "fig11_sas_vs_diffmean.png")
    fig.savefig(output_dir / "fig11_sas_vs_diffmean.pdf")
    plt.close(fig)
    print("  Saved fig11_sas_vs_diffmean.png/pdf")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Plot SAS conditioned evaluation results"
    )
    parser.add_argument(
        "--pitch_json",
        type=pathlib.Path,
        default=None,
        help="Path to conditioned_results_average_pitch.json",
    )
    parser.add_argument(
        "--duration_json",
        type=pathlib.Path,
        default=None,
        help="Path to conditioned_results_average_duration.json",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("plots/conditioned"),
        help="Output directory for plots",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SAS Conditioned Evaluation — Plot Generation")
    print("=" * 60)

    # Load data: prefer JSON, fall back to hardcoded
    if args.pitch_json and args.pitch_json.exists():
        print(f"Loading pitch data from: {args.pitch_json}")
        pitch_data = load_from_json(args.pitch_json)
    else:
        print("Using hardcoded pitch data")
        pitch_data = get_conditioned_pitch_data()

    if args.duration_json and args.duration_json.exists():
        print(f"Loading duration data from: {args.duration_json}")
        duration_data = load_from_json(args.duration_json)
    else:
        print("Using hardcoded duration data")
        duration_data = get_conditioned_duration_data()

    print(f"Output directory: {args.output_dir}")
    print()

    # Generate all plots
    print("Generating plots...")

    print("\n[1/4] Per-song response curves")
    plot_per_song_curves(pitch_data, duration_data, args.output_dir)

    print("\n[2/4] Effect sizes (Cohen's d)")
    plot_effect_sizes(pitch_data, duration_data, args.output_dir)

    print("\n[3/4] Conditioned Δ summary")
    plot_conditioned_summary(pitch_data, duration_data, args.output_dir)

    print("\n[4/4] SAS vs DiffMean comparison")
    plot_sas_vs_diffmean(args.output_dir)

    print("\n" + "=" * 60)
    print("All plots saved!")
    print(f"Output: {args.output_dir.absolute()}")
    print("=" * 60)

    print("\nRecommended figure order in presentation:")
    print("  1. fig11 — SAS vs DiffMean comparison (motivate SAS)")
    print("  2. fig8  — Per-song conditioned curves (show individual songs)")
    print("  3. fig9  — Effect sizes (quantify impact)")
    print("  4. fig10 — Δ summary (all comparisons = SUCCESS)")


if __name__ == "__main__":
    main()
