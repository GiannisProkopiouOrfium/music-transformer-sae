"""Effectiveness-vs-Degradation Pareto Plot for PID Steering.

Creates a figure showing that PID achieves the same steering effect
with less quality degradation, plotted across alpha values.

This replaces the error convergence plot (which requires >50 layers
to show clean PID dynamics) with a practical demonstration of PID's
advantage in the shallow-architecture setting.

Usage:
    python pid_steering/plot_pareto.py \
        --results_json exp/sod/pid_steering/experiments/conditioned/conditioned_pid_results.json
"""

import argparse
import json
import logging
import pathlib
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def load_results(results_path: pathlib.Path):
    """Load conditioned evaluation results."""
    with open(results_path) as f:
        return json.load(f)


def aggregate_by_alpha(concept_results, concept):
    """Group results by (method, alpha) and compute means."""
    groups = defaultdict(lambda: {"changes": [], "degradations": []})

    for r in concept_results:
        alpha = abs(r["alpha"])
        for method, m in r["methods"].items():
            key = (method, alpha)
            groups[key]["changes"].append(abs(m["attribute_change"]))

            deg = m.get("degradation", {})
            if isinstance(deg, dict):
                deg = deg.get("total_degradation", 0)
            groups[key]["degradations"].append(deg)

    aggregated = {}
    for (method, alpha), data in groups.items():
        aggregated[(method, alpha)] = {
            "effectiveness": np.mean(data["changes"]),
            "effectiveness_std": np.std(data["changes"]),
            "degradation": np.mean(data["degradations"]),
            "degradation_std": np.std(data["degradations"]),
            "n": len(data["changes"]),
        }
    return aggregated


def plot_pareto(results_data, output_dir: pathlib.Path):
    """Create effectiveness-vs-degradation Pareto plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not available")
        return

    concepts = list(results_data.keys())
    n_concepts = len(concepts)

    fig, axes = plt.subplots(1, n_concepts, figsize=(6 * n_concepts, 5))
    if n_concepts == 1:
        axes = [axes]

    method_styles = {
        "baseline": {"color": "gray", "marker": "x", "label": "Baseline"},
        "p_only": {"color": "#2196F3", "marker": "o", "label": "P (DiffMean)"},
        "pi": {"color": "#F44336", "marker": "s", "label": "PI"},
        "pid": {"color": "#4CAF50", "marker": "^", "label": "PID"},
    }

    for ax, concept in zip(axes, concepts):
        agg = aggregate_by_alpha(results_data[concept], concept)

        alphas = sorted(set(a for _, a in agg.keys()))
        methods = ["p_only", "pi", "pid"]

        for method in methods:
            style = method_styles.get(
                method, {"color": "black", "marker": ".", "label": method}
            )
            effs = []
            degs = []
            eff_stds = []
            deg_stds = []
            valid_alphas = []

            for alpha in alphas:
                key = (method, alpha)
                if key not in agg:
                    continue
                d = agg[key]
                effs.append(d["effectiveness"])
                degs.append(d["degradation"])
                eff_stds.append(d["effectiveness_std"])
                deg_stds.append(d["degradation_std"])
                valid_alphas.append(alpha)

            ax.errorbar(
                effs,
                degs,
                xerr=eff_stds,
                yerr=deg_stds,
                color=style["color"],
                marker=style["marker"],
                label=style["label"],
                linewidth=2,
                markersize=8,
                capsize=3,
                alpha=0.85,
            )

            # Annotate alpha values
            for eff, deg, alpha in zip(effs, degs, valid_alphas):
                ax.annotate(
                    f"α={alpha}",
                    (eff, deg),
                    textcoords="offset points",
                    xytext=(8, 5),
                    fontsize=7,
                    color=style["color"],
                    alpha=0.7,
                )

        # Ideal region arrow
        ax.annotate(
            "← Better",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=9,
            color="green",
            alpha=0.5,
        )

        concept_label = concept.replace("average_", "").replace("_", " ").title()
        ax.set_xlabel("|Attribute Change| (effectiveness)", fontsize=11)
        ax.set_ylabel("Quality Degradation", fontsize=11)
        ax.set_title(f"{concept_label}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle(
        "PID Steering: Effectiveness vs Quality Degradation",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "pareto_effectiveness_degradation.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Pareto plot to {out_path}")


def plot_degradation_by_alpha(results_data, output_dir: pathlib.Path):
    """Bar chart: degradation by method at each alpha (clearer for paper)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not available")
        return

    concepts = list(results_data.keys())
    n_concepts = len(concepts)

    fig, axes = plt.subplots(1, n_concepts, figsize=(7 * n_concepts, 4.5))
    if n_concepts == 1:
        axes = [axes]

    method_colors = {
        "p_only": "#2196F3",
        "pi": "#F44336",
        "pid": "#4CAF50",
    }
    method_labels = {
        "p_only": "P (DiffMean)",
        "pi": "PI",
        "pid": "PID",
    }

    for ax, concept in zip(axes, concepts):
        agg = aggregate_by_alpha(results_data[concept], concept)
        alphas = sorted(set(a for _, a in agg.keys() if a > 0))
        methods = ["p_only", "pi", "pid"]

        x = np.arange(len(alphas))
        width = 0.25

        for i, method in enumerate(methods):
            degs = []
            stds = []
            for alpha in alphas:
                key = (method, alpha)
                if key in agg:
                    degs.append(agg[key]["degradation"])
                    stds.append(agg[key]["degradation_std"])
                else:
                    degs.append(0)
                    stds.append(0)

            bars = ax.bar(
                x + i * width - width,
                degs,
                width,
                yerr=stds,
                label=method_labels.get(method, method),
                color=method_colors.get(method, "gray"),
                capsize=3,
                alpha=0.85,
            )

            # Add value labels on bars
            for bar, deg in zip(bars, degs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{deg:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        concept_label = concept.replace("average_", "").replace("_", " ").title()
        ax.set_xlabel("Steering Strength (|α|)", fontsize=11)
        ax.set_ylabel("Quality Degradation (↓ better)", fontsize=11)
        ax.set_title(f"{concept_label}", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in alphas])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        "Quality Degradation: P vs PI vs PID",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    out_path = output_dir / "degradation_by_alpha.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved degradation bar chart to {out_path}")


def print_summary_table(results_data):
    """Print a clean summary table for the paper."""
    for concept, concept_results in results_data.items():
        agg = aggregate_by_alpha(concept_results, concept)
        alphas = sorted(set(a for _, a in agg.keys() if a > 0))
        methods = ["p_only", "pi", "pid"]

        print(f"\n{'='*80}")
        print(f"  {concept.replace('_', ' ').title()} — Summary")
        print(f"{'='*80}")
        print(
            f"{'α':<6} {'Method':<10} {'|ΔAttr|':<12} {'Degrad':<12} {'Improvement':<12}"
        )
        print("-" * 52)

        for alpha in alphas:
            p_key = ("p_only", alpha)
            p_deg = agg[p_key]["degradation"] if p_key in agg else float("inf")

            for method in methods:
                key = (method, alpha)
                if key not in agg:
                    continue
                d = agg[key]
                improvement = ""
                if method != "p_only" and p_deg > 0:
                    pct = (p_deg - d["degradation"]) / p_deg * 100
                    improvement = f"{pct:+.0f}%"

                print(
                    f"{alpha:<6.1f} {method:<10} "
                    f"{d['effectiveness']:<12.1f} "
                    f"{d['degradation']:<12.2f} "
                    f"{improvement}"
                )
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Pareto and degradation plots from conditioned PID results"
    )
    parser.add_argument(
        "--results_json",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Output directory for plots (default: same as results_json parent)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    results_data = load_results(args.results_json)
    output_dir = args.output_dir or args.results_json.parent

    print_summary_table(results_data)
    plot_pareto(results_data, output_dir)
    plot_degradation_by_alpha(results_data, output_dir)


if __name__ == "__main__":
    main()
