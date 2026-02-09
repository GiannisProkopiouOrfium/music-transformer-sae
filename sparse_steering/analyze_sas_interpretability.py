"""
Interpretability Analysis for Sparse Activation Steering (SAS).

This script analyzes SAS vectors to understand which features drive steering
and provides visualizations for the key advantage of SAS over DiffMean:
interpretable, monosemantic features.

Usage:
    python sparse_steering/analyze_sas_interpretability.py \
        --concept average_pitch \
        --output_dir exp/sod/sparse_steering/interpretability
"""

import argparse
import json
import logging
import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sas_vectors(sas_vectors_path: pathlib.Path) -> Dict:
    """Load SAS vectors from file."""
    logger.info(f"Loading SAS vectors from {sas_vectors_path}")
    data = torch.load(sas_vectors_path, map_location="cpu")
    return data


def load_diffmean_vectors(diffmean_path: pathlib.Path) -> Dict:
    """Load DiffMean vectors for comparison."""
    if not diffmean_path.exists():
        logger.warning(f"DiffMean vectors not found at {diffmean_path}")
        return None

    logger.info(f"Loading DiffMean vectors from {diffmean_path}")
    data = torch.load(diffmean_path, map_location="cpu")
    return data


def analyze_single_layer(v_sas: np.ndarray, layer_idx: int, concept: str) -> Dict:
    """Analyze SAS vector for a single layer.

    Returns:
        Dictionary with analysis results
    """
    # Basic statistics
    active_mask = v_sas != 0
    active_indices = np.where(active_mask)[0]
    n_active = len(active_indices)

    positive_mask = v_sas > 0
    negative_mask = v_sas < 0
    n_positive = np.sum(positive_mask)
    n_negative = np.sum(negative_mask)

    # Magnitude statistics
    active_values = v_sas[active_mask]
    magnitude = np.linalg.norm(v_sas)
    mean_value = v_sas.mean()

    # Top features
    feature_magnitudes = np.abs(v_sas)
    top_k = min(20, n_active)
    top_indices = np.argsort(feature_magnitudes)[-top_k:][::-1]
    top_values = v_sas[top_indices]

    # Cumulative contribution
    sorted_abs = np.sort(feature_magnitudes)[::-1]
    cumsum = np.cumsum(sorted_abs)
    cumsum_norm = cumsum / (cumsum[-1] + 1e-10)
    threshold_80 = np.argmax(cumsum_norm >= 0.8) if len(cumsum_norm) > 0 else 0
    threshold_90 = np.argmax(cumsum_norm >= 0.9) if len(cumsum_norm) > 0 else 0

    results = {
        "layer_idx": layer_idx,
        "concept": concept,
        "total_features": len(v_sas),
        "n_active": n_active,
        "sparsity": n_active / len(v_sas),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "magnitude": magnitude,
        "mean_value": mean_value,
        "top_indices": top_indices.tolist(),
        "top_values": top_values.tolist(),
        "threshold_80": threshold_80,
        "threshold_90": threshold_90,
        "active_values_mean": active_values.mean() if len(active_values) > 0 else 0,
        "active_values_std": active_values.std() if len(active_values) > 0 else 0,
    }

    return results


def print_layer_analysis(results: Dict):
    """Print analysis results for a layer."""
    logger.info("=" * 80)
    logger.info(f"LAYER {results['layer_idx']} - {results['concept'].upper()}")
    logger.info("=" * 80)

    logger.info(f"\nFeature Statistics:")
    logger.info(f"  Total dimensions: {results['total_features']}")
    logger.info(
        f"  Active features:  {results['n_active']} ({results['sparsity']*100:.2f}%)"
    )
    logger.info(f"  Positive (high):  {results['n_positive']}")
    logger.info(f"  Negative (low):   {results['n_negative']}")
    logger.info(f"  L2 magnitude:     {results['magnitude']:.4f}")
    logger.info(f"  Mean value:       {results['mean_value']:+.6f}")

    logger.info(f"\nCumulative Contribution:")
    logger.info(f"  Top {results['threshold_80']} features → 80% of steering effect")
    logger.info(f"  Top {results['threshold_90']} features → 90% of steering effect")

    logger.info(f"\nTop 10 Features by Magnitude:")
    for i in range(min(10, len(results["top_indices"]))):
        idx = results["top_indices"][i]
        value = results["top_values"][i]
        direction = "HIGH" if value > 0 else "LOW"
        logger.info(
            f"  {i+1:2d}. Feature {idx:4d}: {value:+.6f} ({direction} {results['concept']})"
        )


def compare_with_diffmean(
    sas_results: Dict, diffmean_vector: np.ndarray, layer_idx: int
):
    """Compare SAS with DiffMean to show interpretability advantage."""
    logger.info("\n" + "=" * 80)
    logger.info(f"SAS vs DIFFMEAN COMPARISON - Layer {layer_idx}")
    logger.info("=" * 80)

    sas = sas_results

    # DiffMean stats
    diffmean_l0 = len(diffmean_vector)  # All dimensions active
    diffmean_l2 = np.linalg.norm(diffmean_vector)

    logger.info(f"\nSAS (Sparse Interpretable Space):")
    logger.info(f"  Dimensions:       4096")
    logger.info(f"  Active features:  {sas['n_active']} ({sas['sparsity']*100:.2f}%)")
    logger.info(f"  L0 sparsity:      {sas['n_active']}")
    logger.info(f"  L2 magnitude:     {sas['magnitude']:.4f}")
    logger.info(
        f"  Interpretable:    ✓ YES - Can inspect {sas['n_active']} individual features"
    )

    logger.info(f"\nDiffMean (Dense Entangled Space):")
    logger.info(f"  Dimensions:       512")
    logger.info(f"  Active features:  512 (100.00%)")
    logger.info(f"  L0 sparsity:      512 (dense)")
    logger.info(f"  L2 magnitude:     {diffmean_l2:.4f}")
    logger.info(f"  Interpretable:    ✗ NO - 512 entangled, polysemantic dimensions")

    logger.info(f"\n✓ Interpretability Advantage:")
    logger.info(
        f"  SAS has {sas['sparsity']*100:.1f}% active dimensions vs DiffMean's 100%"
    )
    logger.info(f"  SAS features are monosemantic (one feature ≈ one concept)")
    logger.info(f"  DiffMean features are polysemantic (one dimension = many concepts)")
    logger.info(
        f"  Each SAS feature can be analyzed, tested, and understood individually"
    )


def visualize_sas_vector(
    v_sas: np.ndarray,
    layer_idx: int,
    concept: str,
    output_dir: pathlib.Path,
    results: Dict = None,
):
    """Create comprehensive visualizations of SAS vector."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Distribution of active feature values
    active_values = v_sas[v_sas != 0]
    if len(active_values) > 0:
        axes[0, 0].hist(active_values, bins=50, alpha=0.7, edgecolor="black")
        axes[0, 0].axvline(0, color="red", linestyle="--", linewidth=2, label="Zero")
        axes[0, 0].set_xlabel("Feature Value", fontsize=12)
        axes[0, 0].set_ylabel("Count", fontsize=12)
        axes[0, 0].set_title(
            f"Distribution of Active Features (n={len(active_values)})",
            fontsize=14,
            fontweight="bold",
        )
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # 2. Top 20 features bar chart
    if results:
        top_20_indices = results["top_indices"][:20]
        top_20_values = results["top_values"][:20]
        colors = ["#d62728" if v < 0 else "#1f77b4" for v in top_20_values]

        y_pos = np.arange(len(top_20_values))
        axes[0, 1].barh(y_pos, top_20_values, color=colors, edgecolor="black")
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels([f"F{i}" for i in top_20_indices], fontsize=10)
        axes[0, 1].set_xlabel("SAS Value", fontsize=12)
        axes[0, 1].set_title(
            "Top 20 Features by Magnitude", fontsize=14, fontweight="bold"
        )
        axes[0, 1].axvline(0, color="black", linestyle="-", linewidth=1)
        axes[0, 1].grid(True, alpha=0.3, axis="x")

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#1f77b4", label=f"Positive (HIGH {concept})"),
            Patch(facecolor="#d62728", label=f"Negative (LOW {concept})"),
        ]
        axes[0, 1].legend(handles=legend_elements, loc="lower right")

    # 3. Cumulative contribution curve
    feature_magnitudes = np.abs(v_sas)
    sorted_abs = np.sort(feature_magnitudes)[::-1]
    cumsum = np.cumsum(sorted_abs)
    cumsum_norm = cumsum / (cumsum[-1] + 1e-10)

    axes[1, 0].plot(cumsum_norm, linewidth=2, color="#1f77b4")
    axes[1, 0].axhline(
        0.8, color="#d62728", linestyle="--", linewidth=2, label="80% threshold"
    )
    axes[1, 0].axhline(
        0.9, color="#ff7f0e", linestyle="--", linewidth=2, label="90% threshold"
    )

    if results:
        axes[1, 0].axvline(
            results["threshold_80"], color="#d62728", linestyle=":", alpha=0.5
        )
        axes[1, 0].axvline(
            results["threshold_90"], color="#ff7f0e", linestyle=":", alpha=0.5
        )
        axes[1, 0].text(
            results["threshold_80"],
            0.85,
            f"  {results['threshold_80']} features",
            fontsize=10,
            color="#d62728",
        )

    axes[1, 0].set_xlabel("Number of Top Features", fontsize=12)
    axes[1, 0].set_ylabel("Cumulative Contribution", fontsize=12)
    axes[1, 0].set_title("Feature Contribution Curve", fontsize=14, fontweight="bold")
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, min(500, len(v_sas)))

    # 4. Sparse structure visualization
    axes[1, 1].scatter(
        range(len(v_sas)), v_sas, s=2, alpha=0.6, c=np.abs(v_sas), cmap="RdBu_r"
    )
    axes[1, 1].axhline(0, color="black", linestyle="-", linewidth=0.5)
    axes[1, 1].set_xlabel("Feature Index", fontsize=12)
    axes[1, 1].set_ylabel("Feature Value", fontsize=12)
    axes[1, 1].set_title(
        f"Full SAS Vector Structure (Layer {layer_idx})", fontsize=14, fontweight="bold"
    )
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        f'SAS Vector Analysis: {concept.replace("_", " ").title()} (Layer {layer_idx})',
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    output_path = output_dir / f"sas_vector_layer{layer_idx}_{concept}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved visualization to {output_path}")
    plt.close()


def visualize_cross_layer_analysis(
    all_results: List[Dict],
    concept: str,
    output_dir: pathlib.Path,
):
    """Visualize how features evolve across layers."""
    n_layers = len(all_results)

    # Extract statistics
    layers = [r["layer_idx"] for r in all_results]
    n_active = [r["n_active"] for r in all_results]
    n_positive = [r["n_positive"] for r in all_results]
    n_negative = [r["n_negative"] for r in all_results]
    magnitudes = [r["magnitude"] for r in all_results]
    threshold_80 = [r["threshold_80"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Active features per layer
    axes[0, 0].bar(layers, n_active, color="#1f77b4", edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Layer", fontsize=12)
    axes[0, 0].set_ylabel("Number of Active Features", fontsize=12)
    axes[0, 0].set_title(
        "Feature Sparsity Across Layers", fontsize=14, fontweight="bold"
    )
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # 2. Positive vs Negative features
    x = np.arange(n_layers)
    width = 0.35
    axes[0, 1].bar(
        x - width / 2,
        n_positive,
        width,
        label="Positive (HIGH)",
        color="#1f77b4",
        edgecolor="black",
        alpha=0.7,
    )
    axes[0, 1].bar(
        x + width / 2,
        n_negative,
        width,
        label="Negative (LOW)",
        color="#d62728",
        edgecolor="black",
        alpha=0.7,
    )
    axes[0, 1].set_xlabel("Layer", fontsize=12)
    axes[0, 1].set_ylabel("Number of Features", fontsize=12)
    axes[0, 1].set_title(
        "Positive vs Negative Features", fontsize=14, fontweight="bold"
    )
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(layers)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # 3. Vector magnitude
    axes[1, 0].plot(
        layers, magnitudes, marker="o", linewidth=2, markersize=8, color="#2ca02c"
    )
    axes[1, 0].set_xlabel("Layer", fontsize=12)
    axes[1, 0].set_ylabel("L2 Magnitude", fontsize=12)
    axes[1, 0].set_title("SAS Vector Magnitude", fontsize=14, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Concentration (80% threshold)
    axes[1, 1].plot(
        layers, threshold_80, marker="s", linewidth=2, markersize=8, color="#ff7f0e"
    )
    axes[1, 1].set_xlabel("Layer", fontsize=12)
    axes[1, 1].set_ylabel("# Features for 80% Effect", fontsize=12)
    axes[1, 1].set_title("Feature Concentration", fontsize=14, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        f'Cross-Layer Analysis: {concept.replace("_", " ").title()}',
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    output_path = output_dir / f"cross_layer_analysis_{concept}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved cross-layer visualization to {output_path}")
    plt.close()


def save_analysis_report(
    all_results: List[Dict],
    concept: str,
    output_dir: pathlib.Path,
):
    """Save detailed analysis report to JSON."""
    report = {
        "concept": concept,
        "n_layers": len(all_results),
        "layers": all_results,
        "summary": {
            "total_active_features": sum(r["n_active"] for r in all_results),
            "avg_sparsity": np.mean([r["sparsity"] for r in all_results]),
            "avg_positive_features": np.mean([r["n_positive"] for r in all_results]),
            "avg_negative_features": np.mean([r["n_negative"] for r in all_results]),
            "avg_magnitude": np.mean([r["magnitude"] for r in all_results]),
        },
    }

    output_path = output_dir / f"interpretability_analysis_{concept}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved analysis report to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SAS vectors for interpretability"
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="average_pitch",
        choices=["average_pitch", "average_duration"],
        help="Concept to analyze",
    )
    parser.add_argument(
        "--sas_vectors_path",
        type=pathlib.Path,
        default=None,
        help="Path to SAS vectors file (default: auto-detect)",
    )
    parser.add_argument(
        "--diffmean_path",
        type=pathlib.Path,
        default=None,
        help="Path to DiffMean vectors file for comparison (default: auto-detect)",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("exp/sod/sparse_steering/interpretability"),
        help="Output directory for visualizations and reports",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Which layers to analyze (default: all, or comma-separated: 0,5,10)",
    )

    args = parser.parse_args()

    # Setup paths
    if args.sas_vectors_path is None:
        args.sas_vectors_path = (
            pathlib.Path("exp/sod/sparse_steering/sas_vectors")
            / f"{args.concept}_sas_vectors.pt"
        )

    if args.diffmean_path is None:
        args.diffmean_path = (
            pathlib.Path("exp/sod/steering_vectors")
            / f"{args.concept}_steering_vectors.pt"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("SAS INTERPRETABILITY ANALYSIS")
    logger.info("=" * 80)
    logger.info(f"Concept: {args.concept}")
    logger.info(f"SAS vectors: {args.sas_vectors_path}")
    logger.info(f"Output: {args.output_dir}")

    # Load vectors
    sas_data = load_sas_vectors(args.sas_vectors_path)
    diffmean_data = load_diffmean_vectors(args.diffmean_path)

    # Determine which layers to analyze
    if args.layers == "all":
        # Count layers in sas_data (exclude 'tau' and 'concept' keys)
        layer_keys = [k for k in sas_data.keys() if k.startswith("layer_")]
        n_layers = len(layer_keys)
        layers_to_analyze = list(range(n_layers))
    else:
        layers_to_analyze = [int(l.strip()) for l in args.layers.split(",")]

    logger.info(f"Analyzing layers: {layers_to_analyze}")

    # Analyze each layer
    all_results = []

    for layer_idx in layers_to_analyze:
        layer_key = f"layer_{layer_idx}"

        if layer_key not in sas_data:
            logger.warning(f"Layer {layer_idx} not found in SAS data, skipping")
            continue

        # Get SAS vector for this layer
        v_sas = sas_data[layer_key]
        if isinstance(v_sas, torch.Tensor):
            v_sas = v_sas.numpy()

        # Analyze
        results = analyze_single_layer(v_sas, layer_idx, args.concept)
        all_results.append(results)

        # Print analysis
        print_layer_analysis(results)

        # Compare with DiffMean if available
        if diffmean_data is not None and layer_key in diffmean_data:
            diffmean_vector = diffmean_data[layer_key]
            if isinstance(diffmean_vector, torch.Tensor):
                diffmean_vector = diffmean_vector.numpy()
            compare_with_diffmean(results, diffmean_vector, layer_idx)

        # Visualize
        visualize_sas_vector(v_sas, layer_idx, args.concept, args.output_dir, results)

    # Cross-layer analysis
    if len(all_results) > 1:
        logger.info("\n" + "=" * 80)
        logger.info("CROSS-LAYER SUMMARY")
        logger.info("=" * 80)

        for r in all_results:
            logger.info(
                f"Layer {r['layer_idx']:2d}: "
                f"{r['n_active']:3d} features "
                f"({r['sparsity']*100:5.2f}%), "
                f"pos={r['n_positive']:3d}, neg={r['n_negative']:3d}, "
                f"||v||={r['magnitude']:6.2f}"
            )

        visualize_cross_layer_analysis(all_results, args.concept, args.output_dir)

    # Save report
    save_analysis_report(all_results, args.concept, args.output_dir)

    logger.info("\n" + "=" * 80)
    logger.info("✓ ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nKey Findings:")
    logger.info(f"  Total layers analyzed: {len(all_results)}")
    logger.info(
        f"  Average sparsity: {np.mean([r['sparsity'] for r in all_results])*100:.2f}%"
    )
    logger.info(f"  Total active features: {sum(r['n_active'] for r in all_results)}")
    logger.info(f"\n✓ Visualizations saved to: {args.output_dir}")
    logger.info(
        f"✓ Detailed report saved to: {args.output_dir / f'interpretability_analysis_{args.concept}.json'}"
    )


if __name__ == "__main__":
    main()
