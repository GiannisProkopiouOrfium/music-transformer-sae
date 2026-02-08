"""
Ablation study on τ (tau) threshold for SAS vectors.

This script tests multiple τ values to find the optimal balance between:
- Feature selectivity (higher τ = fewer, more consistent features)
- Vector richness (lower τ = more features for steering)
- Non-zero coverage (avoiding empty vectors across layers)

Usage:
    python sparse_steering/analyze_tau_threshold.py
    python sparse_steering/analyze_tau_threshold.py --tau_values 0.01 0.05 0.10 0.15 0.20
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_mean_vector_sas(sparse_activations, tau):
    """
    Compute mean vector following SAS Algorithm 1.

    For each feature c:
    - Compute frequency: freq[c] = |{r | S[r,c] ≠ 0}| / |D|
    - If freq[c] >= τ: v[c] = mean of non-zero values in S[:,c]
    - Otherwise: v[c] = 0

    Args:
        sparse_activations: (N, D) array of sparse activations
        tau: Frequency threshold

    Returns:
        v: (D,) mean vector
        freq: (D,) frequency array
    """
    N, D = sparse_activations.shape

    # Compute frequency for each feature
    active_mask = sparse_activations != 0  # (N, D)
    freq = active_mask.sum(axis=0) / N  # (D,)

    # Compute mean vector
    v = np.zeros(D)
    for c in range(D):
        if freq[c] >= tau:
            # Average only non-zero activations for this feature
            active_rows = active_mask[:, c]
            if active_rows.sum() > 0:
                v[c] = sparse_activations[active_rows, c].mean()

    return v, freq


def compute_sas_vector(sparse_high, sparse_low, tau):
    """
    Compute SAS vector following Algorithm 1.

    Args:
        sparse_high: (N+, D) sparse activations for high concept
        sparse_low: (N-, D) sparse activations for low concept
        tau: Frequency threshold

    Returns:
        v_sas: (D,) SAS steering vector
        stats: Dictionary of statistics
    """
    # Compute v+ and v- with frequency filtering
    v_pos, freq_high = compute_mean_vector_sas(sparse_high, tau)
    v_neg, freq_low = compute_mean_vector_sas(sparse_low, tau)

    # Count features meeting threshold (before shared removal)
    n_high_features = (v_pos != 0).sum()
    n_low_features = (v_neg != 0).sum()

    # Remove shared features (features active in both)
    common_features = (v_pos != 0) & (v_neg != 0)
    n_shared = common_features.sum()

    v_pos[common_features] = 0
    v_neg[common_features] = 0

    # Compute final SAS vector
    v_sas = v_pos - v_neg

    # Compute statistics
    stats = {
        "n_high_features": n_high_features,
        "n_low_features": n_low_features,
        "n_shared": n_shared,
        "n_final": (v_sas != 0).sum(),
        "norm_vpos": np.linalg.norm(v_pos),
        "norm_vneg": np.linalg.norm(v_neg),
        "norm_vsas": np.linalg.norm(v_sas),
        "mean_freq_high": (
            freq_high[freq_high >= tau].mean() if (freq_high >= tau).any() else 0.0
        ),
        "mean_freq_low": (
            freq_low[freq_low >= tau].mean() if (freq_low >= tau).any() else 0.0
        ),
    }

    return v_sas, stats


def analyze_tau_for_concept(concept, sparse_high, sparse_low, tau_values):
    """
    Analyze multiple τ values for a single concept.

    Returns:
        results: List of dicts with results for each τ
    """
    n_layers = len(sparse_high)
    results = []

    for tau in tau_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing τ = {tau:.3f}")
        logger.info(f"{'='*60}")

        layer_stats = []
        total_features = 0
        non_zero_layers = 0

        for layer_idx in range(n_layers):
            S_high = sparse_high[layer_idx]  # (N+, D)
            S_low = sparse_low[layer_idx]  # (N-, D)

            v_sas, stats = compute_sas_vector(S_high, S_low, tau)

            layer_stats.append(stats)
            total_features += stats["n_final"]
            if stats["n_final"] > 0:
                non_zero_layers += 1

        # Aggregate statistics
        avg_features = total_features / n_layers
        coverage = non_zero_layers / n_layers

        avg_norm = np.mean([s["norm_vsas"] for s in layer_stats])
        min_features = min(s["n_final"] for s in layer_stats)
        max_features = max(s["n_final"] for s in layer_stats)

        avg_shared = np.mean([s["n_shared"] for s in layer_stats])

        result = {
            "tau": tau,
            "avg_features": avg_features,
            "min_features": min_features,
            "max_features": max_features,
            "coverage": coverage,
            "non_zero_layers": non_zero_layers,
            "avg_norm": avg_norm,
            "avg_shared": avg_shared,
            "layer_stats": layer_stats,
        }

        results.append(result)

        logger.info(f"  Average features per layer: {avg_features:.1f}")
        logger.info(f"  Feature range: [{min_features}, {max_features}]")
        logger.info(f"  Non-zero layers: {non_zero_layers}/{n_layers} ({coverage:.1%})")
        logger.info(f"  Average ||v_SAS||: {avg_norm:.4f}")
        logger.info(f"  Average shared features removed: {avg_shared:.1f}")

    return results


def print_summary_table(concept_results):
    """Print a summary table comparing all τ values across concepts."""

    for concept, results in concept_results.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Summary for {concept}")
        logger.info(f"{'='*80}")

        headers = [
            "τ",
            "Avg Features",
            "Min/Max",
            "Coverage",
            "Avg ||v_SAS||",
            "Avg Shared",
        ]

        rows = []
        for r in results:
            row = [
                f"{r['tau']:.3f}",
                f"{r['avg_features']:.1f}",
                f"{r['min_features']}/{r['max_features']}",
                f"{r['coverage']:.1%}",
                f"{r['avg_norm']:.4f}",
                f"{r['avg_shared']:.1f}",
            ]
            rows.append(row)

        print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))


def print_recommendations(concept_results):
    """Provide recommendations based on analysis."""

    logger.info(f"\n{'='*80}")
    logger.info("RECOMMENDATIONS")
    logger.info(f"{'='*80}\n")

    for concept, results in concept_results.items():
        logger.info(f"\n{concept}:")

        # Find τ with best coverage (non-zero layers)
        best_coverage = max(results, key=lambda r: r["coverage"])

        # Find τ with good balance (decent features + high coverage)
        balanced = [r for r in results if r["coverage"] >= 0.8]  # At least 80% coverage
        if balanced:
            best_balanced = max(balanced, key=lambda r: r["avg_features"])
            logger.info(f"  ✓ Recommended τ = {best_balanced['tau']:.3f}")
            logger.info(
                f"    - {best_balanced['avg_features']:.1f} avg features per layer"
            )
            logger.info(f"    - {best_balanced['coverage']:.1%} layer coverage")
            logger.info(f"    - {best_balanced['avg_norm']:.4f} avg vector magnitude")
        else:
            logger.info(
                f"  ⚠ Best coverage: τ = {best_coverage['tau']:.3f} ({best_coverage['coverage']:.1%})"
            )
            logger.info(f"    - Consider lowering τ further for better coverage")

        # Check for empty vectors
        all_zero = [r for r in results if r["non_zero_layers"] == 0]
        if all_zero:
            logger.info(
                f"  ⚠ τ values with all-zero vectors: {[r['tau'] for r in all_zero]}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study on τ threshold for SAS vectors"
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=["average_pitch", "average_duration"],
        help="Concepts to analyze (default: average_pitch average_duration)",
    )
    parser.add_argument(
        "--tau_values",
        nargs="+",
        type=float,
        default=[0.01, 0.03, 0.05, 0.08, 0.10, 0.13, 0.15, 0.20, 0.25, 0.30],
        help="τ values to test",
    )
    parser.add_argument(
        "--exp_dir", type=Path, default=Path("exp/sod"), help="Experiment directory"
    )

    args = parser.parse_args()

    # Paths
    sas_dir = args.exp_dir / "sparse_steering" / "sas_vectors"

    if not sas_dir.exists():
        logger.error(f"SAS directory not found: {sas_dir}")
        logger.error("Run encode_concept_activations.py first")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("SAS τ Threshold Ablation Study")
    logger.info("=" * 80)
    logger.info(f"Concepts: {args.concepts}")
    logger.info(f"τ values: {args.tau_values}")
    logger.info(f"Directory: {sas_dir}")

    concept_results = {}

    for concept in args.concepts:
        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing: {concept}")
        logger.info(f"{'='*80}")

        # Load sparse activations
        high_path = sas_dir / f"{concept}_high_sparse.pt"
        low_path = sas_dir / f"{concept}_low_sparse.pt"

        if not high_path.exists() or not low_path.exists():
            logger.error(f"Missing sparse activations for {concept}")
            logger.error(f"  Expected: {high_path}")
            logger.error(f"           {low_path}")
            continue

        logger.info(f"Loading: {high_path.name}, {low_path.name}")
        sparse_high = torch.load(high_path, map_location="cpu")
        sparse_low = torch.load(low_path, map_location="cpu")

        n_layers = len(sparse_high)
        logger.info(f"Layers: {n_layers}")
        logger.info(f"Samples: {len(sparse_high[0])} high, {len(sparse_low[0])} low")

        # Analyze all τ values
        results = analyze_tau_for_concept(
            concept, sparse_high, sparse_low, args.tau_values
        )

        concept_results[concept] = results

    # Print summary tables
    print_summary_table(concept_results)

    # Print recommendations
    print_recommendations(concept_results)

    logger.info(f"\n{'='*80}")
    logger.info("Ablation study complete!")
    logger.info(f"{'='*80}")
    logger.info("\nNext steps:")
    logger.info("1. Choose optimal τ based on recommendations")
    logger.info(
        "2. Run: python sparse_steering/compute_sas_vectors.py --tau <chosen_value>"
    )
    logger.info("3. Proceed to Stage 3: Inference-time steering")


if __name__ == "__main__":
    main()
