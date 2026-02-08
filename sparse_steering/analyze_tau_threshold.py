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
            "Shared %",
        ]

        rows = []
        for r in results:
            # Calculate shared ratio
            total_before_removal = r["avg_features"] + r["avg_shared"]
            shared_pct = (
                (r["avg_shared"] / total_before_removal * 100)
                if total_before_removal > 0
                else 0
            )

            row = [
                f"{r['tau']:.3f}",
                f"{r['avg_features']:.1f}",
                f"{r['min_features']}/{r['max_features']}",
                f"{r['coverage']:.1%}",
                f"{r['avg_norm']:.4f}",
                f"{r['avg_shared']:.1f}",
                f"{shared_pct:.1f}%",
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

        # Calculate shared ratios for all results
        for r in results:
            total = r["avg_features"] + r["avg_shared"]
            r["shared_ratio"] = (r["avg_shared"] / total) if total > 0 else 0

        # Find candidates with 100% coverage and low shared ratio
        full_coverage = [r for r in results if r["coverage"] == 1.0]

        if full_coverage:
            # Find best balance: low shared ratio, decent features
            # Prefer shared ratio < 15% and sufficient features (> 50)
            good_candidates = [
                r
                for r in full_coverage
                if r["shared_ratio"] < 0.15 and r["avg_features"] > 50
            ]

            if good_candidates:
                # Pick the one with most features among good candidates
                best = max(good_candidates, key=lambda r: r["avg_features"])
                logger.info(f"  ✅ RECOMMENDED: τ = {best['tau']:.3f}")
                logger.info(f"    - {best['avg_features']:.1f} avg features per layer")
                logger.info(f"    - {best['coverage']:.1%} layer coverage")
                logger.info(
                    f"    - {best['shared_ratio']:.1%} shared feature ratio (GOOD)"
                )
                logger.info(f"    - {best['avg_norm']:.4f} avg vector magnitude")
            else:
                # Relax constraints - just find lowest shared ratio with 100% coverage
                best = min(full_coverage, key=lambda r: r["shared_ratio"])
                logger.info(f"  ✅ RECOMMENDED: τ = {best['tau']:.3f}")
                logger.info(f"    - {best['avg_features']:.1f} avg features per layer")
                logger.info(f"    - {best['coverage']:.1%} layer coverage")
                logger.info(f"    - {best['shared_ratio']:.1%} shared feature ratio")
                logger.info(f"    - {best['avg_norm']:.4f} avg vector magnitude")
        else:
            # No 100% coverage, find best available
            best_coverage = max(results, key=lambda r: r["coverage"])
            logger.info(f"  ⚠ Best available: τ = {best_coverage['tau']:.3f}")
            logger.info(f"    - {best_coverage['coverage']:.1%} layer coverage")
            logger.info(f"    - Consider lowering τ for full coverage")

        # Highlight problematic τ values
        high_shared = [r for r in results if r["shared_ratio"] > 0.20]
        if high_shared:
            tau_list = [f"{r['tau']:.2f}" for r in high_shared]
            logger.info(f"  ⚠ High shared ratio (>20%): τ = {', '.join(tau_list)}")
            logger.info(f"    → Poor discriminability, avoid these values")

        # Show alternatives for comparison
        logger.info(f"\n  Alternatives to consider:")
        candidates = [r for r in results if r["coverage"] >= 0.8][:5]
        for r in candidates[:3]:  # Show top 3
            logger.info(
                f"    τ={r['tau']:.2f}: {r['avg_features']:.0f} features, "
                f"{r['shared_ratio']:.1%} shared, "
                f"||v||={r['avg_norm']:.1f}"
            )


def print_per_layer_breakdown(concept_results, tau_values_to_show=None):
    """Print per-layer breakdown for selected τ values."""

    logger.info(f"\n{'='*80}")
    logger.info("PER-LAYER BREAKDOWN")
    logger.info(f"{'='*80}\n")

    for concept, results in concept_results.items():
        # If not specified, show top 3 recommended values
        if tau_values_to_show is None:
            # Get results with 100% coverage, sorted by features
            full_cov = [r for r in results if r["coverage"] == 1.0]
            if full_cov:
                sorted_results = sorted(
                    full_cov, key=lambda r: r["avg_features"], reverse=True
                )
                tau_values_to_show = [r["tau"] for r in sorted_results[:3]]
            else:
                tau_values_to_show = [
                    results[0]["tau"],
                    results[len(results) // 2]["tau"],
                    results[-1]["tau"],
                ]

        selected = [r for r in results if r["tau"] in tau_values_to_show]

        for result in selected:
            logger.info(f"\n{concept} - τ = {result['tau']:.3f}")
            logger.info("-" * 60)

            headers = ["Layer", "Features", "||v_SAS||", "Shared Removed"]
            rows = []

            for i, stats in enumerate(result["layer_stats"]):
                row = [
                    f"L{i}",
                    f"{stats['n_final']}",
                    f"{stats['norm_vsas']:.2f}",
                    f"{stats['n_shared']}",
                ]
                rows.append(row)

            print(tabulate(rows, headers=headers, tablefmt="simple"))


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
    parser.add_argument(
        "--show_per_layer",
        action="store_true",
        help="Show per-layer breakdown for top candidates",
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

    # Print per-layer breakdown if requested
    if args.show_per_layer:
        print_per_layer_breakdown(concept_results)

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
