#!/usr/bin/env python3
"""Phase 0 — Cross-concept overlap analysis between pitch and duration SAS vectors.

Loads the pre-computed SAS vectors for average_pitch and average_duration,
then computes per-layer diagnostics:

  • Number of non-zero features in each vector
  • Number of features active in BOTH vectors (overlap)
  • Overlap as fraction of smaller / larger set
  • Cosine similarity
  • Dot product
  • Feature-level Jaccard index
  • Distribution of overlap magnitudes

These metrics determine which dual-steering composition strategy is appropriate:
  - Low overlap + low cosine → direct addition should work
  - Moderate overlap → cross-concept masking recommended
  - High overlap / high cosine → Gram-Schmidt required

Output:
  - JSON report with all per-layer metrics
  - Console summary table
  - Matplotlib figure (overlap heatmap + cosine bar chart)

Usage:
    python sparse_steering/dual_steering/overlap_analysis.py
    python sparse_steering/dual_steering/overlap_analysis.py \
        --pitch_vectors  exp/sod/sparse_steering/sas_vectors/average_pitch_sas_vectors.pt \
        --duration_vectors exp/sod/sparse_steering/sas_vectors/average_duration_sas_vectors.pt \
        --output_dir exp/sod/sparse_steering/dual_steering/overlap_analysis
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add parent dirs to path
_THIS_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent))  # sparse_steering/
sys.path.insert(0, str(_THIS_DIR.parent.parent))  # project root

from config_dual import (
    PITCH_SAS_VECTORS,
    DURATION_SAS_VECTORS,
    OVERLAP_OUTPUT_DIR,
    NUM_LAYERS,
    SPARSE_DIM,
)


# ════════════════════════════════════════════════════════════════════════════
# Core analysis functions
# ════════════════════════════════════════════════════════════════════════════


def load_sas_vectors(path: pathlib.Path) -> Dict[int, np.ndarray]:
    """Load SAS vectors from .pt file as numpy arrays."""
    import torch

    data = torch.load(path, map_location="cpu", weights_only=False)
    vectors = {}
    for layer_idx, vec in data.items():
        if isinstance(vec, np.ndarray):
            vectors[int(layer_idx)] = vec
        else:
            vectors[int(layer_idx)] = vec.numpy()
    return vectors


def analyze_single_layer(
    v_pitch: np.ndarray, v_duration: np.ndarray, layer_idx: int
) -> dict:
    """Compute all overlap metrics for one layer.

    Args:
        v_pitch:    (sparse_dim,) SAS vector for pitch
        v_duration: (sparse_dim,) SAS vector for duration
        layer_idx:  layer index (for logging)

    Returns:
        Dictionary of metrics for this layer.
    """
    assert (
        v_pitch.shape == v_duration.shape == (SPARSE_DIM,)
    ), f"Shape mismatch: pitch={v_pitch.shape}, duration={v_duration.shape}"

    # Non-zero masks
    nz_pitch = v_pitch != 0
    nz_duration = v_duration != 0

    # Counts
    n_pitch = int(nz_pitch.sum())
    n_duration = int(nz_duration.sum())
    n_overlap = int((nz_pitch & nz_duration).sum())
    n_union = int((nz_pitch | nz_duration).sum())
    n_pitch_only = int((nz_pitch & ~nz_duration).sum())
    n_duration_only = int((~nz_pitch & nz_duration).sum())

    # Jaccard index
    jaccard = n_overlap / n_union if n_union > 0 else 0.0

    # Overlap fractions
    overlap_frac_pitch = n_overlap / n_pitch if n_pitch > 0 else 0.0
    overlap_frac_duration = n_overlap / n_duration if n_duration > 0 else 0.0
    overlap_frac_min = (
        n_overlap / min(n_pitch, n_duration) if min(n_pitch, n_duration) > 0 else 0.0
    )

    # Cosine similarity (over the full 4096-dim vector)
    norm_p = np.linalg.norm(v_pitch)
    norm_d = np.linalg.norm(v_duration)
    cosine = float(np.dot(v_pitch, v_duration) / (norm_p * norm_d + 1e-12))

    # Dot product
    dot = float(np.dot(v_pitch, v_duration))

    # Magnitudes
    magnitude_pitch = float(norm_p)
    magnitude_duration = float(norm_d)

    # For overlapping features — analyze magnitudes
    overlap_indices = np.where(nz_pitch & nz_duration)[0]
    if len(overlap_indices) > 0:
        overlap_pitch_vals = v_pitch[overlap_indices]
        overlap_duration_vals = v_duration[overlap_indices]

        # Sign agreement: are overlapping features pointing same direction?
        same_sign = int(
            np.sum(np.sign(overlap_pitch_vals) == np.sign(overlap_duration_vals))
        )
        opposite_sign = len(overlap_indices) - same_sign

        # Which concept dominates on shared features?
        pitch_dominant = int(
            np.sum(np.abs(overlap_pitch_vals) > np.abs(overlap_duration_vals))
        )
        duration_dominant = len(overlap_indices) - pitch_dominant

        overlap_details = {
            "indices": overlap_indices.tolist(),
            "pitch_magnitudes_mean": float(np.mean(np.abs(overlap_pitch_vals))),
            "duration_magnitudes_mean": float(np.mean(np.abs(overlap_duration_vals))),
            "same_sign_count": same_sign,
            "opposite_sign_count": opposite_sign,
            "pitch_dominant_count": pitch_dominant,
            "duration_dominant_count": duration_dominant,
        }
    else:
        overlap_details = {
            "indices": [],
            "pitch_magnitudes_mean": 0.0,
            "duration_magnitudes_mean": 0.0,
            "same_sign_count": 0,
            "opposite_sign_count": 0,
            "pitch_dominant_count": 0,
            "duration_dominant_count": 0,
        }

    return {
        "layer": layer_idx,
        "n_pitch_features": n_pitch,
        "n_duration_features": n_duration,
        "n_overlap": n_overlap,
        "n_union": n_union,
        "n_pitch_only": n_pitch_only,
        "n_duration_only": n_duration_only,
        "jaccard_index": jaccard,
        "overlap_frac_of_pitch": overlap_frac_pitch,
        "overlap_frac_of_duration": overlap_frac_duration,
        "overlap_frac_of_smaller": overlap_frac_min,
        "cosine_similarity": cosine,
        "dot_product": dot,
        "magnitude_pitch": magnitude_pitch,
        "magnitude_duration": magnitude_duration,
        "overlap_details": overlap_details,
    }


def analyze_all_layers(
    pitch_vectors: Dict[int, np.ndarray],
    duration_vectors: Dict[int, np.ndarray],
) -> list:
    """Run overlap analysis for all available layers."""
    layers = sorted(set(pitch_vectors.keys()) & set(duration_vectors.keys()))
    results = []
    for layer_idx in layers:
        metrics = analyze_single_layer(
            pitch_vectors[layer_idx], duration_vectors[layer_idx], layer_idx
        )
        results.append(metrics)
    return results


def compute_summary(layer_results: list) -> dict:
    """Aggregate layer-level results into a global summary."""
    n_layers = len(layer_results)

    avg = lambda key: float(np.mean([r[key] for r in layer_results]))
    total_overlap = sum(r["n_overlap"] for r in layer_results)
    total_pitch = sum(r["n_pitch_features"] for r in layer_results)
    total_duration = sum(r["n_duration_features"] for r in layer_results)

    # Recommendation logic
    max_cosine = max(r["cosine_similarity"] for r in layer_results)
    avg_overlap_frac = avg("overlap_frac_of_smaller")
    avg_cosine = avg("cosine_similarity")

    if avg_overlap_frac < 0.05 and abs(avg_cosine) < 0.1:
        recommendation = "DIRECT_ADDITION"
        reasoning = (
            "Vectors have <5% feature overlap and near-zero cosine similarity. "
            "SAE successfully disentangled the concepts — direct addition should "
            "work with minimal interference."
        )
    elif avg_overlap_frac < 0.15 and abs(avg_cosine) < 0.3:
        recommendation = "CROSS_CONCEPT_MASKING"
        reasoning = (
            "Moderate overlap exists. Cross-concept masking (assign shared "
            "features to the dominant concept) is the natural sparse-space fix. "
            "Also test direct addition as a baseline."
        )
    else:
        recommendation = "GRAM_SCHMIDT"
        reasoning = (
            "Significant overlap or cosine similarity detected. Gram-Schmidt "
            "orthogonalisation in sparse space recommended, though it may break "
            "sparsity and require re-sparsification."
        )

    return {
        "n_layers_analyzed": n_layers,
        "avg_cosine_similarity": avg_cosine,
        "max_cosine_similarity": float(max_cosine),
        "avg_overlap_frac_of_smaller": avg_overlap_frac,
        "total_overlap_features": total_overlap,
        "total_pitch_features": total_pitch,
        "total_duration_features": total_duration,
        "recommendation": recommendation,
        "reasoning": reasoning,
    }


# ════════════════════════════════════════════════════════════════════════════
# Visualisation
# ════════════════════════════════════════════════════════════════════════════


def plot_overlap_report(layer_results: list, summary: dict, output_dir: pathlib.Path):
    """Generate a multi-panel figure summarising the overlap analysis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping plot")
        return

    n = len(layer_results)
    layers = [r["layer"] for r in layer_results]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # ── Panel 1: Feature counts (stacked bar) ──
    ax = axes[0, 0]
    pitch_only = [r["n_pitch_only"] for r in layer_results]
    dur_only = [r["n_duration_only"] for r in layer_results]
    overlap = [r["n_overlap"] for r in layer_results]
    x = np.arange(n)
    ax.bar(x, pitch_only, label="Pitch only", color="#1f77b4")
    ax.bar(x, dur_only, bottom=pitch_only, label="Duration only", color="#ff7f0e")
    ax.bar(
        x,
        overlap,
        bottom=[p + d for p, d in zip(pitch_only, dur_only)],
        label="Shared",
        color="#d62728",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("# Non-zero Features")
    ax.set_title("Feature Partition by Concept")
    ax.legend(fontsize=8)

    # ── Panel 2: Cosine similarity ──
    ax = axes[0, 1]
    cosines = [r["cosine_similarity"] for r in layer_results]
    colors = [
        "#d62728" if abs(c) > 0.3 else "#2ca02c" if abs(c) < 0.1 else "#ff7f0e"
        for c in cosines
    ]
    ax.bar(x, cosines, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.3, color="grey", ls="--", alpha=0.5, label="|cos|=0.3")
    ax.axhline(-0.3, color="grey", ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Similarity (pitch ↔ duration)")

    # ── Panel 3: Overlap fraction ──
    ax = axes[0, 2]
    frac_pitch = [r["overlap_frac_of_pitch"] for r in layer_results]
    frac_dur = [r["overlap_frac_of_duration"] for r in layer_results]
    w = 0.35
    ax.bar(
        x - w / 2,
        frac_pitch,
        w,
        label="% of pitch features",
        color="#1f77b4",
        alpha=0.7,
    )
    ax.bar(
        x + w / 2,
        frac_dur,
        w,
        label="% of duration features",
        color="#ff7f0e",
        alpha=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Overlap Fraction")
    ax.set_title("Overlap as % of Each Concept's Features")
    ax.legend(fontsize=8)

    # ── Panel 4: Jaccard index ──
    ax = axes[1, 0]
    jaccards = [r["jaccard_index"] for r in layer_results]
    ax.bar(x, jaccards, color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Jaccard Index")
    ax.set_title("Jaccard Index (feature overlap / union)")

    # ── Panel 5: Sign agreement on shared features ──
    ax = axes[1, 1]
    same = [r["overlap_details"]["same_sign_count"] for r in layer_results]
    opp = [r["overlap_details"]["opposite_sign_count"] for r in layer_results]
    ax.bar(x - w / 2, same, w, label="Same sign", color="#2ca02c")
    ax.bar(x + w / 2, opp, w, label="Opposite sign", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("# Shared Features")
    ax.set_title("Sign Agreement on Shared Features")
    ax.legend(fontsize=8)

    # ── Panel 6: Vector magnitudes ──
    ax = axes[1, 2]
    mag_p = [r["magnitude_pitch"] for r in layer_results]
    mag_d = [r["magnitude_duration"] for r in layer_results]
    ax.plot(layers, mag_p, "o-", label="Pitch ||v||", color="#1f77b4")
    ax.plot(layers, mag_d, "s-", label="Duration ||v||", color="#ff7f0e")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 Norm")
    ax.set_title("SAS Vector Magnitudes")
    ax.legend()

    fig.suptitle(
        f"Cross-Concept Overlap Analysis: Pitch ↔ Duration SAS Vectors\n"
        f"Recommendation: {summary['recommendation']}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "overlap_analysis.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "overlap_analysis.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved overlap_analysis.png/pdf")


# ════════════════════════════════════════════════════════════════════════════
# Console output
# ════════════════════════════════════════════════════════════════════════════


def print_table(layer_results: list, summary: dict):
    """Print a nicely-formatted summary table to stdout."""
    header = (
        f"{'Layer':>5} │ {'#Pitch':>7} │ {'#Dur':>7} │ {'#Shared':>7} │ "
        f"{'Overlap%':>8} │ {'Jaccard':>7} │ {'Cosine':>7} │ {'DotProd':>9}"
    )
    sep = "─" * len(header)

    print("\n" + "=" * 72)
    print(" CROSS-CONCEPT OVERLAP ANALYSIS: Pitch ↔ Duration SAS Vectors")
    print("=" * 72)
    print(header)
    print(sep)

    for r in layer_results:
        overlap_pct = r["overlap_frac_of_smaller"] * 100
        print(
            f"{r['layer']:>5} │ {r['n_pitch_features']:>7} │ "
            f"{r['n_duration_features']:>7} │ {r['n_overlap']:>7} │ "
            f"{overlap_pct:>7.1f}% │ {r['jaccard_index']:>7.4f} │ "
            f"{r['cosine_similarity']:>+7.4f} │ {r['dot_product']:>+9.2f}"
        )

    print(sep)
    print(f"\n{'SUMMARY':>5}")
    print(f"  Average cosine similarity:  {summary['avg_cosine_similarity']:+.4f}")
    print(f"  Max |cosine|:               {summary['max_cosine_similarity']:.4f}")
    print(f"  Average overlap fraction:   {summary['avg_overlap_frac_of_smaller']:.1%}")
    print(f"  Total shared features:      {summary['total_overlap_features']}")

    print(f"\n  ┌──────────────────────────────────────────────────────────┐")
    print(f"  │  RECOMMENDATION: {summary['recommendation']:<40} │")
    print(f"  └──────────────────────────────────────────────────────────┘")
    print(f"\n  {summary['reasoning']}")

    # Per-layer breakdown of shared features (helpful for debugging)
    print(f"\n  Shared feature detail per layer:")
    for r in layer_results:
        od = r["overlap_details"]
        if r["n_overlap"] > 0:
            print(
                f"    Layer {r['layer']:2d}: {r['n_overlap']:3d} shared — "
                f"same-sign={od['same_sign_count']}, opp-sign={od['opposite_sign_count']} — "
                f"pitch-dominant={od['pitch_dominant_count']}, "
                f"dur-dominant={od['duration_dominant_count']}"
            )
        else:
            print(f"    Layer {r['layer']:2d}:   0 shared (fully disjoint!)")
    print()


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: Cross-concept overlap analysis for dual SAS steering"
    )
    parser.add_argument(
        "--pitch_vectors",
        type=pathlib.Path,
        default=PITCH_SAS_VECTORS,
        help="Path to pitch SAS vectors (.pt)",
    )
    parser.add_argument(
        "--duration_vectors",
        type=pathlib.Path,
        default=DURATION_SAS_VECTORS,
        help="Path to duration SAS vectors (.pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=OVERLAP_OUTPUT_DIR,
        help="Output directory for report and plots",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip matplotlib plot generation",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load vectors ──
    logger.info(f"Loading pitch SAS vectors: {args.pitch_vectors}")
    pitch_vecs = load_sas_vectors(args.pitch_vectors)
    logger.info(f"  Loaded {len(pitch_vecs)} layers")

    logger.info(f"Loading duration SAS vectors: {args.duration_vectors}")
    duration_vecs = load_sas_vectors(args.duration_vectors)
    logger.info(f"  Loaded {len(duration_vecs)} layers")

    # ── Analyse ──
    logger.info("Computing per-layer overlap metrics...")
    layer_results = analyze_all_layers(pitch_vecs, duration_vecs)
    summary = compute_summary(layer_results)

    # ── Output ──
    print_table(layer_results, summary)

    # Save JSON
    report = {
        "per_layer": layer_results,
        "summary": summary,
        "paths": {
            "pitch_vectors": str(args.pitch_vectors),
            "duration_vectors": str(args.duration_vectors),
        },
    }
    json_path = args.output_dir / "overlap_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved JSON report: {json_path}")

    # Plot
    if not args.no_plot:
        plot_overlap_report(layer_results, summary, args.output_dir)

    print("\n✓ Phase 0 overlap analysis complete.")
    print(f"✓ Results saved to: {args.output_dir}")
    print(f"\nNext step: Implement composition strategies and run Phase 1 grid search.")

    return report


if __name__ == "__main__":
    main()
