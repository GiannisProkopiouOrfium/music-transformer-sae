#!/usr/bin/env python3
"""Phase 0: Analyze Steering Vector Orthogonality.

This script analyzes the relationship between pitch and modality steering vectors
to determine whether orthogonalization is necessary for effective dual-concept steering.

Key metrics:
1. Cosine similarity: Measures angular alignment between vectors
   - |cos| < 0.3: Low interference (direct addition likely sufficient)
   - |cos| 0.3-0.5: Moderate interference (test multiple strategies)
   - |cos| > 0.5: High interference (orthogonalization recommended)

2. Projection magnitude: Measures overlap in vector subspaces
   - ||proj|| / ||vec|| < 0.3: Minimal overlap
   - ||proj|| / ||vec|| > 0.5: Significant overlap

3. Per-layer analysis: Some layers may interfere more than others

Usage:
    python dual_steering/analyze_vector_orthogonality.py \\
        --pitch_vectors steering_interventions/outputs/steering_vectors/average_pitch_steering_vectors.pt \\
        --modality_vectors steering_interventions/modality/outputs/steering_vectors/modality_steering_vectors.pt \\
        --output_dir steering_interventions/dual_steering/outputs/orthogonality_analysis
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from steered_generator import load_steering_vectors


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Calculate cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity (-1 to 1)
    """
    # Normalize vectors (flatten to ensure 1D)
    v1_flat = v1.flatten()
    v2_flat = v2.flatten()
    v1_norm = v1_flat / (v1_flat.norm() + 1e-8)
    v2_norm = v2_flat / (v2_flat.norm() + 1e-8)
    return float(torch.dot(v1_norm, v2_norm))


def projection_magnitude(v_source: torch.Tensor, v_target: torch.Tensor) -> float:
    """Calculate magnitude of v_source's projection onto v_target.

    Args:
        v_source: Vector to project
        v_target: Vector to project onto

    Returns:
        Ratio: ||projection|| / ||v_source||
    """
    # Flatten vectors to ensure 1D
    v_source_flat = v_source.flatten()
    v_target_flat = v_target.flatten()

    v_target_norm = v_target_flat / (v_target_flat.norm() + 1e-8)
    projection = torch.dot(v_source_flat, v_target_norm) * v_target_norm
    proj_magnitude = projection.norm()
    source_magnitude = v_source_flat.norm()

    return float(proj_magnitude / (source_magnitude + 1e-8))


def analyze_vector_pair(
    pitch_vec: torch.Tensor, modality_vec: torch.Tensor, layer_idx: int
) -> Dict:
    """Analyze relationship between pitch and modality vectors for one layer.

    Args:
        pitch_vec: Pitch steering vector
        modality_vec: Modality steering vector
        layer_idx: Layer index

    Returns:
        Dictionary with analysis results
    """
    # Cosine similarity
    cos_sim = cosine_similarity(pitch_vec, modality_vec)

    # Projection magnitudes (bidirectional)
    proj_modality_onto_pitch = projection_magnitude(modality_vec, pitch_vec)
    proj_pitch_onto_modality = projection_magnitude(pitch_vec, modality_vec)

    # Vector norms
    pitch_norm = float(pitch_vec.flatten().norm())
    modality_norm = float(modality_vec.flatten().norm())

    # Angle in degrees
    angle_deg = float(np.arccos(np.clip(cos_sim, -1, 1)) * 180 / np.pi)

    return {
        "layer": layer_idx,
        "cosine_similarity": cos_sim,
        "angle_degrees": angle_deg,
        "projection_modality_onto_pitch": proj_modality_onto_pitch,
        "projection_pitch_onto_modality": proj_pitch_onto_modality,
        "pitch_norm": pitch_norm,
        "modality_norm": modality_norm,
        "norm_ratio": modality_norm / (pitch_norm + 1e-8),
    }


def classify_interference_level(cos_sim: float, proj_mag: float) -> Tuple[str, str]:
    """Classify interference level based on metrics.

    Args:
        cos_sim: Cosine similarity
        proj_mag: Projection magnitude

    Returns:
        (level, recommendation) tuple
    """
    abs_cos = abs(cos_sim)

    if abs_cos < 0.3 and proj_mag < 0.3:
        return ("low", "direct_addition")
    elif abs_cos < 0.5 and proj_mag < 0.5:
        return ("moderate", "test_all_strategies")
    else:
        return ("high", "orthogonalization_recommended")


def visualize_analysis(
    results: Dict, output_dir: pathlib.Path, save_figures: bool = True
):
    """Create visualizations of the analysis.

    Args:
        results: Analysis results dictionary
        output_dir: Directory to save figures
        save_figures: Whether to save figures to disk
    """
    layers = sorted(results.keys())
    n_layers = len(layers)

    # Extract metrics
    cos_sims = [results[layer]["cosine_similarity"] for layer in layers]
    angles = [results[layer]["angle_degrees"] for layer in layers]
    proj_m2p = [results[layer]["projection_modality_onto_pitch"] for layer in layers]
    proj_p2m = [results[layer]["projection_pitch_onto_modality"] for layer in layers]
    pitch_norms = [results[layer]["pitch_norm"] for layer in layers]
    modality_norms = [results[layer]["modality_norm"] for layer in layers]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Steering Vector Orthogonality Analysis: Pitch vs Modality",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Cosine similarity per layer
    ax = axes[0, 0]
    ax.plot(layers, cos_sims, "o-", linewidth=2, markersize=8, color="#2E86AB")
    ax.axhline(y=0.3, color="orange", linestyle="--", alpha=0.7, label="Low threshold")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="High threshold")
    ax.axhline(y=-0.3, color="orange", linestyle="--", alpha=0.7)
    ax.axhline(y=-0.5, color="red", linestyle="--", alpha=0.7)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Cosine Similarity", fontsize=12)
    ax.set_title("Angular Alignment Between Vectors", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Angle in degrees
    ax = axes[0, 1]
    ax.plot(layers, angles, "o-", linewidth=2, markersize=8, color="#A23B72")
    ax.axhline(y=90, color="green", linestyle="--", alpha=0.7, label="Orthogonal")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Angle (degrees)", fontsize=12)
    ax.set_title("Vector Angles (90° = Perfect Orthogonality)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Projection magnitudes
    ax = axes[0, 2]
    ax.plot(
        layers,
        proj_m2p,
        "o-",
        linewidth=2,
        markersize=8,
        label="Modality → Pitch",
        color="#F18F01",
    )
    ax.plot(
        layers,
        proj_p2m,
        "s-",
        linewidth=2,
        markersize=8,
        label="Pitch → Modality",
        color="#C73E1D",
    )
    ax.axhline(y=0.3, color="orange", linestyle="--", alpha=0.7, label="Low threshold")
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="High threshold")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Projection Ratio", fontsize=12)
    ax.set_title("Subspace Overlap (||proj|| / ||vec||)", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Vector norms
    ax = axes[1, 0]
    ax.plot(
        layers,
        pitch_norms,
        "o-",
        linewidth=2,
        markersize=8,
        label="Pitch",
        color="#2E86AB",
    )
    ax.plot(
        layers,
        modality_norms,
        "s-",
        linewidth=2,
        markersize=8,
        label="Modality",
        color="#A23B72",
    )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("L2 Norm", fontsize=12)
    ax.set_title("Steering Vector Magnitudes", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Heatmap of cosine similarities
    ax = axes[1, 1]
    cos_matrix = np.array(cos_sims).reshape(1, -1)
    im = ax.imshow(cos_matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_yticks([0])
    ax.set_yticklabels(["Pitch vs\nModality"])
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_title("Cosine Similarity Heatmap", fontsize=13)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    # 6. Interference classification
    ax = axes[1, 2]
    classifications = []
    colors_map = {"low": "#06A77D", "moderate": "#F18F01", "high": "#D62828"}
    colors = []

    for layer in layers:
        level, _ = classify_interference_level(
            results[layer]["cosine_similarity"],
            max(
                results[layer]["projection_modality_onto_pitch"],
                results[layer]["projection_pitch_onto_modality"],
            ),
        )
        classifications.append(level)
        colors.append(colors_map[level])

    # Count classifications
    from collections import Counter

    counts = Counter(classifications)

    bars = ax.bar(
        ["Low", "Moderate", "High"],
        [counts["low"], counts["moderate"], counts["high"]],
        color=[colors_map["low"], colors_map["moderate"], colors_map["high"]],
    )

    # Add counts on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_ylabel("Number of Layers", fontsize=12)
    ax.set_title("Interference Classification Summary", fontsize=13)
    ax.set_ylim(0, n_layers + 1)

    plt.tight_layout()

    if save_figures:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / "orthogonality_analysis.png"
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved figure: {fig_path}")

    plt.show()


def print_summary(results: Dict, overall_recommendation: str):
    """Print analysis summary to console.

    Args:
        results: Analysis results dictionary
        overall_recommendation: Overall recommendation string
    """
    print("\n" + "=" * 80)
    print("STEERING VECTOR ORTHOGONALITY ANALYSIS")
    print("=" * 80)

    layers = sorted(results.keys())

    # Overall statistics
    cos_sims = [abs(results[layer]["cosine_similarity"]) for layer in layers]
    proj_mags = [
        max(
            results[layer]["projection_modality_onto_pitch"],
            results[layer]["projection_pitch_onto_modality"],
        )
        for layer in layers
    ]

    print(f"\nTotal layers analyzed: {len(layers)}")
    print("\nCosine Similarity (absolute values):")
    print(f"  Mean:   {np.mean(cos_sims):.4f}")
    print(f"  Median: {np.median(cos_sims):.4f}")
    print(f"  Min:    {np.min(cos_sims):.4f}")
    print(f"  Max:    {np.max(cos_sims):.4f}")
    print(f"  Std:    {np.std(cos_sims):.4f}")

    print("\nProjection Magnitudes (maximum):")
    print(f"  Mean:   {np.mean(proj_mags):.4f}")
    print(f"  Median: {np.median(proj_mags):.4f}")
    print(f"  Min:    {np.min(proj_mags):.4f}")
    print(f"  Max:    {np.max(proj_mags):.4f}")
    print(f"  Std:    {np.std(proj_mags):.4f}")

    # Per-layer classifications
    print(f"\n{'='*80}")
    print("PER-LAYER ANALYSIS")
    print("=" * 80)
    print(
        f"{'Layer':<8} {'Cos Sim':<10} {'Angle°':<10} {'Proj Max':<10} {'Level':<12} {'Recommendation':<25}"
    )
    print("-" * 80)

    classifications = []
    recommendations = []

    for layer in layers:
        r = results[layer]
        proj_max = max(
            r["projection_modality_onto_pitch"], r["projection_pitch_onto_modality"]
        )
        level, rec = classify_interference_level(r["cosine_similarity"], proj_max)

        classifications.append(level)
        recommendations.append(rec)

        print(
            f"{layer:<8} {r['cosine_similarity']:>+8.4f}  {r['angle_degrees']:>8.1f}  "
            f"{proj_max:>8.4f}  {level:<12} {rec:<25}"
        )

    # Summary counts
    from collections import Counter

    level_counts = Counter(classifications)
    rec_counts = Counter(recommendations)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nInterference Levels:")
    print(f"  Low:      {level_counts['low']:>2} layers")
    print(f"  Moderate: {level_counts['moderate']:>2} layers")
    print(f"  High:     {level_counts['high']:>2} layers")

    print("\nRecommended Strategies by Layer:")
    print(
        f"  Direct Addition Only:          {rec_counts.get('direct_addition', 0):>2} layers"
    )
    print(
        f"  Test All Strategies:           {rec_counts.get('test_all_strategies', 0):>2} layers"
    )
    print(
        f"  Orthogonalization Recommended: {rec_counts.get('orthogonalization_recommended', 0):>2} layers"
    )

    print(f"\n{'='*80}")
    print("OVERALL RECOMMENDATION")
    print("=" * 80)
    print(f"\n{overall_recommendation}")
    print("\n" + "=" * 80)


def determine_overall_recommendation(results: Dict) -> str:
    """Determine overall recommendation based on layer analysis.

    Args:
        results: Analysis results dictionary

    Returns:
        Recommendation string
    """
    layers = sorted(results.keys())

    # Count interference levels
    low_count = 0
    moderate_count = 0
    high_count = 0

    for layer in layers:
        r = results[layer]
        proj_max = max(
            r["projection_modality_onto_pitch"], r["projection_pitch_onto_modality"]
        )
        level, _ = classify_interference_level(r["cosine_similarity"], proj_max)

        if level == "low":
            low_count += 1
        elif level == "moderate":
            moderate_count += 1
        else:
            high_count += 1

    total = len(layers)

    # Decision logic
    if high_count > total * 0.5:
        return (
            f"🔴 HIGH INTERFERENCE DETECTED ({high_count}/{total} layers)\n\n"
            f"Recommendation: IMPLEMENT ORTHOGONALIZATION\n"
            f"  → Implement: gram_schmidt + symmetric strategies\n"
            f"  → Skip: direct addition (likely to cause significant interference)\n\n"
            f"The steering vectors show substantial overlap. Orthogonalization is\n"
            f"strongly recommended to prevent concepts from interfering with each other."
        )
    elif moderate_count > total * 0.3:
        return (
            f"🟡 MODERATE INTERFERENCE DETECTED ({moderate_count}/{total} layers moderate, {high_count}/{total} high)\n\n"
            f"Recommendation: TEST ALL THREE STRATEGIES\n"
            f"  → Implement: direct, gram_schmidt, symmetric\n"
            f"  → Run Phase 2 validation to compare performance\n\n"
            f"The steering vectors show variable overlap across layers. Empirical\n"
            f"testing is needed to determine which strategy works best."
        )
    else:
        return (
            f"🟢 LOW INTERFERENCE DETECTED ({low_count}/{total} layers low, {moderate_count}/{total} moderate)\n\n"
            f"Recommendation: START WITH DIRECT ADDITION\n"
            f"  → Implement: direct (primary) + gram_schmidt (validation)\n"
            f"  → Direct addition likely sufficient, but include orthogonalization for comparison\n\n"
            f"The steering vectors are largely independent. Simple addition should work well,\n"
            f"but we'll include orthogonalization to validate this assumption."
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze orthogonality between pitch and modality steering vectors"
    )
    parser.add_argument(
        "--pitch_vectors",
        type=pathlib.Path,
        required=True,
        help="Path to pitch steering vectors (.pt file)",
    )
    parser.add_argument(
        "--modality_vectors",
        type=pathlib.Path,
        required=True,
        help="Path to modality steering vectors (.pt file)",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/dual_steering/outputs/orthogonality_analysis"
        ),
        help="Output directory for results",
    )
    parser.add_argument(
        "--save_figures",
        action="store_true",
        default=True,
        help="Save figures to disk",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load steering vectors
    logging.info(f"Loading pitch vectors from: {args.pitch_vectors}")
    pitch_vectors, pitch_metadata = load_steering_vectors(args.pitch_vectors)

    logging.info(f"Loading modality vectors from: {args.modality_vectors}")
    modality_vectors, modality_metadata = load_steering_vectors(args.modality_vectors)

    # Verify layers match
    pitch_layers = set(pitch_vectors.keys())
    modality_layers = set(modality_vectors.keys())

    if pitch_layers != modality_layers:
        logging.warning(
            f"Layer mismatch: pitch has {len(pitch_layers)} layers, "
            f"modality has {len(modality_layers)} layers"
        )

    common_layers = sorted(pitch_layers & modality_layers)
    logging.info(f"Analyzing {len(common_layers)} common layers")

    # Analyze each layer
    results = {}
    for layer in common_layers:
        results[layer] = analyze_vector_pair(
            pitch_vectors[layer], modality_vectors[layer], layer
        )

    # Determine overall recommendation
    overall_recommendation = determine_overall_recommendation(results)

    # Print summary
    print_summary(results, overall_recommendation)

    # Visualize
    logging.info("\nGenerating visualizations...")
    visualize_analysis(results, args.output_dir, args.save_figures)

    # Save detailed results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_file = args.output_dir / "orthogonality_analysis.json"

    # Convert results to JSON-serializable format
    json_results = {}
    for layer, data in results.items():
        json_results[str(layer)] = {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in data.items()
        }

    with open(results_file, "w") as f:
        json.dump(
            {
                "metadata": {
                    "pitch_concept": pitch_metadata.get("concept", "unknown"),
                    "modality_concept": modality_metadata.get("concept", "unknown"),
                    "num_layers": len(common_layers),
                },
                "overall_recommendation": overall_recommendation,
                "per_layer_results": json_results,
            },
            f,
            indent=2,
        )

    logging.info(f"\nSaved detailed results to: {results_file}")
    logging.info("\n" + "=" * 80)
    logging.info("NEXT STEPS:")
    logging.info("=" * 80)
    logging.info("1. Review the recommendation above")
    logging.info(
        "2. Check the visualization: "
        + str(args.output_dir / "orthogonality_analysis.png")
    )
    logging.info("3. Share the results to proceed with implementation")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
