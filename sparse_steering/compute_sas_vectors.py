"""Compute Sparse Activation Steering (SAS) vectors.

This script implements Algorithm 1 from the SAS paper:
"Sparse Activation Steering: Activation Steering with Adaptive Sparse Features"

Algorithm Steps:
1. Extract sparse representations S+ and S- (done in encode_concept_activations.py)
2. Define non-zero rows: R+[c] = {r | S+[r,c] ≠ 0}, R-[c] = {r | S-[r,c] ≠ 0}
3. Frequency filtering: C+_τ = {c | |R+[c]|/|D| ≥ τ}, C-_τ = {c | |R-[c]|/|D| ≥ τ}
4. Compute means (only over non-zero activations):
   - v+[c] = (1/|R+[c]|) * Σ_{r∈R+[c]} S+[r,c] if c ∈ C+_τ, else 0
   - v-[c] = (1/|R-[c]|) * Σ_{r∈R-[c]} S-[r,c] if c ∈ C-_τ, else 0
5. Zero out shared features: C = {c | v+[c] ≠ 0 ∧ v-[c] ≠ 0}, set v+[C] = v-[C] = 0
6. Final vector: v_SAS = v+ - v-

Key difference from simple centroid difference (DiffMean):
- Only averages non-zero activations per feature
- Filters by frequency threshold τ
- Removes shared features at the vector level (not frequency level)

Stage 2, Part 2 of SAS implementation.
"""

import argparse
import logging
import pathlib
import torch
import numpy as np

from config_sas import (
    SAS_VECTORS_DIR,
    NUM_LAYERS,
    SPARSE_DIM,
)


def compute_feature_frequencies(sparse_activations: np.ndarray) -> np.ndarray:
    """Compute frequency that each feature is active (non-zero).

    Corresponds to |R[c]|/|D| in Algorithm 1.

    Args:
        sparse_activations: (N, sparse_dim) array of sparse features

    Returns:
        frequency: (sparse_dim,) array of activation frequencies [0, 1]
    """
    is_active = sparse_activations != 0  # (N, sparse_dim)
    frequency = is_active.mean(axis=0)  # (sparse_dim,)
    return frequency


def compute_mean_vector_sas(
    sparse_activations: np.ndarray,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean vector following SAS Algorithm 1.

    For each feature c:
    - If frequency >= τ: v[c] = mean of non-zero activations
    - Otherwise: v[c] = 0

    Args:
        sparse_activations: (N, sparse_dim) sparse features
        tau: frequency threshold

    Returns:
        v: (sparse_dim,) mean vector
        freq: (sparse_dim,) activation frequencies
    """
    n_samples, sparse_dim = sparse_activations.shape

    # Compute activation frequency for each feature (column)
    is_active = sparse_activations != 0  # (N, sparse_dim)
    freq = is_active.mean(axis=0)  # (sparse_dim,)

    # Identify features that meet frequency threshold
    # C_τ := {c | |R[c]|/|D| >= τ}
    valid_features = freq >= tau

    # Initialize mean vector
    v = np.zeros(sparse_dim)

    # For each valid feature, compute mean over non-zero activations
    # v[c] := (1/|R[c]|) * Σ_{r∈R[c]} S[r, c]
    for c in range(sparse_dim):
        if valid_features[c]:
            # Get non-zero activations for this feature
            active_rows = sparse_activations[:, c] != 0
            if active_rows.any():
                v[c] = sparse_activations[active_rows, c].mean()

    return v, freq


def compute_sas_vector(
    sparse_high: np.ndarray,
    sparse_low: np.ndarray,
    tau: float = 0.3,
) -> tuple[np.ndarray, dict]:
    """Compute SAS steering vector following Algorithm 1.

    Steps:
    1. Compute v+ by averaging non-zero activations for features with freq >= τ
    2. Compute v- by averaging non-zero activations for features with freq >= τ
    3. Zero out common features where v+[c] ≠ 0 AND v-[c] ≠ 0
    4. Compute final vector: v_SAS = v+ - v-

    Args:
        sparse_high: (N_high, sparse_dim) sparse features for positive group
        sparse_low: (N_low, sparse_dim) sparse features for negative group
        tau: frequency threshold (0.3 recommended in paper)

    Returns:
        sas_vector: (sparse_dim,) steering vector in sparse space
        stats: dictionary of statistics
    """
    # Step 1 & 2: Compute v+ and v- following Algorithm 1
    v_pos, freq_high = compute_mean_vector_sas(sparse_high, tau)
    v_neg, freq_low = compute_mean_vector_sas(sparse_low, tau)

    # Step 3: Zero out common columns
    # C = {c | (v+[c] ≠ 0 ∧ v-[c] ≠ 0)}
    common_features = (v_pos != 0) & (v_neg != 0)
    v_pos[common_features] = 0
    v_neg[common_features] = 0

    # Step 4: Compute final SAS vector
    sas_vector = v_pos - v_neg

    # Step 4: Compute final SAS vector
    sas_vector = v_pos - v_neg

    # Compute statistics
    num_features_total = len(sas_vector)

    # Features meeting frequency threshold
    features_high_freq = freq_high >= tau
    features_low_freq = freq_low >= tau
    num_features_high = features_high_freq.sum()
    num_features_low = features_low_freq.sum()

    # Features in final vectors (after frequency filtering)
    features_pos_active = v_pos != 0
    features_neg_active = v_neg != 0

    # Common features that were removed
    num_features_shared = common_features.sum()

    # Features in final SAS vector
    num_features_kept = (sas_vector != 0).sum()

    # Sparsity stats
    l0_high = (sparse_high != 0).sum(axis=1).mean()
    l0_low = (sparse_low != 0).sum(axis=1).mean()
    l0_vector = (sas_vector != 0).sum()

    stats = {
        "tau": tau,
        "n_samples_high": len(sparse_high),
        "n_samples_low": len(sparse_low),
        "n_features_total": num_features_total,
        "n_features_high_freq": num_features_high,
        "n_features_low_freq": num_features_low,
        "n_features_shared_removed": num_features_shared,
        "n_features_final": num_features_kept,
        "freq_high_mean": freq_high.mean(),
        "freq_low_mean": freq_low.mean(),
        "l0_high": l0_high,
        "l0_low": l0_low,
        "l0_vector": l0_vector,
        "vector_magnitude": np.linalg.norm(sas_vector),
        "v_pos_magnitude": np.linalg.norm(v_pos),
        "v_neg_magnitude": np.linalg.norm(v_neg),
    }

    return sas_vector, stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute SAS steering vectors from sparse activations"
    )
    parser.add_argument(
        "--concepts",
        type=str,
        nargs="+",
        default=["pitch", "duration"],
        help="Concepts to process (default: pitch duration)",
    )
    parser.add_argument(
        "--input_dir",
        type=pathlib.Path,
        default=SAS_VECTORS_DIR,
        help="Directory with sparse concept activations",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=SAS_VECTORS_DIR,
        help="Output directory for SAS vectors",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.3,
        help="Feature frequency threshold (default: 0.3)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process each concept
    for concept in args.concepts:
        logging.info("=" * 60)
        logging.info(f"Computing SAS vectors for: {concept}")
        logging.info(f"τ (tau) threshold: {args.tau}")
        logging.info("=" * 60)

        # Load sparse activations
        high_path = args.input_dir / f"{concept}_high_sparse.pt"
        low_path = args.input_dir / f"{concept}_low_sparse.pt"

        if not high_path.exists():
            logging.error(f"High sparse activations not found: {high_path}")
            logging.error("Please run encode_concept_activations.py first")
            continue

        if not low_path.exists():
            logging.error(f"Low sparse activations not found: {low_path}")
            logging.error("Please run encode_concept_activations.py first")
            continue

        logging.info(f"Loading sparse activations...")
        sparse_high_dict = torch.load(high_path, weights_only=False)
        sparse_low_dict = torch.load(low_path, weights_only=False)

        # Compute SAS vector for each layer
        sas_vectors = {}
        all_stats = {}

        for layer_idx in range(NUM_LAYERS):
            if layer_idx not in sparse_high_dict or layer_idx not in sparse_low_dict:
                logging.warning(f"  Layer {layer_idx}: Missing activations, skipping")
                continue

            sparse_high = sparse_high_dict[layer_idx]
            sparse_low = sparse_low_dict[layer_idx]

            # Compute SAS vector
            sas_vector, stats = compute_sas_vector(
                sparse_high=sparse_high,
                sparse_low=sparse_low,
                tau=args.tau,
            )

            sas_vectors[layer_idx] = sas_vector
            all_stats[layer_idx] = stats

            # Log statistics
            logging.info(f"\nLayer {layer_idx}:")
            logging.info(
                f"  Samples: {stats['n_samples_high']} high, {stats['n_samples_low']} low"
            )
            logging.info(f"  Frequency threshold τ = {stats['tau']}")
            logging.info(f"  Features meeting frequency threshold:")
            logging.info(
                f"    High (C+_τ): {stats['n_features_high_freq']}/{stats['n_features_total']} "
                f"({100 * stats['n_features_high_freq'] / stats['n_features_total']:.1f}%)"
            )
            logging.info(
                f"    Low (C-_τ):  {stats['n_features_low_freq']}/{stats['n_features_total']} "
                f"({100 * stats['n_features_low_freq'] / stats['n_features_total']:.1f}%)"
            )
            logging.info(
                f"  Shared features removed: {stats['n_features_shared_removed']}"
            )
            logging.info(
                f"  Final SAS vector features: {stats['n_features_final']}/{stats['n_features_total']} "
                f"({100 * stats['n_features_final'] / stats['n_features_total']:.1f}%)"
            )
            logging.info(f"  Sparsity:")
            logging.info(f"    L0 (high activations): {stats['l0_high']:.1f}")
            logging.info(f"    L0 (low activations):  {stats['l0_low']:.1f}")
            logging.info(f"    L0 (SAS vector): {stats['l0_vector']}")
            logging.info(f"  Vector magnitudes:")
            logging.info(f"    ||v+||: {stats['v_pos_magnitude']:.4f}")
            logging.info(f"    ||v-||: {stats['v_neg_magnitude']:.4f}")
            logging.info(f"    ||v_SAS||: {stats['vector_magnitude']:.4f}")

        # Save SAS vectors
        output_path = args.output_dir / f"{concept}_sas_vectors.pt"
        torch.save(sas_vectors, output_path)
        logging.info(f"\n✓ Saved SAS vectors to: {output_path}")

        # Save statistics
        stats_path = args.output_dir / f"{concept}_sas_stats.pt"
        torch.save(all_stats, stats_path)
        logging.info(f"✓ Saved statistics to: {stats_path}")

    logging.info("\n" + "=" * 60)
    logging.info("✓ SAS vector computation complete!")
    logging.info(f"✓ Saved to: {args.output_dir}")
    logging.info("=" * 60)
    logging.info("\nNext: Implement Stage 3 (inference-time steering)")


if __name__ == "__main__":
    main()
