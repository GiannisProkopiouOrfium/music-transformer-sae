#!/usr/bin/env python3
"""
Pre-Whitening Diagnostic & Post-Hoc SAS Normalization
=====================================================

Checks whether the residual stream activations are anisotropic (non-uniform
covariance), and if so, normalizes SAS vector weights to account for the
dense-space geometry.

Two stages:

  1. DIAGNOSTIC — compute the covariance matrix Σ of dense activations at the
     steering layer, report:
       • Condition number  κ = λ_max / λ_min
       • Top-10 / bottom-10 eigenvalues
       • "Effective rank" (how many principal components explain 90% of variance)
     If κ ≈ 1, the space is already isotropic → normalization is unnecessary.

  2. OPTION B — Post-hoc normalization of SAS features.  For each feature c:
         v_norm[c] = v_sas[c] / √(e_c^T Σ e_c)
     where e_c is the encoder row for feature c.
     Reports how much the feature ranking changes after normalization.

Usage (on EC2):
    cd /home/ubuntu/mmt
    python feature_explainability/whitening_analysis.py

    # With custom paths
    python feature_explainability/whitening_analysis.py \
        --training_data exp/sod/sparse_steering/sae_training_data/random_activations_full.h5 \
        --sae_dir exp/sod/sparse_steering/sae_checkpoints \
        --sas_dir exp/sod/sparse_steering/sas_vectors \
        --output_dir feature_explainability/outputs/whitening
"""

import argparse
import json
import logging
import pathlib
import sys

import numpy as np

try:
    import torch
except ImportError:
    print("ERROR: PyTorch required. pip install torch")
    sys.exit(1)

try:
    import h5py
except ImportError:
    print("ERROR: h5py required. pip install h5py")
    sys.exit(1)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_TRAINING_DATA = (
    PROJECT_ROOT
    / "exp"
    / "sod"
    / "sparse_steering"
    / "sae_training_data"
    / "random_activations_full.h5"
)
DEFAULT_SAE_DIR = PROJECT_ROOT / "exp" / "sod" / "sparse_steering" / "sae_checkpoints"
DEFAULT_SAS_DIR = PROJECT_ROOT / "exp" / "sod" / "sparse_steering" / "sas_vectors"
DEFAULT_OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "outputs" / "whitening"

STEERING_LAYER = 10
CONCEPTS = ["average_pitch", "average_duration"]
CONCEPT_LABELS = {"average_pitch": "Pitch", "average_duration": "Duration"}
SPARSE_DIM = 4096

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)

if HAS_MPL:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1: Diagnostic — Covariance Anisotropy
# ═════════════════════════════════════════════════════════════════════════════


def load_training_activations(h5_path: pathlib.Path, layer: int) -> np.ndarray:
    """Load dense activations from HDF5 training data."""
    logger.info(f"Loading training activations from {h5_path}, layer {layer}")
    with h5py.File(h5_path, "r") as f:
        key = f"layer_{layer}"
        if key not in f:
            raise KeyError(
                f"Layer {layer} not found in {h5_path}. Keys: {list(f.keys())}"
            )
        data = f[key][:]
    logger.info(f"  Shape: {data.shape} (n_samples × d_model)")
    return data.astype(np.float64)


def compute_covariance_diagnostic(activations: np.ndarray) -> dict:
    """Compute the covariance matrix and eigenvalue analysis.

    Returns dict with condition_number, eigenvalues, effective_rank, etc.
    """
    n, d = activations.shape
    logger.info(f"  Computing {d}×{d} covariance matrix from {n} samples...")

    # Center the data
    mean = activations.mean(axis=0)
    centered = activations - mean

    # Covariance matrix
    cov = centered.T @ centered / (n - 1)  # (d, d)

    # Eigendecomposition (symmetric → use eigh for stability)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns sorted ascending, flip to descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Condition number
    lambda_max = eigenvalues[0]
    lambda_min = eigenvalues[-1]
    # Clamp near-zero eigenvalues for numerical stability
    lambda_min_safe = max(lambda_min, 1e-10)
    condition_number = lambda_max / lambda_min_safe

    # Effective rank (number of eigenvalues explaining 90% of total variance)
    total_var = eigenvalues.sum()
    cumvar = np.cumsum(eigenvalues) / total_var
    effective_rank_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    effective_rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1

    # Ratio of largest to 10th, 50th, 100th eigenvalue
    ratio_10 = eigenvalues[0] / eigenvalues[min(9, d - 1)]
    ratio_50 = eigenvalues[0] / eigenvalues[min(49, d - 1)]

    result = {
        "n_samples": n,
        "d_model": d,
        "condition_number": float(condition_number),
        "lambda_max": float(lambda_max),
        "lambda_min": float(lambda_min),
        "ratio_max_to_10th": float(ratio_10),
        "ratio_max_to_50th": float(ratio_50),
        "effective_rank_90pct": effective_rank_90,
        "effective_rank_99pct": effective_rank_99,
        "total_variance": float(total_var),
        "top_10_eigenvalues": eigenvalues[:10].tolist(),
        "bottom_10_eigenvalues": eigenvalues[-10:].tolist(),
        "explained_variance_ratio": (eigenvalues / total_var)[:50].tolist(),
    }

    logger.info(f"  Condition number κ = {condition_number:.1f}")
    logger.info(f"  λ_max = {lambda_max:.4f},  λ_min = {lambda_min:.6f}")
    logger.info(f"  λ_max / λ_10 = {ratio_10:.1f},  λ_max / λ_50 = {ratio_50:.1f}")
    logger.info(f"  Effective rank (90% var) = {effective_rank_90}/{d}")
    logger.info(f"  Effective rank (99% var) = {effective_rank_99}/{d}")

    if condition_number < 10:
        logger.info(
            "  ✓ Space is roughly isotropic (κ < 10). Normalization has minor effect."
        )
    elif condition_number < 100:
        logger.info(
            "  ⚠ Moderate anisotropy (10 < κ < 100). Normalization recommended."
        )
    else:
        logger.info(
            "  ⚠ Strong anisotropy (κ > 100). Normalization strongly recommended."
        )

    return result, cov, eigenvalues, eigenvectors, mean


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2: Option B — Post-Hoc SAS Normalization
# ═════════════════════════════════════════════════════════════════════════════


def load_sae_encoder(sae_dir: pathlib.Path, layer: int):
    """Load the SAE encoder weight matrix for the given layer.

    Returns encoder_weight (sparse_dim × input_dim) as numpy array,
    plus the input normalization parameters.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "sparse_steering"))
    from sae_model import SparseAutoencoder

    k = 128 if layer >= 8 else (96 if layer >= 4 else (64 if layer >= 1 else 32))

    sae = SparseAutoencoder(
        input_dim=512, sparse_dim=4096, k=k, tied_weights=True, normalize_input=True
    )
    ckpt_path = sae_dir / f"sae_layer_{layer}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    if "_normalization_fitted" not in state_dict:
        state_dict["_normalization_fitted"] = torch.tensor(
            1 if state_dict.get("input_mean", torch.zeros(1)).abs().sum() > 0 else 0
        )
    sae.load_state_dict(state_dict)
    sae.eval()

    # Encoder weight: (sparse_dim, input_dim) = (4096, 512)
    encoder_weight = sae.encoder.weight.detach().numpy()

    # Input normalization
    input_mean = sae.input_mean.detach().numpy()
    input_std = sae.input_std.detach().numpy()

    logger.info(f"  Loaded SAE encoder: {encoder_weight.shape}")
    return encoder_weight, input_mean, input_std


def load_sas_vectors(sas_dir: pathlib.Path, concept: str):
    """Load SAS vectors for a concept."""
    path = sas_dir / f"{concept}_sas_vectors.pt"
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    result = {}
    for k, v in data.items():
        result[int(k)] = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)
    return result


def normalize_sas_vector(
    v_sas: np.ndarray,
    encoder_weight: np.ndarray,
    cov_matrix: np.ndarray,
    input_std: np.ndarray,
) -> np.ndarray:
    """Normalize SAS vector weights by the encoder direction's variance.

    For each feature c:
        v_norm[c] = v_sas[c] / √(e_c^T Σ e_c)

    where e_c is row c of the encoder (in original dense space, accounting
    for the SAE's per-dimension normalization).

    The encoder operates on normalized inputs: x_norm = (x - μ) / σ
    So the effective encoder direction in the original space is:
        e_c_orig[j] = e_c[j] / σ[j]
    and the variance contributed is:
        var_c = e_c_orig^T Σ e_c_orig
    """
    n_features = len(v_sas)
    v_norm = np.zeros(n_features)

    for c in range(n_features):
        if v_sas[c] == 0:
            continue

        # Encoder direction for feature c (in normalized space)
        e_c = encoder_weight[c]  # (512,)

        # Convert to original space: e_c_orig[j] = e_c[j] / σ[j]
        e_c_orig = e_c / (input_std + 1e-8)

        # Variance contributed by this direction: e_c^T Σ e_c
        var_c = e_c_orig @ cov_matrix @ e_c_orig

        if var_c > 0:
            v_norm[c] = v_sas[c] / np.sqrt(var_c)
        else:
            v_norm[c] = v_sas[c]

    return v_norm


def compare_rankings(v_original, v_normalized, top_k=20):
    """Compare feature rankings before and after normalization."""
    orig_top = np.argsort(np.abs(v_original))[::-1][:top_k]
    norm_top = np.argsort(np.abs(v_normalized))[::-1][:top_k]

    # Rank correlation (Kendall tau-like: how many pairs are in same order?)
    orig_ranks = {fid: rank for rank, fid in enumerate(orig_top)}
    norm_ranks = {fid: rank for rank, fid in enumerate(norm_top)}

    shared = set(orig_top) & set(norm_top)
    new_in_top = set(norm_top) - set(orig_top)
    dropped = set(orig_top) - set(norm_top)

    comparisons = []
    for fid in orig_top:
        orig_w = float(v_original[fid])
        norm_w = float(v_normalized[fid])
        orig_r = orig_ranks.get(fid, -1)
        norm_r = norm_ranks.get(fid, -1)
        comparisons.append(
            {
                "feature_id": int(fid),
                "orig_weight": orig_w,
                "norm_weight": norm_w,
                "orig_rank": orig_r + 1,
                "norm_rank": norm_r + 1 if norm_r >= 0 else ">20",
                "sign": "+" if orig_w > 0 else "-",
            }
        )

    return {
        "shared_in_top20": len(shared),
        "new_entries": [int(f) for f in new_in_top],
        "dropped_out": [int(f) for f in dropped],
        "overlap_pct": len(shared) / top_k * 100,
        "comparisons": comparisons,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════


def plot_diagnostic(diagnostic, output_dir):
    """Plot eigenvalue spectrum and cumulative variance."""
    if not HAS_MPL:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Eigenvalue spectrum (log scale)
    evals = diagnostic["top_10_eigenvalues"] + diagnostic["bottom_10_eigenvalues"]
    # Plot full explained variance ratio
    evr = diagnostic["explained_variance_ratio"]
    ax1.semilogy(range(1, len(evr) + 1), evr, "b-", linewidth=2)
    ax1.set_xlabel("Principal Component Index")
    ax1.set_ylabel("Fraction of Total Variance (log)")
    ax1.set_title(
        f"(a) Eigenvalue Spectrum — κ = {diagnostic['condition_number']:.0f}",
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.axhline(
        y=1 / diagnostic["d_model"],
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Uniform = 1/{diagnostic['d_model']}",
    )
    ax1.legend()

    # (b) Cumulative explained variance
    cumvar = np.cumsum(evr)
    ax2.plot(range(1, len(cumvar) + 1), cumvar, "b-", linewidth=2)
    ax2.axhline(0.90, color="orange", linestyle="--", alpha=0.5, label="90% variance")
    ax2.axhline(0.99, color="red", linestyle="--", alpha=0.5, label="99% variance")
    ax2.axvline(
        diagnostic["effective_rank_90pct"], color="orange", linestyle=":", alpha=0.4
    )
    ax2.axvline(
        diagnostic["effective_rank_99pct"], color="red", linestyle=":", alpha=0.4
    )
    ax2.set_xlabel("Number of Principal Components")
    ax2.set_ylabel("Cumulative Variance Explained")
    ax2.set_title(
        f"(b) Effective Rank — "
        f"90%={diagnostic['effective_rank_90pct']}, "
        f"99%={diagnostic['effective_rank_99pct']}",
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "fig_whitening_diagnostic.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_ranking_comparison(all_comparisons, output_dir):
    """Plot before/after ranking for each concept."""
    if not HAS_MPL:
        return

    n = len(all_comparisons)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 7))
    if n == 1:
        axes = [axes]

    for ax, (concept, comp) in zip(axes, all_comparisons.items()):
        label = CONCEPT_LABELS.get(concept, concept)
        data = comp["comparisons"]

        fids = [f"F{d['feature_id']}" for d in data]
        orig_abs = [abs(d["orig_weight"]) for d in data]
        norm_abs = [abs(d["norm_weight"]) for d in data]

        x = np.arange(len(fids))
        width = 0.35

        ax.barh(
            x - width / 2,
            orig_abs,
            width,
            label="Original |w|",
            color="#1565C0",
            alpha=0.8,
        )
        ax.barh(
            x + width / 2,
            norm_abs,
            width,
            label="Normalized |w|",
            color="#C62828",
            alpha=0.8,
        )
        ax.set_yticks(x)
        ax.set_yticklabels(fids, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("|SAS weight|")
        ax.set_title(f"{label} — Original vs Whitening-Normalized", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="x")

        # Highlight features that change rank dramatically
        for i, d in enumerate(data):
            if d["norm_rank"] == ">20":
                ax.annotate(
                    "dropped",
                    (0, i),
                    fontsize=7,
                    color="gray",
                    xytext=(5, 0),
                    textcoords="offset points",
                )

    plt.tight_layout()
    path = output_dir / "fig_whitening_ranking.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Pre-whitening diagnostic & post-hoc SAS normalization"
    )
    parser.add_argument("--training_data", type=str, default=str(DEFAULT_TRAINING_DATA))
    parser.add_argument("--sae_dir", type=str, default=str(DEFAULT_SAE_DIR))
    parser.add_argument("--sas_dir", type=str, default=str(DEFAULT_SAS_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--layer", type=int, default=STEERING_LAYER)
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("  Pre-Whitening Diagnostic & SAS Normalization")
    logger.info("=" * 72)

    # ── Stage 1: Diagnostic ──
    logger.info("")
    logger.info("━━━ Stage 1: Covariance Anisotropy Diagnostic ━━━")

    activations = load_training_activations(
        pathlib.Path(args.training_data), args.layer
    )
    diagnostic, cov_matrix, eigenvalues, eigenvectors, mean = (
        compute_covariance_diagnostic(activations)
    )

    diag_path = output_dir / "covariance_diagnostic.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostic, f, indent=2)
    logger.info(f"Saved: {diag_path}")

    plot_diagnostic(diagnostic, output_dir)

    # ── Stage 2: Post-Hoc Normalization ──
    logger.info("")
    logger.info("━━━ Stage 2: Post-Hoc SAS Vector Normalization ━━━")

    encoder_weight, input_mean, input_std = load_sae_encoder(
        pathlib.Path(args.sae_dir), args.layer
    )

    all_comparisons = {}
    all_normalized_vectors = {}

    for concept in CONCEPTS:
        label = CONCEPT_LABELS[concept]
        logger.info(f"─── {label} ───")

        sas_vecs = load_sas_vectors(pathlib.Path(args.sas_dir), concept)
        if sas_vecs is None:
            logger.warning(f"  SAS vectors not found for {concept}")
            continue

        v_sas = sas_vecs[args.layer]
        v_norm = normalize_sas_vector(v_sas, encoder_weight, cov_matrix, input_std)

        # Compare rankings
        comparison = compare_rankings(v_sas, v_norm, top_k=20)
        all_comparisons[concept] = comparison
        all_normalized_vectors[concept] = v_norm

        logger.info(
            f"  Top-20 overlap: {comparison['overlap_pct']:.0f}% "
            f"({comparison['shared_in_top20']}/20)"
        )
        if comparison["new_entries"]:
            logger.info(f"  New in top-20: {comparison['new_entries']}")
        if comparison["dropped_out"]:
            logger.info(f"  Dropped from top-20: {comparison['dropped_out']}")

        logger.info(f"  Feature ranking changes:")
        for d in comparison["comparisons"][:10]:
            norm_rank = d["norm_rank"]
            rank_str = (
                f"→ #{norm_rank}" if isinstance(norm_rank, int) else f"→ {norm_rank}"
            )
            logger.info(
                f"    F{d['feature_id']:4d} ({d['sign']}) "
                f"#{d['orig_rank']:2d} {rank_str:>6s}  "
                f"|w| {abs(d['orig_weight']):.3f} → {abs(d['norm_weight']):.3f}"
            )

    # Save results
    results_path = output_dir / "normalization_results.json"
    with open(results_path, "w") as f:
        json.dump(all_comparisons, f, indent=2, default=str)
    logger.info(f"Saved: {results_path}")

    # Save normalized vectors
    for concept, v_norm in all_normalized_vectors.items():
        vec_path = output_dir / f"{concept}_sas_vectors_whitened.npy"
        np.save(vec_path, v_norm)
        logger.info(f"Saved: {vec_path}")

    plot_ranking_comparison(all_comparisons, output_dir)

    logger.info("")
    logger.info("=" * 72)
    logger.info("  ✓ Whitening Analysis Complete!")
    logger.info("=" * 72)
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
