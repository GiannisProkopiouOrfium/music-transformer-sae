"""Visualize Activation Space with PCA and KDE.

This script visualizes the separation between high and low concept activations
using PCA projection and 2D Kernel Density Estimation.

Usage:
    python experiments/visualize_activation_space.py \\
        --concept average_pitch \\
        --activations_dir steering_interventions/outputs/activations \\
        --output_dir steering_interventions/experiments/results/activation_viz
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directories to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def setup_logging(output_dir: pathlib.Path):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "visualize_activation_space.log"),
            logging.StreamHandler(),
        ],
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize activation space")
    parser.add_argument(
        "--concept", type=str, required=True, help="Concept name (e.g., average_pitch)"
    )
    parser.add_argument(
        "--activations_dir",
        type=pathlib.Path,
        default=pathlib.Path("steering_interventions/outputs/activations"),
        help="Directory containing activation .h5 files",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/experiments/results/activation_viz"
        ),
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to visualize (comma-separated or 'all')",
    )
    return parser.parse_args()


def load_activations(h5_file: pathlib.Path, layer: int) -> np.ndarray:
    """Load activations for a specific layer from H5 file."""
    with h5py.File(h5_file, "r") as f:
        layer_key = f"layer_{layer}"
        if layer_key not in f:
            raise ValueError(f"Layer {layer} not found in {h5_file}")
        activations = f[layer_key][:]
    return activations


def compute_pca_projection(
    high_acts: np.ndarray, low_acts: np.ndarray, n_components: int = 2
) -> Tuple[PCA, np.ndarray, np.ndarray]:
    """Compute PCA projection for both high and low activations."""
    # Combine for fitting
    all_acts = np.vstack([high_acts, low_acts])

    # Fit PCA
    pca = PCA(n_components=n_components)
    pca.fit(all_acts)

    # Transform
    high_2d = pca.transform(high_acts)
    low_2d = pca.transform(low_acts)

    return pca, high_2d, low_2d


def compute_separation_metrics(
    high_2d: np.ndarray, low_2d: np.ndarray
) -> Dict[str, float]:
    """Compute separation metrics between high and low clusters."""
    # Mean positions
    mean_high = high_2d.mean(axis=0)
    mean_low = low_2d.mean(axis=0)

    # Euclidean distance
    euclidean_dist = np.linalg.norm(mean_high - mean_low)

    # Cosine similarity
    cos_sim = cosine_similarity([mean_high], [mean_low])[0, 0]

    # Overlap coefficient (using KDE)
    try:
        kde_high = gaussian_kde(high_2d.T)
        kde_low = gaussian_kde(low_2d.T)

        # Sample points from the combined range
        x_range = np.linspace(
            min(high_2d[:, 0].min(), low_2d[:, 0].min()),
            max(high_2d[:, 0].max(), low_2d[:, 0].max()),
            100,
        )
        y_range = np.linspace(
            min(high_2d[:, 1].min(), low_2d[:, 1].min()),
            max(high_2d[:, 1].max(), low_2d[:, 1].max()),
            100,
        )
        X, Y = np.meshgrid(x_range, y_range)
        positions = np.vstack([X.ravel(), Y.ravel()])

        # Compute densities
        Z_high = kde_high(positions).reshape(X.shape)
        Z_low = kde_low(positions).reshape(X.shape)

        # Overlap coefficient (intersection over minimum)
        overlap = np.sum(np.minimum(Z_high, Z_low)) / np.sum(
            np.minimum(np.sum(Z_high), np.sum(Z_low))
        )
    except Exception as e:
        logging.warning(f"Could not compute overlap coefficient: {e}")
        overlap = np.nan

    return {
        "euclidean_distance": float(euclidean_dist),
        "cosine_similarity": float(cos_sim),
        "overlap_coefficient": float(overlap),
    }


def plot_kde_contours(
    high_2d: np.ndarray,
    low_2d: np.ndarray,
    pca: PCA,
    layer: int,
    output_path: pathlib.Path,
):
    """Create KDE contour plot for a single layer."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter points
    ax.scatter(
        high_2d[:, 0],
        high_2d[:, 1],
        alpha=0.3,
        c="red",
        s=10,
        label="High Concept",
    )
    ax.scatter(
        low_2d[:, 0], low_2d[:, 1], alpha=0.3, c="blue", s=10, label="Low Concept"
    )

    # KDE contours
    try:
        for data, color, label in [
            (high_2d, "red", "High"),
            (low_2d, "blue", "Low"),
        ]:
            kde = gaussian_kde(data.T)
            x_range = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
            y_range = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
            X, Y = np.meshgrid(x_range, y_range)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            ax.contour(X, Y, Z, colors=color, alpha=0.7, levels=5, linewidths=1.5)
    except Exception as e:
        logging.warning(f"Could not plot KDE contours: {e}")

    # Add steering vector
    mean_high = high_2d.mean(axis=0)
    mean_low = low_2d.mean(axis=0)
    ax.arrow(
        mean_low[0],
        mean_low[1],
        mean_high[0] - mean_low[0],
        mean_high[1] - mean_low[1],
        head_width=0.3,
        head_length=0.3,
        fc="green",
        ec="green",
        linewidth=2,
        label="Steering Vector",
    )

    # Labels
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    ax.set_title(
        f"Layer {layer} - PCA Projection with KDE", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_layers_grid(
    projections: Dict[int, Tuple[np.ndarray, np.ndarray, PCA]],
    output_path: pathlib.Path,
):
    """Create grid of subplots for all layers."""
    n_layers = len(projections)
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for idx, (layer, (high_2d, low_2d, pca)) in enumerate(sorted(projections.items())):
        ax = axes[idx]

        # Scatter with smaller points
        ax.scatter(high_2d[:, 0], high_2d[:, 1], alpha=0.2, c="red", s=5)
        ax.scatter(low_2d[:, 0], low_2d[:, 1], alpha=0.2, c="blue", s=5)

        # Simplified contours
        try:
            for data, color in [(high_2d, "red"), (low_2d, "blue")]:
                kde = gaussian_kde(data.T)
                x_range = np.linspace(data[:, 0].min(), data[:, 0].max(), 50)
                y_range = np.linspace(data[:, 1].min(), data[:, 1].max(), 50)
                X, Y = np.meshgrid(x_range, y_range)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                ax.contour(X, Y, Z, colors=color, alpha=0.5, levels=3, linewidths=1)
        except Exception:
            pass

        ax.set_title(f"Layer {layer}", fontsize=10)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_separation_metrics(
    separation_metrics: Dict[int, Dict], output_path: pathlib.Path
):
    """Plot separation metrics vs layer."""
    layers = sorted(separation_metrics.keys())
    euclidean = [separation_metrics[l]["euclidean_distance"] for l in layers]
    cosine = [separation_metrics[l]["cosine_similarity"] for l in layers]
    overlap = [separation_metrics[l]["overlap_coefficient"] for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Euclidean distance
    axes[0].plot(layers, euclidean, "o-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Layer", fontsize=12)
    axes[0].set_ylabel("Euclidean Distance", fontsize=12)
    axes[0].set_title(
        "Cluster Separation (Higher = Better)", fontsize=12, fontweight="bold"
    )
    axes[0].grid(True, alpha=0.3)

    # Cosine similarity
    axes[1].plot(layers, cosine, "o-", linewidth=2, markersize=8, color="orange")
    axes[1].set_xlabel("Layer", fontsize=12)
    axes[1].set_ylabel("Cosine Similarity", fontsize=12)
    axes[1].set_title("Direction Alignment (-1 to 1)", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Overlap coefficient
    axes[2].plot(layers, overlap, "o-", linewidth=2, markersize=8, color="green")
    axes[2].set_xlabel("Layer", fontsize=12)
    axes[2].set_ylabel("Overlap Coefficient", fontsize=12)
    axes[2].set_title(
        "Distribution Overlap (Lower = Better)", fontsize=12, fontweight="bold"
    )
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir)

    logging.info("=" * 80)
    logging.info("ACTIVATION SPACE VISUALIZATION")
    logging.info("=" * 80)
    logging.info(f"Concept: {args.concept}")
    logging.info(f"Activations dir: {args.activations_dir}")

    # Find activation files
    high_file = args.activations_dir / f"high_{args.concept}_activations.h5"
    low_file = args.activations_dir / f"low_{args.concept}_activations.h5"

    if not high_file.exists():
        raise FileNotFoundError(f"High activations file not found: {high_file}")
    if not low_file.exists():
        raise FileNotFoundError(f"Low activations file not found: {low_file}")

    logging.info(f"High file: {high_file}")
    logging.info(f"Low file: {low_file}")

    # Determine layers to process
    with h5py.File(high_file, "r") as f:
        available_layers = sorted([int(k.split("_")[1]) for k in f.keys()])

    if args.layers == "all":
        layers = available_layers
    else:
        layers = [int(l) for l in args.layers.split(",")]

    logging.info(f"Processing layers: {layers}")

    # Process each layer
    projections = {}
    separation_metrics = {}

    for layer in layers:
        logging.info(f"\nProcessing layer {layer}...")

        # Load activations
        high_acts = load_activations(high_file, layer)
        low_acts = load_activations(low_file, layer)

        logging.info(f"  High shape: {high_acts.shape}")
        logging.info(f"  Low shape: {low_acts.shape}")

        # Compute PCA
        pca, high_2d, low_2d = compute_pca_projection(high_acts, low_acts)
        projections[layer] = (high_2d, low_2d, pca)

        logging.info(
            f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, "
            f"PC2={pca.explained_variance_ratio_[1]:.3f}, "
            f"Total={pca.explained_variance_ratio_[:2].sum():.3f}"
        )

        # Compute separation metrics
        sep_metrics = compute_separation_metrics(high_2d, low_2d)
        separation_metrics[layer] = sep_metrics

        logging.info(f"  Euclidean distance: {sep_metrics['euclidean_distance']:.3f}")
        logging.info(f"  Cosine similarity: {sep_metrics['cosine_similarity']:.3f}")
        logging.info(f"  Overlap coefficient: {sep_metrics['overlap_coefficient']:.3f}")

        # Plot individual layer
        output_path = args.output_dir / f"layer_{layer}_kde.png"
        plot_kde_contours(high_2d, low_2d, pca, layer, output_path)
        logging.info(f"  Saved plot: {output_path}")

    # Plot grid of all layers
    logging.info("\nCreating summary visualizations...")
    grid_path = args.output_dir / f"{args.concept}_all_layers_grid.png"
    plot_all_layers_grid(projections, grid_path)
    logging.info(f"Saved grid plot: {grid_path}")

    # Plot separation metrics
    metrics_path = args.output_dir / f"{args.concept}_separation_metrics.png"
    plot_separation_metrics(separation_metrics, metrics_path)
    logging.info(f"Saved metrics plot: {metrics_path}")

    # Save metrics to JSON
    metrics_json = args.output_dir / f"{args.concept}_separation_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump({"concept": args.concept, "layers": separation_metrics}, f, indent=2)
    logging.info(f"Saved metrics JSON: {metrics_json}")

    logging.info("\nVisualization complete!")


if __name__ == "__main__":
    main()
