import argparse
import pathlib
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze extracted activations")
    parser.add_argument(
        "-f",
        "--file",
        type=pathlib.Path,
        required=True,
        help="HDF5 file containing activations",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=pathlib.Path,
        required=True,
        help="Output directory for plots",
    )
    parser.add_argument(
        "-l",
        "--layer",
        type=int,
        default=None,
        help="Layer to analyze (if None, analyzes all)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Maximum number of samples for visualization",
    )
    return parser.parse_args()


def analyze_layer_activations(activations, layer_name, output_dir, max_samples=10000):
    """Analyze activations from a single layer."""
    logging.info(f"Analyzing {layer_name}: {activations.shape}")

    # Subsample for visualization if needed
    if activations.shape[0] > max_samples:
        indices = np.random.choice(activations.shape[0], max_samples, replace=False)
        activations_subset = activations[indices]
    else:
        activations_subset = activations

    # Basic statistics
    mean_activation = np.mean(activations, axis=0)
    std_activation = np.std(activations, axis=0)
    sparsity = np.mean(activations == 0)

    logging.info(f"  Mean activation magnitude: {np.mean(np.abs(mean_activation)):.4f}")
    logging.info(f"  Std activation magnitude: {np.mean(std_activation):.4f}")
    logging.info(f"  Sparsity (fraction of zeros): {sparsity:.4f}")

    # Activation distribution
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(activations.flatten(), bins=100, alpha=0.7)
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.title(f"{layer_name}: Activation Distribution")
    plt.yscale("log")

    plt.subplot(1, 3, 2)
    plt.plot(mean_activation)
    plt.xlabel("Neuron Index")
    plt.ylabel("Mean Activation")
    plt.title(f"{layer_name}: Mean Activations")

    plt.subplot(1, 3, 3)
    plt.plot(std_activation)
    plt.xlabel("Neuron Index")
    plt.ylabel("Std Activation")
    plt.title(f"{layer_name}: Activation Std")

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{layer_name}_statistics.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    # PCA analysis
    logging.info("  Running PCA...")
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(activations_subset)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Principal Component")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"{layer_name}: PCA Explained Variance")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=1)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"{layer_name}: PCA Projection")

    plt.tight_layout()
    plt.savefig(output_dir / f"{layer_name}_pca.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Correlation matrix (sample of neurons)
    if activations.shape[1] > 100:
        # Sample neurons for correlation analysis
        neuron_indices = np.random.choice(activations.shape[1], 100, replace=False)
        corr_matrix = np.corrcoef(activations[:, neuron_indices].T)
    else:
        corr_matrix = np.corrcoef(activations.T)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True)
    plt.title(f"{layer_name}: Neuron Correlations")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"{layer_name}_correlations.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    return {
        "mean_magnitude": np.mean(np.abs(mean_activation)),
        "std_magnitude": np.mean(std_activation),
        "sparsity": sparsity,
        "pca_variance_ratio": pca.explained_variance_ratio_,
        "shape": activations.shape,
    }


def main():
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load activations
    logging.info(f"Loading activations from {args.file}")

    results = {}
    with h5py.File(args.file, "r") as f:
        available_layers = list(f.keys())
        logging.info(f"Available layers: {available_layers}")

        if args.layer is not None:
            layer_key = f"layer_{args.layer}"
            if layer_key in f:
                activations = f[layer_key][:]
                results[layer_key] = analyze_layer_activations(
                    activations, layer_key, args.output_dir, args.max_samples
                )
            else:
                logging.error(f"Layer {args.layer} not found in file")
        else:
            # Analyze all layers
            for layer_key in available_layers:
                activations = f[layer_key][:]
                results[layer_key] = analyze_layer_activations(
                    activations, layer_key, args.output_dir, args.max_samples
                )

    # Save summary
    logging.info("Analysis complete. Summary:")
    for layer, stats in results.items():
        logging.info(
            f"  {layer}: {stats['shape']} tokens, "
            f"sparsity={stats['sparsity']:.3f}, "
            f"mean_mag={stats['mean_magnitude']:.4f}"
        )


if __name__ == "__main__":
    main()
