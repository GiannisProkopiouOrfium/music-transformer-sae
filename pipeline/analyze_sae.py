#!/usr/bin/env python3
"""
Standalone SAE analysis module for SAE pipeline.

This module can be run independently to analyze trained sparse autoencoders
and generate musical interpretations of discovered features.
"""

import argparse
import logging
import sys
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add mmt to path
sys.path.append(str(Path(__file__).parent.parent))


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np

    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def setup_logging(level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


class SparseAutoencoder(torch.nn.Module):
    """Sparse Autoencoder model (for loading)."""

    def __init__(self, input_dim: int, hidden_dim: int, sparsity_coeff: float = 1e-4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coeff = sparsity_coeff

        self.encoder = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x):
        hidden = torch.relu(self.encoder(x))
        reconstructed = self.decoder(hidden)
        return reconstructed, hidden


def load_model(model_path: str, device: torch.device) -> SparseAutoencoder:
    """Load trained SAE model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = SparseAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        sparsity_coeff=checkpoint.get("sparsity_coeff", 1e-4),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def load_activations(activations_path: str, max_samples: int = None) -> np.ndarray:
    """Load activations from HDF5 file."""
    with h5py.File(activations_path, "r") as f:
        activations = f["activations"][:]

        if max_samples and len(activations) > max_samples:
            # Random subsample for analysis
            indices = np.random.choice(len(activations), max_samples, replace=False)
            activations = activations[indices]

    logging.info(f"Loaded activations for analysis: {activations.shape}")
    return activations


def analyze_reconstruction_quality(
    model: SparseAutoencoder, activations: np.ndarray, device: torch.device
) -> Dict[str, float]:
    """Analyze SAE reconstruction quality."""

    activations_tensor = torch.FloatTensor(activations).to(device)

    with torch.no_grad():
        reconstructed, hidden = model(activations_tensor)

    # Convert back to numpy
    activations_np = activations_tensor.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()
    hidden_np = hidden.cpu().numpy()

    # Compute metrics
    mse = np.mean((activations_np - reconstructed_np) ** 2)
    mae = np.mean(np.abs(activations_np - reconstructed_np))

    # Explained variance
    total_var = np.var(activations_np)
    residual_var = np.var(activations_np - reconstructed_np)
    explained_variance = 1 - (residual_var / total_var) if total_var > 0 else 0

    # Sparsity metrics
    sparsity_percent = np.mean(hidden_np == 0) * 100
    l1_sparsity = np.mean(np.sum(np.abs(hidden_np), axis=1))

    # Feature activation statistics
    feature_activations = np.mean(hidden_np > 0, axis=0) * 100
    active_features = np.sum(feature_activations > 1)  # Features active >1% of time
    dead_features = np.sum(feature_activations == 0)  # Features never active

    metrics = {
        "reconstruction_mse": float(mse),
        "reconstruction_mae": float(mae),
        "explained_variance": float(explained_variance),
        "sparsity_percent": float(sparsity_percent),
        "l1_sparsity": float(l1_sparsity),
        "active_features": int(active_features),
        "dead_features": int(dead_features),
        "total_features": int(hidden_np.shape[1]),
    }

    return metrics, hidden_np, feature_activations


def analyze_feature_patterns(
    hidden_activations: np.ndarray, feature_activations: np.ndarray, top_k: int = 20
) -> Dict[str, Any]:
    """Analyze patterns in SAE features."""

    # Find most interesting features (moderate activation frequency)
    interesting_mask = (feature_activations > 1) & (feature_activations < 50)
    interesting_features = np.where(interesting_mask)[0]

    # Sort by selectivity (high variance/mean ratio for non-zero activations)
    selectivity_scores = []
    for feat_idx in interesting_features:
        feat_acts = hidden_activations[:, feat_idx]
        active_mask = feat_acts > 0
        if np.sum(active_mask) > 10:  # Need enough active samples
            mean_act = np.mean(feat_acts[active_mask])
            std_act = np.std(feat_acts[active_mask])
            selectivity = std_act / (mean_act + 1e-8)
            selectivity_scores.append((feat_idx, selectivity))

    # Sort by selectivity
    selectivity_scores.sort(key=lambda x: x[1], reverse=True)
    most_selective = [idx for idx, _ in selectivity_scores[:top_k]]

    # Find most active features
    most_active = np.argsort(feature_activations)[-top_k:][::-1]

    # Find sparsest features (low but non-zero activation)
    sparse_mask = (feature_activations > 0.1) & (feature_activations < 5)
    sparsest_candidates = np.where(sparse_mask)[0]
    if len(sparsest_candidates) > 0:
        sparsest = np.argsort(feature_activations[sparsest_candidates])[:top_k]
        sparsest_features = sparsest_candidates[sparsest]
    else:
        sparsest_features = []

    return {
        "interesting_features": interesting_features.tolist(),
        "most_selective": most_selective,
        "most_active": most_active.tolist(),
        "sparsest_features": sparsest_features.tolist(),
        "num_interesting": len(interesting_features),
    }


def generate_musical_interpretations(
    feature_patterns: Dict[str, Any],
    hidden_activations: np.ndarray,
    model: SparseAutoencoder,
) -> Dict[str, Any]:
    """Generate musical interpretations for discovered features."""

    interpretations = {}

    # Analyze decoder patterns to understand what each feature reconstructs
    decoder_weights = model.decoder.weight.data.cpu().numpy()  # [input_dim, hidden_dim]

    for category, feature_list in feature_patterns.items():
        if category in ["most_selective", "most_active", "sparsest_features"]:
            category_interpretations = {}

            for feat_idx in feature_list[:10]:  # Top 10 features per category
                # Analyze what this feature reconstructs
                decoder_pattern = decoder_weights[:, feat_idx]

                # Find strongest reconstruction weights
                top_weights_idx = np.argsort(np.abs(decoder_pattern))[-5:][::-1]
                top_weights = decoder_pattern[top_weights_idx]

                # Activation statistics
                feat_activations = hidden_activations[:, feat_idx]
                activation_stats = {
                    "mean": float(np.mean(feat_activations)),
                    "std": float(np.std(feat_activations)),
                    "max": float(np.max(feat_activations)),
                    "activation_rate": float(np.mean(feat_activations > 0) * 100),
                    "top_reconstruction_dims": top_weights_idx.tolist(),
                    "top_reconstruction_weights": top_weights.tolist(),
                }

                # Musical interpretation heuristics
                musical_properties = analyze_musical_properties(
                    feat_activations, decoder_pattern, feat_idx
                )

                category_interpretations[str(feat_idx)] = {
                    "activation_stats": activation_stats,
                    "musical_properties": musical_properties,
                }

            interpretations[category] = category_interpretations

    return interpretations


def analyze_musical_properties(
    activations: np.ndarray, decoder_pattern: np.ndarray, feature_idx: int
) -> Dict[str, Any]:
    """Analyze potential musical properties of a feature."""

    # Basic heuristics for musical interpretation
    properties = {
        "feature_type": "unknown",
        "selectivity": "medium",
        "potential_musical_role": [],
    }

    # Activation frequency analysis
    activation_rate = np.mean(activations > 0) * 100

    if activation_rate < 1:
        properties["selectivity"] = "very_sparse"
        properties["potential_musical_role"].append("rare_musical_event")
    elif activation_rate < 5:
        properties["selectivity"] = "sparse"
        properties["potential_musical_role"].append("specific_musical_pattern")
    elif activation_rate < 20:
        properties["selectivity"] = "moderate"
        properties["potential_musical_role"].append("common_musical_element")
    else:
        properties["selectivity"] = "frequent"
        properties["potential_musical_role"].append("general_musical_feature")

    # Decoder pattern analysis
    decoder_norm = np.linalg.norm(decoder_pattern)
    max_weight = np.max(np.abs(decoder_pattern))

    if decoder_norm > np.percentile(np.abs(decoder_pattern), 95):
        properties["feature_type"] = "strong_pattern"

    if max_weight > 3 * np.std(decoder_pattern):
        properties["potential_musical_role"].append("focused_reconstruction")

    # Assign musical interpretation based on heuristics
    if activation_rate < 2 and decoder_norm > 0.5:
        properties["potential_musical_role"].append("chord_or_harmony_specific")
    elif 2 <= activation_rate < 10:
        properties["potential_musical_role"].append("rhythmic_or_melodic_motif")
    elif activation_rate >= 10:
        properties["potential_musical_role"].append("general_texture_or_timbre")

    return properties


def create_visualizations(
    metrics: Dict[str, Any],
    hidden_activations: np.ndarray,
    feature_activations: np.ndarray,
    feature_patterns: Dict[str, Any],
    output_dir: Path,
):
    """Create analysis visualizations."""

    plt.figure(figsize=(15, 10))

    # Plot 1: Feature activation frequency distribution
    plt.subplot(2, 3, 1)
    plt.hist(feature_activations, bins=50, alpha=0.7, color="skyblue")
    plt.xlabel("Activation Frequency (%)")
    plt.ylabel("Number of Features")
    plt.title("Feature Activation Distribution")
    plt.axvline(
        np.mean(feature_activations),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(feature_activations):.1f}%",
    )
    plt.legend()

    # Plot 2: Sparsity visualization
    plt.subplot(2, 3, 2)
    sample_activations = hidden_activations[:200, :100]  # Sample for visualization
    plt.imshow(sample_activations.T, aspect="auto", cmap="viridis")
    plt.xlabel("Sample Index")
    plt.ylabel("Feature Index (first 100)")
    plt.title("Feature Activation Heatmap")
    plt.colorbar()

    # Plot 3: Feature selectivity
    plt.subplot(2, 3, 3)
    # Calculate selectivity for all features
    selectivity = []
    for i in range(hidden_activations.shape[1]):
        feat_acts = hidden_activations[:, i]
        if np.sum(feat_acts > 0) > 5:
            selectivity.append(np.std(feat_acts) / (np.mean(feat_acts) + 1e-8))
        else:
            selectivity.append(0)

    plt.scatter(feature_activations, selectivity, alpha=0.6, s=20)
    plt.xlabel("Activation Frequency (%)")
    plt.ylabel("Selectivity (std/mean)")
    plt.title("Feature Selectivity vs Frequency")

    # Plot 4: Reconstruction quality
    plt.subplot(2, 3, 4)
    metrics_to_plot = ["reconstruction_mse", "explained_variance", "sparsity_percent"]
    values = [metrics[key] for key in metrics_to_plot]
    plt.bar(range(len(metrics_to_plot)), values)
    plt.xticks(
        range(len(metrics_to_plot)), ["MSE", "Expl. Var.", "Sparsity %"], rotation=45
    )
    plt.title("Reconstruction Metrics")

    # Plot 5: Feature categories
    plt.subplot(2, 3, 5)
    categories = ["Dead", "Active", "Interesting"]
    counts = [
        metrics["dead_features"],
        metrics["active_features"],
        feature_patterns["num_interesting"],
    ]
    plt.pie(counts, labels=categories, autopct="%1.1f%%")
    plt.title("Feature Categories")

    # Plot 6: Top features comparison
    plt.subplot(2, 3, 6)
    top_selective = feature_patterns["most_selective"][:10]
    top_active = feature_patterns["most_active"][:10]

    plt.scatter(
        range(len(top_selective)),
        [feature_activations[i] for i in top_selective],
        label="Most Selective",
        alpha=0.7,
        s=30,
    )
    plt.scatter(
        range(len(top_active)),
        [feature_activations[i] for i in top_active],
        label="Most Active",
        alpha=0.7,
        s=30,
    )
    plt.xlabel("Feature Rank")
    plt.ylabel("Activation Frequency (%)")
    plt.title("Top Features Comparison")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "sae_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    logging.info(f"Visualizations saved to: {output_dir / 'sae_analysis.png'}")


def main():
    """Analyze trained Sparse Autoencoder."""
    parser = argparse.ArgumentParser(
        description="Analyze trained SAE and generate musical interpretations"
    )
    parser.add_argument(
        "--model", required=True, help="Path to trained SAE model (.pt file)"
    )
    parser.add_argument(
        "--activations", required=True, help="Path to original activations HDF5 file"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for analysis results"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum samples to analyze (default: 10000)",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=20,
        help="Number of top features to analyze (default: 20)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for analysis",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("🎵 Sparse Autoencoder Analysis")
    logger.info(f"Model: {args.model}")
    logger.info(f"Activations: {args.activations}")
    logger.info(f"Output: {args.output}")

    try:
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine device
        if args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)

        logger.info(f"Using device: {device}")

        # Load model
        logger.info("📊 Loading SAE model...")
        model, checkpoint = load_model(args.model, device)
        logger.info(f"Model architecture: {checkpoint.get('architecture', 'Unknown')}")

        # Load activations
        logger.info("📊 Loading activations...")
        activations = load_activations(args.activations, args.max_samples)

        # Analyze reconstruction quality
        logger.info("🔄 Analyzing reconstruction quality...")
        metrics, hidden_activations, feature_activations = (
            analyze_reconstruction_quality(model, activations, device)
        )

        # Analyze feature patterns
        logger.info("🔄 Analyzing feature patterns...")
        feature_patterns = analyze_feature_patterns(
            hidden_activations, feature_activations, args.top_features
        )

        # Generate musical interpretations
        logger.info("🎼 Generating musical interpretations...")
        interpretations = generate_musical_interpretations(
            feature_patterns, hidden_activations, model
        )

        # Create visualizations
        logger.info("📊 Creating visualizations...")
        create_visualizations(
            metrics,
            hidden_activations,
            feature_activations,
            feature_patterns,
            output_dir,
        )

        # Save results
        results = {
            "metrics": metrics,
            "feature_patterns": feature_patterns,
            "interpretations": interpretations,
            "model_info": {
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
                "expansion_factor": model.hidden_dim / model.input_dim,
                "sparsity_coeff": model.sparsity_coeff,
            },
        }

        # Save as JSON (convert numpy types to native Python types)
        results_path = output_dir / "analysis_results.json"
        with open(results_path, "w") as f:
            json.dump(convert_numpy_types(results), f, indent=2)

        # Save detailed metrics as HDF5
        metrics_path = output_dir / "detailed_metrics.h5"
        with h5py.File(metrics_path, "w") as f:
            f.create_dataset("hidden_activations", data=hidden_activations)
            f.create_dataset("feature_activations", data=feature_activations)
            f.create_dataset("original_activations", data=activations)

            for key, value in metrics.items():
                f.attrs[key] = value

        # Print summary
        logger.info("✅ Analysis completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info("📊 Summary:")
        logger.info(f"  Total Features: {metrics['total_features']}")
        logger.info(f"  Active Features: {metrics['active_features']}")
        logger.info(f"  Dead Features: {metrics['dead_features']}")
        logger.info(f"  Sparsity: {metrics['sparsity_percent']:.1f}%")
        logger.info(f"  Reconstruction MSE: {metrics['reconstruction_mse']:.4f}")
        logger.info(f"  Explained Variance: {metrics['explained_variance']:.1%}")

        # Print interesting features
        print("\n" + "=" * 60)
        print("🎼 DISCOVERED MUSICAL FEATURES:")
        print("=" * 60)

        for category, features in feature_patterns.items():
            if (
                category in ["most_selective", "most_active", "sparsest_features"]
                and features
            ):
                print(f"\n{category.replace('_', ' ').title()}:")
                for feat_idx in features[:5]:  # Top 5 per category
                    activation_rate = feature_activations[feat_idx]
                    print(
                        f"  Feature {feat_idx:4d}: {activation_rate:5.1f}% activation"
                    )

        print(f"\n📊 Detailed results: {results_path}")
        print(f"📊 Visualizations: {output_dir / 'sae_analysis.png'}")

    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
