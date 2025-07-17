"""Analysis pipeline for SAE interpretability."""

import logging
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
import json

# Add mmt to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.train import SparseAutoencoder
from interpretation.interpret_music_sae import (
    analyze_sae_features,
    create_feature_visualizations,
)
from interpretation.musical_context_analysis import analyze_musical_contexts
from utils.storage import HDF5Manager


def load_trained_sae(model_path: str, device: torch.device) -> SparseAutoencoder:
    """Load trained SAE model from checkpoint."""
    logger = logging.getLogger(__name__)

    logger.info(f"Loading SAE model from {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint["model_config"]

    model = SparseAutoencoder(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        l1_alpha=model_config["l1_alpha"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        f"Loaded SAE model: {model_config['input_dim']} -> {model_config['hidden_dim']} -> {model_config['input_dim']}"
    )
    return model


def analyze_feature_activations(
    model: SparseAutoencoder,
    activations: np.ndarray,
    tokens: np.ndarray = None,
    sample_size: int = 10000,
) -> Dict[str, Any]:
    """Analyze SAE feature activations."""
    logger = logging.getLogger(__name__)

    logger.info("Analyzing feature activations...")

    # Sample activations for analysis
    if len(activations) > sample_size:
        indices = np.random.choice(len(activations), sample_size, replace=False)
        sample_activations = activations[indices]
        sample_tokens = tokens[indices] if tokens is not None else None
    else:
        sample_activations = activations
        sample_tokens = tokens

    # Convert to torch tensor
    sample_tensor = torch.from_numpy(sample_activations).float()

    # Get feature activations
    with torch.no_grad():
        feature_activations = model.get_feature_activations(sample_tensor).numpy()

    # Analyze sparsity
    sparsity_per_feature = (feature_activations == 0).mean(axis=0)
    active_features = np.sum(
        sparsity_per_feature < 0.99
    )  # Features that activate > 1% of time

    # Analyze activation magnitudes
    mean_activations = np.mean(feature_activations, axis=0)
    max_activations = np.max(feature_activations, axis=0)
    std_activations = np.std(feature_activations, axis=0)

    # Find top activating features
    top_features = np.argsort(mean_activations)[-20:][::-1]

    results = {
        "num_features": feature_activations.shape[1],
        "active_features": int(active_features),
        "sparsity_overall": float(np.mean(sparsity_per_feature)),
        "sparsity_per_feature": sparsity_per_feature.tolist(),
        "mean_activations": mean_activations.tolist(),
        "max_activations": max_activations.tolist(),
        "std_activations": std_activations.tolist(),
        "top_features": top_features.tolist(),
        "sample_size": len(sample_activations),
    }

    logger.info(
        f"Analysis completed: {active_features}/{feature_activations.shape[1]} active features"
    )
    logger.info(f"Overall sparsity: {results['sparsity_overall']:.3f}")

    return results


def create_analysis_visualizations(
    analysis_results: Dict[str, Any], output_dir: str
) -> List[str]:
    """Create visualization plots for SAE analysis."""
    logger = logging.getLogger(__name__)

    logger.info("Creating analysis visualizations...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    saved_plots = []

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # 1. Sparsity distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sparsity_data = analysis_results["sparsity_per_feature"]
    ax.hist(sparsity_data, bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Sparsity (fraction of zero activations)")
    ax.set_ylabel("Number of features")
    ax.set_title("Distribution of Feature Sparsity")
    ax.axvline(
        np.mean(sparsity_data),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(sparsity_data):.3f}",
    )
    ax.legend()

    sparsity_plot = output_path / "feature_sparsity_distribution.png"
    plt.savefig(sparsity_plot, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(str(sparsity_plot))

    # 2. Activation magnitude distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    mean_acts = analysis_results["mean_activations"]
    max_acts = analysis_results["max_activations"]

    # Mean activations
    ax1.hist(mean_acts, bins=50, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Mean activation magnitude")
    ax1.set_ylabel("Number of features")
    ax1.set_title("Distribution of Mean Feature Activations")
    ax1.set_yscale("log")

    # Max activations
    ax2.hist(max_acts, bins=50, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Max activation magnitude")
    ax2.set_ylabel("Number of features")
    ax2.set_title("Distribution of Max Feature Activations")
    ax2.set_yscale("log")

    activation_plot = output_path / "activation_magnitude_distributions.png"
    plt.savefig(activation_plot, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(str(activation_plot))

    # 3. Top features analysis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    top_features = analysis_results["top_features"][:10]
    top_means = [analysis_results["mean_activations"][i] for i in top_features]
    top_sparsity = [analysis_results["sparsity_per_feature"][i] for i in top_features]

    x = range(len(top_features))
    bars = ax.bar(x, top_means, alpha=0.7)

    # Color bars by sparsity
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(1 - top_sparsity[i]))

    ax.set_xlabel("Top Features (by mean activation)")
    ax.set_ylabel("Mean activation magnitude")
    ax.set_title("Top 10 Most Active Features")
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{f}" for f in top_features], rotation=45)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Activation Frequency (1 - sparsity)")

    top_features_plot = output_path / "top_features_analysis.png"
    plt.savefig(top_features_plot, dpi=300, bbox_inches="tight")
    plt.close()
    saved_plots.append(str(top_features_plot))

    logger.info(f"Created {len(saved_plots)} visualization plots")
    return saved_plots


def generate_analysis_report(
    analysis_results: Dict[str, Any],
    experiment_config: Dict[str, Any],
    output_file: str,
) -> None:
    """Generate comprehensive analysis report."""
    logger = logging.getLogger(__name__)

    logger.info(f"Generating analysis report: {output_file}")

    with open(output_file, "w") as f:
        f.write("# SAE Analysis Report\n\n")

        # Experiment info
        f.write("## Experiment Configuration\n")
        f.write(f"- Experiment Name: {experiment_config['experiment']['name']}\n")
        f.write(f"- Model Layer: {experiment_config['model']['layer_idx']}\n")
        f.write(f"- SAE Architecture: {analysis_results['num_features']} features\n\n")

        # Sparsity analysis
        f.write("## Sparsity Analysis\n")
        f.write(f"- Total Features: {analysis_results['num_features']}\n")
        f.write(f"- Active Features: {analysis_results['active_features']}\n")
        f.write(
            f"- Active Ratio: {analysis_results['active_features']/analysis_results['num_features']:.3f}\n"
        )
        f.write(f"- Overall Sparsity: {analysis_results['sparsity_overall']:.3f}\n\n")

        # Top features
        f.write("## Top Active Features\n")
        top_features = analysis_results["top_features"][:10]
        for i, feature_idx in enumerate(top_features):
            mean_act = analysis_results["mean_activations"][feature_idx]
            sparsity = analysis_results["sparsity_per_feature"][feature_idx]
            f.write(
                f"{i+1}. Feature {feature_idx}: Mean={mean_act:.4f}, Sparsity={sparsity:.3f}\n"
            )

        f.write("\n")

        # Statistics
        f.write("## Statistical Summary\n")
        f.write(
            f"- Mean activation (all features): {np.mean(analysis_results['mean_activations']):.6f}\n"
        )
        f.write(
            f"- Std activation (all features): {np.mean(analysis_results['std_activations']):.6f}\n"
        )
        f.write(
            f"- Max activation observed: {np.max(analysis_results['max_activations']):.6f}\n"
        )
        f.write(f"- Sample size analyzed: {analysis_results['sample_size']}\n")

    logger.info("Analysis report generated successfully")


def run_analysis_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete SAE analysis pipeline."""
    logger = logging.getLogger(__name__)

    logger.info("Initializing SAE analysis pipeline")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Analysis device: {device}")

    # Load trained SAE model
    models_dir = Path(config["experiment"]["directories"]["models"])
    model_path = models_dir / "best_sae_model.pt"

    if not model_path.exists():
        model_path = models_dir / "final_sae_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"No trained SAE model found in {models_dir}")

    model = load_trained_sae(str(model_path), device)

    # Load activation data
    extraction_config = config["extraction"]
    activations_dir = config["experiment"]["directories"]["activations"]
    activation_file = Path(activations_dir) / extraction_config["output_file"]

    logger.info(f"Loading activations from {activation_file}")

    with h5py.File(activation_file, "r") as f:
        activations = f["activations"][:]
        tokens = f["tokens"][:] if "tokens" in f else None

    # Run feature analysis
    analysis_config = config.get("analysis", {})
    sample_size = analysis_config.get("sample_size", 10000)

    analysis_results = analyze_feature_activations(
        model, activations, tokens, sample_size
    )

    # Create visualizations
    results_dir = config["experiment"]["directories"]["results"]
    plots = create_analysis_visualizations(analysis_results, results_dir)

    # Generate report
    report_file = Path(results_dir) / "sae_analysis_report.md"
    generate_analysis_report(analysis_results, config, str(report_file))

    # Save analysis results
    results_file = Path(results_dir) / "analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(analysis_results, f, indent=2)

    results = {
        "experiment_name": config["experiment"]["name"],
        "model_path": str(model_path),
        "num_features": analysis_results["num_features"],
        "active_features": analysis_results["active_features"],
        "sparsity_overall": analysis_results["sparsity_overall"],
        "visualization_plots": plots,
        "report_file": str(report_file),
        "results_file": str(results_file),
        "analysis_config": analysis_config,
    }

    logger.info("SAE analysis pipeline completed successfully")
    return results
