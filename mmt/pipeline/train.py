"""
SAE Training Pipeline

Trains sparse autoencoders on extracted transformer activations.
"""

import logging
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Tuple, Optional
import sys

# Add parent directory to path to import mmt modules
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from sae.sae_data import create_sae_dataloader
from .config import PipelineConfig


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for transformer activation analysis.

    Features:
    - L1 sparsity penalty or top-k sparsity
    - Normalized decoder weights
    - Optional bias terms
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsity_coeff: float = 1e-4,
        k_sparse: Optional[int] = None,
        use_bias: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coeff = sparsity_coeff
        self.k_sparse = k_sparse
        self.use_bias = use_bias

        # Encoder and decoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        # Kaiming initialization for encoder
        nn.init.kaiming_uniform_(self.encoder.weight, a=0, mode="fan_in")
        if self.use_bias:
            nn.init.zeros_(self.encoder.bias)

        # Initialize decoder and normalize
        nn.init.kaiming_uniform_(self.decoder.weight, a=0, mode="fan_out")
        self._normalize_decoder()

    def _normalize_decoder(self):
        """Normalize decoder columns to unit norm."""
        with torch.no_grad():
            norms = torch.norm(self.decoder.weight, dim=0, keepdim=True)
            self.decoder.weight.div_(norms + 1e-8)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input activations [batch_size, input_dim]

        Returns:
            reconstructed: Reconstructed activations [batch_size, input_dim]
            hidden: Hidden activations [batch_size, hidden_dim]
        """
        # Encode
        hidden = self.encoder(x)
        hidden = torch.relu(hidden)  # ReLU activation for sparsity

        # Apply sparsity constraint
        if self.k_sparse is not None:
            hidden = self._apply_topk_sparsity(hidden, self.k_sparse)

        # Decode
        reconstructed = self.decoder(hidden)

        return reconstructed, hidden

    def _apply_topk_sparsity(self, hidden: torch.Tensor, k: int) -> torch.Tensor:
        """Apply top-k sparsity constraint."""
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(hidden, k, dim=-1)

        # Create sparse tensor
        sparse_hidden = torch.zeros_like(hidden)
        sparse_hidden.scatter_(-1, topk_indices, topk_values)

        return sparse_hidden

    def compute_loss(
        self, x: torch.Tensor, reconstructed: torch.Tensor, hidden: torch.Tensor
    ) -> dict:
        """
        Compute loss components.

        Args:
            x: Original activations
            reconstructed: Reconstructed activations
            hidden: Hidden activations

        Returns:
            Dictionary with loss components
        """
        # Reconstruction loss (MSE)
        recon_loss = torch.mean((x - reconstructed) ** 2)

        # Sparsity loss
        if self.k_sparse is None:
            # L1 sparsity penalty
            sparsity_loss = self.sparsity_coeff * torch.mean(torch.abs(hidden))
        else:
            # For top-k, sparsity is enforced architecturally
            sparsity_loss = torch.tensor(0.0, device=x.device)

        # Total loss
        total_loss = recon_loss + sparsity_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "sparsity_loss": sparsity_loss,
            "sparsity_ratio": torch.mean((hidden > 0).float()),
        }


def setup_device(config: PipelineConfig) -> torch.device:
    """Setup compute device."""
    if config.sae.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.sae.device)

    logging.info(f"Using device: {device}")
    return device


def load_activations_data(activations_path: pathlib.Path, config: PipelineConfig):
    """Load activations and create data loaders."""
    logging.info(f"Loading activations from: {activations_path}")

    if not activations_path.exists():
        raise FileNotFoundError(f"Activations file not found: {activations_path}")

    # Find the layer key
    with h5py.File(activations_path, "r") as f:
        available_keys = list(f.keys())
        logging.info(f"Available activation layers: {available_keys}")

        # Use first layer if multiple available
        if len(available_keys) == 1:
            layer_key = available_keys[0]
        else:
            # Prefer the middle layer from extraction config
            target_layer = (
                config.extraction.layers[0] if config.extraction.layers else 3
            )
            layer_key = f"layer_{target_layer}"
            if layer_key not in available_keys:
                layer_key = available_keys[0]

        logging.info(f"Using layer: {layer_key}")

    # Get subsample parameters from config
    train_subsample = getattr(config.sae, 'subsample_training', None)
    val_subsample = getattr(config.sae, 'subsample_validation', 5000)
    
    # Create data loaders with intelligent subsampling
    train_loader, train_info = create_sae_dataloader(
        str(activations_path),
        layer_key=layer_key,
        batch_size=config.sae.batch_size,
        shuffle=True,
        subsample=train_subsample,  # Use config-specified subsample for speed
        num_workers=0,  # Avoid multiprocessing issues
        memory_efficient=True,  # Enable memory-efficient loading for large datasets
    )

    val_loader, val_info = create_sae_dataloader(
        str(activations_path),
        layer_key=layer_key,
        batch_size=config.sae.batch_size,
        shuffle=False,
        subsample=val_subsample,  # Config-specified validation set size
        num_workers=0,  # Avoid multiprocessing issues
        memory_efficient=True,  # Enable memory-efficient loading
    )

    # Extract input dimension from shape
    input_dim = train_info["shape"][1]  # Shape is [num_samples, input_dim]
    
    # Log data usage information
    total_samples = train_info["shape"][0]
    logging.info(f"Created data loaders with input dimension: {input_dim}")
    logging.info(f"Training data: {total_samples:,} samples (subsampled from larger dataset)")
    logging.info(f"Validation data: {val_info['shape'][0]:,} samples")
    
    # Estimate training time
    batches_per_epoch = len(train_loader)
    estimated_minutes = (batches_per_epoch * config.sae.num_epochs * 3) // 60  # ~3 sec per batch
    logging.info(f"Estimated training time: ~{estimated_minutes} minutes ({batches_per_epoch} batches/epoch)")
    print(f"📊 Data loaded: {total_samples:,} training samples, {val_info['shape'][0]:,} validation samples")
    print(f"⏱️ Estimated training time: ~{estimated_minutes} minutes")

    return train_loader, val_loader, input_dim


def train_sae_epoch(
    model, train_loader, optimizer, device, gradient_accumulation_steps=1
):
    """Train SAE for one epoch with gradient accumulation and memory optimization."""
    model.train()
    total_losses = []
    recon_losses = []
    sparsity_losses = []
    sparsity_ratios = []

    # Clear cache to free up memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    accumulation_loss = 0.0
    
    # Optimize for speed: use autocast for mixed precision and compile model
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device, non_blocking=True)

        # Mixed precision forward pass for speed
        if use_amp:
            with torch.amp.autocast('cuda'):
                reconstructed, hidden = model(batch)
                losses = model.compute_loss(batch, reconstructed, hidden)
                scaled_loss = losses["total_loss"] / gradient_accumulation_steps
        else:
            reconstructed, hidden = model(batch)
            losses = model.compute_loss(batch, reconstructed, hidden)
            scaled_loss = losses["total_loss"] / gradient_accumulation_steps

        # Backward pass with gradient scaling
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        accumulation_loss += losses["total_loss"].item()

        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Normalize decoder weights (maintain unit norm constraint)
            model._normalize_decoder()

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()

            # Track metrics using accumulated loss
            total_losses.append(accumulation_loss)
            recon_losses.append(losses["reconstruction_loss"].item())
            sparsity_losses.append(losses["sparsity_loss"].item())
            sparsity_ratios.append(losses["sparsity_ratio"].item())

            accumulation_loss = 0.0

        # Clear intermediate tensors to save memory
        del batch, reconstructed, hidden, losses

        # More frequent but lightweight progress reporting
        if batch_idx % max(5, (25 * gradient_accumulation_steps)) == 0:
            if total_losses:  # Make sure we have losses to report
                batch_msg = (
                    f"Batch {batch_idx}: Loss={total_losses[-1]:.4f}, "
                    f"Recon={recon_losses[-1]:.4f}, "
                    f"Sparsity={sparsity_ratios[-1]:.3f}"
                )
                logging.info(batch_msg)
                print(batch_msg)  # Ensure visibility

        # Less frequent memory cleanup to avoid overhead
        if batch_idx % (200 * gradient_accumulation_steps) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Handle any remaining gradients
    if accumulation_loss > 0:
        model._normalize_decoder()
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    return {
        "total_loss": np.mean(total_losses) if total_losses else 0.0,
        "reconstruction_loss": np.mean(recon_losses) if recon_losses else 0.0,
        "sparsity_loss": np.mean(sparsity_losses) if sparsity_losses else 0.0,
        "sparsity_ratio": np.mean(sparsity_ratios) if sparsity_ratios else 0.0,
    }


def validate_sae(model, val_loader, device):
    """Validate SAE performance."""
    model.eval()
    total_losses = []
    recon_losses = []
    sparsity_ratios = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            # Forward pass
            reconstructed, hidden = model(batch)

            # Compute losses
            losses = model.compute_loss(batch, reconstructed, hidden)

            total_losses.append(losses["total_loss"].item())
            recon_losses.append(losses["reconstruction_loss"].item())
            sparsity_ratios.append(losses["sparsity_ratio"].item())

    return {
        "total_loss": np.mean(total_losses),
        "reconstruction_loss": np.mean(recon_losses),
        "sparsity_ratio": np.mean(sparsity_ratios),
    }


def analyze_sae_training(model, val_loader, device, epoch, output_dir, save_plots=True):
    """
    Comprehensive analysis of SAE training progress and learned features.

    Args:
        model: Trained SAE model
        val_loader: Validation data loader
        device: Compute device
        epoch: Current epoch number
        output_dir: Directory to save analysis results
        save_plots: Whether to save plots

    Returns:
        Dictionary with analysis results
    """
    model.eval()

    # Collect activations for analysis
    all_inputs = []
    all_hidden = []
    all_reconstructed = []
    all_losses = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            reconstructed, hidden = model(batch)

            # Compute per-sample reconstruction error
            recon_errors = torch.mean((batch - reconstructed) ** 2, dim=1)

            all_inputs.append(batch.cpu())
            all_hidden.append(hidden.cpu())
            all_reconstructed.append(reconstructed.cpu())
            all_losses.append(recon_errors.cpu())

    inputs = torch.cat(all_inputs, dim=0)
    hidden_acts = torch.cat(all_hidden, dim=0)
    reconstructed = torch.cat(all_reconstructed, dim=0)
    recon_errors = torch.cat(all_losses, dim=0)

    # Feature analysis
    feature_frequencies = torch.mean((hidden_acts > 0).float(), dim=0)
    feature_magnitudes = torch.mean(hidden_acts, dim=0)
    feature_max_acts = torch.max(hidden_acts, dim=0)[0]

    # Decoder analysis
    decoder_weights = model.decoder.weight.data.cpu()
    decoder_norms = torch.norm(decoder_weights, dim=0)

    # Compute feature selectivity (variance/mean ratio)
    feature_selectivity = torch.std(hidden_acts, dim=0) / (
        torch.mean(hidden_acts, dim=0) + 1e-8
    )

    analysis_results = {
        "epoch": epoch,
        "feature_statistics": {
            "activation_frequencies": feature_frequencies.numpy(),
            "mean_magnitudes": feature_magnitudes.numpy(),
            "max_activations": feature_max_acts.numpy(),
            "decoder_norms": decoder_norms.numpy(),
            "selectivity": feature_selectivity.numpy(),
        },
        "reconstruction_statistics": {
            "mean_error": torch.mean(recon_errors).item(),
            "median_error": torch.median(recon_errors).item(),
            "std_error": torch.std(recon_errors).item(),
        },
        "sparsity_statistics": {
            "overall_sparsity": torch.mean((hidden_acts == 0).float()).item(),
            "active_features": torch.sum(feature_frequencies > 0).item(),
            "highly_active_features": torch.sum(feature_frequencies > 0.1).item(),
            "rare_features": torch.sum(feature_frequencies < 0.01).item(),
        },
    }

    if save_plots:
        create_sae_analysis_plots(
            analysis_results,
            hidden_acts,
            recon_errors,
            feature_frequencies,
            decoder_norms,
            output_dir,
            epoch,
        )

    return analysis_results


def create_sae_analysis_plots(
    analysis_results,
    hidden_acts,
    recon_errors,
    feature_frequencies,
    decoder_norms,
    output_dir,
    epoch,
):
    """Create comprehensive visualization plots for SAE analysis."""

    # Create analysis directory
    analysis_dir = pathlib.Path(output_dir) / "sae_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Set up matplotlib
    plt.style.use("default")

    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 15))

    # Plot 1: Feature activation frequency histogram
    plt.subplot(3, 4, 1)
    plt.hist(
        feature_frequencies.numpy(),
        bins=50,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )
    plt.xlabel("Activation Frequency")
    plt.ylabel("Number of Features")
    plt.title("Feature Activation Frequency Distribution")
    plt.axvline(0.01, color="red", linestyle="--", alpha=0.7, label="1% line")
    plt.axvline(0.1, color="orange", linestyle="--", alpha=0.7, label="10% line")
    plt.legend()

    # Plot 2: Feature ranking by frequency
    plt.subplot(3, 4, 2)
    sorted_freq, _ = torch.sort(feature_frequencies, descending=True)
    plt.plot(sorted_freq.numpy(), color="coral")
    plt.xlabel("Feature Rank")
    plt.ylabel("Activation Frequency")
    plt.title("Features Ranked by Activation Frequency")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    # Plot 3: Reconstruction error distribution
    plt.subplot(3, 4, 3)
    plt.hist(
        recon_errors.numpy(), bins=50, alpha=0.7, color="lightgreen", edgecolor="black"
    )
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Number of Samples")
    plt.title("Per-Sample Reconstruction Error")
    mean_error = torch.mean(recon_errors).item()
    plt.axvline(
        mean_error, color="red", linestyle="--", label=f"Mean: {mean_error:.4f}"
    )
    plt.legend()

    # Plot 4: Decoder weight norms
    plt.subplot(3, 4, 4)
    plt.hist(decoder_norms.numpy(), bins=50, alpha=0.7, color="plum", edgecolor="black")
    plt.xlabel("Decoder Weight Norm")
    plt.ylabel("Number of Features")
    plt.title("Decoder Pattern Strength Distribution")

    # Plot 5: Feature selectivity vs frequency
    selectivity = analysis_results["feature_statistics"]["selectivity"]
    plt.subplot(3, 4, 5)
    plt.scatter(
        feature_frequencies.numpy(), selectivity, alpha=0.6, s=20, color="darkgreen"
    )
    plt.xlabel("Activation Frequency")
    plt.ylabel("Selectivity (std/mean)")
    plt.title("Feature Selectivity vs Activation Frequency")
    plt.grid(True, alpha=0.3)

    # Plot 6: Sparsity statistics pie chart
    plt.subplot(3, 4, 6)
    sparsity_stats = analysis_results["sparsity_statistics"]
    total_features = len(feature_frequencies)
    rare_features = sparsity_stats["rare_features"]
    active_features = sparsity_stats["highly_active_features"]
    moderate_features = total_features - rare_features - active_features

    labels = ["Rare (<1%)", "Moderate (1-10%)", "Highly Active (>10%)"]
    sizes = [rare_features, moderate_features, active_features]
    colors = ["lightcoral", "lightskyblue", "lightgreen"]
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    plt.title("Feature Activity Distribution")

    # Plot 7: Feature activation heatmap (sample)
    plt.subplot(3, 4, 7)
    # Sample most interesting features (moderate frequency)
    interesting_mask = (feature_frequencies > 0.05) & (feature_frequencies < 0.5)
    interesting_indices = torch.where(interesting_mask)[0]

    if len(interesting_indices) > 0:
        sample_size = min(50, len(interesting_indices))
        sample_features = interesting_indices[
            torch.randperm(len(interesting_indices))[:sample_size]
        ]
        sample_data = hidden_acts[:100, sample_features]  # First 100 samples

        im = plt.imshow(
            sample_data.T.numpy(),
            aspect="auto",
            cmap="viridis",
            interpolation="nearest",
        )
        plt.xlabel("Sample Index")
        plt.ylabel("Feature Index")
        plt.title(f"Feature Activation Heatmap\n({sample_size} features × 100 samples)")
        plt.colorbar(im, fraction=0.046, pad=0.04)
    else:
        plt.text(
            0.5,
            0.5,
            "No interesting\nfeatures found",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.title("Feature Activation Heatmap")

    # Plot 8: Feature co-activation correlation matrix
    plt.subplot(3, 4, 8)
    if len(interesting_indices) > 0:
        n_features_for_corr = min(20, len(interesting_indices))
        corr_features = interesting_indices[:n_features_for_corr]
        correlation_matrix = torch.corrcoef(hidden_acts[:, corr_features].T)

        sns.heatmap(
            correlation_matrix.numpy(),
            cmap="coolwarm",
            center=0,
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        plt.title(f"Feature Co-activation Matrix\n({n_features_for_corr} features)")
    else:
        plt.text(
            0.5,
            0.5,
            "Insufficient features\nfor correlation",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.title("Feature Co-activation Matrix")

    # Plot 9: Most and least active features
    plt.subplot(3, 4, 9)
    sorted_freq, sorted_indices = torch.sort(feature_frequencies, descending=True)
    top_10 = sorted_indices[:10]
    bottom_10 = sorted_indices[-10:]

    y_pos = np.arange(10)
    plt.barh(y_pos, sorted_freq[:10].numpy(), color="lightblue", alpha=0.7)
    plt.xlabel("Activation Frequency")
    plt.title("Top 10 Most Active Features")
    plt.yticks(y_pos, [f"F{idx.item()}" for idx in top_10])
    plt.grid(axis="x", alpha=0.3)

    # Plot 10: Dead features analysis
    plt.subplot(3, 4, 10)
    dead_features = torch.sum(feature_frequencies == 0).item()
    barely_active = torch.sum(
        (feature_frequencies > 0) & (feature_frequencies < 0.001)
    ).item()
    low_active = torch.sum(
        (feature_frequencies >= 0.001) & (feature_frequencies < 0.01)
    ).item()
    moderate_active = torch.sum(
        (feature_frequencies >= 0.01) & (feature_frequencies < 0.1)
    ).item()
    high_active = torch.sum(feature_frequencies >= 0.1).item()

    categories = [
        "Dead\n(0%)",
        "Barely\n(<0.1%)",
        "Low\n(0.1-1%)",
        "Moderate\n(1-10%)",
        "High\n(>10%)",
    ]
    counts = [dead_features, barely_active, low_active, moderate_active, high_active]
    colors = ["red", "orange", "yellow", "lightgreen", "green"]

    plt.bar(categories, counts, color=colors, alpha=0.7, edgecolor="black")
    plt.ylabel("Number of Features")
    plt.title("Feature Activity Categories")
    plt.xticks(rotation=45)

    # Plot 11: Reconstruction quality scatter
    plt.subplot(3, 4, 11)
    # Compute per-sample input norm and reconstruction error
    input_norms = torch.norm(hidden_acts, dim=1)
    plt.scatter(
        input_norms.numpy(), recon_errors.numpy(), alpha=0.5, s=10, color="purple"
    )
    plt.xlabel("Input Activation Norm")
    plt.ylabel("Reconstruction Error")
    plt.title("Reconstruction Error vs Input Norm")
    plt.grid(True, alpha=0.3)

    # Plot 12: Training statistics summary (text)
    plt.subplot(3, 4, 12)
    plt.axis("off")

    stats_text = f"""
SAE Training Analysis - Epoch {epoch}

SPARSITY METRICS:
• Overall sparsity: {analysis_results['sparsity_statistics']['overall_sparsity']:.3f}
• Active features: {analysis_results['sparsity_statistics']['active_features']}/{total_features}
• Highly active (>10%): {analysis_results['sparsity_statistics']['highly_active_features']}
• Rare features (<1%): {analysis_results['sparsity_statistics']['rare_features']}

RECONSTRUCTION METRICS:
• Mean error: {analysis_results['reconstruction_statistics']['mean_error']:.4f}
• Median error: {analysis_results['reconstruction_statistics']['median_error']:.4f}
• Std error: {analysis_results['reconstruction_statistics']['std_error']:.4f}

FEATURE METRICS:
• Max frequency: {torch.max(feature_frequencies):.3f}
• Min frequency: {torch.min(feature_frequencies):.3f}
• Mean decoder norm: {torch.mean(decoder_norms):.3f}
    """

    plt.text(
        0.05,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()

    # Save the plot
    plot_filename = analysis_dir / f"sae_analysis_epoch_{epoch:03d}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_filename


def create_training_progress_plots(training_history, output_dir):
    """Create plots showing training progress over epochs."""

    if not training_history:
        return

    analysis_dir = pathlib.Path(output_dir) / "sae_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    epochs = [entry["epoch"] for entry in training_history]
    train_losses = [entry["train"]["total_loss"] for entry in training_history]
    val_losses = [entry["val"]["total_loss"] for entry in training_history]
    train_recon = [entry["train"]["reconstruction_loss"] for entry in training_history]
    val_recon = [entry["val"]["reconstruction_loss"] for entry in training_history]
    train_sparsity = [entry["train"]["sparsity_ratio"] for entry in training_history]
    val_sparsity = [entry["val"]["sparsity_ratio"] for entry in training_history]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Total loss
    axes[0, 0].plot(epochs, train_losses, "b-", label="Train", linewidth=2)
    axes[0, 0].plot(epochs, val_losses, "r-", label="Validation", linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reconstruction loss
    axes[0, 1].plot(epochs, train_recon, "b-", label="Train", linewidth=2)
    axes[0, 1].plot(epochs, val_recon, "r-", label="Validation", linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Reconstruction Loss")
    axes[0, 1].set_title("Reconstruction Loss Progress")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Sparsity ratio
    axes[1, 0].plot(epochs, train_sparsity, "b-", label="Train", linewidth=2)
    axes[1, 0].plot(epochs, val_sparsity, "r-", label="Validation", linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Sparsity Ratio")
    axes[1, 0].set_title("Sparsity Ratio Progress")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Loss components comparison
    if len(training_history) > 0 and "sparsity_loss" in training_history[-1]["train"]:
        train_sparsity_loss = [
            entry["train"]["sparsity_loss"] for entry in training_history
        ]
        val_sparsity_loss = [
            entry["val"].get("sparsity_loss", 0) for entry in training_history
        ]

        axes[1, 1].plot(
            epochs, train_recon, "b-", label="Reconstruction Loss", linewidth=2
        )
        axes[1, 1].plot(
            epochs, train_sparsity_loss, "g-", label="Sparsity Loss", linewidth=2
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss Value")
        axes[1, 1].set_title("Loss Components (Training)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale("log")
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "Sparsity loss data\nnot available",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )
        axes[1, 1].set_title("Loss Components")

    plt.tight_layout()

    # Save the plot
    plot_filename = analysis_dir / "training_progress.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
    plt.close()

    return plot_filename


def save_analysis_summary(analysis_results, training_history, output_dir, model_config):
    """Save comprehensive analysis summary as JSON."""

    analysis_dir = pathlib.Path(output_dir) / "sae_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Prepare summary data
    summary = {
        "model_config": model_config,
        "final_analysis": analysis_results,
        "training_history": training_history,
        "analysis_metadata": {
            "total_epochs": len(training_history),
            "best_epoch": (
                min(training_history, key=lambda x: x["val"]["total_loss"])["epoch"]
                if training_history
                else None
            ),
            "final_sparsity": analysis_results["sparsity_statistics"][
                "overall_sparsity"
            ],
            "final_reconstruction_error": analysis_results["reconstruction_statistics"][
                "mean_error"
            ],
        },
    }

    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    summary = convert_numpy(summary)

    # Save summary
    summary_file = analysis_dir / "training_analysis_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary_file


def train_sae_pipeline(
    config: PipelineConfig, activations_path: pathlib.Path
) -> pathlib.Path:
    """
    Train sparse autoencoder on extracted activations.

    Args:
        config: Pipeline configuration
        activations_path: Path to extracted activations

    Returns:
        Path to saved SAE model
    """
    logging.info("=== STARTING SAE TRAINING ===")
    print("=== STARTING SAE TRAINING ===")  # Ensure visibility

    # Setup device
    device = setup_device(config)

    # Load data
    train_loader, val_loader, input_dim = load_activations_data(
        activations_path, config
    )

    # Create model
    model = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=config.sae.hidden_dim,
        sparsity_coeff=config.sae.sparsity_coeff,
        k_sparse=config.sae.k_sparse,
        use_bias=False,  # Standard SAE doesn't use bias in encoder
    ).to(device)

    model_info = f"Created SAE: {input_dim} -> {config.sae.hidden_dim} -> {input_dim}"
    logging.info(model_info)
    print(model_info)  # Ensure visibility
    logging.info(f"Expansion factor: {config.sae.hidden_dim / input_dim:.1f}x")

    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.sae.learning_rate,
        weight_decay=config.sae.weight_decay,
    )

    # Setup scheduler
    if config.sae.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.sae.num_epochs
        )
    elif config.sae.scheduler == "linear":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=config.sae.num_epochs,
        )
    else:
        scheduler = None

    # Training loop
    best_val_loss = float("inf")
    training_history = []

    for epoch in range(config.sae.num_epochs):
        logging.info(f"Epoch {epoch + 1}/{config.sae.num_epochs}")
        print(f"Epoch {epoch + 1}/{config.sae.num_epochs}")  # Ensure visibility

        # Train
        # Calculate gradient accumulation steps for memory efficiency
        # Use smaller effective batch size for faster training
        effective_batch_size = 64  # Reduced for faster updates
        gradient_accumulation_steps = max(
            1, effective_batch_size // config.sae.batch_size
        )

        train_metrics = train_sae_epoch(
            model, train_loader, optimizer, device, gradient_accumulation_steps
        )

        # Validate
        val_metrics = validate_sae(model, val_loader, device)

        # Step scheduler
        if scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            logging.info(f"Learning rate: {current_lr:.6f}")

        # Log metrics
        progress_msg = (
            f"Train Loss: {train_metrics['total_loss']:.4f}, "
            f"Val Loss: {val_metrics['total_loss']:.4f}, "
            f"Sparsity: {val_metrics['sparsity_ratio']:.3f}"
        )
        logging.info(progress_msg)
        print(progress_msg)  # Ensure visibility

        # Save metrics
        epoch_metrics = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        training_history.append(epoch_metrics)

        # Periodic detailed analysis (every 10 epochs or last epoch)
        if (epoch + 1) % 10 == 0 or epoch == config.sae.num_epochs - 1:
            print(f"📊 Running detailed analysis for epoch {epoch + 1}...")
            output_dir = config.output_dir or pathlib.Path(
                f"exp/{config.model.dataset}/{config.model.representation}"
            )

            try:
                analysis_results = analyze_sae_training(
                    model, val_loader, device, epoch + 1, output_dir, save_plots=True
                )
                print(
                    f"✅ Analysis complete - plots saved to {output_dir}/sae_analysis/"
                )
            except Exception as e:
                print(f"⚠️ Analysis failed: {e}")
                logging.warning(f"SAE analysis failed at epoch {epoch + 1}: {e}")

        # Save best model
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]

            # Save model
            output_dir = config.output_dir or pathlib.Path(
                f"exp/{config.model.dataset}/{config.model.representation}"
            )
            sae_dir = output_dir / "sae_models"
            sae_dir.mkdir(parents=True, exist_ok=True)

            model_path = sae_dir / f"sae_layer_{config.sae.hidden_dim}d.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.sae.__dict__,
                    "input_dim": input_dim,
                    "training_history": training_history,
                    "activations_path": str(activations_path),
                    "best_val_loss": best_val_loss,
                    "final_sparsity": val_metrics["sparsity_ratio"],
                    "layer_key": (
                        train_loader.dataset.layer_key
                        if hasattr(train_loader.dataset, "layer_key")
                        else "layer_3"
                    ),
                },
                model_path,
            )

            logging.info(f"Saved best model: {model_path}")
            print(f"Saved best model: {model_path}")  # Ensure visibility

    # Final comprehensive analysis and visualization
    print("\n🎯 GENERATING FINAL COMPREHENSIVE ANALYSIS...")
    output_dir = config.output_dir or pathlib.Path(
        f"exp/{config.model.dataset}/{config.model.representation}"
    )

    try:
        # Final detailed analysis
        final_analysis = analyze_sae_training(
            model,
            val_loader,
            device,
            config.sae.num_epochs,
            output_dir,
            save_plots=True,
        )

        # Create training progress plots
        progress_plot = create_training_progress_plots(training_history, output_dir)
        print(f"📈 Training progress plots saved: {progress_plot}")

        # Save comprehensive analysis summary
        summary_file = save_analysis_summary(
            final_analysis, training_history, output_dir, config.sae.__dict__
        )
        print(f"📋 Analysis summary saved: {summary_file}")

        # Print final statistics
        print("\n" + "=" * 60)
        print("🎼 FINAL SAE TRAINING ANALYSIS")
        print("=" * 60)

        print(f"🏆 Best validation loss: {best_val_loss:.4f}")
        print(
            f"🎯 Final sparsity ratio: {final_analysis['sparsity_statistics']['overall_sparsity']:.3f}"
        )
        print(
            f"📊 Active features: {final_analysis['sparsity_statistics']['active_features']}/{config.sae.hidden_dim}"
        )
        print(
            f"🔥 Highly active features (>10%): {final_analysis['sparsity_statistics']['highly_active_features']}"
        )
        print(
            f"💀 Rare features (<1%): {final_analysis['sparsity_statistics']['rare_features']}"
        )
        print(
            f"🎛️ Mean reconstruction error: {final_analysis['reconstruction_statistics']['mean_error']:.4f}"
        )

        print(f"\n📁 Analysis outputs saved to: {output_dir}/sae_analysis/")
        print(
            f"   - Detailed plots: sae_analysis_epoch_{config.sae.num_epochs:03d}.png"
        )
        print(f"   - Training progress: training_progress.png")
        print(f"   - Analysis summary: training_analysis_summary.json")

    except Exception as e:
        print(f"⚠️ Final analysis failed: {e}")
        logging.error(f"Final SAE analysis failed: {e}")

    completion_msg = f"=== TRAINING COMPLETE: {model_path} ==="
    final_loss_msg = f"Best validation loss: {best_val_loss:.4f}"
    final_sparsity_msg = f"Final sparsity ratio: {val_metrics['sparsity_ratio']:.3f}"

    logging.info(completion_msg)
    logging.info(final_loss_msg)
    logging.info(final_sparsity_msg)
    logging.info(final_loss_msg)
    logging.info(final_sparsity_msg)

    print(completion_msg)  # Ensure visibility
    print(final_loss_msg)  # Ensure visibility
    print(final_sparsity_msg)  # Ensure visibility

    return model_path


if __name__ == "__main__":
    # Example usage
    config = PipelineConfig()
    activations_path = pathlib.Path("exp/sod/ape/activations/activations_layers_3.h5")

    model_path = train_sae_pipeline(config, activations_path)
    print(f"SAE model saved to: {model_path}")
