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

    # Create data loaders
    train_loader, train_info = create_sae_dataloader(
        str(activations_path),
        layer_key=layer_key,
        batch_size=config.sae.batch_size,
        shuffle=True,
        subsample=None,  # Use all data for training
        num_workers=0,  # Avoid multiprocessing issues
    )

    val_loader, val_info = create_sae_dataloader(
        str(activations_path),
        layer_key=layer_key,
        batch_size=config.sae.batch_size,
        shuffle=False,
        subsample=5000,  # Smaller validation set
        num_workers=0,  # Avoid multiprocessing issues
    )

    # Extract input dimension from shape
    input_dim = train_info["shape"][1]  # Shape is [num_samples, input_dim]
    logging.info(f"Created data loaders with input dimension: {input_dim}")
    logging.info(f"Training data shape: {train_info['shape']}")
    logging.info(f"Validation data shape: {val_info['shape']}")

    return train_loader, val_loader, input_dim


def train_sae_epoch(model, train_loader, optimizer, device):
    """Train SAE for one epoch."""
    model.train()
    total_losses = []
    recon_losses = []
    sparsity_losses = []
    sparsity_ratios = []

    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)

        # Forward pass
        reconstructed, hidden = model(batch)

        # Compute losses
        losses = model.compute_loss(batch, reconstructed, hidden)

        # Backward pass
        optimizer.zero_grad()
        losses["total_loss"].backward()

        # Normalize decoder weights (maintain unit norm constraint)
        model._normalize_decoder()

        optimizer.step()

        # Track metrics
        total_losses.append(losses["total_loss"].item())
        recon_losses.append(losses["reconstruction_loss"].item())
        sparsity_losses.append(losses["sparsity_loss"].item())
        sparsity_ratios.append(losses["sparsity_ratio"].item())

        if batch_idx % 100 == 0:
            batch_msg = (
                f"Batch {batch_idx}: Loss={losses['total_loss'].item():.4f}, "
                f"Recon={losses['reconstruction_loss'].item():.4f}, "
                f"Sparsity={losses['sparsity_ratio'].item():.3f}"
            )
            logging.info(batch_msg)
            print(batch_msg)  # Ensure visibility

    return {
        "total_loss": np.mean(total_losses),
        "reconstruction_loss": np.mean(recon_losses),
        "sparsity_loss": np.mean(sparsity_losses),
        "sparsity_ratio": np.mean(sparsity_ratios),
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
        train_metrics = train_sae_epoch(model, train_loader, optimizer, device)

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
                },
                model_path,
            )

            logging.info(f"Saved best model: {model_path}")
            print(f"Saved best model: {model_path}")  # Ensure visibility

    completion_msg = f"=== TRAINING COMPLETE: {model_path} ==="
    final_loss_msg = f"Best validation loss: {best_val_loss:.4f}"
    final_sparsity_msg = f"Final sparsity ratio: {val_metrics['sparsity_ratio']:.3f}"

    logging.info(completion_msg)
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
