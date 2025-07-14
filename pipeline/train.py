"""Training pipeline for SAE models."""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import time

# Add mmt to path
sys.path.append(str(Path(__file__).parent.parent))

from sae_data import ActivationDataset, create_sae_dataloader
from utils.gpu import GPUManager
from utils.memory import MemoryOptimizer
from utils.storage import HDF5Manager


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for music transformer interpretability."""

    def __init__(self, input_dim: int, hidden_dim: int, l1_alpha: float = 0.001):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_alpha = l1_alpha

        # Encoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # Decoder
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        """Forward pass through SAE."""
        # Encode
        hidden = torch.relu(self.encoder(x))

        # Decode
        reconstructed = self.decoder(hidden)

        return reconstructed, hidden

    def get_feature_activations(self, x):
        """Get feature activations without reconstruction."""
        with torch.no_grad():
            hidden = torch.relu(self.encoder(x))
        return hidden


def calculate_sparsity_loss(
    hidden_activations: torch.Tensor, l1_alpha: float
) -> torch.Tensor:
    """Calculate L1 sparsity loss."""
    return l1_alpha * torch.mean(torch.abs(hidden_activations))


def calculate_reconstruction_loss(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> torch.Tensor:
    """Calculate reconstruction loss (MSE)."""
    return nn.functional.mse_loss(reconstructed, original)


def train_epoch(
    model: SparseAutoencoder,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_mixed_precision: bool = False,
) -> Dict[str, float]:
    """Train SAE for one epoch."""
    model.train()

    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_sparsity_loss = 0.0
    total_sparsity = 0.0
    num_batches = 0

    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None

    for batch_idx, batch in enumerate(dataloader):
        activations = batch.to(device)

        optimizer.zero_grad()

        if use_mixed_precision and scaler:
            with torch.cuda.amp.autocast():
                reconstructed, hidden = model(activations)
                reconstruction_loss = calculate_reconstruction_loss(
                    activations, reconstructed
                )
                sparsity_loss = calculate_sparsity_loss(hidden, model.l1_alpha)
                total_loss_batch = reconstruction_loss + sparsity_loss

            scaler.scale(total_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstructed, hidden = model(activations)
            reconstruction_loss = calculate_reconstruction_loss(
                activations, reconstructed
            )
            sparsity_loss = calculate_sparsity_loss(hidden, model.l1_alpha)
            total_loss_batch = reconstruction_loss + sparsity_loss

            total_loss_batch.backward()
            optimizer.step()

        # Calculate sparsity percentage (percentage of zero activations)
        sparsity_percent = (hidden == 0).float().mean().item() * 100

        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_reconstruction_loss += reconstruction_loss.item()
        total_sparsity_loss += sparsity_loss.item()
        total_sparsity += sparsity_percent
        num_batches += 1

        if batch_idx % 100 == 0:
            logging.getLogger(__name__).info(
                f"Batch {batch_idx}, Loss: {total_loss_batch.item():.6f}, "
                f"Reconstruction: {reconstruction_loss.item():.6f}, "
                f"Sparsity: {sparsity_loss.item():.6f}, "
                f"Sparsity %: {sparsity_percent:.1f}%"
            )

    return {
        "total_loss": total_loss / num_batches,
        "reconstruction_loss": total_reconstruction_loss / num_batches,
        "sparsity_loss": total_sparsity_loss / num_batches,
        "sparsity_percent": total_sparsity / num_batches,
    }


def validate_model(
    model: SparseAutoencoder, dataloader: DataLoader, device: torch.device
) -> Dict[str, float]:
    """Validate SAE model."""
    model.eval()

    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_sparsity_loss = 0.0
    total_sparsity = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            activations = batch.to(device)

            reconstructed, hidden = model(activations)
            reconstruction_loss = calculate_reconstruction_loss(
                activations, reconstructed
            )
            sparsity_loss = calculate_sparsity_loss(hidden, model.l1_alpha)
            total_loss_batch = reconstruction_loss + sparsity_loss

            sparsity_percent = (hidden == 0).float().mean().item() * 100

            total_loss += total_loss_batch.item()
            total_reconstruction_loss += reconstruction_loss.item()
            total_sparsity_loss += sparsity_loss.item()
            total_sparsity += sparsity_percent
            num_batches += 1

    return {
        "total_loss": total_loss / num_batches,
        "reconstruction_loss": total_reconstruction_loss / num_batches,
        "sparsity_loss": total_sparsity_loss / num_batches,
        "sparsity_percent": total_sparsity / num_batches,
    }


def save_checkpoint(
    model: SparseAutoencoder,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    config: Dict[str, Any],
    filepath: str,
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config,
        "model_config": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "l1_alpha": model.l1_alpha,
        },
    }
    torch.save(checkpoint, filepath)


def load_activation_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load activation data from HDF5 file."""
    logger = logging.getLogger(__name__)

    # Get activation file path
    extraction_config = config["extraction"]
    activations_dir = config["experiment"]["directories"]["activations"]
    activation_file = Path(activations_dir) / extraction_config["output_file"]

    if not activation_file.exists():
        raise FileNotFoundError(f"Activation file not found: {activation_file}")

    logger.info(f"Loading activations from {activation_file}")

    with h5py.File(activation_file, "r") as f:
        activations = f["activations"][:]
        metadata = {
            "num_samples": f.attrs["num_samples"],
            "hidden_size": f.attrs["hidden_size"],
            "layer_idx": f.attrs["layer_idx"],
        }

    logger.info(
        f"Loaded {metadata['num_samples']} activation samples with hidden size {metadata['hidden_size']}"
    )
    return {"activations": activations, "metadata": metadata}


def run_training_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete SAE training pipeline."""
    logger = logging.getLogger(__name__)

    logger.info("Initializing SAE training pipeline")

    # Setup device and optimization
    gpu_manager = GPUManager()
    memory_optimizer = MemoryOptimizer()

    device = gpu_manager.device
    training_settings = gpu_manager.get_optimal_settings("training")

    logger.info(f"Training on device: {device}")
    logger.info(f"Training settings: {training_settings}")

    # Load activation data
    data_info = load_activation_data(config)
    activations = data_info["activations"]
    metadata = data_info["metadata"]

    # Get training configuration
    training_config = config["training"]

    # Create SAE model
    input_dim = metadata["hidden_size"]
    hidden_dim = training_config.get("hidden_dim", input_dim * 4)
    l1_alpha = training_config.get("l1_alpha", 0.001)

    model = SparseAutoencoder(input_dim, hidden_dim, l1_alpha)
    model.to(device)

    logger.info(f"Created SAE model: {input_dim} -> {hidden_dim} -> {input_dim}")

    # Create data loaders
    batch_size = training_config.get("batch_size", training_settings["batch_size"])

    train_loader, val_loader = create_sae_dataloader(
        activations,
        batch_size=batch_size,
        val_split=training_config.get("val_split", 0.1),
        shuffle=True,
    )

    logger.info(f"Created data loaders with batch size {batch_size}")

    # Setup optimizer
    learning_rate = training_config.get("learning_rate", 0.001)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    num_epochs = training_config.get("num_epochs", 100)
    save_every = training_config.get("save_every", 10)
    use_mixed_precision = training_settings.get("mixed_precision", False)

    best_val_loss = float("inf")
    training_history = []

    model_save_dir = Path(config["experiment"]["directories"]["models"])

    logger.info(f"Starting training for {num_epochs} epochs")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, use_mixed_precision
        )

        # Validation
        val_metrics = validate_model(model, val_loader, device)

        epoch_time = time.time() - epoch_start_time

        # Log metrics
        logger.info(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s):")
        logger.info(
            f"  Train - Loss: {train_metrics['total_loss']:.6f}, "
            f"Reconstruction: {train_metrics['reconstruction_loss']:.6f}, "
            f"Sparsity: {train_metrics['sparsity_percent']:.1f}%"
        )
        logger.info(
            f"  Val   - Loss: {val_metrics['total_loss']:.6f}, "
            f"Reconstruction: {val_metrics['reconstruction_loss']:.6f}, "
            f"Sparsity: {val_metrics['sparsity_percent']:.1f}%"
        )

        # Save training history
        training_history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
                "epoch_time": epoch_time,
            }
        )

        # Save best model
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_model_path = model_save_dir / "best_sae_model.pt"
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics["total_loss"],
                training_config,
                str(best_model_path),
            )
            logger.info(f"Saved best model with validation loss: {best_val_loss:.6f}")

        # Regular checkpoint saving
        if (epoch + 1) % save_every == 0:
            checkpoint_path = model_save_dir / f"sae_checkpoint_epoch_{epoch+1}.pt"
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_metrics["total_loss"],
                training_config,
                str(checkpoint_path),
            )

    # Save final model
    final_model_path = model_save_dir / "final_sae_model.pt"
    save_checkpoint(
        model,
        optimizer,
        num_epochs - 1,
        val_metrics["total_loss"],
        training_config,
        str(final_model_path),
    )

    # Save training history
    history_path = model_save_dir / "training_history.npy"
    np.save(history_path, training_history)

    results = {
        "experiment_name": config["experiment"]["name"],
        "num_epochs": num_epochs,
        "best_val_loss": best_val_loss,
        "final_train_metrics": train_metrics,
        "final_val_metrics": val_metrics,
        "model_config": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "l1_alpha": l1_alpha,
        },
        "training_config": training_config,
        "device": str(device),
    }

    logger.info("SAE training completed successfully")
    return results
