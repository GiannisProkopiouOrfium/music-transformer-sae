#!/usr/bin/env python3
"""
Standalone SAE training module for SAE pipeline.

This module can be run independently to train sparse autoencoders on extracted activations.
"""

import argparse
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any

# Add mmt to path
sys.path.append(str(Path(__file__).parent.parent))


class SparseAutoencoder(nn.Module):
    """Optimized Sparse Autoencoder for music transformer interpretability."""

    def __init__(self, input_dim: int, hidden_dim: int, sparsity_coeff: float = 1e-4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coeff = sparsity_coeff

        # Encoder and decoder (no bias for better interpretability)
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)

        # Normalize decoder weights (each column represents a feature)
        with torch.no_grad():
            self.decoder.weight.div_(
                torch.norm(self.decoder.weight, dim=0, keepdim=True)
            )

    def forward(self, x: torch.Tensor):
        """Forward pass through SAE."""
        # Encode with ReLU activation for sparsity
        hidden = torch.relu(self.encoder(x))

        # Decode
        reconstructed = self.decoder(hidden)

        return reconstructed, hidden

    def loss(self, x: torch.Tensor, reconstructed: torch.Tensor, hidden: torch.Tensor):
        """Compute SAE loss with sparsity regularization."""
        # Reconstruction loss (MSE)
        recon_loss = torch.mean((x - reconstructed) ** 2)

        # Sparsity loss (L1 on hidden activations)
        sparsity_loss = torch.mean(torch.abs(hidden))

        total_loss = recon_loss + self.sparsity_coeff * sparsity_loss

        return total_loss, recon_loss, sparsity_loss


def setup_logging(level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def load_activations(activations_path: str) -> np.ndarray:
    """Load activations from HDF5 file."""
    with h5py.File(activations_path, "r") as f:
        activations = f["activations"][:]

    logging.info(f"Loaded activations: {activations.shape}")
    return activations


def train_sae(
    activations: np.ndarray,
    config: Dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> tuple:
    """Train the Sparse Autoencoder."""

    # SAE configuration
    sae_config = config.get("sae", {})
    input_dim = activations.shape[1]
    expansion_factor = sae_config.get("expansion_factor", 4.0)
    hidden_dim = int(input_dim * expansion_factor)

    # Create SAE model
    sae = SparseAutoencoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        sparsity_coeff=sae_config.get("sparsity_coeff", 1e-4),
    ).to(device)

    logger.info(f"SAE Architecture: {input_dim} → {hidden_dim} → {input_dim}")
    logger.info(f"Expansion factor: {expansion_factor}x")
    logger.info(f"Parameters: {sum(p.numel() for p in sae.parameters()):,}")

    # Training configuration
    learning_rate = sae_config.get("learning_rate", 1e-3)
    num_epochs = sae_config.get("num_epochs", 50)
    batch_size = sae_config.get("batch_size", 64)

    optimizer = optim.Adam(sae.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    # Convert to tensor
    activations_tensor = torch.FloatTensor(activations).to(device)

    # Training loop
    losses = {"total": [], "recon": [], "sparsity": []}

    logger.info(f"Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_losses = {"total": 0, "recon": 0, "sparsity": 0}
        num_batches = 0

        # Random permutation for each epoch
        indices = torch.randperm(activations_tensor.shape[0])

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch_data = activations_tensor[batch_indices]

            # Forward pass
            reconstructed, hidden = sae(batch_data)
            total_loss, recon_loss, sparsity_loss = sae.loss(
                batch_data, reconstructed, hidden
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Accumulate losses
            epoch_losses["total"] += total_loss.item()
            epoch_losses["recon"] += recon_loss.item()
            epoch_losses["sparsity"] += sparsity_loss.item()
            num_batches += 1

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            losses[key].append(epoch_losses[key])

        scheduler.step()

        # Log progress
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            sparsity_pct = torch.mean((hidden == 0).float()).item() * 100
            logger.info(
                f"Epoch {epoch:3d}: Loss={epoch_losses['total']:.4f} "
                f"(Recon={epoch_losses['recon']:.4f}, "
                f"Sparsity={epoch_losses['sparsity']:.4f}), "
                f"Sparsity={sparsity_pct:.1f}%"
            )

    return sae, losses


def save_model(sae: SparseAutoencoder, losses: Dict, output_path: str, config: Dict):
    """Save trained SAE model."""
    checkpoint = {
        "model_state_dict": sae.state_dict(),
        "input_dim": sae.input_dim,
        "hidden_dim": sae.hidden_dim,
        "sparsity_coeff": sae.sparsity_coeff,
        "losses": losses,
        "config": config,
        "architecture": f"{sae.input_dim}→{sae.hidden_dim}→{sae.input_dim}",
    }

    torch.save(checkpoint, output_path)
    logging.info(f"Model saved to: {output_path}")


def main():
    """Train Sparse Autoencoder on extracted activations."""
    parser = argparse.ArgumentParser(
        description="Train Sparse Autoencoder for music transformer interpretability"
    )
    parser.add_argument(
        "--activations", required=True, help="Path to extracted activations HDF5 file"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for trained model"
    )
    parser.add_argument("--config", help="Path to configuration YAML file (optional)")
    parser.add_argument(
        "--expansion-factor",
        type=float,
        default=4.0,
        help="Hidden dimension expansion factor (default: 4.0)",
    )
    parser.add_argument(
        "--sparsity-coeff",
        type=float,
        default=1e-4,
        help="Sparsity coefficient for L1 loss (default: 1e-4)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for training",
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

    logger.info("🎵 Sparse Autoencoder Training")
    logger.info(f"Activations: {args.activations}")
    logger.info(f"Output: {args.output}")

    try:
        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load config or create default
        if args.config:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {"sae": {}}

        # Override config with CLI arguments
        config["sae"]["expansion_factor"] = args.expansion_factor
        config["sae"]["sparsity_coeff"] = args.sparsity_coeff
        config["sae"]["num_epochs"] = args.epochs
        config["sae"]["batch_size"] = args.batch_size
        config["sae"]["learning_rate"] = args.learning_rate

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

        # Load activations
        logger.info("📊 Loading activations...")
        activations = load_activations(args.activations)

        # Train SAE
        logger.info("🔄 Training SAE...")
        sae, losses = train_sae(activations, config, device, logger)

        # Save model
        model_path = output_dir / "sae_model.pt"
        save_model(sae, losses, str(model_path), config)

        # Print results
        logger.info("✅ Training completed successfully!")
        logger.info(f"📁 Model saved to: {model_path}")

        # Final metrics
        final_loss = losses["total"][-1]
        final_recon = losses["recon"][-1]
        final_sparsity = losses["sparsity"][-1]

        logger.info(f"📊 Final metrics:")
        logger.info(f"  Total Loss: {final_loss:.4f}")
        logger.info(f"  Reconstruction Loss: {final_recon:.4f}")
        logger.info(f"  Sparsity Loss: {final_sparsity:.4f}")

        # Usage instructions
        print("\n" + "=" * 60)
        print("🎯 NEXT STEPS:")
        print("=" * 60)
        print("Analyze SAE:")
        print(
            f"  python pipeline/analyze_sae.py --model {model_path} --activations {args.activations} --output {args.output}/analysis"
        )
        print("\nFull pipeline:")
        print(
            f"  python pipeline/main.py --config {args.config} --output-dir {args.output} --experiment-name analysis --stage analyze"
        )

    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
