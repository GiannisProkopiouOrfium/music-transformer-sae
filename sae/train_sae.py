import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pathlib
import logging
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from .sae_data import create_sae_dataloader

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")


class SparseAutoencoder(nn.Module):
    """Simple Sparse Autoencoder for activation analysis."""

    def __init__(self, input_dim: int, hidden_dim: int, sparsity_coeff: float = 1e-4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coeff = sparsity_coeff

        # Encoder and decoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)

        # Normalize decoder weights (columns)
        with torch.no_grad():
            self.decoder.weight.div_(
                torch.norm(self.decoder.weight, dim=0, keepdim=True)
            )

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
        hidden = torch.relu(self.encoder(x))

        # Decode
        reconstructed = self.decoder(hidden)

        return reconstructed, hidden

    def loss(
        self, x: torch.Tensor, reconstructed: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute SAE loss with sparsity regularization.

        Args:
            x: Original activations
            reconstructed: Reconstructed activations
            hidden: Hidden activations

        Returns:
            Total loss
        """
        # Reconstruction loss (MSE)
        recon_loss = torch.mean((x - reconstructed) ** 2)

        # Sparsity loss (L1 on hidden activations)
        sparsity_loss = torch.mean(torch.abs(hidden))

        total_loss = recon_loss + self.sparsity_coeff * sparsity_loss

        return total_loss


def train_sae(
    activations_path: str,
    layer_key: str = None,
    hidden_dim: int = 2048,
    sparsity_coeff: float = 1e-4,
    learning_rate: float = 1e-3,
    batch_size: int = 1024,
    num_epochs: int = 10,
    device: str = "cpu",
):
    """Train a Sparse Autoencoder on the extracted activations."""

    logging.info(f"Training SAE on {activations_path}")

    # Create data loader with memory-efficient loading
    dataloader, data_info = create_sae_dataloader(
        activations_path,
        layer_key=layer_key,
        batch_size=batch_size,
        shuffle=True,
        normalize=True,
        memory_efficient=True,  # Enable memory-efficient loading for large datasets
    )

    logging.info(f"Data info: {data_info}")

    input_dim = data_info["shape"][1]  # Feature dimension

    # Create model
    model = SparseAutoencoder(input_dim, hidden_dim, sparsity_coeff).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logging.info(f"Created SAE: {input_dim} -> {hidden_dim} -> {input_dim}")
    logging.info(f"Sparsity coefficient: {sparsity_coeff}")

    # Training loop
    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []

        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device)

            # Forward pass
            reconstructed, hidden = model(batch)
            loss = model.loss(batch, reconstructed, hidden)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            if batch_idx % 100 == 0:
                sparsity = torch.mean((hidden == 0).float()).item()
                logging.info(
                    f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.6f}, Sparsity = {sparsity:.3f}"
                )

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        logging.info(f"Epoch {epoch}: Average Loss = {avg_loss:.6f}")

    # Save model
    save_dir = pathlib.Path("sae_models")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"sae_{layer_key or 'layer'}_{hidden_dim}d.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "sparsity_coeff": sparsity_coeff,
            "losses": losses,
            "data_info": data_info,
        },
        save_path,
    )

    logging.info(f"Saved SAE model to {save_path}")

    return model, losses


def analyze_sae(model: SparseAutoencoder, dataloader, device: str = "cpu"):
    """Analyze the trained SAE."""
    model.eval()

    all_hidden = []
    all_recon_errors = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstructed, hidden = model(batch)

            recon_error = torch.mean((batch - reconstructed) ** 2, dim=1)
            all_recon_errors.append(recon_error.cpu())
            all_hidden.append(hidden.cpu())

    all_hidden = torch.cat(all_hidden, dim=0)
    all_recon_errors = torch.cat(all_recon_errors, dim=0)

    # Calculate statistics
    sparsity = torch.mean((all_hidden == 0).float()).item()
    mean_recon_error = torch.mean(all_recon_errors).item()

    logging.info(f"SAE Analysis:")
    logging.info(f"  Sparsity: {sparsity:.3f}")
    logging.info(f"  Mean reconstruction error: {mean_recon_error:.6f}")
    logging.info(f"  Hidden activations shape: {all_hidden.shape}")

    return {
        "sparsity": sparsity,
        "mean_recon_error": mean_recon_error,
        "hidden_activations": all_hidden,
        "reconstruction_errors": all_recon_errors,
    }


if __name__ == "__main__":
    # Train SAE on the extracted activations
    activations_path = "exp/sod/ape/activations/activations_layers_3.h5"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Train the SAE
    model, losses = train_sae(
        activations_path=activations_path,
        layer_key=None,  # Auto-detect
        hidden_dim=2048,  # 4x expansion
        sparsity_coeff=1e-4,
        learning_rate=1e-3,
        batch_size=1024,
        num_epochs=5,
        device=device,
    )

    # Create test dataloader for analysis
    test_dataloader, _ = create_sae_dataloader(
        activations_path,
        batch_size=1024,
        shuffle=False,
        subsample=5000,  # Use subset for analysis
    )

    # Analyze the trained SAE
    analysis = analyze_sae(model, test_dataloader, device)

    print("\nTraining complete!")
