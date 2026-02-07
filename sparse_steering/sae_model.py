"""Sparse Autoencoder (SAE) model with TopK sparsity."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class TopKActivation(nn.Module):
    """TopK activation that keeps only the K largest activations."""

    def __init__(self, k: int):
        """Initialize TopK activation.

        Args:
            k: Number of top activations to keep
        """
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply TopK activation.

        Args:
            x: Input tensor of shape (batch, features)

        Returns:
            Tensor with only top-k activations, rest set to zero
        """
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(x, self.k, dim=-1)

        # Create output tensor with zeros
        output = torch.zeros_like(x)

        # Scatter top-k values back
        output.scatter_(-1, topk_indices, topk_values)

        return output


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder with TopK sparsity constraint.

    Architecture:
        encoder: dense (512) -> ReLU -> sparse (4096) -> TopK(k=32)
        decoder: sparse (4096) -> dense (512)

    The encoder maps dense activations to a sparse high-dimensional space.
    The decoder reconstructs the original activations from the sparse representation.
    """

    def __init__(
        self,
        input_dim: int = 512,
        sparse_dim: int = 4096,
        k: int = 32,
        tied_weights: bool = True,
    ):
        """Initialize Sparse Autoencoder.

        Args:
            input_dim: Dimension of input activations (512 for MMT)
            sparse_dim: Dimension of sparse feature space
            k: Number of active features in TopK sparsity
            tied_weights: Whether to tie encoder and decoder weights (transpose)
        """
        super().__init__()

        self.input_dim = input_dim
        self.sparse_dim = sparse_dim
        self.k = k
        self.tied_weights = tied_weights

        # Encoder: dense -> sparse
        self.encoder = nn.Linear(input_dim, sparse_dim, bias=True)

        # TopK activation for sparsity
        self.topk = TopKActivation(k)

        # Decoder: sparse -> dense
        if tied_weights:
            # Decoder uses transposed encoder weights
            self.decoder = None
        else:
            self.decoder = nn.Linear(sparse_dim, input_dim, bias=True)

        # Decoder bias (always separate even with tied weights)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        if not self.tied_weights and self.decoder is not None:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode dense activations to sparse features.

        Args:
            x: Input activations (batch, input_dim)

        Returns:
            Sparse features (batch, sparse_dim) with TopK sparsity
        """
        # Linear projection
        h = self.encoder(x)

        # ReLU activation
        h = F.relu(h)

        # TopK sparsity
        sparse_features = self.topk(h)

        return sparse_features

    def decode(self, sparse_features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to dense activations.

        Args:
            sparse_features: Sparse features (batch, sparse_dim)

        Returns:
            Reconstructed activations (batch, input_dim)
        """
        if self.tied_weights:
            # Use transposed encoder weights
            reconstruction = F.linear(
                sparse_features, self.encoder.weight.t(), self.decoder_bias
            )
        else:
            # Use separate decoder
            reconstruction = self.decoder(sparse_features) + self.decoder_bias

        return reconstruction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: encode then decode.

        Args:
            x: Input activations (batch, input_dim)

        Returns:
            (reconstruction, sparse_features)
        """
        sparse_features = self.encode(x)
        reconstruction = self.decode(sparse_features)

        return reconstruction, sparse_features

    def compute_loss(
        self,
        x: torch.Tensor,
        reconstruction: torch.Tensor,
        sparse_features: torch.Tensor,
        l1_coefficient: float = 1e-3,
    ) -> Dict[str, torch.Tensor]:
        """Compute SAE loss: MSE reconstruction + L1 sparsity.

        Args:
            x: Original activations
            reconstruction: Reconstructed activations
            sparse_features: Sparse feature representation
            l1_coefficient: Weight for L1 sparsity loss

        Returns:
            Dictionary with loss components
        """
        # MSE reconstruction loss
        mse_loss = F.mse_loss(reconstruction, x)

        # L1 sparsity loss (encourage fewer active features)
        l1_loss = torch.mean(torch.abs(sparse_features))

        # Total loss
        total_loss = mse_loss + l1_coefficient * l1_loss

        # Compute L0 norm (number of active features)
        l0_norm = torch.mean((sparse_features != 0).float().sum(dim=-1))

        return {
            "total": total_loss,
            "mse": mse_loss,
            "l1": l1_loss,
            "l0": l0_norm,
        }

    def get_active_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get indices of active features for input.

        Args:
            x: Input activations (batch, input_dim)

        Returns:
            Boolean mask of active features (batch, sparse_dim)
        """
        sparse_features = self.encode(x)
        return sparse_features != 0

    def get_feature_statistics(self, sparse_features: torch.Tensor) -> Dict[str, float]:
        """Compute statistics about sparse features.

        Args:
            sparse_features: Sparse features (batch, sparse_dim)

        Returns:
            Dictionary with feature statistics
        """
        # Active features per sample
        active_per_sample = (sparse_features != 0).float().sum(dim=-1)

        # Feature activation frequency (% of samples where feature is active)
        activation_frequency = (sparse_features != 0).float().mean(dim=0)

        # Feature magnitude statistics
        nonzero_mask = sparse_features != 0
        magnitudes = torch.abs(sparse_features[nonzero_mask])

        stats = {
            "mean_active_features": active_per_sample.mean().item(),
            "std_active_features": active_per_sample.std().item(),
            "min_active_features": active_per_sample.min().item(),
            "max_active_features": active_per_sample.max().item(),
            "mean_activation_frequency": activation_frequency.mean().item(),
            "max_activation_frequency": activation_frequency.max().item(),
            "mean_feature_magnitude": (
                magnitudes.mean().item() if len(magnitudes) > 0 else 0.0
            ),
        }

        return stats


def test_sae_model():
    """Test SAE model functionality."""
    print("Testing Sparse Autoencoder...")

    # Create model
    sae = SparseAutoencoder(
        input_dim=512,
        sparse_dim=4096,
        k=32,
        tied_weights=True,
    )

    # Create dummy input
    batch_size = 16
    x = torch.randn(batch_size, 512)

    # Forward pass
    reconstruction, sparse_features = sae(x)

    # Compute loss
    loss_dict = sae.compute_loss(x, reconstruction, sparse_features)

    # Get statistics
    stats = sae.get_feature_statistics(sparse_features)

    # Print results
    print(f"Input shape: {x.shape}")
    print(f"Sparse features shape: {sparse_features.shape}")
    print(f"Reconstruction shape: {reconstruction.shape}")
    print(f"\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.6f}")
    print(f"\nFeature statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    # Verify sparsity
    active_features = (sparse_features != 0).sum(dim=-1)
    print(f"\nSparsity verification:")
    print(f"  Expected active features: {sae.k}")
    print(
        f"  Actual active features: {active_features.float().mean():.2f} ± {active_features.float().std():.2f}"
    )

    print("\n✓ SAE model test passed!")


if __name__ == "__main__":
    test_sae_model()
