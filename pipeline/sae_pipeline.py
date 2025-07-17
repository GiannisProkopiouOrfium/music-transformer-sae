"""
Unified Cross-Platform SAE Pipeline for Music Transformer Interpretability
===========================================================================

This module provides a streamlined, optimized pipeline for:
1. Extracting activations from music transformer layers with input association
2. Training Sparse Autoencoders with proper sparsity constraints
3. Analyzing and interpreting learned musical features

Key Features:
- Automatic platform detection and optimization (macOS MPS, Linux CUDA, CPU fallback)
- Robust activation extraction with input-output mapping for interpretability
- Memory-efficient training with configurable batch sizes and gradient accumulation
- Comprehensive analysis with musical context correlation
- Easy single-command execution or modular stage-by-stage processing
"""

import sys
import platform
import psutil
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import logging
from typing import Dict, Any, Tuple, List
import warnings

warnings.filterwarnings("ignore")

# Add mmt to path
sys.path.append(str(Path(__file__).parent.parent))

# Import MMT components with graceful fallback
try:
    from mmt.dataset import MusicDataset
    from mmt.music_x_transformers import MusicXTransformer
    from mmt.representation import get_encoding
    # Import custom APE encoding
    sys.path.append(str(Path(__file__).parent.parent))
    from ape_encoding import get_ape_encoding

    HAS_MMT = True
except ImportError as e:
    logging.warning(f"MMT components not available, will use synthetic data: {e}")
    HAS_MMT = False


class PlatformOptimizer:
    """Detect platform and optimize settings accordingly."""

    def __init__(self):
        self.platform = platform.system().lower()
        self.is_macos = self.platform == "darwin"
        self.is_linux = self.platform == "linux"
        self.is_aws_ec2 = self._detect_aws_ec2()
        self.hardware_info = self._get_hardware_info()

    def _detect_aws_ec2(self) -> bool:
        """Detect if running on AWS EC2."""
        if not self.is_linux:
            return False
        try:
            import urllib.request

            req = urllib.request.Request(
                "http://169.254.169.254/latest/meta-data/instance-id",
                headers={"User-Agent": "aws-ec2-metadata/1.0"},
            )
            urllib.request.urlopen(req, timeout=2)
            return True
        except Exception:
            return False

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get detailed hardware information."""
        info = {
            "platform": self.platform,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "has_cuda": torch.cuda.is_available(),
            "has_mps": hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available(),
            "is_aws_ec2": self.is_aws_ec2,
        }

        if info["has_cuda"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_gb"] = torch.cuda.get_device_properties(
                0
            ).total_memory / (1024**3)

        return info

    def optimize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize configuration for current platform."""
        if self.is_macos:
            return self._optimize_macos(config)
        elif self.is_aws_ec2:
            return self._optimize_aws_ec2(config)
        else:
            return self._optimize_linux(config)

    def _optimize_macos(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Conservative settings for macOS development."""
        config.setdefault("extraction", {}).update(
            {
                "batch_size": min(config.get("extraction", {}).get("batch_size", 8), 8),
                "max_samples": min(
                    config.get("extraction", {}).get("max_samples", 2000), 2000
                ),
            }
        )

        config.setdefault("sae", {}).update(
            {
                "batch_size": min(config.get("sae", {}).get("batch_size", 32), 32),
                "num_epochs": min(config.get("sae", {}).get("num_epochs", 30), 30),
            }
        )

        config["platform"] = {
            "device": "mps" if self.hardware_info["has_mps"] else "cpu",
            "num_workers": 0,  # Avoid multiprocessing issues on macOS
            "memory_factor": 0.7,
            "use_synthetic": True,  # Default to synthetic for testing
        }
        return config

    def _optimize_aws_ec2(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Production settings for AWS EC2."""
        memory_gb = self.hardware_info["memory_gb"]

        # Detect instance type and optimize accordingly
        if memory_gb > 50:  # p3.2xlarge or similar
            instance_type = "large"
            config.setdefault("extraction", {}).update(
                {"batch_size": 64, "max_samples": 500000}
            )
            config.setdefault("sae", {}).update({"batch_size": 128, "num_epochs": 100})
        elif memory_gb > 25:  # g4dn.2xlarge
            instance_type = "medium"
            config.setdefault("extraction", {}).update(
                {"batch_size": 32, "max_samples": 100000}
            )
            config.setdefault("sae", {}).update({"batch_size": 64, "num_epochs": 100})
        else:  # g4dn.xlarge
            instance_type = "small"
            config.setdefault("extraction", {}).update(
                {"batch_size": 16, "max_samples": 50000}
            )
            config.setdefault("sae", {}).update({"batch_size": 32, "num_epochs": 50})

        config["platform"] = {
            "device": "cuda" if self.hardware_info["has_cuda"] else "cpu",
            "num_workers": min(self.hardware_info["cpu_count"], 8),
            "memory_factor": 0.85,
            "use_synthetic": False,
            "instance_type": instance_type,
        }
        return config

    def _optimize_linux(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generic Linux settings."""
        config["platform"] = {
            "device": "cuda" if self.hardware_info["has_cuda"] else "cpu",
            "num_workers": min(self.hardware_info["cpu_count"] // 2, 4),
            "memory_factor": 0.8,
            "use_synthetic": False,
        }
        return config


class SparseAutoencoder(nn.Module):
    """Optimized Sparse Autoencoder for music transformer interpretability."""

    def __init__(self, input_dim: int, hidden_dim: int, sparsity_coeff: float = 0.001):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coeff = sparsity_coeff

        # Encoder and decoder with proper initialization
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform and zero biases."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

        # Normalize decoder columns to unit norm (standard SAE practice)
        with torch.no_grad():
            self.decoder.weight.div_(
                torch.norm(self.decoder.weight, dim=0, keepdim=True) + 1e-8
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through SAE."""
        hidden = torch.relu(self.encoder(x))  # ReLU activation for sparsity
        reconstructed = self.decoder(hidden)
        return reconstructed, hidden

    def compute_loss(
        self, x: torch.Tensor, reconstructed: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss with reconstruction and sparsity components."""
        recon_loss = nn.functional.mse_loss(reconstructed, x)
        sparsity_loss = torch.mean(torch.abs(hidden))  # L1 sparsity
        total_loss = recon_loss + self.sparsity_coeff * sparsity_loss

        # Additional metrics
        sparsity_percent = (hidden == 0).float().mean().item() * 100

        metrics = {
            "total_loss": total_loss.item(),
            "reconstruction_loss": recon_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "sparsity_percent": sparsity_percent,
        }

        return total_loss, metrics


class ActivationExtractor:
    """Extract activations from transformer layers with input association."""

    def __init__(self, device: torch.device, logger: logging.Logger):
        self.device = device
        self.logger = logger
        self.activations = []
        self.input_tokens = []
        self.sequence_ids = []

    def extract_real_activations(
        self, config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract real activations from music transformer."""
        if not HAS_MMT:
            self.logger.warning("MMT not available, falling back to synthetic data")
            return self.extract_synthetic_activations(config)

        try:
            model_path = config["model"]["checkpoint_path"]
            if not Path(model_path).exists():
                self.logger.warning(
                    f"Model not found at {model_path}, using synthetic data"
                )
                return self.extract_synthetic_activations(config)

            # Load model and encoding - use APE encoding for this specific model
            encoding = get_ape_encoding()

            # Create model architecture
            model = MusicXTransformer(
                dim=config["model"]["hidden_size"],
                depth=config["model"]["num_layers"],
                heads=config["model"]["num_heads"],
                max_seq_len=config["model"]["max_seq_len"],
                max_beat=encoding["max_beat"],
                encoding=encoding,
            ).to(self.device)

            # Load checkpoint
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

            model.eval()
            self.logger.info(
                f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters"
            )

            # Setup dataset
            data_dir = Path(config["data"]["data_dir"])
            # Use the parent directory's names.txt file
            parent_dir = data_dir.parent
            names_files = list(parent_dir.glob("names.txt")) + list(parent_dir.glob("*names.txt"))
            if names_files:
                names_file = names_files[0]
                self.logger.info(f"Using names file: {names_file}")
            else:
                # Create a simple names file for testing
                names_file = data_dir / "temp_names.txt"
                json_files = list(data_dir.glob("**/*.npy"))[:config["extraction"]["max_samples"]]
                with open(names_file, 'w') as f:
                    for npy_file in json_files:
                        # Write relative path from data_dir without extension
                        rel_path = npy_file.relative_to(data_dir)
                        f.write(f"{rel_path.with_suffix('')}\n")
            
            dataset = MusicDataset(
                filename=str(names_file),
                data_dir=str(data_dir),
                encoding=encoding,
                max_seq_len=config["model"]["max_seq_len"],
                max_beat=encoding["max_beat"],
            )

            # Sample subset if needed
            max_samples = config["extraction"]["max_samples"]
            if max_samples and max_samples < len(dataset):
                indices = torch.randperm(len(dataset))[:max_samples]
                dataset = torch.utils.data.Subset(dataset, indices)

            self.logger.info(f"Dataset ready with {len(dataset)} samples")

            # Create dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config["extraction"]["batch_size"],
                shuffle=False,
                num_workers=config["platform"]["num_workers"],
                pin_memory=True if self.device.type == "cuda" else False,
            )

            # Extract activations from target layer
            target_layer = config["extraction"]["target_layer"]
            return self._extract_from_dataloader(model, dataloader, target_layer)

        except Exception as e:
            self.logger.error(f"Failed to extract real activations: {e}")
            return self.extract_synthetic_activations(config)

    def extract_synthetic_activations(
        self, config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate synthetic transformer-like activations for testing."""
        self.logger.info("Generating synthetic activations for testing")

        hidden_size = config["model"]["hidden_size"]
        max_samples = config["extraction"]["max_samples"]

        # Generate realistic transformer activations
        rng = np.random.default_rng(42)  # Reproducible

        activations = []
        metadata = {
            "type": "synthetic",
            "hidden_size": hidden_size,
            "layer_idx": config["extraction"]["target_layer"],
            "num_samples": max_samples,
            "input_associations": [],
        }

        for i in tqdm(range(0, max_samples, 1000), desc="Generating synthetic data"):
            batch_size = min(1000, max_samples - i)

            # Create transformer-like activations with realistic statistics
            batch_activations = rng.normal(0, 0.1, (batch_size, hidden_size)).astype(
                np.float32
            )

            # Add sparsity (some dimensions should be inactive)
            sparsity_mask = rng.random((batch_size, hidden_size)) < 0.15  # 15% sparsity
            batch_activations[sparsity_mask] = 0

            # Add some structure (correlated dimensions for musical features)
            for j in range(0, hidden_size, 64):  # Groups of 64 features
                if j + 64 <= hidden_size:
                    # Some correlation within groups (simulating musical patterns)
                    correlation_strength = rng.random() * 0.3
                    group_pattern = rng.normal(0, correlation_strength, (batch_size, 1))
                    batch_activations[:, j : j + 64] += group_pattern

            # Ensure non-negative (ReLU-like behavior)
            batch_activations = np.maximum(batch_activations, 0)

            activations.append(batch_activations)

            # Create synthetic input associations for interpretability
            for seq_id in range(batch_size):
                metadata["input_associations"].append(
                    {
                        "sequence_id": i + seq_id,
                        "synthetic_musical_context": {
                            "chord_progression": f"synthetic_chord_{(i + seq_id) % 12}",
                            "rhythm_pattern": f"pattern_{(i + seq_id) % 8}",
                            "instrument": f"inst_{(i + seq_id) % 4}",
                        },
                    }
                )

        activations_array = np.concatenate(activations, axis=0)

        self.logger.info(
            f"Generated {activations_array.shape[0]} synthetic activations"
        )
        self.logger.info(f"Shape: {activations_array.shape}")
        self.logger.info(f"Sparsity: {(activations_array == 0).mean() * 100:.1f}%")

        return activations_array, metadata

    def _extract_from_dataloader(
        self, model, dataloader, target_layer: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Extract activations from specific transformer layer."""
        self.activations = []
        self.input_tokens = []
        self.sequence_ids = []

        def hook_fn(module, input, output):
            """Hook to capture activations."""
            if hasattr(output, "hidden_states"):
                # If output has hidden_states attribute (transformer output)
                activations = output.hidden_states
            else:
                # Direct tensor output
                activations = output

            # Take mean over sequence length to get sentence-level representation
            if len(activations.shape) == 3:  # [batch, seq, hidden]
                activations = activations.mean(dim=1)  # [batch, hidden]

            self.activations.append(activations.detach().cpu())

        # Register hook on target layer
        if hasattr(model, "decoder") and hasattr(model.decoder, "net"):
            # MusicXTransformers structure
            layers = model.decoder.net.attn_layers.layers
            if target_layer < len(layers):
                handle = layers[target_layer].register_forward_hook(hook_fn)
            else:
                self.logger.warning(
                    f"Target layer {target_layer} not found, using last layer"
                )
                handle = layers[-1].register_forward_hook(hook_fn)
        else:
            self.logger.warning(
                "Model structure not recognized, using model-level hook"
            )
            handle = model.register_forward_hook(hook_fn)

        # Extract activations
        seq_id = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Extracting activations")
            ):
                if isinstance(batch, dict):
                    inputs = batch["input_ids"].to(self.device)
                else:
                    inputs = batch.to(self.device)

                # Forward pass
                _ = model(inputs)

                # Store input associations
                batch_size = inputs.shape[0]
                for i in range(batch_size):
                    self.input_tokens.append(inputs[i].cpu().numpy())
                    self.sequence_ids.append(seq_id)
                    seq_id += 1

        # Clean up hook
        handle.remove()

        # Concatenate all activations
        all_activations = torch.cat(self.activations, dim=0).numpy()

        metadata = {
            "type": "real",
            "hidden_size": all_activations.shape[1],
            "layer_idx": target_layer,
            "num_samples": all_activations.shape[0],
            "input_associations": [
                {"sequence_id": sid, "input_tokens": tokens.tolist()}
                for sid, tokens in zip(self.sequence_ids, self.input_tokens)
            ],
        }

        self.logger.info(
            f"Extracted {all_activations.shape[0]} real activations from layer {target_layer}"
        )

        return all_activations, metadata


class SAETrainer:
    """Train Sparse Autoencoder with proper optimization."""

    def __init__(self, device: torch.device, logger: logging.Logger):
        self.device = device
        self.logger = logger

    def train(
        self, activations: np.ndarray, config: Dict[str, Any]
    ) -> Tuple[SparseAutoencoder, Dict[str, List[float]]]:
        """Train SAE on extracted activations."""
        self.logger.info("Training Sparse Autoencoder")

        input_dim = activations.shape[1]
        hidden_dim = int(input_dim * config["sae"]["expansion_factor"])

        # Create model
        model = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sparsity_coeff=config["sae"]["sparsity_coeff"],
        ).to(self.device)

        self.logger.info(f"SAE architecture: {input_dim} → {hidden_dim} → {input_dim}")
        self.logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Setup optimizer and scheduler
        optimizer = optim.Adam(
            model.parameters(), lr=config["sae"]["learning_rate"], weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

        # Convert to tensor
        activations_tensor = torch.FloatTensor(activations).to(self.device)

        # Training loop
        num_epochs = config["sae"]["num_epochs"]
        batch_size = config["sae"]["batch_size"]

        losses = {"total": [], "reconstruction": [], "sparsity": []}
        best_loss = float("inf")

        for epoch in range(num_epochs):
            epoch_metrics = {
                "total": 0,
                "reconstruction": 0,
                "sparsity": 0,
                "sparsity_percent": 0,
            }
            num_batches = 0

            # Shuffle data
            indices = torch.randperm(activations_tensor.shape[0])

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_data = activations_tensor[batch_indices]

                # Forward pass
                reconstructed, hidden = model(batch_data)
                total_loss, metrics = model.compute_loss(
                    batch_data, reconstructed, hidden
                )

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Accumulate metrics
                for key in epoch_metrics:
                    if key in metrics:
                        epoch_metrics[key] += metrics[key]
                num_batches += 1

            # Average metrics
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
                if key in losses:
                    losses[key].append(epoch_metrics[key])

            scheduler.step()

            # Logging
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.logger.info(
                    f"Epoch {epoch:3d}: Loss={epoch_metrics['total']:.4f} "
                    f"(Recon={epoch_metrics['reconstruction']:.4f}, "
                    f"Sparsity={epoch_metrics['sparsity']:.4f}, "
                    f"Sparse%={epoch_metrics['sparsity_percent']:.1f}%)"
                )

            # Save best model
            if epoch_metrics["total"] < best_loss:
                best_loss = epoch_metrics["total"]

        self.logger.info(f"Training completed. Best loss: {best_loss:.4f}")
        return model, losses


class SAEAnalyzer:
    """Analyze trained SAE and create visualizations."""

    def __init__(self, device: torch.device, logger: logging.Logger):
        self.device = device
        self.logger = logger

    def analyze(
        self, model: SparseAutoencoder, activations: np.ndarray
    ) -> Dict[str, Any]:
        """Comprehensive SAE analysis."""
        self.logger.info("Analyzing trained SAE")

        # Sample data for analysis (to avoid memory issues)
        max_analysis_samples = 10000
        if activations.shape[0] > max_analysis_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(
                activations.shape[0], max_analysis_samples, replace=False
            )
            sample_activations = activations[indices]
        else:
            sample_activations = activations

        activations_tensor = torch.FloatTensor(sample_activations).to(self.device)

        # Run model
        with torch.no_grad():
            reconstructed, hidden = model(activations_tensor)

        # Convert to numpy for analysis
        original_np = activations_tensor.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()
        hidden_np = hidden.cpu().numpy()

        # Compute metrics
        mse = np.mean((original_np - reconstructed_np) ** 2)
        mae = np.mean(np.abs(original_np - reconstructed_np))

        # Explained variance
        total_var = np.var(original_np)
        residual_var = np.var(original_np - reconstructed_np)
        explained_variance = 1 - (residual_var / total_var) if total_var > 0 else 0

        # Sparsity analysis
        sparsity_percent = np.mean(hidden_np == 0) * 100
        l1_norm = np.mean(np.sum(np.abs(hidden_np), axis=1))

        # Feature activation analysis
        feature_activation_rates = np.mean(hidden_np > 0, axis=0) * 100
        active_features = np.sum(
            feature_activation_rates > 1
        )  # Features active >1% of time
        dead_features = np.sum(feature_activation_rates == 0)

        metrics = {
            "reconstruction_mse": float(mse),
            "reconstruction_mae": float(mae),
            "explained_variance": float(explained_variance),
            "sparsity_percent": float(sparsity_percent),
            "l1_norm": float(l1_norm),
            "active_features": int(active_features),
            "dead_features": int(dead_features),
            "total_features": int(hidden_np.shape[1]),
            "feature_activation_rates": feature_activation_rates.tolist(),
            "sample_size": len(sample_activations),
        }

        self.logger.info("Analysis results:")
        self.logger.info(f"  Reconstruction MSE: {mse:.4f}")
        self.logger.info(f"  Explained variance: {explained_variance:.3f}")
        self.logger.info(f"  Sparsity: {sparsity_percent:.1f}%")
        self.logger.info(f"  Active features: {active_features}/{hidden_np.shape[1]}")

        return metrics, (original_np, reconstructed_np, hidden_np)

    def create_visualizations(
        self,
        metrics: Dict[str, Any],
        data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray],
        output_dir: Path,
    ):
        """Create comprehensive analysis visualizations."""
        original_np, reconstructed_np, hidden_np = data_tuple

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("SAE Analysis Results", fontsize=16, fontweight="bold")

        # 1. Feature activation distribution
        feature_rates = np.array(metrics["feature_activation_rates"])
        axes[0, 0].hist(feature_rates, bins=50, alpha=0.7, edgecolor="black")
        axes[0, 0].set_xlabel("Activation Rate (%)")
        axes[0, 0].set_ylabel("Number of Features")
        axes[0, 0].set_title("Feature Activation Distribution")
        axes[0, 0].axvline(
            np.mean(feature_rates),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(feature_rates):.1f}%",
        )
        axes[0, 0].legend()

        # 2. Reconstruction quality
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(
            len(original_np), min(1000, len(original_np)), replace=False
        )
        axes[0, 1].scatter(
            original_np[sample_indices].flatten(),
            reconstructed_np[sample_indices].flatten(),
            alpha=0.5,
            s=1,
        )
        axes[0, 1].plot(
            [original_np.min(), original_np.max()],
            [original_np.min(), original_np.max()],
            "r--",
            alpha=0.8,
        )
        axes[0, 1].set_xlabel("Original Activations")
        axes[0, 1].set_ylabel("Reconstructed Activations")
        axes[0, 1].set_title(
            f'Reconstruction Quality (R² = {metrics["explained_variance"]:.3f})'
        )

        # 3. Sparsity heatmap (sample of features)
        n_samples = min(100, hidden_np.shape[0])
        n_features = min(50, hidden_np.shape[1])
        sample_hidden = hidden_np[:n_samples, :n_features]
        im = axes[1, 0].imshow(sample_hidden.T, aspect="auto", cmap="viridis")
        axes[1, 0].set_xlabel("Sample Index")
        axes[1, 0].set_ylabel("Feature Index")
        axes[1, 0].set_title(
            f'Feature Activation Heatmap\n(Sparsity: {metrics["sparsity_percent"]:.1f}%)'
        )
        plt.colorbar(im, ax=axes[1, 0])

        # 4. Summary statistics
        axes[1, 1].axis("off")
        stats_text = f"""
SAE Analysis Summary
{'='*25}

Architecture: {metrics['total_features']} features
Active Features: {metrics['active_features']} ({metrics['active_features']/metrics['total_features']*100:.1f}%)
Dead Features: {metrics['dead_features']} ({metrics['dead_features']/metrics['total_features']*100:.1f}%)

Reconstruction:
  MSE: {metrics['reconstruction_mse']:.4f}
  MAE: {metrics['reconstruction_mae']:.4f}
  Explained Variance: {metrics['explained_variance']:.3f}

Sparsity:
  Zero Activations: {metrics['sparsity_percent']:.1f}%
  L1 Norm: {metrics['l1_norm']:.3f}

Sample Size: {metrics['sample_size']:,}
        """
        axes[1, 1].text(
            0.1,
            0.9,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()
        plot_path = output_dir / "sae_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Analysis visualization saved to {plot_path}")
        return str(plot_path)


class UnifiedSAEPipeline:
    """Main pipeline class that orchestrates the entire SAE workflow."""

    def __init__(
        self,
        config_path: str = None,
        output_dir: str = None,
        experiment_name: str = "sae_experiment",
        config: dict = None,
    ):
        """
        Initialize pipeline.

        Args:
            config_path: Path to config YAML file (optional if config dict provided)
            output_dir: Output directory path
            experiment_name: Name of experiment
            config: Pre-loaded config dictionary (optional if config_path provided)
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name

        # Handle config loading vs direct config
        if config is not None:
            self.config_path = None
            self._config = config
        elif config_path is not None:
            self.config_path = Path(config_path)
            self._config = None
        else:
            raise ValueError("Either config_path or config must be provided")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Platform optimization
        self.optimizer = PlatformOptimizer()
        self.logger.info(f"Platform detected: {self.optimizer.platform}")
        self.logger.info(f"Hardware: {self.optimizer.hardware_info}")

        # Load and optimize config
        self.config = self._load_config()

        # Setup device
        self.device = self._setup_device()

        # Define file paths - create experiment subdirectory
        experiment_dir = self.output_dir / self.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        self.activations_file = experiment_dir / "activations.h5"
        self.model_file = experiment_dir / "sae_model.pt"
        self.analysis_file = experiment_dir / "analysis_results.h5"

        self.logger.info(f"Pipeline initialized: {self.experiment_name}")
        self.logger.info(f"Output directory: {experiment_dir}")
        self.logger.info(f"Device: {self.device}")

    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_file = self.output_dir / f"{self.experiment_name}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self) -> Dict[str, Any]:
        """Load and optimize configuration."""
        if self._config is not None:
            # Use pre-loaded config
            config = self._config
        else:
            # Load from file
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)

        # Platform-specific optimization
        config = self.optimizer.optimize_config(config)

        self.logger.info(
            f"Configuration loaded and optimized for {self.optimizer.platform}"
        )
        return config

    def _setup_device(self) -> torch.device:
        """Setup compute device."""
        device_str = self.config["platform"]["device"]

        if device_str == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif device_str == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            self.logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU")

        return device

    def extract_activations(self) -> Dict[str, Any]:
        """Extract activations from transformer layers."""
        self.logger.info("🔄 Stage 1: Extracting Activations")

        extractor = ActivationExtractor(self.device, self.logger)

        # Choose extraction method based on config
        # if self.config["platform"].get("use_synthetic", False):
        #     activations, metadata = extractor.extract_synthetic_activations(self.config)
        # else:
        activations, metadata = extractor.extract_real_activations(self.config)

        # Save activations with metadata
        with h5py.File(self.activations_file, "w") as f:
            f.create_dataset("activations", data=activations)

            # Save metadata as attributes
            for key, value in metadata.items():
                if key != "input_associations":  # Handle separately
                    f.attrs[key] = value

            # Save input associations as JSON string
            import json

            f.attrs["input_associations"] = json.dumps(metadata["input_associations"])

        self.logger.info(f"✅ Activations saved to {self.activations_file}")

        return {
            "activations_shape": activations.shape,
            "metadata": metadata,
            "file_path": str(self.activations_file),
        }

    def train_sae(self) -> Dict[str, Any]:
        """Train Sparse Autoencoder."""
        self.logger.info("🔄 Stage 2: Training SAE")

        # Load activations
        with h5py.File(self.activations_file, "r") as f:
            activations = f["activations"][:]
            metadata = {key: f.attrs[key] for key in f.attrs.keys()}

        self.logger.info(f"Loaded activations: {activations.shape}")

        # Train SAE
        trainer = SAETrainer(self.device, self.logger)
        model, losses = trainer.train(activations, self.config)

        # Save model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": self.config["sae"],
                "input_dim": model.input_dim,
                "hidden_dim": model.hidden_dim,
                "sparsity_coeff": model.sparsity_coeff,
                "losses": losses,
                "metadata": metadata,
            },
            self.model_file,
        )

        self.logger.info(f"✅ SAE model saved to {self.model_file}")

        return {
            "model_architecture": f"{model.input_dim} → {model.hidden_dim} → {model.input_dim}",
            "training_losses": losses,
            "file_path": str(self.model_file),
        }

    def analyze_sae(self) -> Dict[str, Any]:
        """Analyze trained SAE and create visualizations."""
        self.logger.info("🔄 Stage 3: Analyzing SAE")

        # Load model
        checkpoint = torch.load(
            self.model_file, map_location=self.device, weights_only=False
        )
        model = SparseAutoencoder(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            sparsity_coeff=checkpoint["sparsity_coeff"],
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Load activations
        with h5py.File(self.activations_file, "r") as f:
            activations = f["activations"][:]

        # Analyze
        analyzer = SAEAnalyzer(self.device, self.logger)
        metrics, data_tuple = analyzer.analyze(model, activations)

        # Create visualizations
        plot_path = analyzer.create_visualizations(metrics, data_tuple, self.output_dir)

        # Save analysis results
        with h5py.File(self.analysis_file, "w") as f:
            original_np, reconstructed_np, hidden_np = data_tuple

            f.create_dataset("original_activations", data=original_np)
            f.create_dataset("reconstructed_activations", data=reconstructed_np)
            f.create_dataset("feature_activations", data=hidden_np)

            # Save metrics as attributes
            for key, value in metrics.items():
                if key != "feature_activation_rates":
                    f.attrs[key] = value

            f.create_dataset(
                "feature_activation_rates", data=metrics["feature_activation_rates"]
            )

        self.logger.info(f"✅ Analysis results saved to {self.analysis_file}")

        return {
            "metrics": metrics,
            "visualization_path": plot_path,
            "analysis_file": str(self.analysis_file),
        }

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run the complete SAE pipeline."""
        self.logger.info("🚀 Running Full SAE Pipeline")

        start_time = datetime.now()

        # Stage 1: Extract activations
        extraction_results = self.extract_activations()

        # Stage 2: Train SAE
        training_results = self.train_sae()

        # Stage 3: Analyze SAE
        analysis_results = self.analyze_sae()

        end_time = datetime.now()
        duration = end_time - start_time

        results = {
            "experiment_name": self.experiment_name,
            "duration_seconds": duration.total_seconds(),
            "platform": self.optimizer.platform,
            "device": str(self.device),
            "extraction": extraction_results,
            "training": training_results,
            "analysis": analysis_results,
        }

        self.logger.info(f"🎉 Full pipeline completed in {duration}")
        return results
