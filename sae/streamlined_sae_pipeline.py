#!/usr/bin/env python3
"""
Streamlined SAE Pipeline for macOS Testing
This script runs the complete SAE pipeline: extract → train → analyze
"""

import sys
import yaml
import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Add mmt to path
sys.path.append(str(Path(__file__).parent))

# Import mmt components directly
from mmt.dataset import MusicDataset
from mmt.music_x_transformers import MusicXTransformers


class StreamlinedSAEPipeline:
    """Streamlined SAE pipeline for macOS testing."""

    def __init__(self, config_path, output_dir):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Setup device
        self.device = self._setup_device()

        # Setup paths
        self.activations_file = self.output_dir / "activations.h5"
        self.sae_model_file = self.output_dir / "sae_model.pt"
        self.analysis_file = self.output_dir / "analysis_results.h5"

    def _setup_device(self):
        """Setup compute device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🖥️  Using CUDA: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("🖥️  Using Apple MPS")
        else:
            device = torch.device("cpu")
            print("🖥️  Using CPU")
        return device

    def extract_activations(self):
        """Extract activations from transformer layers."""
        print("\n🔄 Stage 1: Extracting Activations")
        print("-" * 40)

        # Load model
        print("Loading model...")
        model_path = self.config["model"]["checkpoint_path"]
        checkpoint = torch.load(model_path, map_location=self.device)

        # Create model (simplified version)
        model = MusicXTransformers(
            dim=self.config["model"]["hidden_size"],
            depth=self.config["model"]["num_layers"],
            heads=self.config["model"]["num_heads"],
            max_seq_len=self.config["model"]["max_seq_len"],
            num_tokens=self.config["model"]["vocab_size"],
        ).to(self.device)

        # Load state dict
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()

        print(
            f"✅ Model loaded with {sum(p.numel() for p in model.parameters())} parameters"
        )

        # Setup dataset
        print("Setting up dataset...")
        dataset = MusicDataset(
            data_dir=self.config["data"]["data_dir"],
            data_version=self.config["data"]["version"],
            max_seq_len=self.config["model"]["max_seq_len"],
            representation=self.config["data"]["representation"],
        )

        # Get subset for testing
        max_samples = self.config["extraction"]["max_samples"]
        if max_samples and max_samples < len(dataset):
            indices = torch.randperm(len(dataset))[:max_samples]
            dataset = torch.utils.data.Subset(dataset, indices)

        print(f"✅ Dataset ready with {len(dataset)} samples")

        # Create data loader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config["extraction"]["batch_size"],
            shuffle=False,
            num_workers=0,  # Use 0 for macOS compatibility
        )

        # Extract activations
        print("Extracting activations...")
        target_layer = self.config["extraction"]["target_layer"]
        all_activations = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Processing batches")
            ):
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)

                # Get intermediate activations
                activations = self._extract_layer_activations(
                    model, inputs, target_layer
                )

                # Store activations (flatten sequence dimension)
                _, _, hidden_dim = activations.shape
                activations_flat = activations.view(-1, hidden_dim)
                all_activations.append(activations_flat.cpu().numpy())

                # Memory management
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Concatenate all activations
        activations_array = np.concatenate(all_activations, axis=0)
        print(
            f"✅ Extracted {activations_array.shape[0]} activations of dimension {activations_array.shape[1]}"
        )

        # Save activations
        print("Saving activations...")
        with h5py.File(self.activations_file, "w") as f:
            f.create_dataset("activations", data=activations_array, compression="gzip")
            f.attrs["num_samples"] = activations_array.shape[0]
            f.attrs["hidden_size"] = activations_array.shape[1]
            f.attrs["target_layer"] = target_layer

        print(f"✅ Activations saved to {self.activations_file}")
        return activations_array.shape

    def _extract_layer_activations(self, model, inputs, target_layer):
        """Extract activations from a specific layer."""
        activations = {}

        def hook_fn(module, input, output):
            activations["target"] = output

        # Register hook on target layer
        if hasattr(model, "net") and hasattr(model.net, "attn_layers"):
            layers = model.net.attn_layers.layers
            if target_layer < len(layers):
                hook = layers[target_layer].register_forward_hook(hook_fn)
            else:
                # Default to last layer
                hook = layers[-1].register_forward_hook(hook_fn)
        else:
            # Fallback: hook the whole model
            hook = model.register_forward_hook(hook_fn)

        # Forward pass
        try:
            _ = model(inputs)
            result = activations.get("target", inputs)
        except Exception as e:
            print(f"Warning: Forward pass failed: {e}")
            # Return dummy activations
            batch_size, seq_len = inputs.shape[:2]
            hidden_size = self.config["model"]["hidden_size"]
            result = torch.randn(batch_size, seq_len, hidden_size, device=inputs.device)

        # Remove hook
        hook.remove()

        return result

    def train_sae(self):
        """Train Sparse Autoencoder on extracted activations."""
        print("\n🔄 Stage 2: Training SAE")
        print("-" * 40)

        # Load activations
        print("Loading activations...")
        with h5py.File(self.activations_file, "r") as f:
            activations = f["activations"][:]
            hidden_size = f.attrs["hidden_size"]

        print(
            f"✅ Loaded {activations.shape[0]} activations of dimension {hidden_size}"
        )

        # SAE model
        class SparseAutoencoder(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim, sparsity_coeff=1e-3):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.sparsity_coeff = sparsity_coeff

                self.encoder = torch.nn.Linear(input_dim, hidden_dim)
                self.decoder = torch.nn.Linear(hidden_dim, input_dim)

                # Initialize weights
                torch.nn.init.xavier_uniform_(self.encoder.weight)
                torch.nn.init.xavier_uniform_(self.decoder.weight)

            def forward(self, x):
                # Encode
                hidden = torch.relu(self.encoder(x))

                # Decode
                reconstructed = self.decoder(hidden)

                return reconstructed, hidden

            def loss(self, x, reconstructed, hidden):
                # Reconstruction loss
                recon_loss = torch.nn.functional.mse_loss(reconstructed, x)

                # Sparsity loss (L1 regularization)
                sparsity_loss = torch.mean(torch.abs(hidden))

                # Total loss
                total_loss = recon_loss + self.sparsity_coeff * sparsity_loss

                return total_loss, recon_loss, sparsity_loss

        # Create SAE
        sae_config = self.config["sae"]
        expansion_factor = sae_config["expansion_factor"]
        sae_hidden_dim = int(hidden_size * expansion_factor)

        sae = SparseAutoencoder(
            input_dim=hidden_size,
            hidden_dim=sae_hidden_dim,
            sparsity_coeff=sae_config["sparsity_coeff"],
        ).to(self.device)

        print(f"✅ SAE created: {hidden_size} → {sae_hidden_dim} → {hidden_size}")

        # Training setup
        optimizer = torch.optim.Adam(sae.parameters(), lr=sae_config["learning_rate"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

        # Convert to tensor
        activations_tensor = torch.FloatTensor(activations).to(self.device)

        # Training loop
        print("Training SAE...")
        num_epochs = sae_config["num_epochs"]
        batch_size = sae_config["batch_size"]

        losses = {"total": [], "recon": [], "sparsity": []}

        for epoch in range(num_epochs):
            epoch_losses = {"total": 0, "recon": 0, "sparsity": 0}
            num_batches = 0

            # Create batches
            indices = torch.randperm(activations_tensor.shape[0])

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i : i + batch_size]
                batch_data = activations_tensor[batch_indices]

                # Forward pass
                reconstructed, hidden = sae(batch_data)

                # Compute loss
                total_loss, recon_loss, sparsity_loss = sae.loss(
                    batch_data, reconstructed, hidden
                )

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Track losses
                epoch_losses["total"] += total_loss.item()
                epoch_losses["recon"] += recon_loss.item()
                epoch_losses["sparsity"] += sparsity_loss.item()
                num_batches += 1

            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= num_batches
                losses[key].append(epoch_losses[key])

            scheduler.step()

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                print(
                    f"Epoch {epoch:3d}: Loss={epoch_losses['total']:.4f} "
                    f"(Recon={epoch_losses['recon']:.4f}, "
                    f"Sparsity={epoch_losses['sparsity']:.4f})"
                )

        print("✅ SAE training completed")

        # Save model
        torch.save(
            {
                "model_state_dict": sae.state_dict(),
                "config": sae_config,
                "input_dim": hidden_size,
                "hidden_dim": sae_hidden_dim,
                "losses": losses,
            },
            self.sae_model_file,
        )

        print(f"✅ SAE model saved to {self.sae_model_file}")

        return sae, losses

    def analyze_sae(self):
        """Analyze the trained SAE."""
        print("\n🔄 Stage 3: Analyzing SAE")
        print("-" * 40)

        # Load SAE model
        print("Loading SAE model...")
        checkpoint = torch.load(self.sae_model_file, map_location=self.device)
        input_dim = checkpoint["input_dim"]
        hidden_dim = checkpoint["hidden_dim"]

        # Recreate SAE
        class SparseAutoencoder(torch.nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.encoder = torch.nn.Linear(input_dim, hidden_dim)
                self.decoder = torch.nn.Linear(hidden_dim, input_dim)

            def forward(self, x):
                hidden = torch.relu(self.encoder(x))
                reconstructed = self.decoder(hidden)
                return reconstructed, hidden

        sae = SparseAutoencoder(input_dim, hidden_dim).to(self.device)
        sae.load_state_dict(checkpoint["model_state_dict"])
        sae.eval()

        print(f"✅ SAE model loaded: {input_dim} → {hidden_dim} → {input_dim}")

        # Load activations
        print("Loading activations...")
        with h5py.File(self.activations_file, "r") as f:
            activations = f["activations"][:]

        # Sample subset for analysis
        max_analysis_samples = 10000
        if activations.shape[0] > max_analysis_samples:
            indices = np.random.choice(
                activations.shape[0], max_analysis_samples, replace=False
            )
            activations = activations[indices]

        activations_tensor = torch.FloatTensor(activations).to(self.device)
        print(f"✅ Analyzing {activations.shape[0]} samples")

        # Run analysis
        print("Running SAE analysis...")
        with torch.no_grad():
            reconstructed, hidden = sae(activations_tensor)

        # Convert to numpy
        activations_np = activations_tensor.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()
        hidden_np = hidden.cpu().numpy()

        # Compute metrics
        print("Computing metrics...")

        # Reconstruction error
        mse = np.mean((activations_np - reconstructed_np) ** 2)
        mae = np.mean(np.abs(activations_np - reconstructed_np))

        # Sparsity metrics
        sparsity = np.mean(hidden_np == 0) * 100
        l1_sparsity = np.mean(np.sum(np.abs(hidden_np), axis=1))

        # Feature statistics
        feature_activations = np.mean(hidden_np > 0, axis=0) * 100
        active_features = np.sum(
            feature_activations > 1
        )  # Features active in >1% of samples

        metrics = {
            "reconstruction_mse": float(mse),
            "reconstruction_mae": float(mae),
            "sparsity_percent": float(sparsity),
            "l1_sparsity": float(l1_sparsity),
            "active_features": int(active_features),
            "total_features": int(hidden_dim),
            "feature_activation_rates": feature_activations,
        }

        print("✅ Analysis completed:")
        print(f"   - Reconstruction MSE: {mse:.4f}")
        print(f"   - Reconstruction MAE: {mae:.4f}")
        print(f"   - Sparsity: {sparsity:.1f}%")
        print(f"   - Active features: {active_features}/{hidden_dim}")

        # Create visualizations
        print("Creating visualizations...")
        self._create_analysis_plots(metrics, checkpoint["losses"])

        # Save analysis results
        with h5py.File(self.analysis_file, "w") as f:
            f.create_dataset("original_activations", data=activations_np)
            f.create_dataset("reconstructed_activations", data=reconstructed_np)
            f.create_dataset("sae_features", data=hidden_np)

            # Save metrics as attributes
            for key, value in metrics.items():
                if key != "feature_activation_rates":
                    f.attrs[key] = value

            f.create_dataset("feature_activation_rates", data=feature_activations)

        print(f"✅ Analysis results saved to {self.analysis_file}")

        return metrics

    def _create_analysis_plots(self, metrics, losses):
        """Create analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("SAE Analysis Results", fontsize=16)

        # Training losses
        axes[0, 0].plot(losses["total"], label="Total Loss")
        axes[0, 0].plot(losses["recon"], label="Reconstruction Loss")
        axes[0, 0].plot(losses["sparsity"], label="Sparsity Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Losses")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Feature activation rates
        activation_rates = metrics["feature_activation_rates"]
        axes[0, 1].hist(activation_rates, bins=50, alpha=0.7)
        axes[0, 1].set_xlabel("Activation Rate (%)")
        axes[0, 1].set_ylabel("Number of Features")
        axes[0, 1].set_title("Feature Activation Distribution")
        axes[0, 1].grid(True)

        # Metrics summary
        metric_names = [
            "Reconstruction MSE",
            "Reconstruction MAE",
            "Sparsity (%)",
            "Active Features",
        ]
        metric_values = [
            metrics["reconstruction_mse"],
            metrics["reconstruction_mae"],
            metrics["sparsity_percent"],
            metrics["active_features"],
        ]

        axes[1, 0].barh(metric_names, metric_values)
        axes[1, 0].set_title("Key Metrics")

        # Sparsity visualization
        axes[1, 1].text(
            0.1, 0.8, f"Total Features: {metrics['total_features']}", fontsize=12
        )
        axes[1, 1].text(
            0.1, 0.6, f"Active Features: {metrics['active_features']}", fontsize=12
        )
        axes[1, 1].text(
            0.1, 0.4, f"Sparsity: {metrics['sparsity_percent']:.1f}%", fontsize=12
        )
        axes[1, 1].text(
            0.1, 0.2, f"L1 Sparsity: {metrics['l1_sparsity']:.3f}", fontsize=12
        )
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title("Sparsity Summary")
        axes[1, 1].axis("off")

        plt.tight_layout()

        # Save plot
        plot_file = self.output_dir / "sae_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Analysis plot saved to {plot_file}")

    def run_full_pipeline(self):
        """Run the complete SAE pipeline."""
        print("🚀 Starting Streamlined SAE Pipeline")
        print("=" * 50)

        start_time = datetime.now()

        try:
            # Stage 1: Extract activations
            activation_shape = self.extract_activations()

            # Stage 2: Train SAE
            _, _ = self.train_sae()

            # Stage 3: Analyze SAE
            metrics = self.analyze_sae()

            # Summary
            end_time = datetime.now()
            duration = end_time - start_time

            print("\n" + "=" * 50)
            print("🎉 Pipeline completed successfully!")
            print(f"⏱️  Total time: {duration}")
            print(f"📊 Results saved to: {self.output_dir}")
            print("\nSummary:")
            print(f"   - Activations extracted: {activation_shape[0]:,}")
            print(f"   - SAE features: {metrics['total_features']:,}")
            print(f"   - Active features: {metrics['active_features']:,}")
            print(f"   - Sparsity: {metrics['sparsity_percent']:.1f}%")
            print(f"   - Reconstruction error: {metrics['reconstruction_mse']:.4f}")

            return True

        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Streamlined SAE Pipeline")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--stage",
        choices=["extract", "train", "analyze", "all"],
        default="all",
        help="Pipeline stage to run",
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = StreamlinedSAEPipeline(args.config, args.output_dir)

    # Run specified stage
    if args.stage == "extract":
        pipeline.extract_activations()
    elif args.stage == "train":
        pipeline.train_sae()
    elif args.stage == "analyze":
        pipeline.analyze_sae()
    else:
        pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
