#!/usr/bin/env python3
"""
REAL LiMuF Extractor for Single Feature

Uses your existing infrastructure:
- SAE model: exp/sod/ape/sae_models/sae_layer_2048d.pt
- Activations: exp/sod/ape/activations/activations_layers_3.h5
- Interpretations: interpretation/enhanced_diversity/enhanced_diversity_report.json

NO SIMULATION - uses real mean activation on actual feature activations.
"""

import torch
import h5py
import json
from pathlib import Path
import argparse
from tqdm import tqdm

# Import your SAE architecture
from sae.train_sae import SparseAutoencoder


class RealLiMuFExtractor:
    """Extract LiMuFs using real SAE feature activations and residual stream data."""

    def __init__(
        self,
        sae_model_path: str = "exp/sod/ape/sae_models/sae_layer_2048d.pt",
        activations_path: str = "exp/sod/ape/activations/activations_layers_3.h5",
        interpretations_path: str = "interpretation/enhanced_diversity/enhanced_diversity_report.json",
        device: str = "cuda",
    ):
        self.device = device
        self.sae_model_path = Path(sae_model_path)
        self.activations_path = Path(activations_path)
        self.interpretations_path = Path(interpretations_path)

        # Load components
        print("🔬 LOADING REAL SAE INFRASTRUCTURE")
        print("=" * 50)
        self.sae_model = self._load_sae_model()
        self.activations_data = self._load_activations()
        self.interpretations = self._load_interpretations()

        print(f"✅ SAE Model loaded: {type(self.sae_model)}")
        print(f"✅ Activations shape: {self.activations_data.shape}")
        print(f"✅ Interpretations for {len(self.interpretations)} features")
        print()

    def _load_sae_model(self):
        """Load your trained SAE model."""
        print(f"📦 Loading SAE from: {self.sae_model_path}")

        if not self.sae_model_path.exists():
            raise FileNotFoundError(f"SAE model not found: {self.sae_model_path}")

        checkpoint = torch.load(self.sae_model_path, map_location="cpu")

        # Extract model parameters from checkpoint
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Infer model dimensions from state dict
        encoder_weight = None
        for key, value in state_dict.items():
            if "encoder" in key and "weight" in key:
                encoder_weight = value
                break

        if encoder_weight is None:
            raise ValueError("Could not find encoder weight in SAE checkpoint")

        input_dim = encoder_weight.shape[1]  # Input dimension (model hidden size)
        hidden_dim = encoder_weight.shape[0]  # SAE hidden dimension

        print(f"   SAE dimensions: {input_dim} → {hidden_dim}")

        # Create SAE model
        sae_model = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sparsity_coeff=0.1,  # Default value
        )

        # Load state dict
        sae_model.load_state_dict(state_dict)
        sae_model.to(self.device)
        sae_model.eval()

        return sae_model

    def _load_activations(self):
        """Load activations from h5 file."""
        print(f"📁 Loading activations from: {self.activations_path}")

        if not self.activations_path.exists():
            raise FileNotFoundError(f"Activations not found: {self.activations_path}")

        with h5py.File(self.activations_path, "r") as f:
            # Check what's in the file
            print(f"   H5 file keys: {list(f.keys())}")

            # Load activations (adjust key name based on your file structure)
            if "activations" in f:
                activations = torch.tensor(f["activations"][:], dtype=torch.float32)
            elif "layer_3" in f:
                activations = torch.tensor(f["layer_3"][:], dtype=torch.float32)
            else:
                # Try the first available dataset
                key = list(f.keys())[0]
                print(f"   Using key: {key}")
                activations = torch.tensor(f[key][:], dtype=torch.float32)

        print(
            f"   Loaded {activations.shape[0]} activation vectors of dimension {activations.shape[1]}"
        )
        return activations

    def _load_interpretations(self):
        """Load feature interpretations."""
        print(f"📖 Loading interpretations from: {self.interpretations_path}")

        with open(self.interpretations_path, "r") as f:
            data = json.load(f)

        interpretations = data.get("interpretations", {})
        print(f"   Found interpretations for features: {list(interpretations.keys())}")

        return interpretations

    def extract_limuf(
        self,
        feature_id: int,
        activation_threshold: float = 2.0,
        min_samples: int = 100,
        max_samples: int = 10000,
    ):
        """
        Extract LiMuF using real mean activation methodology.

        Args:
            feature_id: Feature ID to extract direction for
            activation_threshold: Minimum activation to consider feature "active"
            min_samples: Minimum samples needed for the feature
            max_samples: Maximum samples to use (for efficiency)
        """
        print(f"🎯 EXTRACTING LIMUF FOR FEATURE {feature_id}")
        print("=" * 50)

        # Get feature description
        desc = self.interpretations.get(str(feature_id), {}).get(
            "interpretation", f"Feature {feature_id}"
        )

        print(f"🎵 Feature {feature_id}: {desc[:80]}...")
        print()

        # Use subset of activations for efficiency
        n_samples = min(len(self.activations_data), max_samples)
        activations_subset = self.activations_data[:n_samples]

        print(f"🔬 Processing {n_samples} activation samples...")

        # Encode activations through SAE to get feature activations
        with torch.no_grad():
            batch_size = 1000  # Process in batches to avoid memory issues
            all_feature_activations = []

            for i in tqdm(range(0, n_samples, batch_size), desc="SAE encoding"):
                batch = activations_subset[i : i + batch_size].to(self.device)

                # Get SAE feature activations using forward pass
                # forward returns (reconstructed, hidden) where hidden is what we need
                _, encoded = self.sae_model.forward(batch)  # [batch_size, n_features]
                all_feature_activations.append(encoded.cpu())

            feature_activations = torch.cat(
                all_feature_activations, dim=0
            )  # [n_samples, n_features]

        print(f"✅ SAE encoding complete: {feature_activations.shape}")

        # Find samples where the feature is active
        feature_strengths = feature_activations[:, feature_id]

        # Create mask for high activation samples
        high_activation_mask = feature_strengths > activation_threshold

        n_active_samples = high_activation_mask.sum().item()

        print("📊 Feature activation analysis:")
        print(
            f"   Feature {feature_id}: {n_active_samples} samples above threshold {activation_threshold}"
        )
        print(f"   Mean activation: {feature_strengths.mean().item():.4f}")
        print(f"   Max activation: {feature_strengths.max().item():.4f}")
        print(f"   Std activation: {feature_strengths.std().item():.4f}")

        if n_active_samples < min_samples:
            print(
                f"⚠️  Warning: Only {n_active_samples} samples for Feature {feature_id} (min: {min_samples})"
            )

        if n_active_samples == 0:
            raise ValueError(
                f"No samples found above threshold {activation_threshold} for feature {feature_id}"
            )

        # Extract corresponding residual stream activations
        activations_when_feature_fires = activations_subset[
            high_activation_mask
        ]  # [n_active_samples, d_model]

        # Calculate mean activation (LiMuF)
        mean_activation = torch.mean(activations_when_feature_fires, dim=0)  # [d_model]
        mean_norm = torch.norm(mean_activation, dim=0).item()

        # Normalize
        limuf_vector_normalized = mean_activation / torch.norm(mean_activation, dim=0)

        print("🎯 LiMuF vector extracted:")
        print(f"   Raw norm: {mean_norm:.4f}")
        print(
            f"   Normalized norm: {torch.norm(limuf_vector_normalized, dim=0).item():.4f}"
        )
        print(f"   Vector shape: {limuf_vector_normalized.shape}")

        # Calculate some statistics
        print("📈 Statistics:")
        print(f"   Active samples: {n_active_samples}")
        print(
            f"   Mean activation strength: {feature_strengths[high_activation_mask].mean().item():.4f}"
        )
        print(f"   Activation percentile: {(n_active_samples / n_samples * 100):.2f}%")

        # Create metadata
        metadata = {
            "feature_id": feature_id,
            "feature_description": desc,
            "active_samples": n_active_samples,
            "total_samples": n_samples,
            "activation_threshold": activation_threshold,
            "limuf_norm": mean_norm,
            "layer": 3,  # Based on your activations file
            "method": "mean_activation",
            "statistics": {
                "mean_activation_strength": feature_strengths[high_activation_mask]
                .mean()
                .item(),
                "max_activation_strength": feature_strengths[high_activation_mask]
                .max()
                .item(),
                "std_activation_strength": feature_strengths[high_activation_mask]
                .std()
                .item(),
                "activation_percentile": n_active_samples / n_samples * 100,
                "overall_mean_activation": feature_strengths.mean().item(),
                "overall_std_activation": feature_strengths.std().item(),
            },
        }

        return limuf_vector_normalized, metadata

    def save_limufs(self, limufs: dict, output_dir: str = "real_limufs_layer3"):
        """Save extracted LiMuFs."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Separate vectors and metadata
        limuf_vectors = {}
        metadata = {}

        for name, (vector, meta) in limufs.items():
            limuf_vectors[name] = vector
            metadata[name] = meta

        # Save
        output_file = output_path / "limufs.pt"
        torch.save(
            {
                "limufs": limuf_vectors,
                "metadata": metadata,
                "extraction_info": {
                    "sae_model_path": str(self.sae_model_path),
                    "activations_path": str(self.activations_path),
                    "interpretations_path": str(self.interpretations_path),
                    "method": "real_mean_activation",
                },
            },
            output_file,
        )

        # Save metadata as JSON for readability
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Saved LiMuFs to: {output_file}")
        print(f"✅ Saved metadata to: {metadata_file}")

        return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Extract real LiMuFs for single features"
    )
    parser.add_argument(
        "--feature-id",
        type=int,
        default=182,
        help="Feature ID to extract direction for",
    )
    parser.add_argument(
        "--threshold", type=float, default=2.0, help="Activation threshold"
    )
    parser.add_argument(
        "--output-dir", default="real_limufs_layer3", help="Output directory"
    )
    parser.add_argument(
        "--sae-model",
        default="exp/sod/ape/sae_models/sae_layer_2048d.pt",
        help="SAE model path",
    )
    parser.add_argument(
        "--activations",
        default="exp/sod/ape/activations/activations_layers_3.h5",
        help="Activations file path",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    print("🎼 REAL LIMUF EXTRACTION")
    print("=" * 40)
    print(f"Feature ID: {args.feature_id}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {args.device}")
    print()

    # Create extractor
    extractor = RealLiMuFExtractor(
        sae_model_path=args.sae_model,
        activations_path=args.activations,
        device=args.device,
    )

    # Extract LiMuF
    limuf_name = f"feature_{args.feature_id}"
    limuf_vector, metadata = extractor.extract_limuf(
        feature_id=args.feature_id,
        activation_threshold=args.threshold,
    )

    # Save results
    limufs = {limuf_name: (limuf_vector, metadata)}
    extractor.save_limufs(limufs=limufs, output_dir=args.output_dir)

    print("\n🎯 SUCCESS!")
    print(f"LiMuF extracted: {limuf_name}")
    print("Ready for intervention testing!")
    print()
    print("🎵 Next step:")
    print("python test_real_interventions.py \\")
    print(f"    --limuf-dir {args.output_dir} \\")
    print(f"    --limuf-name {limuf_name}")


if __name__ == "__main__":
    main()
