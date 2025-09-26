#!/usr/bin/env python3
"""
Flexible LiMuF Extractor for Any Layer and Feature

Supports:
- Any layer (1, 3, 5, etc.) with automatic path detection
- Any feature ID from the interpretations
- Configurable thresholds and parameters
- Automatic metadata loading from interpretation files

Usage examples:
python extract_flexible_limuf.py --layer 1 --feature-id 256 --threshold 2.0
python extract_flexible_limuf.py --layer 5 --feature-id 471 --threshold 1.5
python extract_flexible_limuf.py --layer 3 --feature-id 182 --threshold 2.0
"""

import torch
import h5py
import json
from pathlib import Path
import argparse
from tqdm import tqdm

# Import your SAE architecture
from sae.train_sae import SparseAutoencoder


class FlexibleLiMuFExtractor:
    """Extract LiMuFs from any layer and feature using real SAE infrastructure."""

    def __init__(
        self,
        layer: int,
        device: str = "cuda",
        base_path: str = "exp/sod/ape",
    ):
        self.layer = layer
        self.device = device
        self.base_path = Path(base_path)

        # Auto-detect paths based on layer
        self.sae_model_path = (
            self.base_path / f"layer_{layer}" / "sae_models" / "sae_layer_2048d.pt"
        )
        self.activations_path = (
            self.base_path / "activations" / f"activations_layers_{layer}.h5"
        )
        self.interpretations_path = Path(
            f"layer_interpretations/enhanced_diversity_report_layer{layer}.json"
        )

        # Verify paths exist
        self._verify_paths()

        # Load components
        print(f"🎼 LOADING LAYER {layer} SAE INFRASTRUCTURE")
        print("=" * 50)
        self.sae_model = self._load_sae_model()
        self.activations_data = self._load_activations()
        self.interpretations = self._load_interpretations()

        print(f"✅ SAE Model loaded: {type(self.sae_model)}")
        print(f"✅ Activations shape: {self.activations_data.shape}")
        print(f"✅ Interpretations for {len(self.interpretations)} features")
        print()

    def _verify_paths(self):
        """Verify all required paths exist."""
        missing_paths = []

        if not self.sae_model_path.exists():
            missing_paths.append(f"SAE model: {self.sae_model_path}")
        if not self.activations_path.exists():
            missing_paths.append(f"Activations: {self.activations_path}")
        if not self.interpretations_path.exists():
            missing_paths.append(f"Interpretations: {self.interpretations_path}")

        if missing_paths:
            print("❌ MISSING FILES:")
            for path in missing_paths:
                print(f"   {path}")
            raise FileNotFoundError("Required files not found. Check paths above.")

        print(f"✅ All paths verified for layer {self.layer}")

    def _load_sae_model(self):
        """Load the SAE model for the specified layer."""
        print(f"📦 Loading SAE from: {self.sae_model_path}")

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

        with h5py.File(self.activations_path, "r") as f:
            # Check what's in the file
            print(f"   H5 file keys: {list(f.keys())}")

            # Load activations (try different possible key names)
            possible_keys = [
                "activations",
                f"layer_{self.layer}",
                f"activations_layer_{self.layer}",
                list(f.keys())[0],  # fallback to first key
            ]

            activations = None
            for key in possible_keys:
                if key in f:
                    print(f"   Using key: {key}")
                    activations = torch.tensor(f[key][:], dtype=torch.float32)
                    break

            if activations is None:
                raise ValueError(
                    f"Could not find activations in file. Available keys: {list(f.keys())}"
                )

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
        print(
            f"   Found interpretations for features: {sorted([int(k) for k in interpretations.keys()])}"
        )

        return interpretations

    def get_feature_info(self, feature_id: int):
        """Get information about a specific feature."""
        feature_str = str(feature_id)
        if feature_str not in self.interpretations:
            available_features = sorted([int(k) for k in self.interpretations.keys()])
            raise ValueError(
                f"Feature {feature_id} not found. Available features: {available_features}"
            )

        feature_data = self.interpretations[feature_str]

        print(f"🎵 FEATURE {feature_id} INFO:")
        print(f"   Category: {feature_data.get('musical_category', 'unknown')}")
        print(
            f"   Description: {feature_data.get('interpretation', 'No description')[:100]}..."
        )

        stats = feature_data.get("feature_stats", {})
        if stats:
            print(f"   Total activations: {stats.get('total_activations', 'unknown')}")
            print(f"   Mean activation: {stats.get('mean_activation', 0):.3f}")
            print(f"   Max activation: {stats.get('max_activation', 0):.3f}")
            print(
                f"   Activation frequency: {stats.get('activation_frequency', 0):.6f}"
            )

        return feature_data

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
        print(f"🎯 EXTRACTING LIMUF FOR LAYER {self.layer}, FEATURE {feature_id}")
        print("=" * 60)

        # Get feature information
        feature_info = self.get_feature_info(feature_id)
        desc = feature_info.get("interpretation", f"Feature {feature_id}")
        category = feature_info.get("musical_category", "unknown")

        print(f"🎵 Feature {feature_id} ({category}): {desc[:80]}...")
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
            "layer": self.layer,
            "feature_description": desc,
            "feature_category": category,
            "active_samples": n_active_samples,
            "total_samples": n_samples,
            "activation_threshold": activation_threshold,
            "limuf_norm": mean_norm,
            "method": "mean_activation",
            "extraction_paths": {
                "sae_model": str(self.sae_model_path),
                "activations": str(self.activations_path),
                "interpretations": str(self.interpretations_path),
            },
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

    def save_limufs(self, limufs: dict, output_dir: str = None):
        """Save extracted LiMuFs."""
        if output_dir is None:
            output_dir = f"limufs_layer{self.layer}"

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
                    "layer": self.layer,
                    "method": "real_mean_activation",
                    "extractor_version": "flexible_v1.0",
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
        description="Extract LiMuFs for any layer and feature"
    )

    # Required arguments
    parser.add_argument(
        "--layer", type=int, required=True, help="Layer number (1, 3, 5, etc.)"
    )
    parser.add_argument(
        "--feature-id", type=int, required=True, help="Feature ID to extract"
    )

    # Optional arguments
    parser.add_argument(
        "--threshold", type=float, default=2.0, help="Activation threshold"
    )
    parser.add_argument(
        "--output-dir", help="Output directory (default: limufs_layer{N})"
    )
    parser.add_argument(
        "--max-samples", type=int, default=10000, help="Maximum samples to process"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--base-path", default="exp/sod/ape", help="Base path for SAE files"
    )

    # Info mode
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="List available features for the layer",
    )

    args = parser.parse_args()

    print("🎼 FLEXIBLE LIMUF EXTRACTION")
    print("=" * 40)
    print(f"Layer: {args.layer}")
    print(f"Feature ID: {args.feature_id}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {args.device}")
    print()

    # Create extractor
    extractor = FlexibleLiMuFExtractor(
        layer=args.layer,
        device=args.device,
        base_path=args.base_path,
    )

    # List features mode
    if args.list_features:
        print("📋 AVAILABLE FEATURES:")
        for feature_id_str in sorted(extractor.interpretations.keys(), key=int):
            feature_id = int(feature_id_str)
            feature_data = extractor.interpretations[feature_id_str]
            category = feature_data.get("musical_category", "unknown")
            desc = feature_data.get("interpretation", "No description")[:60]
            print(f"   {feature_id:4d}: {category:15s} - {desc}...")
        return

    # Extract LiMuF
    limuf_name = f"layer{args.layer}_feature_{args.feature_id}"
    limuf_vector, metadata = extractor.extract_limuf(
        feature_id=args.feature_id,
        activation_threshold=args.threshold,
        max_samples=args.max_samples,
    )

    # Save results
    limufs = {limuf_name: (limuf_vector, metadata)}
    output_dir = args.output_dir or f"limufs_layer{args.layer}"
    extractor.save_limufs(limufs=limufs, output_dir=output_dir)

    print("\n🎯 SUCCESS!")
    print(f"Layer: {args.layer}")
    print(f"Feature: {args.feature_id} ({metadata['feature_category']})")
    print(f"LiMuF extracted: {limuf_name}")
    print("Ready for intervention testing!")
    print()
    print("🎵 Next step:")
    print("python test_interventions.py \\")
    print(f"    --limuf-dir {output_dir} \\")
    print(f"    --limuf-name {limuf_name} \\")
    print(f"    --layer {args.layer}")


if __name__ == "__main__":
    main()
