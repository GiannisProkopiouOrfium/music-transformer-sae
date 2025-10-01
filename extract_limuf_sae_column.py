#!/usr/bin/env python3
"""
SAE Column-Based LiMuF Extractor for Any Layer and Feature

This version extracts Linear Musical Features (LiMuFs) directly from the SAE decoder
columns instead of computing empirical means from activations. The SAE decoder column
for each feature represents the learned direction that the SAE uses to reconstruct
that feature's contribution to the original activation space.

This approach is:
- Much faster (no activation processing needed)
- More direct (uses the actual learned feature direction)
- Theoretically cleaner (decoder column IS the feature direction)

Usage examples:
python extract_limuf_sae_column.py --layer 1 --feature-id 256
python extract_limuf_sae_column.py --layer 5 --feature-id 471
python extract_limuf_sae_column.py --layer 3 --feature-id 182
"""

import torch
import json
from pathlib import Path
import argparse

# Import your SAE architecture
from sae.train_sae import SparseAutoencoder


class SAEColumnLiMuFExtractor:
    """Extract LiMuFs directly from SAE decoder columns."""

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
        self.interpretations_path = Path(
            f"layer_interpretations/enhanced_diversity_report_layer{layer}.json"
        )

        # Verify paths exist
        self._verify_paths()

        # Load components
        print(f"🎼 LOADING LAYER {layer} SAE FOR COLUMN EXTRACTION")
        print("=" * 50)
        self.sae_model = self._load_sae_model()
        self.interpretations = self._load_interpretations()

        print(f"✅ SAE Model loaded: {type(self.sae_model)}")
        print(f"✅ SAE decoder shape: {self.sae_model.decoder.weight.shape}")
        print(f"✅ Interpretations for {len(self.interpretations)} features")
        print()

    def _verify_paths(self):
        """Verify all required paths exist."""
        missing_paths = []

        if not self.sae_model_path.exists():
            missing_paths.append(f"SAE model: {self.sae_model_path}")
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

    def extract_limuf_from_column(self, feature_id: int):
        """
        Extract LiMuF directly from SAE decoder column.

        Args:
            feature_id: Feature ID to extract direction for

        Returns:
            tuple: (limuf_vector, metadata)
        """
        print("🎯 EXTRACTING LIMUF FROM SAE COLUMN")
        print(f"   Layer {self.layer}, Feature {feature_id}")
        print("=" * 50)

        # Get feature information
        feature_info = self.get_feature_info(feature_id)
        desc = feature_info.get("interpretation", f"Feature {feature_id}")
        category = feature_info.get("musical_category", "unknown")

        print(f"🎵 Feature {feature_id} ({category}): {desc[:80]}...")
        print()

        # Validate feature_id is within SAE bounds
        n_features = self.sae_model.decoder.weight.shape[0]
        if feature_id >= n_features:
            raise ValueError(
                f"Feature ID {feature_id} exceeds SAE dimension {n_features}"
            )

        # Extract decoder column for this feature
        with torch.no_grad():
            # The decoder weight shape is [n_features, d_model]
            # We want the feature_id-th row, which represents how this feature
            # contributes to reconstructing the original activation space
            limuf_vector_raw = (
                self.sae_model.decoder.weight[feature_id, :].clone().detach()
            )

        # Calculate statistics
        raw_norm = torch.norm(limuf_vector_raw, dim=0).item()

        # Normalize the vector
        limuf_vector_normalized = limuf_vector_raw / torch.norm(limuf_vector_raw, dim=0)
        normalized_norm = torch.norm(limuf_vector_normalized, dim=0).item()

        print("🎯 LiMuF vector extracted from SAE decoder:")
        print(f"   Raw norm: {raw_norm:.4f}")
        print(f"   Normalized norm: {normalized_norm:.4f}")
        print(f"   Vector shape: {limuf_vector_normalized.shape}")
        print(f"   Device: {limuf_vector_normalized.device}")

        # Additional statistics
        print("📈 Vector statistics:")
        print(f"   Mean: {limuf_vector_raw.mean().item():.6f}")
        print(f"   Std: {limuf_vector_raw.std().item():.6f}")
        print(f"   Min: {limuf_vector_raw.min().item():.6f}")
        print(f"   Max: {limuf_vector_raw.max().item():.6f}")
        print(f"   Non-zero elements: {(limuf_vector_raw != 0).sum().item()}")

        # Create metadata
        metadata = {
            "feature_id": feature_id,
            "layer": self.layer,
            "feature_description": desc,
            "feature_category": category,
            "limuf_norm": raw_norm,
            "method": "sae_decoder_column",
            "extraction_source": "sae_decoder_weights",
            "sae_info": {
                "sae_model_path": str(self.sae_model_path),
                "sae_dimensions": {
                    "n_features": self.sae_model.decoder.weight.shape[0],
                    "d_model": self.sae_model.decoder.weight.shape[1],
                },
            },
            "vector_statistics": {
                "raw_norm": raw_norm,
                "mean": limuf_vector_raw.mean().item(),
                "std": limuf_vector_raw.std().item(),
                "min": limuf_vector_raw.min().item(),
                "max": limuf_vector_raw.max().item(),
                "non_zero_elements": (limuf_vector_raw != 0).sum().item(),
                "sparsity": 1.0
                - (limuf_vector_raw != 0).sum().item() / limuf_vector_raw.numel(),
            },
            "feature_interpretation": feature_info,
        }

        return limuf_vector_normalized.cpu(), metadata

    def save_limufs(self, limufs: dict, output_dir: str = None, feature_id: int = None):
        """Save extracted LiMuFs."""
        if output_dir is None:
            output_dir = f"limufs_layer{self.layer}_feature{feature_id}_sae_columns"

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
                    "method": "sae_decoder_column",
                    "extractor_version": "sae_column_v1.0",
                    "description": "LiMuFs extracted directly from SAE decoder columns",
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
        description="Extract LiMuFs from SAE decoder columns for any layer and feature"
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
        "--output-dir", help="Output directory (default: limufs_layer{N}_sae_columns)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load SAE model on",
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

    # Multiple features
    parser.add_argument(
        "--feature-ids",
        nargs="+",
        type=int,
        help="Extract multiple features at once",
    )

    args = parser.parse_args()

    print("🎼 SAE COLUMN-BASED LIMUF EXTRACTION")
    print("=" * 45)
    print(f"Layer: {args.layer}")
    print(f"Device: {args.device}")
    print("Method: Direct SAE decoder column extraction")
    print()

    # Create extractor
    extractor = SAEColumnLiMuFExtractor(
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

    # Determine which features to extract
    feature_ids = []
    if args.feature_ids:
        feature_ids = args.feature_ids
    elif args.feature_id is not None:
        feature_ids = [args.feature_id]
    else:
        raise ValueError("Must specify either --feature-id or --feature-ids")

    print(f"🎯 Extracting LiMuFs for features: {feature_ids}")
    print()

    # Extract LiMuFs
    limufs = {}
    for feature_id in feature_ids:
        print(f"\n{'='*60}")
        print(f"PROCESSING FEATURE {feature_id}")
        print("=" * 60)

        limuf_name = f"layer{args.layer}_feature_{feature_id}"
        limuf_vector, metadata = extractor.extract_limuf_from_column(
            feature_id=feature_id
        )
        limufs[limuf_name] = (limuf_vector, metadata)

        print(f"✅ Feature {feature_id} extracted as '{limuf_name}'")

    # Save results
    output_dir = args.output_dir or f"limufs_layer{args.layer}_sae_columns"
    extractor.save_limufs(limufs=limufs, output_dir=output_dir, feature_id=feature_id)

    print("\n🎯 SUCCESS!")
    print(f"Layer: {args.layer}")
    print(f"Features extracted: {len(limufs)}")
    print("Method: SAE decoder column extraction")
    print("Ready for intervention testing!")
    print()
    print("🎵 Next step:")
    print("python test_interventions.py \\")
    print(f"    --limuf-dir {output_dir} \\")
    print(f"    --layer {args.layer}")


if __name__ == "__main__":
    main()
