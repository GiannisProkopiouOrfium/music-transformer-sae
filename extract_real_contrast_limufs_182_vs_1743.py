#!/usr/bin/env python3
"""
REAL Contrast LiMuF Extractor for Features 182 vs 1743

Uses your existing infrastructure:
- SAE model: exp/sod/ape/sae_models/sae_layer_2048d.pt
- Activations: exp/sod/ape/activations/activations_layers_3.h5
- Interpretations: interpretation/enhanced_diversity/enhanced_diversity_report.json

NO SIMULATION - uses real difference-in-means on actual activations.
"""

import torch
import h5py
import json
from pathlib import Path
import argparse
from tqdm import tqdm

# Import your SAE architecture
from sae.train_sae import SparseAutoencoder


class RealContrastLiMuFExtractor:
    """Extract contrast LiMuFs using real SAE feature activations and residual stream data."""

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

    def extract_contrast_limuf(
        self,
        feature_a_id: int,
        feature_b_id: int,
        activation_threshold: float = 2.0,
        min_samples: int = 100,
        max_samples: int = 10000,
    ):
        """
        Extract contrast LiMuF using real difference-in-means methodology.

        Args:
            feature_a_id: First feature ID (positive direction)
            feature_b_id: Second feature ID (negative direction)
            activation_threshold: Minimum activation to consider feature "active"
            min_samples: Minimum samples needed per feature
            max_samples: Maximum samples to use (for efficiency)
        """
        print(f"🎯 EXTRACTING CONTRAST: Feature {feature_a_id} vs {feature_b_id}")
        print("=" * 60)

        # Get feature descriptions
        desc_a = self.interpretations.get(str(feature_a_id), {}).get(
            "interpretation", f"Feature {feature_a_id}"
        )
        desc_b = self.interpretations.get(str(feature_b_id), {}).get(
            "interpretation", f"Feature {feature_b_id}"
        )

        print(f"🎵 Feature {feature_a_id} (Positive): {desc_a[:80]}...")
        print(f"🎵 Feature {feature_b_id} (Negative): {desc_b[:80]}...")
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

        # Find samples where each feature is active
        feature_a_strengths = feature_activations[:, feature_a_id]
        feature_b_strengths = feature_activations[:, feature_b_id]

        # Create masks for high activation samples
        high_a_mask = feature_a_strengths > activation_threshold
        high_b_mask = feature_b_strengths > activation_threshold

        n_a_samples = high_a_mask.sum().item()
        n_b_samples = high_b_mask.sum().item()

        print(f"📊 Feature activation analysis:")
        print(
            f"   Feature {feature_a_id}: {n_a_samples} samples above threshold {activation_threshold}"
        )
        print(
            f"   Feature {feature_b_id}: {n_b_samples} samples above threshold {activation_threshold}"
        )

        if n_a_samples < min_samples:
            print(
                f"⚠️  Warning: Only {n_a_samples} samples for Feature {feature_a_id} (min: {min_samples})"
            )
        if n_b_samples < min_samples:
            print(
                f"⚠️  Warning: Only {n_b_samples} samples for Feature {feature_b_id} (min: {min_samples})"
            )

        if n_a_samples == 0 or n_b_samples == 0:
            raise ValueError("Not enough samples found for contrast extraction")

        # Extract corresponding residual stream activations
        activations_when_a_fires = activations_subset[
            high_a_mask
        ]  # [n_a_samples, d_model]
        activations_when_b_fires = activations_subset[
            high_b_mask
        ]  # [n_b_samples, d_model]

        # Calculate mean activations (LiReFs difference-in-means)
        mean_a = torch.mean(activations_when_a_fires, dim=0)  # [d_model]
        mean_b = torch.mean(activations_when_b_fires, dim=0)  # [d_model]

        # The contrast vector
        contrast_vector = mean_a - mean_b  # [d_model]
        contrast_norm = torch.norm(contrast_vector, dim=0).item()

        # Normalize
        contrast_vector_normalized = contrast_vector / torch.norm(contrast_vector, dim=0)

        print("🎯 Contrast vector extracted:")
        print(f"   Raw norm: {contrast_norm:.4f}")
        print(
            f"   Normalized norm: {torch.norm(contrast_vector_normalized, dim=0).item():.4f}"
        )
        print(f"   Vector shape: {contrast_vector_normalized.shape}")

        # Calculate some statistics
        dot_product = torch.dot(mean_a, mean_b).item()
        a_norm = torch.norm(mean_a, dim=0).item()
        b_norm = torch.norm(mean_b, dim=0).item()

        print(f"📈 Statistics:")
        print(f"   Mean A norm: {a_norm:.4f}")
        print(f"   Mean B norm: {b_norm:.4f}")
        print(f"   Dot product: {dot_product:.4f}")
        print(f"   Cosine similarity: {dot_product / (a_norm * b_norm):.4f}")

        # Create metadata
        metadata = {
            "feature_a_id": feature_a_id,
            "feature_b_id": feature_b_id,
            "feature_a_description": desc_a,
            "feature_b_description": desc_b,
            "feature_a_samples": n_a_samples,
            "feature_b_samples": n_b_samples,
            "activation_threshold": activation_threshold,
            "contrast_norm": contrast_norm,
            "layer": 3,  # Based on your activations file
            "method": "difference_in_means",
            "statistics": {
                "mean_a_norm": a_norm,
                "mean_b_norm": b_norm,
                "dot_product": dot_product,
                "cosine_similarity": dot_product / (a_norm * b_norm),
            },
        }

        return contrast_vector_normalized, metadata

    def save_contrast_limufs(
        self, contrasts: dict, output_dir: str = "real_contrast_limufs_layer3"
    ):
        """Save extracted contrast LiMuFs."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Separate vectors and metadata
        limufs = {}
        metadata = {}

        for name, (vector, meta) in contrasts.items():
            limufs[name] = vector
            metadata[name] = meta

        # Save
        output_file = output_path / "limufs.pt"
        torch.save(
            {
                "limufs": limufs,
                "metadata": metadata,
                "extraction_info": {
                    "sae_model_path": str(self.sae_model_path),
                    "activations_path": str(self.activations_path),
                    "interpretations_path": str(self.interpretations_path),
                    "method": "real_difference_in_means",
                },
            },
            output_file,
        )

        # Save metadata as JSON for readability
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Saved contrast LiMuFs to: {output_file}")
        print(f"✅ Saved metadata to: {metadata_file}")

        return output_file


def main():
    parser = argparse.ArgumentParser(description="Extract real contrast LiMuFs")
    parser.add_argument(
        "--feature-a",
        type=int,
        default=182,
        help="First feature ID (positive direction)",
    )
    parser.add_argument(
        "--feature-b",
        type=int,
        default=1743,
        help="Second feature ID (negative direction)",
    )
    parser.add_argument(
        "--threshold", type=float, default=2.0, help="Activation threshold"
    )
    parser.add_argument(
        "--output-dir", default="real_contrast_limufs_layer3", help="Output directory"
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

    print("🎼 REAL CONTRAST LIMUF EXTRACTION")
    print("=" * 50)
    print(f"Feature A (positive): {args.feature_a}")
    print(f"Feature B (negative): {args.feature_b}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {args.device}")
    print()

    # Create extractor
    extractor = RealContrastLiMuFExtractor(
        sae_model_path=args.sae_model,
        activations_path=args.activations,
        device=args.device,
    )

    # Extract contrast
    contrast_name = f"feature_{args.feature_a}_vs_{args.feature_b}"
    contrast_vector, metadata = extractor.extract_contrast_limuf(
        feature_a_id=args.feature_a,
        feature_b_id=args.feature_b,
        activation_threshold=args.threshold,
    )

    # Save results
    contrasts = {contrast_name: (contrast_vector, metadata)}
    output_file = extractor.save_contrast_limufs(
        contrasts=contrasts, output_dir=args.output_dir
    )

    print(f"\n🎯 SUCCESS!")
    print(f"Contrast LiMuF extracted: {contrast_name}")
    print(f"Ready for intervention testing!")
    print()
    print(f"🎵 Next step:")
    print(f"python test_real_contrast_interventions.py \\")
    print(f"    --limuf-dir {args.output_dir} \\")
    print(f"    --contrast-name {contrast_name}")


if __name__ == "__main__":
    main()
