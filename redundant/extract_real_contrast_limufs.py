#!/usr/bin/env python3
"""
REAL Contrast LiMuF Extractor for Features 182 vs 1743

This script uses your existing infrastructure:
- SAE model: exp/sod/ape/sae_models/sae_layer_2048d.pt
- Activations: exp/sod/ape/activations/activations_layers_3.h5
- Interpretations: interpretation/enhanced_diversity/enhanced_diversity_report.json

Extracts contrast LiMuFs following LiReFs methodology with difference-in-means.
"""

import torch
import h5py
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm


class RealContrastLiMuFExtractor:
    """Extract contrast LiMuFs using your existing SAE and activations."""

    def __init__(
        self,
        sae_model_path: str,
        activations_path: str,
        interpretations_path: str,
        device: str = "cuda",
    ):
        self.device = device
        self.sae_model_path = Path(sae_model_path)
        self.activations_path = Path(activations_path)
        self.interpretations_path = Path(interpretations_path)

        # Load components
        self.sae_model = self._load_sae_model()
        self.activations = self._load_activations()
        self.interpretations = self._load_interpretations()

        print(f"✅ Loaded SAE model: {self.sae_model}")
        print(f"✅ Loaded activations: {self.activations.shape}")
        print(f"✅ Loaded interpretations for {len(self.interpretations)} features")

    def _load_sae_model(self):
        """Load your trained SAE model."""
        print(f"📦 Loading SAE from: {self.sae_model_path}")

        checkpoint = torch.load(self.sae_model_path, map_location="cpu")

        # Based on your SAE structure (adjust if needed)
        from sae.train_sae import SparseAutoencoder

        # Get model config from checkpoint
        config = checkpoint.get("config", {})
        input_dim = config.get("input_dim", 512)  # Adjust based on your model
        hidden_dim = config.get("hidden_dim", 2048)

        sae_model = SparseAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            sparsity_coeff=config.get("sparsity_coeff", 0.1),
        )

        sae_model.load_state_dict(checkpoint["model_state_dict"])
        sae_model.to(self.device)
        sae_model.eval()

        return sae_model

    def _load_activations(self):
        """Load residual stream activations from h5 file."""
        print(f"📁 Loading activations from: {self.activations_path}")

        with h5py.File(self.activations_path, "r") as f:
            # Your activations are stored as [n_samples, seq_len, d_model]
            activations = torch.tensor(f["activations"][:], dtype=torch.float32)

            # Extract last token activations (following LiReFs)
            if len(activations.shape) == 3:
                last_token_activations = activations[:, -1, :]  # [n_samples, d_model]
            else:
                last_token_activations = activations  # Already [n_samples, d_model]

        print(f"   Shape: {last_token_activations.shape}")
        return last_token_activations

    def _load_interpretations(self):
        """Load feature interpretations."""
        with open(self.interpretations_path, "r") as f:
            data = json.load(f)
        return data["interpretations"]

    def extract_contrast_limuf(
        self,
        feature_a_id: int,
        feature_b_id: int,
        activation_threshold: float = 2.0,
        min_samples: int = 100,
    ):
        """
        Extract contrast LiMuF using difference-in-means methodology.

        Args:
            feature_a_id: First feature (positive direction)
            feature_b_id: Second feature (negative direction)
            activation_threshold: Minimum activation to consider
            min_samples: Minimum samples per feature
        """
        print(f"\n🎯 EXTRACTING CONTRAST: Feature {feature_a_id} vs {feature_b_id}")
        print("=" * 60)

        # Get feature descriptions
        desc_a = self.interpretations.get(str(feature_a_id), {}).get(
            "interpretation", f"Feature {feature_a_id}"
        )
        desc_b = self.interpretations.get(str(feature_b_id), {}).get(
            "interpretation", f"Feature {feature_b_id}"
        )

        print(f"🎵 Feature {feature_a_id} (POSITIVE): {desc_a[:80]}...")
        print(f"🎵 Feature {feature_b_id} (NEGATIVE): {desc_b[:80]}...")
        print()

        # Step 1: Get SAE feature activations for all samples
        print("🔬 Computing SAE feature activations...")
        batch_size = 1000  # Process in batches to avoid memory issues
        n_samples = self.activations.shape[0]

        feature_a_activations = []
        feature_b_activations = []

        with torch.no_grad():
            for i in tqdm(range(0, n_samples, batch_size), desc="SAE encoding"):
                batch_end = min(i + batch_size, n_samples)
                batch_activations = self.activations[i:batch_end].to(self.device)

                # Get SAE feature activations
                _, sae_features = self.sae_model(
                    batch_activations
                )  # [batch, n_features]

                # Extract specific feature activations
                feature_a_activations.append(sae_features[:, feature_a_id].cpu())
                feature_b_activations.append(sae_features[:, feature_b_id].cpu())

        # Concatenate all batches
        feature_a_activations = torch.cat(feature_a_activations, dim=0)
        feature_b_activations = torch.cat(feature_b_activations, dim=0)

        print(f"✅ SAE activations computed for {n_samples} samples")

        # Step 2: Find samples where each feature fires strongly
        high_a_mask = feature_a_activations > activation_threshold
        high_b_mask = feature_b_activations > activation_threshold

        n_a_samples = high_a_mask.sum().item()
        n_b_samples = high_b_mask.sum().item()

        print(
            f"📊 Feature {feature_a_id} fires in {n_a_samples} samples (>{activation_threshold})"
        )
        print(
            f"📊 Feature {feature_b_id} fires in {n_b_samples} samples (>{activation_threshold})"
        )

        if n_a_samples < min_samples:
            print(
                f"❌ Not enough samples for Feature {feature_a_id} ({n_a_samples} < {min_samples})"
            )
            return None

        if n_b_samples < min_samples:
            print(
                f"❌ Not enough samples for Feature {feature_b_id} ({n_b_samples} < {min_samples})"
            )
            return None

        # Step 3: Get corresponding residual stream activations
        activations_when_a_fires = self.activations[
            high_a_mask
        ]  # [n_a_samples, d_model]
        activations_when_b_fires = self.activations[
            high_b_mask
        ]  # [n_b_samples, d_model]

        print(f"🎯 Residual activations when A fires: {activations_when_a_fires.shape}")
        print(f"🎯 Residual activations when B fires: {activations_when_b_fires.shape}")

        # Step 4: Calculate mean activations (difference-in-means)
        mean_a = torch.mean(activations_when_a_fires, dim=0)  # [d_model]
        mean_b = torch.mean(activations_when_b_fires, dim=0)  # [d_model]

        # Step 5: Create contrast vector
        contrast_vector = mean_a - mean_b  # [d_model]
        contrast_vector = contrast_vector / torch.norm(contrast_vector)  # Normalize

        print(f"✅ Contrast vector computed: {contrast_vector.shape}")
        print(f"   Norm: {torch.norm(contrast_vector):.4f}")

        # Step 6: Calculate statistics
        stats = {
            "feature_a_id": feature_a_id,
            "feature_b_id": feature_b_id,
            "feature_a_samples": n_a_samples,
            "feature_b_samples": n_b_samples,
            "feature_a_description": desc_a,
            "feature_b_description": desc_b,
            "activation_threshold": activation_threshold,
            "direction_norm": torch.norm(contrast_vector).item(),
            "mean_a_norm": torch.norm(mean_a).item(),
            "mean_b_norm": torch.norm(mean_b).item(),
            "dot_product_means": torch.dot(mean_a, mean_b).item(),
        }

        return contrast_vector, stats

    def save_contrast_limuf(
        self,
        contrast_vector: torch.Tensor,
        stats: dict,
        output_dir: str,
        contrast_name: str = None,
    ):
        """Save the contrast LiMuF."""
        if contrast_name is None:
            contrast_name = (
                f"feature_{stats['feature_a_id']}_vs_{stats['feature_b_id']}"
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save in the same format as your existing LiMuFs
        contrast_data = {
            "limufs": {contrast_name: contrast_vector},
            "metadata": {
                contrast_name: {
                    **stats,
                    "layer": 3,  # Based on your activations
                    "extraction_method": "difference_in_means",
                    "sae_model_path": str(self.sae_model_path),
                    "activations_path": str(self.activations_path),
                }
            },
        }

        output_file = output_path / "limufs.pt"
        torch.save(contrast_data, output_file)

        # Also save readable metadata
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(contrast_data["metadata"], f, indent=2)

        print(f"\n✅ SAVED CONTRAST LIMUF")
        print(f"   File: {output_file}")
        print(f"   Contrast: {contrast_name}")
        print(f"   Metadata: {metadata_file}")

        return output_file


def main():
    """Extract contrast LiMuF for Features 182 vs 1743."""
    parser = argparse.ArgumentParser(description="Extract real contrast LiMuFs")
    parser.add_argument(
        "--sae-model-path",
        default="exp/sod/ape/sae_models/sae_layer_2048d.pt",
        help="Path to trained SAE model",
    )
    parser.add_argument(
        "--activations-path",
        default="exp/sod/ape/activations/activations_layers_3.h5",
        help="Path to extracted activations",
    )
    parser.add_argument(
        "--interpretations-path",
        default="interpretation/enhanced_diversity/enhanced_diversity_report.json",
        help="Path to feature interpretations",
    )
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
        "--activation-threshold",
        type=float,
        default=2.0,
        help="Minimum activation threshold",
    )
    parser.add_argument(
        "--output-dir",
        default="real_contrast_limufs_182_vs_1743",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()

    print("🎼 REAL CONTRAST LIMUF EXTRACTION")
    print("=" * 50)
    print(f"SAE Model: {args.sae_model_path}")
    print(f"Activations: {args.activations_path}")
    print(f"Feature A: {args.feature_a} (positive direction)")
    print(f"Feature B: {args.feature_b} (negative direction)")
    print(f"Device: {args.device}")
    print()

    # Initialize extractor
    extractor = RealContrastLiMuFExtractor(
        sae_model_path=args.sae_model_path,
        activations_path=args.activations_path,
        interpretations_path=args.interpretations_path,
        device=args.device,
    )

    # Extract contrast
    result = extractor.extract_contrast_limuf(
        feature_a_id=args.feature_a,
        feature_b_id=args.feature_b,
        activation_threshold=args.activation_threshold,
    )

    if result is None:
        print("❌ Failed to extract contrast LiMuF")
        return False

    contrast_vector, stats = result

    # Save contrast
    output_file = extractor.save_contrast_limuf(
        contrast_vector=contrast_vector,
        stats=stats,
        output_dir=args.output_dir,
        contrast_name="steady_pulse_vs_melodic_sequences",
    )

    print(f"\n🎯 NEXT STEPS:")
    print(f"   1. Test interventions: python test_real_contrast_interventions.py")
    print(f"   2. Generate sequences with strengths [-2, -1, 0, +1, +2]")
    print(f"   3. Convert to audio and analyze differences")

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
