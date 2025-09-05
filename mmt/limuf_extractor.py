"""
Linear Music Features (LiMuFs) - Musical equivalent of LiReFs for controlling
specific musical concepts during generation.
"""

import torch
import h5py
from pathlib import Path
import json
from typing import Dict, List, Tuple

# Import your existing SAE model
from sae.train_sae import SparseAutoencoder


class LiMuFExtractor:
    """Extract Linear Music Features from SAE activations."""

    def __init__(
        self, sae_model_path: str, activations_path: str, device: str = "cuda"
    ):
        self.device = device
        self.sae_model_path = Path(sae_model_path)
        self.activations_path = Path(activations_path)

        # Load SAE model
        self.sae_model = self._load_sae_model()

        # Load activations data
        self.activations_data = self._load_activations()

        # Store extracted LiMuFs
        self.limufs: Dict[int, torch.Tensor] = {}
        self.feature_stats: Dict[int, Dict] = {}

    def _load_sae_model(self):
        """Load the trained SAE model."""
        print(f"📦 Loading SAE model from: {self.sae_model_path}")

        checkpoint = torch.load(self.sae_model_path, map_location="cpu")

        # Reconstruct SAE model from checkpoint
        model = SparseAutoencoder(
            input_dim=checkpoint["input_dim"],
            hidden_dim=(
                checkpoint["config"]["hidden_dim"]
                if "config" in checkpoint
                else checkpoint["hidden_dim"]
            ),
            sparsity_coeff=(
                checkpoint["config"]["sparsity_coeff"]
                if "config" in checkpoint
                else checkpoint["sparsity_coeff"]
            ),
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print(f"✅ SAE model loaded: {checkpoint['input_dim']} -> {model.hidden_dim}")

        return model

    def _load_activations(self):
        """Load activation data from h5 file."""
        print(f"📥 Loading activations from {self.activations_path}")

        activations_data = {}
        with h5py.File(self.activations_path, "r") as f:
            # Load the layer activations
            layer_keys = [key for key in f.keys() if key.startswith("layer_")]

            if len(layer_keys) == 1:
                layer_key = layer_keys[0]
                activations_data["activations"] = torch.tensor(f[layer_key][:])
                activations_data["metadata"] = dict(f[layer_key].attrs)
                print(
                    f"✅ Loaded activations shape: {activations_data['activations'].shape}"
                )

                # Also load metadata if available
                for meta_key in ["file_names", "sequence_lengths"]:
                    if meta_key in f:
                        activations_data[meta_key] = list(f[meta_key][:])

            else:
                raise ValueError(f"Expected single layer, found: {layer_keys}")

        return activations_data

    def extract_limuf_for_feature(
        self,
        feature_id: int,
        high_threshold: float = 2.0,
        low_threshold: float = 0.5,
        min_samples: int = 100,
    ) -> torch.Tensor:
        """
        Extract Linear Music Feature direction for a specific SAE feature.

        This implements the difference-in-means methodology from the LiReFs paper,
        adapted for musical concepts.

        Args:
            feature_id: ID of the SAE feature
            high_threshold: Threshold for high activation samples
            low_threshold: Threshold for low activation samples
            min_samples: Minimum samples needed for reliable extraction

        Returns:
            Normalized LiMuF direction vector
        """
        print(f"🎵 Extracting LiMuF for feature {feature_id}")

        # Get original activations
        original_activations = self.activations_data["activations"].to(self.device)

        with torch.no_grad():
            # Pass through SAE encoder to get feature activations
            sae_activations = self.sae_model.encoder(original_activations)

            # Get activations for this specific feature
            feature_activations = sae_activations[:, feature_id]

            # Find high and low activation samples
            high_mask = feature_activations > high_threshold
            low_mask = feature_activations < low_threshold

            high_count = high_mask.sum().item()
            low_count = low_mask.sum().item()

            print(f"   High activation samples: {high_count}")
            print(f"   Low activation samples: {low_count}")
            print(
                f"   Feature activation range: [{feature_activations.min():.3f}, {feature_activations.max():.3f}]"
            )

            if high_count < min_samples or low_count < min_samples:
                print(f"   ⚠️  Insufficient samples for reliable LiMuF extraction")
                print(
                    f"   Need at least {min_samples} samples, got {high_count} high, {low_count} low"
                )
                return None

            # Compute difference-in-means (LiReFs methodology)
            high_mean = original_activations[high_mask].mean(dim=0)
            low_mean = original_activations[low_mask].mean(dim=0)

            # LiMuF direction = difference vector
            limuf_direction = high_mean - low_mean

            # Normalize the direction
            direction_norm = limuf_direction.norm()
            if direction_norm > 0:
                limuf_direction = limuf_direction / direction_norm
            else:
                print(f"   ⚠️  Zero norm direction, skipping feature {feature_id}")
                return None

            # Store statistics
            self.feature_stats[feature_id] = {
                "high_samples": high_count,
                "low_samples": low_count,
                "high_threshold": high_threshold,
                "low_threshold": low_threshold,
                "direction_norm": direction_norm.item(),
                "feature_activation_mean": feature_activations.mean().item(),
                "feature_activation_std": feature_activations.std().item(),
            }

            # Store the LiMuF
            self.limufs[feature_id] = limuf_direction.cpu()

            print(f"   ✅ LiMuF extracted successfully (norm: {direction_norm:.4f})")
            return limuf_direction.cpu()

    def extract_multiple_limufs(
        self,
        feature_interpretations: Dict,
        categories: List[str] = None,
        max_features: int = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract LiMuFs for multiple features from interpretation results.

        Args:
            feature_interpretations: Dict from enhanced_diversity_report
            categories: List of musical categories to extract (e.g., ['rhythmic_specific'])
            max_features: Maximum number of features to extract (None = all)

        Returns:
            Dictionary mapping feature_id to LiMuF direction
        """

        if categories is None:
            categories = [
                "rhythmic_specific",
                "harmonic_specific",
                "textural_specific",
                "dynamic_specific",
                "structural_specific",
            ]

        extracted_limufs = {}
        processed_count = 0

        print(f"🎼 Extracting LiMuFs for categories: {categories}")
        if max_features:
            print(f"   Maximum features to process: {max_features}")

        for feature_id_str, interpretation in feature_interpretations.items():
            if max_features and processed_count >= max_features:
                print(f"   Reached maximum feature limit ({max_features})")
                break

            feature_id = int(feature_id_str)

            # Filter by category if specified
            musical_category = interpretation.get("musical_category", "unknown")
            if musical_category not in categories:
                continue

            # Use feature statistics for thresholds
            feature_stats = interpretation.get("feature_stats", {})
            mean_activation = feature_stats.get("mean_activation", 2.0)

            # Set thresholds based on feature statistics
            # Make thresholds more conservative for better separation
            high_threshold = max(mean_activation * 1.2, 1.5)
            low_threshold = mean_activation * 0.2

            print(f"\n🎯 Processing Feature {feature_id}: {musical_category}")
            description = interpretation.get("interpretation", "No description")
            print(f"   Description: {description[:100]}...")
            print(f"   Thresholds: low={low_threshold:.3f}, high={high_threshold:.3f}")

            limuf = self.extract_limuf_for_feature(
                feature_id=feature_id,
                high_threshold=high_threshold,
                low_threshold=low_threshold,
            )

            if limuf is not None:
                extracted_limufs[feature_id] = limuf
                processed_count += 1

        print(f"\n🎉 Extracted {len(extracted_limufs)} LiMuFs successfully!")
        return extracted_limufs

    def validate_limuf_quality(
        self, feature_id: int, num_test_samples: int = 1000
    ) -> Dict:
        """
        Validate the quality of an extracted LiMuF by testing its discriminative power.

        Args:
            feature_id: Feature to validate
            num_test_samples: Number of samples to use for validation

        Returns:
            Validation metrics
        """
        if feature_id not in self.limufs:
            raise ValueError(f"Feature {feature_id} not found in extracted LiMuFs")

        print(f"🔍 Validating LiMuF quality for feature {feature_id}")

        limuf_direction = self.limufs[feature_id].to(self.device)
        original_activations = self.activations_data["activations"].to(self.device)

        # Randomly sample test data
        total_samples = original_activations.shape[0]
        test_indices = torch.randperm(total_samples)[
            : min(num_test_samples, total_samples)
        ]
        test_activations = original_activations[test_indices]

        with torch.no_grad():
            # Get SAE feature activations
            sae_activations = self.sae_model.encoder(test_activations)
            target_feature_activations = sae_activations[:, feature_id]

            # Project onto LiMuF direction
            limuf_projections = torch.matmul(test_activations, limuf_direction)

            # Compute correlation between SAE feature and LiMuF projection
            correlation = torch.corrcoef(
                torch.stack([target_feature_activations, limuf_projections])
            )[0, 1]

            # Compute other metrics
            feature_stats = self.feature_stats[feature_id]
            high_threshold = feature_stats["high_threshold"]

            high_feature_mask = target_feature_activations > high_threshold
            high_projection_values = limuf_projections[high_feature_mask]
            low_projection_values = limuf_projections[~high_feature_mask]

            separation_score = (
                high_projection_values.mean() - low_projection_values.mean()
            ) / (high_projection_values.std() + low_projection_values.std() + 1e-8)

        validation_metrics = {
            "correlation": correlation.item(),
            "separation_score": separation_score.item(),
            "high_projection_mean": high_projection_values.mean().item(),
            "low_projection_mean": low_projection_values.mean().item(),
            "projection_std": limuf_projections.std().item(),
            "num_test_samples": len(test_indices),
        }

        print(f"   Correlation with SAE feature: {correlation:.4f}")
        print(f"   Separation score: {separation_score:.4f}")

        return validation_metrics

    def save_limufs(self, output_path: str):
        """Save extracted LiMuFs to disk."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save LiMuF directions and metadata
        limufs_data = {
            "limufs": {str(k): v.numpy() for k, v in self.limufs.items()},
            "feature_stats": self.feature_stats,
            "metadata": {
                "sae_model_path": str(self.sae_model_path),
                "activations_path": str(self.activations_path),
                "extraction_method": "difference_in_means",
                "total_features": len(self.limufs),
                "activations_shape": list(self.activations_data["activations"].shape),
            },
        }

        # Save as both pickle and json
        torch.save(limufs_data, output_path / "limufs.pt")

        # Save readable summary
        summary = {
            "extracted_features": list(self.limufs.keys()),
            "feature_stats": self.feature_stats,
            "total_features": len(self.limufs),
            "extraction_metadata": limufs_data["metadata"],
        }

        with open(output_path / "limuf_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"💾 LiMuFs saved to {output_path}")
        print(f"   - limufs.pt: Full LiMuF data")
        print(f"   - limuf_summary.json: Human-readable summary")

    @classmethod
    def load_limufs(cls, limuf_path: str) -> Tuple[Dict[int, torch.Tensor], Dict]:
        """Load previously extracted LiMuFs."""
        limuf_path = Path(limuf_path)

        limuf_file = limuf_path / "limufs.pt"
        if limuf_file.exists():
            print(f"📁 Loading LiMuFs from {limuf_file}")
            data = torch.load(limuf_file, map_location="cpu")
            limufs = {int(k): torch.tensor(v) for k, v in data["limufs"].items()}
            metadata = data.get("metadata", {})
            print(f"✅ Loaded {len(limufs)} LiMuFs")
            return limufs, metadata
        else:
            raise FileNotFoundError(f"LiMuF file not found: {limuf_file}")
