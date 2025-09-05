"""
Validation framework for feature interventions
Measures SAE feature activations to confirm interventions work as expected
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict, List, Optional
from pathlib import Path
from single_feature_intervention import SingleFeatureInterventionWrapper


class InterventionValidator:
    """
    Validates that feature interventions actually affect the targeted features
    """

    def __init__(
        self,
        intervention_wrapper: SingleFeatureInterventionWrapper,
        sae_model_path: str,
    ):
        self.wrapper = intervention_wrapper
        self.device = self.wrapper.device

        # Load the trained SAE for measuring activations
        sae_checkpoint = torch.load(sae_model_path, map_location=self.device)

        # Extract SAE components
        if "encoder.weight" in sae_checkpoint and "encoder.bias" in sae_checkpoint:
            self.sae_encoder_weight = sae_checkpoint[
                "encoder.weight"
            ]  # [n_features, hidden_dim]
            self.sae_encoder_bias = sae_checkpoint["encoder.bias"]  # [n_features]
        else:
            raise ValueError("Could not find SAE encoder weights in checkpoint")

        if "decoder.weight" in sae_checkpoint:
            self.sae_decoder_weight = sae_checkpoint[
                "decoder.weight"
            ]  # [hidden_dim, n_features]
        else:
            raise ValueError("Could not find SAE decoder weights in checkpoint")

        print(f"📊 Loaded SAE with {self.sae_encoder_weight.shape[0]} features")

        # Store activations during generation
        self.collected_activations = []
        self.activation_hooks = []

    def setup_activation_collection(self, layer_idx: int = 3):
        """
        Set up hooks to collect activations from the specified layer during generation
        """
        self.collected_activations = []

        def activation_hook(module, input, output):
            """Hook to collect layer activations"""
            if len(output.shape) == 3:  # [batch, seq, hidden]
                # Store the activations (detach to avoid gradient issues)
                self.collected_activations.append(output.detach().cpu())
            return output

        # Hook the same layer we're intervening on
        target_layer = self.wrapper.model.decoder.net.attn_layers.layers[
            layer_idx * 2 + 1
        ]
        if len(target_layer) >= 3:
            residual_module = target_layer[2]  # Residual connection output
            hook = residual_module.register_forward_hook(activation_hook)
            self.activation_hooks.append(hook)

        print(f"🔗 Set up activation collection on layer {layer_idx}")

    def compute_sae_activations(self, layer_activations: torch.Tensor) -> torch.Tensor:
        """
        Compute SAE feature activations from layer activations

        Args:
            layer_activations: [batch, seq, hidden_dim] or [seq, hidden_dim]

        Returns:
            sae_activations: [batch, seq, n_features] or [seq, n_features]
        """
        original_shape = layer_activations.shape

        # Flatten to [batch*seq, hidden_dim] if needed
        if len(original_shape) == 3:
            batch_size, seq_len, hidden_dim = original_shape
            layer_activations = layer_activations.view(-1, hidden_dim)
        elif len(original_shape) == 2:
            batch_size, seq_len = 1, original_shape[0]
        else:
            raise ValueError(f"Unexpected activation shape: {original_shape}")

        # Move to device
        layer_activations = layer_activations.to(self.device)
        encoder_weight = self.sae_encoder_weight.to(self.device)
        encoder_bias = self.sae_encoder_bias.to(self.device)

        # Apply SAE encoder: activations = ReLU(input @ encoder.T + bias)
        sae_features = F.linear(layer_activations, encoder_weight, encoder_bias)
        sae_features = F.relu(sae_features)  # SAE typically uses ReLU

        # Reshape back to original structure
        if len(original_shape) == 3:
            sae_features = sae_features.view(batch_size, seq_len, -1)

        return sae_features

    def validate_intervention(
        self,
        start_tokens: torch.Tensor,
        feature_id: int,
        intervention_strengths: List[float] = [0.0, 0.5, 1.0, 2.0],
        seq_len: int = 128,
        layer_idx: int = 3,
    ) -> Dict:
        """
        Validate that intervention actually affects the target feature

        Returns detailed measurements of feature activations
        """

        results = {
            "feature_id": feature_id,
            "feature_name": self.wrapper.get_available_features().get(
                feature_id, f"Feature_{feature_id}"
            ),
            "measurements": {},
            "summary": {},
        }

        print(f"\n🔬 Validating intervention on Feature {feature_id}")
        print(f"   {results['feature_name'][:80]}...")

        # Test different intervention strengths
        for strength in intervention_strengths:
            print(f"   Testing strength {strength}...")

            strength_results = {"promote": {}, "suppress": {}}

            # Test promotion and suppression
            for intervention_type in ["promote", "suppress"]:
                # Set up activation collection
                self.setup_activation_collection(layer_idx)

                try:
                    if strength == 0.0:
                        # Baseline - no intervention
                        with torch.no_grad():
                            output = self.wrapper.model.generate(start_tokens, seq_len)
                    else:
                        # Generate with intervention
                        output = self.wrapper.generate_with_feature_intervention(
                            start_tokens=start_tokens,
                            seq_len=seq_len,
                            feature_id=feature_id,
                            intervention_strength=strength,
                            intervention_type=intervention_type,
                            layer_idx=layer_idx,
                        )

                    # Process collected activations
                    if self.collected_activations:
                        # Concatenate all activations from generation
                        all_activations = torch.cat(
                            self.collected_activations, dim=1
                        )  # [batch, total_seq, hidden]

                        # Compute SAE feature activations
                        sae_activations = self.compute_sae_activations(
                            all_activations
                        )  # [batch, total_seq, n_features]

                        # Extract target feature activations
                        target_feature_acts = (
                            sae_activations[0, :, feature_id].cpu().numpy()
                        )  # [seq]

                        # Compute statistics
                        stats = {
                            "mean_activation": float(np.mean(target_feature_acts)),
                            "max_activation": float(np.max(target_feature_acts)),
                            "total_activation": float(np.sum(target_feature_acts)),
                            "active_positions": int(
                                np.sum(target_feature_acts > 0.1)
                            ),  # Positions with significant activation
                            "activation_frequency": float(
                                np.mean(target_feature_acts > 0.1)
                            ),
                            "raw_activations": target_feature_acts.tolist(),
                        }

                        strength_results[intervention_type] = {
                            "generated_output": output.cpu().numpy(),
                            "target_feature_stats": stats,
                            "all_sae_activations": sae_activations.cpu().numpy(),
                        }

                finally:
                    # Clean up hooks
                    self.cleanup_hooks()

            results["measurements"][strength] = strength_results

        # Analyze the results
        results["summary"] = self._analyze_intervention_effects(
            results["measurements"], feature_id
        )

        return results

    def _analyze_intervention_effects(
        self, measurements: Dict, feature_id: int
    ) -> Dict:
        """
        Analyze whether interventions had the expected effects
        """
        summary = {
            "intervention_successful": False,
            "baseline_activation": 0.0,
            "promotion_effects": {},
            "suppression_effects": {},
            "recommendations": [],
        }

        # Get baseline (strength 0.0)
        if 0.0 in measurements:
            baseline_stats = measurements[0.0]["promote"][
                "target_feature_stats"
            ]  # Same for both
            summary["baseline_activation"] = baseline_stats["mean_activation"]

        # Analyze promotion effects
        promotion_working = False
        for strength in [0.5, 1.0, 2.0]:
            if strength in measurements:
                promote_stats = measurements[strength]["promote"][
                    "target_feature_stats"
                ]
                baseline_mean = summary["baseline_activation"]
                promote_mean = promote_stats["mean_activation"]

                effect_size = (promote_mean - baseline_mean) / (baseline_mean + 1e-8)

                summary["promotion_effects"][strength] = {
                    "mean_activation": promote_mean,
                    "effect_size": effect_size,
                    "increase": promote_mean > baseline_mean,
                    "relative_increase": effect_size > 0.1,  # 10% increase threshold
                }

                if effect_size > 0.1:
                    promotion_working = True

        # Analyze suppression effects
        suppression_working = False
        for strength in [0.5, 1.0, 2.0]:
            if strength in measurements:
                suppress_stats = measurements[strength]["suppress"][
                    "target_feature_stats"
                ]
                baseline_mean = summary["baseline_activation"]
                suppress_mean = suppress_stats["mean_activation"]

                effect_size = (baseline_mean - suppress_mean) / (baseline_mean + 1e-8)

                summary["suppression_effects"][strength] = {
                    "mean_activation": suppress_mean,
                    "effect_size": effect_size,
                    "decrease": suppress_mean < baseline_mean,
                    "relative_decrease": effect_size > 0.1,  # 10% decrease threshold
                }

                if effect_size > 0.1:
                    suppression_working = True

        # Overall assessment
        summary["intervention_successful"] = promotion_working or suppression_working

        # Recommendations
        if not promotion_working:
            summary["recommendations"].append(
                "Promotion not working - try higher intervention strength or different layer"
            )
        if not suppression_working:
            summary["recommendations"].append(
                "Suppression not working - try higher intervention strength or different layer"
            )
        if summary["baseline_activation"] < 0.01:
            summary["recommendations"].append(
                "Very low baseline activation - feature may not be relevant for this input"
            )

        return summary

    def batch_validate_features(
        self,
        start_tokens: torch.Tensor,
        feature_ids: List[int],
        intervention_strengths: List[float] = [0.0, 1.0, 2.0],
        seq_len: int = 128,
        layer_idx: int = 3,
    ) -> Dict:
        """
        Validate multiple features in batch
        """
        results = {"validation_results": {}, "summary_statistics": {}}

        print(f"🧪 Batch validating {len(feature_ids)} features...")

        for i, feature_id in enumerate(feature_ids):
            print(f"\n[{i+1}/{len(feature_ids)}] Validating Feature {feature_id}")

            feature_results = self.validate_intervention(
                start_tokens=start_tokens,
                feature_id=feature_id,
                intervention_strengths=intervention_strengths,
                seq_len=seq_len,
                layer_idx=layer_idx,
            )

            results["validation_results"][feature_id] = feature_results

        # Compute summary statistics
        successful_features = 0
        promotion_success = 0
        suppression_success = 0

        for feature_id, feature_results in results["validation_results"].items():
            summary = feature_results["summary"]
            if summary["intervention_successful"]:
                successful_features += 1

            # Check promotion success
            for strength, effect in summary["promotion_effects"].items():
                if effect.get("relative_increase", False):
                    promotion_success += 1
                    break

            # Check suppression success
            for strength, effect in summary["suppression_effects"].items():
                if effect.get("relative_decrease", False):
                    suppression_success += 1
                    break

        results["summary_statistics"] = {
            "total_features_tested": len(feature_ids),
            "successful_interventions": successful_features,
            "success_rate": successful_features / len(feature_ids),
            "promotion_success_count": promotion_success,
            "suppression_success_count": suppression_success,
        }

        print(f"\n📊 Batch Validation Summary:")
        print(f"   Total features tested: {len(feature_ids)}")
        print(
            f"   Successful interventions: {successful_features}/{len(feature_ids)} ({successful_features/len(feature_ids)*100:.1f}%)"
        )
        print(f"   Promotion working: {promotion_success} features")
        print(f"   Suppression working: {suppression_success} features")

        return results

    def visualize_feature_activation_changes(
        self, validation_results: Dict, feature_id: int, save_path: Optional[str] = None
    ):
        """
        Create visualizations showing how feature activations change with intervention
        """
        try:
            import matplotlib.pyplot as plt

            if feature_id not in validation_results["validation_results"]:
                print(f"Feature {feature_id} not found in validation results")
                return

            feature_data = validation_results["validation_results"][feature_id]
            measurements = feature_data["measurements"]

            # Extract data for plotting
            strengths = []
            promote_means = []
            suppress_means = []

            for strength in sorted(measurements.keys()):
                if strength == 0.0:
                    baseline_mean = measurements[strength]["promote"][
                        "target_feature_stats"
                    ]["mean_activation"]
                    strengths.append(strength)
                    promote_means.append(baseline_mean)
                    suppress_means.append(baseline_mean)
                else:
                    strengths.append(strength)
                    promote_means.append(
                        measurements[strength]["promote"]["target_feature_stats"][
                            "mean_activation"
                        ]
                    )
                    suppress_means.append(
                        measurements[strength]["suppress"]["target_feature_stats"][
                            "mean_activation"
                        ]
                    )

            # Create plot
            plt.figure(figsize=(10, 6))
            plt.plot(
                strengths,
                promote_means,
                "o-",
                label="Promotion",
                color="green",
                linewidth=2,
            )
            plt.plot(
                strengths,
                suppress_means,
                "o-",
                label="Suppression",
                color="red",
                linewidth=2,
            )
            plt.axhline(
                y=baseline_mean,
                color="gray",
                linestyle="--",
                alpha=0.7,
                label="Baseline",
            )

            plt.xlabel("Intervention Strength")
            plt.ylabel("Mean Feature Activation")
            plt.title(
                f'Feature {feature_id} Activation vs Intervention Strength\n{feature_data["feature_name"][:60]}...'
            )
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                print(f"💾 Saved plot to {save_path}")
            else:
                plt.show()

        except ImportError:
            print("📊 matplotlib not available - skipping visualization")

    def save_validation_results(self, results: Dict, output_path: str):
        """
        Save validation results to file
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save main results (excluding large numpy arrays)
        save_data = {
            "summary_statistics": results.get("summary_statistics", {}),
            "feature_summaries": {},
        }

        for feature_id, feature_data in results.get("validation_results", {}).items():
            save_data["feature_summaries"][feature_id] = {
                "feature_name": feature_data["feature_name"],
                "summary": feature_data["summary"],
            }

        with open(output_dir / "validation_summary.json", "w") as f:
            json.dump(save_data, f, indent=2)

        # Save detailed results separately
        for feature_id, feature_data in results.get("validation_results", {}).items():
            feature_file = output_dir / f"feature_{feature_id}_detailed.json"

            # Remove large numpy arrays for JSON serialization
            simplified_data = {
                "feature_id": feature_data["feature_id"],
                "feature_name": feature_data["feature_name"],
                "summary": feature_data["summary"],
            }

            with open(feature_file, "w") as f:
                json.dump(simplified_data, f, indent=2)

        print(f"💾 Validation results saved to {output_dir}")

    def cleanup_hooks(self):
        """Remove all activation collection hooks"""
        for hook in self.activation_hooks:
            hook.remove()
        self.activation_hooks = []
        self.collected_activations = []


def run_intervention_validation():
    """
    Main function to run intervention validation
    """
    print("🔬 Starting Intervention Validation...")

    # Load models
    from single_feature_intervention import load_model_and_create_wrapper

    wrapper = load_model_and_create_wrapper(
        model_path="exp/sod/ape/checkpoints/best_model.pt",
        sae_path="exp/sod/ape/sae_models/sae_layer_2048d.pt",
        interpretations_path="layer_interpretations/enhanced_diversity_report.json",
        config_path="exp/sod/ape/train-args.json",
    )

    # Create validator
    validator = InterventionValidator(
        intervention_wrapper=wrapper,
        sae_model_path="exp/sod/ape/sae_models/sae_layer_2048d.pt",
    )

    # Test with specific features
    device = wrapper.device
    start_tokens = torch.randint(0, 4, (1, 32, 6), device=device)

    # Test key features from your interpretations
    test_features = [1327, 889]  # Triplet rhythms, Syncopated patterns

    # Run validation
    results = validator.batch_validate_features(
        start_tokens=start_tokens,
        feature_ids=test_features,
        intervention_strengths=[0.0, 0.5, 1.0, 2.0],
        seq_len=128,
        layer_idx=3,
    )

    # Save results
    validator.save_validation_results(results, "validation_results")

    # Create visualizations for each feature
    for feature_id in test_features:
        validator.visualize_feature_activation_changes(
            results,
            feature_id,
            save_path=f"validation_results/feature_{feature_id}_activation_plot.png",
        )

    print("✅ Validation complete!")
    return results


if __name__ == "__main__":
    results = run_intervention_validation()
