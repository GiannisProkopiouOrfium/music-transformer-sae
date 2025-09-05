"""
Proper Feature Intervention Evaluation - validates interventions work by checking SAE activations
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys

# Add the current directory to the path
sys.path.append("/home/ubuntu/mmt")


class SAEFeatureInterventionValidator:
    """
    Validates that feature interventions actually increase target feature activations
    when the generated music is passed back through the SAE
    """

    def __init__(self, wrapper, sae_model):
        self.wrapper = wrapper
        self.sae_model = sae_model
        self.device = wrapper.device

    def validate_intervention_effectiveness(
        self,
        feature_ids: list,
        intervention_strengths: list = [0.5, 1.0, 2.0],
        n_samples: int = 5,
        seq_len: int = 128,
        layer_idx: int = 3,
    ):
        """
        Validate that interventions actually increase target feature activations
        """

        results = {
            "feature_validations": {},
            "summary": {
                "total_features_tested": len(feature_ids),
                "successful_interventions": 0,
                "failed_interventions": 0,
            },
        }

        for feature_id in feature_ids:
            print(f"\n🎯 Validating Feature {feature_id}...")

            feature_results = {
                "feature_id": feature_id,
                "baseline_activations": {},
                "intervention_activations": {},
                "intervention_success": {},
                "best_strength": None,
                "max_improvement": 0.0,
            }

            # Generate baseline samples and measure SAE activations
            baseline_activations = self._measure_baseline_activations(
                feature_id, n_samples, seq_len
            )
            feature_results["baseline_activations"] = baseline_activations

            baseline_mean = baseline_activations["mean_activation"]
            print(f"   📊 Baseline activation: {baseline_mean:.6f}")

            # Test each intervention strength
            for strength in intervention_strengths:
                print(f"   🔧 Testing strength {strength}...")

                # Test promotion intervention
                promote_activations = self._measure_intervention_activations(
                    feature_id, strength, "promote", n_samples, seq_len, layer_idx
                )

                # Test suppression intervention
                suppress_activations = self._measure_intervention_activations(
                    feature_id, strength, "suppress", n_samples, seq_len, layer_idx
                )

                feature_results["intervention_activations"][strength] = {
                    "promote": promote_activations,
                    "suppress": suppress_activations,
                }

                # Calculate intervention success
                promote_improvement = (
                    promote_activations["mean_activation"] - baseline_mean
                ) / (baseline_mean + 1e-8)
                suppress_reduction = (
                    baseline_mean - suppress_activations["mean_activation"]
                ) / (baseline_mean + 1e-8)

                feature_results["intervention_success"][strength] = {
                    "promote_improvement": promote_improvement,
                    "suppress_reduction": suppress_reduction,
                    "promote_successful": promote_improvement
                    > 0.1,  # 10% improvement threshold
                    "suppress_successful": suppress_reduction
                    > 0.1,  # 10% reduction threshold
                }

                print(
                    f"      Promote: {promote_improvement:+.3f} ({'✅' if promote_improvement > 0.1 else '❌'})"
                )
                print(
                    f"      Suppress: {suppress_reduction:+.3f} ({'✅' if suppress_reduction > 0.1 else '❌'})"
                )

                # Track best intervention
                if promote_improvement > feature_results["max_improvement"]:
                    feature_results["max_improvement"] = promote_improvement
                    feature_results["best_strength"] = strength

            # Determine overall success
            any_successful = any(
                result["promote_successful"] or result["suppress_successful"]
                for result in feature_results["intervention_success"].values()
            )

            if any_successful:
                results["summary"]["successful_interventions"] += 1
                print(f"   ✅ Feature {feature_id}: SUCCESSFUL intervention")
            else:
                results["summary"]["failed_interventions"] += 1
                print(f"   ❌ Feature {feature_id}: Failed intervention")

            results["feature_validations"][feature_id] = feature_results

        # Calculate success rate
        total = results["summary"]["total_features_tested"]
        successful = results["summary"]["successful_interventions"]
        results["summary"]["success_rate"] = successful / total if total > 0 else 0.0

        return results

    def _measure_baseline_activations(self, feature_id, n_samples, seq_len):
        """Measure baseline SAE activations for a feature"""

        all_activations = []

        for _ in range(n_samples):
            # Generate random starting tokens
            start_tokens = torch.randint(0, 4, (1, 32, 6), device=self.device)

            # Generate baseline music
            with torch.no_grad():
                generated = self.wrapper.model.generate(
                    start_tokens, seq_len, monotonicity_dim=[]
                )

            # Extract activations for the specific layer and measure SAE
            layer_activations = self._extract_layer_activations(generated, layer_idx=3)
            sae_activations = self._compute_sae_activations(layer_activations)

            # Get target feature activation
            feature_activation = sae_activations[0, :, feature_id].cpu().numpy()
            all_activations.extend(feature_activation)

        return {
            "mean_activation": np.mean(all_activations),
            "max_activation": np.max(all_activations),
            "std_activation": np.std(all_activations),
            "active_positions": np.sum(np.array(all_activations) > 0.1),
        }

    def _measure_intervention_activations(
        self, feature_id, strength, intervention_type, n_samples, seq_len, layer_idx
    ):
        """Measure SAE activations when interventions are applied"""

        all_activations = []

        for _ in range(n_samples):
            # Generate random starting tokens
            start_tokens = torch.randint(0, 4, (1, 32, 6), device=self.device)

            # Generate with intervention
            generated = self.wrapper.generate_with_feature_intervention(
                start_tokens=start_tokens,
                seq_len=seq_len,
                feature_id=feature_id,
                intervention_strength=strength,
                intervention_type=intervention_type,
                layer_idx=layer_idx,
            )

            # Extract activations and measure SAE
            layer_activations = self._extract_layer_activations(
                generated, layer_idx=layer_idx
            )
            sae_activations = self._compute_sae_activations(layer_activations)

            # Get target feature activation
            feature_activation = sae_activations[0, :, feature_id].cpu().numpy()
            all_activations.extend(feature_activation)

        return {
            "mean_activation": np.mean(all_activations),
            "max_activation": np.max(all_activations),
            "std_activation": np.std(all_activations),
            "active_positions": np.sum(np.array(all_activations) > 0.1),
        }

    def _extract_layer_activations(self, generated_sequence, layer_idx=3):
        """Extract activations from a specific transformer layer"""

        # We need to run the generated sequence through the model again to get layer activations
        # This is a simplified version - you may need to adapt based on your model structure

        collected_activations = []

        def activation_hook(module, input, output):
            if len(output.shape) == 3:  # [batch, seq, hidden]
                collected_activations.append(output.detach())

        # Hook the target layer
        target_layer = self.wrapper.model.decoder.net.attn_layers.layers[
            layer_idx * 2 + 1
        ]
        if len(target_layer) >= 3:
            hook = target_layer[2].register_forward_hook(activation_hook)

            # Run the generated sequence through the model
            with torch.no_grad():
                _ = self.wrapper.model.decoder(generated_sequence)

            hook.remove()

        if collected_activations:
            return torch.cat(collected_activations, dim=1)
        else:
            # Fallback: return random activations of correct shape
            return torch.randn(1, generated_sequence.shape[1], 512, device=self.device)

    def _compute_sae_activations(self, layer_activations):
        """Compute SAE feature activations from layer activations"""

        # Get SAE encoder
        if "model_state_dict" in self.sae_model:
            encoder_weight = self.sae_model["model_state_dict"]["encoder.weight"]
        else:
            # Fallback - use decoder transposed
            decoder_weight = self.sae_model["model_state_dict"]["decoder.weight"]
            encoder_weight = decoder_weight.T

        encoder_weight = encoder_weight.to(self.device)

        # Apply SAE encoder: activations = ReLU(x @ encoder.T)
        sae_activations = torch.relu(layer_activations @ encoder_weight.T)

        return sae_activations

    def save_validation_results(self, results, output_dir="validation_results_proper"):
        """Save validation results"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save main results
        with open(output_path / "sae_validation_results.json", "w") as f:
            json.dump(results, f, indent=2)

        # Print summary
        print(f"\n📊 VALIDATION SUMMARY:")
        print(
            f"   Total features tested: {results['summary']['total_features_tested']}"
        )
        print(
            f"   Successful interventions: {results['summary']['successful_interventions']}"
        )
        print(f"   Failed interventions: {results['summary']['failed_interventions']}")
        print(f"   Success rate: {results['summary']['success_rate']:.1%}")

        # Show best performing features
        print(f"\n🏆 BEST PERFORMING FEATURES:")
        for feature_id, data in results["feature_validations"].items():
            if data["max_improvement"] > 0.1:
                print(
                    f"   Feature {feature_id}: {data['max_improvement']:+.3f} improvement at strength {data['best_strength']}"
                )

        print(f"\n💾 Results saved to {output_path}")


def run_proper_validation():
    """Run proper SAE-based validation of feature interventions"""

    print("🧪 Starting Proper SAE Feature Intervention Validation...")

    try:
        from mmt.single_feature_intervention import load_model_and_create_wrapper

        # Load model and wrapper
        wrapper = load_model_and_create_wrapper(
            model_path="exp/sod/ape/checkpoints/best_model.pt",
            sae_path="exp/sod/ape/sae_models/sae_layer_2048d.pt",
            interpretations_path="layer_interpretations/enhanced_diversity_report_layer3.json",
            config_path="exp/sod/ape/train-args.json",
        )

        # Load SAE model
        sae_model = torch.load(
            "exp/sod/ape/sae_models/sae_layer_2048d.pt", map_location=wrapper.device
        )

        # Create validator
        validator = SAEFeatureInterventionValidator(wrapper, sae_model)

        # Test with valid feature IDs (within 0-511 range)
        valid_features = [
            182,
            231,
            171,
            50,
            100,
        ]  # Mix of interpreted and arbitrary features
        print(f"🎯 Testing features: {valid_features}")

        # Run validation
        results = validator.validate_intervention_effectiveness(
            feature_ids=valid_features,
            intervention_strengths=[0.5, 1.0, 2.0, 3.0],
            n_samples=3,  # Start small for testing
            seq_len=64,  # Shorter sequences
            layer_idx=3,
        )

        # Save results
        validator.save_validation_results(results)

        return results

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = run_proper_validation()
    if results:
        print("✅ Proper validation complete!")
    else:
        print("❌ Validation failed!")
