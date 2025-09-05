"""
Single Feature Intervention for Music Transformer
Based on individual SAE features rather than contrasting pairs
"""

import torch
import json
import numpy as np
from typing import Dict, List
from mmt import music_x_transformers, representation


class SingleFeatureInterventionWrapper:
    """
    Wrapper that can manipulate individual SAE features during inference
    Rather than using contrasting pairs, directly modulate specific features
    """

    def __init__(self, model, sae_model_path: str, feature_interpretations_path: str):
        self.model = model
        self.hooks = []
        self.device = next(model.parameters()).device

        # Load SAE model to get feature directions
        self.sae_model = torch.load(sae_model_path, map_location=self.device)

        # Extract decoder weights (these are the feature directions)
        if "model_state_dict" in self.sae_model:
            state_dict = self.sae_model["model_state_dict"]
            if "decoder.weight" in state_dict:
                # Shape: [hidden_dim, n_features]
                self.feature_directions = state_dict[
                    "decoder.weight"
                ].T  # [n_features, hidden_dim]
            else:
                raise ValueError(
                    f"Could not find decoder.weight in model_state_dict. Available keys: {list(state_dict.keys())}"
                )
        elif "decoder.weight" in self.sae_model:
            # Fallback for older format
            self.feature_directions = self.sae_model[
                "decoder.weight"
            ].T  # [n_features, hidden_dim]
        else:
            raise ValueError(
                f"Could not find decoder weights in SAE model. Available keys: {list(self.sae_model.keys())}"
            )

        # Load feature interpretations
        with open(feature_interpretations_path) as f:
            self.interpretations = json.load(f)

        print(
            f"🎵 Loaded {self.feature_directions.shape[0]} features with directions of dim {self.feature_directions.shape[1]}"
        )

    def get_available_features(self) -> Dict[int, str]:
        """Get all available features with their interpretations"""
        available = {}
        if "interpretations" in self.interpretations:
            for feature_id, data in self.interpretations["interpretations"].items():
                available[int(feature_id)] = data.get(
                    "interpretation", "No description"
                )
        return available

    def register_intervention_hook(
        self,
        layer_idx: int = 3,
        feature_id: int = None,
        intervention_strength: float = 1.0,
        intervention_type: str = "promote",
    ):
        """
        Register hook to intervene on a specific feature

        Args:
            layer_idx: Which transformer layer to intervene on (0-based)
            feature_id: Which SAE feature to manipulate
            intervention_strength: How strong the intervention (can be negative)
            intervention_type: "promote" (positive) or "suppress" (negative)
        """

        if feature_id is None or feature_id >= self.feature_directions.shape[0]:
            raise ValueError(f"Invalid feature_id {feature_id}")

        # Get the feature direction vector
        feature_direction = self.feature_directions[feature_id].to(
            self.device
        )  # [hidden_dim]

        # Determine intervention sign
        if intervention_type == "suppress":
            intervention_strength = -abs(intervention_strength)
        else:  # promote
            intervention_strength = abs(intervention_strength)

        def intervention_hook(module, input, output):
            """Hook function that modifies activations"""
            if len(output.shape) == 3:  # [batch, seq, hidden]
                batch_size, seq_len, hidden_dim = output.shape

                # Add the feature direction to all positions
                # Broadcasting: [hidden_dim] -> [1, 1, hidden_dim] -> [batch, seq, hidden_dim]
                intervention_vector = (
                    (intervention_strength * feature_direction)
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                output = output + intervention_vector.expand(batch_size, seq_len, -1)

            return output

        # Hook the residual stream at the specified layer
        # This targets the output of the FFN (after residual connection)
        target_layer = self.model.decoder.net.attn_layers.layers[
            layer_idx * 2 + 1
        ]  # FFN layer
        if len(target_layer) >= 3:  # Should have [PreNorm, FFN, Residual]
            residual_module = target_layer[2]  # Residual connection
            hook = residual_module.register_forward_hook(intervention_hook)
            self.hooks.append(hook)
            return hook
        else:
            raise ValueError(f"Unexpected layer structure at layer {layer_idx}")

    def generate_with_feature_intervention(
        self,
        start_tokens: torch.Tensor,
        seq_len: int,
        feature_id: int,
        intervention_strength: float = 1.0,
        intervention_type: str = "promote",
        layer_idx: int = 3,
        **generation_kwargs,
    ) -> torch.Tensor:
        """
        Generate music with intervention on a specific feature

        Args:
            start_tokens: Initial tokens to condition generation
            seq_len: How many tokens to generate
            feature_id: Which SAE feature to manipulate
            intervention_strength: Strength of intervention
            intervention_type: "promote" or "suppress"
            layer_idx: Which transformer layer to intervene on
        """

        # Register the intervention hook
        hook = self.register_intervention_hook(
            layer_idx=layer_idx,
            feature_id=feature_id,
            intervention_strength=intervention_strength,
            intervention_type=intervention_type,
        )

        try:
            # Generate with intervention
            with torch.no_grad():
                output = self.model.generate(
                    start_tokens=start_tokens,
                    seq_len=seq_len,
                    monotonicity_dim=[],
                    **generation_kwargs,
                )
            return output

        finally:
            # Always clean up the hook
            if hook:
                hook.remove()
            self.remove_hooks()

    def batch_feature_analysis(
        self,
        start_tokens: torch.Tensor,
        seq_len: int,
        feature_ids: List[int],
        intervention_strengths: List[float] = [0.5, 1.0, 2.0],
        n_samples: int = 5,
        layer_idx: int = 3,
    ) -> Dict:
        """
        Analyze effects of multiple features at different strengths
        """
        results = {"baseline": [], "interventions": {}}

        # Generate baseline samples
        print("🎼 Generating baseline samples...")
        for i in range(n_samples):
            baseline_output = self.model.generate(
                start_tokens, seq_len, monotonicity_dim=[]
            )
            results["baseline"].append(baseline_output.cpu().numpy())

        # Test each feature at each strength
        for feature_id in feature_ids:
            feature_name = self.get_available_features().get(
                feature_id, f"Feature_{feature_id}"
            )
            print(f"\n🎵 Testing Feature {feature_id}: {feature_name[:50]}...")

            results["interventions"][feature_id] = {
                "name": feature_name,
                "promote": {},
                "suppress": {},
            }

            for strength in intervention_strengths:
                print(f"  Testing strength {strength}...")

                # Test promotion
                promote_samples = []
                for i in range(n_samples):
                    output = self.generate_with_feature_intervention(
                        start_tokens=start_tokens,
                        seq_len=seq_len,
                        feature_id=feature_id,
                        intervention_strength=strength,
                        intervention_type="promote",
                        layer_idx=layer_idx,
                    )
                    promote_samples.append(output.cpu().numpy())

                results["interventions"][feature_id]["promote"][
                    strength
                ] = promote_samples

                # Test suppression
                suppress_samples = []
                for i in range(n_samples):
                    output = self.generate_with_feature_intervention(
                        start_tokens=start_tokens,
                        seq_len=seq_len,
                        feature_id=feature_id,
                        intervention_strength=strength,
                        intervention_type="suppress",
                        layer_idx=layer_idx,
                    )
                    suppress_samples.append(output.cpu().numpy())

                results["interventions"][feature_id]["suppress"][
                    strength
                ] = suppress_samples

        return results

    def remove_hooks(self):
        """Clean up all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def load_model_and_create_wrapper(
    model_path: str, sae_path: str, interpretations_path: str, config_path: str
) -> SingleFeatureInterventionWrapper:
    """Load the music transformer and create intervention wrapper"""

    # Load configuration
    with open(config_path) as f:
        train_args = json.load(f)

    # Load encoding
    encoding = representation.load_encoding("data/sod/processed/notes/encoding.json")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = music_x_transformers.MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rotary_pos_emb=train_args["rel_pos_emb"],
        use_abs_pos_emb=train_args["abs_pos_emb"],
    ).to(device)

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Create intervention wrapper
    wrapper = SingleFeatureInterventionWrapper(
        model=model,
        sae_model_path=sae_path,
        feature_interpretations_path=interpretations_path,
    )

    return wrapper


# Musical analysis functions for evaluation
def analyze_musical_properties(samples: np.ndarray, encoding) -> Dict:
    """
    Analyze musical properties of generated samples
    You can expand this based on your specific musical metrics
    """
    analysis = {
        "rhythmic_complexity": [],
        "pitch_variety": [],
        "dynamic_range": [],
        "sequence_lengths": [],
    }

    for sample in samples:
        # Basic analysis - you can expand this
        if len(sample.shape) == 3:  # [batch, seq, features]
            sample = sample[0]  # Take first in batch

        # Pitch variety (assuming pitch is in a specific dimension)
        pitch_dim = 3  # Based on your representation
        if sample.shape[1] > pitch_dim:
            pitches = sample[:, pitch_dim]
            unique_pitches = len(np.unique(pitches[pitches > 0]))  # Exclude padding
            analysis["pitch_variety"].append(unique_pitches)

        # Sequence length
        analysis["sequence_lengths"].append(len(sample))

        # Add more metrics as needed
        analysis["rhythmic_complexity"].append(calculate_rhythmic_complexity(sample))
        analysis["dynamic_range"].append(calculate_dynamic_range(sample))

    # Convert to summary statistics
    summary = {}
    for key, values in analysis.items():
        if values:
            summary[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

    return summary


def calculate_rhythmic_complexity(sample: np.ndarray) -> float:
    """Calculate a simple rhythmic complexity metric"""
    # This is a placeholder - implement based on your representation
    if len(sample.shape) == 2 and sample.shape[1] > 1:
        beat_dim = 1  # Assuming beat is dimension 1
        beats = sample[:, beat_dim]
        unique_beats = len(np.unique(beats[beats > 0]))
        return unique_beats / len(beats) if len(beats) > 0 else 0.0
    return 0.0


def calculate_dynamic_range(sample: np.ndarray) -> float:
    """Calculate dynamic range based on velocity if available"""
    # Placeholder - implement based on your representation
    return np.random.random()  # Replace with actual calculation


if __name__ == "__main__":
    # Example usage
    print("🎵 Loading model and creating intervention wrapper...")

    wrapper = load_model_and_create_wrapper(
        model_path="exp/sod/ape/checkpoints/best_model.pt",  # Adjust path
        sae_path="exp/sod/ape/sae_models/sae_layer_2048d.pt",
        interpretations_path="interpretation/enhanced_diversity/enhanced_diversity_report.json",
        config_path="exp/sod/ape/train-args.json",  # Adjust path
    )

    # Show available features
    features = wrapper.get_available_features()
    print(f"\n📋 Available features:")
    for fid, desc in list(features.items())[:5]:  # Show first 5
        print(f"  {fid}: {desc[:80]}...")

    # Example: Test triplet rhythm feature (ID 1327 from your data)
    device = wrapper.device
    start_tokens = torch.randint(0, 4, (1, 32, 6), device=device)

    print(f"\n🎼 Testing feature 1327 (triplet rhythms)...")

    # Generate with promotion
    promoted_output = wrapper.generate_with_feature_intervention(
        start_tokens=start_tokens,
        seq_len=256,
        feature_id=1327,
        intervention_strength=1.5,
        intervention_type="promote",
    )

    # Generate with suppression
    suppressed_output = wrapper.generate_with_feature_intervention(
        start_tokens=start_tokens,
        seq_len=256,
        feature_id=1327,
        intervention_strength=1.5,
        intervention_type="suppress",
    )

    print(
        "✅ Generation complete! Check the outputs for differences in triplet patterns."
    )
