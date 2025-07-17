#!/usr/bin/env python3
"""
SAE Feature Interpreter with OpenAI Integration
==============================================

This script:
1. Loads a trained SAE model and passes inputs through it
2. Analyzes feature activations and groups similar features
3. Uses OpenAI API to interpret what each feature/group represents
4. Validates interpretations by predicting activations on new data
5. Provides confidence scores for each interpretation
"""

import os
import json
import h5py
import torch
import numpy as np
import openai
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Set OpenAI API key (you'll need to set this)
# openai.api_key = "your-api-key-here"  # Uncomment and add your key


class SparseAutoencoder(torch.nn.Module):
    """Sparse Autoencoder for feature analysis."""

    def __init__(self, input_dim: int, hidden_dim: int, sparsity_coeff: float = 0.001):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coeff = sparsity_coeff

        # Encoder
        self.encoder = torch.nn.Linear(input_dim, hidden_dim, bias=False)
        self.encoder_bias = torch.nn.Parameter(torch.zeros(hidden_dim))

        # Decoder
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=False)
        self.decoder_bias = torch.nn.Parameter(torch.zeros(input_dim))

        # ReLU activation
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # Encode
        hidden = self.relu(self.encoder(x) + self.encoder_bias)

        # Decode
        reconstructed = self.decoder(hidden) + self.decoder_bias

        return reconstructed, hidden


class SAEFeatureInterpreter:
    """Main class for SAE feature interpretation and validation."""

    def __init__(self, results_dir, json_dir=None, device="cpu"):
        """
        Initialize the interpreter.

        Args:
            results_dir: Path to SAE experiment results
            json_dir: Path to JSON MIDI files for musical context
            device: Device to run computations on
        """
        self.results_dir = Path(results_dir)
        self.json_dir = Path(json_dir) if json_dir else None
        self.device = torch.device(device)

        # Load SAE model and data
        self.model = self._load_sae_model()
        self.data = self._load_experiment_data()
        self.musical_contexts = self._load_musical_contexts() if json_dir else None

        # Feature analysis results
        self.feature_stats = {}
        self.feature_groups = {}
        self.interpretations = {}
        self.validation_scores = {}

    def _load_sae_model(self):
        """Load the trained SAE model."""
        model_path = self.results_dir / "sae_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"SAE model not found at {model_path}")

        # Load model state
        checkpoint = torch.load(model_path, map_location=self.device)

        # Determine model dimensions from checkpoint
        encoder_weight = checkpoint["encoder.weight"]
        hidden_dim, input_dim = encoder_weight.shape

        # Create and load model
        model = SparseAutoencoder(input_dim, hidden_dim)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()

        print(f"Loaded SAE model: {input_dim} → {hidden_dim} → {input_dim}")
        return model

    def _load_experiment_data(self):
        """Load activations and SAE features from experiment."""
        data = {}

        # Load original activations
        with h5py.File(self.results_dir / "activations.h5", "r") as f:
            data["activations"] = f["activations"][:]

        # Load SAE analysis results
        with h5py.File(self.results_dir / "analysis_results.h5", "r") as f:
            data["sae_features"] = f["feature_activations"][:]
            data["reconstructed"] = f["reconstructed_activations"][:]

            if "feature_activation_rates" in f:
                data["activation_rates"] = f["feature_activation_rates"][:]

        print(
            f"Loaded {data['activations'].shape[0]} samples with {data['sae_features'].shape[1]} features"
        )
        return data

    def _load_musical_contexts(self):
        """Load musical contexts from JSON files."""
        if not self.json_dir or not self.json_dir.exists():
            return None

        contexts = []
        json_files = list(self.json_dir.glob("**/*.json"))

        print(f"Loading musical contexts from {len(json_files)} files...")

        for i, json_file in enumerate(json_files[: self.data["activations"].shape[0]]):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                context = self._extract_musical_context(data)
                contexts.append(context)

                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1} files...")

            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                contexts.append(None)

        return contexts

    def _extract_musical_context(self, data):
        """Extract musical context from JSON data."""
        context = {
            "title": data.get("metadata", {}).get("title", "Unknown"),
            "instruments": [],
            "pitch_ranges": [],
            "intervals": [],
            "tempos": [],
            "keys": [],
            "durations": [],
        }

        # Extract tempo
        tempos = data.get("tempos", [])
        if tempos:
            context["tempos"] = [t["qpm"] for t in tempos]

        # Extract key signatures
        keys = data.get("key_signatures", [])
        if keys:
            context["keys"] = [(k["root"], k["mode"]) for k in keys]

        # Extract track information
        for track in data.get("tracks", []):
            if track.get("is_drum", False):
                continue

            notes = track.get("notes", [])
            if len(notes) < 5:
                continue

            context["instruments"].append(track.get("program", 0))

            pitches = [note["pitch"] for note in notes]
            context["pitch_ranges"].append((min(pitches), max(pitches)))

            durations = [note["duration"] for note in notes]
            context["durations"].extend(durations[:20])  # First 20 durations

            # Calculate intervals
            intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
            context["intervals"].extend(intervals[:20])  # First 20 intervals

        return context

    def analyze_feature_statistics(self):
        """Analyze statistics for all features."""
        print("Analyzing feature statistics...")

        sae_features = self.data["sae_features"]

        for feature_idx in range(sae_features.shape[1]):
            activations = sae_features[:, feature_idx]

            self.feature_stats[feature_idx] = {
                "mean_activation": float(np.mean(activations)),
                "max_activation": float(np.max(activations)),
                "std_activation": float(np.std(activations)),
                "activation_frequency": float(np.mean(activations > 0)),
                "sparsity": float(np.mean(activations == 0)),
                "top_activating_samples": np.argsort(activations)[-10:].tolist(),
                "activation_values": activations[
                    np.argsort(activations)[-10:]
                ].tolist(),
            }

        print(f"Computed statistics for {len(self.feature_stats)} features")

    def group_similar_features(self, n_groups=20, min_activation_freq=0.05):
        """Group features with similar activation patterns."""
        print(f"Grouping features into {n_groups} groups...")

        # Filter features by activation frequency
        valid_features = [
            idx
            for idx, stats in self.feature_stats.items()
            if stats["activation_frequency"] >= min_activation_freq
        ]

        if len(valid_features) < n_groups:
            n_groups = len(valid_features)

        print(
            f"Using {len(valid_features)} features with activation frequency >= {min_activation_freq}"
        )

        # Prepare feature activation matrix for clustering
        feature_matrix = self.data["sae_features"][
            :, valid_features
        ].T  # [features, samples]

        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix_scaled)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(feature_matrix_scaled, cluster_labels)
        print(f"Silhouette score: {silhouette_avg:.3f}")

        # Group features by cluster
        for i, feature_idx in enumerate(valid_features):
            cluster_id = cluster_labels[i]

            if cluster_id not in self.feature_groups:
                self.feature_groups[cluster_id] = {
                    "features": [],
                    "representative_feature": None,
                    "group_stats": {},
                }

            self.feature_groups[cluster_id]["features"].append(feature_idx)

        # Find representative feature for each group (highest mean activation)
        for cluster_id, group_info in self.feature_groups.items():
            features = group_info["features"]

            # Find feature with highest activation strength
            best_feature = max(
                features,
                key=lambda f: self.feature_stats[f]["mean_activation"]
                * self.feature_stats[f]["activation_frequency"],
            )

            group_info["representative_feature"] = best_feature

            # Compute group statistics
            group_activations = np.mean(self.data["sae_features"][:, features], axis=1)
            group_info["group_stats"] = {
                "size": len(features),
                "mean_activation": float(np.mean(group_activations)),
                "activation_frequency": float(np.mean(group_activations > 0)),
                "features_list": features,
            }

        print(f"Created {len(self.feature_groups)} feature groups")

        # Print group summary
        for cluster_id, group_info in sorted(self.feature_groups.items()):
            stats = group_info["group_stats"]
            rep_feature = group_info["representative_feature"]
            print(
                f"Group {cluster_id}: {stats['size']} features, "
                f"rep={rep_feature}, freq={stats['activation_frequency']:.3f}"
            )

    def get_musical_context_for_samples(self, sample_indices):
        """Get musical context for specific samples."""
        if not self.musical_contexts:
            return "No musical context available (using synthetic data)"

        contexts = []
        for idx in sample_indices:
            if idx < len(self.musical_contexts) and self.musical_contexts[idx]:
                contexts.append(self.musical_contexts[idx])

        if not contexts:
            return "No valid musical contexts found"

        # Aggregate contexts
        all_instruments = []
        all_pitch_ranges = []
        all_intervals = []
        all_tempos = []
        all_keys = []
        all_durations = []

        for ctx in contexts:
            all_instruments.extend(ctx.get("instruments", []))
            all_pitch_ranges.extend(ctx.get("pitch_ranges", []))
            all_intervals.extend(ctx.get("intervals", []))
            all_tempos.extend(ctx.get("tempos", []))
            all_keys.extend(ctx.get("keys", []))
            all_durations.extend(ctx.get("durations", []))

        # Summarize
        summary = {
            "common_instruments": Counter(all_instruments).most_common(3),
            "pitch_range": {
                "min": (
                    min([r[0] for r in all_pitch_ranges]) if all_pitch_ranges else None
                ),
                "max": (
                    max([r[1] for r in all_pitch_ranges]) if all_pitch_ranges else None
                ),
                "avg_min": (
                    np.mean([r[0] for r in all_pitch_ranges])
                    if all_pitch_ranges
                    else None
                ),
                "avg_max": (
                    np.mean([r[1] for r in all_pitch_ranges])
                    if all_pitch_ranges
                    else None
                ),
            },
            "common_intervals": Counter(all_intervals).most_common(5),
            "avg_tempo": np.mean(all_tempos) if all_tempos else None,
            "common_keys": Counter(all_keys).most_common(3),
            "common_durations": Counter(all_durations).most_common(5),
        }

        return summary

    def generate_interpretation_prompt(self, group_id):
        """Generate OpenAI prompt for feature group interpretation."""
        group_info = self.feature_groups[group_id]
        rep_feature = group_info["representative_feature"]
        group_stats = group_info["group_stats"]

        # Get top activating samples for the representative feature
        top_samples = self.feature_stats[rep_feature]["top_activating_samples"]

        # Get musical context
        musical_context = self.get_musical_context_for_samples(top_samples)

        prompt = f"""You are a music theory expert analyzing neural network features from a music transformer model. 

## Feature Group Analysis

**Group ID**: {group_id}
**Group Size**: {group_stats['size']} features  
**Representative Feature**: {rep_feature}
**Activation Frequency**: {group_stats['activation_frequency']:.1%}
**Mean Activation Strength**: {group_stats['mean_activation']:.4f}

## Musical Context of Highly Activating Samples

"""

        if isinstance(musical_context, dict):
            if musical_context.get("common_instruments"):
                instruments = musical_context["common_instruments"]
                prompt += f"**Most Common Instruments**: {[f'MIDI {inst[0]} (×{inst[1]})' for inst in instruments]}\n"

            if (
                musical_context.get("pitch_range")
                and musical_context["pitch_range"]["avg_min"]
            ):
                pr = musical_context["pitch_range"]
                prompt += f"**Pitch Range**: {pr['avg_min']:.0f}-{pr['avg_max']:.0f} MIDI (avg), {pr['min']}-{pr['max']} (absolute)\n"

            if musical_context.get("common_intervals"):
                intervals = musical_context["common_intervals"]
                prompt += f"**Common Intervals**: {[f'{inv[0]} semitones (×{inv[1]})' for inv in intervals[:3]]}\n"

            if musical_context.get("avg_tempo"):
                prompt += f"**Average Tempo**: {musical_context['avg_tempo']:.1f} BPM\n"

            if musical_context.get("common_keys"):
                keys = musical_context["common_keys"]
                prompt += f"**Common Keys**: {[f'Root {key[0][0]} {key[0][1]} (×{key[1]})' for key in keys[:2]]}\n"

            if musical_context.get("common_durations"):
                durations = musical_context["common_durations"]
                prompt += f"**Common Durations**: {[f'{dur[0]} ticks (×{dur[1]})' for dur in durations[:3]]}\n"
        else:
            prompt += f"{musical_context}\n"

        prompt += f"""
## Analysis Request

Based on this information, please provide:

1. **Feature Description**: What specific musical concept does this feature group represent? Be specific about:
   - Harmonic patterns (chord types, progressions, voice leading)
   - Melodic patterns (scales, motifs, contour)
   - Rhythmic patterns (specific note values, syncopation, meter)
   - Textural elements (monophony, polyphony, counterpoint)
   - Instrumental techniques or timbres

2. **Theoretical Explanation**: Why would a neural network learn to detect this pattern? What makes it musically significant or distinctive?

3. **Predictive Features**: What specific musical characteristics should reliably predict when this feature will activate strongly? List 3-5 concrete features.

4. **Feature Name**: Suggest a concise, descriptive name for this feature group.

5. **Confidence**: How confident are you in this interpretation? (1-10 scale, where 10 is very confident)

Please be specific and concise. Focus on concrete musical elements rather than vague descriptions.
"""

        return prompt

    def interpret_features_with_openai(self, api_key=None):
        """Use OpenAI API to interpret feature groups."""
        if api_key:
            openai.api_key = api_key
        elif not openai.api_key:
            print(
                "Warning: OpenAI API key not set. Please set openai.api_key or pass api_key parameter."
            )
            return

        print("Generating interpretations using OpenAI API...")

        for group_id in tqdm(self.feature_groups.keys(), desc="Interpreting groups"):
            try:
                prompt = self.generate_interpretation_prompt(group_id)

                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert music theorist and computational musicologist analyzing neural network features.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=800,
                    temperature=0.3,
                )

                interpretation = response.choices[0].message.content

                self.interpretations[group_id] = {
                    "prompt": prompt,
                    "interpretation": interpretation,
                    "model": "gpt-4",
                    "timestamp": str(np.datetime64("now")),
                }

                print(f"\nGroup {group_id} Interpretation:")
                print("-" * 50)
                print(interpretation)
                print("-" * 50)

            except Exception as e:
                print(f"Error interpreting group {group_id}: {e}")
                self.interpretations[group_id] = {
                    "error": str(e),
                    "timestamp": str(np.datetime64("now")),
                }

    def validate_interpretations(self, test_split=0.2):
        """Validate interpretations by predicting activations on held-out data."""
        print("Validating interpretations...")

        n_samples = self.data["sae_features"].shape[0]
        n_test = int(n_samples * test_split)

        # Split data
        test_indices = np.random.choice(n_samples, n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

        test_features = self.data["sae_features"][test_indices]
        test_contexts = [
            (
                self.musical_contexts[i]
                if self.musical_contexts and i < len(self.musical_contexts)
                else None
            )
            for i in test_indices
        ]

        for group_id, group_info in self.feature_groups.items():
            if (
                group_id not in self.interpretations
                or "error" in self.interpretations[group_id]
            ):
                continue

            # Get group activations
            group_features = group_info["features"]
            group_activations_test = np.mean(test_features[:, group_features], axis=1)

            # Calculate correlation between predicted and actual activations
            # For now, use simple heuristics - this could be made more sophisticated
            predicted_activations = self._predict_activations_from_context(
                test_contexts, group_id
            )

            if predicted_activations is not None:
                correlation, p_value = pearsonr(
                    group_activations_test, predicted_activations
                )

                self.validation_scores[group_id] = {
                    "correlation": float(correlation),
                    "p_value": float(p_value),
                    "test_samples": len(test_indices),
                    "mean_test_activation": float(np.mean(group_activations_test)),
                    "prediction_accuracy": float(correlation**2),  # R-squared
                }

                print(
                    f"Group {group_id}: Correlation = {correlation:.3f} (p={p_value:.3f})"
                )

    def _predict_activations_from_context(self, contexts, group_id):
        """Simple prediction based on musical context - this is a placeholder."""
        # This is a simplified prediction method
        # In practice, you'd want more sophisticated feature extraction

        if not contexts or not any(contexts):
            return None

        predictions = []

        for ctx in contexts:
            if ctx is None:
                predictions.append(0.0)
                continue

            # Simple heuristic prediction based on context
            score = 0.0

            # Example heuristics (these would be refined based on interpretations)
            if ctx.get("instruments"):
                # Higher score for string instruments
                if any(40 <= inst <= 51 for inst in ctx["instruments"]):
                    score += 0.3

            if ctx.get("intervals"):
                # Higher score for stepwise motion
                stepwise = sum(1 for interval in ctx["intervals"] if abs(interval) <= 2)
                score += (
                    (stepwise / len(ctx["intervals"])) * 0.4 if ctx["intervals"] else 0
                )

            if ctx.get("tempos"):
                # Moderate tempo preference
                avg_tempo = np.mean(ctx["tempos"])
                if 80 <= avg_tempo <= 120:
                    score += 0.3

            predictions.append(score)

        return np.array(predictions)

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        report = {
            "experiment_info": {
                "results_dir": str(self.results_dir),
                "n_samples": self.data["activations"].shape[0],
                "n_features": self.data["sae_features"].shape[1],
                "n_groups": len(self.feature_groups),
                "has_musical_context": self.musical_contexts is not None,
            },
            "feature_groups": {},
            "interpretations": self.interpretations,
            "validation_scores": self.validation_scores,
        }

        # Add feature group summaries
        for group_id, group_info in self.feature_groups.items():
            report["feature_groups"][group_id] = {
                "size": group_info["group_stats"]["size"],
                "representative_feature": group_info["representative_feature"],
                "activation_frequency": group_info["group_stats"][
                    "activation_frequency"
                ],
                "mean_activation": group_info["group_stats"]["mean_activation"],
                "features": group_info["features"],
                "has_interpretation": group_id in self.interpretations,
                "has_validation": group_id in self.validation_scores,
            }

        return report

    def save_results(self, output_path=None):
        """Save all results to JSON file."""
        if output_path is None:
            output_path = self.results_dir / "feature_interpretation_results.json"

        report = self.generate_summary_report()

        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        report = convert_numpy_types(report)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Results saved to {output_path}")

        # Also save human-readable summary
        summary_path = output_path.parent / "interpretation_summary.txt"
        self._save_human_readable_summary(summary_path)

    def _save_human_readable_summary(self, output_path):
        """Save human-readable summary of interpretations."""
        with open(output_path, "w") as f:
            f.write("SAE Feature Interpretation Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Dataset: {self.data['activations'].shape[0]} samples\n")
            f.write(f"Features: {self.data['sae_features'].shape[1]} total\n")
            f.write(f"Groups: {len(self.feature_groups)}\n")
            f.write(f"Interpreted: {len(self.interpretations)}\n")
            f.write(f"Validated: {len(self.validation_scores)}\n\n")

            # Sort groups by validation score
            sorted_groups = sorted(
                self.feature_groups.keys(),
                key=lambda g: self.validation_scores.get(g, {}).get("correlation", -1),
                reverse=True,
            )

            for group_id in sorted_groups:
                group_info = self.feature_groups[group_id]
                f.write(f"\nGroup {group_id}\n")
                f.write("-" * 20 + "\n")
                f.write(f"Size: {group_info['group_stats']['size']} features\n")
                f.write(
                    f"Representative: Feature {group_info['representative_feature']}\n"
                )
                f.write(
                    f"Activation Frequency: {group_info['group_stats']['activation_frequency']:.1%}\n"
                )

                if group_id in self.validation_scores:
                    score = self.validation_scores[group_id]
                    f.write(f"Validation Correlation: {score['correlation']:.3f}\n")
                    f.write(
                        f"Prediction Accuracy (R²): {score['prediction_accuracy']:.3f}\n"
                    )

                if group_id in self.interpretations:
                    f.write("Interpretation:\n")
                    f.write(self.interpretations[group_id]["interpretation"])
                    f.write("\n")

        print(f"Human-readable summary saved to {output_path}")


def main():
    """Main execution function."""
    # Configuration
    results_dir = "results/layer3_experiment/quick_sae_test"
    json_dir = "data/sod/processed/json"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Initialize interpreter
    print("Initializing SAE Feature Interpreter...")
    interpreter = SAEFeatureInterpreter(results_dir, json_dir, device)

    # Analyze features
    print("\n1. Analyzing feature statistics...")
    interpreter.analyze_feature_statistics()

    print("\n2. Grouping similar features...")
    interpreter.group_similar_features(n_groups=15, min_activation_freq=0.1)

    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nWarning: OPENAI_API_KEY not found in environment variables.")
        print("Please set your OpenAI API key to enable interpretation:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nSkipping OpenAI interpretation step...")
    else:
        print("\n3. Interpreting features with OpenAI...")
        interpreter.interpret_features_with_openai(api_key)

        print("\n4. Validating interpretations...")
        interpreter.validate_interpretations()

    # Save results
    print("\n5. Saving results...")
    interpreter.save_results()

    print("\nAnalysis complete!")
    return interpreter


if __name__ == "__main__":
    interpreter = main()
