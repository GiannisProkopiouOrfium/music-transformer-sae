#!/usr/bin/env python3
"""
SAE Feature Analysis Demo - No External APIs Required
===================================================

This script demonstrates:
1. Loading trained SAE and passing inputs through it
2. Analyzing and grouping feature activations
3. Creating interpretable descriptions of features
4. Scoring how well descriptions predict new activations

Run this first to see the framework, then set up OpenAI API for full interpretation.
"""

import json
import h5py
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")


class SparseAutoencoder(torch.nn.Module):
    """Sparse Autoencoder - same as training script."""

    def __init__(self, input_dim: int, hidden_dim: int, sparsity_coeff: float = 0.001):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coeff = sparsity_coeff

        self.encoder = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = torch.nn.Linear(hidden_dim, input_dim, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        hidden = self.relu(self.encoder(x))
        reconstructed = self.decoder(hidden)
        return reconstructed, hidden


class SAEAnalyzer:
    """Analyze SAE features and create interpretations."""

    def __init__(self, results_dir, json_dir=None):
        self.results_dir = Path(results_dir)
        self.json_dir = Path(json_dir) if json_dir else None
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        # Load model and data
        self.model = self._load_sae_model()
        self.activations = self._load_activations()
        self.sae_features = self._load_sae_features()
        self.musical_contexts = self._load_musical_contexts()

        # Analysis results
        self.feature_groups = {}
        self.group_interpretations = {}
        self.validation_scores = {}

    def _load_sae_model(self):
        """Load the trained SAE model."""
        model_path = self.results_dir / "sae_model.pt"
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )

        # Get dimensions from checkpoint
        input_dim = checkpoint["input_dim"]
        hidden_dim = checkpoint["hidden_dim"]
        sparsity_coeff = checkpoint.get("sparsity_coeff", 0.001)

        model = SparseAutoencoder(input_dim, hidden_dim, sparsity_coeff)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        print(f"Loaded SAE: {input_dim} → {hidden_dim} → {input_dim}")
        return model

    def _load_activations(self):
        """Load original layer activations."""
        with h5py.File(self.results_dir / "activations.h5", "r") as f:
            activations = f["activations"][:]
        print(f"Loaded {activations.shape[0]} activation samples")
        return activations

    def _load_sae_features(self):
        """Load SAE feature activations."""
        with h5py.File(self.results_dir / "analysis_results.h5", "r") as f:
            features = f["feature_activations"][:]
        print(f"Loaded SAE features: {features.shape}")
        return features

    def _load_musical_contexts(self):
        """Load musical contexts from JSON files."""
        if not self.json_dir or not self.json_dir.exists():
            print("No JSON directory provided - using synthetic data analysis")
            return None

        contexts = []
        json_files = list(self.json_dir.glob("**/*.json"))[: self.activations.shape[0]]

        print(f"Loading musical contexts from {len(json_files)} files...")

        for json_file in json_files:
            try:
                context = self._extract_musical_features(json_file)
                contexts.append(context)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                contexts.append(None)

        return contexts

    def _extract_musical_features(self, json_path):
        """Extract musical features from JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)

        features = {
            "instruments": [],
            "pitch_stats": {},
            "rhythm_stats": {},
            "harmony_stats": {},
            "tempo": None,
            "key": None,
        }

        # Extract tempo
        if "tempos" in data and data["tempos"]:
            features["tempo"] = data["tempos"][0]["qpm"]

        # Extract key
        if "key_signatures" in data and data["key_signatures"]:
            key = data["key_signatures"][0]
            features["key"] = (key["root"], key["mode"])

        # Analyze tracks
        all_pitches = []
        all_durations = []
        all_intervals = []

        for track in data.get("tracks", []):
            if track.get("is_drum", False):
                continue

            notes = track.get("notes", [])
            if len(notes) < 5:
                continue

            features["instruments"].append(track.get("program", 0))

            pitches = [note["pitch"] for note in notes]
            durations = [note["duration"] for note in notes]

            all_pitches.extend(pitches)
            all_durations.extend(durations)

            # Calculate intervals
            intervals = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]
            all_intervals.extend(intervals)

        # Compute statistics
        if all_pitches:
            features["pitch_stats"] = {
                "mean": np.mean(all_pitches),
                "std": np.std(all_pitches),
                "range": max(all_pitches) - min(all_pitches),
                "min": min(all_pitches),
                "max": max(all_pitches),
            }

        if all_durations:
            features["rhythm_stats"] = {
                "mean_duration": np.mean(all_durations),
                "duration_variety": len(set(all_durations)),
                "common_duration": Counter(all_durations).most_common(1)[0][0],
            }

        if all_intervals:
            features["harmony_stats"] = {
                "mean_interval": np.mean(np.abs(all_intervals)),
                "stepwise_ratio": sum(1 for x in all_intervals if abs(x) <= 2)
                / len(all_intervals),
                "common_intervals": [
                    x[0] for x in Counter(all_intervals).most_common(3)
                ],
            }

        return features

    def pass_new_inputs_through_sae(self, new_activations):
        """Pass new activations through the trained SAE."""
        print("Passing new inputs through SAE...")

        new_activations_tensor = torch.FloatTensor(new_activations).to(self.device)

        with torch.no_grad():
            reconstructed, sae_features = self.model(new_activations_tensor)

        return {
            "original": new_activations,
            "reconstructed": reconstructed.cpu().numpy(),
            "sae_features": sae_features.cpu().numpy(),
        }

    def group_features_by_similarity(self, n_groups=12, min_activation_freq=0.05):
        """Group features with similar activation patterns."""
        print(f"Grouping features into {n_groups} groups...")

        # Calculate feature statistics
        feature_stats = {}
        for i in range(self.sae_features.shape[1]):
            activations = self.sae_features[:, i]
            feature_stats[i] = {
                "activation_freq": np.mean(activations > 0),
                "mean_activation": np.mean(activations),
                "max_activation": np.max(activations),
            }

        # Filter features by activation frequency
        valid_features = [
            i
            for i, stats in feature_stats.items()
            if stats["activation_freq"] >= min_activation_freq
        ]

        print(
            f"Using {len(valid_features)} features (activation freq >= {min_activation_freq})"
        )

        if len(valid_features) < n_groups:
            n_groups = len(valid_features)

        # Prepare feature matrix for clustering
        feature_matrix = self.sae_features[:, valid_features].T  # [features, samples]
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        # Cluster features
        kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix_scaled)

        # Create feature groups
        for i, feature_idx in enumerate(valid_features):
            cluster_id = cluster_labels[i]

            if cluster_id not in self.feature_groups:
                self.feature_groups[cluster_id] = {
                    "features": [],
                    "representative": None,
                    "stats": {},
                }

            self.feature_groups[cluster_id]["features"].append(feature_idx)

        # Find representative feature for each group
        for cluster_id, group in self.feature_groups.items():
            # Choose feature with highest combined activation strength and frequency
            best_feature = max(
                group["features"],
                key=lambda f: feature_stats[f]["mean_activation"]
                * feature_stats[f]["activation_freq"],
            )

            group["representative"] = best_feature

            # Compute group statistics
            group_activations = np.mean(self.sae_features[:, group["features"]], axis=1)
            group["stats"] = {
                "size": len(group["features"]),
                "activation_freq": float(np.mean(group_activations > 0)),
                "mean_activation": float(np.mean(group_activations)),
                "top_samples": np.argsort(group_activations)[
                    -20:
                ].tolist(),  # Top 20 samples
            }

        print(f"Created {len(self.feature_groups)} feature groups")
        return self.feature_groups

    def create_musical_interpretations(self):
        """Create interpretations based on musical context analysis."""
        print("Creating musical interpretations for feature groups...")

        for group_id, group in self.feature_groups.items():
            interpretation = self._analyze_group_musical_patterns(group_id, group)
            self.group_interpretations[group_id] = interpretation

            print(f"\nGroup {group_id} ({group['stats']['size']} features):")
            print(f"  Activation frequency: {group['stats']['activation_freq']:.1%}")
            print(f"  Representative feature: {group['representative']}")
            print(f"  Interpretation: {interpretation['description']}")
            print(
                f"  Predicted features: {', '.join(interpretation['predictive_features'])}"
            )

    def _analyze_group_musical_patterns(self, group_id, group):
        """Analyze musical patterns for a feature group."""
        top_samples = group["stats"]["top_samples"]

        if self.musical_contexts is None:
            # Synthetic data analysis
            return {
                "description": f"Synthetic feature group {group_id} with {group['stats']['size']} features",
                "predictive_features": ["synthetic_pattern_1", "synthetic_pattern_2"],
                "confidence": 5,
                "data_type": "synthetic",
            }

        # Analyze musical contexts for top activating samples
        relevant_contexts = [
            self.musical_contexts[i]
            for i in top_samples
            if i < len(self.musical_contexts) and self.musical_contexts[i]
        ]

        if not relevant_contexts:
            return {
                "description": f"Feature group {group_id} - insufficient musical context",
                "predictive_features": ["unknown"],
                "confidence": 1,
                "data_type": "insufficient",
            }

        # Aggregate musical characteristics
        analysis = self._aggregate_musical_characteristics(relevant_contexts)

        # Generate interpretation based on patterns
        interpretation = self._generate_interpretation_from_analysis(group_id, analysis)

        return interpretation

    def _aggregate_musical_characteristics(self, contexts):
        """Aggregate musical characteristics across contexts."""
        aggregated = {
            "instruments": [],
            "tempos": [],
            "keys": [],
            "pitch_ranges": [],
            "stepwise_ratios": [],
            "common_intervals": [],
            "rhythm_patterns": [],
        }

        for ctx in contexts:
            if ctx.get("instruments"):
                aggregated["instruments"].extend(ctx["instruments"])

            if ctx.get("tempo"):
                aggregated["tempos"].append(ctx["tempo"])

            if ctx.get("key"):
                aggregated["keys"].append(ctx["key"])

            if ctx.get("pitch_stats"):
                ps = ctx["pitch_stats"]
                aggregated["pitch_ranges"].append((ps["min"], ps["max"]))

            if ctx.get("harmony_stats"):
                hs = ctx["harmony_stats"]
                aggregated["stepwise_ratios"].append(hs["stepwise_ratio"])
                aggregated["common_intervals"].extend(hs["common_intervals"])

            if ctx.get("rhythm_stats"):
                rs = ctx["rhythm_stats"]
                aggregated["rhythm_patterns"].append(rs["common_duration"])

        # Summarize patterns
        summary = {}

        if aggregated["instruments"]:
            summary["dominant_instrument"] = Counter(
                aggregated["instruments"]
            ).most_common(1)[0]

        if aggregated["tempos"]:
            summary["avg_tempo"] = np.mean(aggregated["tempos"])
            summary["tempo_category"] = self._categorize_tempo(summary["avg_tempo"])

        if aggregated["keys"]:
            summary["common_key"] = Counter(aggregated["keys"]).most_common(1)[0]

        if aggregated["stepwise_ratios"]:
            summary["avg_stepwise_ratio"] = np.mean(aggregated["stepwise_ratios"])
            summary["motion_type"] = (
                "stepwise" if summary["avg_stepwise_ratio"] > 0.6 else "leaping"
            )

        if aggregated["common_intervals"]:
            summary["frequent_intervals"] = Counter(
                aggregated["common_intervals"]
            ).most_common(3)

        if aggregated["pitch_ranges"]:
            ranges = aggregated["pitch_ranges"]
            summary["avg_range"] = np.mean([r[1] - r[0] for r in ranges])
            summary["range_category"] = (
                "narrow" if summary["avg_range"] < 24 else "wide"
            )

        return summary

    def _categorize_tempo(self, bpm):
        """Categorize tempo into musical terms."""
        if bpm < 60:
            return "very_slow"
        elif bpm < 80:
            return "slow"
        elif bpm < 120:
            return "moderate"
        elif bpm < 160:
            return "fast"
        else:
            return "very_fast"

    def _generate_interpretation_from_analysis(self, group_id, analysis):
        """Generate human-readable interpretation from analysis."""
        description_parts = []
        predictive_features = []

        # Build description based on patterns
        if "dominant_instrument" in analysis:
            instrument, count = analysis["dominant_instrument"]
            description_parts.append(
                f"primarily associated with MIDI instrument {instrument}"
            )
            predictive_features.append(f"instrument_{instrument}")

        if "motion_type" in analysis:
            motion = analysis["motion_type"]
            description_parts.append(f"featuring {motion} melodic motion")
            predictive_features.append(f"motion_{motion}")

        if "tempo_category" in analysis:
            tempo_cat = analysis["tempo_category"]
            description_parts.append(f"in {tempo_cat} tempo contexts")
            predictive_features.append(f"tempo_{tempo_cat}")

        if "range_category" in analysis:
            range_cat = analysis["range_category"]
            description_parts.append(f"with {range_cat} pitch ranges")
            predictive_features.append(f"range_{range_cat}")

        if "frequent_intervals" in analysis:
            intervals = analysis["frequent_intervals"]
            if intervals:
                top_interval = intervals[0][0]
                if top_interval == 0:
                    description_parts.append("emphasizing repeated notes")
                    predictive_features.append("repeated_notes")
                elif abs(top_interval) <= 2:
                    description_parts.append("with stepwise melodic patterns")
                    predictive_features.append("stepwise_motion")
                else:
                    description_parts.append("featuring larger intervallic leaps")
                    predictive_features.append("large_intervals")

        # Combine description
        if description_parts:
            description = f"Musical pattern {' '.join(description_parts)}"
        else:
            description = (
                f"Feature group {group_id} with unclear musical characteristics"
            )

        # Estimate confidence based on data quality
        confidence = min(10, len(description_parts) * 2 + 2)

        return {
            "description": description,
            "predictive_features": predictive_features,
            "confidence": confidence,
            "data_type": "real",
            "analysis_details": analysis,
        }

    def validate_interpretations_with_prediction(self, test_fraction=0.3):
        """Validate interpretations by predicting activations on test data."""
        print("Validating interpretations with prediction scoring...")

        n_samples = self.sae_features.shape[0]
        n_test = int(n_samples * test_fraction)

        # Create train/test split
        rng = np.random.default_rng(42)
        test_indices = rng.choice(n_samples, n_test, replace=False)
        train_indices = np.setdiff1d(np.arange(n_samples), test_indices)

        print(
            f"Using {len(train_indices)} samples for training, {len(test_indices)} for testing"
        )

        for group_id, group in self.feature_groups.items():
            print(f"\nValidating Group {group_id}...")

            # Get group activations
            group_features = group["features"]
            group_activations = np.mean(self.sae_features[:, group_features], axis=1)

            train_activations = group_activations[train_indices]
            test_activations = group_activations[test_indices]

            # Predict test activations based on musical features
            predicted_activations = self._predict_from_musical_features(
                test_indices, group_id
            )

            if predicted_activations is not None:
                # Calculate correlation
                correlation, p_value = pearsonr(test_activations, predicted_activations)
                r_squared = correlation**2

                self.validation_scores[group_id] = {
                    "correlation": float(correlation),
                    "p_value": float(p_value),
                    "r_squared": float(r_squared),
                    "prediction_accuracy": float(r_squared),
                    "test_samples": len(test_indices),
                    "mean_test_activation": float(np.mean(test_activations)),
                }

                print(f"  Correlation: {correlation:.3f} (p={p_value:.3f})")
                print(f"  R-squared: {r_squared:.3f}")
                print(f"  Prediction accuracy: {r_squared:.1%}")
            else:
                print(f"  Could not validate - insufficient data")
                self.validation_scores[group_id] = {"error": "insufficient_data"}

    def _predict_from_musical_features(self, test_indices, group_id):
        """Predict feature activations based on musical characteristics."""
        if self.musical_contexts is None:
            return None

        interpretation = self.group_interpretations.get(group_id)
        if not interpretation or interpretation["data_type"] != "real":
            return None

        predictions = []

        for idx in test_indices:
            if idx >= len(self.musical_contexts) or not self.musical_contexts[idx]:
                predictions.append(0.0)
                continue

            ctx = self.musical_contexts[idx]
            prediction_score = self._score_musical_context(ctx, interpretation)
            predictions.append(prediction_score)

        return np.array(predictions)

    def _score_musical_context(self, context, interpretation):
        """Score how well a musical context matches an interpretation."""
        score = 0.0
        max_score = 0.0

        predictive_features = interpretation["predictive_features"]

        for feature in predictive_features:
            max_score += 1.0

            if feature.startswith("instrument_"):
                target_instrument = int(feature.split("_")[1])
                if target_instrument in context.get("instruments", []):
                    score += 1.0

            elif feature.startswith("motion_"):
                motion_type = feature.split("_")[1]
                if context.get("harmony_stats"):
                    stepwise_ratio = context["harmony_stats"]["stepwise_ratio"]
                    if motion_type == "stepwise" and stepwise_ratio > 0.6:
                        score += 1.0
                    elif motion_type == "leaping" and stepwise_ratio <= 0.6:
                        score += 1.0

            elif feature.startswith("tempo_"):
                tempo_category = feature.split("_")[1]
                if context.get("tempo"):
                    actual_category = self._categorize_tempo(context["tempo"])
                    if actual_category == tempo_category:
                        score += 1.0

            elif feature.startswith("range_"):
                range_category = feature.split("_")[1]
                if context.get("pitch_stats"):
                    actual_range = context["pitch_stats"]["range"]
                    actual_category = "narrow" if actual_range < 24 else "wide"
                    if actual_category == range_category:
                        score += 1.0

            elif feature == "repeated_notes":
                if context.get("harmony_stats"):
                    common_intervals = context["harmony_stats"]["common_intervals"]
                    if 0 in common_intervals:
                        score += 1.0

            elif feature == "stepwise_motion":
                if context.get("harmony_stats"):
                    stepwise_ratio = context["harmony_stats"]["stepwise_ratio"]
                    if stepwise_ratio > 0.6:
                        score += 1.0

        return score / max_score if max_score > 0 else 0.0

    def generate_summary_report(self):
        """Generate comprehensive summary of analysis."""
        print("\n" + "=" * 60)
        print("SAE FEATURE ANALYSIS SUMMARY")
        print("=" * 60)

        print(
            f"Dataset: {self.sae_features.shape[0]} samples, {self.sae_features.shape[1]} features"
        )
        print(f"Feature groups: {len(self.feature_groups)}")
        print(f"Interpretations: {len(self.group_interpretations)}")
        print(f"Validated groups: {len(self.validation_scores)}")

        # Sort groups by validation score
        validated_groups = [
            (gid, score)
            for gid, score in self.validation_scores.items()
            if "correlation" in score
        ]
        validated_groups.sort(key=lambda x: x[1]["r_squared"], reverse=True)

        print(f"\nTOP VALIDATED INTERPRETATIONS:")
        print("-" * 40)

        for group_id, score in validated_groups[:5]:
            interpretation = self.group_interpretations[group_id]
            group = self.feature_groups[group_id]

            print(f"\nGroup {group_id} (R² = {score['r_squared']:.3f}):")
            print(f"  Size: {group['stats']['size']} features")
            print(f"  Description: {interpretation['description']}")
            print(f"  Prediction accuracy: {score['r_squared']:.1%}")
            print(f"  Confidence: {interpretation['confidence']}/10")

        return {
            "summary": {
                "n_samples": self.sae_features.shape[0],
                "n_features": self.sae_features.shape[1],
                "n_groups": len(self.feature_groups),
                "n_interpretations": len(self.group_interpretations),
                "n_validated": len(
                    [s for s in self.validation_scores.values() if "correlation" in s]
                ),
            },
            "feature_groups": self.feature_groups,
            "interpretations": self.group_interpretations,
            "validation_scores": self.validation_scores,
        }

    def save_results(self, output_path=None):
        """Save all results to file."""
        if output_path is None:
            output_path = self.results_dir / "sae_analysis_results.json"

        report = self.generate_summary_report()

        # Convert numpy types for JSON
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        report = convert_numpy(report)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nResults saved to: {output_path}")


def main():
    """Main execution function."""
    # Set paths
    results_dir = "results/layer3_experiment/quick_sae_test"
    json_dir = "data/sod/processed/json"  # Set to None if no JSON data

    print("SAE Feature Analysis Demo")
    print("=" * 50)

    # Initialize analyzer
    analyzer = SAEAnalyzer(results_dir, json_dir)

    # Demonstrate passing new inputs through SAE
    print("\n1. Testing SAE with new inputs...")
    test_indices = np.random.choice(analyzer.activations.shape[0], 100, replace=False)
    test_activations = analyzer.activations[test_indices]

    sae_results = analyzer.pass_new_inputs_through_sae(test_activations)
    print(f"   Processed {len(test_activations)} new samples")
    print(
        f"   Reconstruction MSE: {np.mean((sae_results['original'] - sae_results['reconstructed'])**2):.4f}"
    )

    # Group features
    print("\n2. Grouping similar features...")
    analyzer.group_features_by_similarity(n_groups=10, min_activation_freq=0.1)

    # Create interpretations
    print("\n3. Creating musical interpretations...")
    analyzer.create_musical_interpretations()

    # Validate interpretations
    print("\n4. Validating interpretations...")
    analyzer.validate_interpretations_with_prediction()

    # Generate and save report
    print("\n5. Generating final report...")
    analyzer.save_results()

    print("\nDemo completed! 🎵")
    print("\nTo use OpenAI for advanced interpretation:")
    print("1. Set OPENAI_API_KEY environment variable")
    print("2. Run: python sae_feature_interpreter.py")

    return analyzer


if __name__ == "__main__":
    analyzer = main()
