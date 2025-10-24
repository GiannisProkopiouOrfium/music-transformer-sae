#!/usr/bin/env python3
"""
Deterministic Evaluation of Contrasting Interventions

This script uses quantitative musical metrics to evaluate intervention success
without relying on LLMs. Uses MusPy and music21 for MIDI analysis.

Usage:
    # Single directory evaluation
    python evaluate_interventions_deterministic.py \
        --input-dir contrasting_intervention_results_5_1950/layer5 \
        --output-file evaluation_deterministic_layer5.json
    
    # Batch evaluation
    python evaluate_interventions_deterministic.py \
        --input-dirs contrasting_intervention_results_1_325/layer1 \
                     contrasting_intervention_results_3_182/layer3 \
        --output-file evaluation_deterministic_all.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

try:
    import muspy
except ImportError:
    print("Error: muspy package not installed. Run: pip install muspy")
    sys.exit(1)

try:
    import music21
except ImportError:
    print("Error: music21 package not installed. Run: pip install music21")
    sys.exit(1)


# Feature-specific metric definitions
FEATURE_METRICS = {
    "1_325": {  # rhythmic_displacement_syncopation
        "name": "rhythmic_displacement_syncopation",
        "primary_metrics": ["syncopation_score", "rhythmic_complexity"],
        "expected_direction": "increase",  # positive additions should increase
    },
    "1_256": {  # dynamic_contrast_accents
        "name": "dynamic_contrast_accents",
        "primary_metrics": [
            "velocity_variance",
            "velocity_range",
            "dynamic_transitions",
        ],
        "expected_direction": "increase",
    },
    "1_1323": {  # wide_pitch_range_texture
        "name": "wide_pitch_range_texture",
        "primary_metrics": ["pitch_range", "pitch_class_entropy"],
        "expected_direction": "increase",
    },
    "3_182": {  # steady_pulse_march_rhythm
        "name": "steady_pulse_march_rhythm",
        "primary_metrics": [
            "rhythmic_regularity",
            "tempo_stability",
            "metric_strength",
        ],
        "expected_direction": "increase",
    },
    "3_855": {  # dynamic_contrast_tension
        "name": "dynamic_contrast_tension",
        "primary_metrics": [
            "velocity_variance",
            "velocity_range",
            "dynamic_transitions",
        ],
        "expected_direction": "increase",
    },
    "3_997": {  # antiphonal_call_response
        "name": "antiphonal_call_response",
        "primary_metrics": ["part_independence", "rhythmic_offset_between_parts"],
        "expected_direction": "increase",
    },
    "5_471": {  # unison_doubling_octaves
        "name": "unison_doubling_octaves",
        "primary_metrics": ["octave_doubling_ratio", "simultaneity_density"],
        "expected_direction": "increase",
    },
    "5_904": {  # rhythmic_augmentation
        "name": "rhythmic_augmentation",
        "primary_metrics": ["average_note_duration", "duration_variance"],
        "expected_direction": "increase",
    },
    "5_1950": {  # dramatic_dynamic_swells
        "name": "dramatic_dynamic_swells",
        "primary_metrics": [
            "velocity_variance",
            "velocity_range",
            "dynamic_transitions",
        ],
        "expected_direction": "increase",
    },
}


class MIDIMetricsAnalyzer:
    """Extract quantitative metrics from MIDI files."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_midi(self, midi_path: Path) -> Dict:
        """Extract all relevant metrics from a MIDI file."""
        try:
            # Load with both MusPy and music21 for comprehensive analysis
            music_muspy = muspy.read_midi(str(midi_path))
            music_m21 = music21.converter.parse(str(midi_path))

            # Collect all notes from all tracks
            all_notes = []
            for track in music_muspy.tracks:
                all_notes.extend(track.notes)

            metrics = {}

            # Basic statistics
            metrics["n_notes"] = len(all_notes)
            metrics["duration"] = music_muspy.get_end_time()

            # Pitch-based metrics
            metrics.update(self._analyze_pitch(all_notes, music_m21))

            # Rhythm-based metrics
            metrics.update(self._analyze_rhythm(all_notes, music_muspy, music_m21))

            # Dynamics-based metrics
            metrics.update(self._analyze_dynamics(all_notes))

            # Multi-part metrics
            metrics.update(self._analyze_parts(all_notes, music_m21))

            return metrics

        except Exception as e:
            self.logger.error(f"Failed to analyze MIDI {midi_path}: {e}")
            return {"error": str(e)}

    def _analyze_pitch(self, notes, music_m21) -> Dict:
        """Analyze pitch-related metrics."""
        metrics = {}

        if len(notes) == 0:
            return {
                "pitch_range": 0,
                "pitch_mean": 0,
                "pitch_std": 0,
                "pitch_class_entropy": 0,
            }

        pitches = [note.pitch for note in notes]

        # Pitch range (in semitones)
        metrics["pitch_range"] = int(max(pitches) - min(pitches))
        metrics["pitch_mean"] = float(np.mean(pitches))
        metrics["pitch_std"] = float(np.std(pitches))

        # Pitch class entropy (12-tone distribution)
        pitch_classes = [p % 12 for p in pitches]
        pc_counts = np.bincount(pitch_classes, minlength=12)
        pc_probs = pc_counts / pc_counts.sum()
        pc_probs = pc_probs[pc_probs > 0]  # Remove zeros
        metrics["pitch_class_entropy"] = float(-np.sum(pc_probs * np.log2(pc_probs)))

        return metrics

    def _analyze_rhythm(self, notes, music_muspy, music_m21) -> Dict:
        """Analyze rhythm-related metrics."""
        metrics = {}

        if len(notes) == 0:
            return {
                "average_note_duration": 0,
                "duration_variance": 0,
                "rhythmic_complexity": 0,
                "syncopation_score": 0,
                "rhythmic_regularity": 0,
                "metric_strength": 0,
                "tempo_stability": 1.0,
            }

        # Note durations
        durations = [note.duration for note in notes]
        metrics["average_note_duration"] = float(np.mean(durations))
        metrics["duration_variance"] = float(np.var(durations))

        # Unique duration types (rhythmic complexity)
        unique_durations = len(set(durations))
        metrics["rhythmic_complexity"] = int(unique_durations)

        # Inter-onset intervals
        onsets = sorted([note.time for note in notes])
        if len(onsets) > 1:
            iois = np.diff(onsets)
            metrics["ioi_mean"] = float(np.mean(iois))
            metrics["ioi_std"] = float(np.std(iois))

            # Rhythmic regularity (inverse of IOI coefficient of variation)
            if metrics["ioi_mean"] > 0:
                cv = metrics["ioi_std"] / metrics["ioi_mean"]
                metrics["rhythmic_regularity"] = float(1.0 / (1.0 + cv))
            else:
                metrics["rhythmic_regularity"] = 0.0

            # Syncopation score (simplified: variance of onset positions within beat grid)
            beat_positions = [onset % music_muspy.resolution for onset in onsets]
            metrics["syncopation_score"] = float(
                np.std(beat_positions) if len(beat_positions) > 1 else 0
            )
        else:
            metrics["ioi_mean"] = 0.0
            metrics["ioi_std"] = 0.0
            metrics["rhythmic_regularity"] = 0.0
            metrics["syncopation_score"] = 0.0

        # Metric strength (using music21)
        try:
            score = music_m21.flat.notes
            if len(score) > 0:
                # Count notes on strong beats vs weak beats
                strong_beat_count = 0
                total_count = 0
                for note in score:
                    if hasattr(note, "beat"):
                        total_count += 1
                        # Beat 1 is typically strong
                        if note.beat == 1.0:
                            strong_beat_count += 1

                metrics["metric_strength"] = (
                    strong_beat_count / total_count if total_count > 0 else 0
                )
            else:
                metrics["metric_strength"] = 0
        except:
            metrics["metric_strength"] = 0

        # Tempo stability (check for tempo changes)
        try:
            tempos = music_m21.flat.getElementsByClass("MetronomeMark")
            if len(tempos) > 1:
                tempo_values = [t.number for t in tempos]
                metrics["tempo_stability"] = 1.0 - (
                    np.std(tempo_values) / np.mean(tempo_values)
                )
            else:
                metrics["tempo_stability"] = 1.0
        except:
            metrics["tempo_stability"] = 1.0

        return metrics

    def _analyze_dynamics(self, notes) -> Dict:
        """Analyze dynamics-related metrics."""
        metrics = {}

        if len(notes) == 0:
            return {
                "velocity_mean": 0,
                "velocity_std": 0,
                "velocity_range": 0,
                "velocity_variance": 0,
                "dynamic_transitions": 0,
            }

        velocities = [note.velocity for note in notes]

        metrics["velocity_mean"] = float(np.mean(velocities))
        metrics["velocity_std"] = float(np.std(velocities))
        metrics["velocity_range"] = int(max(velocities) - min(velocities))
        metrics["velocity_variance"] = float(np.var(velocities))

        # Count significant dynamic transitions (velocity changes > 20)
        if len(velocities) > 1:
            vel_diffs = np.abs(np.diff(velocities))
            metrics["dynamic_transitions"] = int(np.sum(vel_diffs > 20))
        else:
            metrics["dynamic_transitions"] = 0

        return metrics

    def _analyze_parts(self, notes, music_m21) -> Dict:
        """Analyze multi-part interaction metrics."""
        metrics = {}

        # Part independence (simplified: ratio of non-simultaneous to total notes)
        if len(notes) > 1:
            onsets = [note.time for note in notes]
            unique_onsets = len(set(onsets))
            metrics["part_independence"] = float(unique_onsets / len(onsets))

            # Simultaneity density (average notes per unique onset time)
            metrics["simultaneity_density"] = float(len(onsets) / unique_onsets)
        else:
            metrics["part_independence"] = 1.0
            metrics["simultaneity_density"] = 1.0

        # Octave doubling detection
        try:
            # Group notes by onset time
            onset_groups = {}
            for note in notes:
                onset = note.time
                if onset not in onset_groups:
                    onset_groups[onset] = []
                onset_groups[onset].append(note.pitch)

            # Check for octave relationships
            octave_doublings = 0
            total_simultaneities = 0
            for onset, pitches in onset_groups.items():
                if len(pitches) > 1:
                    total_simultaneities += 1
                    # Check if any pitches are octaves apart
                    for i, p1 in enumerate(pitches):
                        for p2 in pitches[i + 1 :]:
                            if abs(p1 - p2) % 12 == 0:  # Same pitch class
                                octave_doublings += 1
                                break

            metrics["octave_doubling_ratio"] = float(
                octave_doublings / total_simultaneities
                if total_simultaneities > 0
                else 0
            )
        except:
            metrics["octave_doubling_ratio"] = 0.0

        # Rhythmic offset between parts (measure of call-and-response)
        try:
            # Using music21 to analyze parts separately
            if len(music_m21.parts) > 1:
                part_onsets = []
                for part in music_m21.parts:
                    onsets = [float(n.offset) for n in part.flat.notes]
                    if onsets:
                        part_onsets.append(onsets)

                # Calculate offset correlation between parts
                if len(part_onsets) >= 2:
                    # Simplified: check if onset patterns are shifted
                    offsets_diff = []
                    for i in range(len(part_onsets) - 1):
                        if len(part_onsets[i]) > 0 and len(part_onsets[i + 1]) > 0:
                            # Measure minimum time distance between parts
                            min_dist = min(
                                [
                                    abs(o1 - o2)
                                    for o1 in part_onsets[i][:10]
                                    for o2 in part_onsets[i + 1][:10]
                                ]
                            )
                            offsets_diff.append(min_dist)

                    metrics["rhythmic_offset_between_parts"] = float(
                        np.mean(offsets_diff) if offsets_diff else 0
                    )
                else:
                    metrics["rhythmic_offset_between_parts"] = 0.0
            else:
                metrics["rhythmic_offset_between_parts"] = 0.0
        except:
            metrics["rhythmic_offset_between_parts"] = 0.0

        return metrics


class DeterministicEvaluator:
    """Evaluate interventions using deterministic metrics."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize evaluator."""
        self.analyzer = MIDIMetricsAnalyzer()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def get_feature_key(self, intervention_dir: Path) -> Optional[str]:
        """Extract layer_feature key from directory path."""
        parts = intervention_dir.parts

        layer = None
        feature = None

        for part in parts:
            if part.startswith("layer"):
                layer = part.replace("layer", "")
            elif part.startswith("feature"):
                feature = part.split("_")[0].replace("feature", "")

        if layer and feature:
            return f"{layer}_{feature}"
        return None

    def compare_metrics(
        self, baseline_metrics: Dict, intervention_metrics: Dict, feature_key: str
    ) -> Dict:
        """Compare intervention metrics against baseline."""

        if "error" in baseline_metrics or "error" in intervention_metrics:
            return {"error": "Failed to extract metrics"}

        feature_config = FEATURE_METRICS.get(feature_key, {})
        primary_metrics = feature_config.get("primary_metrics", [])
        expected_direction = feature_config.get("expected_direction", "increase")

        comparison = {
            "feature_key": feature_key,
            "feature_name": feature_config.get("name", "unknown"),
            "primary_metrics": primary_metrics,
            "metric_changes": {},
            "metric_values": {
                "baseline": {},
                "intervention": {},
            },
        }

        # Calculate changes for all metrics
        for metric_name in baseline_metrics:
            if metric_name == "error" or not isinstance(
                baseline_metrics[metric_name], (int, float)
            ):
                continue

            baseline_val = baseline_metrics[metric_name]
            intervention_val = intervention_metrics.get(metric_name, 0)

            # Calculate relative change
            if baseline_val != 0:
                relative_change = (intervention_val - baseline_val) / abs(baseline_val)
            else:
                relative_change = 1.0 if intervention_val > 0 else 0.0

            comparison["metric_changes"][metric_name] = {
                "baseline": float(baseline_val),
                "intervention": float(intervention_val),
                "absolute_change": float(intervention_val - baseline_val),
                "relative_change": float(relative_change),
            }

            comparison["metric_values"]["baseline"][metric_name] = float(baseline_val)
            comparison["metric_values"]["intervention"][metric_name] = float(
                intervention_val
            )

        # Aggregate primary metrics
        if primary_metrics:
            changes = []
            for metric in primary_metrics:
                if metric in comparison["metric_changes"]:
                    changes.append(
                        comparison["metric_changes"][metric]["relative_change"]
                    )

            if changes:
                comparison["primary_metric_avg_change"] = float(np.mean(changes))
                comparison["primary_metric_direction"] = (
                    "increase" if np.mean(changes) > 0 else "decrease"
                )
            else:
                comparison["primary_metric_avg_change"] = 0.0
                comparison["primary_metric_direction"] = "unchanged"

        return comparison

    def evaluate_intervention_set(self, intervention_dir: Path) -> Dict:
        """Evaluate all interventions in a directory."""

        self.logger.info(f"Evaluating intervention set: {intervention_dir}")

        # Find baseline MIDI
        baseline_files = list(intervention_dir.glob("*_baseline.mid"))
        if not baseline_files:
            self.logger.error(f"No baseline file found in {intervention_dir}")
            return {"error": "No baseline file found"}

        baseline_midi = baseline_files[0]
        song_name = baseline_midi.stem.replace("_baseline", "")

        # Get feature key
        feature_key = self.get_feature_key(intervention_dir)
        if not feature_key or feature_key not in FEATURE_METRICS:
            self.logger.warning(
                f"Unknown feature key: {feature_key}, using generic metrics"
            )

        # Analyze baseline
        self.logger.info(f"Analyzing baseline: {baseline_midi.name}")
        baseline_metrics = self.analyzer.analyze_midi(baseline_midi)

        # Find all intervention MIDIs
        intervention_midis = [
            f
            for f in intervention_dir.glob(f"{song_name}_*.mid")
            if "baseline" not in f.stem and "ablation" not in f.stem
        ]

        # Find ablation if exists
        ablation_files = list(intervention_dir.glob("*_ablation.mid"))
        ablation_midi = ablation_files[0] if ablation_files else None

        results = {
            "song_name": song_name,
            "baseline": baseline_midi.name,
            "baseline_metrics": baseline_metrics,
            "comparisons": [],
        }

        # Compare each intervention with baseline
        for intervention_midi in intervention_midis:
            # Extract strength from filename
            stem = intervention_midi.stem
            if "_add_" in stem:
                strength_str = stem.split("_add_")[-1]
                strength = float(strength_str)
            else:
                continue

            self.logger.info(f"Analyzing intervention: {intervention_midi.name}")
            intervention_metrics = self.analyzer.analyze_midi(intervention_midi)

            comparison = self.compare_metrics(
                baseline_metrics, intervention_metrics, feature_key
            )
            comparison["strength"] = strength
            comparison["intervention_file"] = intervention_midi.name

            # Determine success based on expected direction and strength sign
            if "primary_metric_avg_change" in comparison:
                change = comparison["primary_metric_avg_change"]

                # Positive strength should increase metrics
                # Negative strength should decrease metrics
                if strength > 0:
                    comparison["success"] = change > 0
                elif strength < 0:
                    comparison["success"] = change < 0
                else:
                    comparison["success"] = abs(change) < 0.01
            else:
                comparison["success"] = False

            results["comparisons"].append(comparison)

        # Compare ablation with baseline if exists
        if ablation_midi:
            self.logger.info(f"Analyzing ablation: {ablation_midi.name}")
            ablation_metrics = self.analyzer.analyze_midi(ablation_midi)

            comparison = self.compare_metrics(
                baseline_metrics, ablation_metrics, feature_key
            )
            comparison["strength"] = "ablation"
            comparison["intervention_file"] = ablation_midi.name

            # Ablation should decrease primary metrics
            if "primary_metric_avg_change" in comparison:
                change = comparison["primary_metric_avg_change"]
                comparison["success"] = change < 0
            else:
                comparison["success"] = False

            results["ablation_comparison"] = comparison

        return results

    def evaluate_directory(self, input_dir: Path) -> Dict:
        """Evaluate all intervention sets in a directory."""

        self.logger.info(f"Scanning directory: {input_dir}")

        # Find all feature directories
        feature_dirs = []
        for feature_dir in input_dir.glob("feature*"):
            if feature_dir.is_dir():
                for intervention_dir in feature_dir.glob("feature*_song*"):
                    if intervention_dir.is_dir():
                        feature_dirs.append(intervention_dir)

        self.logger.info(f"Found {len(feature_dirs)} intervention sets to evaluate")

        results = {
            "input_dir": str(input_dir),
            "timestamp": datetime.now().isoformat(),
            "total_sets": len(feature_dirs),
            "evaluations": [],
        }

        for i, intervention_dir in enumerate(feature_dirs, 1):
            self.logger.info(
                f"[{i}/{len(feature_dirs)}] Evaluating: {intervention_dir.name}"
            )
            evaluation = self.evaluate_intervention_set(intervention_dir)
            evaluation["intervention_dir"] = str(intervention_dir)
            results["evaluations"].append(evaluation)

        return results

    def calculate_accuracy(self, results: Dict) -> Dict:
        """Calculate accuracy metrics from evaluation results."""

        total_comparisons = 0
        correct_additions = 0
        correct_ablations = 0
        total_additions = 0
        total_ablations = 0

        # Handle both single directory and multiple directories
        evaluations_list = []
        if "directories" in results:
            for directory_result in results["directories"]:
                evaluations_list.extend(directory_result.get("evaluations", []))
        else:
            evaluations_list = results.get("evaluations", [])

        for evaluation in evaluations_list:
            # Check addition interventions
            for comparison in evaluation.get("comparisons", []):
                if "success" not in comparison:
                    continue

                total_comparisons += 1
                total_additions += 1

                if comparison["success"]:
                    correct_additions += 1

            # Check ablation
            ablation_comp = evaluation.get("ablation_comparison")
            if ablation_comp and "success" in ablation_comp:
                total_comparisons += 1
                total_ablations += 1

                if ablation_comp["success"]:
                    correct_ablations += 1

        accuracy_metrics = {
            "total_comparisons": total_comparisons,
            "total_additions": total_additions,
            "total_ablations": total_ablations,
            "correct_additions": correct_additions,
            "correct_ablations": correct_ablations,
            "addition_accuracy": (
                correct_additions / total_additions if total_additions > 0 else 0
            ),
            "ablation_accuracy": (
                correct_ablations / total_ablations if total_ablations > 0 else 0
            ),
            "overall_accuracy": (
                (correct_additions + correct_ablations) / total_comparisons
                if total_comparisons > 0
                else 0
            ),
        }

        return accuracy_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate contrasting interventions using deterministic metrics"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Input directory containing intervention results",
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        help="Multiple input directories for batch evaluation",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Output JSON file for evaluation results",
    )

    args = parser.parse_args()

    # Determine input directories
    input_dirs = []
    if args.input_dir:
        input_dirs.append(args.input_dir)
    if args.input_dirs:
        input_dirs.extend(args.input_dirs)

    if not input_dirs:
        print("Error: Must specify --input-dir or --input-dirs")
        sys.exit(1)

    # Validate directories exist
    for input_dir in input_dirs:
        if not input_dir.exists():
            print(f"Error: Directory not found: {input_dir}")
            sys.exit(1)

    # Create evaluator
    evaluator = DeterministicEvaluator()

    # Evaluate each directory
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "method": "deterministic_metrics",
        "directories": [],
    }

    for input_dir in input_dirs:
        print(f"\n{'='*80}")
        print(f"Evaluating: {input_dir}")
        print(f"{'='*80}\n")

        results = evaluator.evaluate_directory(input_dir)
        all_results["directories"].append(results)

    # Calculate overall accuracy
    all_results["accuracy_metrics"] = evaluator.calculate_accuracy(all_results)

    # Save results
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {args.output_file}")
    print("\nAccuracy Metrics:")
    print(
        f"  - Addition accuracy: {all_results['accuracy_metrics']['addition_accuracy']:.1%}"
    )
    print(
        f"  - Ablation accuracy: {all_results['accuracy_metrics']['ablation_accuracy']:.1%}"
    )
    print(
        f"  - Overall accuracy: {all_results['accuracy_metrics']['overall_accuracy']:.1%}"
    )
    print(
        f"  - Total comparisons: {all_results['accuracy_metrics']['total_comparisons']}"
    )


if __name__ == "__main__":
    main()
