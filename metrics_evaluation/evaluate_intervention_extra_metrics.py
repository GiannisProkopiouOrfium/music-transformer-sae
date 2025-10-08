#!/usr/bin/env python3
"""
Extended Metrics Evaluation for Feature Interventions

This script evaluates generated musical sequences using intervention-specific metrics:
- Musical coherence preservation
- Harmonic progression quality
- Rhythmic consistency
- Note density and range statistics

It extends the base metrics evaluation with more detailed musical analysis.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

import muspy
import numpy as np
import torch

# Add mmt to path for imports
sys.path.append(str(Path(__file__).parent / "mmt"))
import representation

# Import our benchmarks module
try:
    from benchmarks import (
        calculate_cohens_d,
        interpret_cohens_d,
    )

    BENCHMARKS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import benchmarks module: {e}")
    print("Running without Cohen's d calculations...")
    BENCHMARKS_AVAILABLE = False


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging for the evaluation."""
    log_file = output_dir / "extended_metrics_evaluation.log"

    logger = logging.getLogger("extended_metrics_evaluation")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def compute_note_density_metrics(music: muspy.Music) -> Dict[str, float]:
    """Compute note density and distribution metrics."""
    if not music.tracks:
        return {"note_density": 0.0, "notes_per_beat": 0.0, "total_notes": 0}

    total_notes = sum(len(track.notes) for track in music.tracks)
    duration_beats = (
        music.get_end_time() / music.resolution if music.resolution > 0 else 1
    )

    # Notes per beat
    notes_per_beat = total_notes / duration_beats if duration_beats > 0 else 0

    # Note density (notes per quarter note equivalent)
    note_density = (
        total_notes / (music.get_end_time() / music.resolution)
        if music.get_end_time() > 0
        else 0
    )

    return {
        "note_density": float(note_density),
        "notes_per_beat": float(notes_per_beat),
        "total_notes": int(total_notes),
        "duration_beats": float(duration_beats),
    }


def compute_pitch_range_metrics(music: muspy.Music) -> Dict[str, float]:
    """Compute pitch range and distribution metrics."""
    if not music.tracks:
        return {"pitch_range": 0.0, "mean_pitch": 0.0, "pitch_std": 0.0}

    all_pitches = []
    for track in music.tracks:
        all_pitches.extend([note.pitch for note in track.notes])

    if not all_pitches:
        return {"pitch_range": 0.0, "mean_pitch": 0.0, "pitch_std": 0.0}

    pitches = np.array(all_pitches)

    return {
        "pitch_range": float(np.max(pitches) - np.min(pitches)),
        "mean_pitch": float(np.mean(pitches)),
        "pitch_std": float(np.std(pitches)),
        "min_pitch": int(np.min(pitches)),
        "max_pitch": int(np.max(pitches)),
    }


def compute_rhythmic_consistency_metrics(music: muspy.Music) -> Dict[str, float]:
    """Compute rhythmic consistency and regularity metrics."""
    if not music.tracks:
        return {"rhythmic_regularity": 0.0, "tempo_stability": 0.0}

    # Collect all note onsets
    all_onsets = []
    for track in music.tracks:
        all_onsets.extend([note.time for note in track.notes])

    if len(all_onsets) < 2:
        return {"rhythmic_regularity": 0.0, "tempo_stability": 0.0}

    all_onsets = sorted(all_onsets)

    # Compute inter-onset intervals
    intervals = np.diff(all_onsets)

    if len(intervals) == 0:
        return {"rhythmic_regularity": 0.0, "tempo_stability": 0.0}

    # Rhythmic regularity: inverse of coefficient of variation of intervals
    if np.mean(intervals) > 0:
        regularity = 1.0 / (1.0 + np.std(intervals) / np.mean(intervals))
    else:
        regularity = 0.0

    # Tempo stability: consistency of intervals
    tempo_stability = 1.0 - (np.std(intervals) / (np.mean(intervals) + 1e-8))
    tempo_stability = max(0.0, tempo_stability)  # Clamp to [0, 1]

    return {
        "rhythmic_regularity": float(regularity),
        "tempo_stability": float(tempo_stability),
        "mean_interval": float(np.mean(intervals)),
        "interval_std": float(np.std(intervals)),
    }


def compute_harmonic_progression_metrics(music: muspy.Music) -> Dict[str, float]:
    """Compute harmonic progression quality metrics."""
    if not music.tracks:
        return {"harmonic_consistency": 0.0, "chord_progression_quality": 0.0}

    try:
        # Use MusPy's built-in harmonic analysis where possible
        harmonic_consistency = muspy.scale_consistency(music)

        # Compute chord progression metrics if possible
        chord_progression_quality = 0.0

        # Simple harmonic quality based on consonance/dissonance
        all_pitches_by_time = defaultdict(list)
        for track in music.tracks:
            for note in track.notes:
                # Group notes by time windows (quarter note resolution)
                time_bucket = (note.time // music.resolution) * music.resolution
                all_pitches_by_time[time_bucket].append(
                    note.pitch % 12
                )  # Pitch classes

        # Compute consonance for each time bucket
        consonances = []
        for pitches in all_pitches_by_time.values():
            if len(pitches) > 1:
                unique_pitches = set(pitches)
                # Simple consonance measure: fewer unique pitch classes = more consonant
                consonance = (
                    1.0 - (len(unique_pitches) - 1) / 11.0
                )  # Normalize to [0, 1]
                consonances.append(max(0.0, consonance))

        if consonances:
            chord_progression_quality = float(np.mean(consonances))

        return {
            "harmonic_consistency": float(harmonic_consistency),
            "chord_progression_quality": float(chord_progression_quality),
            "consonance_variance": float(np.var(consonances)) if consonances else 0.0,
        }

    except Exception as e:
        return {
            "harmonic_consistency": 0.0,
            "chord_progression_quality": 0.0,
            "error": str(e),
        }


def compute_musical_coherence_metrics(music: muspy.Music) -> Dict[str, float]:
    """Compute overall musical coherence metrics."""
    if not music.tracks:
        return {"structural_coherence": 0.0, "melodic_coherence": 0.0}

    # Structural coherence: consistency of phrase lengths and patterns
    phrase_lengths = []
    for track in music.tracks:
        if len(track.notes) > 1:
            # Simple phrase detection based on rests
            current_phrase_length = 0
            last_end_time = 0

            for note in sorted(track.notes, key=lambda n: n.time):
                # If there's a gap, consider it a phrase boundary
                if note.time - last_end_time > music.resolution:  # Quarter note gap
                    if current_phrase_length > 0:
                        phrase_lengths.append(current_phrase_length)
                    current_phrase_length = 1
                else:
                    current_phrase_length += 1
                last_end_time = note.time + note.duration

            if current_phrase_length > 0:
                phrase_lengths.append(current_phrase_length)

    # Structural coherence: regularity of phrase lengths
    if len(phrase_lengths) > 1:
        structural_coherence = 1.0 / (
            1.0 + np.std(phrase_lengths) / np.mean(phrase_lengths)
        )
    else:
        structural_coherence = 0.5  # Neutral if not enough phrases

    # Melodic coherence: smoothness of melodic contour
    melodic_coherence_scores = []
    for track in music.tracks:
        if len(track.notes) > 2:
            pitches = [note.pitch for note in sorted(track.notes, key=lambda n: n.time)]
            intervals = np.abs(np.diff(pitches))
            # Melodic coherence: prefer smaller intervals (smoother motion)
            avg_interval = np.mean(intervals)
            coherence = 1.0 / (1.0 + avg_interval / 12.0)  # Normalize by octave
            melodic_coherence_scores.append(coherence)

    melodic_coherence = (
        float(np.mean(melodic_coherence_scores)) if melodic_coherence_scores else 0.0
    )

    return {
        "structural_coherence": float(structural_coherence),
        "melodic_coherence": float(melodic_coherence),
        "phrase_count": len(phrase_lengths),
        "mean_phrase_length": float(np.mean(phrase_lengths)) if phrase_lengths else 0.0,
    }


def evaluate_sequence_extended_metrics(
    sequence: np.ndarray, encoding: Dict
) -> Dict[str, Any]:
    """Evaluate a sequence using extended intervention-specific metrics."""
    try:
        # Convert to MusPy Music object
        music = representation.decode(sequence, encoding)

        # Trim the music to reasonable length for evaluation
        if music.resolution:
            music.trim(music.resolution * 64)

        # Check if music has tracks
        if not music.tracks:
            return {
                "error": "No tracks in generated music",
                "note_density_metrics": {},
                "pitch_range_metrics": {},
                "rhythmic_consistency_metrics": {},
                "harmonic_progression_metrics": {},
                "musical_coherence_metrics": {},
            }

        # Compute all metric categories
        result = {
            "note_density_metrics": compute_note_density_metrics(music),
            "pitch_range_metrics": compute_pitch_range_metrics(music),
            "rhythmic_consistency_metrics": compute_rhythmic_consistency_metrics(music),
            "harmonic_progression_metrics": compute_harmonic_progression_metrics(music),
            "musical_coherence_metrics": compute_musical_coherence_metrics(music),
        }

        return result

    except Exception as e:
        return {
            "error": f"Failed to decode sequence: {str(e)}",
            "note_density_metrics": {},
            "pitch_range_metrics": {},
            "rhythmic_consistency_metrics": {},
            "harmonic_progression_metrics": {},
            "musical_coherence_metrics": {},
        }


def load_base_metrics_results(base_results_file: Path) -> Dict:
    """Load base metrics results to extend."""
    with open(base_results_file, "r") as f:
        return json.load(f)


def extend_results_with_detailed_metrics(
    base_results: Dict, encoding: Dict, logger: logging.Logger
) -> Dict:
    """Extend base results with detailed intervention-specific metrics."""

    logger.info("🔍 Extending base results with detailed metrics...")

    extended_results = base_results.copy()
    extended_results["evaluation_info"]["metrics_evaluated"].extend(
        [
            "note_density_metrics",
            "pitch_range_metrics",
            "rhythmic_consistency_metrics",
            "harmonic_progression_metrics",
            "musical_coherence_metrics",
        ]
    )

    # Process each detailed result
    for i, result in enumerate(extended_results["detailed_results"]):
        layer = result["layer"]
        feature_id = result["feature_id"]
        feature_name = result["feature_name"]

        logger.info(
            f"📊 [{i+1}/{len(extended_results['detailed_results'])}] Processing Layer {layer}, Feature {feature_id}"
        )

        # Extend each condition with detailed metrics
        for condition_name, condition_data in result["conditions"].items():
            if "tensor_file" in condition_data:
                try:
                    # Load sequence
                    tensor_file = Path(condition_data["tensor_file"])
                    tensor_data = torch.load(tensor_file, map_location="cpu")

                    if "generated" in tensor_data:
                        sequence = tensor_data["generated"]
                    else:
                        # Find the sequence tensor
                        for key, value in tensor_data.items():
                            if (
                                isinstance(value, torch.Tensor)
                                and len(value.shape) >= 2
                            ):
                                sequence = value
                                break
                        else:
                            continue

                    # Convert to numpy and remove batch dimension if present
                    seq_np = sequence.numpy()
                    if len(seq_np.shape) == 3 and seq_np.shape[0] == 1:
                        seq_np = seq_np[0]

                    # Compute extended metrics
                    extended_metrics = evaluate_sequence_extended_metrics(
                        seq_np, encoding
                    )

                    # Add to condition data
                    condition_data["extended_metrics"] = extended_metrics

                    logger.info(f"  ✅ {condition_name}: Extended metrics computed")

                except Exception as e:
                    logger.error(
                        f"  ❌ {condition_name}: Failed to compute extended metrics: {e}"
                    )
                    condition_data["extended_metrics"] = {"error": str(e)}

    logger.info("✅ Extended metrics computation complete")
    return extended_results


def create_extended_summary(extended_results: Dict) -> Dict:
    """Create summary of extended metrics across all features and conditions."""

    summary = {
        "extended_metrics_summary": {},
        "condition_comparisons": {},
        "layer_comparisons": {},
    }

    # Collect all extended metrics
    all_extended_metrics = defaultdict(lambda: defaultdict(list))
    condition_extended_metrics = defaultdict(lambda: defaultdict(list))
    layer_extended_metrics = defaultdict(lambda: defaultdict(list))

    for result in extended_results["detailed_results"]:
        layer = result["layer"]

        for condition_name, condition_data in result["conditions"].items():
            if (
                "extended_metrics" in condition_data
                and "error" not in condition_data["extended_metrics"]
            ):
                extended_metrics = condition_data["extended_metrics"]

                # Process each metric category
                for category_name, category_metrics in extended_metrics.items():
                    if (
                        isinstance(category_metrics, dict)
                        and "error" not in category_metrics
                    ):
                        for metric_name, metric_value in category_metrics.items():
                            if not isinstance(metric_value, str) and not np.isnan(
                                metric_value
                            ):
                                full_metric_name = f"{category_name}.{metric_name}"
                                all_extended_metrics[full_metric_name]["all"].append(
                                    metric_value
                                )
                                condition_extended_metrics[condition_name][
                                    full_metric_name
                                ].append(metric_value)
                                layer_extended_metrics[layer][full_metric_name].append(
                                    metric_value
                                )

    # Compute overall extended metrics summary
    for metric_name, metric_data in all_extended_metrics.items():
        values = metric_data["all"]
        if values:
            summary["extended_metrics_summary"][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

    # Compute condition comparisons
    for condition_name, metrics_dict in condition_extended_metrics.items():
        summary["condition_comparisons"][condition_name] = {}
        for metric_name, values in metrics_dict.items():
            if values:
                summary["condition_comparisons"][condition_name][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "count": len(values),
                }

    # Compute layer comparisons
    for layer, metrics_dict in layer_extended_metrics.items():
        summary["layer_comparisons"][layer] = {}
        for metric_name, values in metrics_dict.items():
            if values:
                summary["layer_comparisons"][layer][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "count": len(values),
                }

    return summary


def _extract_baseline_metrics(conditions: Dict) -> Dict:
    """Extract baseline metrics from conditions."""
    baseline_values = {}
    if "baseline" in conditions and "extended_metrics" in conditions["baseline"]:
        baseline_metrics = conditions["baseline"]["extended_metrics"]

        for category_name, category_metrics in baseline_metrics.items():
            if isinstance(category_metrics, dict) and "error" not in category_metrics:
                for metric_name, metric_value in category_metrics.items():
                    if not isinstance(metric_value, str) and not np.isnan(metric_value):
                        full_metric_name = f"{category_name}.{metric_name}"
                        baseline_values[full_metric_name] = metric_value
    return baseline_values


def _compute_effect_size_for_condition(
    intervention_metrics: Dict, baseline_values: Dict
) -> Dict:
    """Compute effect sizes for a single intervention condition."""
    condition_effects = {}

    for category_name, category_metrics in intervention_metrics.items():
        if isinstance(category_metrics, dict) and "error" not in category_metrics:
            for metric_name, metric_value in category_metrics.items():
                if not isinstance(metric_value, str) and not np.isnan(metric_value):
                    full_metric_name = f"{category_name}.{metric_name}"

                    if full_metric_name in baseline_values:
                        baseline_val = baseline_values[full_metric_name]
                        intervention_val = metric_value

                        # Calculate Cohen's d
                        cohens_d = calculate_cohens_d(
                            [baseline_val], [intervention_val]
                        )

                        condition_effects[full_metric_name] = {
                            "baseline_value": baseline_val,
                            "intervention_value": intervention_val,
                            "cohens_d": cohens_d,
                            "effect_size_interpretation": interpret_cohens_d(cohens_d),
                            "percent_change": (
                                ((intervention_val - baseline_val) / baseline_val * 100)
                                if baseline_val != 0
                                else 0.0
                            ),
                        }

    return condition_effects


def add_intervention_effect_sizes(extended_results: Dict) -> Dict:
    """Add Cohen's d effect size calculations for interventions vs baseline."""

    if not BENCHMARKS_AVAILABLE:
        return extended_results

    # Add effect size analysis section
    extended_results["intervention_effect_sizes"] = {}

    # Process each feature
    for result in extended_results["detailed_results"]:
        layer = result["layer"]
        feature_id = result["feature_id"]
        feature_key = f"layer{layer}_feature{feature_id}"

        # Get baseline values
        baseline_values = _extract_baseline_metrics(result["conditions"])

        # Compare each intervention to baseline
        feature_effects = {}
        for condition_name, condition_data in result["conditions"].items():
            if condition_name == "baseline" or "extended_metrics" not in condition_data:
                continue

            condition_effects = _compute_effect_size_for_condition(
                condition_data["extended_metrics"], baseline_values
            )

            if condition_effects:
                feature_effects[condition_name] = condition_effects

        if feature_effects:
            extended_results["intervention_effect_sizes"][feature_key] = feature_effects

    return extended_results


def main():
    """Main extended evaluation function."""

    parser = argparse.ArgumentParser(
        description="Evaluate interventions using extended intervention-specific metrics"
    )
    parser.add_argument(
        "--base-results",
        type=Path,
        default="evaluation_results/base_metrics_evaluation_results.json",
        help="Path to base metrics evaluation results",
    )
    parser.add_argument(
        "--encoding-path",
        type=Path,
        default="data/sod/processed/notes/encoding.json",
        help="Path to encoding JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="evaluation_results",
        help="Output directory for evaluation results",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(args.output_dir)

    logger.info("🎼 STARTING EXTENDED METRICS EVALUATION FOR INTERVENTIONS")
    logger.info("=" * 70)
    logger.info(f"Base results file: {args.base_results}")
    logger.info(f"Encoding path: {args.encoding_path}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load base results
    logger.info("📄 Loading base evaluation results...")
    try:
        base_results = load_base_metrics_results(args.base_results)
        logger.info(f"✅ Loaded base results from {args.base_results}")
    except Exception as e:
        logger.error(f"❌ Failed to load base results: {e}")
        return False

    # Load encoding
    logger.info("📄 Loading encoding...")
    try:
        encoding = representation.load_encoding(args.encoding_path)
        logger.info(f"✅ Loaded encoding from {args.encoding_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load encoding: {e}")
        return False

    # Extend results with detailed metrics
    logger.info("🎯 COMPUTING EXTENDED METRICS")
    extended_results = extend_results_with_detailed_metrics(
        base_results, encoding, logger
    )

    # Create extended summary
    logger.info("📊 CREATING EXTENDED SUMMARY")
    extended_summary = create_extended_summary(extended_results)

    # Add intervention effect sizes
    logger.info("📊 COMPUTING INTERVENTION EFFECT SIZES")
    extended_results = add_intervention_effect_sizes(extended_results)

    # Combine with existing summary
    extended_results["overall_summary"].update(extended_summary)

    # Save extended results
    extended_results_file = args.output_dir / "extended_metrics_evaluation_results.json"
    with open(extended_results_file, "w") as f:
        json.dump(extended_results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("🎉 EXTENDED METRICS EVALUATION COMPLETE!")
    logger.info(f"📊 Extended results saved to: {extended_results_file}")

    logger.info("\n📈 EXTENDED METRICS SUMMARY:")
    for metric_name, stats in extended_summary["extended_metrics_summary"].items():
        logger.info(
            f"  {metric_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f} (n={stats['count']})"
        )

    logger.info(
        f"\n📁 Detailed logs: {args.output_dir}/extended_metrics_evaluation.log"
    )
    logger.info("🎵 Ready for comparative analysis!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
