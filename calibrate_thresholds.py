#!/usr/bin/env python3
"""
Threshold Calibration Script for Deterministic Analysis

Analyzes training data to calculate realistic metric thresholds based on
actual music distribution. This replaces arbitrary thresholds with data-driven ones.

Usage:
    python calibrate_thresholds.py --data-dir data/sod/processed/json --output calibrated_thresholds.json
    python calibrate_thresholds.py --data-dir data/sod/processed/json/Kunstderfuge --output calibrated_thresholds.json
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging
from tqdm import tqdm
import sys

# Add project modules to path
sys.path.insert(0, str(Path(__file__).parent))

from deterministic_analysis.midi_feature_extractors import MIDIFeatureExtractor
import muspy


class ThresholdCalibrator:
    """Calculate data-driven thresholds from training corpus."""

    def __init__(self, data_dir: str, max_files: Optional[int] = None):
        """
        Initialize calibrator.

        Args:
            data_dir: Directory containing MusPy JSON training files
            max_files: Maximum number of files to process (None = all)
        """
        self.data_dir = Path(data_dir)
        self.max_files = max_files
        self.midi_extractor = MIDIFeatureExtractor()

        # Storage for all extracted metrics
        self.all_metrics = []

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def find_json_files(self) -> List[Path]:
        """Find all MusPy JSON files in the data directory."""
        json_files = list(self.data_dir.rglob("*.json"))

        if self.max_files:
            json_files = json_files[: self.max_files]

        self.logger.info(f"Found {len(json_files)} JSON files in {self.data_dir}")
        return json_files

    def load_muspy_json(self, file_path: Path) -> Optional[muspy.Music]:
        """Load a MusPy JSON file."""
        try:
            # Use muspy.load() for JSON files
            music = muspy.load(str(file_path))
            return music

        except Exception as e:
            self.logger.warning(f"Failed to load {file_path}: {e}")
            return None

    def extract_metrics_from_corpus(self) -> Dict[str, List[float]]:
        """
        Extract all metrics from the training corpus.

        Returns:
            Dictionary mapping metric paths to lists of values
        """
        json_files = self.find_json_files()

        self.logger.info("Extracting metrics from training corpus...")

        # Dictionary to store metric values: {"category.metric": [values]}
        metric_values = {}

        successful_files = 0
        failed_files = 0

        for json_file in tqdm(json_files, desc="Processing files"):
            music = self.load_muspy_json(json_file)

            if music is None:
                failed_files += 1
                continue

            try:
                # Extract all features for this file
                features = self.midi_extractor.extract_all_features(music)

                # Flatten the nested dictionary and store values
                for category, metrics in features.items():
                    if isinstance(metrics, dict):
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)) and not np.isnan(value):
                                metric_key = f"{category}.{metric_name}"

                                if metric_key not in metric_values:
                                    metric_values[metric_key] = []

                                metric_values[metric_key].append(float(value))

                successful_files += 1

            except Exception as e:
                self.logger.warning(f"Failed to extract metrics from {json_file}: {e}")
                failed_files += 1

        self.logger.info(f"Successfully processed: {successful_files} files")
        self.logger.info(f"Failed to process: {failed_files} files")
        self.logger.info(f"Total unique metrics extracted: {len(metric_values)}")

        return metric_values

    def calculate_percentiles(
        self, metric_values: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate percentile-based thresholds for each metric.

        Args:
            metric_values: Dictionary mapping metric paths to value lists

        Returns:
            Dictionary with percentile thresholds for each metric
        """
        self.logger.info("Calculating percentile thresholds...")

        thresholds = {}

        for metric_key, values in tqdm(
            metric_values.items(), desc="Computing percentiles"
        ):
            if len(values) < 10:  # Skip metrics with too few samples
                self.logger.warning(
                    f"Skipping {metric_key}: only {len(values)} samples"
                )
                continue

            values_array = np.array(values)

            # Calculate percentiles
            percentiles = {
                "min": float(np.min(values_array)),
                "p5": float(np.percentile(values_array, 5)),
                "p10": float(np.percentile(values_array, 10)),
                "p25": float(np.percentile(values_array, 25)),
                "p50": float(np.percentile(values_array, 50)),  # median
                "p75": float(np.percentile(values_array, 75)),
                "p90": float(np.percentile(values_array, 90)),
                "p95": float(np.percentile(values_array, 95)),
                "max": float(np.max(values_array)),
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "count": len(values),
            }

            # Define quality ranges based on percentiles
            # "Good" range: 25th-75th percentile (middle 50% of data)
            # "Acceptable" range: 10th-90th percentile (middle 80% of data)
            # "Poor": Outside 10th-90th percentile

            quality_ranges = {
                "excellent": {"min": percentiles["p75"], "max": percentiles["max"]},
                "good": {"min": percentiles["p50"], "max": percentiles["p75"]},
                "moderate": {"min": percentiles["p25"], "max": percentiles["p50"]},
                "acceptable": {"min": percentiles["p10"], "max": percentiles["p90"]},
                "poor": {"min": percentiles["min"], "max": percentiles["p10"]},
            }

            thresholds[metric_key] = {
                "percentiles": percentiles,
                "quality_ranges": quality_ranges,
            }

        return thresholds

    def generate_feature_specific_thresholds(
        self, general_thresholds: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate feature-specific threshold recommendations based on the 9 features.

        This maps the general metric thresholds to the specific metrics used
        by each of the 9 interpretable features.
        """

        # Feature profiles mapping (using actual extracted metric names)
        feature_profiles = {
            "325": {
                "name": "dynamic_emphasis",
                "layer": 1,
                "primary_metrics": [
                    "note_patterns.average_velocity",
                    "velocity_dynamics.dynamic_range",
                    "velocity_dynamics.dynamic_variance",
                ],
            },
            "256": {
                "name": "note_timing",
                "layer": 1,
                "primary_metrics": [
                    "rhythmic_analysis.average_ioi",
                    "rhythmic_analysis.ioi_std",
                    "note_patterns.average_note_duration",
                ],
            },
            "1323": {
                "name": "note_density_control",
                "layer": 1,
                "primary_metrics": [
                    "note_patterns.note_density",
                    "note_patterns.notes_per_beat",
                    "musical_complexity.polyphonic_complexity",
                ],
            },
            "182": {
                "name": "rhythmic_pattern",
                "layer": 3,
                "primary_metrics": [
                    "rhythmic_analysis.rhythmic_regularity",
                    "rhythmic_analysis.syncopation_score",
                    "rhythmic_analysis.rhythmic_complexity",
                ],
            },
            "855": {
                "name": "phrase_structure",
                "layer": 3,
                "primary_metrics": [
                    "structural_analysis.phrase_count",
                    "structural_analysis.average_phrase_length",
                    "structural_analysis.structural_coherence",
                ],
            },
            "997": {
                "name": "melodic_contour",
                "layer": 3,
                "primary_metrics": [
                    "pitch_analysis.pitch_range",
                    "pitch_analysis.average_interval_size",
                    "pitch_analysis.step_motion_ratio",
                ],
            },
            "471": {
                "name": "harmonic_progression",
                "layer": 5,
                "primary_metrics": [
                    "harmonic_analysis.harmonic_rhythm",
                    "harmonic_analysis.chord_count",
                    "harmonic_analysis.scale_consistency",
                ],
            },
            "904": {
                "name": "musical_structure",
                "layer": 5,
                "primary_metrics": [
                    "structural_analysis.structural_coherence",
                    "structural_analysis.phrase_length_std",
                    "musical_complexity.overall_complexity_score",
                ],
            },
            "1950": {
                "name": "tonal_center",
                "layer": 5,
                "primary_metrics": [
                    "harmonic_analysis.pitch_class_entropy",
                    "harmonic_analysis.scale_consistency",
                    "pitch_analysis.pitch_std",
                ],
            },
        }

        feature_thresholds = {}

        for feature_id, profile in feature_profiles.items():
            feature_thresholds[feature_id] = {
                "name": profile["name"],
                "layer": profile["layer"],
                "metric_thresholds": {},
            }

            # Extract thresholds for each primary metric
            for metric_path in profile["primary_metrics"]:
                if metric_path in general_thresholds:
                    feature_thresholds[feature_id]["metric_thresholds"][metric_path] = (
                        general_thresholds[metric_path]
                    )
                else:
                    self.logger.warning(
                        f"Metric {metric_path} for feature {feature_id} not found in calibrated thresholds"
                    )

        return feature_thresholds

    def save_thresholds(self, thresholds: Dict[str, Any], output_path: str):
        """Save calibrated thresholds to JSON file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        calibration_data = {
            "metadata": {
                "data_directory": str(self.data_dir),
                "num_files_processed": (
                    len(self.all_metrics) if self.all_metrics else "N/A"
                ),
                "calibration_date": str(np.datetime64("now")),
                "description": "Data-driven metric thresholds calculated from training corpus",
            },
            "general_thresholds": thresholds["general"],
            "feature_specific_thresholds": thresholds["feature_specific"],
        }

        with open(output_file, "w") as f:
            json.dump(calibration_data, f, indent=2)

        self.logger.info(f"✅ Thresholds saved to: {output_file}")

    def print_summary(self, thresholds: Dict[str, Dict[str, Any]]):
        """Print a summary of calibrated thresholds."""
        print("\n" + "=" * 80)
        print("THRESHOLD CALIBRATION SUMMARY")
        print("=" * 80)

        general = thresholds["general"]
        print(f"\n📊 Total metrics calibrated: {len(general)}")

        # Show sample thresholds for key metrics
        key_metrics = [
            "note_patterns.average_velocity",
            "note_density.note_density",
            "pitch.pitch_range",
            "rhythm.rhythm_regularity",
            "harmony.harmonic_rhythm",
        ]

        print("\n🔑 Sample Key Metrics:")
        print("-" * 80)

        for metric in key_metrics:
            if metric in general:
                stats = general[metric]["percentiles"]
                print(f"\n{metric}:")
                print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
                print(f"  Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
                print(f"  Median (P50): {stats['p50']:.3f}")
                print(
                    f"  Good range (P25-P75): [{stats['p25']:.3f}, {stats['p75']:.3f}]"
                )
                print(f"  Samples: {stats['count']}")

        # Feature-specific summary
        feature_specific = thresholds["feature_specific"]
        print(f"\n📁 Feature-specific thresholds: {len(feature_specific)} features")

        for feature_id, data in feature_specific.items():
            print(f"\n  Feature {feature_id} ({data['name']}, Layer {data['layer']}):")
            print(f"    Calibrated metrics: {len(data['metric_thresholds'])}")
            for metric_path in data["metric_thresholds"].keys():
                print(f"      - {metric_path}")

        print("\n" + "=" * 80)

    def run_calibration(self, output_path: str) -> Dict[str, Any]:
        """
        Run the complete calibration pipeline.

        Args:
            output_path: Path to save calibrated thresholds

        Returns:
            Calibrated thresholds dictionary
        """
        print("🎵 Starting Threshold Calibration")
        print("=" * 80)

        # Step 1: Extract metrics from corpus
        metric_values = self.extract_metrics_from_corpus()

        if not metric_values:
            self.logger.error("No metrics extracted from corpus!")
            return None

        # Step 2: Calculate percentile thresholds
        general_thresholds = self.calculate_percentiles(metric_values)

        # Step 3: Generate feature-specific thresholds
        feature_specific_thresholds = self.generate_feature_specific_thresholds(
            general_thresholds
        )

        # Combine all thresholds
        all_thresholds = {
            "general": general_thresholds,
            "feature_specific": feature_specific_thresholds,
        }

        # Step 4: Save thresholds
        self.save_thresholds(all_thresholds, output_path)

        # Step 5: Print summary
        self.print_summary(all_thresholds)

        return all_thresholds


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Calibrate metric thresholds from training data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sod/processed/json",
        help="Directory containing MusPy JSON training files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="calibrated_thresholds.json",
        help="Output path for calibrated thresholds JSON",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process (default: all)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Use specific subset (e.g., 'Kunstderfuge' for data/sod/processed/json/Kunstderfuge)",
    )

    args = parser.parse_args()

    # Adjust data directory if subset specified
    data_dir = args.data_dir
    if args.subset:
        data_dir = str(Path(args.data_dir) / args.subset)

    # Check if directory exists
    if not Path(data_dir).exists():
        print(f"❌ Error: Directory not found: {data_dir}")
        sys.exit(1)

    # Initialize calibrator
    calibrator = ThresholdCalibrator(data_dir=data_dir, max_files=args.max_files)

    # Run calibration
    thresholds = calibrator.run_calibration(args.output)

    if thresholds:
        print(f"\n✅ Calibration complete!")
        print(f"📁 Thresholds saved to: {args.output}")
        print(f"\nNext steps:")
        print("1. Review the calibrated thresholds in the JSON file")
        print("2. Update your analysis to use these data-driven thresholds")
        print("3. Re-run deterministic analysis with calibrated thresholds")
    else:
        print("\n❌ Calibration failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
