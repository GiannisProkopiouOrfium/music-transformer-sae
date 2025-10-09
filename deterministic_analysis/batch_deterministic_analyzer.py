#!/usr/bin/env python3
"""
Batch Deterministic Analyzer

Orchestrates the complete deterministic analysis pipeline for all feature interventions.
Processes baseline + intervention pairs, extracts musical features, and provides comprehensive assessments.

This module handles:
- Batch processing of all 54 intervention results (9 features × 6 conditions)
- Integration with existing file formats (numpy, CSV, txt, muspy JSON)
- Musical quality assessment and intervention effectiveness scoring
- Decision-making support for intervention success
- Report generation and result aggregation
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import argparse
from datetime import datetime
import traceback

# Import our analysis modules
from midi_feature_extractors import MIDIFeatureExtractor
from feature_specific_analyzers import (
    FeatureSpecificAnalyzer,
    analyze_intervention_batch,
)

# Import existing MMT modules
import sys

sys.path.append(str(Path(__file__).parent.parent))

try:
    from mmt.baseline.representation_remi import represent_midi_with_remi
    from mmt.baseline.representation_mmm import represent_midi_with_mmm
    import muspy
except ImportError as e:
    print(f"Warning: Could not import MMT modules: {e}")
    print("Continuing with basic functionality...")


class BatchDeterministicAnalyzer:
    """Orchestrate complete deterministic analysis pipeline."""

    def __init__(self, output_dir: str = "deterministic_results"):
        """Initialize batch analyzer with configuration."""

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize analyzers
        self.midi_extractor = MIDIFeatureExtractor()
        self.feature_analyzer = FeatureSpecificAnalyzer()

        # Configure logging
        self.setup_logging()

        # Feature configuration - our 9 selected features
        self.features_config = {
            # Layer 1 - Early Processing (Note Level)
            "325": {"layer": 1, "name": "dynamic_emphasis", "type": "early"},
            "256": {"layer": 1, "name": "note_timing", "type": "early"},
            "1323": {"layer": 1, "name": "note_density_control", "type": "early"},
            # Layer 3 - Mid Processing (Phrase/Rhythm Level)
            "182": {"layer": 3, "name": "rhythmic_pattern", "type": "mid"},
            "855": {"layer": 3, "name": "phrase_structure", "type": "mid"},
            "997": {"layer": 3, "name": "melodic_contour", "type": "mid"},
            # Layer 5 - Late Processing (Structure/Harmony Level)
            "471": {"layer": 5, "name": "harmonic_progression", "type": "late"},
            "904": {"layer": 5, "name": "musical_structure", "type": "late"},
            "1950": {"layer": 5, "name": "tonal_center", "type": "late"},
        }

        # Intervention strengths typically used
        self.intervention_strengths = [-5.0, -2.0, -1.0, 1.0, 2.0, 5.0]

        self.logger.info(f"BatchDeterministicAnalyzer initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Configured for {len(self.features_config)} features")

    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "analysis.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        self.logger = logging.getLogger(__name__)

    def find_intervention_files(self, base_dir: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Find all intervention result files in the specified directory.

        Expected structure:
        - baseline files: *_baseline.*
        - intervention files: *_feature{ID}_strength{STRENGTH}.*

        Args:
            base_dir: Directory containing intervention results

        Returns:
            Dictionary mapping feature_id -> condition -> list of files
        """

        base_path = Path(base_dir)
        if not base_path.exists():
            self.logger.error(f"Base directory does not exist: {base_dir}")
            return {}

        files_by_feature = {}

        # Initialize structure for all features
        for feature_id in self.features_config.keys():
            files_by_feature[feature_id] = {"baseline": [], "interventions": {}}
            for strength in self.intervention_strengths:
                files_by_feature[feature_id]["interventions"][str(strength)] = []

        # Scan for files
        for file_path in base_path.rglob("*"):
            if file_path.is_file():
                filename = file_path.name

                # Check for baseline files
                if "baseline" in filename.lower():
                    # Add to all features (baseline is shared)
                    for feature_id in self.features_config.keys():
                        files_by_feature[feature_id]["baseline"].append(str(file_path))

                # Check for intervention files
                for feature_id in self.features_config.keys():
                    if (
                        f"feature{feature_id}" in filename
                        or f"feature_{feature_id}" in filename
                    ):
                        for strength in self.intervention_strengths:
                            strength_patterns = [
                                f"strength{strength}",
                                f"strength_{strength}",
                                f"str{strength}",
                                f"str_{strength}",
                                f"{strength:.1f}",
                                f"s{strength}",
                            ]

                            if any(
                                pattern in filename for pattern in strength_patterns
                            ):
                                files_by_feature[feature_id]["interventions"][
                                    str(strength)
                                ].append(str(file_path))
                                break

        # Log findings
        total_files = 0
        for feature_id, conditions in files_by_feature.items():
            baseline_count = len(conditions["baseline"])
            intervention_count = sum(
                len(files) for files in conditions["interventions"].values()
            )
            total_files += baseline_count + intervention_count

            self.logger.info(
                f"Feature {feature_id}: {baseline_count} baseline, {intervention_count} intervention files"
            )

        self.logger.info(f"Total files found: {total_files}")
        return files_by_feature

    def load_musical_data(self, file_path: str) -> Optional[Any]:
        """
        Load musical data from various file formats.

        Supports:
        - .mid/.midi files (direct MIDI)
        - .json files (muspy JSON format)
        - .npy files (numpy arrays)
        - .csv files (CSV data)
        - .txt files (text representations)

        Args:
            file_path: Path to the musical data file

        Returns:
            Loaded musical data or None if failed
        """

        path = Path(file_path)

        try:
            if path.suffix.lower() in [".mid", ".midi"]:
                # Direct MIDI file
                return muspy.read_midi(str(path))

            elif path.suffix.lower() == ".json":
                # MusPy JSON format or other JSON
                with open(path, "r") as f:
                    data = json.load(f)

                # Try to convert to MusPy Music object
                if isinstance(data, dict) and "tracks" in data:
                    return muspy.from_dict(data)
                else:
                    return data

            elif path.suffix.lower() == ".npy":
                # Numpy array - try to interpret as MIDI representation
                data = np.load(str(path))
                return data

            elif path.suffix.lower() == ".csv":
                # CSV file - load as numpy array
                data = np.loadtxt(str(path), delimiter=",")
                return data

            elif path.suffix.lower() == ".txt":
                # Text representation - attempt to parse
                with open(path, "r") as f:
                    content = f.read().strip()

                # Try to parse as JSON first
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Return as text representation
                    return content

            else:
                self.logger.warning(f"Unsupported file format: {path.suffix}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            return None

    def convert_to_muspy_music(
        self, data: Any, file_path: str
    ) -> Optional[muspy.Music]:
        """
        Convert various data formats to MusPy Music object for analysis.

        Args:
            data: Loaded musical data
            file_path: Original file path for context

        Returns:
            MusPy Music object or None if conversion failed
        """

        try:
            # Already a MusPy Music object
            if isinstance(data, muspy.Music):
                return data

            # Dictionary - try to convert from dict
            elif isinstance(data, dict):
                if "tracks" in data:
                    return muspy.from_dict(data)
                else:
                    self.logger.warning(
                        f"Dictionary format not recognized for {file_path}"
                    )
                    return None

            # Numpy array - interpret as token sequence
            elif isinstance(data, np.ndarray):
                # This would need specific conversion based on representation format
                # For now, return None - would need representation-specific conversion
                self.logger.warning(
                    f"Numpy array conversion not implemented for {file_path}"
                )
                return None

            # String - try to parse as text representation
            elif isinstance(data, str):
                # This would need format-specific parsing (REMI, MMM, etc.)
                self.logger.warning(
                    f"String representation conversion not implemented for {file_path}"
                )
                return None

            else:
                self.logger.warning(
                    f"Unsupported data type for {file_path}: {type(data)}"
                )
                return None

        except Exception as e:
            self.logger.error(f"Failed to convert data from {file_path}: {e}")
            return None

    def analyze_single_intervention(
        self,
        feature_id: str,
        baseline_files: List[str],
        intervention_files: List[str],
        strength: float,
    ) -> Dict[str, Any]:
        """
        Analyze a single feature intervention condition.

        Args:
            feature_id: ID of the intervened feature
            baseline_files: List of baseline condition files
            intervention_files: List of intervention condition files
            strength: Intervention strength used

        Returns:
            Analysis results for this intervention condition
        """

        analysis_results = {
            "feature_id": feature_id,
            "strength": strength,
            "feature_config": self.features_config[feature_id],
            "baseline_analysis": None,
            "intervention_analysis": None,
            "comparative_analysis": None,
            "files_processed": {"baseline": [], "intervention": []},
            "processing_errors": [],
        }

        try:
            # Process baseline files
            baseline_features_list = []
            for baseline_file in baseline_files:
                data = self.load_musical_data(baseline_file)
                if data is not None:
                    music = self.convert_to_muspy_music(data, baseline_file)
                    if music is not None:
                        features = self.midi_extractor.extract_all_features(music)
                        baseline_features_list.append(features)
                        analysis_results["files_processed"]["baseline"].append(
                            baseline_file
                        )
                    else:
                        analysis_results["processing_errors"].append(
                            f"Failed to convert baseline file: {baseline_file}"
                        )
                else:
                    analysis_results["processing_errors"].append(
                        f"Failed to load baseline file: {baseline_file}"
                    )

            # Process intervention files
            intervention_features_list = []
            for intervention_file in intervention_files:
                data = self.load_musical_data(intervention_file)
                if data is not None:
                    music = self.convert_to_muspy_music(data, intervention_file)
                    if music is not None:
                        features = self.midi_extractor.extract_all_features(music)
                        intervention_features_list.append(features)
                        analysis_results["files_processed"]["intervention"].append(
                            intervention_file
                        )
                    else:
                        analysis_results["processing_errors"].append(
                            f"Failed to convert intervention file: {intervention_file}"
                        )
                else:
                    analysis_results["processing_errors"].append(
                        f"Failed to load intervention file: {intervention_file}"
                    )

            # Aggregate features across files (average)
            if baseline_features_list:
                analysis_results["baseline_analysis"] = self._aggregate_features(
                    baseline_features_list
                )

            if intervention_features_list:
                analysis_results["intervention_analysis"] = self._aggregate_features(
                    intervention_features_list
                )

            # Perform comparative analysis
            if (
                analysis_results["baseline_analysis"]
                and analysis_results["intervention_analysis"]
            ):
                analysis_results["comparative_analysis"] = (
                    self.feature_analyzer.analyze_feature_intervention(
                        feature_id=feature_id,
                        baseline_features=analysis_results["baseline_analysis"],
                        intervention_features=analysis_results["intervention_analysis"],
                        strength=strength,
                        intervention_type="addition" if strength > 0 else "ablation",
                    )
                )

        except Exception as e:
            self.logger.error(
                f"Error analyzing feature {feature_id} strength {strength}: {e}"
            )
            analysis_results["processing_errors"].append(f"Analysis error: {str(e)}")
            analysis_results["processing_errors"].append(traceback.format_exc())

        return analysis_results

    def _aggregate_features(self, features_list: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate feature dictionaries across multiple files.

        Args:
            features_list: List of feature dictionaries to aggregate

        Returns:
            Aggregated features dictionary with averaged values
        """

        if not features_list:
            return {}

        if len(features_list) == 1:
            return features_list[0]

        # Initialize aggregated dictionary
        aggregated = {}

        # Get all keys from first dictionary
        first_features = features_list[0]

        for category, category_data in first_features.items():
            if isinstance(category_data, dict):
                aggregated[category] = {}

                for metric, value in category_data.items():
                    if isinstance(value, (int, float)):
                        # Aggregate numeric values by averaging
                        values = []
                        for features in features_list:
                            if category in features and metric in features[category]:
                                metric_value = features[category][metric]
                                if isinstance(metric_value, (int, float)):
                                    values.append(metric_value)

                        if values:
                            aggregated[category][metric] = np.mean(values)
                        else:
                            aggregated[category][metric] = value
                    else:
                        # Non-numeric values - take first occurrence
                        aggregated[category][metric] = value
            else:
                # Non-dictionary categories - take first occurrence
                aggregated[category] = category_data

        return aggregated

    def run_batch_analysis(
        self,
        input_dir: str,
        features_to_analyze: Optional[List[str]] = None,
        strengths_to_analyze: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Run complete batch analysis on all interventions.

        Args:
            input_dir: Directory containing intervention result files
            features_to_analyze: Specific features to analyze (default: all)
            strengths_to_analyze: Specific strengths to analyze (default: all)

        Returns:
            Complete batch analysis results
        """

        self.logger.info("Starting batch deterministic analysis...")

        # Find all intervention files
        intervention_files = self.find_intervention_files(input_dir)

        if not intervention_files:
            self.logger.error("No intervention files found")
            return {"error": "No intervention files found"}

        # Filter features and strengths if specified
        if features_to_analyze:
            intervention_files = {
                fid: data
                for fid, data in intervention_files.items()
                if fid in features_to_analyze
            }

        if strengths_to_analyze:
            for feature_data in intervention_files.values():
                feature_data["interventions"] = {
                    str(s): files
                    for s, files in feature_data["interventions"].items()
                    if float(s) in strengths_to_analyze
                }

        # Initialize batch results
        batch_results = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "input_directory": input_dir,
                "output_directory": str(self.output_dir),
                "features_analyzed": list(intervention_files.keys()),
                "total_conditions": 0,
            },
            "individual_analyses": {},
            "batch_summary": None,
            "processing_stats": {
                "successful_analyses": 0,
                "failed_analyses": 0,
                "total_files_processed": 0,
                "processing_errors": [],
            },
        }

        # Process each feature
        for feature_id, conditions in intervention_files.items():
            self.logger.info(f"Analyzing feature {feature_id}...")

            feature_results = {
                "feature_config": self.features_config[feature_id],
                "conditions": {},
            }

            baseline_files = conditions["baseline"]

            # Process each intervention strength
            for strength_str, intervention_files_list in conditions[
                "interventions"
            ].items():
                if not intervention_files_list:
                    continue

                strength = float(strength_str)
                self.logger.info(f"  Processing strength {strength}...")

                condition_analysis = self.analyze_single_intervention(
                    feature_id=feature_id,
                    baseline_files=baseline_files,
                    intervention_files=intervention_files_list,
                    strength=strength,
                )

                feature_results["conditions"][strength_str] = condition_analysis

                # Update stats
                batch_results["analysis_metadata"]["total_conditions"] += 1

                if condition_analysis.get("comparative_analysis"):
                    batch_results["processing_stats"]["successful_analyses"] += 1
                else:
                    batch_results["processing_stats"]["failed_analyses"] += 1

                files_count = len(
                    condition_analysis["files_processed"]["baseline"]
                ) + len(condition_analysis["files_processed"]["intervention"])
                batch_results["processing_stats"][
                    "total_files_processed"
                ] += files_count

                if condition_analysis["processing_errors"]:
                    batch_results["processing_stats"]["processing_errors"].extend(
                        condition_analysis["processing_errors"]
                    )

            batch_results["individual_analyses"][feature_id] = feature_results

        # Generate batch summary
        self.logger.info("Generating batch summary...")
        batch_results["batch_summary"] = self._generate_batch_summary(batch_results)

        # Save results
        self._save_batch_results(batch_results)

        self.logger.info("Batch analysis completed!")
        self.logger.info(
            f"Successful analyses: {batch_results['processing_stats']['successful_analyses']}"
        )
        self.logger.info(
            f"Failed analyses: {batch_results['processing_stats']['failed_analyses']}"
        )

        return batch_results

    def _generate_batch_summary(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of batch analysis results."""

        summary = {
            "overview": {
                "total_features": len(batch_results["individual_analyses"]),
                "total_conditions": batch_results["analysis_metadata"][
                    "total_conditions"
                ],
                "success_rate": 0.0,
                "average_quality_score": 0.0,
            },
            "feature_performance": {},
            "layer_analysis": {"early": [], "mid": [], "late": []},
            "strength_analysis": {},
            "quality_distribution": {
                "excellent": 0,
                "good": 0,
                "moderate": 0,
                "poor": 0,
                "failed": 0,
            },
            "recommendations": [],
            "best_interventions": [],
            "problematic_interventions": [],
        }

        # Collect all comparative analyses
        all_analyses = {}
        quality_scores = []

        for feature_id, feature_data in batch_results["individual_analyses"].items():
            feature_config = self.features_config[feature_id]
            layer_type = feature_config["type"]

            for strength_str, condition_data in feature_data["conditions"].items():
                if condition_data.get("comparative_analysis"):
                    analysis_key = f"{feature_id}_{strength_str}"
                    all_analyses[analysis_key] = condition_data["comparative_analysis"]

                    # Track by layer
                    if "overall_decision" in condition_data["comparative_analysis"]:
                        decision = condition_data["comparative_analysis"][
                            "overall_decision"
                        ]
                        score = decision["overall_score"]
                        quality_scores.append(score)

                        summary["layer_analysis"][layer_type].append(
                            {
                                "feature_id": feature_id,
                                "strength": strength_str,
                                "score": score,
                            }
                        )

                        # Track quality distribution
                        quality_level = decision["quality_level"]
                        if quality_level in summary["quality_distribution"]:
                            summary["quality_distribution"][quality_level] += 1

        # Calculate overview stats
        if quality_scores:
            summary["overview"]["success_rate"] = len(
                [s for s in quality_scores if s >= 0.5]
            ) / len(quality_scores)
            summary["overview"]["average_quality_score"] = np.mean(quality_scores)

        # Use existing batch analysis function
        if all_analyses:
            detailed_summary = analyze_intervention_batch(all_analyses)
            summary.update(detailed_summary)

        return summary

    def _save_batch_results(self, batch_results: Dict[str, Any]):
        """Save batch analysis results to files."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save complete results as JSON
        results_file = self.output_dir / f"batch_analysis_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(batch_results, f, indent=2, default=str)

        # Save summary report
        summary_file = self.output_dir / f"analysis_summary_{timestamp}.md"
        self._generate_markdown_report(batch_results, summary_file)

        # Save CSV summary for easier analysis
        csv_file = self.output_dir / f"intervention_scores_{timestamp}.csv"
        self._generate_csv_summary(batch_results, csv_file)

        self.logger.info(f"Results saved:")
        self.logger.info(f"  Complete results: {results_file}")
        self.logger.info(f"  Summary report: {summary_file}")
        self.logger.info(f"  CSV summary: {csv_file}")

    def _generate_markdown_report(
        self, batch_results: Dict[str, Any], output_file: Path
    ):
        """Generate markdown summary report."""

        summary = batch_results.get("batch_summary", {})

        with open(output_file, "w") as f:
            f.write("# Deterministic Analysis Report\n\n")

            # Overview
            f.write("## Overview\n\n")
            overview = summary.get("overview", {})
            f.write(
                f"- **Total Features Analyzed**: {overview.get('total_features', 0)}\n"
            )
            f.write(f"- **Total Conditions**: {overview.get('total_conditions', 0)}\n")
            f.write(f"- **Success Rate**: {overview.get('success_rate', 0):.1%}\n")
            f.write(
                f"- **Average Quality Score**: {overview.get('average_quality_score', 0):.3f}\n\n"
            )

            # Quality Distribution
            f.write("## Quality Distribution\n\n")
            quality_dist = summary.get("quality_distribution", {})
            for level, count in quality_dist.items():
                f.write(f"- **{level.title()}**: {count}\n")
            f.write("\n")

            # Layer Performance
            f.write("## Layer Performance\n\n")
            layer_analysis = summary.get("layer_analysis", {})
            for layer_type, analyses in layer_analysis.items():
                if analyses:
                    scores = [a["score"] for a in analyses]
                    avg_score = np.mean(scores)
                    f.write(f"### {layer_type.title()} Layer\n")
                    f.write(f"- Average Score: {avg_score:.3f}\n")
                    f.write(f"- Conditions: {len(analyses)}\n\n")

            # Recommendations
            recommendations = summary.get("recommendations", [])
            if recommendations:
                f.write("## Recommendations\n\n")
                for i, rec in enumerate(recommendations, 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")

            # Processing Stats
            f.write("## Processing Statistics\n\n")
            stats = batch_results.get("processing_stats", {})
            f.write(
                f"- **Successful Analyses**: {stats.get('successful_analyses', 0)}\n"
            )
            f.write(f"- **Failed Analyses**: {stats.get('failed_analyses', 0)}\n")
            f.write(
                f"- **Total Files Processed**: {stats.get('total_files_processed', 0)}\n"
            )

            errors = stats.get("processing_errors", [])
            if errors:
                f.write(f"- **Processing Errors**: {len(errors)}\n")

    def _generate_csv_summary(self, batch_results: Dict[str, Any], output_file: Path):
        """Generate CSV summary of intervention scores."""

        import csv

        with open(output_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(
                [
                    "feature_id",
                    "feature_name",
                    "layer",
                    "layer_type",
                    "strength",
                    "overall_score",
                    "quality_level",
                    "effectiveness_score",
                    "quality_preservation_score",
                    "layer_appropriateness_score",
                    "success",
                    "files_processed",
                ]
            )

            # Data rows
            for feature_id, feature_data in batch_results[
                "individual_analyses"
            ].items():
                feature_config = self.features_config[feature_id]

                for strength_str, condition_data in feature_data["conditions"].items():
                    if (
                        condition_data.get("comparative_analysis")
                        and "overall_decision" in condition_data["comparative_analysis"]
                    ):

                        analysis = condition_data["comparative_analysis"]
                        decision = analysis["overall_decision"]

                        files_count = len(
                            condition_data["files_processed"]["baseline"]
                        ) + len(condition_data["files_processed"]["intervention"])

                        writer.writerow(
                            [
                                feature_id,
                                feature_config["name"],
                                feature_config["layer"],
                                feature_config["type"],
                                strength_str,
                                decision["overall_score"],
                                decision["quality_level"],
                                decision["component_scores"]["effectiveness"],
                                decision["component_scores"]["quality_preservation"],
                                decision["component_scores"]["layer_appropriateness"],
                                decision["success_criteria"]["overall_success"],
                                files_count,
                            ]
                        )


def main():
    """Main function for command-line usage."""

    parser = argparse.ArgumentParser(
        description="Batch Deterministic Analysis for Feature Interventions"
    )
    parser.add_argument(
        "input_dir", help="Directory containing intervention result files"
    )
    parser.add_argument(
        "--output-dir",
        default="deterministic_results",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--features", nargs="+", help="Specific feature IDs to analyze (default: all)"
    )
    parser.add_argument(
        "--strengths",
        nargs="+",
        type=float,
        help="Specific intervention strengths to analyze (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = BatchDeterministicAnalyzer(output_dir=args.output_dir)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run analysis
    try:
        results = analyzer.run_batch_analysis(
            input_dir=args.input_dir,
            features_to_analyze=args.features,
            strengths_to_analyze=args.strengths,
        )

        if "error" in results:
            print(f"Analysis failed: {results['error']}")
            return 1

        print("Analysis completed successfully!")
        print(f"Results saved to: {analyzer.output_dir}")

        # Print quick summary
        summary = results.get("batch_summary", {})
        overview = summary.get("overview", {})
        print(f"\nQuick Summary:")
        print(f"- Features analyzed: {overview.get('total_features', 0)}")
        print(f"- Success rate: {overview.get('success_rate', 0):.1%}")
        print(f"- Average quality: {overview.get('average_quality_score', 0):.3f}")

        return 0

    except Exception as e:
        print(f"Analysis failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
