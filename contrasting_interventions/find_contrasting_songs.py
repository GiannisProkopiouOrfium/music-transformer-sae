#!/usr/bin/env python3
"""
Find Contrasting Songs for Dramatic Interventions

This script identifies songs from the training corpus that have contrasting properties
to target features, ensuring interventions produce dramatic, obvious changes.

Three parallel selection criteria:
1. SAE Activations: Songs that minimally activate target features
2. Musical Feature Analysis: Direct metric analysis showing low feature presence
3. Diversity Report Exclusion: Not in the high-activation list from diversity reports

Usage:
    python find_contrasting_songs.py \
        --layer 3 \
        --features 182 855 997 \
        --sae-path exp/sod/ape/sae_models/sae_layer_2048d.pt \
        --diversity-report layer_interpretations/enhanced_diversity_report_layer3.json \
        --data-dir data/sod/processed/json/Kunstderfuge \
        --output-dir contrasting_songs_results \
        --top-n 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import torch
import muspy
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deterministic_analysis.midi_feature_extractors import MIDIFeatureExtractor


class ContrastingSongFinder:
    """Find songs with contrasting properties to target features."""

    def __init__(
        self,
        layer: int,
        feature_ids: List[int],
        sae_path: Path,
        diversity_report_path: Optional[Path],
        data_dir: Path,
        output_dir: Path,
    ):
        self.layer = layer
        self.feature_ids = feature_ids
        self.sae_path = sae_path
        self.diversity_report_path = diversity_report_path
        self.data_dir = data_dir
        self.output_dir = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(output_dir / "contrasting_songs.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

        # Load SAE model
        self.logger.info(f"Loading SAE model from {sae_path}")
        self.sae = self._load_sae()

        # Load diversity report if provided
        self.high_activation_songs = set()
        if diversity_report_path and diversity_report_path.exists():
            self.logger.info(f"Loading diversity report from {diversity_report_path}")
            self.high_activation_songs = self._extract_high_activation_songs()

        # Initialize feature extractor for musical analysis
        self.midi_extractor = MIDIFeatureExtractor(
            encoding_path="../data/sod/processed/encoding.json"
        )

        # Feature-specific metric mappings
        self.feature_metric_mapping = {
            182: {  # steady_pulse
                "name": "steady_pulse_march_rhythm",
                "metrics": [
                    "rhythmic_analysis.rhythmic_regularity",
                    "rhythmic_analysis.rhythmic_complexity",
                    "rhythmic_analysis.syncopation_score",
                ],
                "preference": "low_regularity",  # Want irregular for contrast
            },
            855: {  # dynamic_contrast
                "name": "dynamic_contrast_tension",
                "metrics": [
                    "velocity_dynamics.dynamic_range",
                    "velocity_dynamics.dynamic_variance",
                    "note_patterns.average_velocity",
                ],
                "preference": "low_variance",  # Want flat dynamics for contrast
            },
            997: {  # antiphonal_texture
                "name": "antiphonal_call_response",
                "metrics": [
                    "musical_complexity.polyphonic_complexity",
                    "structural_analysis.phrase_count",
                    "note_patterns.unique_programs",
                ],
                "preference": "low_complexity",  # Want simple texture for contrast
            },
            325: {  # rhythmic_displacement (Layer 1)
                "name": "rhythmic_displacement",
                "metrics": [
                    "rhythmic_analysis.syncopation_score",
                    "rhythmic_analysis.ioi_std",
                ],
                "preference": "low_syncopation",
            },
            256: {  # dynamic_contrasts (Layer 1)
                "name": "dynamic_contrasts",
                "metrics": [
                    "velocity_dynamics.dynamic_range",
                    "velocity_dynamics.dynamic_variance",
                ],
                "preference": "low_variance",
            },
            1323: {  # wide_pitch_range (Layer 1)
                "name": "wide_pitch_range",
                "metrics": [
                    "pitch_analysis.pitch_range",
                    "pitch_analysis.pitch_std",
                ],
                "preference": "narrow_range",
            },
            471: {  # unison_doubling (Layer 5)
                "name": "unison_doubling",
                "metrics": [
                    "musical_complexity.polyphonic_complexity",
                    "note_patterns.unique_programs",
                ],
                "preference": "low_complexity",
            },
            904: {  # rhythmic_augmentation (Layer 5)
                "name": "rhythmic_augmentation",
                "metrics": [
                    "rhythmic_analysis.rhythmic_complexity",
                    "note_patterns.average_note_duration",
                ],
                "preference": "simple_rhythm",
            },
            1950: {  # dynamic_contrasts_layer5 (Layer 5)
                "name": "dynamic_contrasts_layer5",
                "metrics": [
                    "velocity_dynamics.dynamic_range",
                    "velocity_dynamics.dynamic_variance",
                ],
                "preference": "low_variance",
            },
        }

    def _load_sae(self) -> torch.nn.Module:
        """Load the SAE model."""
        try:
            checkpoint = torch.load(self.sae_path, map_location="cpu")
            # Extract model architecture from checkpoint
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            # Get dimensions from state dict
            encoder_weight = state_dict["encoder.weight"]
            d_model, d_sae = encoder_weight.shape

            self.logger.info(f"SAE dimensions: d_model={d_model}, d_sae={d_sae}")

            # Create SAE model (simplified version)
            from sae.sae_data import SAE

            sae = SAE(d_model, d_sae)
            sae.load_state_dict(state_dict)
            sae.eval()

            return sae

        except Exception as e:
            self.logger.error(f"Failed to load SAE: {e}")
            self.logger.warning("Continuing without SAE activation analysis...")
            return None

    def _extract_high_activation_songs(self) -> Set[str]:
        """Extract song IDs that highly activate target features from diversity report."""
        try:
            with open(self.diversity_report_path, "r") as f:
                report = json.load(f)

            high_activation_songs = set()

            for feature_id in self.feature_ids:
                feature_key = str(feature_id)
                if feature_key in report.get("interpretations", {}):
                    feature_data = report["interpretations"][feature_key]
                    if "feature_stats" in feature_data:
                        top_activations = feature_data["feature_stats"].get(
                            "top_activations", []
                        )
                        for activation in top_activations:
                            song_id = activation.get("song_id")
                            if song_id:
                                # Extract just the song name without path
                                song_name = Path(song_id).stem
                                high_activation_songs.add(song_name)

            self.logger.info(
                f"Found {len(high_activation_songs)} songs with high activations for target features"
            )
            return high_activation_songs

        except Exception as e:
            self.logger.warning(
                f"Could not load diversity report: {e}. Continuing without exclusion list."
            )
            return set()

    def find_json_files(self) -> List[Path]:
        """Find all MusPy JSON files in the data directory."""
        json_files = list(self.data_dir.rglob("*.json"))
        self.logger.info(f"Found {len(json_files)} JSON files in {self.data_dir}")
        return json_files

    def load_muspy_json(self, file_path: Path) -> Optional[muspy.Music]:
        """Load a MusPy JSON file."""
        try:
            music = muspy.load(str(file_path))
            return music
        except Exception as e:
            self.logger.debug(f"Failed to load {file_path}: {e}")
            return None

    def calculate_sae_activations(
        self, music: muspy.Music, song_id: str
    ) -> Optional[Dict[int, float]]:
        """
        Calculate SAE activations for target features.

        Returns:
            Dictionary mapping feature_id to average activation strength
        """
        if self.sae is None:
            return None

        try:
            # This is a simplified version - you'll need to adapt based on your actual
            # SAE forward pass pipeline. The key is to get activations for specific features.

            # For now, return None to skip SAE-based filtering
            # TODO: Implement actual SAE forward pass with music data
            return None

        except Exception as e:
            self.logger.debug(f"SAE activation failed for {song_id}: {e}")
            return None

    def calculate_musical_contrast_score(
        self, music: muspy.Music, feature_id: int
    ) -> float:
        """
        Calculate how much this song contrasts with the target feature.

        Higher score = better contrast (more different from feature characteristics)

        Returns:
            Contrast score (0-1, higher is better)
        """
        try:
            # Extract all musical features
            features = self.midi_extractor.extract_all_features(music)

            if feature_id not in self.feature_metric_mapping:
                self.logger.warning(f"No metric mapping for feature {feature_id}")
                return 0.0

            feature_config = self.feature_metric_mapping[feature_id]
            metrics_to_check = feature_config["metrics"]
            preference = feature_config["preference"]

            contrast_scores = []

            for metric_path in metrics_to_check:
                category, metric_name = metric_path.split(".")

                if category in features and metric_name in features[category]:
                    value = features[category][metric_name]

                    # Normalize and score based on preference
                    if (
                        "low" in preference
                        or "narrow" in preference
                        or "simple" in preference
                    ):
                        # Want low values for contrast
                        # Normalize: assume typical range is 0-1 or 0-100
                        if "regularity" in metric_name or "consistency" in metric_name:
                            # Lower regularity = higher contrast score
                            normalized = 1.0 - min(value, 1.0)
                        elif "range" in metric_name:
                            # Lower range = higher contrast (normalize by typical max 88)
                            normalized = 1.0 - min(value / 88.0, 1.0)
                        elif "variance" in metric_name or "std" in metric_name:
                            # Lower variance = higher contrast
                            normalized = 1.0 - min(value / 50.0, 1.0)
                        elif "complexity" in metric_name:
                            # Lower complexity = higher contrast
                            normalized = 1.0 - min(value / 10.0, 1.0)
                        elif "syncopation" in metric_name:
                            # Lower syncopation = higher contrast
                            normalized = 1.0 - min(value, 1.0)
                        else:
                            # Generic low preference
                            normalized = 1.0 - min(value / 100.0, 1.0)
                    else:
                        # Want high values for contrast (less common)
                        normalized = min(value / 100.0, 1.0)

                    contrast_scores.append(max(0.0, min(1.0, normalized)))

            # Return average contrast score
            if contrast_scores:
                return np.mean(contrast_scores)
            else:
                return 0.0

        except Exception as e:
            self.logger.debug(f"Failed to calculate contrast score: {e}")
            return 0.0

    def score_all_songs(self, json_files: List[Path]) -> List[Tuple[str, float, Dict]]:
        """
        Score all songs for contrast with target features.

        Returns:
            List of (song_id, total_contrast_score, details) sorted by score (highest first)
        """
        song_scores = []

        self.logger.info("Scoring songs for contrast with target features...")

        for json_file in tqdm(json_files, desc="Analyzing songs"):
            song_name = json_file.stem

            # Skip if in high activation list
            if song_name in self.high_activation_songs:
                self.logger.debug(
                    f"Skipping {song_name}: in high activation exclusion list"
                )
                continue

            # Load music
            music = self.load_muspy_json(json_file)
            if music is None:
                continue

            # Calculate contrast scores for each target feature
            feature_scores = {}
            for feature_id in self.feature_ids:
                contrast_score = self.calculate_musical_contrast_score(
                    music, feature_id
                )
                feature_scores[feature_id] = contrast_score

            # Calculate total score (average across features)
            total_score = np.mean(list(feature_scores.values()))

            # Calculate SAE activations (if available)
            sae_activations = self.calculate_sae_activations(music, song_name)

            details = {
                "song_path": str(json_file),
                "feature_contrast_scores": feature_scores,
                "sae_activations": sae_activations,
                "excluded_from_diversity_report": song_name
                not in self.high_activation_songs,
            }

            song_scores.append((song_name, total_score, details))

        # Sort by total score (highest contrast first)
        song_scores.sort(key=lambda x: x[1], reverse=True)

        return song_scores

    def save_results(self, song_scores: List[Tuple[str, float, Dict]], top_n: int):
        """Save contrasting song results."""

        # Save full results
        full_results = {
            "metadata": {
                "layer": self.layer,
                "target_features": self.feature_ids,
                "feature_names": [
                    self.feature_metric_mapping[fid]["name"]
                    for fid in self.feature_ids
                    if fid in self.feature_metric_mapping
                ],
                "total_songs_analyzed": len(song_scores),
                "high_activation_exclusions": len(self.high_activation_songs),
                "top_n_selected": top_n,
            },
            "top_contrasting_songs": [],
            "all_scores": [],
        }

        # Add top N songs
        for i, (song_name, score, details) in enumerate(song_scores[:top_n]):
            full_results["top_contrasting_songs"].append(
                {
                    "rank": i + 1,
                    "song_name": song_name,
                    "contrast_score": float(score),
                    "details": details,
                }
            )

        # Add all scores for reference
        for song_name, score, details in song_scores:
            full_results["all_scores"].append(
                {"song_name": song_name, "contrast_score": float(score)}
            )

        # Save JSON
        output_file = (
            self.output_dir / f"contrasting_songs_layer{self.layer}_results.json"
        )
        with open(output_file, "w") as f:
            json.dump(full_results, f, indent=2)

        self.logger.info(f"✅ Saved results to {output_file}")

        # Save top N song names to text file for easy use
        names_file = (
            self.output_dir / f"contrasting_songs_layer{self.layer}_top{top_n}.txt"
        )
        with open(names_file, "w") as f:
            for song_name, _, _ in song_scores[:top_n]:
                f.write(f"{song_name}\n")

        self.logger.info(f"✅ Saved song names to {names_file}")

        return output_file, names_file

    def print_summary(self, song_scores: List[Tuple[str, float, Dict]], top_n: int):
        """Print summary of contrasting songs."""
        print("\n" + "=" * 80)
        print("CONTRASTING SONGS SELECTION SUMMARY")
        print("=" * 80)

        print(f"\n🎯 Target Configuration:")
        print(f"  Layer: {self.layer}")
        print(f"  Features: {self.feature_ids}")
        for fid in self.feature_ids:
            if fid in self.feature_metric_mapping:
                print(
                    f"    - Feature {fid}: {self.feature_metric_mapping[fid]['name']}"
                )

        print(f"\n📊 Analysis Results:")
        print(f"  Total songs analyzed: {len(song_scores)}")
        print(f"  High activation exclusions: {len(self.high_activation_songs)}")
        print(f"  Top N selected: {top_n}")

        print(f"\n🏆 Top {min(top_n, 10)} Contrasting Songs:")
        print("-" * 80)

        for i, (song_name, score, details) in enumerate(song_scores[:10]):
            print(f"\n{i+1}. {song_name}")
            print(f"   Overall Contrast Score: {score:.3f}")
            print(f"   Feature-specific scores:")
            for fid, fscore in details["feature_contrast_scores"].items():
                fname = self.feature_metric_mapping.get(fid, {}).get("name", f"F{fid}")
                print(f"     - {fname}: {fscore:.3f}")

        print("\n" + "=" * 80)

    def run(self, top_n: int = 20) -> Tuple[Path, Path]:
        """
        Run the complete contrasting song finding pipeline.

        Args:
            top_n: Number of top contrasting songs to select

        Returns:
            Tuple of (results_json_path, names_txt_path)
        """
        self.logger.info("🚀 Starting Contrasting Song Finder")
        self.logger.info("=" * 80)

        # Find all JSON files
        json_files = self.find_json_files()

        if not json_files:
            self.logger.error("No JSON files found!")
            return None, None

        # Score all songs
        song_scores = self.score_all_songs(json_files)

        if not song_scores:
            self.logger.error("No songs scored successfully!")
            return None, None

        # Save results
        results_file, names_file = self.save_results(song_scores, top_n)

        # Print summary
        self.print_summary(song_scores, top_n)

        return results_file, names_file


def main():
    parser = argparse.ArgumentParser(
        description="Find songs with contrasting properties for dramatic interventions"
    )
    parser.add_argument(
        "--layer", type=int, required=True, help="Target layer (1, 3, or 5)"
    )
    parser.add_argument(
        "--features",
        nargs="+",
        type=int,
        required=True,
        help="Target feature IDs (e.g., 182 855 997)",
    )
    parser.add_argument(
        "--sae-path",
        type=Path,
        required=True,
        help="Path to SAE model (e.g., exp/sod/ape/sae_models/sae_layer_2048d.pt)",
    )
    parser.add_argument(
        "--diversity-report",
        type=Path,
        default=None,
        help="Path to diversity report JSON for exclusion list",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/sod/processed/json/Kunstderfuge"),
        help="Directory containing MusPy JSON training files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("contrasting_songs_results"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top contrasting songs to select",
    )

    args = parser.parse_args()

    # Create finder and run
    finder = ContrastingSongFinder(
        layer=args.layer,
        feature_ids=args.features,
        sae_path=args.sae_path,
        diversity_report_path=args.diversity_report,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    results_file, names_file = finder.run(top_n=args.top_n)

    if results_file and names_file:
        print("\n✅ Contrasting song selection complete!")
        print(f"📁 Results saved to: {results_file}")
        print(f"📁 Song names saved to: {names_file}")
        print(
            f"\nNext step: Run interventions on these contrasting songs for dramatic effects!"
        )
    else:
        print("\n❌ Contrasting song selection failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
