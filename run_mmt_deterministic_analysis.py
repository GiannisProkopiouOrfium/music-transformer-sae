#!/usr/bin/env python3
"""
Extended Batch Analyzer for MMT PyTorch Tensor Interventions

This extends the deterministic analysis to handle your specific .pt tensor format
without requiring conversion.
"""

import sys
import os
import logging
from pathlib import Path
import torch
import numpy as np

# Add MMT modules to path
current_dir = Path(__file__).parent
mmt_dir = current_dir / "mmt"
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(mmt_dir))

# Import MMT modules
from mmt import representation
from mmt.representation import RESOLUTION
import muspy

# Import deterministic analysis modules directly
from deterministic_analysis.batch_deterministic_analyzer import (
    BatchDeterministicAnalyzer,
)


class MMTBatchAnalyzer(BatchDeterministicAnalyzer):
    """Extended analyzer that handles MMT PyTorch tensor format."""

    def __init__(
        self, output_dir: str = "deterministic_results", encoding_path: str = None
    ):
        """Initialize with MMT-specific configuration."""
        super().__init__(output_dir)

        # Load encoding
        if encoding_path is None:
            # Try common paths
            possible_paths = [
                "data/sod/processed/notes/encoding.json",
                "mmt/data/sod/processed/notes/encoding.json",
                "../data/sod/processed/notes/encoding.json",
                "baseline/encoding.json",
            ]
            for path in possible_paths:
                if Path(path).exists():
                    encoding_path = path
                    break
            else:
                raise FileNotFoundError("Could not find encoding.json file")

        self.encoding = representation.load_encoding(encoding_path)
        print(f"Loaded encoding from: {encoding_path}")

    def load_musical_data(self, file_path: str):
        """Extended to handle MMT PyTorch tensors."""
        path = Path(file_path)

        try:
            if path.suffix.lower() == ".pt":
                # Load MMT PyTorch tensor
                tensor_data = torch.load(str(path), map_location="cpu")

                # Extract the generated sequence
                if "generated" in tensor_data:
                    sequence = tensor_data["generated"]
                else:
                    self.logger.warning(f"No 'generated' key in {path.name}")
                    return None

                # Convert to numpy
                if isinstance(sequence, torch.Tensor):
                    seq_np = sequence.numpy()
                else:
                    seq_np = sequence

                # Remove batch dimension if present
                if len(seq_np.shape) == 3 and seq_np.shape[0] == 1:
                    seq_np = seq_np[0]

                # Debug: Check what's in the sequence
                self.logger.debug(f"Sequence shape: {seq_np.shape}")
                self.logger.debug(f"Sequence dtype: {seq_np.dtype}")

                # Check if the sequence contains Note objects instead of numeric codes
                if seq_np.size > 0:
                    sample_element = seq_np.flat[0]
                    self.logger.debug(f"Sample element type: {type(sample_element)}")
                    self.logger.debug(f"Sample element: {sample_element}")

                    # If we have Note objects, try to create a MusPy Music object directly
                    if hasattr(sample_element, "__class__") and "Note" in str(
                        type(sample_element)
                    ):
                        self.logger.info(
                            f"Sequence contains Note objects - attempting direct MusPy conversion for {path.name}"
                        )
                        try:
                            # Create a MusPy Music object from the Note objects
                            music = muspy.Music(resolution=RESOLUTION)
                            track = muspy.Track(program=0, is_drum=False)

                            # Convert the Note objects to MusPy notes
                            notes = []
                            for note_obj in seq_np.flat:
                                if (
                                    hasattr(note_obj, "pitch")
                                    and hasattr(note_obj, "start")
                                    and hasattr(note_obj, "duration")
                                ):
                                    muspy_note = muspy.Note(
                                        start=(
                                            int(note_obj.start)
                                            if hasattr(note_obj, "start")
                                            else 0
                                        ),
                                        pitch=(
                                            int(note_obj.pitch)
                                            if hasattr(note_obj, "pitch")
                                            else 60
                                        ),
                                        duration=(
                                            int(note_obj.duration)
                                            if hasattr(note_obj, "duration")
                                            else 12
                                        ),
                                        velocity=(
                                            int(note_obj.velocity)
                                            if hasattr(note_obj, "velocity")
                                            else 80
                                        ),
                                    )
                                    notes.append(muspy_note)

                            track.notes = notes
                            music.tracks = [track]
                            return music

                        except Exception as e:
                            self.logger.error(
                                f"Failed to convert Note objects to MusPy: {e}"
                            )
                            return None

                    # If we have non-numeric data, try to convert
                    if not isinstance(
                        sample_element, (int, float, np.integer, np.floating)
                    ):
                        self.logger.warning(
                            f"Sequence contains non-numeric data: {type(sample_element)}"
                        )
                        try:
                            # Try to convert to numeric array
                            seq_np = seq_np.astype(int)
                        except (ValueError, TypeError) as e:
                            self.logger.error(
                                f"Cannot convert sequence to numeric: {e}"
                            )
                            return None

                # Convert to MusPy Music object using your representation
                music = representation.decode(seq_np, self.encoding)
                return music

            elif path.suffix.lower() in [".wav", ".mp3", ".mp4", ".flac"]:
                # Skip audio files - they're not suitable for symbolic analysis
                self.logger.debug(f"Skipping audio file: {path.name}")
                return None

            else:
                # Fall back to parent class for other formats
                return super().load_musical_data(file_path)

        except Exception as e:
            self.logger.error(f"Failed to load MMT tensor {file_path}: {e}")
            return None

    def convert_to_muspy_music(self, data, file_path: str):
        """Override - data should already be MusPy Music from load_musical_data."""
        if isinstance(data, muspy.Music):
            return data
        else:
            return super().convert_to_muspy_music(data, file_path)

    def find_intervention_files(self, base_dir: str):
        """Override to exclude audio files from processing."""
        # Get files from parent method
        files_by_feature = super().find_intervention_files(base_dir)

        # Filter out audio files
        audio_extensions = {".wav", ".mp3", ".mp4", ".flac", ".aiff", ".ogg"}

        for feature_id in files_by_feature:
            # Filter baseline files
            files_by_feature[feature_id]["baseline"] = [
                f
                for f in files_by_feature[feature_id]["baseline"]
                if Path(f).suffix.lower() not in audio_extensions
            ]

            # Filter intervention files
            for strength in files_by_feature[feature_id]["interventions"]:
                files_by_feature[feature_id]["interventions"][strength] = [
                    f
                    for f in files_by_feature[feature_id]["interventions"][strength]
                    if Path(f).suffix.lower() not in audio_extensions
                ]

        return files_by_feature


def run_mmt_analysis(
    interventions_dir: str, output_dir: str = "mmt_deterministic_results"
):
    """
    Run deterministic analysis on MMT intervention results.

    Args:
        interventions_dir: Path to your interventions directory
        output_dir: Output directory for analysis results
    """

    print("🎵 MMT Intervention Analysis")
    print("=" * 50)
    print(f"Input: {interventions_dir}")
    print(f"Output: {output_dir}")
    print()

    try:
        # Initialize MMT-specific analyzer
        analyzer = MMTBatchAnalyzer(output_dir=output_dir)

        # Run analysis
        results = analyzer.run_batch_analysis(
            input_dir=interventions_dir,
            # You can specify specific features if needed
            # features_to_analyze=["325", "256", "1323", "182", "855", "997", "471", "904", "1950"]
        )

        if "error" not in results:
            summary = results.get("batch_summary", {})
            overview = summary.get("overview", {})

            print("\n🎉 Analysis Complete!")
            print(f"Features analyzed: {overview.get('total_features', 0)}")
            print(f"Total conditions: {overview.get('total_conditions', 0)}")
            print(f"Success rate: {overview.get('success_rate', 0):.1%}")
            print(f"Average quality: {overview.get('average_quality_score', 0):.3f}")
            print(f"\nResults saved to: {output_dir}/")

            return results
        else:
            print(f"Analysis failed: {results['error']}")
            return None

    except Exception as e:
        print(f"Error running analysis: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run deterministic analysis on MMT interventions"
    )
    parser.add_argument(
        "interventions_dir", help="Directory containing intervention .pt files"
    )
    parser.add_argument(
        "--output-dir",
        default="mmt_deterministic_results",
        help="Output directory for analysis results",
    )
    parser.add_argument("--encoding-path", help="Path to encoding.json file")

    args = parser.parse_args()

    # Run the analysis
    results = run_mmt_analysis(args.interventions_dir, args.output_dir)

    if results:
        print("\n✅ Success! Check the output directory for detailed results.")
    else:
        print("\n❌ Analysis failed. Check the error messages above.")
        exit(1)
