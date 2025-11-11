#!/usr/bin/env python3
"""Phase 1: Analyze Modality Distribution in Dataset.

This script validates that music21 key detection works on your dataset
and determines optimal sampling strategy for major/minor classification.

Usage:
    # Analyze sample of dataset
    python modality/analyze_modality_distribution.py --sample_size 500

    # Analyze full dataset
    python modality/analyze_modality_distribution.py --sample_size None

    # Test different confidence thresholds
    python modality/analyze_modality_distribution.py --confidence_thresholds "0.5,0.6,0.7,0.8"
"""

import argparse
import json
import logging
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import config

# Import music21 (will check if available)
try:
    from music21 import note, stream

    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("WARNING: music21 not available. Install with: pip install music21")


def detect_key_from_json(
    json_path: pathlib.Path, min_notes: int = 10
) -> Tuple[str, str, float, int]:
    """Detect musical key from JSON file.

    Args:
        json_path: Path to JSON file
        min_notes: Minimum number of notes required for detection

    Returns:
        (mode, tonic, confidence, num_notes)
        Returns ("unknown", "N/A", 0.0, 0) if detection fails
    """
    if not MUSIC21_AVAILABLE:
        return ("unknown", "N/A", 0.0, 0)

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Extract all pitches from all tracks
        pitches = []
        for track in data.get("tracks", []):
            for n in track.get("notes", []):
                pitch = n.get("pitch")
                if pitch is not None:
                    pitches.append(pitch)

        # Check if we have enough notes
        if len(pitches) < min_notes:
            return ("unknown", "N/A", 0.0, len(pitches))

        # Create music21 stream
        s = stream.Stream()
        for p in pitches:
            s.append(note.Note(p))

        # Analyze key
        key = s.analyze("key")
        mode = key.mode  # 'major' or 'minor'
        tonic = key.tonic.name  # e.g., 'C', 'A-', 'F#'
        confidence = key.correlationCoefficient  # 0.0 to 1.0

        return (mode, tonic, confidence, len(pitches))

    except Exception as e:
        logging.warning(f"Key detection failed for {json_path.name}: {e}")
        return ("unknown", "N/A", 0.0, 0)


def detect_key_from_segment(
    tracks: List[Dict],
    start_beat: int,
    n_beats: int,
    resolution: int = 12,
    min_notes: int = 10,
) -> Tuple[str, str, float, int]:
    """Detect key from a segment of tracks.

    Args:
        tracks: List of track dictionaries
        start_beat: Starting beat index
        n_beats: Number of beats in segment
        resolution: Time resolution (ticks per beat)
        min_notes: Minimum notes required

    Returns:
        (mode, tonic, confidence, num_notes)
    """
    if not MUSIC21_AVAILABLE:
        return ("unknown", "N/A", 0.0, 0)

    start_time = start_beat * resolution
    end_time = (start_beat + n_beats) * resolution

    # Extract pitches in this segment
    pitches = []
    for track in tracks:
        for n in track.get("notes", []):
            time = n.get("time", 0)
            pitch = n.get("pitch")
            if start_time <= time < end_time and pitch is not None:
                pitches.append(pitch)

    # Check if we have enough notes
    if len(pitches) < min_notes:
        return ("unknown", "N/A", 0.0, len(pitches))

    try:
        # Create music21 stream
        s = stream.Stream()
        for p in pitches:
            s.append(note.Note(p))

        # Analyze key
        key = s.analyze("key")
        mode = key.mode
        tonic = key.tonic.name
        confidence = key.correlationCoefficient

        return (mode, tonic, confidence, len(pitches))

    except Exception as e:
        logging.warning(f"Key detection failed for segment: {e}")
        return ("unknown", "N/A", 0.0, len(pitches))


def analyze_distribution(
    n_beats: int = None,
    sample_size: int = None,
    confidence_thresholds: List[float] = None,
    use_train_split: bool = True,
    min_notes: int = 10,
):
    """Analyze modality distribution in dataset.

    Args:
        n_beats: Segment length (None = whole songs)
        sample_size: Number of files to sample (None = all)
        confidence_thresholds: List of thresholds to test
        use_train_split: Whether to use train split only
        min_notes: Minimum notes required for detection
    """
    if confidence_thresholds is None:
        confidence_thresholds = [0.5, 0.6, 0.7, 0.8]

    print("=" * 70)
    print("MODALITY DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"Segment length: {n_beats if n_beats else 'Full songs'} beats")
    print(f"Minimum notes: {min_notes}")
    print(f"Confidence thresholds to test: {confidence_thresholds}")
    print("=" * 70)

    # Get JSON files
    json_files = sorted(config.JSON_DIR.rglob("*.json"))

    # Filter by train split
    if use_train_split:
        train_names_file = config.DATA_DIR / "train-names.txt"
        if train_names_file.exists():
            with open(train_names_file, "r") as f:
                train_names = set(line.strip() for line in f)

            json_files = [
                jf
                for jf in json_files
                if jf.relative_to(config.JSON_DIR).with_suffix("").as_posix()
                in train_names
            ]
            print(f"\nFound {len(json_files)} files in train split")
        else:
            print(f"\nWARNING: train-names.txt not found, using all files")
    else:
        print(f"\nFound {len(json_files)} total files")

    # Sample if requested
    if sample_size and sample_size < len(json_files):
        np.random.seed(42)
        json_files = list(np.random.choice(json_files, sample_size, replace=False))
        print(f"Sampling {sample_size} files for faster analysis")

    # Collect results
    print(f"\nAnalyzing files...")
    all_results = []

    for json_path in tqdm(json_files, desc="Processing files"):
        if n_beats is None:
            # Analyze whole song
            mode, tonic, confidence, num_notes = detect_key_from_json(
                json_path, min_notes
            )
            all_results.append(
                {
                    "file": json_path.name,
                    "mode": mode,
                    "tonic": tonic,
                    "confidence": confidence,
                    "num_notes": num_notes,
                    "segment_idx": None,
                }
            )
        else:
            # Analyze segments
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                tracks = data.get("tracks", [])

                # Calculate total beats
                max_time = 0
                for track in tracks:
                    for n in track.get("notes", []):
                        time = n.get("time", 0)
                        if time > max_time:
                            max_time = time

                resolution = 12  # Assuming 12 ticks per beat
                total_beats = (max_time // resolution) + 1

                # Process segments
                for start_beat in range(0, total_beats, n_beats):
                    if start_beat + n_beats > total_beats:
                        break  # Skip incomplete segments

                    mode, tonic, confidence, num_notes = detect_key_from_segment(
                        tracks, start_beat, n_beats, resolution, min_notes
                    )

                    all_results.append(
                        {
                            "file": json_path.name,
                            "mode": mode,
                            "tonic": tonic,
                            "confidence": confidence,
                            "num_notes": num_notes,
                            "segment_idx": start_beat // n_beats,
                        }
                    )

            except Exception as e:
                logging.warning(f"Failed to process {json_path.name}: {e}")
                continue

    # Analyze results
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print("=" * 70)
    print(f"Total segments analyzed: {len(all_results):,}")

    # Count by mode
    mode_counts = defaultdict(int)
    for result in all_results:
        mode_counts[result["mode"]] += 1

    major_count = mode_counts["major"]
    minor_count = mode_counts["minor"]
    unknown_count = mode_counts["unknown"]
    total = len(all_results)

    print(f"\nMode Distribution:")
    print(f"  Major:   {major_count:,} ({100*major_count/total:.1f}%)")
    print(f"  Minor:   {minor_count:,} ({100*minor_count/total:.1f}%)")
    print(f"  Unknown: {unknown_count:,} ({100*unknown_count/total:.1f}%)")

    # Confidence distribution
    confidences = [r["confidence"] for r in all_results if r["mode"] != "unknown"]
    if confidences:
        print(f"\nConfidence Scores (for detected keys):")
        print(f"  Mean:   {np.mean(confidences):.3f}")
        print(f"  Median: {np.median(confidences):.3f}")
        print(f"  Std:    {np.std(confidences):.3f}")
        print(f"  Min:    {np.min(confidences):.3f}")
        print(f"  Max:    {np.max(confidences):.3f}")

        # Percentiles
        print(f"\nConfidence Percentiles:")
        for p in [10, 25, 50, 75, 90]:
            val = np.percentile(confidences, p)
            print(f"  {p}th: {val:.3f}")

    # Notes per segment
    note_counts = [r["num_notes"] for r in all_results]
    print(f"\nNotes Per Segment:")
    print(f"  Mean:   {np.mean(note_counts):.1f}")
    print(f"  Median: {np.median(note_counts):.1f}")
    print(f"  Min:    {np.min(note_counts)}")
    print(f"  Max:    {np.max(note_counts)}")

    # Analyze different confidence thresholds
    print(f"\n{'='*70}")
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 70)

    for threshold in confidence_thresholds:
        filtered = [r for r in all_results if r["confidence"] >= threshold]
        major_filtered = [r for r in filtered if r["mode"] == "major"]
        minor_filtered = [r for r in filtered if r["mode"] == "minor"]

        total_filtered = len(filtered)
        major_filtered_count = len(major_filtered)
        minor_filtered_count = len(minor_filtered)

        print(f"\nThreshold: {threshold:.1f}")
        print(
            f"  Total segments:  {total_filtered:,} ({100*total_filtered/total:.1f}% of all)"
        )
        print(
            f"  Major:           {major_filtered_count:,} ({100*major_filtered_count/total_filtered:.1f}%)"
            if total_filtered > 0
            else "  Major: 0"
        )
        print(
            f"  Minor:           {minor_filtered_count:,} ({100*minor_filtered_count/total_filtered:.1f}%)"
            if total_filtered > 0
            else "  Minor: 0"
        )

        if total_filtered > 0:
            ratio = (
                major_filtered_count / minor_filtered_count
                if minor_filtered_count > 0
                else float("inf")
            )
            print(f"  Major/Minor ratio: {ratio:.2f}")

    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Find best threshold for balance
    best_threshold = None
    best_balance = float("inf")

    for threshold in confidence_thresholds:
        filtered = [r for r in all_results if r["confidence"] >= threshold]
        major_filtered_count = sum(1 for r in filtered if r["mode"] == "major")
        minor_filtered_count = sum(1 for r in filtered if r["mode"] == "minor")

        if minor_filtered_count > 0 and major_filtered_count > 0:
            ratio = major_filtered_count / minor_filtered_count
            imbalance = abs(ratio - 1.0)

            if (
                imbalance < best_balance and len(filtered) > 100
            ):  # Need at least 100 samples
                best_balance = imbalance
                best_threshold = threshold

    if best_threshold:
        filtered = [r for r in all_results if r["confidence"] >= best_threshold]
        major_filtered_count = sum(1 for r in filtered if r["mode"] == "major")
        minor_filtered_count = sum(1 for r in filtered if r["mode"] == "minor")

        print(f"\nRecommended confidence_threshold: {best_threshold:.1f}")
        print(f"  → Major samples:  {major_filtered_count:,}")
        print(f"  → Minor samples:  {minor_filtered_count:,}")
        print(f"  → Total samples:  {len(filtered):,}")
        print(f"  → Balance ratio:  {major_filtered_count/minor_filtered_count:.2f}:1")
        print(f"\nThis provides the best balance while maintaining sufficient samples.")
    else:
        print("\nWARNING: Could not find suitable threshold with balanced data.")
        print("Consider:")
        print("  1. Lower minimum notes requirement")
        print("  2. Use longer segments (more beats)")
        print("  3. Accept imbalanced data and use weighted sampling")

    # Top tonics
    print(f"\n{'='*70}")
    print("MOST COMMON TONICS")
    print("=" * 70)

    tonic_counts = defaultdict(lambda: {"major": 0, "minor": 0})
    for result in all_results:
        if result["mode"] in ["major", "minor"]:
            tonic_counts[result["tonic"]][result["mode"]] += 1

    # Sort by total count
    sorted_tonics = sorted(
        tonic_counts.items(), key=lambda x: x[1]["major"] + x[1]["minor"], reverse=True
    )[:10]

    print("\nTop 10 tonics:")
    for tonic, counts in sorted_tonics:
        total = counts["major"] + counts["minor"]
        print(
            f"  {tonic:>3}: {total:>4} ({counts['major']:>3} major, {counts['minor']:>3} minor)"
        )

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    # Save detailed results
    output_dir = pathlib.Path("steering_interventions/modality/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "modality_distribution_analysis.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "config": {
                    "n_beats": n_beats,
                    "sample_size": sample_size,
                    "min_notes": min_notes,
                    "use_train_split": use_train_split,
                },
                "summary": {
                    "total_segments": len(all_results),
                    "major_count": major_count,
                    "minor_count": minor_count,
                    "unknown_count": unknown_count,
                    "recommended_threshold": best_threshold,
                },
                "detailed_results": all_results[:100],  # Save first 100 for inspection
            },
            f,
            indent=2,
        )

    print(f"\nDetailed results saved to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze modality distribution")
    parser.add_argument(
        "--n_beats",
        type=int,
        default=16,
        help="Segment length in beats (None = whole songs). Use 0 for whole songs.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=500,
        help="Number of files to sample (None = all). Use 0 for all files.",
    )
    parser.add_argument(
        "--confidence_thresholds",
        type=str,
        default="0.5,0.6,0.7,0.8",
        help="Comma-separated confidence thresholds to test",
    )
    parser.add_argument(
        "--min_notes",
        type=int,
        default=10,
        help="Minimum notes required for key detection",
    )
    parser.add_argument(
        "--no_train_split",
        action="store_true",
        help="Use all files instead of train split only",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,  # Only show warnings/errors during processing
        format="%(levelname)s - %(message)s",
    )

    # Parse arguments
    n_beats = None if args.n_beats == 0 else args.n_beats
    sample_size = None if args.sample_size == 0 else args.sample_size
    confidence_thresholds = [
        float(t.strip()) for t in args.confidence_thresholds.split(",")
    ]

    # Check music21
    if not MUSIC21_AVAILABLE:
        print("\nERROR: music21 is not installed!")
        print("Install with: pip install music21")
        sys.exit(1)

    analyze_distribution(
        n_beats=n_beats,
        sample_size=sample_size,
        confidence_thresholds=confidence_thresholds,
        use_train_split=not args.no_train_split,
        min_notes=args.min_notes,
    )


if __name__ == "__main__":
    main()
