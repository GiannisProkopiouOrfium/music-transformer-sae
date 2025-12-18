"""Step A: Dataset Curation for Modality (Major/Minor).

This module:
1. Loads JSON files and detects musical key (major vs. minor)
2. Segments songs into fixed-length chunks (N beats)
3. Filters segments by confidence threshold
4. Balances major/minor datasets by sampling
5. Maps to corresponding .npy/.csv tokenized files
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import config

# Import music21
try:
    from music21 import note, stream

    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("ERROR: music21 not available. Install with: pip install music21")
    sys.exit(1)


def calculate_modality_metric(segment: Dict) -> Tuple[str, float]:
    """Calculate modality (major/minor) from segment.

    Args:
        segment: Segment dictionary with 'all_notes'

    Returns:
        (mode, confidence): ("major"/"minor"/"unknown", correlation_coefficient)
    """
    notes = segment["all_notes"]

    concept_config = config.CONCEPTS.get("modality", {})
    min_notes = concept_config.get("min_notes", 10)

    # Extract pitches
    pitches = [note.get("pitch") for note in notes if note.get("pitch") is not None]

    if len(pitches) < min_notes:
        return ("unknown", 0.0)

    try:
        # Create music21 stream
        s = stream.Stream()
        for p in pitches:
            s.append(note.Note(p))

        # Analyze key
        key = s.analyze("key")
        mode = key.mode  # 'major' or 'minor'
        confidence = key.correlationCoefficient  # 0.0 to 1.0

        return (mode, confidence)

    except Exception as e:
        logging.warning(f"Modality detection failed: {e}")
        return ("unknown", 0.0)


def get_segment_beats(notes: List[Dict], resolution: int = 12) -> int:
    """Calculate the number of beats covered by notes.

    Args:
        notes: List of note dictionaries with 'time' field
        resolution: Time resolution (ticks per beat)

    Returns:
        Number of beats
    """
    if not notes:
        return 0
    max_time = max(note.get("time", 0) for note in notes)
    return (max_time // resolution) + 1


def segment_track_by_beats(
    track: Dict, start_beat: int, n_beats: int, resolution: int = 12
) -> Dict:
    """Extract a segment of notes from a track within a beat range.

    Args:
        track: Track dictionary with 'notes' list
        start_beat: Starting beat index
        n_beats: Number of beats to extract
        resolution: Time resolution (ticks per beat)

    Returns:
        New track dictionary with segmented notes
    """
    start_time = start_beat * resolution
    end_time = (start_beat + n_beats) * resolution

    segmented_notes = [
        note
        for note in track.get("notes", [])
        if start_time <= note.get("time", 0) < end_time
    ]

    # Adjust times to start from 0
    for note in segmented_notes:
        note["time"] -= start_time

    return {**track, "notes": segmented_notes}


def load_and_segment_json(
    json_path: pathlib.Path,
    n_beats: int = None,
    resolution: int = 12,
    first_segment_only: bool = False,
) -> List[Dict]:
    """Load a JSON file and optionally segment it.

    Args:
        json_path: Path to JSON file
        n_beats: Number of beats per segment (None = no segmentation)
        resolution: Time resolution (ticks per beat)
        first_segment_only: If True, only return the first segment

    Returns:
        List of segments (each segment is a dict with metadata and tracks)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    tracks = data.get("tracks", [])

    if n_beats is None:
        # No segmentation - return whole song
        all_notes = []
        for track in tracks:
            all_notes.extend(track.get("notes", []))

        total_beats = get_segment_beats(all_notes, resolution)

        return [
            {
                "file_name": json_path.stem,
                "segment_idx": 0,
                "start_beat": 0,
                "n_beats": total_beats,
                "tracks": tracks,
                "all_notes": all_notes,
            }
        ]

    else:
        # Segmentation mode
        max_time = 0
        for track in tracks:
            for note in track.get("notes", []):
                max_time = max(max_time, note.get("time", 0))

        total_beats = (max_time // resolution) + 1
        n_segments = max(1, total_beats // n_beats)

        segments = []
        segment_range = range(1) if first_segment_only else range(n_segments)

        for seg_idx in segment_range:
            start_beat = seg_idx * n_beats

            # Segment each track
            segmented_tracks = []
            all_segmented_notes = []

            for track in tracks:
                seg_track = segment_track_by_beats(
                    track, start_beat, n_beats, resolution
                )
                if seg_track["notes"]:
                    segmented_tracks.append(seg_track)
                    all_segmented_notes.extend(seg_track["notes"])

            if all_segmented_notes:
                segments.append(
                    {
                        "file_name": json_path.stem,
                        "segment_idx": seg_idx,
                        "start_beat": start_beat,
                        "n_beats": n_beats,
                        "tracks": segmented_tracks,
                        "all_notes": all_segmented_notes,
                    }
                )

        return segments


def curate_modality_datasets(
    n_beats: int = 16,
    resolution: int = 12,
    confidence_threshold: float = 0.7,
    use_train_split: bool = True,
    first_segment_only: bool = False,
    balance_classes: bool = True,
    max_samples_per_class: int = None,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Curate major/minor datasets.

    Args:
        n_beats: Segment length in beats (None = no segmentation)
        resolution: Time resolution
        confidence_threshold: Minimum confidence for classification
        use_train_split: Whether to use train split only
        first_segment_only: If True, only use first segment from each file
        balance_classes: If True, sample equal numbers from each class
        max_samples_per_class: Maximum samples per class (None = use all)

    Returns:
        (major_segments, minor_segments, metadata)
    """
    concept_config = config.CONCEPTS["modality"]

    logging.info("Curating datasets for concept: modality (major/minor)")
    logging.info(f"Confidence threshold: {confidence_threshold}")
    logging.info(f"Balance classes: {balance_classes}")
    if n_beats:
        seg_info = f"{n_beats} beats" + (
            " (first segment only)" if first_segment_only else " (all segments)"
        )
        logging.info(f"Segmentation: {seg_info}")
    else:
        logging.info("No segmentation")

    # Get list of JSON files
    json_files = sorted(config.JSON_DIR.rglob("*.json"))

    # Filter by train split
    if use_train_split:
        train_names_file = config.DATA_DIR / "train-names.txt"
        if train_names_file.exists():
            with open(train_names_file, "r") as f:
                train_names = set(line.strip() for line in f if line.strip())

            json_files = [
                jf
                for jf in json_files
                if jf.relative_to(config.JSON_DIR).with_suffix("").as_posix()
                in train_names
            ]
            logging.info(f"Using train split: {len(json_files)} files")
        else:
            logging.warning(f"Train split file not found: {train_names_file}")

    logging.info(f"Processing {len(json_files)} JSON files...")

    major_segments = []
    minor_segments = []
    all_confidences = []
    unknown_count = 0

    for json_path in tqdm(json_files, desc="Processing files"):
        try:
            segments = load_and_segment_json(
                json_path, n_beats, resolution, first_segment_only
            )

            for segment in segments:
                mode, confidence = calculate_modality_metric(segment)

                # Only use high-confidence segments
                if confidence >= confidence_threshold:
                    # Add metadata
                    segment["mode"] = mode
                    segment["confidence"] = confidence
                    segment["concept"] = "modality"

                    all_confidences.append(confidence)

                    if mode == "major":
                        major_segments.append(segment)
                    elif mode == "minor":
                        minor_segments.append(segment)
                    else:
                        unknown_count += 1
                else:
                    unknown_count += 1

        except Exception as e:
            logging.error(f"Error processing {json_path}: {e}")
            continue

    logging.info(f"Major segments (before balancing): {len(major_segments)}")
    logging.info(f"Minor segments (before balancing): {len(minor_segments)}")
    logging.info(f"Low confidence/unknown segments: {unknown_count}")

    # Balance classes if requested
    if balance_classes:
        min_class_size = min(len(major_segments), len(minor_segments))

        if max_samples_per_class:
            min_class_size = min(min_class_size, max_samples_per_class)

        # Random sample with fixed seed for reproducibility
        np.random.seed(42)

        if len(major_segments) > min_class_size:
            indices = np.random.choice(
                len(major_segments), min_class_size, replace=False
            )
            major_segments = [major_segments[i] for i in indices]

        if len(minor_segments) > min_class_size:
            indices = np.random.choice(
                len(minor_segments), min_class_size, replace=False
            )
            minor_segments = [minor_segments[i] for i in indices]

        logging.info(f"Balanced to {min_class_size} samples per class")

    metadata = {
        "concept": "modality",
        "confidence_threshold": confidence_threshold,
        "n_beats": n_beats,
        "resolution": resolution,
        "n_major_segments": len(major_segments),
        "n_minor_segments": len(minor_segments),
        "balance_classes": balance_classes,
        "confidence_stats": {
            "mean": float(np.mean(all_confidences)) if all_confidences else 0.0,
            "std": float(np.std(all_confidences)) if all_confidences else 0.0,
            "min": float(np.min(all_confidences)) if all_confidences else 0.0,
            "max": float(np.max(all_confidences)) if all_confidences else 0.0,
            "median": float(np.median(all_confidences)) if all_confidences else 0.0,
        },
    }

    logging.info(f"Final major segments: {len(major_segments)}")
    logging.info(f"Final minor segments: {len(minor_segments)}")
    logging.info(f"Confidence stats: {metadata['confidence_stats']}")

    return major_segments, minor_segments, metadata


def save_datasets(
    major_segments: List[Dict],
    minor_segments: List[Dict],
    metadata: Dict,
    output_dir: pathlib.Path = None,
):
    """Save curated datasets to disk.

    Args:
        major_segments: List of major-mode segments
        minor_segments: List of minor-mode segments
        metadata: Metadata dictionary
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = pathlib.Path("steering_interventions/modality/outputs/datasets")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save segment lists
    major_file = output_dir / "major_modality_segments.json"
    minor_file = output_dir / "minor_modality_segments.json"
    metadata_file = output_dir / "modality_metadata.json"

    with open(major_file, "w") as f:
        json.dump(major_segments, f, indent=2)

    with open(minor_file, "w") as f:
        json.dump(minor_segments, f, indent=2)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Saved major segments to: {major_file}")
    logging.info(f"Saved minor segments to: {minor_file}")
    logging.info(f"Saved metadata to: {metadata_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Curate major/minor datasets for modality steering"
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="modality",
        help="Concept name (should be 'modality')",
    )
    parser.add_argument(
        "--n_beats",
        type=int,
        default=16,
        help="Segment length in beats (0 = no segmentation)",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="Minimum confidence for classification",
    )
    parser.add_argument(
        "--no_segment",
        action="store_true",
        help="Disable segmentation (use whole songs)",
    )
    parser.add_argument(
        "--use_all_data",
        action="store_true",
        help="Use all data instead of just train split",
    )
    parser.add_argument(
        "--first_segment_only",
        action="store_true",
        help="Only use the first N-beat segment from each file",
    )
    parser.add_argument(
        "--no_balance",
        action="store_true",
        help="Don't balance classes (use all samples)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per class (None = use all available after balancing)",
    )
    parser.add_argument(
        "--output_dir", type=pathlib.Path, default=None, help="Output directory"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.concept != "modality":
        logging.warning(
            f"Concept '{args.concept}' specified, but this script is for 'modality'"
        )

    n_beats = None if args.no_segment or args.n_beats == 0 else args.n_beats

    major_segments, minor_segments, metadata = curate_modality_datasets(
        n_beats=n_beats,
        resolution=12,
        confidence_threshold=args.confidence_threshold,
        use_train_split=not args.use_all_data,
        first_segment_only=args.first_segment_only,
        balance_classes=not args.no_balance,
        max_samples_per_class=args.max_samples,
    )

    save_datasets(major_segments, minor_segments, metadata, args.output_dir)

    logging.info("Dataset curation complete!")


if __name__ == "__main__":
    main()
