"""Step A: Dataset Curation and Segmentation.

This module:
1. Loads JSON files and calculates musical metrics (e.g., average velocity)
2. Optionally segments songs into fixed-length chunks (N beats)
3. Filters segments into high/low datasets based on thresholds
4. Maps to corresponding .npy/.csv tokenized files
"""

import argparse
import json
import logging
import pathlib
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import config


def calculate_velocity_metric(notes: List[Dict]) -> float:
    """Calculate average velocity from a list of notes.

    Args:
        notes: List of note dictionaries with 'velocity' field

    Returns:
        Average velocity across all notes
    """
    if not notes:
        return 0.0
    velocities = [note.get("velocity", 0) for note in notes]
    return np.mean(velocities)


def calculate_pitch_range_metric(notes: List[Dict]) -> float:
    """Calculate pitch range (max - min) from a list of notes.

    Args:
        notes: List of note dictionaries with 'pitch' field

    Returns:
        Pitch range in semitones
    """
    if not notes:
        return 0.0
    pitches = [note.get("pitch", 0) for note in notes]
    return max(pitches) - min(pitches)


def calculate_average_pitch_metric(notes: List[Dict]) -> float:
    """Calculate average pitch from a list of notes.

    Args:
        notes: List of note dictionaries with 'pitch' field

    Returns:
        Average pitch (MIDI note number)
    """
    if not notes:
        return 0.0
    pitches = [note.get("pitch", 0) for note in notes]
    return sum(pitches) / len(pitches)


def calculate_note_density_metric(
    notes: List[Dict], n_beats: int, resolution: int = 12
) -> float:
    """Calculate note density (notes per beat).

    Args:
        notes: List of note dictionaries
        n_beats: Number of beats in the segment
        resolution: Time resolution (ticks per beat)

    Returns:
        Average number of notes per beat
    """
    if n_beats == 0:
        return 0.0
    return len(notes) / n_beats


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
        first_segment_only: If True, only return the first segment (ignores rest of song)

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
        # Find the maximum time across all tracks
        max_time = 0
        for track in tracks:
            for note in track.get("notes", []):
                max_time = max(max_time, note.get("time", 0))

        total_beats = (max_time // resolution) + 1
        n_segments = max(1, total_beats // n_beats)

        segments = []
        # If first_segment_only is True, only process the first segment
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
                if seg_track["notes"]:  # Only include tracks with notes
                    segmented_tracks.append(seg_track)
                    all_segmented_notes.extend(seg_track["notes"])

            if all_segmented_notes:  # Only add segment if it has notes
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


def calculate_metric(segment: Dict, concept: str) -> float:
    """Calculate a metric for a segment based on the concept.

    Args:
        segment: Segment dictionary with 'all_notes'
        concept: Concept name from config.CONCEPTS

    Returns:
        Metric value
    """
    notes = segment["all_notes"]

    if concept == "velocity":
        return calculate_velocity_metric(notes)
    elif concept == "pitch_range":
        return calculate_pitch_range_metric(notes)
    elif concept == "average_pitch":
        return calculate_average_pitch_metric(notes)
    elif concept == "note_density":
        return calculate_note_density_metric(notes, segment["n_beats"])
    else:
        raise ValueError(f"Unknown concept: {concept}")


def curate_datasets(
    concept: str = "velocity",
    n_beats: int = None,
    resolution: int = 12,
    use_train_split: bool = True,
    first_segment_only: bool = False,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Main function to curate high/low datasets.

    Args:
        concept: Which concept to use (from config.CONCEPTS)
        n_beats: Segment length in beats (None = no segmentation)
        resolution: Time resolution
        use_train_split: Whether to use train split (True) or all data (False)
        first_segment_only: If True, only use the first N-beat segment from each file

    Returns:
        (high_segments, low_segments, metadata)
    """
    concept_config = config.CONCEPTS[concept]
    high_threshold = concept_config["high_threshold"]
    low_threshold = concept_config["low_threshold"]

    logging.info(f"Curating datasets for concept: {concept}")
    logging.info(f"High threshold: {high_threshold}, Low threshold: {low_threshold}")
    if n_beats:
        seg_info = f"{n_beats} beats" + (
            " (first segment only)" if first_segment_only else " (all segments)"
        )
        logging.info(f"Segmentation: {seg_info}")
    else:
        logging.info("No segmentation")

    # Get list of JSON files
    json_files = sorted(config.JSON_DIR.rglob("*.json"))

    # Filter by train split if requested
    if use_train_split:
        train_names_file = config.DATA_DIR / "train-names.txt"
        if train_names_file.exists():
            with open(train_names_file, "r") as f:
                train_names = set(line.strip() for line in f if line.strip())

            # Filter JSON files to only those in train split
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

    high_segments = []
    low_segments = []
    all_metrics = []

    for json_path in tqdm(json_files, desc="Processing files"):
        try:
            segments = load_and_segment_json(
                json_path, n_beats, resolution, first_segment_only
            )

            for segment in segments:
                metric_value = calculate_metric(segment, concept)
                all_metrics.append(metric_value)

                # Add metric to segment metadata
                segment["metric_value"] = metric_value
                segment["concept"] = concept

                # Classify into high/low
                if metric_value >= high_threshold:
                    high_segments.append(segment)
                elif metric_value <= low_threshold:
                    low_segments.append(segment)

        except Exception as e:
            logging.error(f"Error processing {json_path}: {e}")
            continue

    metadata = {
        "concept": concept,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "n_beats": n_beats,
        "resolution": resolution,
        "n_high_segments": len(high_segments),
        "n_low_segments": len(low_segments),
        "metric_stats": {
            "mean": float(np.mean(all_metrics)),
            "std": float(np.std(all_metrics)),
            "min": float(np.min(all_metrics)),
            "max": float(np.max(all_metrics)),
            "median": float(np.median(all_metrics)),
        },
    }

    logging.info(f"High segments: {len(high_segments)}")
    logging.info(f"Low segments: {len(low_segments)}")
    logging.info(f"Metric stats: {metadata['metric_stats']}")

    return high_segments, low_segments, metadata


def save_datasets(
    high_segments: List[Dict],
    low_segments: List[Dict],
    metadata: Dict,
    output_dir: pathlib.Path = None,
):
    """Save curated datasets to disk.

    Args:
        high_segments: List of high-metric segments
        low_segments: List of low-metric segments
        metadata: Metadata dictionary
        output_dir: Output directory (default: config.OUTPUT_DIR / "datasets")
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR / "datasets"

    output_dir.mkdir(parents=True, exist_ok=True)

    concept = metadata["concept"]

    # Save segment lists
    high_file = output_dir / f"high_{concept}_segments.json"
    low_file = output_dir / f"low_{concept}_segments.json"
    metadata_file = output_dir / f"{concept}_metadata.json"

    with open(high_file, "w") as f:
        json.dump(high_segments, f, indent=2)

    with open(low_file, "w") as f:
        json.dump(low_segments, f, indent=2)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Saved high segments to: {high_file}")
    logging.info(f"Saved low segments to: {low_file}")
    logging.info(f"Saved metadata to: {metadata_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Curate datasets for steering interventions"
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="velocity",
        choices=list(config.CONCEPTS.keys()),
        help="Which concept to use",
    )
    parser.add_argument(
        "--n_beats",
        type=int,
        default=config.SEGMENT_N_BEATS,
        help="Segment length in beats (0 = no segmentation)",
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
        help="Only use the first N-beat segment from each file (ignore rest of song)",
    )
    parser.add_argument(
        "--output_dir", type=pathlib.Path, default=None, help="Output directory"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    n_beats = None if args.no_segment or args.n_beats == 0 else args.n_beats

    high_segments, low_segments, metadata = curate_datasets(
        concept=args.concept,
        n_beats=n_beats,
        resolution=12,
        use_train_split=not args.use_all_data,
        first_segment_only=args.first_segment_only,
    )

    save_datasets(high_segments, low_segments, metadata, args.output_dir)

    logging.info("Dataset curation complete!")


if __name__ == "__main__":
    main()
