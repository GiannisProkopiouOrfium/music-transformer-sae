"""
Export SAS steering results to WAV files for listening.

This script:
1. Finds baseline (λ=0) and best steered samples from SAS experiments
2. Converts them to MIDI and WAV format
3. Organizes them for easy comparison

Usage:
    python sparse_steering/export_steered_samples_to_wav.py \
        --concept average_pitch \
        --samples_dir exp/sod/sparse_steering/steered_samples/average_pitch \
        --output_dir exp/sod/sparse_steering/audio_samples
"""

import argparse
import json
import logging
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import torch

# Import from mmt
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
import representation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_encoding(encoding_path: pathlib.Path) -> dict:
    """Load encoding dictionary."""
    with open(encoding_path) as f:
        return json.load(f)


def load_token_sequences(samples_dir: pathlib.Path) -> Dict[float, List[np.ndarray]]:
    """Load all token sequences organized by lambda value.

    Returns:
        Dictionary mapping lambda -> list of token arrays
    """
    logger.info(f"Loading token sequences from {samples_dir}")

    lambda_samples = {}

    # Find all .npy files
    npy_files = list(samples_dir.glob("*.npy"))

    if not npy_files:
        logger.warning(f"No .npy files found in {samples_dir}")
        return lambda_samples

    logger.info(f"Found {len(npy_files)} .npy files")

    # Parse filenames like: sample_lambda_pos1.0_num_0.npy or sample_lambda_neg0.5_num_2.npy
    for npy_file in npy_files:
        try:
            # Extract lambda from filename
            name = npy_file.stem  # e.g., "sample_lambda_pos1.0_num_0"
            parts = name.split("_")

            # Find lambda value
            lambda_str = None
            for i, part in enumerate(parts):
                if part == "lambda" and i + 1 < len(parts):
                    lambda_str = parts[i + 1]
                    break

            if lambda_str is None:
                logger.warning(f"Could not parse lambda from {npy_file.name}")
                continue

            # Parse lambda value (e.g., "pos1.0" -> 1.0, "neg0.5" -> -0.5)
            if lambda_str.startswith("pos"):
                lambda_val = float(lambda_str[3:])
            elif lambda_str.startswith("neg"):
                lambda_val = -float(lambda_str[3:])
            else:
                logger.warning(f"Unknown lambda format: {lambda_str}")
                continue

            # Load tokens
            tokens = np.load(npy_file)

            # Add to dictionary
            if lambda_val not in lambda_samples:
                lambda_samples[lambda_val] = []
            lambda_samples[lambda_val].append(tokens)

        except Exception as e:
            logger.warning(f"Error loading {npy_file.name}: {e}")
            continue

    # Sort by lambda
    lambda_samples = dict(sorted(lambda_samples.items()))

    logger.info(f"Loaded samples for {len(lambda_samples)} lambda values:")
    for lam, samples in lambda_samples.items():
        logger.info(f"  λ={lam:+.1f}: {len(samples)} samples")

    return lambda_samples


def compute_metrics(tokens: np.ndarray, encoding: dict) -> dict:
    """Compute pitch and duration metrics for a token sequence."""
    from sparse_steering.steered_generator_sas import (
        extract_pitches_from_tokens,
        extract_durations_from_tokens,
    )

    pitches = extract_pitches_from_tokens(tokens, encoding)
    durations = extract_durations_from_tokens(tokens, encoding)

    return {
        "pitch_mean": float(np.mean(pitches)) if pitches else 0.0,
        "duration_mean": float(np.mean(durations)) if durations else 0.0,
        "n_notes": len(pitches),
    }


def select_samples_to_export(
    lambda_samples: Dict[float, List[np.ndarray]],
    encoding: dict,
    concept: str,
) -> List[Tuple[float, int, np.ndarray, str]]:
    """Select which samples to export (baseline + extremes).

    Returns:
        List of (lambda, sample_idx, tokens, description) tuples
    """
    selected = []

    # 1. Baseline (λ=0)
    if 0.0 in lambda_samples and lambda_samples[0.0]:
        tokens = lambda_samples[0.0][0]  # First sample
        metrics = compute_metrics(tokens, encoding)
        desc = f"baseline_lambda_0.0_pitch_{metrics['pitch_mean']:.1f}_dur_{metrics['duration_mean']:.1f}"
        selected.append((0.0, 0, tokens, desc))
        logger.info(
            f"✓ Baseline: λ=0.0, pitch={metrics['pitch_mean']:.1f}, duration={metrics['duration_mean']:.1f}"
        )

    # 2. Find best steering based on concept
    if concept == "average_pitch":
        # Highest pitch (should be positive lambda)
        best_lambda, best_metrics = find_extreme_sample(
            lambda_samples, encoding, "pitch", maximize=True
        )
        if best_lambda is not None:
            tokens = lambda_samples[best_lambda][0]
            desc = f"high_pitch_lambda_{best_lambda:+.1f}_pitch_{best_metrics['pitch_mean']:.1f}"
            selected.append((best_lambda, 0, tokens, desc))
            logger.info(
                f"✓ High pitch: λ={best_lambda:+.1f}, pitch={best_metrics['pitch_mean']:.1f}"
            )

        # Lowest pitch (should be negative lambda)
        best_lambda, best_metrics = find_extreme_sample(
            lambda_samples, encoding, "pitch", maximize=False
        )
        if best_lambda is not None:
            tokens = lambda_samples[best_lambda][0]
            desc = f"low_pitch_lambda_{best_lambda:+.1f}_pitch_{best_metrics['pitch_mean']:.1f}"
            selected.append((best_lambda, 0, tokens, desc))
            logger.info(
                f"✓ Low pitch: λ={best_lambda:+.1f}, pitch={best_metrics['pitch_mean']:.1f}"
            )

    elif concept == "average_duration":
        # Longest duration (should be positive lambda)
        best_lambda, best_metrics = find_extreme_sample(
            lambda_samples, encoding, "duration", maximize=True
        )
        if best_lambda is not None:
            tokens = lambda_samples[best_lambda][0]
            desc = f"long_duration_lambda_{best_lambda:+.1f}_dur_{best_metrics['duration_mean']:.1f}"
            selected.append((best_lambda, 0, tokens, desc))
            logger.info(
                f"✓ Long duration: λ={best_lambda:+.1f}, duration={best_metrics['duration_mean']:.1f}"
            )

        # Shortest duration (should be negative lambda)
        best_lambda, best_metrics = find_extreme_sample(
            lambda_samples, encoding, "duration", maximize=False
        )
        if best_lambda is not None:
            tokens = lambda_samples[best_lambda][0]
            desc = f"short_duration_lambda_{best_lambda:+.1f}_dur_{best_metrics['duration_mean']:.1f}"
            selected.append((best_lambda, 0, tokens, desc))
            logger.info(
                f"✓ Short duration: λ={best_lambda:+.1f}, duration={best_metrics['duration_mean']:.1f}"
            )

    return selected


def find_extreme_sample(
    lambda_samples: Dict[float, List[np.ndarray]],
    encoding: dict,
    metric: str,  # "pitch" or "duration"
    maximize: bool,
) -> Tuple[float, dict]:
    """Find lambda with highest/lowest average metric value.

    Returns:
        (best_lambda, metrics) or (None, None)
    """
    best_lambda = None
    best_value = -float("inf") if maximize else float("inf")
    best_metrics = None

    for lam, samples in lambda_samples.items():
        if not samples:
            continue

        # Compute average across all samples for this lambda
        all_metrics = [compute_metrics(tokens, encoding) for tokens in samples]

        if metric == "pitch":
            avg_value = np.mean([m["pitch_mean"] for m in all_metrics])
        elif metric == "duration":
            avg_value = np.mean([m["duration_mean"] for m in all_metrics])
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Check if this is better
        is_better = avg_value > best_value if maximize else avg_value < best_value

        if is_better:
            best_lambda = lam
            best_value = avg_value
            best_metrics = all_metrics[0]  # Use first sample's metrics

    return best_lambda, best_metrics


def export_to_midi_and_wav(
    tokens: np.ndarray,
    encoding: dict,
    output_dir: pathlib.Path,
    filename_base: str,
) -> Tuple[pathlib.Path, pathlib.Path]:
    """Convert tokens to MIDI and WAV files.

    Returns:
        (midi_path, wav_path)
    """
    # Decode to muspy Music
    music = representation.decode(tokens, encoding)

    # Trim to 64 bars (same as training)
    music.trim(music.resolution * 64)

    # Create output directories
    midi_dir = output_dir / "midi"
    wav_dir = output_dir / "wav"
    midi_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Save MIDI
    midi_path = midi_dir / f"{filename_base}.mid"
    music.write(midi_path)

    # Save WAV
    wav_path = wav_dir / f"{filename_base}.wav"
    music.write(wav_path)

    return midi_path, wav_path


def main():
    parser = argparse.ArgumentParser(
        description="Export SAS steering results to WAV files"
    )
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        choices=["average_pitch", "average_duration"],
        help="Concept that was steered",
    )
    parser.add_argument(
        "--samples_dir",
        type=pathlib.Path,
        default=None,
        help="Directory containing .npy token files (default: auto-detect)",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("exp/sod/sparse_steering/audio_samples"),
        help="Output directory for MIDI and WAV files",
    )
    parser.add_argument(
        "--encoding_path",
        type=pathlib.Path,
        default=pathlib.Path("data/sod/processed/notes/encoding.json"),
        help="Path to encoding JSON file",
    )
    parser.add_argument(
        "--export_all",
        action="store_true",
        help="Export all samples, not just baseline + extremes",
    )

    args = parser.parse_args()

    # Auto-detect samples directory if not provided
    if args.samples_dir is None:
        args.samples_dir = (
            pathlib.Path("exp/sod/sparse_steering/steered_samples") / args.concept
        )

    logger.info("=" * 80)
    logger.info("EXPORT SAS SAMPLES TO WAV")
    logger.info("=" * 80)
    logger.info(f"Concept: {args.concept}")
    logger.info(f"Samples directory: {args.samples_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(
        f"Export mode: {'all samples' if args.export_all else 'baseline + extremes'}"
    )

    # Check if samples directory exists
    if not args.samples_dir.exists():
        logger.error(f"Samples directory not found: {args.samples_dir}")
        logger.error("Please run steered_generator_sas.py first to generate samples")
        return

    # Check if encoding file exists
    if not args.encoding_path.exists():
        logger.error(f"Encoding file not found: {args.encoding_path}")
        logger.error("Common locations:")
        logger.error("  - data/sod/processed/notes/encoding.json")
        logger.error("  - baseline/encoding_remi.json")
        logger.error("  - mmt/encoding.json")
        return

    # Load encoding
    logger.info(f"\nLoading encoding from {args.encoding_path}")
    encoding = load_encoding(args.encoding_path)

    # Load all token sequences
    lambda_samples = load_token_sequences(args.samples_dir)

    if not lambda_samples:
        logger.error("No samples found!")
        return

    # Select samples to export
    logger.info("\n" + "=" * 80)
    logger.info("SELECTING SAMPLES")
    logger.info("=" * 80)

    if args.export_all:
        # Export all samples
        selected = []
        for lam, samples in lambda_samples.items():
            for i, tokens in enumerate(samples):
                metrics = compute_metrics(tokens, encoding)
                desc = f"lambda_{lam:+.1f}_sample_{i}_pitch_{metrics['pitch_mean']:.1f}_dur_{metrics['duration_mean']:.1f}"
                selected.append((lam, i, tokens, desc))
        logger.info(f"Selected all {len(selected)} samples")
    else:
        # Export baseline + extremes
        selected = select_samples_to_export(lambda_samples, encoding, args.concept)
        logger.info(f"Selected {len(selected)} samples (baseline + extremes)")

    # Export samples
    logger.info("\n" + "=" * 80)
    logger.info("EXPORTING TO MIDI AND WAV")
    logger.info("=" * 80)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    exported_files = []

    for lam, sample_idx, tokens, description in selected:
        logger.info(f"\nExporting: {description}")

        try:
            midi_path, wav_path = export_to_midi_and_wav(
                tokens, encoding, args.output_dir, description
            )

            logger.info(f"  ✓ MIDI: {midi_path}")
            logger.info(f"  ✓ WAV:  {wav_path}")

            exported_files.append(
                {
                    "lambda": lam,
                    "sample_idx": sample_idx,
                    "description": description,
                    "midi": str(midi_path),
                    "wav": str(wav_path),
                }
            )

        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            continue

    # Save manifest
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "concept": args.concept,
                "samples_dir": str(args.samples_dir),
                "n_exported": len(exported_files),
                "files": exported_files,
            },
            f,
            indent=2,
        )

    logger.info("\n" + "=" * 80)
    logger.info("✓ EXPORT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Exported {len(exported_files)} samples")
    logger.info(f"MIDI files: {args.output_dir / 'midi'}")
    logger.info(f"WAV files:  {args.output_dir / 'wav'}")
    logger.info(f"Manifest:   {manifest_path}")
    logger.info(f"\nYou can now listen to the WAV files to hear the steering effect!")


if __name__ == "__main__":
    main()
