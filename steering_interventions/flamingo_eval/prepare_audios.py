#!/usr/bin/env python3
"""Step 1: Prepare Audio Files for Music Flamingo Evaluation.

This script:
1. Scans all experiment directories (single/dual × unconditional/conditional)
2. Selects best samples per alpha combination via degradation calculation
3. Converts selected .npy files to .wav audio
4. Creates manifest.json with metadata for evaluation

Workflow:
- Category I (Unconditional Single): 7 alphas × 2 concepts = 14 files
- Category II (Unconditional Dual): 4 scenarios + baseline = ~37 files
- Category III (Conditional Single): 2 concepts × 2 directions × ~4 alphas × ~5 songs = ~80 files  
- Category IV (Conditional Dual): Use pre-selected listening_samples (~20 files)

Total: ~151 WAV files

Usage:
    python steering_interventions/flamingo_eval/1_prepare_audio.py \\
        --config steering_interventions/flamingo_eval/config.yaml \\
        --force  # Overwrite existing files
"""

import argparse
import json
import logging
import pathlib
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from tqdm import tqdm

# Add paths for imports
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))

import representation


def load_config(config_path: pathlib.Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_degradation(metrics: dict, ground_truth: dict) -> float:
    """Calculate quality degradation from ground truth metrics.

    Args:
        metrics: Sample quality metrics
        ground_truth: Ground truth metrics from config

    Returns:
        Total degradation score (lower is better)
    """
    entropy_diff = abs(
        metrics.get("pitch_class_entropy", ground_truth["pitch_class_entropy"])
        - ground_truth["pitch_class_entropy"]
    )

    scale_diff = max(
        0,
        ground_truth["scale_consistency"]
        - metrics.get("scale_consistency", ground_truth["scale_consistency"]),
    )

    groove_diff = max(
        0,
        ground_truth["groove_consistency"]
        - metrics.get("groove_consistency", ground_truth["groove_consistency"]),
    )

    return entropy_diff + scale_diff + groove_diff


def convert_npy_to_wav(
    npy_path: pathlib.Path, wav_path: pathlib.Path, encoding: dict, force: bool = False
) -> bool:
    """Convert .npy token file to .wav audio.

    Args:
        npy_path: Path to .npy file
        wav_path: Output .wav path
        encoding: Encoding dictionary
        force: Overwrite if exists

    Returns:
        True if successful
    """
    if wav_path.exists() and not force:
        return True  # Already exists

    try:
        # Load tokens
        tokens = np.load(npy_path)

        # Decode to music
        music = representation.decode(tokens, encoding)

        # Create output directory
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        # Write audio
        music.write_audio(str(wav_path))

        return True

    except Exception as e:
        logging.error(f"Error converting {npy_path.name}: {e}")
        return False


def process_category_1_unconditional_single(
    data_root: pathlib.Path,
    output_root: pathlib.Path,
    encoding: dict,
    config: dict,
    force: bool = False,
) -> List[dict]:
    """Process Category I: Unconditional Single Steering.

    Returns:
        List of manifest entries
    """
    logging.info("\n" + "=" * 70)
    logging.info("CATEGORY I: Unconditional Single Steering")
    logging.info("=" * 70)

    manifest = []
    ground_truth = config["ground_truth"]
    alphas = config["categories"]["category_1"]["alphas"]

    for concept in ["pitch", "duration"]:
        logging.info(f"\nProcessing concept: {concept}")

        # Path to concept directory
        concept_dir = data_root / "single" / "unconditional" / f"average_{concept}"

        if not concept_dir.exists():
            logging.warning(f"Directory not found: {concept_dir}")
            continue

        # Load sample metrics
        metrics_file = concept_dir / "sample_metrics.json"
        if not metrics_file.exists():
            logging.warning(f"Metrics file not found: {metrics_file}")
            continue

        with open(metrics_file, "r") as f:
            all_samples = json.load(f)

        # Group by alpha
        by_alpha = defaultdict(list)
        for sample in all_samples:
            alpha = sample["alpha"]
            by_alpha[alpha].append(sample)

        # Select best sample for each alpha
        for alpha in alphas:
            if alpha not in by_alpha:
                logging.warning(f"Alpha {alpha} not found for {concept}")
                continue

            samples = by_alpha[alpha]

            # Calculate degradation for each sample
            for sample in samples:
                sample["degradation"] = calculate_degradation(sample, ground_truth)

            # Select best (lowest degradation)
            best_sample = min(samples, key=lambda x: x["degradation"])

            # Convert to WAV
            npy_path = pathlib.Path(best_sample["filepath"])
            wav_filename = f"alpha_{alpha:+.1f}_best.wav"
            wav_path = (
                output_root
                / "prepared_audio"
                / "category_1_unconditional_single"
                / concept
                / wav_filename
            )

            success = convert_npy_to_wav(npy_path, wav_path, encoding, force)

            if success:
                manifest_entry = {
                    "id": f"cat1_{concept}_alpha{alpha:+.1f}",
                    "category": "unconditional_single",
                    "concept": concept,
                    "alpha": alpha,
                    "wav_path": str(wav_path.relative_to(output_root)),
                    "npy_path": str(npy_path),
                    "sample_num": best_sample.get("sample_num", 0),
                    "degradation": best_sample["degradation"],
                    "metrics": {
                        "mean_pitch": best_sample.get("mean_pitch"),
                        "mean_duration": best_sample.get("mean_duration"),
                        "pitch_class_entropy": best_sample.get("pitch_class_entropy"),
                        "scale_consistency": best_sample.get("scale_consistency"),
                        "groove_consistency": best_sample.get("groove_consistency"),
                    },
                }
                manifest.append(manifest_entry)
                logging.info(
                    f"  ✓ {concept} α={alpha:+.1f}: degradation={best_sample['degradation']:.3f}"
                )
            else:
                logging.error(f"  ✗ Failed to convert {concept} α={alpha}")

    logging.info(f"\nCategory I complete: {len(manifest)} files")
    return manifest


def process_category_2_unconditional_dual(
    data_root: pathlib.Path,
    output_root: pathlib.Path,
    encoding: dict,
    config: dict,
    force: bool = False,
) -> List[dict]:
    """Process Category II: Unconditional Dual Steering.

    Returns:
        List of manifest entries
    """
    logging.info("\n" + "=" * 70)
    logging.info("CATEGORY II: Unconditional Dual Steering")
    logging.info("=" * 70)

    manifest = []
    ground_truth = config["ground_truth"]
    baseline_sample_id = config["sample_selection"]["baseline_sample"]  # "num_0"

    scenarios = ["low_long", "low_short", "high_long", "high_short"]

    for scenario in scenarios:
        logging.info(f"\nProcessing scenario: {scenario}")

        scenario_dir = data_root / "dual" / "unconditional" / scenario

        if not scenario_dir.exists():
            logging.warning(f"Directory not found: {scenario_dir}")
            continue

        # Load all sample_metrics_p*.json files
        all_samples = []
        for metrics_file in scenario_dir.glob("sample_metrics_p*.json"):
            with open(metrics_file, "r") as f:
                samples = json.load(f)
                all_samples.extend(samples)

        if not all_samples:
            logging.warning(f"No sample metrics found in {scenario_dir}")
            continue

        # Group by (alpha_pitch, alpha_duration)
        by_alpha = defaultdict(list)
        for sample in all_samples:
            key = (sample["alpha_pitch"], sample["alpha_duration"])
            by_alpha[key].append(sample)

        # Select best sample for each alpha combo
        for (alpha_p, alpha_d), samples in by_alpha.items():
            # Calculate degradation
            for sample in samples:
                sample["degradation"] = calculate_degradation(sample, ground_truth)

            # Select best
            best_sample = min(samples, key=lambda x: x["degradation"])

            # Convert to WAV
            npy_path = pathlib.Path(best_sample["filepath"])
            wav_filename = f"alpha_p{alpha_p:+.1f}_d{alpha_d:+.1f}_best.wav"
            wav_path = (
                output_root
                / "prepared_audio"
                / "category_2_unconditional_dual"
                / scenario
                / wav_filename
            )

            success = convert_npy_to_wav(npy_path, wav_path, encoding, force)

            if success:
                manifest_entry = {
                    "id": f"cat2_{scenario}_p{alpha_p:+.1f}_d{alpha_d:+.1f}",
                    "category": "unconditional_dual",
                    "scenario": scenario,
                    "alpha_pitch": alpha_p,
                    "alpha_duration": alpha_d,
                    "wav_path": str(wav_path.relative_to(output_root)),
                    "npy_path": str(npy_path),
                    "sample_num": best_sample.get("sample_num", 0),
                    "degradation": best_sample["degradation"],
                    "metrics": {
                        "mean_pitch": best_sample.get("mean_pitch"),
                        "mean_duration": best_sample.get("mean_duration"),
                        "pitch_class_entropy": best_sample.get("pitch_class_entropy"),
                        "scale_consistency": best_sample.get("scale_consistency"),
                        "groove_consistency": best_sample.get("groove_consistency"),
                    },
                }
                manifest.append(manifest_entry)
                logging.info(
                    f"  ✓ p={alpha_p:+.1f}, d={alpha_d:+.1f}: degradation={best_sample['degradation']:.3f}"
                )

    # Process baseline from neutral folder (p0.0_d0.0 only)
    logging.info("\nProcessing baseline (neutral folder)")
    neutral_dir = data_root / "dual" / "unconditional" / "neutral"

    if neutral_dir.exists():
        # Find baseline samples (p0.0_d0.0)
        baseline_pattern = f"sample_alpha_p0.0_d0.0_{baseline_sample_id}.npy"
        baseline_files = list(neutral_dir.glob(baseline_pattern))

        if baseline_files:
            npy_path = baseline_files[0]

            # Load metrics
            metrics_file = neutral_dir / "sample_metrics_p0.0_d0.0.json"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    baseline_samples = json.load(f)

                # Find matching sample
                sample_num = (
                    int(baseline_sample_id.split("_")[1])
                    if "_" in baseline_sample_id
                    else 0
                )
                best_sample = next(
                    (s for s in baseline_samples if s.get("sample_num") == sample_num),
                    baseline_samples[0],
                )
                best_sample["degradation"] = calculate_degradation(
                    best_sample, ground_truth
                )
            else:
                logging.warning(f"Baseline metrics not found: {metrics_file}")
                best_sample = {"degradation": 0.0, "sample_num": sample_num}

            # Convert to WAV
            wav_path = (
                output_root
                / "prepared_audio"
                / "category_2_unconditional_dual"
                / "baseline"
                / "alpha_p0.0_d0.0_baseline.wav"
            )

            success = convert_npy_to_wav(npy_path, wav_path, encoding, force)

            if success:
                manifest_entry = {
                    "id": "cat2_baseline_p0.0_d0.0",
                    "category": "unconditional_dual",
                    "scenario": "baseline",
                    "alpha_pitch": 0.0,
                    "alpha_duration": 0.0,
                    "wav_path": str(wav_path.relative_to(output_root)),
                    "npy_path": str(npy_path),
                    "sample_num": best_sample.get("sample_num", 0),
                    "degradation": best_sample.get("degradation", 0.0),
                    "metrics": {
                        "mean_pitch": best_sample.get("mean_pitch"),
                        "mean_duration": best_sample.get("mean_duration"),
                        "pitch_class_entropy": best_sample.get("pitch_class_entropy"),
                        "scale_consistency": best_sample.get("scale_consistency"),
                        "groove_consistency": best_sample.get("groove_consistency"),
                    },
                }
                manifest.append(manifest_entry)
                logging.info(
                    f"  ✓ Baseline: degradation={best_sample.get('degradation', 0.0):.3f}"
                )
        else:
            logging.warning(f"Baseline file not found: {baseline_pattern}")

    logging.info(f"\nCategory II complete: {len(manifest)} files")
    return manifest


def process_category_3_conditional_single(
    data_root: pathlib.Path,
    output_root: pathlib.Path,
    encoding: dict,
    config: dict,
    force: bool = False,
) -> List[dict]:
    """Process Category III: Conditional Single Steering.

    Returns:
        List of manifest entries
    """
    logging.info("\n" + "=" * 70)
    logging.info("CATEGORY III: Conditional Single Steering")
    logging.info("=" * 70)

    manifest = []
    ground_truth = config["ground_truth"]

    # Map directory names to override types
    override_mapping = {
        "high_pitch": {"concept": "pitch", "direction": "high_to_low"},
        "low_pitch": {"concept": "pitch", "direction": "low_to_high"},
        "high_duration": {"concept": "duration", "direction": "high_to_low"},
        "low_duration": {"concept": "duration", "direction": "low_to_high"},
    }

    base_dir = data_root / "single" / "conditional" / "flamingo_exp"

    for dir_name, override_info in override_mapping.items():
        logging.info(
            f"\nProcessing {override_info['concept']} - {override_info['direction']}"
        )

        category_dir = base_dir / dir_name

        if not category_dir.exists():
            logging.warning(f"Directory not found: {category_dir}")
            continue

        # Load sample metrics
        metrics_file = category_dir / "sample_metrics.json"
        if not metrics_file.exists():
            logging.warning(f"Metrics file not found: {metrics_file}")
            continue

        with open(metrics_file, "r") as f:
            all_samples = json.load(f)

        # Group by (song_name, alpha)
        by_song_alpha = defaultdict(list)
        for sample in all_samples:
            key = (sample["song_name"], sample["alpha"])
            by_song_alpha[key].append(sample)

        # Select best sample for each (song, alpha) combo
        for (song_name, alpha), samples in by_song_alpha.items():
            # Calculate/use degradation
            for sample in samples:
                if "total_degradation" in sample:
                    sample["degradation"] = sample["total_degradation"]
                else:
                    sample["degradation"] = calculate_degradation(sample, ground_truth)

            # Select best
            best_sample = min(samples, key=lambda x: x["degradation"])

            # Convert to WAV
            npy_path = pathlib.Path(best_sample["filepath"])
            wav_filename = f"{song_name}_alpha_{alpha:+.1f}_best.wav"
            wav_path = (
                output_root
                / "prepared_audio"
                / "category_3_conditional_single"
                / override_info["concept"]
                / override_info["direction"]
                / wav_filename
            )

            success = convert_npy_to_wav(npy_path, wav_path, encoding, force)

            if success:
                manifest_entry = {
                    "id": f"cat3_{override_info['concept']}_{override_info['direction']}_{song_name}_a{alpha:+.1f}",
                    "category": "conditional_single",
                    "concept": override_info["concept"],
                    "override_direction": override_info["direction"],
                    "song_name": song_name,
                    "alpha": alpha,
                    "wav_path": str(wav_path.relative_to(output_root)),
                    "npy_path": str(npy_path),
                    "sample_num": best_sample.get("sample_num", 0),
                    "degradation": best_sample["degradation"],
                    "metrics": {
                        "initial_pitch": best_sample.get("initial_pitch"),
                        "initial_duration": best_sample.get("initial_duration"),
                        "generated_mean_pitch": best_sample.get("generated_mean_pitch"),
                        "generated_mean_duration": best_sample.get(
                            "generated_mean_duration"
                        ),
                        "pitch_change": best_sample.get("pitch_change"),
                        "duration_change": best_sample.get("duration_change"),
                    },
                }
                manifest.append(manifest_entry)
                logging.info(
                    f"  ✓ {song_name} α={alpha:+.1f}: degradation={best_sample['degradation']:.3f}"
                )

    logging.info(f"\nCategory III complete: {len(manifest)} files")
    return manifest


def process_category_4_conditional_dual(
    data_root: pathlib.Path,
    output_root: pathlib.Path,
    encoding: dict,
    config: dict,
    force: bool = False,
) -> List[dict]:
    """Process Category IV: Conditional Dual Steering.

    Uses pre-selected best samples from listening_samples directory.

    Returns:
        List of manifest entries
    """
    logging.info("\n" + "=" * 70)
    logging.info("CATEGORY IV: Conditional Dual Steering")
    logging.info("=" * 70)

    manifest = []

    # Path to listening samples (already pre-selected best examples)
    listening_dir = (
        data_root / "dual" / "conditional" / "phase4_conditioned" / "listening_samples"
    )

    if not listening_dir.exists():
        logging.warning(f"Listening samples directory not found: {listening_dir}")
        return manifest

    # Parse filenames to extract metadata
    # Format: 01_Kunstderfuge-1241_highpitchshortdurationtolowlong_ap-1.0_ad+0.8.wav
    # or: Kunstderfuge-1241_highpitchshortdurationtolowlong_BASELINE.wav

    pattern = re.compile(
        r"(?:(\d+)_)?([^_]+)_([^_]+)_(?:ap([+-]?\d+\.?\d*)_ad([+-]?\d+\.?\d*)|BASELINE)\.wav"
    )

    wav_files = sorted(listening_dir.glob("*.wav"))

    for wav_file in wav_files:
        match = pattern.match(wav_file.name)

        if not match:
            logging.warning(f"Could not parse filename: {wav_file.name}")
            continue

        rank, song_name, scenario, alpha_p, alpha_d = match.groups()

        # Check if baseline
        is_baseline = "BASELINE" in wav_file.name

        if is_baseline:
            alpha_p = 0.0
            alpha_d = 0.0
        else:
            alpha_p = float(alpha_p)
            alpha_d = float(alpha_d)

        # Convert to output location (symlink or copy)
        wav_filename = wav_file.name
        wav_path = (
            output_root
            / "prepared_audio"
            / "category_4_conditional_dual"
            / scenario
            / wav_filename
        )

        # Create directory
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy or symlink
        if not wav_path.exists() or force:
            import shutil

            shutil.copy(wav_file, wav_path)

        # Create manifest entry
        manifest_entry = {
            "id": f"cat4_{scenario}_{song_name}_p{alpha_p:+.1f}_d{alpha_d:+.1f}",
            "category": "conditional_dual",
            "scenario": scenario,
            "song_name": song_name,
            "alpha_pitch": alpha_p,
            "alpha_duration": alpha_d,
            "is_baseline": is_baseline,
            "rank": int(rank) if rank else None,
            "wav_path": str(wav_path.relative_to(output_root)),
            "source_wav_path": str(wav_file),
        }
        manifest.append(manifest_entry)

        if is_baseline:
            logging.info(f"  ✓ {song_name} - BASELINE")
        else:
            logging.info(f"  ✓ {song_name} p={alpha_p:+.1f}, d={alpha_d:+.1f}")

    logging.info(f"\nCategory IV complete: {len(manifest)} files")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Prepare audio files for Music Flamingo evaluation"
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=pathlib.Path("steering_interventions/flamingo_eval/config.yaml"),
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing WAV files",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["1", "2", "3", "4"],
        default=["1", "2", "3", "4"],
        help="Which categories to process (default: all)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup logging
    log_level = getattr(logging, config["logging"]["level"])
    handlers = [logging.StreamHandler()]

    if config["logging"]["save_to_file"]:
        log_file = pathlib.Path(config["logging"]["log_file"])
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    logging.info("=" * 70)
    logging.info("Music Flamingo Evaluation - Audio Preparation")
    logging.info("=" * 70)
    logging.info(f"Config: {args.config}")
    logging.info(f"Force overwrite: {args.force}")
    logging.info(f"Categories: {args.categories}")

    # Setup paths
    data_root = pathlib.Path(config["paths"]["data_root"])
    output_root = pathlib.Path(config["paths"]["output_root"])
    encoding_file = pathlib.Path(config["paths"]["encoding_file"])

    logging.info(f"Data root: {data_root}")
    logging.info(f"Output root: {output_root}")

    # Load encoding
    logging.info("Loading encoding...")
    encoding = representation.load_encoding(encoding_file)

    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)

    # Process each category
    manifest = []

    if "1" in args.categories and config["categories"]["category_1"]["enabled"]:
        cat1_manifest = process_category_1_unconditional_single(
            data_root, output_root, encoding, config, args.force
        )
        manifest.extend(cat1_manifest)

    if "2" in args.categories and config["categories"]["category_2"]["enabled"]:
        cat2_manifest = process_category_2_unconditional_dual(
            data_root, output_root, encoding, config, args.force
        )
        manifest.extend(cat2_manifest)

    if "3" in args.categories and config["categories"]["category_3"]["enabled"]:
        cat3_manifest = process_category_3_conditional_single(
            data_root, output_root, encoding, config, args.force
        )
        manifest.extend(cat3_manifest)

    if "4" in args.categories and config["categories"]["category_4"]["enabled"]:
        cat4_manifest = process_category_4_conditional_dual(
            data_root, output_root, encoding, config, args.force
        )
        manifest.extend(cat4_manifest)

    # Save manifest
    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logging.info("\n" + "=" * 70)
    logging.info("SUMMARY")
    logging.info("=" * 70)
    logging.info(f"Total files prepared: {len(manifest)}")

    # Breakdown by category
    by_category = defaultdict(int)
    for entry in manifest:
        by_category[entry["category"]] += 1

    for category, count in sorted(by_category.items()):
        logging.info(f"  {category}: {count} files")

    logging.info(f"\nManifest saved to: {manifest_path}")
    logging.info("=" * 70)
    logging.info("Audio preparation complete!")
    logging.info("Next step: Run 2_evaluate_music_flamingo.py")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
