#!/usr/bin/env python3
"""Phase 4: Conditioned Dual-Steering Testing for Pitch + Duration.

Tests dual-steering ability to override strong conditioning context from extreme songs.

Test scenarios (ALL FIGHT BOTH CONCEPTS):
1. Low pitch + short duration → steer high + long (fight both)
2. High pitch + long duration → steer low + short (fight both)
3. Low pitch + long duration → steer high + short (fight both with opposite)
4. High pitch + short duration → steer low + long (fight both with opposite)

For each scenario:
- Find extreme conditioning songs (extreme pitch and duration combinations)
- Extract first N beats as conditioning
- Generate with various alpha combinations
- Measure steering success vs baseline
- Calculate quality degradation

Output:
- conditioned_results.json: Full metrics per config
- listening_list.json: Best context-fighting examples ranked by audibility + quality
- Generated MIDI/WAV files

Usage:
    python steering_interventions/dual_steering/test_multi_conditioned.py \\
        --model_checkpoint exp/sod/checkpoints/best_model.pt \\
        --pitch_vectors outputs/steering_vectors/average_pitch_steering_vectors.pt \\
        --duration_vectors outputs/steering_vectors/average_duration_steering_vectors.pt \\
        --output_dir steering_interventions/dual_steering/outputs/phase4_conditioned \\
        --n_songs 5 \\
        --conditioning_beats 8 \\
        --strategies gram_schmidt_pitch
"""

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import config
import music_x_transformers
import muspy
import representation
import utils

# No music21 needed - using token-based duration measurement

# Ground truth metrics from paper
GROUND_TRUTH_METRICS = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}


def load_song_tokens(filepath: pathlib.Path, encoding: Dict) -> np.ndarray:
    """Load song tokens from .npy file.

    Args:
        filepath: Path to .npy file in subfolder
        encoding: Encoding dictionary

    Returns:
        Token array (seq_len, 6)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load notes in 5D format
    notes = np.load(filepath)

    if len(notes.shape) != 2 or notes.shape[1] != 5:
        raise ValueError(f"Invalid shape {notes.shape}, expected (seq_len, 5)")

    # Convert to 6D codes
    codes = representation.encode_notes(notes, encoding)
    return codes


def extract_pitches_from_tokens(tokens: np.ndarray, encoding: dict) -> list:
    """Extract pitch values from tokens."""
    try:
        music = representation.decode(tokens, encoding)
        pitches = []
        for track in music.tracks:
            for note in track.notes:
                pitches.append(note.pitch)
        return pitches
    except Exception as e:
        logging.error(f"Error extracting pitches: {e}")
        return []


def measure_duration_from_tokens(tokens: np.ndarray, encoding: dict) -> Dict:
    """Measure duration statistics from tokens (token[4] = duration in ticks).

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        Dictionary with duration statistics
    """
    try:
        note_type = encoding["type_code_map"]["note"]
        durations = []

        for token in tokens:
            if token[0] == note_type:
                duration_value = token[4]
                durations.append(duration_value)

        if len(durations) < 10:
            return {
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "n_notes": len(durations),
            }

        return {
            "mean": float(np.mean(durations)),
            "std": float(np.std(durations)),
            "min": float(np.min(durations)),
            "max": float(np.max(durations)),
            "n_notes": len(durations),
        }
    except Exception as e:
        logging.error(f"Error measuring duration: {e}")
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "n_notes": 0,
        }


# Key detection removed - using duration measurement instead


def evaluate_quality_metrics(tokens: np.ndarray, encoding: dict) -> Dict:
    """Evaluate objective quality metrics."""
    try:
        music = representation.decode(tokens, encoding)
        music.trim(music.resolution * 64)

        if not music.tracks:
            return {
                "pitch_class_entropy": np.nan,
                "scale_consistency": np.nan,
                "groove_consistency": np.nan,
            }

        return {
            "pitch_class_entropy": muspy.pitch_class_entropy(music),
            "scale_consistency": muspy.scale_consistency(music) * 100,
            "groove_consistency": muspy.groove_consistency(music, 4 * music.resolution)
            * 100,
        }
    except Exception as e:
        logging.error(f"Error evaluating quality: {e}")
        return {
            "pitch_class_entropy": np.nan,
            "scale_consistency": np.nan,
            "groove_consistency": np.nan,
        }


def calculate_degradation(metrics: Dict, baseline: Dict) -> Dict:
    """Calculate quality degradation from baseline."""
    entropy_diff = abs(metrics["pitch_class_entropy"] - baseline["pitch_class_entropy"])
    scale_diff = max(0, baseline["scale_consistency"] - metrics["scale_consistency"])
    groove_diff = max(0, baseline["groove_consistency"] - metrics["groove_consistency"])
    total_degradation = entropy_diff + scale_diff + groove_diff

    return {
        "entropy_diff": float(entropy_diff),
        "scale_diff": float(scale_diff),
        "groove_diff": float(groove_diff),
        "total_degradation": float(total_degradation),
    }


def find_extreme_songs(
    notes_dir: pathlib.Path,
    encoding: Dict,
    n_songs: int = 5,
    conditioning_beats: int = 8,
    pitch_high_threshold: float = 67.6,  # From config.py
    pitch_low_threshold: float = 60.0,  # From config.py
    duration_high_threshold: float = 14.5,  # From config.py - "long" notes
    duration_low_threshold: float = 6.5,  # From config.py - "short" notes
    cache_file: pathlib.Path = None,
) -> Dict[str, List[Tuple[pathlib.Path, float, float]]]:
    """Find songs with extreme pitch and duration characteristics.

    Scans all songs and classifies them into 4 categories using config thresholds:
    - Low pitch: mean_pitch < pitch_low_threshold
    - High pitch: mean_pitch > pitch_high_threshold
    - Short duration: mean_duration < duration_low_threshold
    - Long duration: mean_duration > duration_high_threshold

    Results are cached to avoid re-scanning on subsequent runs.

    Args:
        notes_dir: Directory containing .npy files
        encoding: Encoding dictionary
        n_songs: Number of songs per category
        conditioning_beats: Beats to analyze for classification
        pitch_high_threshold: Threshold for high pitch (from config.py)
        pitch_low_threshold: Threshold for low pitch (from config.py)
        duration_high_threshold: Threshold for long duration (from config.py)
        duration_low_threshold: Threshold for short duration (from config.py)
        cache_file: Path to cache file (if None, uses notes_dir/extreme_songs_cache.json)

    Returns:
        Dictionary with 4 categories:
        - "low_pitch_short_duration": [(filepath, mean_pitch, mean_duration)]
        - "low_pitch_long_duration": [(filepath, mean_pitch, mean_duration)]
        - "high_pitch_short_duration": [(filepath, mean_pitch, mean_duration)]
        - "high_pitch_long_duration": [(filepath, mean_pitch, mean_duration)]
    """
    # Setup cache file
    if cache_file is None:
        cache_file = notes_dir / "extreme_songs_duration_cache.json"

    # Try to load from cache
    if cache_file.exists():
        logging.info(f"Loading cached extreme songs from {cache_file}...")
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)

            # Verify cache parameters match
            if (
                cached_data["conditioning_beats"] == conditioning_beats
                and cached_data["pitch_high_threshold"] == pitch_high_threshold
                and cached_data["pitch_low_threshold"] == pitch_low_threshold
                and cached_data["duration_high_threshold"] == duration_high_threshold
                and cached_data["duration_low_threshold"] == duration_low_threshold
            ):
                # Convert cached data back to proper format
                result = {}
                for category, songs in cached_data["extreme_songs"].items():
                    result[category] = [
                        (pathlib.Path(path), pitch, duration)
                        for path, pitch, duration in songs[:n_songs]
                    ]

                logging.info("✅ Loaded from cache successfully")
                for category, songs in result.items():
                    if songs:
                        pitches = [p for _, p, _ in songs]
                        durations = [d for _, _, d in songs]
                        logging.info(f"  {category.upper()}: {len(songs)} songs")
                        logging.info(
                            f"    Pitch range: {min(pitches):.1f} - {max(pitches):.1f}"
                        )
                        logging.info(
                            f"    Duration range: {min(durations):.1f} - {max(durations):.1f}"
                        )
                return result
            else:
                logging.info("Cache parameters mismatch, re-scanning...")
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}, re-scanning...")

    # Cache miss or invalid - scan all songs
    logging.info(f"Scanning {notes_dir} for songs with extreme characteristics...")

    candidates = {
        "low_pitch_short_duration": [],
        "low_pitch_long_duration": [],
        "high_pitch_short_duration": [],
        "high_pitch_long_duration": [],
    }

    # Scan for .npy files
    scanned_count = 0
    for subfolder in notes_dir.iterdir():
        if not subfolder.is_dir():
            continue

        for filepath in subfolder.glob("*.npy"):
            try:
                scanned_count += 1
                if scanned_count % 100 == 0:
                    logging.info(f"  Scanned {scanned_count} songs...")

                tokens = load_song_tokens(filepath, encoding)

                # Extract first N beats
                max_beat = conditioning_beats
                cond_len = 0
                for i, token in enumerate(tokens):
                    if token[1] >= max_beat:
                        cond_len = i
                        break
                if cond_len == 0:
                    cond_len = min(len(tokens), 256)

                cond_tokens = tokens[:cond_len]

                # Get pitch
                pitches = extract_pitches_from_tokens(cond_tokens, encoding)
                if len(pitches) < 10:
                    continue
                mean_pitch = float(np.mean(pitches))

                # Get duration
                duration_stats = measure_duration_from_tokens(cond_tokens, encoding)
                if duration_stats["n_notes"] < 10:
                    continue
                mean_duration = duration_stats["mean"]

                # Classify into categories using thresholds
                # Low pitch + short duration
                if (
                    mean_pitch < pitch_low_threshold
                    and mean_duration < duration_low_threshold
                ):
                    candidates["low_pitch_short_duration"].append(
                        (filepath, mean_pitch, mean_duration)
                    )

                # Low pitch + long duration
                if (
                    mean_pitch < pitch_low_threshold
                    and mean_duration > duration_high_threshold
                ):
                    candidates["low_pitch_long_duration"].append(
                        (filepath, mean_pitch, mean_duration)
                    )

                # High pitch + short duration
                if (
                    mean_pitch > pitch_high_threshold
                    and mean_duration < duration_low_threshold
                ):
                    candidates["high_pitch_short_duration"].append(
                        (filepath, mean_pitch, mean_duration)
                    )

                # High pitch + long duration
                if (
                    mean_pitch > pitch_high_threshold
                    and mean_duration > duration_high_threshold
                ):
                    candidates["high_pitch_long_duration"].append(
                        (filepath, mean_pitch, mean_duration)
                    )

            except Exception as e:
                logging.debug(f"Error processing {filepath.name}: {e}")

    logging.info(f"  Scanned {scanned_count} songs total")

    # For each category, sort by extremeness and take top N
    # Low pitch + short duration: most extreme = lowest pitch, shortest duration
    candidates["low_pitch_short_duration"].sort(key=lambda x: (x[1], x[2]))
    candidates["low_pitch_short_duration"] = candidates["low_pitch_short_duration"][
        :n_songs
    ]

    # Low pitch + long duration: most extreme = lowest pitch, longest duration
    candidates["low_pitch_long_duration"].sort(key=lambda x: (x[1], -x[2]))
    candidates["low_pitch_long_duration"] = candidates["low_pitch_long_duration"][
        :n_songs
    ]

    # High pitch + short duration: most extreme = highest pitch, shortest duration
    candidates["high_pitch_short_duration"].sort(key=lambda x: (-x[1], x[2]))
    candidates["high_pitch_short_duration"] = candidates["high_pitch_short_duration"][
        :n_songs
    ]

    # High pitch + long duration: most extreme = highest pitch, longest duration
    candidates["high_pitch_long_duration"].sort(key=lambda x: (-x[1], -x[2]))
    candidates["high_pitch_long_duration"] = candidates["high_pitch_long_duration"][
        :n_songs
    ]

    # Log results
    for category, songs in candidates.items():
        logging.info(f"\n{category.upper()}: {len(songs)} songs")
        if songs:
            pitches = [p for _, p, _ in songs]
            durations = [d for _, _, d in songs]
            logging.info(f"  Pitch range: {min(pitches):.1f} - {max(pitches):.1f}")
            logging.info(
                f"  Duration range: {min(durations):.1f} - {max(durations):.1f} ticks"
            )

    # Save to cache for future runs
    try:
        cache_data = {
            "conditioning_beats": conditioning_beats,
            "pitch_high_threshold": pitch_high_threshold,
            "pitch_low_threshold": pitch_low_threshold,
            "duration_high_threshold": duration_high_threshold,
            "duration_low_threshold": duration_low_threshold,
            "extreme_songs": {
                category: [
                    (str(path), pitch, duration) for path, pitch, duration in songs
                ]
                for category, songs in candidates.items()
            },
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        logging.info(f"✅ Cached extreme songs to {cache_file}")
    except Exception as e:
        logging.warning(f"Failed to save cache: {e}")

    return candidates


def extract_conditioning_prefix(tokens: np.ndarray, n_beats: int = 8) -> torch.Tensor:
    """Extract first N beats as conditioning."""
    max_beat = n_beats
    cond_len = 0

    for i, token in enumerate(tokens):
        beat = token[1]
        if beat >= max_beat:
            cond_len = i
            break

    if cond_len == 0:
        cond_len = len(tokens)

    cond_tokens = tokens[:cond_len]
    cond_tensor = torch.from_numpy(cond_tokens).long().unsqueeze(0)
    return cond_tensor


def filter_alphas_for_scenario(
    scenario: str,
    alpha_pitch_list: List[float],
    alpha_duration_list: List[float],
) -> Tuple[List[float], List[float]]:
    """Filter alpha values based on scenario to test only relevant combinations.

    Rules based on validation grid findings:
    1. low_pitch_short_duration_to_high_long: positive pitch + positive duration + baseline
    2. high_pitch_long_duration_to_low_short: negative pitch + negative duration + baseline
    3. low_pitch_long_duration_to_high_short: positive pitch + negative duration + baseline
    4. high_pitch_short_duration_to_low_long: negative pitch + positive duration + baseline

    Args:
        scenario: Scenario name
        alpha_pitch_list: Full list of pitch alphas
        alpha_duration_list: Full list of duration alphas

    Returns:
        (filtered_pitch_alphas, filtered_duration_alphas)
    """
    # Always include baseline (0.0)
    baseline = [0.0]

    if scenario == "low_pitch_short_duration_to_high_long":
        # Fight both: need positive for both
        pitch_alphas = baseline + [a for a in alpha_pitch_list if a > 0]
        duration_alphas = baseline + [a for a in alpha_duration_list if a > 0]

    elif scenario == "high_pitch_long_duration_to_low_short":
        # Fight both: need negative for both
        pitch_alphas = baseline + [a for a in alpha_pitch_list if a < 0]
        duration_alphas = baseline + [a for a in alpha_duration_list if a < 0]

    elif scenario == "low_pitch_long_duration_to_high_short":
        # Fight pitch (positive), fight duration opposite (negative)
        pitch_alphas = baseline + [a for a in alpha_pitch_list if a > 0]
        duration_alphas = baseline + [a for a in alpha_duration_list if a < 0]

    elif scenario == "high_pitch_short_duration_to_low_long":
        # Fight pitch (negative), fight duration opposite (positive)
        pitch_alphas = baseline + [a for a in alpha_pitch_list if a < 0]
        duration_alphas = baseline + [a for a in alpha_duration_list if a > 0]

    else:
        # Fallback: use all alphas
        pitch_alphas = alpha_pitch_list
        duration_alphas = alpha_duration_list

    # Remove duplicates and sort
    pitch_alphas = sorted(set(pitch_alphas))
    duration_alphas = sorted(set(duration_alphas))

    return pitch_alphas, duration_alphas


def conditioned_generate_and_evaluate(
    model,
    composer,
    strategy: str,
    song_list: List[Tuple[pathlib.Path, float, float]],
    scenario: str,
    alpha_pitch_list: List[float],
    alpha_duration_list: List[float],
    encoding: dict,
    device: torch.device,
    conditioning_beats: int = 8,
    continuation_len: int = 512,
    output_dir: pathlib.Path = None,
) -> List[Dict]:
    """Generate conditioned continuations with dual steering.

    Args:
        model: Transformer model
        composer: VectorComposer instance
        strategy: Composition strategy (e.g., "gram_schmidt_pitch")
        song_list: List of (filepath, mean_pitch, mean_duration)
        scenario: Scenario name (e.g., "low_pitch_short_duration_to_high_long")
        alpha_pitch_list: Pitch alpha values to test
        alpha_duration_list: Duration alpha values to test
        encoding: Encoding dictionary
        device: Torch device
        conditioning_beats: Beats for conditioning
        continuation_len: Target sequence length
        output_dir: Where to save results

    Returns:
        List of result dictionaries
    """
    from multi_steered_generator import MultiSteeringGenerator

    results = []
    eos = encoding["type_code_map"]["end-of-song"]

    # Filter alphas based on scenario to test only relevant combinations
    alpha_pitch_list, alpha_duration_list = filter_alphas_for_scenario(
        scenario, alpha_pitch_list, alpha_duration_list
    )

    logging.info(f"Testing {len(alpha_pitch_list)} pitch alphas: {alpha_pitch_list}")
    logging.info(
        f"Testing {len(alpha_duration_list)} duration alphas: {alpha_duration_list}"
    )
    logging.info(
        f"Total combinations: {len(alpha_pitch_list) * len(alpha_duration_list)}"
    )

    # Calculate total iterations for progress bar
    total_iterations = len(song_list) * len(alpha_pitch_list) * len(alpha_duration_list)
    pbar = tqdm(total=total_iterations, desc=f"{scenario}", unit="generation")

    for song_idx, (filepath, cond_pitch, cond_duration) in enumerate(song_list):
        logging.info(f"\n{'='*70}")
        logging.info(
            f"Song {song_idx+1}/{len(song_list)}: {filepath.name} | Scenario: {scenario}"
        )
        logging.info(
            f"Conditioning: pitch={cond_pitch:.1f}, duration={cond_duration:.1f} ticks"
        )
        logging.info(f"{'='*70}")

        # Load and extract conditioning
        tokens = load_song_tokens(filepath, encoding)
        conditioning = extract_conditioning_prefix(tokens, conditioning_beats)
        conditioning = conditioning.to(device)

        logging.info(
            f"Conditioning: {conditioning.shape[1]} tokens ({conditioning_beats} beats)"
        )

        # Test alpha combinations
        for alpha_pitch in alpha_pitch_list:
            for alpha_duration in alpha_duration_list:
                try:
                    # Check if already generated (for resuming)
                    if output_dir is not None:
                        save_dir = (
                            output_dir
                            / scenario
                            / strategy
                            / f"ap{alpha_pitch:+.1f}_ad{alpha_duration:+.1f}"
                        )
                        result_marker = save_dir / f"{filepath.stem}.npy"
                        if result_marker.exists():
                            logging.info(
                                f"  α_p={alpha_pitch:+.1f}, α_d={alpha_duration:+.1f}: SKIPPING (already generated)"
                            )
                            pbar.update(1)  # Update progress bar
                            continue

                    # Create generator
                    generator = MultiSteeringGenerator(
                        model=model,
                        composer=composer,
                        alpha_pitch=alpha_pitch,
                        alpha_duration=alpha_duration,
                        strategy=strategy,
                        intervention_position="last",
                    )

                    # Generate continuation
                    output = generator.generate(
                        conditioning,
                        continuation_len,
                        eos_token=eos,
                        temperature=1.0,
                        filter_logits_fn="top_k",
                        filter_thres=0.9,
                    )

                    # Convert to numpy (generated portion only)
                    generated_tokens = output.cpu().numpy()[0]

                    # Extract pitches
                    pitches = extract_pitches_from_tokens(generated_tokens, encoding)

                    # Extract durations
                    duration_stats = measure_duration_from_tokens(
                        generated_tokens, encoding
                    )

                    if len(pitches) >= 10 and duration_stats["n_notes"] >= 10:
                        # Pitch statistics
                        gen_pitch_mean = float(np.mean(pitches))
                        gen_pitch_std = float(np.std(pitches))

                        # Duration statistics
                        gen_duration_mean = duration_stats["mean"]
                        gen_duration_std = duration_stats["std"]

                        # Quality metrics
                        quality = evaluate_quality_metrics(generated_tokens, encoding)

                        # Degradation
                        degradation = calculate_degradation(
                            quality, GROUND_TRUTH_METRICS
                        )

                        # Calculate steering success
                        pitch_change = gen_pitch_mean - cond_pitch
                        duration_change = gen_duration_mean - cond_duration

                        # Determine if steering succeeded based on scenario
                        pitch_success = False
                        duration_success = False

                        # Check pitch steering (based on directional change)
                        if "low_pitch" in scenario and "to_high" in scenario:
                            # Want to increase pitch
                            pitch_success = gen_pitch_mean > cond_pitch
                        elif "high_pitch" in scenario and "to_low" in scenario:
                            # Want to decrease pitch
                            pitch_success = gen_pitch_mean < cond_pitch

                        # Check duration steering
                        if (
                            "short_duration" in scenario
                            and "to" in scenario
                            and "long" in scenario
                        ):
                            # Want to increase duration
                            duration_success = gen_duration_mean > cond_duration
                        elif (
                            "long_duration" in scenario
                            and "to" in scenario
                            and "short" in scenario
                        ):
                            # Want to decrease duration
                            duration_success = gen_duration_mean < cond_duration

                        result = {
                            "song_name": filepath.stem,
                            "scenario": scenario,
                            "strategy": strategy,
                            "alpha_pitch": alpha_pitch,
                            "alpha_duration": alpha_duration,
                            # Conditioning
                            "conditioning_pitch": cond_pitch,
                            "conditioning_duration": cond_duration,
                            "conditioning_beats": conditioning_beats,
                            "conditioning_tokens": conditioning.shape[1],
                            # Generated
                            "generated_pitch_mean": gen_pitch_mean,
                            "generated_pitch_std": gen_pitch_std,
                            "generated_duration_mean": gen_duration_mean,
                            "generated_duration_std": gen_duration_std,
                            "generated_n_notes": len(pitches),
                            # Steering effect
                            "pitch_change": pitch_change,
                            "duration_change": duration_change,
                            "pitch_steering_success": pitch_success,
                            "duration_steering_success": duration_success,
                            "overall_success": pitch_success and duration_success,
                            # Quality
                            "quality_metrics": quality,
                            "degradation": degradation,
                        }

                        results.append(result)

                        logging.info(
                            f"  α_p={alpha_pitch:+.1f}, α_d={alpha_duration:+.1f}: "
                            f"pitch={gen_pitch_mean:.1f} (Δ{pitch_change:+.1f}), "
                            f"dur={gen_duration_mean:.1f} (Δ{duration_change:+.1f}), "
                            f"deg={degradation['total_degradation']:.2f}"
                        )

                        # Save if output_dir provided
                        if output_dir is not None:
                            save_dir = (
                                output_dir
                                / scenario
                                / strategy
                                / f"ap{alpha_pitch:+.1f}_ad{alpha_duration:+.1f}"
                            )
                            save_dir.mkdir(parents=True, exist_ok=True)

                            # Save full sequence (conditioning + generated)
                            full_seq = (
                                torch.cat((conditioning, output), 1).cpu().numpy()[0]
                            )

                            # Only save .npy (tokens) to save space
                            # MIDI/WAV can be generated later from .npy files
                            np.save(save_dir / f"{filepath.stem}.npy", full_seq)

                            # Skip MIDI/WAV generation to save space
                            # Uncomment below to generate audio files:
                            # try:
                            #     music = representation.decode(full_seq, encoding)
                            #     music.write(str(save_dir / f"{filepath.stem}.mid"))
                            #     music.write_audio(str(save_dir / f"{filepath.stem}.wav"))
                            # except Exception as e:
                            #     logging.error(f"Error saving audio: {e}")

                        pbar.update(1)  # Update progress bar

                except Exception as e:
                    logging.error(
                        f"Generation failed for α_p={alpha_pitch}, α_d={alpha_duration}: {e}"
                    )
                    pbar.update(1)  # Update progress bar even on error
                    continue

    pbar.close()  # Close progress bar
    return results


def extract_listening_list(results: List[Dict], top_n: int = 20) -> List[Dict]:
    """Extract best context-fighting examples ranked by audibility + quality.

    Ensures balanced representation from all scenarios by selecting top examples
    from each scenario proportionally.

    Scoring criteria:
    - Steering magnitude (how much did pitch/duration change?)
    - Quality preservation (low degradation)
    - Both concepts changed successfully

    Args:
        results: All generation results
        top_n: Number of examples to include

    Returns:
        Ranked list of best examples with scenario diversity
    """
    scored_results = []

    for r in results:
        if r["generated_n_notes"] < 10:
            continue

        # Exclude baseline examples (alpha 0.0 for either concept)
        # We want both concepts actively steered
        if abs(r["alpha_pitch"]) < 1e-9 or abs(r["alpha_duration"]) < 1e-9:
            continue

        # Calculate audibility score
        pitch_magnitude = abs(r["pitch_change"]) / 10.0  # Normalize by 10 semitones
        duration_magnitude = abs(r["duration_change"]) / 10.0  # Normalize by 10 ticks
        success_score = 1.0 if r["overall_success"] else 0.0

        # Calculate quality score (inverse of degradation)
        degradation = r["degradation"]["total_degradation"]
        quality_score = max(0, 1.0 - degradation / 10.0)  # Normalize by 10

        # Combined score
        audibility = (pitch_magnitude + duration_magnitude + success_score) / 3.0
        overall_score = 0.6 * audibility + 0.4 * quality_score

        scored_results.append(
            {
                **r,
                "audibility_score": audibility,
                "quality_score": quality_score,
                "listening_score": overall_score,
            }
        )

    # Group by scenario
    scenarios = {r["scenario"] for r in scored_results}
    n_scenarios = len(scenarios)

    if n_scenarios == 0:
        return []

    # Ensure each scenario gets at least top_n // n_scenarios examples
    per_scenario = max(1, top_n // n_scenarios)
    listening_list = []

    for scenario in sorted(scenarios):
        scenario_results = [r for r in scored_results if r["scenario"] == scenario]
        scenario_results.sort(key=lambda x: x["listening_score"], reverse=True)

        # Add top examples from this scenario
        listening_list.extend(scenario_results[:per_scenario])

    # Fill remaining slots with best overall examples not yet included
    remaining_slots = top_n - len(listening_list)
    if remaining_slots > 0:
        included_ids = {id(r) for r in listening_list}
        remaining_candidates = [r for r in scored_results if id(r) not in included_ids]
        remaining_candidates.sort(key=lambda x: x["listening_score"], reverse=True)
        listening_list.extend(remaining_candidates[:remaining_slots])

    # Final sort by score
    listening_list.sort(key=lambda x: x["listening_score"], reverse=True)

    return listening_list[:top_n]


def analyze_conditioned_results(results: List[Dict]) -> Dict:
    """Analyze conditioned steering results.

    Args:
        results: List of result dictionaries

    Returns:
        Analysis summary
    """
    analysis = {
        "summary_by_scenario": {},
        "summary_by_strategy": {},
        "summary_by_alpha": {},
        "best_configs": {},
    }

    # Group by scenario
    for scenario in {r["scenario"] for r in results}:
        scenario_results = [r for r in results if r["scenario"] == scenario]
        valid = [r for r in scenario_results if r["generated_n_notes"] >= 10]

        if valid:
            analysis["summary_by_scenario"][scenario] = {
                "n_samples": len(valid),
                "pitch_success_rate": np.mean(
                    [r["pitch_steering_success"] for r in valid]
                ),
                "duration_success_rate": np.mean(
                    [r["duration_steering_success"] for r in valid]
                ),
                "overall_success_rate": np.mean([r["overall_success"] for r in valid]),
                "mean_degradation": np.mean(
                    [r["degradation"]["total_degradation"] for r in valid]
                ),
                "mean_pitch_change": np.mean([r["pitch_change"] for r in valid]),
                "mean_duration_change": np.mean([r["duration_change"] for r in valid]),
            }

    # Group by strategy
    for strategy in {r["strategy"] for r in results}:
        strategy_results = [r for r in results if r["strategy"] == strategy]
        valid = [r for r in strategy_results if r["generated_n_notes"] >= 10]

        if valid:
            analysis["summary_by_strategy"][strategy] = {
                "n_samples": len(valid),
                "overall_success_rate": np.mean([r["overall_success"] for r in valid]),
                "mean_degradation": np.mean(
                    [r["degradation"]["total_degradation"] for r in valid]
                ),
            }

    # Find best alpha configs per scenario
    for scenario in {r["scenario"] for r in results}:
        scenario_results = [
            r
            for r in results
            if r["scenario"] == scenario and r["generated_n_notes"] >= 10
        ]

        if scenario_results:
            # Sort by overall success, then by low degradation
            scenario_results.sort(
                key=lambda x: (
                    x["overall_success"],
                    -x["degradation"]["total_degradation"],
                ),
                reverse=True,
            )

            best = scenario_results[0]
            analysis["best_configs"][scenario] = {
                "alpha_pitch": best["alpha_pitch"],
                "alpha_duration": best["alpha_duration"],
                "strategy": best["strategy"],
                "success_rate": best["overall_success"],
                "degradation": best["degradation"]["total_degradation"],
            }

    return analysis


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 4: Conditioned dual-steering testing"
    )
    parser.add_argument(
        "--model_checkpoint", type=pathlib.Path, default=None, help="Model checkpoint"
    )
    parser.add_argument(
        "--pitch_vectors",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/outputs/steering_vectors/average_pitch_steering_vectors.pt"
        ),
        help="Pitch steering vectors",
    )
    parser.add_argument(
        "--duration_vectors",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/outputs/steering_vectors/average_duration_steering_vectors.pt"
        ),
        help="Duration steering vectors",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/dual_steering/outputs/phase4_conditioned"
        ),
        help="Output directory",
    )
    parser.add_argument("--n_songs", type=int, default=5, help="Songs per category")
    parser.add_argument(
        "--conditioning_beats", type=int, default=8, help="Beats for conditioning"
    )
    parser.add_argument(
        "--continuation_len", type=int, default=512, help="Tokens to generate"
    )
    parser.add_argument(
        "--alphas_pitch",
        type=str,
        default="0.25,1.0,1.25,-1.25",
        help="Pitch alphas (based on validation grid findings)",
    )
    parser.add_argument(
        "--alphas_duration",
        type=str,
        default="0.75,-1.25",
        help="Duration alphas (based on validation grid findings)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["gram_schmidt_pitch"],
        help="Strategies to test (gram_schmidt_pitch recommended)",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU number")
    parser.add_argument(
        "--listening_list_size", type=int, default=20, help="Listening list size"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "phase4_conditioned.log"),
            logging.StreamHandler(),
        ],
    )

    # Setup device
    if args.gpu is not None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
            logging.info(f"Using CUDA device: GPU {args.gpu}")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA not available, using CPU")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # Load model
    logging.info("Loading model...")
    train_args = utils.load_json(config.MODEL_DIR / "train-args.json")
    encoding = representation.load_encoding(config.NOTES_DIR / "encoding.json")

    model = music_x_transformers.MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rotary_pos_emb=train_args["rel_pos_emb"],
        use_abs_pos_emb=train_args["abs_pos_emb"],
        emb_dropout=train_args["dropout"],
        attn_dropout=train_args["dropout"],
        ff_dropout=train_args["dropout"],
    ).to(device)

    if args.model_checkpoint is None:
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    else:
        checkpoint_path = args.model_checkpoint

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    logging.info("Model loaded")

    # Create composer
    logging.info("Creating vector composer...")
    from vector_composition import load_and_create_composer

    composer = load_and_create_composer(
        str(args.pitch_vectors), str(args.duration_vectors)
    )

    # Parse alphas
    alphas_pitch = [float(a.strip()) for a in args.alphas_pitch.split(",")]
    alphas_duration = [float(a.strip()) for a in args.alphas_duration.split(",")]

    logging.info(
        f"Testing {len(alphas_pitch)} pitch alphas × {len(alphas_duration)} duration alphas"
    )

    # Find extreme songs
    logging.info("\n" + "=" * 80)
    logging.info("FINDING EXTREME SONGS FOR CONDITIONING")
    logging.info("=" * 80)

    # Setup cache file path
    cache_file = args.output_dir / "extreme_songs_cache.json"

    # Load thresholds from config
    pitch_high = config.CONCEPTS["average_pitch"]["high_threshold"]
    pitch_low = config.CONCEPTS["average_pitch"]["low_threshold"]
    duration_high = config.CONCEPTS["average_duration"]["high_threshold"]
    duration_low = config.CONCEPTS["average_duration"]["low_threshold"]

    logging.info("Using thresholds from config.py:")
    logging.info(f"  Pitch: low < {pitch_low}, high > {pitch_high}")
    logging.info(f"  Duration: short < {duration_low}, long > {duration_high}")

    extreme_songs = find_extreme_songs(
        config.NOTES_DIR,
        encoding,
        args.n_songs,
        args.conditioning_beats,
        pitch_high_threshold=pitch_high,
        pitch_low_threshold=pitch_low,
        duration_high_threshold=duration_high,
        duration_low_threshold=duration_low,
        cache_file=cache_file,
    )

    # Check if we have songs for all categories
    missing_categories = [
        cat for cat, songs in extreme_songs.items() if len(songs) == 0
    ]
    if missing_categories:
        logging.error(f"Missing songs for categories: {missing_categories}")
        logging.error("Cannot proceed with conditioned testing")
        return

    # Define test scenarios
    scenarios = [
        {
            "name": "low_pitch_short_duration_to_high_long",
            "songs": extreme_songs["low_pitch_short_duration"],
            "description": "Low pitch + short duration → steer high + long (fight both)",
        },
        {
            "name": "high_pitch_long_duration_to_low_short",
            "songs": extreme_songs["high_pitch_long_duration"],
            "description": "High pitch + long duration → steer low + short (fight both)",
        },
        {
            "name": "low_pitch_long_duration_to_high_short",
            "songs": extreme_songs["low_pitch_long_duration"],
            "description": "Low pitch + long duration → steer high + short (fight both opposite)",
        },
        {
            "name": "high_pitch_short_duration_to_low_long",
            "songs": extreme_songs["high_pitch_short_duration"],
            "description": "High pitch + short duration → steer low + long (fight both opposite)",
        },
    ]

    # Run experiments
    logging.info("\n" + "=" * 80)
    logging.info("RUNNING CONDITIONED DUAL-STEERING EXPERIMENTS")
    logging.info("=" * 80)

    all_results = []
    start_time = time.time()

    for strategy in args.strategies:
        logging.info(f"\n### STRATEGY: {strategy.upper()} ###")

        for scenario in scenarios:
            logging.info(f"\n{scenario['description']}")

            results = conditioned_generate_and_evaluate(
                model=model,
                composer=composer,
                strategy=strategy,
                song_list=scenario["songs"],
                scenario=scenario["name"],
                alpha_pitch_list=alphas_pitch,
                alpha_duration_list=alphas_duration,
                encoding=encoding,
                device=device,
                conditioning_beats=args.conditioning_beats,
                continuation_len=args.continuation_len,
                output_dir=args.output_dir,
            )

            all_results.extend(results)

    elapsed_time = time.time() - start_time

    # Analyze results
    logging.info("\nAnalyzing results...")
    analysis = analyze_conditioned_results(all_results)

    # Extract listening list
    logging.info("\nExtracting listening list...")
    listening_list = extract_listening_list(all_results, args.listening_list_size)

    # Save results
    results_file = args.output_dir / "conditioned_results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "config": {
                    "n_songs_per_category": args.n_songs,
                    "conditioning_beats": args.conditioning_beats,
                    "continuation_len": args.continuation_len,
                    "alphas_pitch": alphas_pitch,
                    "alphas_duration": alphas_duration,
                    "strategies": args.strategies,
                    "total_generations": len(all_results),
                    "elapsed_time_seconds": elapsed_time,
                },
                "extreme_songs": {
                    k: [(str(f), p, d) for f, p, d in v]
                    for k, v in extreme_songs.items()
                },
                "results": all_results,
                "analysis": analysis,
            },
            f,
            indent=2,
        )

    logging.info(f"Saved results to: {results_file}")

    # Save listening list
    listening_file = args.output_dir / "listening_list.json"
    with open(listening_file, "w") as f:
        json.dump(
            {
                "top_examples": listening_list,
                "selection_criteria": {
                    "audibility": "Magnitude of pitch/duration change",
                    "quality": "Low degradation from ground truth",
                    "success": "Both concepts successfully steered",
                },
            },
            f,
            indent=2,
        )

    logging.info(f"Saved listening list to: {listening_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("PHASE 4: CONDITIONED DUAL-STEERING SUMMARY")
    print("=" * 80)

    print(f"\nTotal generations: {len(all_results)}")
    print(f"Total time: {elapsed_time/60:.1f} minutes")

    print("\n### Success Rates by Scenario ###")
    for scenario, stats in analysis["summary_by_scenario"].items():
        print(f"\n{scenario}:")
        print(f"  Overall success: {stats['overall_success_rate']*100:.1f}%")
        print(f"  Pitch success: {stats['pitch_success_rate']*100:.1f}%")
        print(f"  Duration success: {stats['duration_success_rate']*100:.1f}%")
        print(f"  Mean degradation: {stats['mean_degradation']:.2f}")
        print(f"  Mean pitch change: {stats['mean_pitch_change']:+.1f} semitones")
        print(f"  Mean duration change: {stats['mean_duration_change']:+.1f} ticks")

    print("\n### Best Strategy ###")
    for strategy, stats in analysis["summary_by_strategy"].items():
        print(f"\n{strategy}:")
        print(f"  Overall success: {stats['overall_success_rate']*100:.1f}%")
        print(f"  Mean degradation: {stats['mean_degradation']:.2f}")

    print("\n### Best Configurations ###")
    for scenario, best_config in analysis["best_configs"].items():
        print(f"\n{scenario}:")
        print(f"  α_pitch: {best_config['alpha_pitch']:+.1f}")
        print(f"  α_duration: {best_config['alpha_duration']:+.1f}")
        print(f"  Strategy: {best_config['strategy']}")
        print(f"  Success: {best_config['success_rate']*100:.1f}%")
        print(f"  Degradation: {best_config['degradation']:.2f}")

    print("\n### Listening List (Top 5) ###")
    for i, example in enumerate(listening_list[:5], 1):
        print(f"\n{i}. {example['song_name']} - {example['scenario']}")
        print(f"   Strategy: {example['strategy']}")
        print(
            f"   α_p={example['alpha_pitch']:+.1f}, α_d={example['alpha_duration']:+.1f}"
        )
        print(
            f"   Pitch: {example['conditioning_pitch']:.1f} → {example['generated_pitch_mean']:.1f} (Δ{example['pitch_change']:+.1f})"
        )
        print(
            f"   Duration: {example['conditioning_duration']:.1f} → {example['generated_duration_mean']:.1f} (Δ{example['duration_change']:+.1f})"
        )
        print(
            f"   Score: {example['listening_score']:.3f} (audibility={example['audibility_score']:.2f}, quality={example['quality_score']:.2f})"
        )

    print("\n" + "=" * 80)
    print("✅ PHASE 4 COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
