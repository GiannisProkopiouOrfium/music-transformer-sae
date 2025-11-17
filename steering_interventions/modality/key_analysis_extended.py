"""Conditioned Modality Steering Test with Key Analysis.

This script:
1. Finds songs with extreme modality (clear major vs clear minor) in conditioning segment
2. Uses the first N beats as conditioning prompts
3. Generates continuations with steering
4. Measures bidirectional steering effect (major ↔ minor)
5. Tests if steering can override the conditioning context
6. Identifies initial and final key (e.g., "A minor" → "G major")
7. Analyzes key transition patterns and distributions

Expected behavior (positive α → major, negative α → minor):
  - Major conditioning + positive α: reinforces major (alignment)
  - Major conditioning + negative α: shifts toward minor (fighting context)
  - Minor conditioning + positive α: shifts toward major (fighting context)
  - Minor conditioning + negative α: reinforces minor (alignment)
"""

import argparse
import json
import logging
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add parent directories to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import config
import music_x_transformers
import muspy
import representation
import utils
from steered_generator import SteeredGenerator, load_steering_vectors

# Import music21
try:
    from music21 import note, stream, pitch

    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    print("ERROR: music21 not available. Install with: pip install music21")
    sys.exit(1)


# Ground truth metrics from paper
GROUND_TRUTH_METRICS = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}


def load_song_tokens(filepath: pathlib.Path, encoding: Dict) -> np.ndarray:
    """Load song tokens from .npy file.

    Args:
        filepath: Path to .npy file
        encoding: Encoding dictionary

    Returns:
        Token array (seq_len, 6)
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load notes in 5D format: [beat, position, pitch, duration, program]
    notes = np.load(filepath)

    if len(notes.shape) != 2 or notes.shape[1] != 5:
        raise ValueError(f"Invalid shape {notes.shape}, expected (seq_len, 5)")

    # Convert to 6D codes
    codes = representation.encode_notes(notes, encoding)
    return codes


def detect_key_from_tokens(
    tokens: np.ndarray, encoding: Dict
) -> Tuple[str, str, float]:
    """Detect full key (tonic + mode) from tokens.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        (tonic, mode, confidence): e.g., ("A", "minor", 0.85)
    """
    try:
        note_type = encoding["type_code_map"]["note"]
        pitches = []

        for token in tokens:
            if token[0] == note_type:
                pitch_value = token[3]
                pitches.append(pitch_value)

        if len(pitches) < 10:
            return ("unknown", "unknown", 0.0)

        s = stream.Stream()
        for p in pitches:
            s.append(note.Note(p))

        key = s.analyze("key")

        # Extract tonic name (e.g., "A", "C#", "Bb")
        tonic = key.tonic.name
        mode = key.mode
        confidence = key.correlationCoefficient

        return (tonic, mode, confidence)

    except Exception as e:
        logging.warning(f"Key detection failed: {e}")
        return ("unknown", "unknown", 0.0)


def get_key_string(tonic: str, mode: str) -> str:
    """Create readable key string.

    Args:
        tonic: Tonic name (e.g., "A", "C#")
        mode: Mode ("major" or "minor")

    Returns:
        Key string (e.g., "A minor", "C# major")
    """
    if tonic == "unknown" or mode == "unknown":
        return "unknown"
    return f"{tonic} {mode}"


def calculate_tonic_interval(tonic1: str, tonic2: str) -> Tuple[int, str]:
    """Calculate interval between two tonics.

    Args:
        tonic1: First tonic (e.g., "A")
        tonic2: Second tonic (e.g., "G")

    Returns:
        (semitones, interval_name): e.g., (-2, "major 2nd down")
    """
    if tonic1 == "unknown" or tonic2 == "unknown":
        return (0, "unknown")

    try:
        p1 = pitch.Pitch(tonic1)
        p2 = pitch.Pitch(tonic2)

        # Calculate semitone difference
        semitones = p2.midi - p1.midi

        # Handle octave wrapping (keep within -6 to +6 semitones)
        while semitones > 6:
            semitones -= 12
        while semitones < -6:
            semitones += 12

        # Name the interval
        interval_names = {
            0: "unison",
            1: "minor 2nd",
            2: "major 2nd",
            3: "minor 3rd",
            4: "major 3rd",
            5: "perfect 4th",
            6: "tritone",
            -1: "minor 2nd down",
            -2: "major 2nd down",
            -3: "minor 3rd down",
            -4: "major 3rd down",
            -5: "perfect 4th down",
            -6: "tritone down",
        }

        interval_name = interval_names.get(semitones, f"{abs(semitones)} semitones")

        return (semitones, interval_name)

    except Exception as e:
        logging.warning(f"Interval calculation failed: {e}")
        return (0, "unknown")


def classify_transition_type(tonic1: str, mode1: str, tonic2: str, mode2: str) -> str:
    """Classify the type of key transition.

    Args:
        tonic1: Initial tonic
        mode1: Initial mode
        tonic2: Final tonic
        mode2: Final mode

    Returns:
        Transition type classification
    """
    tonic_changed = tonic1 != tonic2
    mode_changed = mode1 != mode2

    if not tonic_changed and not mode_changed:
        return "no_change"
    elif tonic_changed and not mode_changed:
        return "tonic_change_only"
    elif not tonic_changed and mode_changed:
        return "parallel_key"  # Same tonic, different mode (e.g., C major ↔ C minor)
    else:
        # Both changed - check if relative keys
        if mode1 != mode2:
            # Check for relative key relationship (e.g., C major ↔ A minor)
            try:
                p1 = pitch.Pitch(tonic1)
                p2 = pitch.Pitch(tonic2)
                semitones = abs(p2.midi - p1.midi) % 12

                if semitones == 3:  # Minor 3rd apart = relative keys
                    return "relative_key"
            except:
                pass

        return "both_change"


def evaluate_quality_metrics(tokens: np.ndarray, encoding: Dict) -> Dict:
    """Evaluate objective quality metrics.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        Dictionary with quality metrics
    """
    try:
        import muspy

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
            "error": str(e),
        }


def calculate_degradation(metrics: Dict, baseline: Dict) -> Dict:
    """Calculate quality degradation from baseline.

    Args:
        metrics: Current quality metrics
        baseline: Baseline quality metrics

    Returns:
        Degradation scores
    """
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


def find_extreme_modality_songs(
    notes_dir: pathlib.Path,
    encoding: Dict,
    n_songs: int = 10,
    conditioning_beats: int = 8,
    min_confidence: float = 0.8,
) -> Tuple[
    List[Tuple[pathlib.Path, str, str, float]],
    List[Tuple[pathlib.Path, str, str, float]],
]:
    """Find songs with extreme modality (clear major vs clear minor).

    Args:
        notes_dir: Directory containing .npy files
        encoding: Encoding dictionary
        n_songs: Number of songs to find per category
        conditioning_beats: Number of beats to analyze for conditioning
        min_confidence: Minimum confidence for classification

    Returns:
        (major_songs, minor_songs) - lists of (filepath, tonic, mode, confidence)
    """
    logging.info(f"Scanning {notes_dir} for songs with extreme modality...")

    major_songs = []
    minor_songs = []

    # Scan for .npy files
    for subfolder in notes_dir.iterdir():
        if not subfolder.is_dir():
            continue

        for filepath in subfolder.glob("*.npy"):
            try:
                tokens = load_song_tokens(filepath, encoding)

                # Extract conditioning prefix
                cond_tokens = extract_conditioning_prefix(tokens, conditioning_beats)

                # Detect key in conditioning region
                tonic, mode, confidence = detect_key_from_tokens(
                    cond_tokens.numpy()[0], encoding
                )

                if confidence >= min_confidence:
                    if mode == "major":
                        major_songs.append((filepath, tonic, mode, confidence))
                    elif mode == "minor":
                        minor_songs.append((filepath, tonic, mode, confidence))

            except Exception as e:
                logging.debug(f"Error processing {filepath.name}: {e}")

    # Sort by confidence (highest first)
    major_songs.sort(key=lambda x: x[3], reverse=True)
    minor_songs.sort(key=lambda x: x[3], reverse=True)

    # Take top N
    major_songs = major_songs[:n_songs]
    minor_songs = minor_songs[:n_songs]

    logging.info(f"Found {len(major_songs)} high-confidence major songs")
    if major_songs:
        logging.info(
            f"  Examples: {[f'{get_key_string(t, m)} ({c:.3f})' for _, t, m, c in major_songs[:3]]}"
        )

    logging.info(f"Found {len(minor_songs)} high-confidence minor songs")
    if minor_songs:
        logging.info(
            f"  Examples: {[f'{get_key_string(t, m)} ({c:.3f})' for _, t, m, c in minor_songs[:3]]}"
        )

    return major_songs, minor_songs


def extract_conditioning_prefix(tokens: np.ndarray, n_beats: int = 8) -> torch.Tensor:
    """Extract first N beats as conditioning.

    Args:
        tokens: Full song tokens (seq_len, 6)
        n_beats: Number of beats to extract

    Returns:
        Conditioning tokens (1, cond_len, 6)
    """
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


def conditioned_generate_and_evaluate(
    model,
    steering_vectors: Dict[int, torch.Tensor],
    encoding: Dict,
    device: torch.device,
    song_list: List[Tuple[pathlib.Path, str, str, float]],
    category: str,
    alphas: List[float],
    target_layers: List[int],
    conditioning_beats: int,
    continuation_len: int,
    output_dir: pathlib.Path = None,
) -> List[Dict]:
    """Generate conditioned continuations with steering.

    Args:
        model: The model
        steering_vectors: Steering vectors
        encoding: Encoding dictionary
        device: Device
        song_list: List of (filepath, tonic, mode, confidence) tuples
        category: "major" or "minor" (conditioning category)
        alphas: List of alpha values to test
        target_layers: Layers to apply steering
        conditioning_beats: Beats for conditioning
        continuation_len: Length to generate
        output_dir: Where to save results (optional)

    Returns:
        List of result dictionaries
    """
    generator = SteeredGenerator(model, steering_vectors, encoding)
    eos = encoding["type_code_map"]["end-of-song"]

    results = []

    for i, (filepath, cond_tonic, cond_mode, cond_confidence) in enumerate(song_list):
        cond_key = get_key_string(cond_tonic, cond_mode)

        logging.info(f"\n{'='*70}")
        logging.info(f"Song {i+1}/{len(song_list)}: {filepath.name}")
        logging.info(f"Conditioning: {cond_key} (confidence: {cond_confidence:.3f})")
        logging.info(f"{'='*70}")

        # Load and extract conditioning
        tokens = load_song_tokens(filepath, encoding)
        conditioning = extract_conditioning_prefix(tokens, conditioning_beats)
        conditioning = conditioning.to(device)

        logging.info(
            f"Conditioning: {conditioning.shape[1]} tokens ({conditioning_beats} beats)"
        )

        # Generate with different alphas
        for alpha in alphas:
            logging.info(f"\n  Testing α = {alpha}...")

            # Generate continuation
            generated = generator.generate(
                conditioning,
                continuation_len,
                alpha=alpha,
                target_layers=target_layers,
                eos_token=eos,
                temperature=config.GENERATION_TEMPERATURE,
                filter_logits_fn=config.GENERATION_FILTER,
                filter_thres=config.GENERATION_FILTER_THRESHOLD,
                monotonicity_dim=("type", "beat"),
            )

            # Extract generated portion only
            generated_only = generated.cpu().numpy()[0]

            # Detect key in generated portion
            gen_tonic, gen_mode, gen_confidence = detect_key_from_tokens(
                generated_only, encoding
            )
            gen_key = get_key_string(gen_tonic, gen_mode)

            # Evaluate quality metrics on generated portion
            quality_metrics = evaluate_quality_metrics(generated_only, encoding)

            # Calculate key transition details
            key_transition = f"{cond_key} → {gen_key}"
            tonic_interval, interval_name = calculate_tonic_interval(
                cond_tonic, gen_tonic
            )
            transition_type = classify_transition_type(
                cond_tonic, cond_mode, gen_tonic, gen_mode
            )

            # Calculate bidirectional metric
            # +100 = pure major, -100 = pure minor
            if gen_mode == "major" and gen_confidence >= 0.5:
                bidirectional_score = +100
            elif gen_mode == "minor" and gen_confidence >= 0.5:
                bidirectional_score = -100
            else:
                bidirectional_score = 0  # Unknown

            result = {
                "song_name": filepath.stem,
                "conditioning_category": category,
                # Conditioning key info
                "conditioning_tonic": cond_tonic,
                "conditioning_mode": cond_mode,
                "conditioning_key": cond_key,
                "conditioning_confidence": float(cond_confidence),
                # Generated key info
                "generated_tonic": gen_tonic,
                "generated_mode": gen_mode,
                "generated_key": gen_key,
                "generated_confidence": float(gen_confidence),
                # Transition analysis
                "key_transition": key_transition,
                "tonic_changed": cond_tonic != gen_tonic,
                "mode_changed": cond_mode != gen_mode,
                "tonic_interval_semitones": tonic_interval,
                "tonic_interval_name": interval_name,
                "transition_type": transition_type,
                # Other metrics
                "alpha": alpha,
                "target_layers": target_layers if target_layers else "all",
                "bidirectional_score": bidirectional_score,
                "conditioning_beats": conditioning_beats,
                "continuation_tokens": len(generated_only),
                "quality_metrics": quality_metrics,
            }

            results.append(result)

            logging.info(f"    Generated: {gen_key} (conf: {gen_confidence:.3f})")
            logging.info(f"    Transition: {key_transition}")
            logging.info(f"    Type: {transition_type}, Interval: {interval_name}")
            logging.info(
                f"    Quality: entropy={quality_metrics.get('pitch_class_entropy', 0):.3f}, "
                f"scale={quality_metrics.get('scale_consistency', 0):.1f}%, "
                f"groove={quality_metrics.get('groove_consistency', 0):.1f}%"
            )

            # Save if output_dir provided
            if output_dir is not None:
                # Combine conditioning + generated for full sequence
                full_seq = torch.cat((conditioning, generated), 1).cpu().numpy()[0]

                save_dir = output_dir / category / f"alpha_{alpha}"
                save_dir.mkdir(parents=True, exist_ok=True)

                # Save tokens
                np.save(save_dir / f"{filepath.stem}.npy", full_seq)

                # Save audio
                try:
                    music = representation.decode(full_seq, encoding)
                    music.write(str(save_dir / f"{filepath.stem}.mid"))
                    music.write_audio(str(save_dir / f"{filepath.stem}.wav"))
                except Exception as e:
                    logging.error(f"Error saving audio: {e}")

    return results


def analyze_key_transitions(results: List[Dict]) -> Dict:
    """Analyze key transition patterns.

    Args:
        results: List of result dictionaries

    Returns:
        Analysis dictionary with transition statistics
    """
    analysis = {
        "transition_frequency": {},
        "transition_by_alpha": {},
        "transition_by_type": {},
        "tonic_intervals": {},
        "most_common_transitions": [],
    }

    # Filter confident results
    confident = [r for r in results if r["generated_confidence"] >= 0.5]

    if not confident:
        return analysis

    # Count transition frequencies
    transition_counts = defaultdict(lambda: {"count": 0, "alphas": [], "scores": []})

    for r in confident:
        transition = r["key_transition"]
        transition_counts[transition]["count"] += 1
        transition_counts[transition]["alphas"].append(r["alpha"])

        # Calculate a simple score (confidence - degradation if available)
        if "degradation" in r:
            score = (
                r["generated_confidence"] * 100 - r["degradation"]["total_degradation"]
            )
        else:
            score = r["generated_confidence"] * 100
        transition_counts[transition]["scores"].append(score)

    # Convert to regular dict with averages
    for transition, data in transition_counts.items():
        analysis["transition_frequency"][transition] = {
            "count": data["count"],
            "avg_alpha": float(np.mean(data["alphas"])),
            "avg_score": float(np.mean(data["scores"])),
        }

    # Most common transitions
    sorted_transitions = sorted(
        analysis["transition_frequency"].items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    )
    analysis["most_common_transitions"] = [
        {
            "transition": t,
            "count": data["count"],
            "avg_alpha": data["avg_alpha"],
            "avg_score": data["avg_score"],
        }
        for t, data in sorted_transitions[:20]  # Top 20
    ]

    # Group by alpha
    for alpha in sorted(set(r["alpha"] for r in confident)):
        alpha_results = [r for r in confident if r["alpha"] == alpha]

        transition_types = defaultdict(int)
        tonic_intervals = defaultdict(int)

        for r in alpha_results:
            transition_types[r["transition_type"]] += 1
            if r["tonic_interval_name"] != "unknown":
                tonic_intervals[r["tonic_interval_name"]] += 1

        total = len(alpha_results)

        analysis["transition_by_alpha"][f"alpha_{alpha}"] = {
            "alpha": alpha,
            "total_samples": total,
            "transition_types": {
                t: {"count": c, "percentage": 100 * c / total}
                for t, c in transition_types.items()
            },
            "tonic_intervals": dict(tonic_intervals),
        }

    # Group by transition type (overall)
    transition_types_overall = defaultdict(int)
    for r in confident:
        transition_types_overall[r["transition_type"]] += 1

    total = len(confident)
    analysis["transition_by_type"] = {
        t: {"count": c, "percentage": 100 * c / total}
        for t, c in transition_types_overall.items()
    }

    # Tonic interval analysis (overall)
    tonic_intervals_overall = defaultdict(int)
    for r in confident:
        if r["tonic_interval_name"] != "unknown":
            tonic_intervals_overall[r["tonic_interval_name"]] += 1

    analysis["tonic_intervals"] = dict(tonic_intervals_overall)

    return analysis


def analyze_musical_relationships(results: List[Dict]) -> Dict:
    """Analyze musical relationships in key transitions.

    Args:
        results: List of result dictionaries

    Returns:
        Analysis of musical relationships
    """
    analysis = {
        "parallel_keys": [],
        "relative_keys": [],
        "common_tonic_shifts_positive": {},
        "common_tonic_shifts_negative": {},
    }

    confident = [r for r in results if r["generated_confidence"] >= 0.5]

    # Find parallel key transitions (same tonic, different mode)
    parallel = [r for r in confident if r["transition_type"] == "parallel_key"]
    analysis["parallel_keys"] = [
        {
            "transition": r["key_transition"],
            "alpha": r["alpha"],
            "song": r["song_name"],
        }
        for r in parallel
    ]

    # Find relative key transitions
    relative = [r for r in confident if r["transition_type"] == "relative_key"]
    analysis["relative_keys"] = [
        {
            "transition": r["key_transition"],
            "alpha": r["alpha"],
            "song": r["song_name"],
        }
        for r in relative
    ]

    # Common tonic shifts by alpha direction
    positive_alpha = [r for r in confident if r["alpha"] > 0]
    negative_alpha = [r for r in confident if r["alpha"] < 0]

    # Count intervals for positive steering
    positive_intervals = defaultdict(int)
    for r in positive_alpha:
        if r["tonic_interval_name"] != "unknown":
            positive_intervals[r["tonic_interval_name"]] += 1

    analysis["common_tonic_shifts_positive"] = dict(
        sorted(positive_intervals.items(), key=lambda x: x[1], reverse=True)
    )

    # Count intervals for negative steering
    negative_intervals = defaultdict(int)
    for r in negative_alpha:
        if r["tonic_interval_name"] != "unknown":
            negative_intervals[r["tonic_interval_name"]] += 1

    analysis["common_tonic_shifts_negative"] = dict(
        sorted(negative_intervals.items(), key=lambda x: x[1], reverse=True)
    )

    return analysis


def calculate_musical_coherence(results: List[Dict]) -> Dict:
    """Calculate how often transitions follow music theory principles.

    Args:
        results: List of result dictionaries

    Returns:
        Musical coherence metrics
    """
    confident = [r for r in results if r["generated_confidence"] >= 0.5]

    if not confident:
        return {}

    # Circle of fifths relationships (±7 semitones or ±5 semitones)
    circle_of_fifths = [
        "perfect 4th",
        "perfect 4th down",
        "perfect 5th",
        "perfect 5th down",
    ]

    # Relative key relationships (minor 3rd apart with mode change)
    relative_keys = [r for r in confident if r["transition_type"] == "relative_key"]

    # Parallel keys (same tonic, mode change)
    parallel_keys = [r for r in confident if r["transition_type"] == "parallel_key"]

    # Circle of fifths transitions
    circle_fifths_transitions = [
        r
        for r in confident
        if any(interval in r["tonic_interval_name"] for interval in circle_of_fifths)
    ]

    # Common modulations (relative, parallel, circle of fifths)
    common_modulations = (
        len(relative_keys) + len(parallel_keys) + len(circle_fifths_transitions)
    )

    # Chromatic modulations (minor 2nd, major 2nd, tritone)
    chromatic_intervals = [
        "minor 2nd",
        "major 2nd",
        "tritone",
        "minor 2nd down",
        "major 2nd down",
        "tritone down",
    ]
    chromatic_modulations = [
        r
        for r in confident
        if any(interval in r["tonic_interval_name"] for interval in chromatic_intervals)
        and r["transition_type"] not in ["relative_key", "parallel_key"]
    ]

    total = len(confident)

    coherence = {
        "total_transitions": total,
        "relative_keys": {
            "count": len(relative_keys),
            "percentage": 100 * len(relative_keys) / total,
        },
        "parallel_keys": {
            "count": len(parallel_keys),
            "percentage": 100 * len(parallel_keys) / total,
        },
        "circle_of_fifths": {
            "count": len(circle_fifths_transitions),
            "percentage": 100 * len(circle_fifths_transitions) / total,
        },
        "common_modulations_total": {
            "count": common_modulations,
            "percentage": 100 * common_modulations / total,
        },
        "chromatic_modulations": {
            "count": len(chromatic_modulations),
            "percentage": 100 * len(chromatic_modulations) / total,
        },
        "no_change": {
            "count": sum(1 for r in confident if r["transition_type"] == "no_change"),
            "percentage": 100
            * sum(1 for r in confident if r["transition_type"] == "no_change")
            / total,
        },
    }

    # Per-alpha analysis
    coherence["by_alpha"] = {}
    for alpha in sorted(set(r["alpha"] for r in confident)):
        alpha_results = [r for r in confident if r["alpha"] == alpha]
        alpha_total = len(alpha_results)

        alpha_relative = sum(
            1 for r in alpha_results if r["transition_type"] == "relative_key"
        )
        alpha_parallel = sum(
            1 for r in alpha_results if r["transition_type"] == "parallel_key"
        )
        alpha_circle = sum(
            1
            for r in alpha_results
            if any(
                interval in r["tonic_interval_name"] for interval in circle_of_fifths
            )
        )
        alpha_common = alpha_relative + alpha_parallel + alpha_circle

        coherence["by_alpha"][f"alpha_{alpha}"] = {
            "alpha": alpha,
            "common_modulations_percentage": (
                100 * alpha_common / alpha_total if alpha_total > 0 else 0
            ),
            "relative_keys_percentage": (
                100 * alpha_relative / alpha_total if alpha_total > 0 else 0
            ),
            "parallel_keys_percentage": (
                100 * alpha_parallel / alpha_total if alpha_total > 0 else 0
            ),
            "circle_of_fifths_percentage": (
                100 * alpha_circle / alpha_total if alpha_total > 0 else 0
            ),
        }

    return coherence


def compare_alpha_distributions(results: List[Dict]) -> Dict:
    """Statistical tests comparing transition distributions across alphas.

    Args:
        results: List of result dictionaries

    Returns:
        Statistical test results
    """
    confident = [r for r in results if r["generated_confidence"] >= 0.5]

    if not confident:
        return {}

    # Group by alpha
    alphas = sorted(set(r["alpha"] for r in confident))

    # Contingency table for transition types
    transition_types = [
        "no_change",
        "tonic_change_only",
        "parallel_key",
        "relative_key",
        "both_change",
    ]
    contingency = []

    for alpha in alphas:
        alpha_results = [r for r in confident if r["alpha"] == alpha]
        row = []
        for t_type in transition_types:
            count = sum(1 for r in alpha_results if r["transition_type"] == t_type)
            row.append(count)
        contingency.append(row)

    # Chi-square test for independence
    contingency_array = np.array(contingency)
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_array)

    # Pairwise comparisons between baseline (alpha=0) and others
    pairwise_tests = {}
    baseline_alpha = 0.0

    if baseline_alpha in alphas:
        baseline_results = [r for r in confident if r["alpha"] == baseline_alpha]
        baseline_types = [r["transition_type"] for r in baseline_results]

        for alpha in alphas:
            if alpha == baseline_alpha:
                continue

            alpha_results = [r for r in confident if r["alpha"] == alpha]
            alpha_types = [r["transition_type"] for r in alpha_results]

            # Create contingency table for this pair
            pair_contingency = []
            for t_type in transition_types:
                baseline_count = sum(1 for t in baseline_types if t == t_type)
                alpha_count = sum(1 for t in alpha_types if t == t_type)
                pair_contingency.append([baseline_count, alpha_count])

            pair_contingency_array = np.array(pair_contingency).T

            try:
                pair_chi2, pair_p_value, pair_dof, _ = stats.chi2_contingency(
                    pair_contingency_array
                )
                pairwise_tests[f"baseline_vs_{alpha}"] = {
                    "alpha": alpha,
                    "chi2": float(pair_chi2),
                    "p_value": float(pair_p_value),
                    "dof": int(pair_dof),
                    "significant": pair_p_value < 0.05,
                }
            except Exception:
                pairwise_tests[f"baseline_vs_{alpha}"] = {
                    "alpha": alpha,
                    "error": "Could not compute chi-square test",
                }

    # Test for mode change rates across alphas
    mode_change_by_alpha = []
    for alpha in alphas:
        alpha_results = [r for r in confident if r["alpha"] == alpha]
        mode_changed = sum(1 for r in alpha_results if r["mode_changed"])
        mode_unchanged = len(alpha_results) - mode_changed
        mode_change_by_alpha.append([mode_changed, mode_unchanged])

    mode_chi2, mode_p_value, mode_dof, _ = stats.chi2_contingency(mode_change_by_alpha)

    return {
        "overall_chi_square": {
            "test": "Chi-square test of independence",
            "hypothesis": "Transition type distribution is independent of alpha value",
            "chi2": float(chi2),
            "p_value": float(p_value),
            "dof": int(dof),
            "significant": p_value < 0.05,
            "interpretation": (
                "Reject null hypothesis: Alpha significantly affects transition types"
                if p_value < 0.05
                else "Cannot reject null hypothesis: No significant effect"
            ),
        },
        "pairwise_comparisons": pairwise_tests,
        "mode_change_test": {
            "test": "Chi-square test for mode change rates",
            "chi2": float(mode_chi2),
            "p_value": float(mode_p_value),
            "dof": int(mode_dof),
            "significant": mode_p_value < 0.05,
        },
        "alphas_tested": alphas,
        "transition_types": transition_types,
        "contingency_table": contingency,
    }


def visualize_transition_matrices(transition_matrices: Dict, output_dir: pathlib.Path):
    """Create heatmap visualizations of transition matrices.

    Args:
        transition_matrices: Dictionary of transition matrices per alpha
        output_dir: Directory to save visualizations
    """
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Get all unique keys across all matrices
    all_keys = set()
    for matrix in transition_matrices.values():
        all_keys.update(matrix.keys())
        for inner_dict in matrix.values():
            all_keys.update(inner_dict.keys())

    all_keys = sorted(all_keys)

    # Create a heatmap for each alpha
    for alpha_key, matrix in transition_matrices.items():
        alpha_value = alpha_key.replace("alpha_", "")

        # Create 2D array for heatmap
        matrix_array = np.zeros((len(all_keys), len(all_keys)))

        for i, cond_key in enumerate(all_keys):
            if cond_key in matrix:
                for j, gen_key in enumerate(all_keys):
                    if gen_key in matrix[cond_key]:
                        matrix_array[i, j] = matrix[cond_key][gen_key]

        # Only create visualization if there's data
        if matrix_array.sum() > 0:
            plt.figure(figsize=(14, 12))
            sns.heatmap(
                matrix_array,
                xticklabels=all_keys,
                yticklabels=all_keys,
                annot=True,
                fmt=".0f",
                cmap="YlOrRd",
                cbar_kws={"label": "Transition Count"},
                square=True,
            )
            plt.xlabel("Generated Key", fontsize=12)
            plt.ylabel("Conditioning Key", fontsize=12)
            plt.title(
                f"Key Transition Matrix (α = {alpha_value})",
                fontsize=14,
                fontweight="bold",
            )
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()

            output_file = viz_dir / f"transition_matrix_alpha_{alpha_value}.png"
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            logging.info(f"Saved heatmap: {output_file}")

    # Create a combined comparison plot (all alphas side by side)
    n_alphas = len(transition_matrices)
    if n_alphas > 0:
        fig, axes = plt.subplots(1, min(n_alphas, 4), figsize=(20, 5))
        if n_alphas == 1:
            axes = [axes]

        for idx, (alpha_key, matrix) in enumerate(
            sorted(transition_matrices.items())[:4]
        ):
            alpha_value = alpha_key.replace("alpha_", "")

            # Create 2D array
            matrix_array = np.zeros((len(all_keys), len(all_keys)))
            for i, cond_key in enumerate(all_keys):
                if cond_key in matrix:
                    for j, gen_key in enumerate(all_keys):
                        if gen_key in matrix[cond_key]:
                            matrix_array[i, j] = matrix[cond_key][gen_key]

            sns.heatmap(
                matrix_array,
                xticklabels=all_keys if idx == 0 else [],
                yticklabels=all_keys if idx == 0 else [],
                annot=False,
                cmap="YlOrRd",
                cbar=True,
                square=True,
                ax=axes[idx],
            )
            axes[idx].set_title(f"α = {alpha_value}", fontsize=12)
            if idx == 0:
                axes[idx].set_ylabel("Conditioning Key", fontsize=10)
            axes[idx].set_xlabel("Generated Key", fontsize=10)

        plt.tight_layout()
        comparison_file = viz_dir / "transition_matrices_comparison.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches="tight")
        plt.close()

        logging.info(f"Saved comparison plot: {comparison_file}")


def create_transition_matrix(results: List[Dict], alpha: float = None) -> Dict:
    """Create transition matrix for visualization.

    Args:
        results: List of result dictionaries
        alpha: Optional alpha value to filter by

    Returns:
        Transition matrix data
    """
    confident = [r for r in results if r["generated_confidence"] >= 0.5]

    if alpha is not None:
        confident = [r for r in confident if r["alpha"] == alpha]

    matrix = defaultdict(lambda: defaultdict(int))

    for r in confident:
        cond_key = r["conditioning_key"]
        gen_key = r["generated_key"]
        matrix[cond_key][gen_key] += 1

    # Convert to regular dict
    return {k: dict(v) for k, v in matrix.items()}


def rank_by_key_transition(
    results: List[Dict], output_dir: pathlib.Path = None
) -> List[Dict]:
    """Rank generations by key transition clarity and interest.

    Args:
        results: List of result dictionaries
        output_dir: Output directory for file paths

    Returns:
        Ranked list of results
    """
    ranked = []

    for r in results:
        if r["generated_confidence"] < 0.5:
            continue  # Skip low confidence
        if "degradation" not in r:
            continue  # Skip if no quality metrics

        # Base score on both key confidences
        clarity_score = (r["conditioning_confidence"] + r["generated_confidence"]) * 50

        # Bonus for interesting transitions
        interest_bonus = 0
        if r["transition_type"] == "both_change":
            interest_bonus = 20  # Both tonic and mode changed
        elif r["transition_type"] == "relative_key":
            interest_bonus = 15  # Relative key relationship
        elif r["transition_type"] == "parallel_key":
            interest_bonus = 10  # Parallel key relationship
        elif r["transition_type"] == "tonic_change_only":
            interest_bonus = 5  # Tonic changed

        # Penalty for degradation
        degradation_penalty = r["degradation"]["total_degradation"]

        total_score = clarity_score + interest_bonus - degradation_penalty

        # Construct file paths if output_dir provided
        category = r["conditioning_category"]
        alpha = r["alpha"]
        song_name = r["song_name"]

        file_paths = {}
        if output_dir is not None:
            base_path = output_dir / category / f"alpha_{alpha}" / song_name
            file_paths = {
                "midi_file": str(base_path.with_suffix(".mid")),
                "wav_file": str(base_path.with_suffix(".wav")),
                "npy_file": str(base_path.with_suffix(".npy")),
            }

        ranked.append(
            {
                "song_name": r["song_name"],
                "conditioning_category": r["conditioning_category"],
                "conditioning_key": r["conditioning_key"],
                "generated_key": r["generated_key"],
                "key_transition": r["key_transition"],
                "alpha": r["alpha"],
                "transition_type": r["transition_type"],
                "tonic_interval": r["tonic_interval_name"],
                "conditioning_confidence": r["conditioning_confidence"],
                "generated_confidence": r["generated_confidence"],
                "degradation": r["degradation"]["total_degradation"],
                "clarity_score": clarity_score,
                "interest_bonus": interest_bonus,
                "total_score": total_score,
                "file_paths": file_paths,
            }
        )

    # Sort by total score
    ranked.sort(key=lambda x: x["total_score"], reverse=True)

    return ranked


def main():
    parser = argparse.ArgumentParser(
        description="Test conditioned modality steering with key analysis"
    )
    parser.add_argument(
        "--concept", type=str, default="modality", help="Concept (should be modality)"
    )
    parser.add_argument(
        "--steering_vectors",
        type=pathlib.Path,
        default=None,
        help="Path to steering vectors file",
    )
    parser.add_argument(
        "--checkpoint", type=pathlib.Path, default=None, help="Model checkpoint path"
    )
    parser.add_argument(
        "--n_songs", type=int, default=5, help="Number of songs per category"
    )
    parser.add_argument(
        "--conditioning_beats", type=int, default=8, help="Beats for conditioning"
    )
    parser.add_argument(
        "--continuation_len", type=int, default=256, help="Tokens to generate"
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="-2.0,-1.0,0.0,1.0,2.0",
        help="Comma-separated alpha values",
    )
    parser.add_argument(
        "--target_layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (default: all layers)",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.8,
        help="Minimum confidence for conditioning songs",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU number")
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.output_dir is None:
        args.output_dir = pathlib.Path(
            "steering_interventions/modality/outputs/key_transition_analysis"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if args.gpu is not None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
            logging.info(f"Using CUDA device: GPU {args.gpu}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using MPS device")
        else:
            device = torch.device("cpu")
            logging.warning("CUDA/MPS not available, using CPU")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")

    # Load steering vectors
    if args.steering_vectors is None:
        steering_path = pathlib.Path(
            "steering_interventions/modality/outputs/steering_vectors/modality_steering_vectors.pt"
        )
    else:
        steering_path = args.steering_vectors

    steering_vectors, _ = load_steering_vectors(steering_path)
    steering_vectors = {k: v.to(device) for k, v in steering_vectors.items()}

    # Load model
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

    if args.checkpoint is None:
        checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    else:
        checkpoint_path = args.checkpoint

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    logging.info("Model loaded")

    # Parse alphas
    alphas = [float(a.strip()) for a in args.alphas.split(",")]

    # Parse target layers
    target_layers = None
    if args.target_layers is not None:
        target_layers = [int(layer.strip()) for layer in args.target_layers.split(",")]
        logging.info(f"Will apply steering to layers: {target_layers}")
    else:
        logging.info("Will apply steering to all layers")

    # Find extreme modality songs
    logging.info("\n" + "=" * 70)
    logging.info("FINDING EXTREME MODALITY SONGS")
    logging.info("=" * 70)

    major_songs, minor_songs = find_extreme_modality_songs(
        config.NOTES_DIR,
        encoding,
        args.n_songs,
        args.conditioning_beats,
        args.min_confidence,
    )

    if len(major_songs) == 0 or len(minor_songs) == 0:
        logging.error("Not enough songs found! Try lowering --min_confidence")
        sys.exit(1)

    # Generate and evaluate
    logging.info("\n" + "=" * 70)
    logging.info("GENERATING WITH CONDITIONED STEERING")
    logging.info("=" * 70)

    all_results = []

    # Major conditioning
    logging.info("\n### MAJOR CONDITIONING ###")
    major_results = conditioned_generate_and_evaluate(
        model,
        steering_vectors,
        encoding,
        device,
        major_songs,
        "major",
        alphas,
        target_layers,
        args.conditioning_beats,
        args.continuation_len,
        args.output_dir,
    )
    all_results.extend(major_results)

    # Minor conditioning
    logging.info("\n### MINOR CONDITIONING ###")
    minor_results = conditioned_generate_and_evaluate(
        model,
        steering_vectors,
        encoding,
        device,
        minor_songs,
        "minor",
        alphas,
        target_layers,
        args.conditioning_beats,
        args.continuation_len,
        args.output_dir,
    )
    all_results.extend(minor_results)

    # Calculate degradation for all results
    logging.info("\nCalculating quality metrics...")
    for r in all_results:
        if "quality_metrics" in r and r["quality_metrics"]:
            r["degradation"] = calculate_degradation(
                r["quality_metrics"], GROUND_TRUTH_METRICS
            )

    # Analyze key transitions
    logging.info("\nAnalyzing key transitions...")
    transition_analysis = analyze_key_transitions(all_results)

    logging.info("\nAnalyzing musical relationships...")
    relationship_analysis = analyze_musical_relationships(all_results)

    # Create transition matrices
    logging.info("\nCreating transition matrices...")
    transition_matrices = {}
    for alpha in alphas:
        transition_matrices[f"alpha_{alpha}"] = create_transition_matrix(
            all_results, alpha
        )

    # Rank by key transition quality
    logging.info("\nRanking by key transition clarity and interest...")
    ranked = rank_by_key_transition(all_results, args.output_dir)

    # Calculate musical coherence
    logging.info("\nCalculating musical coherence metrics...")
    coherence_analysis = calculate_musical_coherence(all_results)

    # Statistical tests
    logging.info("\nPerforming statistical tests...")
    statistical_tests = compare_alpha_distributions(all_results)

    # Create visualizations
    logging.info("\nCreating transition matrix visualizations...")
    try:
        visualize_transition_matrices(transition_matrices, args.output_dir)
    except Exception as e:
        logging.warning(f"Could not create visualizations: {e}")

    # Save comprehensive results
    results_file = args.output_dir / "key_transition_analysis.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "results": all_results,
                "transition_analysis": transition_analysis,
                "relationship_analysis": relationship_analysis,
                "transition_matrices": transition_matrices,
                "ranked_transitions": ranked,
                "musical_coherence": coherence_analysis,
                "statistical_tests": statistical_tests,
            },
            f,
            indent=2,
        )

    logging.info(f"Saved results to: {results_file}")

    # Save top ranked transitions to separate file for easy access
    top_ranked_file = args.output_dir / "top_ranked_transitions.json"
    with open(top_ranked_file, "w") as f:
        json.dump(
            {
                "top_50_transitions": ranked[:50],
                "instructions": "Use the file_paths in each entry to locate the generated audio files",
            },
            f,
            indent=2,
        )
    logging.info(f"Saved top ranked transitions to: {top_ranked_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("KEY TRANSITION ANALYSIS SUMMARY")
    print("=" * 80)

    print("\n### Most Common Transitions ###")
    for item in transition_analysis["most_common_transitions"][:10]:
        print(
            f"  {item['transition']:<30} "
            f"Count: {item['count']:>3}  "
            f"Avg α: {item['avg_alpha']:>+5.1f}  "
            f"Avg Score: {item['avg_score']:>6.1f}"
        )

    print("\n### Transition Types (Overall) ###")
    for t_type, data in sorted(
        transition_analysis["transition_by_type"].items(),
        key=lambda x: x[1]["count"],
        reverse=True,
    ):
        print(f"  {t_type:<20} {data['count']:>3} ({data['percentage']:>5.1f}%)")

    print("\n### Tonic Interval Distribution ###")
    sorted_intervals = sorted(
        transition_analysis["tonic_intervals"].items(), key=lambda x: x[1], reverse=True
    )
    for interval, count in sorted_intervals[:8]:
        print(f"  {interval:<20} {count:>3}")

    print("\n### Musical Relationships ###")
    print(
        f"  Parallel keys (same tonic): {len(relationship_analysis['parallel_keys'])}"
    )
    print(f"  Relative keys: {len(relationship_analysis['relative_keys'])}")

    if relationship_analysis["common_tonic_shifts_positive"]:
        print("\n  Common tonic shifts (positive α):")
        for interval, count in list(
            relationship_analysis["common_tonic_shifts_positive"].items()
        )[:5]:
            print(f"    {interval}: {count}")

    if relationship_analysis["common_tonic_shifts_negative"]:
        print("\n  Common tonic shifts (negative α):")
        for interval, count in list(
            relationship_analysis["common_tonic_shifts_negative"].items()
        )[:5]:
            print(f"    {interval}: {count}")

    print("\n### Top Ranked Transitions (by clarity + interest) ###")
    print("  See 'top_ranked_transitions.json' for full details with file paths")
    print(
        f"  {'Rank':<6} {'Transition':<35} {'Alpha':<7} {'Type':<18} "
        f"{'Score':<7} {'Degrad':<8}"
    )
    print("  " + "-" * 90)
    for i, item in enumerate(ranked[:20], 1):
        print(
            f"  {i:<6} {item['key_transition']:<35} "
            f"{item['alpha']:>+5.1f}  {item['transition_type']:<18} "
            f"{item['total_score']:>6.1f} {item['degradation']:>7.2f}"
        )

    # Print file path example
    if ranked and ranked[0].get("file_paths"):
        print("\n  Example file path (Rank #1):")
        print(f"    MIDI: {ranked[0]['file_paths'].get('midi_file', 'N/A')}")
        print(f"    WAV:  {ranked[0]['file_paths'].get('wav_file', 'N/A')}")

    print("\n### Transition Patterns by Alpha ###")
    for alpha_key in sorted(transition_analysis["transition_by_alpha"].keys()):
        data = transition_analysis["transition_by_alpha"][alpha_key]
        alpha = data["alpha"]
        print(f"\nα = {alpha:+.1f} (n={data['total_samples']}):")

        for t_type, t_data in sorted(
            data["transition_types"].items(), key=lambda x: x[1]["count"], reverse=True
        ):
            print(
                f"  {t_type:<20} {t_data['count']:>3} ({t_data['percentage']:>5.1f}%)"
            )

    print("\n### Musical Coherence Analysis ###")
    if coherence_analysis:
        print(
            f"  Common modulations (relative/parallel/circle of fifths): {coherence_analysis['common_modulations_total']['percentage']:.1f}%"
        )
        print(
            f"    - Relative keys: {coherence_analysis['relative_keys']['percentage']:.1f}%"
        )
        print(
            f"    - Parallel keys: {coherence_analysis['parallel_keys']['percentage']:.1f}%"
        )
        print(
            f"    - Circle of fifths: {coherence_analysis['circle_of_fifths']['percentage']:.1f}%"
        )
        print(
            f"  Chromatic modulations: {coherence_analysis['chromatic_modulations']['percentage']:.1f}%"
        )
        print(f"  No change: {coherence_analysis['no_change']['percentage']:.1f}%")

        print("\n  Common modulations by alpha:")
        for alpha_key in sorted(coherence_analysis["by_alpha"].keys()):
            alpha_data = coherence_analysis["by_alpha"][alpha_key]
            print(
                f"    α = {alpha_data['alpha']:>+5.1f}: {alpha_data['common_modulations_percentage']:>5.1f}%"
            )

    print("\n### Statistical Tests ###")
    if statistical_tests:
        overall = statistical_tests["overall_chi_square"]
        print(
            f"  Chi-square test: χ² = {overall['chi2']:.2f}, p = {overall['p_value']:.4f}, dof = {overall['dof']}"
        )
        print(f"  Result: {overall['interpretation']}")

        if statistical_tests.get("pairwise_comparisons"):
            print("\n  Pairwise comparisons vs baseline (α = 0.0):")
            for key, test in sorted(statistical_tests["pairwise_comparisons"].items()):
                if "error" not in test:
                    significance = (
                        "***"
                        if test["p_value"] < 0.001
                        else (
                            "**"
                            if test["p_value"] < 0.01
                            else "*" if test["p_value"] < 0.05 else "ns"
                        )
                    )
                    print(
                        f"    α = {test['alpha']:>+5.1f}: χ² = {test['chi2']:>6.2f}, p = {test['p_value']:.4f} {significance}"
                    )

    print("\n### Visualizations ###")
    print(f"  Heatmaps saved to: {args.output_dir / 'visualizations'}")
    print(f"    - Individual transition matrices per alpha")
    print(f"    - Comparison plot (up to 4 alphas side-by-side)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
