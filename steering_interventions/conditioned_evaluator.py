"""Conditioned Steering Evaluation.

This script:
1. Finds songs with extreme initial pitch values (very high or very low)
2. Uses the first N beats as conditioning
3. Generates continuations with and without steering
4. Evaluates if steering can "correct" the pitch direction
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

import config
import music_x_transformers
import representation
import utils
from steered_generator import SteeredGenerator, load_steering_vectors


# Ground truth metrics from paper
GROUND_TRUTH_METRICS = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}


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
        baseline: Baseline quality metrics (GROUND_TRUTH_METRICS)

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


def load_song_tokens(filepath: pathlib.Path, encoding: Dict) -> np.ndarray:
    """Load song tokens from .npy file (not .pt).

    Args:
        filepath: Path to .npy file (in subfolder structure)
        encoding: Encoding dictionary

    Returns:
        Token array (seq_len, 6)
    """
    # The filepath should be a .npy file in data/sod/processed/notes/[subfolder]/[filename].npy

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Load notes in 5D format: [beat, position, pitch, duration, program]
    notes = np.load(filepath)

    # Validate shape
    if len(notes.shape) != 2 or notes.shape[1] != 5:
        raise ValueError(f"Invalid shape {notes.shape}, expected (seq_len, 5)")

    # Convert notes (5D) to codes (6D) using representation.encode_notes
    # This adds the type and instrument dimensions
    codes = representation.encode_notes(notes, encoding)

    # codes shape: (seq_len, 6) with format [type, beat, position, pitch, duration, instrument]
    return codes


def calculate_initial_pitch(
    tokens: np.ndarray, encoding: Dict, n_beats: int = 4
) -> Optional[float]:
    """Calculate average pitch in first N beats.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary
        n_beats: Number of beats to analyze

    Returns:
        Average pitch in first N beats, or None if no notes
    """
    note_type = encoding["type_code_map"]["note"]

    # Find notes in first n_beats
    pitches = []
    for i, token in enumerate(tokens):
        if token[0] == note_type:
            beat = token[1]  # Beat is in dimension 1
            if beat < n_beats:
                pitch = token[3]  # Pitch is in dimension 3
                pitches.append(pitch)

    if not pitches:
        return None

    return float(np.mean(pitches))


def calculate_initial_duration(
    tokens: np.ndarray, encoding: Dict, n_beats: int = 4
) -> Optional[float]:
    """Calculate average duration in first N beats.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary
        n_beats: Number of beats to analyze

    Returns:
        Average duration in first N beats (in ticks), or None if no notes
    """
    note_type = encoding["type_code_map"]["note"]

    # Find notes in first n_beats
    durations = []
    for i, token in enumerate(tokens):
        if token[0] == note_type:
            beat = token[1]  # Beat is in dimension 1
            if beat < n_beats:
                duration = token[4]  # Duration is in dimension 4
                durations.append(duration)

    if not durations:
        return None

    return float(np.mean(durations))


def find_extreme_pitch_songs(
    notes_dir: pathlib.Path,
    encoding: Dict,
    n_songs: int = 10,
    conditioning_beats: int = 4,
) -> Tuple[List[Tuple[pathlib.Path, float]], List[Tuple[pathlib.Path, float]]]:
    """Find songs with extreme initial pitch values.

    Args:
        notes_dir: Directory containing .npy files in subfolders
        encoding: Encoding dictionary
        n_songs: Number of songs to find per category
        conditioning_beats: Number of beats to analyze

    Returns:
        (low_pitch_songs, high_pitch_songs) - lists of (filepath, avg_pitch)
    """
    logging.info(f"Scanning {notes_dir} for songs with extreme initial pitch...")

    song_pitches = []

    # Scan subfolders for .npy files
    for subfolder in notes_dir.iterdir():
        if not subfolder.is_dir():
            continue

        for filepath in subfolder.glob("*.npy"):
            try:
                tokens = load_song_tokens(filepath, encoding)
                avg_pitch = calculate_initial_pitch(
                    tokens, encoding, conditioning_beats
                )

                if avg_pitch is not None:
                    song_pitches.append((filepath, avg_pitch))
            except Exception as e:
                logging.debug(f"Error processing {filepath.name}: {e}")

    # Sort by pitch
    song_pitches.sort(key=lambda x: x[1])

    # Get extreme songs
    low_pitch_songs = song_pitches[:n_songs]
    high_pitch_songs = song_pitches[-n_songs:]

    logging.info(f"Found {len(song_pitches)} valid songs")
    logging.info(f"Low pitch songs: {[f'{p:.1f}' for _, p in low_pitch_songs]}")
    logging.info(f"High pitch songs: {[f'{p:.1f}' for _, p in high_pitch_songs]}")

    return low_pitch_songs, high_pitch_songs


def find_extreme_duration_songs(
    notes_dir: pathlib.Path,
    encoding: Dict,
    n_songs: int = 10,
    conditioning_beats: int = 4,
) -> Tuple[List[Tuple[pathlib.Path, float]], List[Tuple[pathlib.Path, float]]]:
    """Find songs with extreme initial duration values.

    Args:
        notes_dir: Directory containing .npy files in subfolders
        encoding: Encoding dictionary
        n_songs: Number of songs to find per category
        conditioning_beats: Number of beats to analyze

    Returns:
        (low_duration_songs, high_duration_songs) - lists of (filepath, avg_duration)
    """
    logging.info(f"Scanning {notes_dir} for songs with extreme initial duration...")

    song_durations = []

    # Scan subfolders for .npy files
    for subfolder in notes_dir.iterdir():
        if not subfolder.is_dir():
            continue

        for filepath in subfolder.glob("*.npy"):
            try:
                tokens = load_song_tokens(filepath, encoding)
                avg_duration = calculate_initial_duration(
                    tokens, encoding, conditioning_beats
                )

                if avg_duration is not None:
                    song_durations.append((filepath, avg_duration))
            except Exception as e:
                logging.debug(f"Error processing {filepath.name}: {e}")

    # Sort by duration
    song_durations.sort(key=lambda x: x[1])

    # Get extreme songs
    low_duration_songs = song_durations[:n_songs]
    high_duration_songs = song_durations[-n_songs:]

    logging.info(f"Found {len(song_durations)} valid songs")
    logging.info(f"Low duration songs: {[f'{d:.1f}' for _, d in low_duration_songs]}")
    logging.info(f"High duration songs: {[f'{d:.1f}' for _, d in high_duration_songs]}")

    return low_duration_songs, high_duration_songs


def extract_conditioning_prefix(tokens: np.ndarray, n_beats: int = 4) -> torch.Tensor:
    """Extract first N beats as conditioning.

    Args:
        tokens: Full song tokens (seq_len, 6)
        n_beats: Number of beats to extract

    Returns:
        Conditioning tokens (1, cond_len, 6)
    """
    # Find where conditioning ends
    max_beat = n_beats
    cond_len = 0

    for i, token in enumerate(tokens):
        beat = token[1]
        if beat >= max_beat:
            cond_len = i
            break

    if cond_len == 0:
        cond_len = len(tokens)

    # Extract and convert to tensor
    cond_tokens = tokens[:cond_len]
    cond_tensor = torch.from_numpy(cond_tokens).long().unsqueeze(0)  # Add batch dim

    return cond_tensor


def measure_pitch_from_tokens(tokens: np.ndarray, encoding: Dict) -> Dict:
    """Measure pitch and duration statistics from generated tokens."""
    try:
        music = representation.decode(tokens, encoding)

        pitches = []
        durations = []
        for track in music.tracks:
            for note in track.notes:
                pitches.append(note.pitch)
                durations.append(note.duration)

        if not pitches:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0,
                "max": 0,
                "n_notes": 0,
                "duration_mean": 0.0,
                "duration_std": 0.0,
            }

        return {
            "mean": float(np.mean(pitches)),
            "std": float(np.std(pitches)),
            "min": int(np.min(pitches)),
            "max": int(np.max(pitches)),
            "n_notes": len(pitches),
            "duration_mean": float(np.mean(durations)),
            "duration_std": float(np.std(durations)),
        }
    except Exception as e:
        logging.error(f"Error decoding: {e}")
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "n_notes": 0,
            "duration_mean": 0.0,
            "duration_std": 0.0,
            "error": str(e),
        }


def conditioned_generate_and_evaluate(
    model,
    steering_vectors: Dict[int, torch.Tensor],
    encoding: Dict,
    device: torch.device,
    song_list: List[Tuple[pathlib.Path, float]],
    category: str,
    alpha: float,
    conditioning_beats: int,
    continuation_len: int,
    output_dir: pathlib.Path,
) -> List[Dict]:
    """Generate conditioned continuations with and without steering.

    Args:
        model: The model
        steering_vectors: Steering vectors
        encoding: Encoding dictionary
        device: Device
        song_list: List of (filepath, initial_pitch) tuples
        category: "low_pitch" or "high_pitch"
        alpha: Steering strength (0 for baseline)
        conditioning_beats: Beats to use as conditioning
        continuation_len: Length to generate
        output_dir: Where to save results

    Returns:
        List of result dictionaries
    """
    generator = SteeredGenerator(model, steering_vectors, encoding)
    eos = encoding["type_code_map"]["end-of-song"]

    results = []

    for i, (filepath, initial_pitch) in enumerate(song_list):
        logging.info(f"\n{'='*60}")
        logging.info(f"Song {i+1}/{len(song_list)}: {filepath.name}")
        logging.info(f"Initial pitch: {initial_pitch:.1f}, Alpha: {alpha}")
        logging.info(f"{'='*60}")

        # Load and extract conditioning
        tokens = load_song_tokens(filepath, encoding)
        conditioning = extract_conditioning_prefix(tokens, conditioning_beats)
        conditioning = conditioning.to(device)

        logging.info(
            f"Conditioning: {conditioning.shape[1]} tokens ({conditioning_beats} beats)"
        )

        # Calculate conditioning duration for comparison
        conditioning_duration = (
            calculate_initial_duration(tokens, encoding, conditioning_beats) or 0.0
        )

        # Generate continuation
        generated = generator.generate(
            conditioning,
            continuation_len,
            alpha=alpha,
            target_layers=None,
            eos_token=eos,
            temperature=config.GENERATION_TEMPERATURE,
            filter_logits_fn=config.GENERATION_FILTER,
            filter_thres=config.GENERATION_FILTER_THRESHOLD,
            monotonicity_dim=("type", "beat"),
        )

        # Combine conditioning + generated
        full_seq = torch.cat((conditioning, generated), 1).cpu().numpy()[0]

        # Measure pitch in GENERATED portion only (exclude conditioning)
        generated_only = generated.cpu().numpy()[0]
        metrics = measure_pitch_from_tokens(generated_only, encoding)

        # Also measure full sequence for reference
        full_metrics = measure_pitch_from_tokens(full_seq, encoding)

        # Evaluate quality metrics on full sequence
        quality_metrics = evaluate_quality_metrics(full_seq, encoding)
        degradation = calculate_degradation(quality_metrics, GROUND_TRUTH_METRICS)

        result = {
            "song_name": filepath.stem,
            "category": category,
            "initial_pitch": float(initial_pitch),
            "initial_duration": float(conditioning_duration),
            "alpha": alpha,
            "conditioning_beats": conditioning_beats,
            "conditioning_tokens": conditioning.shape[1],
            "conditioning_duration": float(conditioning_duration),
            "generated_mean_pitch": metrics["mean"],
            "generated_std_pitch": metrics["std"],
            "generated_mean_duration": metrics["duration_mean"],
            "generated_std_duration": metrics["duration_std"],
            "generated_n_notes": metrics["n_notes"],
            "full_mean_pitch": full_metrics["mean"],
            "full_mean_duration": full_metrics["duration_mean"],
            "full_n_notes": full_metrics["n_notes"],
            "pitch_change": float(metrics["mean"] - initial_pitch),
            "duration_change": float(metrics["duration_mean"] - conditioning_duration),
            "quality_metrics": quality_metrics,
            "degradation": degradation,
        }

        results.append(result)

        logging.info(
            f"Conditioning: pitch={initial_pitch:.1f}, duration={conditioning_duration:.2f} ticks"
        )
        logging.info(
            f"Generated: {metrics['n_notes']} notes, "
            f"mean pitch={metrics['mean']:.1f}, "
            f"mean duration={metrics['duration_mean']:.2f} ticks"
        )
        logging.info(
            f"Change: pitch={metrics['mean'] - initial_pitch:+.1f}, "
            f"duration={metrics['duration_mean'] - conditioning_duration:+.2f} ticks"
        )
        logging.info(
            f"Quality: entropy={quality_metrics.get('pitch_class_entropy', 0):.3f}, "
            f"scale={quality_metrics.get('scale_consistency', 0):.1f}%, "
            f"groove={quality_metrics.get('groove_consistency', 0):.1f}%, "
            f"degradation={degradation.get('total_degradation', 0):.2f}"
        )

        # Save if output_dir provided
        if output_dir is not None:
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


def analyze_conditioned_results(results: List[Dict]) -> Dict:
    """Analyze conditioned generation results.

    Args:
        results: List of result dictionaries

    Returns:
        Analysis summary
    """
    # Group by category and alpha
    grouped = {}
    for r in results:
        key = (r["category"], r["alpha"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)

    analysis = {
        "groups": {},
        "comparisons": {},
    }

    # Summarize each group
    for (category, alpha), group_results in grouped.items():
        valid = [r for r in group_results if r["generated_n_notes"] > 0]

        if valid:
            analysis["groups"][f"{category}_alpha_{alpha}"] = {
                "n_samples": len(valid),
                "mean_initial_pitch": float(
                    np.mean([r["initial_pitch"] for r in valid])
                ),
                "mean_initial_duration": float(
                    np.mean([r.get("initial_duration", 0) for r in valid])
                ),
                "mean_generated_pitch": float(
                    np.mean([r["generated_mean_pitch"] for r in valid])
                ),
                "mean_generated_duration": float(
                    np.mean([r["generated_mean_duration"] for r in valid])
                ),
                "std_generated_pitch": float(
                    np.std([r["generated_mean_pitch"] for r in valid])
                ),
                "std_generated_duration": float(
                    np.std([r["generated_mean_duration"] for r in valid])
                ),
                "pitch_change": float(
                    np.mean(
                        [r["generated_mean_pitch"] - r["initial_pitch"] for r in valid]
                    )
                ),
                "duration_change": float(
                    np.mean(
                        [
                            r["generated_mean_duration"]
                            - r.get("initial_duration", r["generated_mean_duration"])
                            for r in valid
                        ]
                    )
                ),
            }

    # Compare steering vs baseline for each category
    for category in ["low_pitch", "high_pitch"]:
        baseline_key = f"{category}_alpha_0.0"

        if baseline_key in analysis["groups"]:
            baseline = analysis["groups"][baseline_key]

            # Find steering results
            for key in analysis["groups"]:
                if key.startswith(category) and key != baseline_key:
                    steered = analysis["groups"][key]
                    alpha = float(key.split("_")[-1])

                    comparison_key = f"{category}_alpha_{alpha}_vs_baseline"

                    # Determine success criteria based on category type
                    is_pitch_category = "pitch" in category
                    is_duration_category = "duration" in category

                    if is_pitch_category:
                        success = (
                            alpha > 0
                            and steered["mean_generated_pitch"]
                            > baseline["mean_generated_pitch"]
                        ) or (
                            alpha < 0
                            and steered["mean_generated_pitch"]
                            < baseline["mean_generated_pitch"]
                        )
                    elif is_duration_category:
                        success = (
                            alpha > 0
                            and steered["mean_generated_duration"]
                            > baseline["mean_generated_duration"]
                        ) or (
                            alpha < 0
                            and steered["mean_generated_duration"]
                            < baseline["mean_generated_duration"]
                        )
                    else:
                        success = False

                    analysis["comparisons"][comparison_key] = {
                        "category": category,
                        "alpha": alpha,
                        "baseline_mean_pitch": baseline["mean_generated_pitch"],
                        "baseline_mean_duration": baseline["mean_generated_duration"],
                        "steered_mean_pitch": steered["mean_generated_pitch"],
                        "steered_mean_duration": steered["mean_generated_duration"],
                        "pitch_difference": steered["mean_generated_pitch"]
                        - baseline["mean_generated_pitch"],
                        "duration_difference": steered["mean_generated_duration"]
                        - baseline["mean_generated_duration"],
                        "success": success,
                    }

    return analysis


def generate_listening_priority_list(
    results: List[Dict], output_dir: pathlib.Path
) -> None:
    """Generate a listening priority list based on quality and steering effectiveness.

    Args:
        results: List of result dictionaries
        output_dir: Output directory to save the list
    """
    # Filter out results with errors or no notes
    valid_results = [
        r
        for r in results
        if r.get("generated_n_notes", 0) > 0
        and "degradation" in r
        and not np.isnan(r["degradation"].get("total_degradation", np.nan))
    ]

    # Calculate steering score for each result
    for r in valid_results:
        # Check if steering is in correct direction
        is_duration_concept = "duration" in r["category"]

        if is_duration_concept:
            # For duration: low_duration + positive α should increase, high_duration + negative α should decrease
            if "low" in r["category"] and r["alpha"] > 0:
                steering_score = r["duration_change"]  # Positive change is good
            elif "high" in r["category"] and r["alpha"] < 0:
                steering_score = -r["duration_change"]  # Negative change is good
            else:
                steering_score = 0
        else:
            # For pitch: same logic
            if "low" in r["category"] and r["alpha"] > 0:
                steering_score = r["pitch_change"]
            elif "high" in r["category"] and r["alpha"] < 0:
                steering_score = -r["pitch_change"]
            else:
                steering_score = 0

        r["steering_score"] = abs(steering_score)  # Magnitude of correct steering

        # Combined score: high steering effect + low degradation
        # Normalize: steering_score (0-50) and degradation (0-100)
        normalized_steering = min(r["steering_score"] / 50.0, 1.0)
        normalized_degradation = 1.0 - min(
            r["degradation"]["total_degradation"] / 100.0, 1.0
        )
        r["priority_score"] = (
            normalized_steering * 0.6 + normalized_degradation * 0.4
        ) * 100

    # Sort by priority score (highest first)
    valid_results.sort(key=lambda x: x["priority_score"], reverse=True)

    # Generate listening list
    listening_list = []
    listening_list.append("=" * 100)
    listening_list.append("LISTENING PRIORITY LIST")
    listening_list.append(
        "Ranked by: Steering Effectiveness (60%) + Low Degradation (40%)"
    )
    listening_list.append("=" * 100)
    listening_list.append("")

    for i, r in enumerate(valid_results[:20], 1):  # Top 20
        listening_list.append(f"{i}. [{r['priority_score']:.1f}] {r['song_name']}")
        listening_list.append(f"   Category: {r['category']}, Alpha: {r['alpha']:+.1f}")
        listening_list.append(
            f"   Steering: {r['steering_score']:.2f}, Degradation: {r['degradation']['total_degradation']:.2f}"
        )

        if "duration" in r["category"]:
            listening_list.append(
                f"   Duration: {r['conditioning_duration']:.1f} → {r['generated_mean_duration']:.1f} ticks "
                f"(change: {r['duration_change']:+.1f})"
            )
        else:
            listening_list.append(
                f"   Pitch: {r['initial_pitch']:.1f} → {r['generated_mean_pitch']:.1f} "
                f"(change: {r['pitch_change']:+.1f})"
            )

        listening_list.append(
            f"   Quality: entropy={r['quality_metrics']['pitch_class_entropy']:.2f}, "
            f"scale={r['quality_metrics']['scale_consistency']:.1f}%, "
            f"groove={r['quality_metrics']['groove_consistency']:.1f}%"
        )

        # File path
        alpha_str = f"alpha_{r['alpha']}"
        file_path = output_dir / r["category"] / alpha_str / f"{r['song_name']}.wav"
        listening_list.append(f"   File: {file_path.relative_to(output_dir)}")
        listening_list.append("")

    # Save to file
    list_file = output_dir / "listening_priority_list.txt"
    with open(list_file, "w") as f:
        f.write("\n".join(listening_list))

    # Also print to console
    print("\n" + "\n".join(listening_list[:200]))  # Print first few
    print(f"\nFull listening list saved to: {list_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate conditioned steering")
    parser.add_argument(
        "--concept", type=str, default="average_pitch", help="Concept to evaluate"
    )
    parser.add_argument("--n_songs", type=int, default=5, help="Songs per category")
    parser.add_argument(
        "--conditioning_beats", type=int, default=4, help="Beats for conditioning"
    )
    parser.add_argument(
        "--continuation_len", type=int, default=256, help="Tokens to generate"
    )
    parser.add_argument(
        "--alphas", type=str, default="0.0,0.5,-0.5", help="Alpha values to test"
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU number")
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config.OUTPUT_DIR / "conditioned_evaluation",
        help="Output directory",
    )
    # add experiment name subfolder to output_dir
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Subfolder name for this experiment",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.experiment_name is not None:
        args.output_dir = args.output_dir / args.experiment_name

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Load steering vectors
    steering_path = (
        config.OUTPUT_DIR / "steering_vectors" / f"{args.concept}_steering_vectors.pt"
    )
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

    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    logging.info("Model loaded")

    # Find extreme songs based on concept
    if args.concept == "average_duration":
        low_songs, high_songs = find_extreme_duration_songs(
            config.NOTES_DIR, encoding, args.n_songs, args.conditioning_beats
        )
        low_category = "low_duration"
        high_category = "high_duration"
    else:  # default to pitch
        low_songs, high_songs = find_extreme_pitch_songs(
            config.NOTES_DIR, encoding, args.n_songs, args.conditioning_beats
        )
        low_category = "low_pitch"
        high_category = "high_pitch"

    # Parse alphas
    alphas = [float(a.strip()) for a in args.alphas.split(",")]

    # Generate and evaluate
    all_results = []

    for alpha in alphas:
        logging.info(f"\n{'='*80}")
        logging.info(f"ALPHA = {alpha}")
        logging.info(f"{'='*80}")

        # Low category songs
        logging.info(f"\nProcessing {low_category.upper().replace('_', ' ')} songs...")
        low_results = conditioned_generate_and_evaluate(
            model,
            steering_vectors,
            encoding,
            device,
            low_songs,
            low_category,
            alpha,
            args.conditioning_beats,
            args.continuation_len,
            args.output_dir,
        )
        all_results.extend(low_results)

        # High category songs
        logging.info(f"\nProcessing {high_category.upper().replace('_', ' ')} songs...")
        high_results = conditioned_generate_and_evaluate(
            model,
            steering_vectors,
            encoding,
            device,
            high_songs,
            high_category,
            alpha,
            args.conditioning_beats,
            args.continuation_len,
            args.output_dir,
        )
        all_results.extend(high_results)

    # Analyze
    logging.info("\nAnalyzing results...")
    analysis = analyze_conditioned_results(all_results)

    # Save results
    results_file = args.output_dir / "conditioned_results.json"
    with open(results_file, "w") as f:
        json.dump({"results": all_results, "analysis": analysis}, f, indent=2)

    logging.info(f"Saved results to: {results_file}")

    # Generate listening priority list
    logging.info("\nGenerating listening priority list...")
    generate_listening_priority_list(all_results, args.output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("CONDITIONED STEERING EVALUATION SUMMARY")
    print("=" * 80)

    for comp_key, comp in analysis["comparisons"].items():
        print(
            f"\n{comp['category'].upper().replace('_', ' ')} songs, alpha={comp['alpha']}"
        )
        print(f"  Baseline:")
        print(f"    Pitch:    {comp['baseline_mean_pitch']:.1f}")
        print(f"    Duration: {comp['baseline_mean_duration']:.2f} ticks")
        print(f"  Steered:")
        print(f"    Pitch:    {comp['steered_mean_pitch']:.1f}")
        print(f"    Duration: {comp['steered_mean_duration']:.2f} ticks")
        print(f"  Change:")
        print(f"    Pitch:    {comp['pitch_difference']:+.1f}")
        print(
            f"    Duration: {comp['duration_difference']:+.2f} ticks ({(comp['duration_difference']/comp['baseline_mean_duration']*100):+.1f}%)"
        )
        print(f"  Success: {'✓' if comp['success'] else '✗'}")

    print("=" * 80)
    logging.info("Evaluation complete!")


if __name__ == "__main__":
    main()
