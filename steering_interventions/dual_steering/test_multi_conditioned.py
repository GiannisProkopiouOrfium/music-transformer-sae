#!/usr/bin/env python3
"""Phase 4: Conditioned Dual-Steering Testing.

Tests dual-steering ability to override strong conditioning context from extreme songs.

Test scenarios (ALL FIGHT BOTH CONCEPTS):
1. Low pitch + minor → steer high + major (fight both)
2. High pitch + major → steer low + minor (fight both)
3. Low pitch + major → steer high + minor (fight both)
4. High pitch + minor → steer low + major (fight both)

For each scenario:
- Find extreme conditioning songs
- Extract first N beats as conditioning
- Generate with various alpha combinations
- Measure steering success vs baseline
- Calculate quality degradation

Output:
- conditioned_results.json: Full metrics per config
- listening_list.json: Best context-fighting examples ranked by audibility + quality
- Generated MIDI/WAV files

Usage:
    python dual_steering/test_multi_conditioned.py \\
        --model_checkpoint path/to/model.ckpt \\
        --pitch_vectors steering_interventions/outputs/steering_vectors/average_pitch_steering_vectors.pt \\
        --modality_vectors steering_interventions/modality/outputs/steering_vectors/modality_steering_vectors.pt \\
        --output_dir steering_interventions/dual_steering/outputs/phase4_conditioned \\
        --n_songs 5 \\
        --conditioning_beats 8 \\
        --strategies gram_schmidt direct
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

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import config
import music_x_transformers
import muspy
import representation
import utils

# Import music21
try:
    from music21 import note, stream

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


def detect_key_from_tokens(tokens: np.ndarray, encoding: dict) -> tuple:
    """Detect key (tonic + mode) from tokens using music21."""
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
        return (key.tonic.name, key.mode, key.correlationCoefficient)

    except Exception as e:
        logging.warning(f"Key detection failed: {e}")
        return ("unknown", "unknown", 0.0)


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
    min_confidence: float = 0.7,
    cache_file: pathlib.Path = None,
) -> Dict[str, List[Tuple[pathlib.Path, float, str, float]]]:
    """Find songs with extreme pitch and modality characteristics.

    Ranks ALL songs by pitch and modality, then selects top N from each category.
    Results are cached to avoid re-scanning on subsequent runs.

    Args:
        notes_dir: Directory containing .npy files
        encoding: Encoding dictionary
        n_songs: Number of songs per category
        conditioning_beats: Beats to analyze for classification
        min_confidence: Minimum modality detection confidence (default: 0.5)
        cache_file: Path to cache file (if None, uses notes_dir/extreme_songs_cache.json)

    Returns:
        Dictionary with 4 categories:
        - "low_pitch_minor": (filepath, mean_pitch, mode, confidence)
        - "low_pitch_major": (filepath, mean_pitch, mode, confidence)
        - "high_pitch_minor": (filepath, mean_pitch, mode, confidence)
        - "high_pitch_major": (filepath, mean_pitch, mode, confidence)
    """
    # Setup cache file
    if cache_file is None:
        cache_file = notes_dir / "extreme_songs_cache.json"

    # Try to load from cache
    if cache_file.exists():
        logging.info(f"Loading cached extreme songs from {cache_file}...")
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)

            # Verify cache parameters match
            if (
                cached_data["conditioning_beats"] == conditioning_beats
                and cached_data["min_confidence"] == min_confidence
            ):
                # Convert cached data back to proper format
                result = {}
                for category, songs in cached_data["extreme_songs"].items():
                    result[category] = [
                        (pathlib.Path(path), pitch, mode, conf)
                        for path, pitch, mode, conf in songs[:n_songs]
                    ]

                logging.info("✅ Loaded from cache successfully")
                for category, songs in result.items():
                    if songs:
                        pitches = [p for _, p, _, _ in songs]
                        confidences = [c for _, _, _, c in songs]
                        logging.info(f"  {category.upper()}: {len(songs)} songs")
                        logging.info(
                            f"    Pitch range: {min(pitches):.1f} - {max(pitches):.1f}"
                        )
                        logging.info(
                            f"    Confidence range: {min(confidences):.2f} - {max(confidences):.2f}"
                        )
                return result
            else:
                logging.info("Cache parameters mismatch, re-scanning...")
        except Exception as e:
            logging.warning(f"Failed to load cache: {e}, re-scanning...")

    # Cache miss or invalid - scan all songs
    logging.info(f"Scanning {notes_dir} for songs with extreme characteristics...")

    candidates = {
        "low_pitch_minor": [],
        "low_pitch_major": [],
        "high_pitch_minor": [],
        "high_pitch_major": [],
    }

    # Scan for .npy files
    for subfolder in notes_dir.iterdir():
        if not subfolder.is_dir():
            continue

        for filepath in subfolder.glob("*.npy"):
            try:
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

                # Get modality
                _, mode, confidence = detect_key_from_tokens(cond_tokens, encoding)

                # Only require valid mode detection with minimum confidence
                if mode not in ["major", "minor"] or confidence < min_confidence:
                    continue

                # Classify into categories (no hard thresholds, just categorize)
                if mode == "minor":
                    candidates["low_pitch_minor"].append(
                        (filepath, mean_pitch, mode, confidence)
                    )
                    candidates["high_pitch_minor"].append(
                        (filepath, mean_pitch, mode, confidence)
                    )
                else:  # major
                    candidates["low_pitch_major"].append(
                        (filepath, mean_pitch, mode, confidence)
                    )
                    candidates["high_pitch_major"].append(
                        (filepath, mean_pitch, mode, confidence)
                    )

            except Exception as e:
                logging.debug(f"Error processing {filepath.name}: {e}")

    # Rank and select top N per category
    for category in candidates:
        if "low_pitch" in category:
            # Sort by lowest pitch (ascending), then by highest confidence
            candidates[category].sort(key=lambda x: (x[1], -x[3]))
        else:  # high_pitch
            # Sort by highest pitch (descending), then by highest confidence
            candidates[category].sort(key=lambda x: (-x[1], -x[3]))

        # Take top N
        candidates[category] = candidates[category][:n_songs]

    # Log results
    for category, songs in candidates.items():
        logging.info(f"\n{category.upper()}: {len(songs)} songs")
        if songs:
            pitches = [p for _, p, _, _ in songs]
            confidences = [c for _, _, _, c in songs]
            logging.info(f"  Pitch range: {min(pitches):.1f} - {max(pitches):.1f}")
            logging.info(
                f"  Confidence range: {min(confidences):.2f} - {max(confidences):.2f}"
            )

    # Save to cache for future runs
    try:
        cache_data = {
            "conditioning_beats": conditioning_beats,
            "min_confidence": min_confidence,
            "extreme_songs": {
                category: [
                    (str(path), pitch, mode, conf) for path, pitch, mode, conf in songs
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


def conditioned_generate_and_evaluate(
    model,
    composer,
    strategy: str,
    song_list: List[Tuple[pathlib.Path, float, str, float]],
    scenario: str,
    alpha_pitch_list: List[float],
    alpha_modality_list: List[float],
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
        strategy: "direct" or "gram_schmidt"
        song_list: List of (filepath, pitch, mode, confidence)
        scenario: Scenario name (e.g., "low_pitch_minor_to_high_major")
        alpha_pitch_list: Pitch alpha values to test
        alpha_modality_list: Modality alpha values to test
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

    for song_idx, (filepath, cond_pitch, cond_mode, cond_confidence) in enumerate(
        song_list
    ):
        logging.info(f"\n{'='*70}")
        logging.info(
            f"Song {song_idx+1}/{len(song_list)}: {filepath.name} | Scenario: {scenario}"
        )
        logging.info(
            f"Conditioning: pitch={cond_pitch:.1f}, mode={cond_mode}, conf={cond_confidence:.2f}"
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
            for alpha_modality in alpha_modality_list:
                try:
                    # Create generator
                    generator = MultiSteeringGenerator(
                        model=model,
                        composer=composer,
                        alpha_pitch=alpha_pitch,
                        alpha_modality=alpha_modality,
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
                    generated_tokens = output[0, conditioning.shape[1] :].cpu().numpy()

                    # Extract pitches
                    pitches = extract_pitches_from_tokens(generated_tokens, encoding)

                    if len(pitches) >= 10:
                        # Pitch statistics
                        gen_pitch_mean = float(np.mean(pitches))
                        gen_pitch_std = float(np.std(pitches))

                        # Detect key/modality
                        _, gen_mode, gen_confidence = detect_key_from_tokens(
                            generated_tokens, encoding
                        )

                        # Quality metrics
                        quality = evaluate_quality_metrics(generated_tokens, encoding)

                        # Degradation
                        degradation = calculate_degradation(
                            quality, GROUND_TRUTH_METRICS
                        )

                        # Calculate steering success
                        pitch_change = gen_pitch_mean - cond_pitch
                        mode_changed = gen_mode != cond_mode

                        # Determine if steering succeeded based on scenario
                        pitch_success = False
                        mode_success = False

                        # Check pitch steering (based on directional change)
                        if "low_pitch" in scenario and "to_high" in scenario:
                            # Want to increase pitch: success if generated pitch is higher
                            pitch_success = gen_pitch_mean > cond_pitch
                        elif "high_pitch" in scenario and "to_low" in scenario:
                            # Want to decrease pitch: success if generated pitch is lower
                            pitch_success = gen_pitch_mean < cond_pitch

                        # Check modality steering (based on target mode achievement)
                        if "to_high_major" in scenario or "to_low_major" in scenario:
                            # Want major mode: success if generated mode is major with good confidence
                            mode_success = gen_mode == "major" and gen_confidence >= 0.5
                        elif "to_high_minor" in scenario or "to_low_minor" in scenario:
                            # Want minor mode: success if generated mode is minor with good confidence
                            mode_success = gen_mode == "minor" and gen_confidence >= 0.5

                        result = {
                            "song_name": filepath.stem,
                            "scenario": scenario,
                            "strategy": strategy,
                            "alpha_pitch": alpha_pitch,
                            "alpha_modality": alpha_modality,
                            # Conditioning
                            "conditioning_pitch": cond_pitch,
                            "conditioning_mode": cond_mode,
                            "conditioning_confidence": cond_confidence,
                            "conditioning_beats": conditioning_beats,
                            "conditioning_tokens": conditioning.shape[1],
                            # Generated
                            "generated_pitch_mean": gen_pitch_mean,
                            "generated_pitch_std": gen_pitch_std,
                            "generated_mode": gen_mode,
                            "generated_confidence": gen_confidence,
                            "generated_n_notes": len(pitches),
                            # Steering effect
                            "pitch_change": pitch_change,
                            "mode_changed": mode_changed,
                            "pitch_steering_success": pitch_success,
                            "mode_steering_success": mode_success,
                            "overall_success": pitch_success and mode_success,
                            # Quality
                            "quality_metrics": quality,
                            "degradation": degradation,
                        }

                        results.append(result)

                        logging.info(
                            f"  α_p={alpha_pitch:+.1f}, α_m={alpha_modality:+.1f}: "
                            f"pitch={gen_pitch_mean:.1f} (Δ{pitch_change:+.1f}), "
                            f"mode={gen_mode} (conf={gen_confidence:.2f}), "
                            f"deg={degradation['total_degradation']:.2f}"
                        )

                        # Save if output_dir provided
                        if output_dir is not None:
                            save_dir = (
                                output_dir
                                / scenario
                                / strategy
                                / f"ap{alpha_pitch:+.1f}_am{alpha_modality:+.1f}"
                            )
                            save_dir.mkdir(parents=True, exist_ok=True)

                            # Save full sequence (conditioning + generated)
                            full_seq = output[0].cpu().numpy()
                            np.save(save_dir / f"{filepath.stem}.npy", full_seq)

                            # Save as MIDI/WAV
                            try:
                                music = representation.decode(full_seq, encoding)
                                music.write(str(save_dir / f"{filepath.stem}.mid"))
                                music.write_audio(
                                    str(save_dir / f"{filepath.stem}.wav")
                                )
                            except Exception as e:
                                logging.error(f"Error saving audio: {e}")

                except Exception as e:
                    logging.error(
                        f"Generation failed for α_p={alpha_pitch}, α_m={alpha_modality}: {e}"
                    )
                    continue

    return results


def extract_listening_list(results: List[Dict], top_n: int = 20) -> List[Dict]:
    """Extract best context-fighting examples ranked by audibility + quality.

    Scoring criteria:
    - Steering magnitude (how much did pitch/mode change?)
    - Quality preservation (low degradation)
    - Confidence (high confidence in detected mode)

    Args:
        results: All generation results
        top_n: Number of examples to include

    Returns:
        Ranked list of best examples
    """
    scored_results = []

    for r in results:
        if r["generated_n_notes"] < 10:
            continue

        # Calculate audibility score
        pitch_magnitude = abs(r["pitch_change"]) / 10.0  # Normalize by 10 semitones
        mode_change_score = 1.0 if r["mode_changed"] else 0.0
        confidence_score = r["generated_confidence"]

        # Calculate quality score (inverse of degradation)
        degradation = r["degradation"]["total_degradation"]
        quality_score = max(0, 1.0 - degradation / 10.0)  # Normalize by 10

        # Combined score
        audibility = (pitch_magnitude + mode_change_score + confidence_score) / 3.0
        overall_score = 0.6 * audibility + 0.4 * quality_score

        scored_results.append(
            {
                **r,
                "audibility_score": audibility,
                "quality_score": quality_score,
                "listening_score": overall_score,
            }
        )

    # Sort by listening score
    scored_results.sort(key=lambda x: x["listening_score"], reverse=True)

    return scored_results[:top_n]


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
    for scenario in set(r["scenario"] for r in results):
        scenario_results = [r for r in results if r["scenario"] == scenario]
        valid = [r for r in scenario_results if r["generated_n_notes"] >= 10]

        if valid:
            analysis["summary_by_scenario"][scenario] = {
                "n_samples": len(valid),
                "pitch_success_rate": np.mean(
                    [r["pitch_steering_success"] for r in valid]
                ),
                "mode_success_rate": np.mean(
                    [r["mode_steering_success"] for r in valid]
                ),
                "overall_success_rate": np.mean([r["overall_success"] for r in valid]),
                "mean_degradation": np.mean(
                    [r["degradation"]["total_degradation"] for r in valid]
                ),
                "mean_pitch_change": np.mean([r["pitch_change"] for r in valid]),
                "mode_change_rate": np.mean([r["mode_changed"] for r in valid]),
            }

    # Group by strategy
    for strategy in set(r["strategy"] for r in results):
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
    for scenario in set(r["scenario"] for r in results):
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
                "alpha_modality": best["alpha_modality"],
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
            "outputs/steering_vectors/average_pitch_steering_vectors.pt"
        ),
        help="Pitch steering vectors",
    )
    parser.add_argument(
        "--modality_vectors",
        type=pathlib.Path,
        default=pathlib.Path(
            "steering_interventions/modality/outputs/steering_vectors/modality_steering_vectors.pt"
        ),
        help="Modality steering vectors",
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
        default="-2.5,-2.0,-1.5,0.0,1.5,2.0,2.5",
        help="Pitch alphas (optimized from Phase 3)",
    )
    parser.add_argument(
        "--alphas_modality",
        type=str,
        default="-2.5,-2.0,-1.5,0.0,1.5,2.0,2.5",
        help="Modality alphas (optimized from Phase 3)",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["gram_schmidt", "direct"],
        help="Strategies to test",
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
        str(args.pitch_vectors), str(args.modality_vectors)
    )

    # Parse alphas
    alphas_pitch = [float(a.strip()) for a in args.alphas_pitch.split(",")]
    alphas_modality = [float(a.strip()) for a in args.alphas_modality.split(",")]

    logging.info(
        f"Testing {len(alphas_pitch)} pitch alphas × {len(alphas_modality)} modality alphas"
    )

    # Find extreme songs
    logging.info("\n" + "=" * 80)
    logging.info("FINDING EXTREME SONGS FOR CONDITIONING")
    logging.info("=" * 80)

    # Setup cache file path
    cache_file = args.output_dir / "extreme_songs_cache.json"

    extreme_songs = find_extreme_songs(
        config.NOTES_DIR,
        encoding,
        args.n_songs,
        args.conditioning_beats,
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
            "name": "low_pitch_minor_to_high_major",
            "songs": extreme_songs["low_pitch_minor"],
            "description": "Low pitch + minor → steer high + major (fight both)",
        },
        {
            "name": "high_pitch_major_to_low_minor",
            "songs": extreme_songs["high_pitch_major"],
            "description": "High pitch + major → steer low + minor (fight both)",
        },
        {
            "name": "low_pitch_major_to_high_minor",
            "songs": extreme_songs["low_pitch_major"],
            "description": "Low pitch + major → steer high + minor (fight both)",
        },
        {
            "name": "high_pitch_minor_to_low_major",
            "songs": extreme_songs["high_pitch_minor"],
            "description": "High pitch + minor → steer low + major (fight both)",
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
                alpha_modality_list=alphas_modality,
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
                    "alphas_modality": alphas_modality,
                    "strategies": args.strategies,
                    "total_generations": len(all_results),
                    "elapsed_time_seconds": elapsed_time,
                },
                "extreme_songs": {
                    k: [(str(f), p, m, c) for f, p, m, c in v]
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
                    "audibility": "Magnitude of pitch/mode change",
                    "quality": "Low degradation from ground truth",
                    "confidence": "High modality detection confidence",
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
        print(f"  Mode success: {stats['mode_success_rate']*100:.1f}%")
        print(f"  Mean degradation: {stats['mean_degradation']:.2f}")
        print(f"  Mean pitch change: {stats['mean_pitch_change']:+.1f} semitones")

    print("\n### Best Strategy ###")
    for strategy, stats in analysis["summary_by_strategy"].items():
        print(f"\n{strategy}:")
        print(f"  Overall success: {stats['overall_success_rate']*100:.1f}%")
        print(f"  Mean degradation: {stats['mean_degradation']:.2f}")

    print("\n### Best Configurations ###")
    for scenario, best_config in analysis["best_configs"].items():
        print(f"\n{scenario}:")
        print(f"  α_pitch: {best_config['alpha_pitch']:+.1f}")
        print(f"  α_modality: {best_config['alpha_modality']:+.1f}")
        print(f"  Strategy: {best_config['strategy']}")
        print(f"  Success: {best_config['success_rate']*100:.1f}%")
        print(f"  Degradation: {best_config['degradation']:.2f}")

    print("\n### Listening List (Top 5) ###")
    for i, example in enumerate(listening_list[:5], 1):
        print(f"\n{i}. {example['song_name']} - {example['scenario']}")
        print(f"   Strategy: {example['strategy']}")
        print(
            f"   α_p={example['alpha_pitch']:+.1f}, α_m={example['alpha_modality']:+.1f}"
        )
        print(
            f"   Pitch: {example['conditioning_pitch']:.1f} → {example['generated_pitch_mean']:.1f} (Δ{example['pitch_change']:+.1f})"
        )
        print(f"   Mode: {example['conditioning_mode']} → {example['generated_mode']}")
        print(
            f"   Score: {example['listening_score']:.3f} (audibility={example['audibility_score']:.2f}, quality={example['quality_score']:.2f})"
        )

    print("\n" + "=" * 80)
    print("✅ PHASE 4 COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
