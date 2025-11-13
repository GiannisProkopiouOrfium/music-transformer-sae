"""Conditioned Modality Steering Test.

This script:
1. Finds songs with extreme modality (clear major vs clear minor) in conditioning segment
2. Uses the first N beats as conditioning prompts
3. Generates continuations with steering
4. Measures bidirectional steering effect (major ↔ minor)
5. Tests if steering can override the conditioning context

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
from typing import Dict, List, Tuple

import numpy as np
import torch

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


def detect_modality_from_tokens(
    tokens: np.ndarray, encoding: Dict
) -> Tuple[str, float]:
    """Detect modality from tokens.

    Args:
        tokens: Token array (seq_len, 6)
        encoding: Encoding dictionary

    Returns:
        (mode, confidence): ("major"/"minor"/"unknown", correlation_coefficient)
    """
    try:
        note_type = encoding["type_code_map"]["note"]
        pitches = []

        for token in tokens:
            if token[0] == note_type:
                pitch = token[3]
                pitches.append(pitch)

        if len(pitches) < 10:
            return ("unknown", 0.0)

        s = stream.Stream()
        for p in pitches:
            s.append(note.Note(p))

        key = s.analyze("key")
        return (key.mode, key.correlationCoefficient)

    except Exception as e:
        logging.warning(f"Modality detection failed: {e}")
        return ("unknown", 0.0)


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
            "scale_consistency": muspy.scale_consistency(music),
            "groove_consistency": muspy.groove_consistency(music, 4 * music.resolution),
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
    List[Tuple[pathlib.Path, str, float]], List[Tuple[pathlib.Path, str, float]]
]:
    """Find songs with extreme modality (clear major vs clear minor).

    Args:
        notes_dir: Directory containing .npy files
        encoding: Encoding dictionary
        n_songs: Number of songs to find per category
        conditioning_beats: Number of beats to analyze for conditioning
        min_confidence: Minimum confidence for classification

    Returns:
        (major_songs, minor_songs) - lists of (filepath, mode, confidence)
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

                # Detect modality in conditioning region
                mode, confidence = detect_modality_from_tokens(
                    cond_tokens.numpy()[0], encoding
                )

                if confidence >= min_confidence:
                    if mode == "major":
                        major_songs.append((filepath, mode, confidence))
                    elif mode == "minor":
                        minor_songs.append((filepath, mode, confidence))

            except Exception as e:
                logging.debug(f"Error processing {filepath.name}: {e}")

    # Sort by confidence (highest first)
    major_songs.sort(key=lambda x: x[2], reverse=True)
    minor_songs.sort(key=lambda x: x[2], reverse=True)

    # Take top N
    major_songs = major_songs[:n_songs]
    minor_songs = minor_songs[:n_songs]

    logging.info(f"Found {len(major_songs)} high-confidence major songs")
    logging.info(f"  Confidences: {[f'{c:.3f}' for _, _, c in major_songs[:5]]}")
    logging.info(f"Found {len(minor_songs)} high-confidence minor songs")
    logging.info(f"  Confidences: {[f'{c:.3f}' for _, _, c in minor_songs[:5]]}")

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
    song_list: List[Tuple[pathlib.Path, str, float]],
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
        song_list: List of (filepath, mode, confidence) tuples
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

    for i, (filepath, cond_mode, cond_confidence) in enumerate(song_list):
        logging.info(f"\n{'='*70}")
        logging.info(f"Song {i+1}/{len(song_list)}: {filepath.name}")
        logging.info(f"Conditioning: {cond_mode} (confidence: {cond_confidence:.3f})")
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

            # Detect modality in generated portion
            gen_mode, gen_confidence = detect_modality_from_tokens(
                generated_only, encoding
            )

            # Evaluate quality metrics on generated portion
            quality_metrics = evaluate_quality_metrics(generated_only, encoding)

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
                "conditioning_mode": cond_mode,
                "conditioning_confidence": float(cond_confidence),
                "alpha": alpha,
                "target_layers": target_layers if target_layers else "all",
                "generated_mode": gen_mode,
                "generated_confidence": float(gen_confidence),
                "bidirectional_score": bidirectional_score,
                "mode_shift": gen_mode != cond_mode,  # Did mode change?
                "conditioning_beats": conditioning_beats,
                "continuation_tokens": len(generated_only),
                "quality_metrics": quality_metrics,
            }

            results.append(result)

            logging.info(
                f"    Generated: {gen_mode} (conf: {gen_confidence:.3f}), "
                f"Shift: {'YES' if result['mode_shift'] else 'NO'}, "
                f"Bidirectional score: {bidirectional_score:+4d}"
            )
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


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze conditioned steering results.

    Args:
        results: List of result dictionaries

    Returns:
        Analysis summary
    """
    analysis = {
        "summary_by_alpha": {},
        "summary_by_category": {},
        "steering_effectiveness": {},
    }

    # Group by alpha
    for alpha in sorted(set(r["alpha"] for r in results)):
        alpha_results = [r for r in results if r["alpha"] == alpha]
        confident = [r for r in alpha_results if r["generated_confidence"] >= 0.5]

        if confident:
            major_count = sum(1 for r in confident if r["generated_mode"] == "major")
            minor_count = sum(1 for r in confident if r["generated_mode"] == "minor")
            total = len(confident)

            # Bidirectional metric: average score
            avg_bidirectional = np.mean([r["bidirectional_score"] for r in confident])

            analysis["summary_by_alpha"][f"alpha_{alpha}"] = {
                "alpha": alpha,
                "n_samples": total,
                "major_count": major_count,
                "minor_count": minor_count,
                "major_percentage": 100 * major_count / total,
                "minor_percentage": 100 * minor_count / total,
                "avg_bidirectional_score": float(avg_bidirectional),
                "mode_shift_count": sum(r["mode_shift"] for r in confident),
                "mode_shift_percentage": 100
                * sum(r["mode_shift"] for r in confident)
                / total,
            }

    # Group by conditioning category
    for category in ["major", "minor"]:
        cat_results = [r for r in results if r["conditioning_category"] == category]

        for alpha in sorted(set(r["alpha"] for r in cat_results)):
            alpha_results = [r for r in cat_results if r["alpha"] == alpha]
            confident = [r for r in alpha_results if r["generated_confidence"] >= 0.5]

            if confident:
                major_count = sum(
                    1 for r in confident if r["generated_mode"] == "major"
                )
                minor_count = sum(
                    1 for r in confident if r["generated_mode"] == "minor"
                )
                total = len(confident)

                key = f"{category}_conditioning_alpha_{alpha}"
                analysis["summary_by_category"][key] = {
                    "conditioning": category,
                    "alpha": alpha,
                    "n_samples": total,
                    "major_percentage": 100 * major_count / total,
                    "minor_percentage": 100 * minor_count / total,
                    "mode_shift_percentage": 100
                    * sum(r["mode_shift"] for r in confident)
                    / total,
                }

    # Steering effectiveness: measure shift magnitude
    # Expected (positive α steers toward major, negative α steers toward minor):
    #   Major conditioning + positive α → stays major or reinforces major
    #   Major conditioning + negative α → shifts toward minor (fighting context)
    #   Minor conditioning + positive α → shifts toward major (fighting context)
    #   Minor conditioning + negative α → stays minor or reinforces minor

    for category in ["major", "minor"]:
        baseline_results = [
            r
            for r in results
            if r["conditioning_category"] == category
            and r["alpha"] == 0.0
            and r["generated_confidence"] >= 0.5
        ]

        if baseline_results:
            baseline_major_pct = (
                100
                * sum(1 for r in baseline_results if r["generated_mode"] == "major")
                / len(baseline_results)
            )

            analysis["steering_effectiveness"][f"{category}_baseline"] = {
                "conditioning": category,
                "baseline_major_percentage": baseline_major_pct,
                "baseline_minor_percentage": 100 - baseline_major_pct,
            }

            # Compare each alpha to baseline
            for alpha in sorted(set(r["alpha"] for r in results if r["alpha"] != 0.0)):
                alpha_results = [
                    r
                    for r in results
                    if r["conditioning_category"] == category
                    and r["alpha"] == alpha
                    and r["generated_confidence"] >= 0.5
                ]

                if alpha_results:
                    steered_major_pct = (
                        100
                        * sum(
                            1 for r in alpha_results if r["generated_mode"] == "major"
                        )
                        / len(alpha_results)
                    )

                    shift = steered_major_pct - baseline_major_pct

                    # Determine if steering worked as expected
                    # Positive α → increase major%, Negative α → decrease major%
                    if category == "major":
                        # Major conditioning: positive α should keep/increase major, negative α should decrease major
                        expected_success = (alpha > 0 and shift >= 0) or (
                            alpha < 0 and shift <= 0
                        )
                    else:  # minor
                        # Minor conditioning: positive α should increase major, negative α should keep/decrease major
                        expected_success = (alpha > 0 and shift >= 0) or (
                            alpha < 0 and shift <= 0
                        )

                    key = f"{category}_alpha_{alpha}"
                    analysis["steering_effectiveness"][key] = {
                        "conditioning": category,
                        "alpha": alpha,
                        "steered_major_percentage": steered_major_pct,
                        "shift_from_baseline": shift,
                        "expected_direction": expected_success,
                    }

    # Calculate degradation for each result
    for r in results:
        if "quality_metrics" in r and r["quality_metrics"]:
            r["degradation"] = calculate_degradation(
                r["quality_metrics"], GROUND_TRUTH_METRICS
            )

    return analysis


def rank_best_generations(results: List[Dict]) -> Dict:
    """Rank best generations by steering effect and quality.

    Score evaluates correct directional shift:
    - Major conditioning + positive α: should stay major (score = generated_major_pct)
    - Major conditioning + negative α: should shift to minor (score = generated_minor_pct)
    - Minor conditioning + positive α: should shift to major (score = generated_major_pct)
    - Minor conditioning + negative α: should stay minor (score = generated_minor_pct)

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary with ranked lists and detailed analysis
    """
    ranked_results = []

    for r in results:
        if r["generated_confidence"] < 0.5:
            continue  # Skip low confidence
        if "degradation" not in r:
            continue  # Skip if no quality metrics

        # Calculate percentages
        # Initial (conditioning)
        if r["conditioning_mode"] == "major":
            initial_major_pct = r["conditioning_confidence"] * 100
            initial_minor_pct = (1 - r["conditioning_confidence"]) * 100
        else:  # minor
            initial_minor_pct = r["conditioning_confidence"] * 100
            initial_major_pct = (1 - r["conditioning_confidence"]) * 100

        # Final (generated)
        if r["generated_mode"] == "major":
            final_major_pct = r["generated_confidence"] * 100
            final_minor_pct = (1 - r["generated_confidence"]) * 100
        else:  # minor
            final_minor_pct = r["generated_confidence"] * 100
            final_major_pct = (1 - r["generated_confidence"]) * 100

        # Calculate shift
        major_shift = final_major_pct - initial_major_pct
        minor_shift = final_minor_pct - initial_minor_pct

        # Calculate score based on expected behavior
        alpha = r["alpha"]
        conditioning = r["conditioning_category"]
        
        if conditioning == "major":
            if alpha > 0:
                # Positive α: should stay/reinforce major
                score = final_major_pct
                expected_behavior = "stay_major"
            elif alpha < 0:
                # Negative α: should shift to minor
                score = final_minor_pct
                expected_behavior = "shift_to_minor"
            else:
                # Baseline
                score = 0
                expected_behavior = "baseline"
        else:  # minor conditioning
            if alpha > 0:
                # Positive α: should shift to major
                score = final_major_pct
                expected_behavior = "shift_to_major"
            elif alpha < 0:
                # Negative α: should stay/reinforce minor
                score = final_minor_pct
                expected_behavior = "stay_minor"
            else:
                # Baseline
                score = 0
                expected_behavior = "baseline"

        # Penalize for quality degradation
        degradation = r["degradation"]["total_degradation"]
        score = score - degradation

        ranked_results.append({
            "song_name": r["song_name"],
            "conditioning": conditioning,
            "alpha": alpha,
            "initial_major_pct": initial_major_pct,
            "initial_minor_pct": initial_minor_pct,
            "final_major_pct": final_major_pct,
            "final_minor_pct": final_minor_pct,
            "major_shift": major_shift,
            "minor_shift": minor_shift,
            "degradation": degradation,
            "score": score,
            "expected_behavior": expected_behavior,
            "quality_metrics": r["quality_metrics"],
        })

    # Sort by score (high = correct behavior with low degradation)
    ranked_results.sort(key=lambda x: x["score"], reverse=True)

    # Separate by category for backward compatibility
    minor_to_major = [r for r in ranked_results if r["expected_behavior"] == "shift_to_major"]
    major_to_minor = [r for r in ranked_results if r["expected_behavior"] == "shift_to_minor"]

    return {
        "all_ranked": ranked_results,
        "minor_to_major": minor_to_major,
        "major_to_minor": major_to_minor,
    }


def main():
    parser = argparse.ArgumentParser(description="Test conditioned modality steering")
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
            "steering_interventions/modality/outputs/conditioned_evaluation"
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

    # Analyze
    logging.info("\nAnalyzing results...")
    analysis = analyze_results(all_results)

    # Save results
    results_file = args.output_dir / "conditioned_steering_results.json"
    with open(results_file, "w") as f:
        json.dump({"results": all_results, "analysis": analysis}, f, indent=2)

    logging.info(f"Saved results to: {results_file}")

    # Rank best generations
    logging.info("\nRanking best generations by steering + quality...")
    ranked = rank_best_generations(all_results)

    # Save ranked results
    ranked_file = args.output_dir / "ranked_best_generations.json"
    with open(ranked_file, "w") as f:
        json.dump(ranked, f, indent=2)

    logging.info(f"Saved ranked results to: {ranked_file}")

    # Print ranked summary
    print("\n" + "=" * 80)
    print("DETAILED STEERING RESULTS (All Songs & Alphas)")
    print("=" * 80)

    for category_name, category_key in [
        ("MINOR → MAJOR (Positive Steering)", "minor_to_major"),
        ("MAJOR → MINOR (Negative Steering)", "major_to_minor"),
    ]:
        print(f"\n### {category_name} ###")
        ranked_list = ranked[category_key]

        if not ranked_list:
            print("  (No confident examples found)")
            continue

        print(
            f"  {'Rank':<6} {'Song':<25} {'Alpha':<7} {'Score':<7} "
            f"{'Init_Maj%':<10} {'Final_Maj%':<10} {'Shift_Maj':<10} "
            f"{'Init_Min%':<10} {'Final_Min%':<10} {'Shift_Min':<10} {'Degrad':<8}"
        )
        print("  " + "-" * 120)

        # Show all examples
        for i, gen in enumerate(ranked_list, 1):
            print(
                f"  {i:<6} {gen['song_name']:<25} "
                f"{gen['alpha']:>+5.1f}  {gen['score']:>6.1f} "
                f"{gen['initial_major_pct']:>9.1f} {gen['final_major_pct']:>9.1f} "
                f"{gen['major_shift']:>+9.1f} "
                f"{gen['initial_minor_pct']:>9.1f} {gen['final_minor_pct']:>9.1f} "
                f"{gen['minor_shift']:>+9.1f} {gen['degradation']:>7.2f}"
            )

    print("=" * 80)

    # Print summary with all results by song and alpha
    print("\n" + "=" * 80)
    print("COMPLETE ANALYSIS BY SONG AND ALPHA")
    print("=" * 80)

    # Group by conditioning category
    for category in ["major", "minor"]:
        category_results = [r for r in ranked["all_ranked"] if r["conditioning"] == category]
        
        if not category_results:
            continue

        print(f"\n### {category.upper()} CONDITIONING ###")
        
        # Group by song
        songs = {}
        for r in category_results:
            if r["song_name"] not in songs:
                songs[r["song_name"]] = []
            songs[r["song_name"]].append(r)
        
        for song_name in sorted(songs.keys()):
            print(f"\n{song_name}:")
            print(
                f"  {'Alpha':<7} {'Behavior':<16} {'Init_Maj%':<10} {'Final_Maj%':<10} "
                f"{'Shift_Maj':<10} {'Init_Min%':<10} {'Final_Min%':<10} {'Shift_Min':<10} "
                f"{'Score':<7} {'Degrad':<8}"
            )
            print("  " + "-" * 110)
            
            # Sort by alpha
            song_results = sorted(songs[song_name], key=lambda x: x["alpha"])
            
            for r in song_results:
                behavior_label = r["expected_behavior"].replace("_", " ").title()
                print(
                    f"  {r['alpha']:>+5.1f}  {behavior_label:<16} "
                    f"{r['initial_major_pct']:>9.1f} {r['final_major_pct']:>9.1f} "
                    f"{r['major_shift']:>+9.1f} "
                    f"{r['initial_minor_pct']:>9.1f} {r['final_minor_pct']:>9.1f} "
                    f"{r['minor_shift']:>+9.1f} {r['score']:>6.1f} {r['degradation']:>7.2f}"
                )

    print("=" * 80)

    # Print average behavior by alpha
    print("\n" + "=" * 80)
    print("AVERAGE STEERING BEHAVIOR BY ALPHA")
    print("=" * 80)

    for category in ["major", "minor"]:
        category_results = [r for r in ranked["all_ranked"] if r["conditioning"] == category]
        
        if not category_results:
            continue

        print(f"\n### {category.upper()} CONDITIONING ###")
        print(
            f"  {'Alpha':<7} {'N':<4} {'Avg_Init_Maj%':<14} {'Avg_Final_Maj%':<14} "
            f"{'Avg_Shift_Maj':<14} {'Avg_Score':<10} {'Avg_Degrad':<11}"
        )
        print("  " + "-" * 90)
        
        # Group by alpha
        alphas_data = {}
        for r in category_results:
            alpha = r["alpha"]
            if alpha not in alphas_data:
                alphas_data[alpha] = []
            alphas_data[alpha].append(r)
        
        # Calculate averages for each alpha
        for alpha in sorted(alphas_data.keys()):
            alpha_results = alphas_data[alpha]
            n = len(alpha_results)
            avg_init_maj = np.mean([r["initial_major_pct"] for r in alpha_results])
            avg_final_maj = np.mean([r["final_major_pct"] for r in alpha_results])
            avg_shift_maj = np.mean([r["major_shift"] for r in alpha_results])
            avg_score = np.mean([r["score"] for r in alpha_results])
            avg_degrad = np.mean([r["degradation"] for r in alpha_results])
            
            print(
                f"  {alpha:>+5.1f}  {n:<4} {avg_init_maj:>13.1f} {avg_final_maj:>13.1f} "
                f"{avg_shift_maj:>+13.1f} {avg_score:>9.1f} {avg_degrad:>10.2f}"
            )

    print("=" * 80)

    # Print summary
    print("\n" + "=" * 70)
    print("CONDITIONED MODALITY STEERING SUMMARY")
    print("=" * 70)

    print("\n### Overall by Alpha ###")
    for key in sorted(analysis["summary_by_alpha"].keys()):
        summary = analysis["summary_by_alpha"][key]
        print(
            f"α = {summary['alpha']:+5.1f}: "
            f"{summary['major_percentage']:5.1f}% major, "
            f"{summary['minor_percentage']:5.1f}% minor, "
            f"Bidirectional score: {summary['avg_bidirectional_score']:+6.1f}, "
            f"Mode shifts: {summary['mode_shift_percentage']:.1f}%"
        )

    print("\n### By Conditioning Category ###")
    for category in ["major", "minor"]:
        print(f"\n{category.upper()} Conditioning:")
        for key in sorted(analysis["summary_by_category"].keys()):
            if key.startswith(category):
                summary = analysis["summary_by_category"][key]
                print(
                    f"  α = {summary['alpha']:+5.1f}: "
                    f"{summary['major_percentage']:5.1f}% major, "
                    f"Mode shifts: {summary['mode_shift_percentage']:.1f}%"
                )

    print("\n### Steering Effectiveness ###")
    for key in sorted(analysis["steering_effectiveness"].keys()):
        if "baseline" not in key:
            eff = analysis["steering_effectiveness"][key]
            symbol = "✓" if eff["expected_direction"] else "✗"
            print(
                f"{symbol} {eff['conditioning'].capitalize()} cond, α={eff['alpha']:+5.1f}: "
                f"{eff['shift_from_baseline']:+6.1f}% shift from baseline"
            )

    print("=" * 70)


if __name__ == "__main__":
    main()
