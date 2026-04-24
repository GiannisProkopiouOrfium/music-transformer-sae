"""Conditioned PID Steering Evaluation.

Mirrors steering_interventions/conditioned_evaluator.py and
sparse_steering/conditioned_evaluator_sas.py for PID steering.

Pipeline:
1. Find songs with extreme initial attribute values (high/low pitch or duration)
2. Use first N beats as conditioning prefix
3. Generate continuations with P-only, PI, PID steering
4. Evaluate: concept shift (generated portion) + quality (full sequence)
5. Compare against DiffMean and SAS baselines
6. Export MIDI + WAV for qualitative evaluation

Key difference from run_single_attribute.py:
- Conditioned (not unconditioned) generation
- Direction-aware: positive alpha on low-value songs, negative on high-value
- Per-song metrics with statistical analysis
- Comparable to existing DM/SAS evaluations
"""

import argparse
import json
import logging
import math
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pid_steering.pid_vector_calculator import (
    load_diffmean_vectors,
    compute_pid_variants,
)
from pid_steering.pid_steered_generator import PIDSteeredGenerator, load_model

import config_pid
import representation

logger = logging.getLogger(__name__)


# ─── Reuse utility functions from conditioned_evaluator ──────────────────────


def load_song_tokens(filepath: pathlib.Path, encoding: Dict) -> np.ndarray:
    """Load song tokens from .npy file (5D notes → 6D codes)."""
    notes = np.load(filepath)
    if len(notes.shape) != 2 or notes.shape[1] != 5:
        raise ValueError(f"Invalid shape {notes.shape}, expected (seq_len, 5)")
    codes = representation.encode_notes(notes, encoding)
    return codes


def extract_conditioning_prefix(tokens: np.ndarray, n_beats: int = 16) -> torch.Tensor:
    """Extract first N beats as conditioning prefix.

    Returns:
        Conditioning tensor (1, cond_len, 6).
    """
    cond_len = 0
    for i, token in enumerate(tokens):
        if token[1] >= n_beats:
            cond_len = i
            break
    if cond_len == 0:
        cond_len = len(tokens)

    cond_tensor = torch.from_numpy(tokens[:cond_len]).long().unsqueeze(0)
    return cond_tensor


def calculate_initial_attribute(
    tokens: np.ndarray, encoding: Dict, concept: str, n_beats: int = 16
) -> Optional[float]:
    """Calculate average attribute (pitch or duration) in first N beats."""
    note_type = encoding["type_code_map"]["note"]
    attr_idx = 3 if "pitch" in concept else 4  # pitch=3, duration=4

    values = []
    for token in tokens:
        if token[0] == note_type and token[1] < n_beats:
            values.append(token[attr_idx])

    return float(np.mean(values)) if values else None


def find_extreme_songs(
    notes_dir: pathlib.Path,
    encoding: Dict,
    concept: str,
    n_songs: int = 5,
    conditioning_beats: int = 16,
) -> Tuple[List[Tuple[pathlib.Path, float]], List[Tuple[pathlib.Path, float]]]:
    """Find songs with extreme initial attribute values.

    Returns:
        (low_songs, high_songs) — lists of (filepath, avg_value).
    """
    logger.info(f"Scanning {notes_dir} for extreme {concept} songs...")

    song_values = []
    for subfolder in notes_dir.iterdir():
        if not subfolder.is_dir():
            continue
        for filepath in subfolder.glob("*.npy"):
            try:
                tokens = load_song_tokens(filepath, encoding)
                val = calculate_initial_attribute(
                    tokens, encoding, concept, conditioning_beats
                )
                if val is not None:
                    song_values.append((filepath, val))
            except Exception:
                pass

    song_values.sort(key=lambda x: x[1])
    low_songs = song_values[:n_songs]
    high_songs = song_values[-n_songs:]

    logger.info(f"Found {len(song_values)} valid songs")
    logger.info(f"Low: {[f'{v:.1f}' for _, v in low_songs]}")
    logger.info(f"High: {[f'{v:.1f}' for _, v in high_songs]}")

    return low_songs, high_songs


def evaluate_quality_metrics(tokens: np.ndarray, encoding: Dict) -> Dict:
    """Evaluate muspy quality metrics on full sequence."""
    try:
        import muspy

        music = representation.decode(tokens, encoding)
        if not music.tracks:
            return {
                k: np.nan
                for k in [
                    "pitch_class_entropy",
                    "scale_consistency",
                    "groove_consistency",
                ]
            }
        return {
            "pitch_class_entropy": muspy.pitch_class_entropy(music),
            "scale_consistency": muspy.scale_consistency(music) * 100,
            "groove_consistency": muspy.groove_consistency(music, 4 * music.resolution)
            * 100,
        }
    except Exception as e:
        logger.error(f"Quality eval error: {e}")
        return {
            k: np.nan
            for k in ["pitch_class_entropy", "scale_consistency", "groove_consistency"]
        }


def measure_concept_from_tokens(tokens: np.ndarray, encoding: Dict) -> Dict:
    """Measure pitch/duration stats from token sequence."""
    try:
        music = representation.decode(tokens, encoding)
        pitches, durations = [], []
        for track in music.tracks:
            for note in track.notes:
                pitches.append(note.pitch)
                durations.append(note.duration)
        if not pitches:
            return {
                "pitch_mean": 0.0,
                "pitch_std": 0.0,
                "duration_mean": 0.0,
                "duration_std": 0.0,
                "n_notes": 0,
            }
        return {
            "pitch_mean": float(np.mean(pitches)),
            "pitch_std": float(np.std(pitches)),
            "duration_mean": float(np.mean(durations)),
            "duration_std": float(np.std(durations)),
            "n_notes": len(pitches),
        }
    except Exception as e:
        logger.error(f"Measure error: {e}")
        return {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "duration_mean": 0.0,
            "duration_std": 0.0,
            "n_notes": 0,
        }


def calculate_degradation(quality: Dict) -> Dict:
    """Calculate degradation from ground truth."""
    gt = config_pid.GROUND_TRUTH
    entropy_diff = abs(
        quality.get("pitch_class_entropy", 0) - gt["pitch_class_entropy"]
    )
    scale_diff = max(0, gt["scale_consistency"] - quality.get("scale_consistency", 0))
    groove_diff = max(
        0, gt["groove_consistency"] - quality.get("groove_consistency", 0)
    )
    return {
        "entropy_diff": float(entropy_diff),
        "scale_diff": float(scale_diff),
        "groove_diff": float(groove_diff),
        "total_degradation": float(entropy_diff + scale_diff + groove_diff),
    }


# ─── Main conditioned generation + evaluation ────────────────────────────────


def conditioned_pid_evaluate(
    generator: PIDSteeredGenerator,
    pid_variants: Dict[str, Dict[int, torch.Tensor]],
    encoding: Dict,
    device: torch.device,
    song_list: List[Tuple[pathlib.Path, float]],
    category: str,
    alpha: float,
    concept: str,
    conditioning_beats: int = 16,
    continuation_len: int = 512,
    output_dir: Optional[pathlib.Path] = None,
    skip_wav: bool = False,
) -> List[Dict]:
    """Generate conditioned continuations with P/PI/PID and evaluate.

    Args:
        generator: PIDSteeredGenerator instance.
        pid_variants: Dict with "p_only", "pi", "pid" steering vectors.
        encoding: Encoding dictionary.
        device: Torch device.
        song_list: (filepath, initial_value) tuples.
        category: "low" or "high".
        alpha: Steering strength.
        concept: "average_pitch" or "average_duration".
        conditioning_beats: Beats used as prefix.
        continuation_len: Tokens to generate.
        output_dir: Where to save outputs.
        skip_wav: Skip WAV rendering.

    Returns:
        List of per-song result dictionaries.
    """
    eos = encoding["type_code_map"]["end-of-song"]
    gen_kwargs = {
        "eos_token": eos,
        "temperature": config_pid.TEMPERATURE,
        "filter_logits_fn": "top_k",
        "filter_thres": config_pid.FILTER_THRESHOLD,
        "monotonicity_dim": ("type", "beat"),
    }

    # Methods: baseline (no steering) + P/PI/PID
    method_configs = {"baseline": None}
    method_configs.update(pid_variants)

    results = []

    for i, (filepath, initial_value) in enumerate(song_list):
        logger.info(
            f"\nSong {i+1}/{len(song_list)}: {filepath.name} "
            f"(initial {concept}={initial_value:.1f})"
        )

        tokens = load_song_tokens(filepath, encoding)
        conditioning = extract_conditioning_prefix(tokens, conditioning_beats).to(
            device
        )
        cond_len = conditioning.shape[1]

        song_result = {
            "song_name": filepath.stem,
            "category": category,
            "initial_value": float(initial_value),
            "concept": concept,
            "alpha": alpha,
            "conditioning_tokens": cond_len,
            "methods": {},
        }

        for method_name, vectors in method_configs.items():
            method_alpha = alpha if method_name != "baseline" else 0.0

            # Apply steering hooks
            if vectors is not None and method_alpha != 0.0:
                generator.apply_precomputed_steering(vectors, method_alpha)
            else:
                generator.remove_steering()

            with torch.no_grad():
                generated = generator.model.generate(
                    conditioning, continuation_len, **gen_kwargs
                )

            generator.remove_steering()

            full_seq = torch.cat((conditioning, generated), 1).cpu().numpy()[0]
            generated_only = generated.cpu().numpy()[0]

            # Concept metrics on generated portion only
            gen_metrics = measure_concept_from_tokens(generated_only, encoding)

            # Quality metrics on full sequence
            quality = evaluate_quality_metrics(full_seq, encoding)
            degradation = calculate_degradation(quality)

            attr_key = "pitch_mean" if "pitch" in concept else "duration_mean"
            attr_change = gen_metrics[attr_key] - initial_value

            method_result = {
                "generated_pitch_mean": gen_metrics["pitch_mean"],
                "generated_duration_mean": gen_metrics["duration_mean"],
                "generated_n_notes": gen_metrics["n_notes"],
                "attribute_change": float(attr_change),
                "quality": quality,
                "degradation": degradation,
            }
            song_result["methods"][method_name] = method_result

            logger.info(
                f"  {method_name:8s} α={method_alpha:+.1f}: "
                f"pitch={gen_metrics['pitch_mean']:.1f}, "
                f"dur={gen_metrics['duration_mean']:.1f}, "
                f"change={attr_change:+.1f}, "
                f"degrad={degradation['total_degradation']:.2f}"
            )

            # Save outputs
            if output_dir is not None:
                save_dir = output_dir / concept / category / method_name
                save_dir.mkdir(parents=True, exist_ok=True)

                np.save(save_dir / f"{filepath.stem}_alpha{alpha}.npy", full_seq)

                try:
                    music = representation.decode(full_seq, encoding)
                    music.write(str(save_dir / f"{filepath.stem}_alpha{alpha}.mid"))
                    if not skip_wav:
                        music.write_audio(
                            str(save_dir / f"{filepath.stem}_alpha{alpha}.wav")
                        )
                except Exception as e:
                    logger.warning(f"Audio export failed: {e}")

        results.append(song_result)

    return results


def aggregate_results(results: List[Dict], concept: str) -> Dict:
    """Aggregate per-song results into summary statistics."""
    methods = set()
    for r in results:
        methods.update(r["methods"].keys())

    summary = {}
    attr_key = (
        "generated_pitch_mean" if "pitch" in concept else "generated_duration_mean"
    )

    for method in sorted(methods):
        changes = []
        degradations = []
        attr_vals = []
        entropies = []
        scales = []
        grooves = []

        for r in results:
            if method not in r["methods"]:
                continue
            m = r["methods"][method]
            changes.append(m["attribute_change"])
            degradations.append(m["degradation"]["total_degradation"])
            attr_vals.append(m[attr_key])
            entropies.append(m["quality"].get("pitch_class_entropy", np.nan))
            scales.append(m["quality"].get("scale_consistency", np.nan))
            grooves.append(m["quality"].get("groove_consistency", np.nan))

        summary[method] = {
            "n_songs": len(changes),
            "attr_change_mean": float(np.mean(changes)),
            "attr_change_std": float(np.std(changes)),
            "attr_value_mean": float(np.mean(attr_vals)),
            "degradation_mean": float(np.mean(degradations)),
            "degradation_std": float(np.std(degradations)),
            "pitch_class_entropy_mean": float(np.nanmean(entropies)),
            "scale_consistency_mean": float(np.nanmean(scales)),
            "groove_consistency_mean": float(np.nanmean(grooves)),
        }

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Conditioned PID steering evaluation")
    parser.add_argument(
        "--concepts",
        type=str,
        nargs="+",
        default=["average_pitch", "average_duration"],
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 1.5],
        help="Alpha values (applied as +α on low songs, -α on high songs)",
    )
    parser.add_argument(
        "--n_songs", type=int, default=5, help="Songs per extreme category (low/high)"
    )
    parser.add_argument(
        "--conditioning_beats", type=int, default=config_pid.CONDITIONING_BEATS
    )
    parser.add_argument(
        "--continuation_len", type=int, default=config_pid.CONTINUATION_LEN
    )
    parser.add_argument(
        "--Kp", type=float, default=config_pid.SPATIAL_PID_DEFAULTS["Kp"]
    )
    parser.add_argument(
        "--Ki", type=float, default=config_pid.SPATIAL_PID_DEFAULTS["Ki"]
    )
    parser.add_argument(
        "--Kd", type=float, default=config_pid.SPATIAL_PID_DEFAULTS["Kd"]
    )
    parser.add_argument(
        "--max_I", type=float, default=config_pid.SPATIAL_PID_DEFAULTS["max_I"]
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "conditioned",
    )
    parser.add_argument("--skip_wav", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model, encoding, _ = load_model(
        config_pid.MODEL_CHECKPOINT,
        config_pid.TRAIN_ARGS_PATH,
        config_pid.ENCODING_PATH,
        device,
    )
    generator = PIDSteeredGenerator(model, encoding)

    notes_dir = config_pid.PROJECT_ROOT / "data" / "sod" / "processed" / "notes"

    all_results = {}

    for concept in args.concepts:
        logger.info(f"\n{'='*70}")
        logger.info(f"Concept: {concept}")
        logger.info(f"{'='*70}")

        # Load DiffMean vectors and compute P/PI/PID variants
        dm_path = config_pid.STEERING_VECTORS_DIR / f"{concept}_steering_vectors.pt"
        dm_vectors = load_diffmean_vectors(dm_path)
        variants = compute_pid_variants(
            dm_vectors,
            Kp=args.Kp,
            Ki=args.Ki,
            Kd=args.Kd,
            max_I=args.max_I,
            dim=config_pid.MODEL_DIM,
        )

        # Find extreme songs
        low_songs, high_songs = find_extreme_songs(
            notes_dir,
            encoding,
            concept,
            n_songs=args.n_songs,
            conditioning_beats=args.conditioning_beats,
        )

        concept_results = []

        for alpha in args.alphas:
            # Low-value songs: steer UP (positive alpha)
            logger.info(f"\n--- Low {concept} songs, α=+{alpha} (steer up) ---")
            low_results = conditioned_pid_evaluate(
                generator,
                variants,
                encoding,
                device,
                low_songs,
                "low",
                alpha=alpha,
                concept=concept,
                conditioning_beats=args.conditioning_beats,
                continuation_len=args.continuation_len,
                output_dir=args.output_dir,
                skip_wav=args.skip_wav,
            )
            concept_results.extend(low_results)

            # High-value songs: steer DOWN (negative alpha)
            logger.info(f"\n--- High {concept} songs, α=-{alpha} (steer down) ---")
            high_results = conditioned_pid_evaluate(
                generator,
                variants,
                encoding,
                device,
                high_songs,
                "high",
                alpha=-alpha,
                concept=concept,
                conditioning_beats=args.conditioning_beats,
                continuation_len=args.continuation_len,
                output_dir=args.output_dir,
                skip_wav=args.skip_wav,
            )
            concept_results.extend(high_results)

        all_results[concept] = concept_results

        # Aggregate and print summary
        # Separate by alpha magnitude for clearer view
        for alpha in args.alphas:
            subset = [r for r in concept_results if abs(r["alpha"]) == alpha]
            if not subset:
                continue
            summary = aggregate_results(subset, concept)

            print(f"\n{'='*90}")
            print(f"Conditioned PID Results — {concept} | |α|={alpha}")
            print(f"{'='*90}")
            print(
                f"{'Method':<12} {'N':<5} {'Attr Change':<15} "
                f"{'Attr Value':<12} {'Degrad':<12} "
                f"{'PC Ent':<10} {'Scale%':<10} {'Groove%':<10}"
            )
            print("-" * 86)
            for method, s in summary.items():
                print(
                    f"{method:<12} {s['n_songs']:<5} "
                    f"{s['attr_change_mean']:+8.2f} ± {s['attr_change_std']:<5.1f} "
                    f"{s['attr_value_mean']:<12.1f} "
                    f"{s['degradation_mean']:<12.2f} "
                    f"{s['pitch_class_entropy_mean']:<10.3f} "
                    f"{s['scale_consistency_mean']:<10.1f} "
                    f"{s['groove_consistency_mean']:<10.1f}"
                )
            print(f"{'='*90}")

    # Save all results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "conditioned_pid_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
