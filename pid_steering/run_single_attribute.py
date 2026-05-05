"""Main Experiment: Single-Attribute Steering Comparison.

Generates the primary results table for the paper:
For each concept (Pitch, Duration) × each method (unconditioned, P, PI, PID)
× each alpha in the sweep grid → generate samples → compute metrics.

This produces the data for:
- Table 1: Main attribute-steering results (P, PI, PID)
- Figure 2: Attribute metric vs alpha curves

Metrics computed per generation:
- pitch_class_entropy (for pitch steering)
- scale_consistency (for pitch steering)
- groove_consistency (for duration steering)
- average_pitch (for pitch steering)
- average_duration (for duration steering)
"""

import argparse
import json
import logging
import math
import pathlib
import sys
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pid_steering.pid_vector_calculator import (
    load_diffmean_vectors,
    compute_pid_variants,
)
from pid_steering.pid_steered_generator import PIDSteeredGenerator, load_model

import config_pid
from mmt import representation

logger = logging.getLogger(__name__)


def compute_generation_metrics(sequence: np.ndarray, encoding=None) -> Dict[str, float]:
    """Compute pitch/duration/rhythm metrics from a generated sequence.

    Extracts metrics from the 6-tuple token representation.
    """
    # Token structure: (type, beat, position, pitch, duration, instrument)
    TYPE_IDX = 0
    PITCH_IDX = 3
    DURATION_IDX = 4

    # Filter note tokens (type == 3 per TYPE_CODE_MAP in representation.py)
    note_mask = sequence[:, TYPE_IDX] == 3
    if note_mask.sum() < 5:
        return {
            "n_notes": int(note_mask.sum()),
            "average_pitch": 0.0,
            "average_duration": 0.0,
            "pitch_class_entropy": 0.0,
            "scale_consistency": 0.0,
            "groove_consistency": 0.0,
        }

    pitches = sequence[note_mask, PITCH_IDX]
    durations = sequence[note_mask, DURATION_IDX]
    beats = sequence[note_mask, 1]  # beat values

    avg_pitch = pitches.mean()
    avg_duration = durations.mean()

    # Pitch class entropy
    pitch_classes = pitches % 12
    counts = np.bincount(pitch_classes, minlength=12).astype(float)
    counts = counts / counts.sum()
    counts = counts[counts > 0]
    pc_entropy = -np.sum(counts * np.log2(counts))

    # Scale consistency: fraction of notes that fit the most common 7-note scale
    # Try all major/minor scales
    major_pattern = [0, 2, 4, 5, 7, 9, 11]
    best_fit = 0
    for root in range(12):
        scale = set((root + p) % 12 for p in major_pattern)
        notes_in_scale = sum(1 for p in pitches if (p % 12) in scale)
        fit = notes_in_scale / len(pitches)
        best_fit = max(best_fit, fit)
    scale_consistency = best_fit * 100.0

    # Groove consistency: use muspy's Hamming distance formula if available,
    # otherwise fall back to note-density regularity (approximate)
    groove_consistency = np.nan
    try:
        import muspy

        if encoding is not None:
            music = representation.decode(sequence, encoding)
        else:
            music = None
        if music is not None and any(len(t.notes) > 0 for t in music.tracks):
            groove_consistency = (
                muspy.groove_consistency(music, 4 * music.resolution) * 100
            )
    except Exception:
        # Fallback: note-density regularity (not the canonical metric)
        unique_beats = np.unique(beats)
        if len(unique_beats) > 1:
            notes_per_beat = np.array(
                [np.sum(beats == b) for b in unique_beats], dtype=float
            )
            if notes_per_beat.mean() > 0:
                groove_consistency = (
                    1.0 - notes_per_beat.std() / notes_per_beat.mean()
                ) * 100
                groove_consistency = max(0.0, groove_consistency)

    return {
        "n_notes": int(note_mask.sum()),
        "average_pitch": float(avg_pitch),
        "average_duration": float(avg_duration),
        "pitch_class_entropy": float(pc_entropy),
        "scale_consistency": float(scale_consistency),
        "groove_consistency": float(groove_consistency),
    }


def run_single_experiment(
    generator: PIDSteeredGenerator,
    pid_vectors: Dict[int, torch.Tensor],
    alpha: float,
    n_samples: int,
    seq_len: int,
    device: torch.device,
    encoding=None,
) -> Dict[str, float]:
    """Run a single (method, alpha) experiment point.

    Returns:
        Dict of averaged metrics across samples.
    """
    sequences = generator.generate_samples(
        pid_vectors=pid_vectors,
        alpha=alpha,
        n_samples=n_samples,
        seq_len=seq_len,
    )

    all_metrics = []
    for seq in sequences:
        m = compute_generation_metrics(seq, encoding)
        all_metrics.append(m)

    # Average
    avg = {}
    for key in all_metrics[0]:
        values = [m[key] for m in all_metrics]
        avg[f"{key}_mean"] = float(np.nanmean(values))
        avg[f"{key}_std"] = float(np.nanstd(values))

    return avg


def main():
    parser = argparse.ArgumentParser(
        description="Run single-attribute PID steering experiments"
    )
    parser.add_argument(
        "--concepts",
        nargs="+",
        default=["average_pitch", "average_duration"],
    )
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
    parser.add_argument(
        "--alpha_grid",
        type=float,
        nargs="+",
        default=config_pid.ALPHA_GRID,
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
        default=config_pid.PID_EXPERIMENTS_DIR / "single_attribute",
    )
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    model, encoding, train_args = load_model(
        config_pid.MODEL_CHECKPOINT,
        config_pid.TRAIN_ARGS_PATH,
        config_pid.ENCODING_PATH,
        device,
    )
    generator = PIDSteeredGenerator(model, encoding)

    # Unconditioned baseline
    logger.info("Generating unconditioned baseline...")
    baseline_metrics = []
    for _ in range(args.n_samples):
        seqs = generator.generate_samples(
            pid_vectors=None,
            alpha=0.0,
            n_samples=1,
            seq_len=args.seq_len,
        )
        baseline_metrics.append(compute_generation_metrics(seqs[0], encoding))

    baseline_avg = {}
    for key in baseline_metrics[0]:
        values = [m[key] for m in baseline_metrics]
        baseline_avg[f"{key}_mean"] = float(np.nanmean(values))
        baseline_avg[f"{key}_std"] = float(np.nanstd(values))

    all_results = {"baseline": baseline_avg}

    for concept in args.concepts:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing concept: {concept}")
        logger.info(f"{'='*60}")

        # Load raw DiffMean vectors
        dm_path = config_pid.STEERING_VECTORS_DIR / f"{concept}_steering_vectors.pt"
        dm_vectors = load_diffmean_vectors(dm_path)

        # Compute P, PI, PID variants
        variants = compute_pid_variants(
            dm_vectors,
            Kp=args.Kp,
            Ki=args.Ki,
            Kd=args.Kd,
            max_I=args.max_I,
            dim=config_pid.MODEL_DIM,
        )

        concept_results = {}

        for method_name, method_vectors in variants.items():
            method_results = {}

            for alpha in args.alpha_grid:
                logger.info(
                    f"  {concept}/{method_name}/alpha={alpha}: "
                    f"generating {args.n_samples} samples..."
                )

                avg = run_single_experiment(
                    generator,
                    method_vectors,
                    alpha,
                    args.n_samples,
                    args.seq_len,
                    device,
                    encoding,
                )
                method_results[str(alpha)] = avg

                key_metric = (
                    "average_pitch_mean"
                    if "pitch" in concept
                    else "average_duration_mean"
                )
                logger.info(f"    → {key_metric}={avg.get(key_metric, 0):.2f}")

            concept_results[method_name] = method_results

        all_results[concept] = concept_results

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "single_attribute_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nSaved results to {results_path}")

    # Print summary table
    print("\n" + "=" * 110)
    print("Single-Attribute Steering Results Summary")
    print("=" * 110)

    # Print baseline first
    if "baseline" in all_results:
        b = all_results["baseline"]
        gc_baseline = b.get("groove_consistency_mean", float("nan"))
        gc_baseline_str = (
            f"{gc_baseline:.1f}%" if not math.isnan(gc_baseline) else "N/A"
        )
        print(f"\n--- Unconditioned Baseline ---")
        print(
            f"  Avg Pitch: {b.get('average_pitch_mean', 0):.2f} ± {b.get('average_pitch_std', 0):.2f}  |  "
            f"Avg Duration: {b.get('average_duration_mean', 0):.2f} ± {b.get('average_duration_std', 0):.2f}  |  "
            f"PC Entropy: {b.get('pitch_class_entropy_mean', 0):.3f}  |  "
            f"Scale Cons: {b.get('scale_consistency_mean', 0):.1f}%  |  "
            f"Groove Cons: {gc_baseline_str}  |  "
            f"N Notes: {b.get('n_notes_mean', 0):.0f}"
        )
        print(
            f"  Ground Truth — PC Entropy: {config_pid.GROUND_TRUTH['pitch_class_entropy']:.3f}  |  "
            f"Scale Cons: {config_pid.GROUND_TRUTH['scale_consistency']:.1f}%  |  "
            f"Groove Cons: {config_pid.GROUND_TRUTH['groove_consistency']:.1f}%"
        )

    for concept in args.concepts:
        print(f"\n--- {concept} ---")
        print(
            f"{'Method':<10} {'Alpha':<8} {'Avg Pitch':<12} {'Avg Dur':<10} "
            f"{'PC Ent':<10} {'Scale%':<10} {'Groove%':<10} "
            f"{'N Notes':<10} {'Degrad':<10}"
        )
        print("-" * 100)
        if concept in all_results:
            for method, alphas in all_results[concept].items():
                for alpha_str, m in alphas.items():
                    # Degradation: average relative deviation from ground truth
                    # Uses only PC entropy and scale consistency (reliable metrics)
                    gt = config_pid.GROUND_TRUTH
                    pc_dev = (
                        abs(
                            m.get("pitch_class_entropy_mean", 0)
                            - gt["pitch_class_entropy"]
                        )
                        / gt["pitch_class_entropy"]
                    )
                    sc_dev = (
                        abs(
                            m.get("scale_consistency_mean", 0) - gt["scale_consistency"]
                        )
                        / gt["scale_consistency"]
                    )
                    deviations = [pc_dev, sc_dev]

                    # Include groove if available (not NaN)
                    gc_val = m.get("groove_consistency_mean", float("nan"))
                    if not math.isnan(gc_val):
                        gc_dev = (
                            abs(gc_val - gt["groove_consistency"])
                            / gt["groove_consistency"]
                        )
                        deviations.append(gc_dev)

                    degradation = sum(deviations) / len(deviations) * 100

                    gc_str = (
                        f"{gc_val:<10.1f}" if not math.isnan(gc_val) else f"{'N/A':<10}"
                    )

                    print(
                        f"{method:<10} {alpha_str:<8} "
                        f"{m.get('average_pitch_mean', 0):<12.2f} "
                        f"{m.get('average_duration_mean', 0):<10.2f} "
                        f"{m.get('pitch_class_entropy_mean', 0):<10.3f} "
                        f"{m.get('scale_consistency_mean', 0):<10.1f} "
                        f"{gc_str} "
                        f"{m.get('n_notes_mean', 0):<10.0f} "
                        f"{degradation:<10.1f}%"
                    )
    print("=" * 110)


if __name__ == "__main__":
    main()
