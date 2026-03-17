#!/usr/bin/env python3
"""
Whitening Steering Comparison
=============================

Compares original vs whitened SAS vectors across:
  • Single-concept steering (pitch-only, duration-only)
  • Dual-concept steering (pitch+duration, expanded_k_2x & gram_schmidt_ek2)
  • Conditioned + unconditioned generation
  • λ from -1.5 to +1.5 (step 0.5)

Measures:
  • Steering success rate (% of samples steered in the intended direction)
  • Quality degradation (pitch_class_entropy, scale_consistency, groove_consistency)

All .npy token files are saved for later listening/conversion to audio.

Usage (on EC2):
    cd /home/ubuntu/mmt

    python feature_explainability/whitening_steering_comparison.py \
        --n_songs 10 --n_unconditioned 5

    # Fewer samples for quick test
    python feature_explainability/whitening_steering_comparison.py \
        --n_songs 3 --n_unconditioned 2

    # Specific mode only
    python feature_explainability/whitening_steering_comparison.py \
        --mode single_conditioned --n_songs 5
"""

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "sparse_steering"))
sys.path.insert(0, str(PROJECT_ROOT / "sparse_steering" / "dual_steering"))
sys.path.insert(0, str(PROJECT_ROOT / "mmt"))

import torch
import torch.nn as nn

import music_x_transformers
import representation
import utils

from steered_generator_sas import (
    SASSteeringHook,
    extract_durations_from_tokens,
    extract_pitches_from_tokens,
    evaluate_quality_metrics,
    get_adaptive_k,
    load_model,
    load_sae_models,
    load_sas_vectors,
    register_steering_hooks,
    remove_hooks,
)
from dual_steered_generator import (
    register_expanded_k_hooks,
    remove_hooks as remove_dual_hooks,
)
from sparse_vector_composer import SparseVectorComposer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_SAS_DIR = PROJECT_ROOT / "exp" / "sod" / "sparse_steering" / "sas_vectors"
DEFAULT_SAE_DIR = PROJECT_ROOT / "exp" / "sod" / "sparse_steering" / "sae_checkpoints"
DEFAULT_WHITENED_DIR = PROJECT_ROOT / "feature_explainability" / "outputs" / "whitening"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "exp" / "sod" / "ape" / "checkpoints" / "best_model.pt"
)
DEFAULT_TRAIN_ARGS = PROJECT_ROOT / "exp" / "sod" / "ape" / "train-args.json"
DEFAULT_ENCODING = (
    PROJECT_ROOT / "data" / "sod" / "processed" / "notes" / "encoding.json"
)
DEFAULT_NOTES_DIR = PROJECT_ROOT / "data" / "sod" / "processed" / "notes"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "feature_explainability" / "outputs" / "whitening_comparison"
)

LAYERS_TO_STEER = [10]
SEQ_LEN = 512
CONDITIONING_BEATS = 16

LAMBDAS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
DUAL_STRATEGIES = ["expanded_k_2x", "gram_schmidt_ek2"]

GROUND_TRUTH_METRICS = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def calculate_degradation(metrics: dict) -> dict:
    """Quality degradation from ground truth."""
    gt = GROUND_TRUTH_METRICS
    ed = abs(metrics["pitch_class_entropy"] - gt["pitch_class_entropy"])
    sd = max(0, gt["scale_consistency"] - metrics["scale_consistency"])
    gd = max(0, gt["groove_consistency"] - metrics["groove_consistency"])
    return {
        "entropy_diff": float(ed),
        "scale_diff": float(sd),
        "groove_diff": float(gd),
        "total_degradation": float(ed + sd + gd),
    }


def _load_sas_vectors_dict(sas_dir: pathlib.Path) -> Dict[str, Dict[int, np.ndarray]]:
    """Load SAS vectors for both concepts from a directory."""
    result = {}
    for concept in ("average_pitch", "average_duration"):
        path = sas_dir / f"{concept}_sas_vectors.pt"
        if not path.exists():
            logger.warning(f"Not found: {path}")
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        vecs = {}
        for k, v in data.items():
            vecs[int(k)] = v.numpy() if isinstance(v, torch.Tensor) else np.array(v)
        result[concept] = vecs
    return result


def find_extreme_songs(
    notes_dir: pathlib.Path,
    encoding: dict,
    concept: str,
    n_songs: int = 10,
) -> Tuple[List[Tuple[pathlib.Path, float]], List[Tuple[pathlib.Path, float]]]:
    """Find songs with extreme values for a concept.

    Returns (low_songs, high_songs) where each is [(filepath, value), ...].
    """
    from conditioned_evaluator_sas import (
        load_song_tokens,
        calculate_initial_pitch,
        calculate_initial_duration,
    )

    calc_fn = (
        calculate_initial_pitch if "pitch" in concept else calculate_initial_duration
    )
    song_values = []

    for subfolder in notes_dir.iterdir():
        if not subfolder.is_dir():
            continue
        for filepath in subfolder.glob("*.npy"):
            try:
                tokens = load_song_tokens(filepath, encoding)
                val = calc_fn(tokens, encoding, CONDITIONING_BEATS)
                if val is not None:
                    song_values.append((filepath, val))
            except Exception:
                continue

    song_values.sort(key=lambda x: x[1])
    low = song_values[:n_songs]
    high = song_values[-n_songs:]
    return low, high


def extract_conditioning(filepath: pathlib.Path, encoding: dict) -> torch.Tensor:
    """Extract conditioning prefix from a song file."""
    from conditioned_evaluator_sas import load_song_tokens

    tokens = load_song_tokens(filepath, encoding)
    cond_len = 0
    for i, token in enumerate(tokens):
        if token[1] >= CONDITIONING_BEATS:
            cond_len = i
            break
    if cond_len == 0:
        cond_len = len(tokens)
    return torch.from_numpy(tokens[:cond_len]).long().unsqueeze(0)


def measure_sample(tokens: np.ndarray, encoding: dict) -> dict:
    """Measure pitch, duration, and quality from generated tokens."""
    pitches = extract_pitches_from_tokens(tokens, encoding)
    durations = extract_durations_from_tokens(tokens, encoding)
    qm = evaluate_quality_metrics(tokens, encoding)
    deg = calculate_degradation(qm)
    return {
        "mean_pitch": float(np.mean(pitches)) if pitches else None,
        "mean_duration": float(np.mean(durations)) if durations else None,
        "n_notes": len(pitches),
        "quality": qm,
        "degradation": deg,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Single-Concept Conditioned
# ═════════════════════════════════════════════════════════════════════════════


def run_single_conditioned(
    model,
    encoding: dict,
    sae_models: dict,
    original_vecs: Dict[str, Dict[int, np.ndarray]],
    whitened_vecs: Dict[str, Dict[int, np.ndarray]],
    device: torch.device,
    notes_dir: pathlib.Path,
    output_dir: pathlib.Path,
    n_songs: int = 10,
):
    """Single-concept conditioned steering: original vs whitened."""
    logger.info("=" * 72)
    logger.info("  Single-Concept Conditioned Steering Comparison")
    logger.info("=" * 72)

    all_results = []

    for concept in ("average_pitch", "average_duration"):
        label = "Pitch" if "pitch" in concept else "Duration"
        logger.info(f"\n{'━' * 60}")
        logger.info(f"  Concept: {label}")
        logger.info(f"{'━' * 60}")

        low_songs, high_songs = find_extreme_songs(
            notes_dir, encoding, concept, n_songs
        )
        logger.info(
            f"  Low: {[f'{v:.1f}' for _, v in low_songs]}  "
            f"High: {[f'{v:.1f}' for _, v in high_songs]}"
        )

        for vec_label, vec_dict in [
            ("original", original_vecs),
            ("whitened", whitened_vecs),
        ]:
            if concept not in vec_dict:
                continue
            sas_vectors = vec_dict[concept]

            for lam in LAMBDAS:
                # Positive λ → low songs (push up), Negative λ → high songs (push down)
                if lam > 0:
                    songs = low_songs
                    expected_direction = +1
                elif lam < 0:
                    songs = high_songs
                    expected_direction = -1
                else:
                    songs = low_songs[:3]  # Baseline: just a few
                    expected_direction = 0

                for filepath, init_val in songs:
                    cond = extract_conditioning(filepath, encoding).to(device)
                    eos = encoding["type_code_map"]["end-of-song"]

                    handles = register_steering_hooks(
                        model,
                        sae_models,
                        sas_vectors,
                        concept,
                        steering_strength=lam,
                        layers_to_steer=LAYERS_TO_STEER,
                    )
                    try:
                        with torch.no_grad():
                            generated = model.generate(
                                cond.clone(),
                                SEQ_LEN,
                                eos_token=eos,
                            )
                        continuation = generated[0].cpu().numpy()
                    finally:
                        remove_hooks(handles)

                    # Measure
                    m = measure_sample(continuation, encoding)
                    concept_key = (
                        "mean_pitch" if "pitch" in concept else "mean_duration"
                    )
                    gen_val = m[concept_key]

                    success = None
                    if gen_val is not None and expected_direction != 0:
                        success = (gen_val - init_val) * expected_direction > 0

                    # Save .npy
                    save_dir = (
                        output_dir
                        / "single_conditioned"
                        / vec_label
                        / concept
                        / f"lam{lam:+.1f}"
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)
                    full_seq = np.concatenate(
                        [cond[0].cpu().numpy(), continuation], axis=0
                    )
                    np.save(save_dir / f"{filepath.stem}.npy", full_seq)

                    result = {
                        "mode": "single_conditioned",
                        "vectors": vec_label,
                        "concept": concept,
                        "lambda": lam,
                        "song": filepath.stem,
                        "initial_value": init_val,
                        "generated_value": gen_val,
                        "delta": (gen_val - init_val) if gen_val else None,
                        "success": success,
                        **m,
                    }
                    all_results.append(result)

                    status = "✓" if success else ("○" if success is None else "✗")
                    logger.info(
                        f"  {status} [{vec_label:8s}] λ={lam:+.1f} "
                        f"init={init_val:.1f} gen={gen_val:.1f} "
                        f"deg={m['degradation']['total_degradation']:.2f}"
                        if gen_val
                        else f"  ✗ [{vec_label:8s}] λ={lam:+.1f} no notes"
                    )

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# Single-Concept Unconditioned
# ═════════════════════════════════════════════════════════════════════════════


def run_single_unconditioned(
    model,
    encoding: dict,
    sae_models: dict,
    original_vecs: Dict[str, Dict[int, np.ndarray]],
    whitened_vecs: Dict[str, Dict[int, np.ndarray]],
    device: torch.device,
    output_dir: pathlib.Path,
    n_samples: int = 5,
):
    """Single-concept unconditioned steering: original vs whitened."""
    logger.info("=" * 72)
    logger.info("  Single-Concept Unconditioned Steering Comparison")
    logger.info("=" * 72)

    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]
    all_results = []

    # First generate baseline samples (λ=0) to determine reference values
    logger.info("  Generating baseline (λ=0) samples...")
    baseline_pitches = []
    baseline_durations = []
    tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
    tgt_start[:, 0, 0] = sos

    for i in range(n_samples):
        with torch.no_grad():
            generated = model.generate(tgt_start.clone(), SEQ_LEN, eos_token=eos)
        tokens = torch.cat((tgt_start, generated), 1)[0].cpu().numpy()
        m = measure_sample(tokens, encoding)
        if m["mean_pitch"] is not None:
            baseline_pitches.append(m["mean_pitch"])
        if m["mean_duration"] is not None:
            baseline_durations.append(m["mean_duration"])

    baseline_pitch = float(np.mean(baseline_pitches)) if baseline_pitches else 60.0
    baseline_dur = float(np.mean(baseline_durations)) if baseline_durations else 5.0
    logger.info(f"  Baseline: pitch={baseline_pitch:.1f}, duration={baseline_dur:.1f}")

    for concept in ("average_pitch", "average_duration"):
        label = "Pitch" if "pitch" in concept else "Duration"
        logger.info(f"\n{'━' * 60}")
        logger.info(f"  Concept: {label}")
        logger.info(f"{'━' * 60}")

        baseline_val = baseline_pitch if "pitch" in concept else baseline_dur

        for vec_label, vec_dict in [
            ("original", original_vecs),
            ("whitened", whitened_vecs),
        ]:
            if concept not in vec_dict:
                continue
            sas_vectors = vec_dict[concept]

            for lam in LAMBDAS:
                if lam == 0.0:
                    continue  # Already measured baseline

                expected_direction = +1 if lam > 0 else -1

                for i in range(n_samples):
                    handles = register_steering_hooks(
                        model,
                        sae_models,
                        sas_vectors,
                        concept,
                        steering_strength=lam,
                        layers_to_steer=LAYERS_TO_STEER,
                    )
                    try:
                        with torch.no_grad():
                            generated = model.generate(
                                tgt_start.clone(),
                                SEQ_LEN,
                                eos_token=eos,
                            )
                        tokens = torch.cat((tgt_start, generated), 1)[0].cpu().numpy()
                    finally:
                        remove_hooks(handles)

                    m = measure_sample(tokens, encoding)
                    concept_key = (
                        "mean_pitch" if "pitch" in concept else "mean_duration"
                    )
                    gen_val = m[concept_key]

                    success = None
                    if gen_val is not None:
                        success = (gen_val - baseline_val) * expected_direction > 0

                    # Save .npy
                    save_dir = (
                        output_dir
                        / "single_unconditioned"
                        / vec_label
                        / concept
                        / f"lam{lam:+.1f}"
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)
                    np.save(save_dir / f"sample_{i}.npy", tokens)

                    result = {
                        "mode": "single_unconditioned",
                        "vectors": vec_label,
                        "concept": concept,
                        "lambda": lam,
                        "sample_idx": i,
                        "baseline_value": baseline_val,
                        "generated_value": gen_val,
                        "delta": (gen_val - baseline_val) if gen_val else None,
                        "success": success,
                        **m,
                    }
                    all_results.append(result)

                    status = "✓" if success else ("○" if success is None else "✗")
                    logger.info(
                        f"  {status} [{vec_label:8s}] λ={lam:+.1f} "
                        f"gen={gen_val:.1f} (Δ={gen_val - baseline_val:+.1f}) "
                        f"deg={m['degradation']['total_degradation']:.2f}"
                        if gen_val
                        else f"  ✗ [{vec_label:8s}] λ={lam:+.1f} no notes"
                    )

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# Dual-Concept Conditioned
# ═════════════════════════════════════════════════════════════════════════════


def run_dual_conditioned(
    model,
    encoding: dict,
    sae_models: dict,
    original_vecs: Dict[str, Dict[int, np.ndarray]],
    whitened_vecs: Dict[str, Dict[int, np.ndarray]],
    device: torch.device,
    notes_dir: pathlib.Path,
    output_dir: pathlib.Path,
    n_songs: int = 10,
):
    """Dual-concept conditioned steering: original vs whitened."""
    logger.info("=" * 72)
    logger.info("  Dual-Concept Conditioned Steering Comparison")
    logger.info("=" * 72)

    all_results = []

    # Find extreme songs for BOTH concepts
    low_pitch, high_pitch = find_extreme_songs(
        notes_dir, encoding, "average_pitch", n_songs
    )
    low_dur, high_dur = find_extreme_songs(
        notes_dir, encoding, "average_duration", n_songs
    )

    # Use low-pitch songs for positive pitch λ, high-pitch for negative
    # Same for duration. We pair them: low_pitch[i] with its own
    # initial pitch and duration values.
    from conditioned_evaluator_sas import (
        load_song_tokens,
        calculate_initial_pitch,
        calculate_initial_duration,
    )

    # Enrich with both metrics
    def enrich(songs):
        enriched = []
        for filepath, val in songs:
            tokens = load_song_tokens(filepath, encoding)
            p = calculate_initial_pitch(tokens, encoding, CONDITIONING_BEATS) or 0.0
            d = calculate_initial_duration(tokens, encoding, CONDITIONING_BEATS) or 0.0
            enriched.append((filepath, p, d))
        return enriched

    low_pitch_enriched = enrich(low_pitch)
    high_pitch_enriched = enrich(high_pitch)

    for vec_label, vec_dict in [
        ("original", original_vecs),
        ("whitened", whitened_vecs),
    ]:
        pitch_vecs = vec_dict.get("average_pitch")
        dur_vecs = vec_dict.get("average_duration")
        if pitch_vecs is None or dur_vecs is None:
            continue

        composer = SparseVectorComposer(pitch_vecs, dur_vecs)

        for strategy in DUAL_STRATEGIES:
            logger.info(f"\n{'━' * 60}")
            logger.info(f"  [{vec_label}] Strategy: {strategy}")
            logger.info(f"{'━' * 60}")

            for lam in LAMBDAS:
                if lam == 0.0:
                    lp, ld = 0.0, 0.0
                    songs = low_pitch_enriched[:3]
                    pitch_dir, dur_dir = 0, 0
                elif lam > 0:
                    lp, ld = lam, lam
                    songs = low_pitch_enriched
                    pitch_dir, dur_dir = +1, +1
                else:
                    lp, ld = lam, lam
                    songs = high_pitch_enriched
                    pitch_dir, dur_dir = -1, -1

                for filepath, init_pitch, init_dur in songs:
                    cond = extract_conditioning(filepath, encoding).to(device)
                    eos = encoding["type_code_map"]["end-of-song"]

                    # Compose vectors and register hooks
                    combined = composer.compose(lp, ld, strategy)
                    if strategy == "expanded_k_2x":
                        handles = register_expanded_k_hooks(
                            model,
                            sae_models,
                            combined,
                            LAYERS_TO_STEER,
                            k_multiplier=2.0,
                        )
                    elif strategy == "gram_schmidt_ek2":
                        handles = register_expanded_k_hooks(
                            model,
                            sae_models,
                            combined,
                            LAYERS_TO_STEER,
                            k_multiplier=2.0,
                        )
                    else:
                        from dual_steered_generator import register_dual_hooks

                        handles = register_dual_hooks(
                            model,
                            sae_models,
                            combined,
                            LAYERS_TO_STEER,
                        )

                    try:
                        with torch.no_grad():
                            generated = model.generate(
                                cond.clone(),
                                SEQ_LEN,
                                eos_token=eos,
                            )
                        continuation = generated[0].cpu().numpy()
                    finally:
                        remove_dual_hooks(handles)

                    m = measure_sample(continuation, encoding)
                    gen_pitch = m["mean_pitch"]
                    gen_dur = m["mean_duration"]

                    pitch_success = None
                    dur_success = None
                    if gen_pitch is not None and pitch_dir != 0:
                        pitch_success = (gen_pitch - init_pitch) * pitch_dir > 0
                    if gen_dur is not None and dur_dir != 0:
                        dur_success = (gen_dur - init_dur) * dur_dir > 0

                    both_success = (
                        (pitch_success is True and dur_success is True)
                        if (lp != 0 or ld != 0)
                        else None
                    )

                    # Save .npy
                    save_dir = (
                        output_dir
                        / "dual_conditioned"
                        / vec_label
                        / strategy
                        / f"lam{lam:+.1f}"
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)
                    full_seq = np.concatenate(
                        [cond[0].cpu().numpy(), continuation], axis=0
                    )
                    np.save(save_dir / f"{filepath.stem}.npy", full_seq)

                    result = {
                        "mode": "dual_conditioned",
                        "vectors": vec_label,
                        "strategy": strategy,
                        "lambda_pitch": lp,
                        "lambda_duration": ld,
                        "song": filepath.stem,
                        "initial_pitch": init_pitch,
                        "initial_duration": init_dur,
                        "generated_pitch": gen_pitch,
                        "generated_duration": gen_dur,
                        "pitch_delta": (gen_pitch - init_pitch) if gen_pitch else None,
                        "duration_delta": (gen_dur - init_dur) if gen_dur else None,
                        "pitch_success": pitch_success,
                        "duration_success": dur_success,
                        "both_success": both_success,
                        **m,
                    }
                    all_results.append(result)

                    status = (
                        "✓" if both_success else ("○" if both_success is None else "✗")
                    )
                    logger.info(
                        f"  {status} [{vec_label:8s}] {strategy} λ={lam:+.1f} "
                        f"p={gen_pitch:.1f} d={gen_dur:.1f}"
                        if gen_pitch and gen_dur
                        else f"  ✗ [{vec_label:8s}] {strategy} λ={lam:+.1f} no notes"
                    )

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# Dual-Concept Unconditioned
# ═════════════════════════════════════════════════════════════════════════════


def run_dual_unconditioned(
    model,
    encoding: dict,
    sae_models: dict,
    original_vecs: Dict[str, Dict[int, np.ndarray]],
    whitened_vecs: Dict[str, Dict[int, np.ndarray]],
    device: torch.device,
    output_dir: pathlib.Path,
    n_samples: int = 5,
    baseline_pitch: float = 60.0,
    baseline_dur: float = 5.0,
):
    """Dual-concept unconditioned steering: original vs whitened."""
    logger.info("=" * 72)
    logger.info("  Dual-Concept Unconditioned Steering Comparison")
    logger.info("=" * 72)

    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]
    tgt_start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
    tgt_start[:, 0, 0] = sos

    all_results = []

    for vec_label, vec_dict in [
        ("original", original_vecs),
        ("whitened", whitened_vecs),
    ]:
        pitch_vecs = vec_dict.get("average_pitch")
        dur_vecs = vec_dict.get("average_duration")
        if pitch_vecs is None or dur_vecs is None:
            continue

        composer = SparseVectorComposer(pitch_vecs, dur_vecs)

        for strategy in DUAL_STRATEGIES:
            logger.info(f"\n{'━' * 60}")
            logger.info(f"  [{vec_label}] Strategy: {strategy}")
            logger.info(f"{'━' * 60}")

            for lam in LAMBDAS:
                if lam == 0.0:
                    continue  # Baseline already measured

                lp, ld = lam, lam
                pitch_dir = +1 if lam > 0 else -1
                dur_dir = +1 if lam > 0 else -1

                for i in range(n_samples):
                    combined = composer.compose(lp, ld, strategy)
                    if strategy in ("expanded_k_2x", "gram_schmidt_ek2"):
                        handles = register_expanded_k_hooks(
                            model,
                            sae_models,
                            combined,
                            LAYERS_TO_STEER,
                            k_multiplier=2.0,
                        )
                    else:
                        from dual_steered_generator import register_dual_hooks

                        handles = register_dual_hooks(
                            model,
                            sae_models,
                            combined,
                            LAYERS_TO_STEER,
                        )

                    try:
                        with torch.no_grad():
                            generated = model.generate(
                                tgt_start.clone(),
                                SEQ_LEN,
                                eos_token=eos,
                            )
                        tokens = torch.cat((tgt_start, generated), 1)[0].cpu().numpy()
                    finally:
                        remove_dual_hooks(handles)

                    m = measure_sample(tokens, encoding)
                    gen_pitch = m["mean_pitch"]
                    gen_dur = m["mean_duration"]

                    pitch_success = None
                    dur_success = None
                    if gen_pitch is not None:
                        pitch_success = (gen_pitch - baseline_pitch) * pitch_dir > 0
                    if gen_dur is not None:
                        dur_success = (gen_dur - baseline_dur) * dur_dir > 0

                    both_success = pitch_success is True and dur_success is True

                    save_dir = (
                        output_dir
                        / "dual_unconditioned"
                        / vec_label
                        / strategy
                        / f"lam{lam:+.1f}"
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)
                    np.save(save_dir / f"sample_{i}.npy", tokens)

                    result = {
                        "mode": "dual_unconditioned",
                        "vectors": vec_label,
                        "strategy": strategy,
                        "lambda_pitch": lp,
                        "lambda_duration": ld,
                        "sample_idx": i,
                        "baseline_pitch": baseline_pitch,
                        "baseline_duration": baseline_dur,
                        "generated_pitch": gen_pitch,
                        "generated_duration": gen_dur,
                        "pitch_delta": (
                            gen_pitch - baseline_pitch if gen_pitch else None
                        ),
                        "duration_delta": (gen_dur - baseline_dur if gen_dur else None),
                        "pitch_success": pitch_success,
                        "duration_success": dur_success,
                        "both_success": both_success,
                        **m,
                    }
                    all_results.append(result)

                    status = "✓" if both_success else "✗"
                    logger.info(
                        f"  {status} [{vec_label:8s}] {strategy} λ={lam:+.1f} "
                        f"p={gen_pitch:.1f} d={gen_dur:.1f}"
                        if gen_pitch and gen_dur
                        else f"  ✗ [{vec_label:8s}] {strategy} λ={lam:+.1f} no notes"
                    )

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
# Analysis & Summary
# ═════════════════════════════════════════════════════════════════════════════


def analyze_results(all_results: List[dict]) -> dict:
    """Compute aggregate statistics from all results."""
    summary = {}

    # Group by (mode, vectors, concept_or_strategy, lambda)
    groups = {}
    for r in all_results:
        mode = r["mode"]
        vecs = r["vectors"]

        if "dual" in mode:
            concept_key = r.get("strategy", "unknown")
        else:
            concept_key = r.get("concept", "unknown")

        lam = r.get("lambda", r.get("lambda_pitch", 0.0))
        key = (mode, vecs, concept_key, lam)
        groups.setdefault(key, []).append(r)

    # Compute per-group statistics
    for key, group in sorted(groups.items()):
        mode, vecs, concept_key, lam = key
        steered = [r for r in group if lam != 0.0]
        if not steered:
            continue

        if "dual" in mode:
            pitch_successes = sum(1 for r in steered if r.get("pitch_success") is True)
            dur_successes = sum(1 for r in steered if r.get("duration_success") is True)
            both_successes = sum(1 for r in steered if r.get("both_success") is True)
            n = len(steered)
            pitch_deltas = [
                r.get("pitch_delta")
                for r in steered
                if r.get("pitch_delta") is not None
            ]
            dur_deltas = [
                r.get("duration_delta")
                for r in steered
                if r.get("duration_delta") is not None
            ]
            entry = {
                "n": n,
                "pitch_success_rate": pitch_successes / n if n else 0,
                "duration_success_rate": dur_successes / n if n else 0,
                "both_success_rate": both_successes / n if n else 0,
                "mean_pitch_delta": float(np.mean(pitch_deltas)) if pitch_deltas else 0,
                "mean_duration_delta": float(np.mean(dur_deltas)) if dur_deltas else 0,
                "mean_degradation": float(
                    np.nanmean([r["degradation"]["total_degradation"] for r in steered])
                ),
            }
        else:
            successes = sum(1 for r in steered if r.get("success") is True)
            n = len(steered)
            deltas = [r.get("delta") for r in steered if r.get("delta") is not None]
            entry = {
                "n": n,
                "success_rate": successes / n if n else 0,
                "mean_delta": float(np.mean(deltas)) if deltas else 0,
                "mean_degradation": float(
                    np.nanmean([r["degradation"]["total_degradation"] for r in steered])
                ),
            }

        summary[f"{mode}/{vecs}/{concept_key}/lam={lam:+.1f}"] = entry

    return summary


def print_comparison_table(summary: dict, all_results: List[dict]):
    """Print a human-readable comparison table."""
    print("\n" + "=" * 90)
    print("  WHITENING STEERING COMPARISON — SUMMARY")
    print("=" * 90)

    # Group by mode
    modes = sorted(set(k.split("/")[0] for k in summary.keys()))

    for mode in modes:
        print(f"\n{'─' * 90}")
        print(f"  MODE: {mode}")
        print(f"{'─' * 90}")

        # Get all entries for this mode
        entries = {k: v for k, v in summary.items() if k.startswith(mode)}

        # Group by concept/strategy
        concepts = sorted(set(k.split("/")[2] for k in entries.keys()))

        for concept in concepts:
            print(f"\n  {concept}:")
            ce = {k: v for k, v in entries.items() if k.split("/")[2] == concept}

            # Header
            if "dual" in mode:
                print(
                    f"  {'λ':>6} {'Vectors':>10} {'N':>4} "
                    f"{'Pitch%':>7} {'Dur%':>7} {'Both%':>7} "
                    f"{'ΔP':>7} {'ΔD':>7} {'Deg':>6}"
                )
                print("  " + "-" * 80)
            else:
                print(
                    f"  {'λ':>6} {'Vectors':>10} {'N':>4} "
                    f"{'Success%':>9} {'ΔMean':>8} {'Deg':>6}"
                )
                print("  " + "-" * 55)

            # Sort by lambda then vectors
            sorted_keys = sorted(
                ce.keys(),
                key=lambda k: (
                    float(k.split("lam=")[1]),
                    k.split("/")[1],
                ),
            )

            for k in sorted_keys:
                v = ce[k]
                parts = k.split("/")
                vecs = parts[1]
                lam_str = parts[3]

                if "dual" in mode:
                    print(
                        f"  {lam_str:>6} {vecs:>10} {v['n']:>4} "
                        f"{v['pitch_success_rate']*100:>6.1f}% "
                        f"{v['duration_success_rate']*100:>6.1f}% "
                        f"{v['both_success_rate']*100:>6.1f}% "
                        f"{v['mean_pitch_delta']:>+7.1f} "
                        f"{v['mean_duration_delta']:>+7.1f} "
                        f"{v['mean_degradation']:>6.2f}"
                    )
                else:
                    print(
                        f"  {lam_str:>6} {vecs:>10} {v['n']:>4} "
                        f"{v['success_rate']*100:>8.1f}% "
                        f"{v['mean_delta']:>+8.1f} "
                        f"{v['mean_degradation']:>6.2f}"
                    )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare original vs whitened SAS vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "single_conditioned",
            "single_unconditioned",
            "dual_conditioned",
            "dual_unconditioned",
        ],
        default="all",
        help="Which experiment modes to run (default: all)",
    )
    parser.add_argument("--sas_dir", type=str, default=str(DEFAULT_SAS_DIR))
    parser.add_argument("--whitened_dir", type=str, default=str(DEFAULT_WHITENED_DIR))
    parser.add_argument("--sae_dir", type=str, default=str(DEFAULT_SAE_DIR))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--train_args", type=str, default=str(DEFAULT_TRAIN_ARGS))
    parser.add_argument("--encoding", type=str, default=str(DEFAULT_ENCODING))
    parser.add_argument("--notes_dir", type=str, default=str(DEFAULT_NOTES_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--n_songs",
        type=int,
        default=10,
        help="Seed songs per extreme group for conditioned experiments",
    )
    parser.add_argument(
        "--n_unconditioned",
        type=int,
        default=5,
        help="Samples per lambda for unconditioned experiments",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )

    logger.info("=" * 72)
    logger.info("  Whitening Steering Comparison")
    logger.info("=" * 72)
    logger.info(f"  Device: {device}")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Lambdas: {LAMBDAS}")
    logger.info(f"  Dual strategies: {DUAL_STRATEGIES}")
    logger.info(f"  Output: {output_dir}")
    logger.info("")

    # Load model and SAEs
    model, encoding, _ = load_model(
        args.checkpoint, args.train_args, args.encoding, device
    )
    sae_models = load_sae_models(pathlib.Path(args.sae_dir), device)

    # Load both vector sets
    original_vecs = _load_sas_vectors_dict(pathlib.Path(args.sas_dir))
    whitened_vecs = _load_sas_vectors_dict(pathlib.Path(args.whitened_dir))

    logger.info(f"  Original vectors: {list(original_vecs.keys())}")
    logger.info(f"  Whitened vectors: {list(whitened_vecs.keys())}")

    all_results = []
    t0 = time.time()

    # Run experiments
    if args.mode in ("all", "single_conditioned"):
        results = run_single_conditioned(
            model,
            encoding,
            sae_models,
            original_vecs,
            whitened_vecs,
            device,
            pathlib.Path(args.notes_dir),
            output_dir,
            args.n_songs,
        )
        all_results.extend(results)

    if args.mode in ("all", "single_unconditioned"):
        results = run_single_unconditioned(
            model,
            encoding,
            sae_models,
            original_vecs,
            whitened_vecs,
            device,
            output_dir,
            args.n_unconditioned,
        )
        all_results.extend(results)

    # Get baseline values for unconditioned dual
    baseline_pitch = 60.0
    baseline_dur = 5.0
    uncond_baselines = [
        r
        for r in all_results
        if r["mode"] == "single_unconditioned" and r.get("baseline_value")
    ]
    if uncond_baselines:
        bp = [
            r["baseline_value"]
            for r in uncond_baselines
            if "pitch" in r.get("concept", "")
        ]
        bd = [
            r["baseline_value"]
            for r in uncond_baselines
            if "duration" in r.get("concept", "")
        ]
        if bp:
            baseline_pitch = bp[0]
        if bd:
            baseline_dur = bd[0]

    if args.mode in ("all", "dual_conditioned"):
        results = run_dual_conditioned(
            model,
            encoding,
            sae_models,
            original_vecs,
            whitened_vecs,
            device,
            pathlib.Path(args.notes_dir),
            output_dir,
            args.n_songs,
        )
        all_results.extend(results)

    if args.mode in ("all", "dual_unconditioned"):
        results = run_dual_unconditioned(
            model,
            encoding,
            sae_models,
            original_vecs,
            whitened_vecs,
            device,
            output_dir,
            args.n_unconditioned,
            baseline_pitch,
            baseline_dur,
        )
        all_results.extend(results)

    elapsed = time.time() - t0

    # Save raw results
    results_path = output_dir / "whitening_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Saved: {results_path}")

    # Analyze and print summary
    summary = analyze_results(all_results)
    summary_path = output_dir / "whitening_comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved: {summary_path}")

    print_comparison_table(summary, all_results)

    logger.info("")
    logger.info("=" * 72)
    logger.info(f"  ✓ Whitening Comparison Complete! ({elapsed:.0f}s)")
    logger.info("=" * 72)
    logger.info(f"  Results:  {results_path}")
    logger.info(f"  Summary:  {summary_path}")
    logger.info(f"  Tokens:   {output_dir}/*/")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
