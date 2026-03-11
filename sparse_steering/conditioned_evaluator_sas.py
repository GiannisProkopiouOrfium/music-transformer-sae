"""
Conditioned Evaluation of Sparse Activation Steering (SAS).

Finds songs with extreme initial pitch/duration values, uses the first N beats
as conditioning, then generates continuations with SAS steering to evaluate
whether the steering can intervene (push the continuation in the opposite
direction of the conditioning).

This is the SAS counterpart of steering_interventions/conditioned_evaluator.py
which uses DiffMean steering vectors.

Usage:
    # Evaluate pitch steering at layer 10
    python sparse_steering/conditioned_evaluator_sas.py \
        --concept average_pitch \
        --n_songs 5 \
        --conditioning_beats 16 \
        --lambdas 0.0,0.25,0.5,0.75,-0.25,-0.5,-0.75,-1.0,-1.5

    # Evaluate duration steering at layer 10
    python sparse_steering/conditioned_evaluator_sas.py \
        --concept average_duration \
        --n_songs 5 \
        --conditioning_beats 16 \
        --lambdas 0.0,0.5,1.0,1.5,-0.5,-1.0,-1.5
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats as scipy_stats

# Add mmt directory to path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))

import music_x_transformers
import representation
import utils

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Default paths (relative to repository root) ─────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "exp" / "sod" / "ape" / "checkpoints" / "best_model.pt"
DEFAULT_TRAIN_ARGS = REPO_ROOT / "exp" / "sod" / "ape" / "train-args.json"
DEFAULT_ENCODING = REPO_ROOT / "data" / "sod" / "processed" / "notes" / "encoding.json"
DEFAULT_SAE_DIR = REPO_ROOT / "exp" / "sod" / "sparse_steering" / "sae_checkpoints"
DEFAULT_SAS_DIR = REPO_ROOT / "exp" / "sod" / "sparse_steering" / "sas_vectors"
DEFAULT_NOTES_DIR = REPO_ROOT / "data" / "sod" / "processed" / "notes"

# Ground truth quality metrics from the paper
GROUND_TRUTH_METRICS = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}


# ── SAS infrastructure imports ──────────────────────────────────────────────


def _import_sas_modules():
    """Import SAS modules from steered_generator_sas."""
    sparse_dir = pathlib.Path(__file__).parent
    if str(sparse_dir) not in sys.path:
        sys.path.insert(0, str(sparse_dir))

    from steered_generator_sas import (
        load_model,
        load_sae_models,
        load_sas_vectors,
        register_steering_hooks,
        remove_hooks,
    )
    from smooth_steering_sas import register_smooth_steering_hooks

    return (
        load_model,
        load_sae_models,
        load_sas_vectors,
        register_steering_hooks,
        remove_hooks,
        register_smooth_steering_hooks,
    )


# ── Quality evaluation helpers ───────────────────────────────────────────────


def evaluate_quality_metrics(tokens: np.ndarray, encoding: dict) -> dict:
    """Evaluate objective quality metrics using muspy."""
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
        logger.error(f"Error evaluating quality: {e}")
        return {
            "pitch_class_entropy": np.nan,
            "scale_consistency": np.nan,
            "groove_consistency": np.nan,
            "error": str(e),
        }


def calculate_degradation(metrics: dict, baseline: dict) -> dict:
    """Calculate quality degradation from ground-truth baseline."""
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


# ── Song loading and feature extraction ──────────────────────────────────────


def load_song_tokens(filepath: pathlib.Path, encoding: dict) -> np.ndarray:
    """Load song tokens from .npy file (5-D notes) and convert to 6-D codes.

    Returns:
        Token array (seq_len, 6): [type, beat, position, pitch, duration, instrument]
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    notes = np.load(filepath)
    if len(notes.shape) != 2 or notes.shape[1] != 5:
        raise ValueError(f"Invalid shape {notes.shape}, expected (seq_len, 5)")

    codes = representation.encode_notes(notes, encoding)
    return codes


def calculate_initial_pitch(
    tokens: np.ndarray, encoding: dict, n_beats: int = 16
) -> Optional[float]:
    """Average MIDI pitch in the first *n_beats*."""
    note_type = encoding["type_code_map"]["note"]
    pitches = []
    for token in tokens:
        if token[0] == note_type and token[1] < n_beats:
            pitches.append(token[3])
    return float(np.mean(pitches)) if pitches else None


def calculate_initial_duration(
    tokens: np.ndarray, encoding: dict, n_beats: int = 16
) -> Optional[float]:
    """Average duration (ticks) in the first *n_beats*."""
    note_type = encoding["type_code_map"]["note"]
    durations = []
    for token in tokens:
        if token[0] == note_type and token[1] < n_beats:
            durations.append(token[4])
    return float(np.mean(durations)) if durations else None


def find_extreme_pitch_songs(
    notes_dir: pathlib.Path,
    encoding: dict,
    n_songs: int = 10,
    conditioning_beats: int = 16,
) -> Tuple[List[Tuple[pathlib.Path, float]], List[Tuple[pathlib.Path, float]]]:
    """Find songs with extreme initial pitch values."""
    logger.info(
        f"Scanning {notes_dir} for extreme-pitch songs (first {conditioning_beats} beats)..."
    )

    song_pitches = []
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
                logger.debug(f"Error processing {filepath.name}: {e}")

    song_pitches.sort(key=lambda x: x[1])
    low = song_pitches[:n_songs]
    high = song_pitches[-n_songs:]

    logger.info(f"Found {len(song_pitches)} valid songs")
    logger.info(f"Low-pitch:  {[f'{p:.1f}' for _, p in low]}")
    logger.info(f"High-pitch: {[f'{p:.1f}' for _, p in high]}")
    return low, high


def find_extreme_duration_songs(
    notes_dir: pathlib.Path,
    encoding: dict,
    n_songs: int = 10,
    conditioning_beats: int = 16,
) -> Tuple[List[Tuple[pathlib.Path, float]], List[Tuple[pathlib.Path, float]]]:
    """Find songs with extreme initial duration values."""
    logger.info(
        f"Scanning {notes_dir} for extreme-duration songs (first {conditioning_beats} beats)..."
    )

    song_durations = []
    for subfolder in notes_dir.iterdir():
        if not subfolder.is_dir():
            continue
        for filepath in subfolder.glob("*.npy"):
            try:
                tokens = load_song_tokens(filepath, encoding)
                avg_dur = calculate_initial_duration(
                    tokens, encoding, conditioning_beats
                )
                if avg_dur is not None:
                    song_durations.append((filepath, avg_dur))
            except Exception as e:
                logger.debug(f"Error processing {filepath.name}: {e}")

    song_durations.sort(key=lambda x: x[1])
    low = song_durations[:n_songs]
    high = song_durations[-n_songs:]

    logger.info(f"Found {len(song_durations)} valid songs")
    logger.info(f"Low-duration:  {[f'{d:.1f}' for _, d in low]}")
    logger.info(f"High-duration: {[f'{d:.1f}' for _, d in high]}")
    return low, high


def extract_conditioning_prefix(tokens: np.ndarray, n_beats: int = 16) -> torch.Tensor:
    """Extract the first *n_beats* from a token array.

    Returns:
        Conditioning tensor (1, cond_len, 6)
    """
    cond_len = 0
    for i, token in enumerate(tokens):
        if token[1] >= n_beats:
            cond_len = i
            break
    if cond_len == 0:
        cond_len = len(tokens)

    cond_tokens = tokens[:cond_len]
    return torch.from_numpy(cond_tokens).long().unsqueeze(0)


def measure_pitch_from_tokens(tokens: np.ndarray, encoding: dict) -> dict:
    """Measure pitch and duration statistics via muspy decoding."""
    try:
        music = representation.decode(tokens, encoding)
        pitches, durations = [], []
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
        logger.error(f"Error decoding: {e}")
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


# ── Core generation loop ────────────────────────────────────────────────────


def conditioned_generate_and_evaluate(
    model,
    encoding: dict,
    sae_models: dict,
    sas_vectors: dict,
    concept: str,
    device: torch.device,
    song_list: List[Tuple[pathlib.Path, float]],
    category: str,
    lam: float,
    layers_to_steer: List[int],
    conditioning_beats: int,
    continuation_len: int,
    output_dir: Optional[pathlib.Path],
    register_steering_hooks_fn,
    remove_hooks_fn,
    smooth: bool = False,
    schedule: str = "cosine",
    n_ramp: int = 64,
    n_decay: int = 0,
    lambda_maintain: float = 1.0,
    register_smooth_hooks_fn=None,
) -> List[dict]:
    """Generate conditioned continuations with SAS steering and evaluate.

    Args:
        model: MMT model (eval mode, on device)
        encoding: Encoding dictionary
        sae_models: Dictionary of SAE models keyed by layer index
        sas_vectors: Dictionary of SAS vectors keyed by layer index
        concept: "average_pitch" or "average_duration"
        device: torch device
        song_list: List of (filepath, initial_value) tuples
        category: e.g. "low_pitch", "high_pitch", "low_duration", "high_duration"
        lam: Steering strength λ (0 = baseline)
        layers_to_steer: Which layers to steer (e.g. [10])
        conditioning_beats: Beats used as conditioning
        continuation_len: Tokens to generate
        output_dir: Where to save results (None = don't save)
        register_steering_hooks_fn: Hook registration function
        remove_hooks_fn: Hook removal function

    Returns:
        List of result dictionaries
    """
    eos = encoding["type_code_map"]["end-of-song"]
    results = []
    sample_metrics = []

    for i, (filepath, initial_value) in enumerate(song_list):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Song {i + 1}/{len(song_list)}: {filepath.name}")
        logger.info(f"Initial value: {initial_value:.1f}, λ: {lam}")
        logger.info(f"{'=' * 60}")

        # Load tokens and extract conditioning prefix
        tokens = load_song_tokens(filepath, encoding)
        conditioning = extract_conditioning_prefix(tokens, conditioning_beats)
        conditioning = conditioning.to(device)
        logger.info(
            f"Conditioning: {conditioning.shape[1]} tokens ({conditioning_beats} beats)"
        )

        # Also compute conditioning duration for reference
        conditioning_pitch = (
            calculate_initial_pitch(tokens, encoding, conditioning_beats) or 0.0
        )
        conditioning_duration = (
            calculate_initial_duration(tokens, encoding, conditioning_beats) or 0.0
        )

        # ── Register SAS hooks ───────────────────────────────────────
        if smooth and lam != 0.0 and register_smooth_hooks_fn is not None:
            handles = register_smooth_hooks_fn(
                model,
                sae_models,
                sas_vectors,
                concept,
                steering_strength=lam,
                layers_to_steer=layers_to_steer,
                schedule=schedule,
                n_ramp=n_ramp,
                n_decay=n_decay,
                lambda_maintain=lambda_maintain,
            )
        else:
            handles = register_steering_hooks_fn(
                model,
                sae_models,
                sas_vectors,
                concept,
                lam,
                layers_to_steer,
            )

        try:
            with torch.no_grad():
                generated = model.generate(
                    conditioning,
                    continuation_len,
                    eos_token=eos,
                    temperature=1.0,
                    filter_logits_fn="top_k",
                    filter_thres=0.9,
                    monotonicity_dim=("type", "beat"),
                )
        finally:
            remove_hooks_fn(handles)

        # ── Combine and measure ──────────────────────────────────────
        full_seq = torch.cat((conditioning, generated), 1).cpu().numpy()[0]
        generated_only = generated.cpu().numpy()[0]

        metrics = measure_pitch_from_tokens(generated_only, encoding)
        full_metrics = measure_pitch_from_tokens(full_seq, encoding)
        quality_metrics = evaluate_quality_metrics(full_seq, encoding)
        degradation = calculate_degradation(quality_metrics, GROUND_TRUTH_METRICS)

        result = {
            "song_name": filepath.stem,
            "category": category,
            "initial_pitch": float(conditioning_pitch),
            "initial_duration": float(conditioning_duration),
            "lambda": float(lam),
            "layers_to_steer": layers_to_steer,
            "conditioning_beats": conditioning_beats,
            "conditioning_tokens": conditioning.shape[1],
            "generated_mean_pitch": metrics["mean"],
            "generated_std_pitch": metrics["std"],
            "generated_mean_duration": metrics["duration_mean"],
            "generated_std_duration": metrics["duration_std"],
            "generated_n_notes": metrics["n_notes"],
            "full_mean_pitch": full_metrics["mean"],
            "full_mean_duration": full_metrics["duration_mean"],
            "full_n_notes": full_metrics["n_notes"],
            "pitch_change": float(metrics["mean"] - conditioning_pitch),
            "duration_change": float(metrics["duration_mean"] - conditioning_duration),
            "quality_metrics": quality_metrics,
            "degradation": degradation,
        }
        results.append(result)

        logger.info(
            f"Conditioning: pitch={conditioning_pitch:.1f}, duration={conditioning_duration:.2f} ticks"
        )
        logger.info(
            f"Generated: {metrics['n_notes']} notes, "
            f"mean pitch={metrics['mean']:.1f}, "
            f"mean duration={metrics['duration_mean']:.2f} ticks"
        )
        logger.info(
            f"Change: pitch={metrics['mean'] - conditioning_pitch:+.1f}, "
            f"duration={metrics['duration_mean'] - conditioning_duration:+.2f} ticks"
        )
        logger.info(
            f"Quality: entropy={quality_metrics.get('pitch_class_entropy', 0):.3f}, "
            f"scale={quality_metrics.get('scale_consistency', 0):.1f}%, "
            f"groove={quality_metrics.get('groove_consistency', 0):.1f}%, "
            f"degradation={degradation.get('total_degradation', 0):.2f}"
        )

        # ── Save artefacts ───────────────────────────────────────────
        if output_dir is not None:
            lam_str = f"lambda_{'pos' if lam >= 0 else 'neg'}{abs(lam):.2f}"
            save_dir = output_dir / category / lam_str
            save_dir.mkdir(parents=True, exist_ok=True)

            np.save(save_dir / f"{filepath.stem}.npy", full_seq)

            sample_metrics.append(
                {
                    "filepath": str(save_dir / f"{filepath.stem}.npy"),
                    "song_name": filepath.stem,
                    "category": category,
                    "lambda": float(lam),
                    "sample_num": i,
                    "initial_pitch": float(conditioning_pitch),
                    "initial_duration": float(conditioning_duration),
                    "generated_mean_pitch": metrics["mean"],
                    "generated_std_pitch": metrics["std"],
                    "generated_mean_duration": metrics["duration_mean"],
                    "generated_std_duration": metrics["duration_std"],
                    "generated_n_notes": metrics["n_notes"],
                    "pitch_change": float(metrics["mean"] - conditioning_pitch),
                    "duration_change": float(
                        metrics["duration_mean"] - conditioning_duration
                    ),
                    "pitch_class_entropy": quality_metrics.get(
                        "pitch_class_entropy", np.nan
                    ),
                    "scale_consistency": quality_metrics.get(
                        "scale_consistency", np.nan
                    ),
                    "groove_consistency": quality_metrics.get(
                        "groove_consistency", np.nan
                    ),
                    "total_degradation": degradation.get("total_degradation", np.nan),
                }
            )

            # Save audio
            try:
                music = representation.decode(full_seq, encoding)
                music.write(str(save_dir / f"{filepath.stem}.mid"))
                music.write_audio(str(save_dir / f"{filepath.stem}.wav"))
            except Exception as e:
                logger.error(f"Error saving audio: {e}")

    # Persist per-sample metrics
    if output_dir is not None and sample_metrics:
        lam_str = f"lambda_{'pos' if lam >= 0 else 'neg'}{abs(lam):.2f}"
        metrics_file = output_dir / category / lam_str / "sample_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(sample_metrics, f, indent=2)
        logger.info(f"Saved sample metrics to {metrics_file}")

    return results


# ── Analysis functions ───────────────────────────────────────────────────────


def analyze_conditioned_results(results: List[dict]) -> dict:
    """Group results by (category, λ) and compare steering vs baseline."""
    grouped: Dict[tuple, list] = {}
    for r in results:
        key = (r["category"], r["lambda"])
        grouped.setdefault(key, []).append(r)

    analysis = {"groups": {}, "comparisons": {}}

    for (category, lam), group_results in grouped.items():
        valid = [r for r in group_results if r["generated_n_notes"] > 0]
        if not valid:
            continue

        analysis["groups"][f"{category}_lambda_{lam}"] = {
            "n_samples": len(valid),
            "mean_initial_pitch": float(np.mean([r["initial_pitch"] for r in valid])),
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
                np.mean([r["generated_mean_pitch"] - r["initial_pitch"] for r in valid])
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

    # Compare each non-zero λ group against the λ=0 baseline
    for category in set(r["category"] for r in results):
        baseline_key = f"{category}_lambda_0.0"
        if baseline_key not in analysis["groups"]:
            continue
        baseline = analysis["groups"][baseline_key]

        is_pitch_cat = "pitch" in category
        is_dur_cat = "duration" in category

        for key, grp in analysis["groups"].items():
            if not key.startswith(category) or key == baseline_key:
                continue

            lam = float(key.split("_")[-1])

            if is_pitch_cat:
                success = (
                    lam > 0
                    and grp["mean_generated_pitch"] > baseline["mean_generated_pitch"]
                ) or (
                    lam < 0
                    and grp["mean_generated_pitch"] < baseline["mean_generated_pitch"]
                )
            elif is_dur_cat:
                success = (
                    lam > 0
                    and grp["mean_generated_duration"]
                    > baseline["mean_generated_duration"]
                ) or (
                    lam < 0
                    and grp["mean_generated_duration"]
                    < baseline["mean_generated_duration"]
                )
            else:
                success = False

            analysis["comparisons"][f"{category}_lambda_{lam}_vs_baseline"] = {
                "category": category,
                "lambda": lam,
                "baseline_mean_pitch": baseline["mean_generated_pitch"],
                "baseline_mean_duration": baseline["mean_generated_duration"],
                "steered_mean_pitch": grp["mean_generated_pitch"],
                "steered_mean_duration": grp["mean_generated_duration"],
                "pitch_difference": grp["mean_generated_pitch"]
                - baseline["mean_generated_pitch"],
                "duration_difference": grp["mean_generated_duration"]
                - baseline["mean_generated_duration"],
                "success": success,
            }

    return analysis


def calculate_statistical_analysis(results: List[dict], concept: str) -> dict:
    """Per-song and aggregate statistics: correlations, effect sizes, monotonicity."""
    is_duration = "duration" in concept

    per_song_stats: Dict[str, dict] = {}
    aggregate_stats: Dict[str, dict] = {}

    for r in results:
        if r.get("generated_n_notes", 0) == 0:
            continue
        song = r["song_name"]
        cat = "low" if "low" in r["category"] else "high"
        if song not in per_song_stats:
            per_song_stats[song] = {
                "category": cat,
                "lambdas": [],
                "values": [],
                "initial": r.get(
                    "initial_duration" if is_duration else "initial_pitch", 0
                ),
            }
        per_song_stats[song]["lambdas"].append(r["lambda"])
        per_song_stats[song]["values"].append(
            r["generated_mean_duration"] if is_duration else r["generated_mean_pitch"]
        )

    # ── Per-song analysis ────────────────────────────────────────────────
    song_analyses = {}
    for song, data in per_song_stats.items():
        if len(data["lambdas"]) < 3:
            continue
        lams = np.array(data["lambdas"])
        vals = np.array(data["values"])
        idx = np.argsort(lams)
        lams, vals = lams[idx], vals[idx]

        pearson_r, pearson_p = scipy_stats.pearsonr(lams, vals)
        spearman_r, spearman_p = scipy_stats.spearmanr(lams, vals)
        slope, intercept, r_val, p_val, std_err = scipy_stats.linregress(lams, vals)
        monotonic = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

        bl_idx = np.where(lams == 0.0)[0]
        bl_val = vals[bl_idx[0]] if len(bl_idx) > 0 else data["initial"]

        song_analyses[song] = {
            "category": data["category"],
            "initial_value": data["initial"],
            "n_lambdas": len(lams),
            "lambda_range": [float(lams[0]), float(lams[-1])],
            "value_range": [float(vals[0]), float(vals[-1])],
            "baseline_value": float(bl_val),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "linear_slope": float(slope),
            "linear_r2": float(r_val**2),
            "monotonic": bool(monotonic),
            "min_change": float(vals[0] - bl_val),
            "max_change": float(vals[-1] - bl_val),
        }

    # ── Aggregate per category ───────────────────────────────────────────
    for category in ["low", "high"]:
        cat_results = [
            r
            for r in results
            if (
                "low" in r["category"] if category == "low" else "high" in r["category"]
            )
            and r.get("generated_n_notes", 0) > 0
        ]
        if not cat_results:
            continue

        lams_all = np.array([r["lambda"] for r in cat_results])
        vals_all = np.array(
            [
                (
                    r["generated_mean_duration"]
                    if is_duration
                    else r["generated_mean_pitch"]
                )
                for r in cat_results
            ]
        )

        pearson_r, pearson_p = scipy_stats.pearsonr(lams_all, vals_all)
        spearman_r, spearman_p = scipy_stats.spearmanr(lams_all, vals_all)
        slope, intercept, r_val, p_val, std_err = scipy_stats.linregress(
            lams_all, vals_all
        )

        # Effect sizes (Cohen's d vs baseline)
        lam_groups: Dict[float, list] = {}
        for r in cat_results:
            lam_groups.setdefault(r["lambda"], []).append(
                r["generated_mean_duration"]
                if is_duration
                else r["generated_mean_pitch"]
            )

        baseline_vals = lam_groups.get(0.0, [])
        effect_sizes = {}
        if baseline_vals:
            bl_mean = np.mean(baseline_vals)
            bl_std = np.std(baseline_vals, ddof=1) if len(baseline_vals) > 1 else 0
            bl_n = len(baseline_vals)

            for lam_val in sorted(lam_groups.keys()):
                if lam_val == 0.0:
                    continue
                gvals = lam_groups[lam_val]
                g_mean = np.mean(gvals)
                g_std = np.std(gvals, ddof=1) if len(gvals) > 1 else 0
                g_n = len(gvals)

                pooled = np.sqrt(
                    ((bl_n - 1) * bl_std**2 + (g_n - 1) * g_std**2)
                    / max(bl_n + g_n - 2, 1)
                )
                d = (g_mean - bl_mean) / pooled if pooled > 0 else 0

                effect_sizes[lam_val] = {
                    "cohens_d": float(d),
                    "mean": float(g_mean),
                    "std": float(g_std),
                    "n": int(g_n),
                    "change_from_baseline": float(g_mean - bl_mean),
                }

        aggregate_stats[category] = {
            "n_samples": len(cat_results),
            "n_songs": len(
                [s for s, d in per_song_stats.items() if d["category"] == category]
            ),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "linear_slope": float(slope),
            "linear_r2": float(r_val**2),
            "effect_sizes": effect_sizes,
        }

    return {"per_song": song_analyses, "aggregate": aggregate_stats}


# ── Pretty-print helpers ─────────────────────────────────────────────────────


def _interpret_correlation(r: float, p: float) -> str:
    if p > 0.05:
        return "NOT SIGNIFICANT"
    a = abs(r)
    if a >= 0.9:
        return "VERY STRONG"
    if a >= 0.7:
        return "STRONG"
    if a >= 0.5:
        return "MODERATE"
    if a >= 0.3:
        return "WEAK"
    return "VERY WEAK"


def _interpret_effect_size(d: float) -> str:
    a = abs(d)
    if a >= 0.8:
        return "LARGE"
    if a >= 0.5:
        return "MEDIUM"
    if a >= 0.2:
        return "SMALL"
    return "NEGLIGIBLE"


def print_statistical_summary(stats: dict, concept: str) -> None:
    """Print a formatted summary of statistical analysis."""
    is_duration = "duration" in concept
    metric_name = "Duration (ticks)" if is_duration else "Pitch"

    print("\n" + "=" * 100)
    print("STATISTICAL ANALYSIS SUMMARY  (SAS Conditioned Evaluation)")
    print("=" * 100)

    # ── Aggregate ────────────────────────────────────────────────────────
    print("\nAGGREGATE STATISTICS (across all songs):")
    print("-" * 100)

    for category in ["low", "high"]:
        label = f"LOW {metric_name}" if category == "low" else f"HIGH {metric_name}"
        cs = stats["aggregate"].get(category, {})
        if not cs:
            continue

        print(f"\n{label} Songs:")
        print(f"  Samples: {cs['n_samples']} (from {cs['n_songs']} songs)")
        print(
            f"  Pearson:  r = {cs['pearson_r']:+.4f}, p = {cs['pearson_p']:.4f} "
            f"[{_interpret_correlation(cs['pearson_r'], cs['pearson_p'])}]"
        )
        print(
            f"  Spearman: ρ = {cs['spearman_r']:+.4f}, p = {cs['spearman_p']:.4f} "
            f"[{_interpret_correlation(cs['spearman_r'], cs['spearman_p'])}]"
        )
        print(
            f"  Linear:   R² = {cs['linear_r2']:.4f}, slope = {cs['linear_slope']:+.4f}"
        )

        if cs["effect_sizes"]:
            print(f"\n  Effect Sizes (Cohen's d vs baseline):")
            for lam_val in sorted(cs["effect_sizes"].keys()):
                es = cs["effect_sizes"][lam_val]
                print(
                    f"    λ = {lam_val:+5.2f}: d = {es['cohens_d']:+.3f} "
                    f"[{_interpret_effect_size(es['cohens_d'])}], "
                    f"change = {es['change_from_baseline']:+.2f}"
                )

    # ── Per-song ─────────────────────────────────────────────────────────
    print("\n" + "-" * 100)
    print("PER-SONG ANALYSIS:")
    print("-" * 100)

    sorted_songs = sorted(
        stats["per_song"].items(), key=lambda x: abs(x[1]["pearson_r"]), reverse=True
    )
    for song_name, ss in sorted_songs:
        cat_label = "LOW" if ss["category"] == "low" else "HIGH"
        print(f"\n{song_name} [{cat_label}]:")
        print(f"  Initial value: {ss['initial_value']:.2f}")
        print(
            f"  λ range: {ss['lambda_range']} → Value range: "
            f"[{ss['value_range'][0]:.2f}, {ss['value_range'][1]:.2f}]"
        )
        print(
            f"  Pearson:  r = {ss['pearson_r']:+.4f}, p = {ss['pearson_p']:.4f} "
            f"[{_interpret_correlation(ss['pearson_r'], ss['pearson_p'])}]"
        )
        print(f"  Spearman: ρ = {ss['spearman_r']:+.4f}, p = {ss['spearman_p']:.4f}")
        print(
            f"  Linear:   R² = {ss['linear_r2']:.4f}, slope = {ss['linear_slope']:+.4f}"
        )
        print(f"  Monotonic: {'YES' if ss['monotonic'] else 'NO'}")
        print(
            f"  Changes: min = {ss['min_change']:+.2f}, max = {ss['max_change']:+.2f}"
        )

    # ── Overall assessment ───────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("STEERING EFFECTIVENESS ASSESSMENT:")
    print("=" * 100)

    for category in ["low", "high"]:
        label = f"LOW {metric_name}" if category == "low" else f"HIGH {metric_name}"
        cs = stats["aggregate"].get(category, {})
        if not cs:
            continue

        pr = cs["pearson_r"]
        pp = cs["pearson_p"]
        r2 = cs["linear_r2"]
        # Direction-aware λ design: LOW songs get only positive λ, HIGH songs
        # get only negative λ.  In both cases, as λ moves away from 0 the
        # metric moves in the intended direction, so the correlation between
        # the actual λ values and the metric is POSITIVE for both categories.
        expected_pos = True
        direction_ok = (pr > 0) == expected_pos

        print(f"\n{label} Songs:")
        print(f"  Expected: Positive correlation (direction-aware λ)")
        print(f"  Actual:   {'Positive' if pr > 0 else 'Negative'} (r = {pr:+.3f})")
        print(f"  Direction: {'CORRECT' if direction_ok else 'INCORRECT'}")
        print(f"  Strength: {_interpret_correlation(pr, pp)}")

        if pp <= 0.001:
            print(f"  Significance: HIGHLY SIGNIFICANT (p < 0.001)")
        elif pp <= 0.01:
            print(f"  Significance: VERY SIGNIFICANT (p < 0.01)")
        elif pp <= 0.05:
            print(f"  Significance: SIGNIFICANT (p < 0.05)")
        else:
            print(f"  Significance: NOT SIGNIFICANT (p = {pp:.3f})")

        if r2 > 0.8:
            print(f"  Linearity: EXCELLENT (R² = {r2:.3f})")
        elif r2 > 0.6:
            print(f"  Linearity: GOOD (R² = {r2:.3f})")
        elif r2 > 0.4:
            print(f"  Linearity: MODERATE (R² = {r2:.3f})")
        else:
            print(f"  Linearity: POOR (R² = {r2:.3f})")

        cat_songs = [
            s for s, d in stats["per_song"].items() if d["category"] == category
        ]
        n_mono = sum(1 for s in cat_songs if stats["per_song"][s]["monotonic"])
        print(f"  Monotonic songs: {n_mono}/{len(cat_songs)}")

        if direction_ok and abs(pr) > 0.7 and pp < 0.01:
            print(f"  VERDICT: EXCELLENT steering effectiveness")
        elif direction_ok and abs(pr) > 0.5 and pp < 0.05:
            print(f"  VERDICT: GOOD steering effectiveness")
        elif direction_ok and pp < 0.05:
            print(f"  VERDICT: MODERATE steering effectiveness")
        else:
            print(f"  VERDICT: WEAK or ineffective steering")

    print("\n" + "=" * 100)


def generate_listening_priority_list(
    results: List[dict], output_dir: pathlib.Path, concept: str = ""
) -> None:
    """Rank samples by steering effectiveness + low degradation and save."""
    valid = [
        r
        for r in results
        if r.get("generated_n_notes", 0) > 0
        and "degradation" in r
        and not np.isnan(r["degradation"].get("total_degradation", np.nan))
    ]

    for r in valid:
        is_dur = "duration" in r["category"]
        if is_dur:
            if "low" in r["category"] and r["lambda"] > 0:
                score = r["duration_change"]
            elif "high" in r["category"] and r["lambda"] < 0:
                score = -r["duration_change"]
            else:
                score = 0
        else:
            if "low" in r["category"] and r["lambda"] > 0:
                score = r["pitch_change"]
            elif "high" in r["category"] and r["lambda"] < 0:
                score = -r["pitch_change"]
            else:
                score = 0

        r["steering_score"] = abs(score)
        norm_steer = min(r["steering_score"] / 50.0, 1.0)
        norm_qual = 1.0 - min(r["degradation"]["total_degradation"] / 100.0, 1.0)
        r["priority_score"] = (norm_steer * 0.6 + norm_qual * 0.4) * 100

    valid.sort(key=lambda x: x["priority_score"], reverse=True)

    lines = [
        "=" * 100,
        "LISTENING PRIORITY LIST  (SAS Conditioned Evaluation)",
        "Ranked by: Steering Effectiveness (60%) + Low Degradation (40%)",
        "=" * 100,
        "",
    ]

    for i, r in enumerate(valid[:20], 1):
        lam_str = f"lambda_{'pos' if r['lambda'] >= 0 else 'neg'}{abs(r['lambda']):.2f}"
        lines.append(f"{i}. [{r['priority_score']:.1f}] {r['song_name']}")
        lines.append(f"   Category: {r['category']}, λ: {r['lambda']:+.2f}")
        lines.append(
            f"   Steering: {r['steering_score']:.2f}, "
            f"Degradation: {r['degradation']['total_degradation']:.2f}"
        )
        if "duration" in r["category"]:
            lines.append(
                f"   Duration: {r['initial_duration']:.1f} → "
                f"{r['generated_mean_duration']:.1f} ticks "
                f"(change: {r['duration_change']:+.1f})"
            )
        else:
            lines.append(
                f"   Pitch: {r['initial_pitch']:.1f} → "
                f"{r['generated_mean_pitch']:.1f} "
                f"(change: {r['pitch_change']:+.1f})"
            )
        lines.append(
            f"   Quality: entropy={r['quality_metrics']['pitch_class_entropy']:.2f}, "
            f"scale={r['quality_metrics']['scale_consistency']:.1f}%, "
            f"groove={r['quality_metrics']['groove_consistency']:.1f}%"
        )
        lines.append(f"   File: {r['category']}/{lam_str}/{r['song_name']}.wav")
        lines.append("")

    suffix = f"_{concept}" if concept else ""
    list_file = output_dir / f"listening_priority_list{suffix}.txt"
    with open(list_file, "w") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(lines[:200]))
    print(f"\nFull list saved to: {list_file}")


# ── Main entry point ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Conditioned evaluation of SAS steering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--concept",
        type=str,
        default="average_pitch",
        help="Concept to evaluate (average_pitch | average_duration)",
    )
    parser.add_argument(
        "--n_songs", type=int, default=5, help="Songs per category (low / high)"
    )
    parser.add_argument(
        "--conditioning_beats",
        type=int,
        default=16,
        help="Number of beats to use as conditioning prefix",
    )
    parser.add_argument(
        "--continuation_len",
        type=int,
        default=256,
        help="Number of tokens to generate after the conditioning",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0.0,0.25,0.5,0.75,-0.25,-0.5,-0.75,-1.0,-1.5",
        help="Comma-separated λ values to test",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="10",
        help="Comma-separated layer indices to steer (default: 10)",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU number")

    # Smooth steering options
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Enable smooth steering with gradual lambda ramp-up",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "sigmoid"],
        help="Ramp-up schedule function (default: cosine)",
    )
    parser.add_argument(
        "--n_ramp",
        type=int,
        default=64,
        help="Number of generation steps for ramp-up (default: 64)",
    )
    parser.add_argument(
        "--n_decay",
        type=int,
        default=0,
        help="Number of steps for decay phase (0 = no decay)",
    )
    parser.add_argument(
        "--lambda_maintain",
        type=float,
        default=1.0,
        help="Fraction of lambda to maintain after decay (0-1, default: 1.0)",
    )

    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=REPO_ROOT
        / "exp"
        / "sod"
        / "sparse_steering"
        / "conditioned_evaluation",
        help="Output directory",
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Sub-folder name"
    )

    # Path overrides
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train_args", type=pathlib.Path, default=DEFAULT_TRAIN_ARGS)
    parser.add_argument("--encoding_path", type=pathlib.Path, default=DEFAULT_ENCODING)
    parser.add_argument("--sae_dir", type=pathlib.Path, default=DEFAULT_SAE_DIR)
    parser.add_argument("--sas_dir", type=pathlib.Path, default=DEFAULT_SAS_DIR)
    parser.add_argument("--notes_dir", type=pathlib.Path, default=DEFAULT_NOTES_DIR)

    args = parser.parse_args()

    if args.experiment_name:
        args.output_dir = args.output_dir / args.experiment_name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # ── Parse CLI lists ──────────────────────────────────────────────────
    lambdas = sorted([float(x.strip()) for x in args.lambdas.split(",")])
    layers_to_steer = [int(x.strip()) for x in args.layers.split(",")]
    logger.info(f"λ values: {lambdas}")
    logger.info(f"Layers to steer: {layers_to_steer}")

    # ── Load SAS infrastructure ──────────────────────────────────────────
    (
        load_model,
        load_sae_models,
        load_sas_vectors,
        register_steering_hooks,
        remove_hooks,
        register_smooth_steering_hooks_fn,
    ) = _import_sas_modules()

    model, encoding, train_args = load_model(
        args.checkpoint,
        args.train_args,
        args.encoding_path,
        device,
    )
    sae_models = load_sae_models(args.sae_dir, device)

    sas_vector_path = args.sas_dir / f"{args.concept}_sas_vectors.pt"
    if not sas_vector_path.exists():
        logger.error(f"SAS vectors not found: {sas_vector_path}")
        sys.exit(1)
    sas_vectors = load_sas_vectors(sas_vector_path)

    # ── Find extreme songs ───────────────────────────────────────────────
    if "duration" in args.concept:
        low_songs, high_songs = find_extreme_duration_songs(
            args.notes_dir,
            encoding,
            args.n_songs,
            args.conditioning_beats,
        )
        low_cat, high_cat = "low_duration", "high_duration"
    else:
        low_songs, high_songs = find_extreme_pitch_songs(
            args.notes_dir,
            encoding,
            args.n_songs,
            args.conditioning_beats,
        )
        low_cat, high_cat = "low_pitch", "high_pitch"

    # ── Generate and evaluate ────────────────────────────────────────────
    all_results: List[dict] = []

    # Smooth steering kwargs
    smooth_kwargs = dict(
        smooth=args.smooth,
        schedule=args.schedule,
        n_ramp=args.n_ramp,
        n_decay=args.n_decay,
        lambda_maintain=args.lambda_maintain,
        register_smooth_hooks_fn=register_smooth_steering_hooks_fn,
    )

    for lam in lambdas:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"λ = {lam}")
        logger.info(f"{'=' * 80}")

        # Direction-aware: positive λ for low songs, negative λ for high songs
        if lam >= 0:
            logger.info(
                f"Processing {low_cat.upper().replace('_', ' ')} songs (positive λ → increase)"
            )
            low_results = conditioned_generate_and_evaluate(
                model,
                encoding,
                sae_models,
                sas_vectors,
                args.concept,
                device,
                low_songs,
                low_cat,
                lam,
                layers_to_steer,
                args.conditioning_beats,
                args.continuation_len,
                args.output_dir,
                register_steering_hooks,
                remove_hooks,
                **smooth_kwargs,
            )
            all_results.extend(low_results)

        if lam <= 0:
            logger.info(
                f"Processing {high_cat.upper().replace('_', ' ')} songs (negative λ → decrease)"
            )
            high_results = conditioned_generate_and_evaluate(
                model,
                encoding,
                sae_models,
                sas_vectors,
                args.concept,
                device,
                high_songs,
                high_cat,
                lam,
                layers_to_steer,
                args.conditioning_beats,
                args.continuation_len,
                args.output_dir,
                register_steering_hooks,
                remove_hooks,
                **smooth_kwargs,
            )
            all_results.extend(high_results)

    # ── Analysis ─────────────────────────────────────────────────────────
    logger.info("\nAnalyzing results...")
    analysis = analyze_conditioned_results(all_results)

    logger.info("\nPerforming statistical analysis...")
    stat_analysis = calculate_statistical_analysis(all_results, args.concept)

    # ── Persist ──────────────────────────────────────────────────────────
    results_file = args.output_dir / f"conditioned_results_{args.concept}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "results": all_results,
                "analysis": analysis,
                "statistical_analysis": stat_analysis,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved results to: {results_file}")

    # ── Print reports ────────────────────────────────────────────────────
    print_statistical_summary(stat_analysis, args.concept)

    logger.info("\nGenerating listening priority list...")
    generate_listening_priority_list(all_results, args.output_dir, args.concept)

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CONDITIONED SAS STEERING EVALUATION SUMMARY")
    print(
        f"Concept: {args.concept}  |  Layers: {layers_to_steer}  |  N songs: {args.n_songs}"
    )
    print("=" * 80)

    for comp_key, comp in analysis["comparisons"].items():
        print(
            f"\n{comp['category'].upper().replace('_', ' ')} songs, λ={comp['lambda']:+.2f}"
        )
        print(
            f"  Baseline:  pitch={comp['baseline_mean_pitch']:.1f}, duration={comp['baseline_mean_duration']:.2f}"
        )
        print(
            f"  Steered:   pitch={comp['steered_mean_pitch']:.1f}, duration={comp['steered_mean_duration']:.2f}"
        )
        print(
            f"  Δ pitch={comp['pitch_difference']:+.1f}, Δ duration={comp['duration_difference']:+.2f}"
        )
        print(f"  Success: {'YES' if comp['success'] else 'NO'}")

    print("=" * 80)
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
