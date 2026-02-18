"""
SAS Layer and Lambda Optimization Experiment.

Grid search over layer groups and λ values to find the optimal configuration
for Sparse Activation Steering (SAS) that:
1. Achieves target feature manipulation (positive λ → higher, negative λ → lower)
2. Maintains model quality (pitch_class_entropy, scale_consistency, groove_consistency)

This is the SAS equivalent of steering_interventions/experiments/layer_alpha_optimization.py,
using SAE-based steering hooks instead of DiffMean steering vectors.

Prerequisites:
    - Trained MMT model at: exp/sod/ape/checkpoints/best_model.pt
    - Encoding at: data/sod/processed/notes/encoding.json
    - Trained SAE models at: exp/sod/sparse_steering/sae_checkpoints/
    - SAS vectors at: exp/sod/sparse_steering/sas_vectors/
    (Run compute_sas_vectors.py --tau 0.08 first)

Usage:
    # Full grid search for pitch
    python sparse_steering/layer_lambda_optimization.py \
        --concept average_pitch \
        --n_samples 5 \
        --gpu 0

    # Full grid search for duration
    python sparse_steering/layer_lambda_optimization.py \
        --concept average_duration \
        --n_samples 5 \
        --gpu 0

    # Quick test with fewer configs
    python sparse_steering/layer_lambda_optimization.py \
        --concept average_pitch \
        --n_samples 3 \
        --lambda_values -1.0 0.0 1.0 \
        --gpu 0
"""

import argparse
import json
import logging
import pathlib
import sys
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import muspy
import numpy as np
import torch
from scipy import stats as scipy_stats

# Add paths
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import representation

# Import SAS components from steered_generator_sas
from steered_generator_sas import (
    load_model,
    load_sae_models,
    load_sas_vectors,
    register_steering_hooks,
    remove_hooks,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ground truth quality metrics from the paper
GROUND_TRUTH_METRICS = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}

# Layer groups to test (same structure as layer_alpha_optimization.py)
LAYER_GROUPS = {
    "single_0": [0],
    "single_1": [1],
    "single_2": [2],
    "single_3": [3],
    "single_4": [4],
    "single_5": [5],
    "single_6": [6],
    "single_7": [7],
    "single_8": [8],
    "single_9": [9],
    "single_10": [10],
    "single_11": [11],
    "early": [0, 1, 2, 3],
    "middle": [4, 5, 6, 7],
    "late": [8, 9, 10, 11],
    "all": list(range(12)),
}

# Default lambda values to test
DEFAULT_LAMBDA_VALUES = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]


# ============================================================================
# Metric Extraction
# ============================================================================


def extract_pitches_from_tokens(tokens: np.ndarray, encoding: dict) -> list:
    """Extract pitch values from generated tokens."""
    try:
        music = representation.decode(tokens, encoding)
        pitches = []
        for track in music.tracks:
            for note in track.notes:
                pitches.append(note.pitch)
        return pitches
    except Exception as e:
        logger.warning(f"Error extracting pitches: {e}")
        return []


def extract_durations_from_tokens(tokens: np.ndarray, encoding: dict) -> list:
    """Extract duration values from generated tokens (in ticks)."""
    try:
        music = representation.decode(tokens, encoding)
        durations = []
        for track in music.tracks:
            for note in track.notes:
                durations.append(note.duration)
        return durations
    except Exception as e:
        logger.warning(f"Error extracting durations: {e}")
        return []


def evaluate_quality_metrics(tokens: np.ndarray, encoding: dict) -> dict:
    """Evaluate objective quality metrics using muspy."""
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
        logger.warning(f"Error evaluating quality: {e}")
        return {
            "pitch_class_entropy": np.nan,
            "scale_consistency": np.nan,
            "groove_consistency": np.nan,
        }


def calculate_degradation(metrics: dict, baseline: dict) -> dict:
    """Calculate quality degradation from baseline."""
    entropy_diff = abs(
        metrics.get("pitch_class_entropy", np.nan)
        - baseline.get("pitch_class_entropy", np.nan)
    )
    scale_diff = max(
        0,
        baseline.get("scale_consistency", 0) - metrics.get("scale_consistency", 0),
    )
    groove_diff = max(
        0,
        baseline.get("groove_consistency", 0) - metrics.get("groove_consistency", 0),
    )

    total_degradation = entropy_diff + scale_diff + groove_diff

    return {
        "entropy_diff": float(entropy_diff) if not np.isnan(entropy_diff) else 0.0,
        "scale_diff": float(scale_diff),
        "groove_diff": float(groove_diff),
        "total_degradation": (
            float(total_degradation) if not np.isnan(total_degradation) else 0.0
        ),
    }


# ============================================================================
# Generation
# ============================================================================


def generate_samples_sas(
    model,
    encoding: dict,
    sae_models: dict,
    sas_vectors: dict,
    concept: str,
    steering_strength: float,
    layers_to_steer: Optional[List[int]],
    n_samples: int,
    max_seq_len: int,
    device: torch.device,
) -> List[np.ndarray]:
    """Generate samples with SAS steering at specific layers.

    Args:
        model: MMT model
        encoding: Encoding dictionary
        sae_models: Dictionary of SAE models per layer
        sas_vectors: Dictionary of SAS vectors per layer
        concept: Concept name
        steering_strength: λ parameter
        layers_to_steer: List of layer indices to steer (None = all)
        n_samples: Number of samples to generate
        max_seq_len: Maximum sequence length
        device: Device

    Returns:
        List of generated token sequences (numpy arrays)
    """
    # Register steering hooks
    handles = register_steering_hooks(
        model, sae_models, sas_vectors, concept, steering_strength, layers_to_steer
    )

    # Get SOS and EOS tokens
    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    # Generation config
    temperature = 1.0
    filter_fn = "top_k"
    filter_thresh = 0.9

    try:
        # Try to get config values if available
        try:
            import config as cfg

            temperature = getattr(cfg, "GENERATION_TEMPERATURE", 1.0)
            filter_fn = getattr(cfg, "GENERATION_FILTER", "top_k")
            filter_thresh = getattr(cfg, "GENERATION_FILTER_THRESHOLD", 0.9)
        except ImportError:
            pass
    except Exception:
        pass

    try:
        samples = []
        for _ in range(n_samples):
            # Create start tokens
            start_tokens = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            start_tokens[:, 0, 0] = sos

            # Generate
            with torch.no_grad():
                generated = model.generate(
                    start_tokens,
                    max_seq_len,
                    eos_token=eos,
                    temperature=temperature,
                    filter_logits_fn=filter_fn,
                    filter_thres=filter_thresh,
                    monotonicity_dim=("type", "beat"),
                )

            # Combine start + generated
            full_seq = torch.cat((start_tokens, generated), 1).cpu().numpy()[0]
            samples.append(full_seq)

        return samples

    finally:
        # Always remove hooks
        remove_hooks(handles)


# ============================================================================
# Evaluation
# ============================================================================


def evaluate_configuration(
    model,
    encoding: dict,
    sae_models: dict,
    sas_vectors: dict,
    concept: str,
    layer_group: List[int],
    steering_strength: float,
    n_samples: int,
    max_seq_len: int,
    device: torch.device,
    save_dir: Optional[pathlib.Path] = None,
) -> dict:
    """Evaluate a single (layer_group, λ) configuration.

    Args:
        model: MMT model
        encoding: Encoding dictionary
        sae_models: SAE models per layer
        sas_vectors: SAS vectors per layer
        concept: Concept being steered
        layer_group: List of layer indices to steer
        steering_strength: λ value
        n_samples: Number of samples
        max_seq_len: Max sequence length
        device: Device
        save_dir: Optional directory to save generated .npy files

    Returns:
        Dictionary with aggregated metrics
    """
    # Generate samples
    samples = generate_samples_sas(
        model=model,
        encoding=encoding,
        sae_models=sae_models,
        sas_vectors=sas_vectors,
        concept=concept,
        steering_strength=steering_strength,
        layers_to_steer=layer_group,
        n_samples=n_samples,
        max_seq_len=max_seq_len,
        device=device,
    )

    # Save samples if requested
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(samples):
            np.save(save_dir / f"sample_{i}.npy", sample)

    # Extract metrics from each sample
    all_pitches = []
    all_durations = []
    per_sample_pitch_means = []
    per_sample_duration_means = []
    quality_metrics_list = []

    for sample in samples:
        pitches = extract_pitches_from_tokens(sample, encoding)
        durations = extract_durations_from_tokens(sample, encoding)
        all_pitches.extend(pitches)
        all_durations.extend(durations)

        # Per-sample means for variance estimation
        if pitches:
            per_sample_pitch_means.append(float(np.mean(pitches)))
        if durations:
            per_sample_duration_means.append(float(np.mean(durations)))

        # Quality metrics per sample
        quality = evaluate_quality_metrics(sample, encoding)
        quality_metrics_list.append(quality)

    # Aggregate quality metrics
    avg_entropy = float(
        np.nanmean([q["pitch_class_entropy"] for q in quality_metrics_list])
    )
    avg_scale = float(
        np.nanmean([q["scale_consistency"] for q in quality_metrics_list])
    )
    avg_groove = float(
        np.nanmean([q["groove_consistency"] for q in quality_metrics_list])
    )

    avg_quality = {
        "pitch_class_entropy": avg_entropy,
        "scale_consistency": avg_scale,
        "groove_consistency": avg_groove,
    }

    # Calculate degradation from paper ground truth
    degradation = calculate_degradation(avg_quality, GROUND_TRUTH_METRICS)

    return {
        "mean_pitch": float(np.mean(all_pitches)) if all_pitches else 0.0,
        "std_pitch": (
            float(np.std(per_sample_pitch_means)) if per_sample_pitch_means else 0.0
        ),
        "mean_duration": float(np.mean(all_durations)) if all_durations else 0.0,
        "std_duration": (
            float(np.std(per_sample_duration_means))
            if per_sample_duration_means
            else 0.0
        ),
        "n_notes": len(all_pitches),
        "n_samples": n_samples,
        "pitch_class_entropy": avg_entropy,
        "scale_consistency": avg_scale,
        "groove_consistency": avg_groove,
        "degradation": degradation,
    }


# ============================================================================
# Analysis
# ============================================================================


def analyze_layer_group(
    results: dict, _layer_name: str, lambda_values: List[float], concept: str
) -> dict:
    """Analyze steering effectiveness for a single layer group across lambdas.

    Returns correlation, slope, monotonicity, and a composite score.
    """
    # Need at least 3 lambda values for meaningful analysis
    valid_lambdas = [lv for lv in sorted(lambda_values) if lv in results]
    if len(valid_lambdas) < 3:
        return {"valid": False, "reason": "insufficient lambda values"}

    lambdas_arr = np.array(valid_lambdas)

    # Determine target metric
    if "pitch" in concept.lower():
        target_values = np.array([results[lv]["mean_pitch"] for lv in valid_lambdas])
        metric_name = "pitch"
    else:
        target_values = np.array([results[lv]["mean_duration"] for lv in valid_lambdas])
        metric_name = "duration"

    # Correlation analysis
    pearson_r, pearson_p = scipy_stats.pearsonr(lambdas_arr, target_values)
    spearman_r, spearman_p = scipy_stats.spearmanr(lambdas_arr, target_values)

    # Linear regression
    slope, intercept, r_value, _, std_err = scipy_stats.linregress(
        lambdas_arr, target_values
    )

    # Monotonicity check
    is_monotonic = all(
        target_values[i] <= target_values[i + 1] for i in range(len(target_values) - 1)
    )

    # Range of effect (max - min target value)
    effect_range = float(target_values.max() - target_values.min())

    # Baseline shift (if λ=0 exists)
    baseline_value = results.get(0.0, {}).get(
        f"mean_{metric_name}", target_values.mean()
    )
    max_positive_shift = float(target_values.max() - baseline_value)
    max_negative_shift = float(target_values.min() - baseline_value)

    # Average quality degradation across all lambda values
    avg_degradation = float(
        np.mean(
            [
                results[lv]["degradation"]["total_degradation"]
                for lv in valid_lambdas
                if "degradation" in results[lv]
            ]
        )
    )

    # Degradation at extremes only (non-zero lambdas)
    extreme_lambdas = [lv for lv in valid_lambdas if abs(lv) > 1e-9]
    avg_extreme_degradation = (
        float(
            np.mean(
                [
                    results[lv]["degradation"]["total_degradation"]
                    for lv in extreme_lambdas
                    if "degradation" in results[lv]
                ]
            )
        )
        if extreme_lambdas
        else 0.0
    )

    # Composite score: reward high correlation & range, penalize degradation
    # Score = |r| * effect_range - degradation_penalty
    correlation_score = abs(pearson_r) * effect_range
    degradation_penalty = avg_extreme_degradation * 0.1  # Scale down
    composite_score = correlation_score - degradation_penalty

    return {
        "valid": True,
        "metric_name": metric_name,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "r_squared": float(r_value**2),
        "slope": float(slope),
        "intercept": float(intercept),
        "std_err": float(std_err),
        "is_monotonic": is_monotonic,
        "effect_range": effect_range,
        "max_positive_shift": max_positive_shift,
        "max_negative_shift": max_negative_shift,
        "baseline_value": float(baseline_value),
        "avg_degradation": avg_degradation,
        "avg_extreme_degradation": avg_extreme_degradation,
        "composite_score": float(composite_score),
        "lambda_to_value": {
            str(lv): float(target_values[i]) for i, lv in enumerate(valid_lambdas)
        },
    }


def rank_configurations(all_analyses: dict) -> List[Tuple[str, dict]]:
    """Rank layer groups by composite score.

    Returns sorted list of (layer_name, analysis) tuples.
    """
    valid = [
        (name, analysis)
        for name, analysis in all_analyses.items()
        if analysis.get("valid", False)
    ]
    return sorted(valid, key=lambda x: x[1]["composite_score"], reverse=True)


def print_results_table(
    results: dict,
    analyses: dict,
    lambda_values: List[float],
    concept: str,
):
    """Print a comprehensive results table."""
    metric = "pitch" if "pitch" in concept.lower() else "duration"
    unit = "semitones" if metric == "pitch" else "ticks"

    print("\n" + "=" * 120)
    print(f"SAS LAYER-LAMBDA OPTIMIZATION RESULTS — {concept.upper()}")
    print("=" * 120)

    # Header
    lambda_str = "".join(f"{'λ=' + str(lv):>10s}" for lv in sorted(lambda_values))
    print(
        f"{'Layer Group':>14s} | {lambda_str} | {'r':>6s} {'p':>8s} "
        f"{'R²':>6s} {'slope':>8s} {'range':>7s} {'mono':>5s} "
        f"{'deg':>6s} {'score':>8s}"
    )
    print("-" * 120)

    # Sort by composite score
    ranked = rank_configurations(analyses)
    ranked_names = [name for name, _ in ranked]

    # Also show unranked ones
    all_names = ranked_names + [n for n in results if n not in ranked_names]

    for layer_name in all_names:
        if layer_name not in results:
            continue

        # Lambda values
        vals = []
        for lv in sorted(lambda_values):
            if lv in results[layer_name]:
                v = results[layer_name][lv].get(f"mean_{metric}", 0.0)
                vals.append(f"{v:10.2f}")
            else:
                vals.append(f"{'---':>10s}")
        vals_str = "".join(vals)

        # Analysis
        a = analyses.get(layer_name, {})
        if a.get("valid", False):
            r_str = f"{a['pearson_r']:+6.3f}"
            p_str = f"{a['pearson_p']:8.4f}"
            r2_str = f"{a['r_squared']:6.3f}"
            slope_str = f"{a['slope']:+8.4f}"
            range_str = f"{a['effect_range']:7.2f}"
            mono_str = "  ✓" if a["is_monotonic"] else "  ✗"
            deg_str = f"{a['avg_extreme_degradation']:6.2f}"
            score_str = f"{a['composite_score']:+8.3f}"
        else:
            r_str = p_str = r2_str = slope_str = range_str = "---"
            mono_str = "---"
            deg_str = score_str = "---"

        # Highlight best
        marker = ""
        if ranked and ranked[0][0] == layer_name:
            marker = " ★"

        print(
            f"{layer_name:>14s} | {vals_str} | {r_str} {p_str} "
            f"{r2_str} {slope_str} {range_str} {mono_str} "
            f"{deg_str} {score_str}{marker}"
        )

    # Summary
    print("\n" + "=" * 120)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 120)

    for i, (name, a) in enumerate(ranked[:5], 1):
        p = a["pearson_p"]
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "ns"
        print(
            f"  {i}. {name:14s}: r={a['pearson_r']:+.3f} ({sig}), "
            f"R²={a['r_squared']:.3f}, slope={a['slope']:+.4f} {unit}/λ, "
            f"range={a['effect_range']:.2f} {unit}, "
            f"mono={'✓' if a['is_monotonic'] else '✗'}, "
            f"deg={a['avg_extreme_degradation']:.2f}, "
            f"score={a['composite_score']:+.3f}"
        )

    # Best single layer
    single_ranked = [(n, a) for n, a in ranked if n.startswith("single_")]
    if single_ranked:
        best_single = single_ranked[0]
        print(
            f"\n  Best single layer: {best_single[0]} (score={best_single[1]['composite_score']:+.3f})"
        )

    # Best group
    group_ranked = [(n, a) for n, a in ranked if not n.startswith("single_")]
    if group_ranked:
        best_group = group_ranked[0]
        print(
            f"  Best layer group:  {best_group[0]} (score={best_group[1]['composite_score']:+.3f})"
        )

    print("=" * 120)

    # Quality degradation table
    print("\n" + "=" * 120)
    print("QUALITY METRICS BY CONFIGURATION")
    print("=" * 120)
    print(
        f"{'Layer Group':>14s} | {'λ':>5s} | {'Entropy':>8s} {'Scale%':>8s} "
        f"{'Groove%':>8s} | {'Total Deg':>10s}"
    )
    print(
        f"{'Ground Truth':>14s} | {'':>5s} | "
        f"{GROUND_TRUTH_METRICS['pitch_class_entropy']:8.3f} "
        f"{GROUND_TRUTH_METRICS['scale_consistency']:8.2f} "
        f"{GROUND_TRUTH_METRICS['groove_consistency']:8.2f} | {'---':>10s}"
    )
    print("-" * 80)

    # Show only top 5 + extremes
    for name, _ in ranked[:5]:
        for lv in sorted(lambda_values):
            if lv in results[name]:
                r = results[name][lv]
                d = r.get("degradation", {})
                print(
                    f"{name:>14s} | {lv:+5.1f} | "
                    f"{r.get('pitch_class_entropy', 0):8.3f} "
                    f"{r.get('scale_consistency', 0):8.2f} "
                    f"{r.get('groove_consistency', 0):8.2f} | "
                    f"{d.get('total_degradation', 0):10.2f}"
                )
        print("-" * 80)

    print("=" * 120)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SAS Layer and Lambda Optimization Experiment"
    )
    parser.add_argument(
        "--concept",
        required=True,
        choices=["average_pitch", "average_duration"],
        help="Concept to steer",
    )
    parser.add_argument(
        "--lambda_values",
        nargs="+",
        type=float,
        default=None,
        help=f"Lambda values to test (default: {DEFAULT_LAMBDA_VALUES})",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples per configuration (default: 5)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--exp_dir",
        type=pathlib.Path,
        default=pathlib.Path("exp/sod"),
        help="Experiment directory",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Output directory (default: exp_dir/sparse_steering/layer_lambda_opt)",
    )
    parser.add_argument(
        "--save_samples",
        action="store_true",
        help="Save generated .npy files for each configuration",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device (default: 0)")
    parser.add_argument(
        "--layers",
        nargs="+",
        type=str,
        default=None,
        help=(
            "Layer groups to test (default: all groups). "
            "E.g.: --layers single_8 single_10 single_11 late all"
        ),
    )

    args = parser.parse_args()

    # Set defaults
    if args.lambda_values is None:
        args.lambda_values = DEFAULT_LAMBDA_VALUES

    # Filter layer groups if specified
    if args.layers is not None:
        invalid = [lg for lg in args.layers if lg not in LAYER_GROUPS]
        if invalid:
            parser.error(
                f"Unknown layer groups: {invalid}. "
                f"Valid options: {list(LAYER_GROUPS.keys())}"
            )
        selected_layer_groups = {
            name: LAYER_GROUPS[name] for name in args.layers
        }
    else:
        selected_layer_groups = LAYER_GROUPS

    if args.output_dir is None:
        args.output_dir = (
            args.exp_dir / "sparse_steering" / "layer_lambda_opt" / args.concept
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging to file + console
    file_handler = logging.FileHandler(
        args.output_dir / "layer_lambda_optimization.log"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Setup device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    total_configs = len(selected_layer_groups) * len(args.lambda_values)

    logger.info("=" * 80)
    logger.info("SAS LAYER-LAMBDA OPTIMIZATION EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Concept:            {args.concept}")
    logger.info(f"Lambda values:      {args.lambda_values}")
    logger.info(f"Layer groups:       {len(selected_layer_groups)} — {list(selected_layer_groups.keys())}")
    logger.info(f"Samples per config: {args.n_samples}")
    logger.info(f"Max seq length:     {args.max_seq_len}")
    logger.info(f"Total configs:      {total_configs}")
    logger.info(f"Total generations:  {total_configs * args.n_samples}")
    logger.info(f"Device:             {device}")
    logger.info(f"Output:             {args.output_dir}")

    # ---- Load model, SAE models, SAS vectors ----
    checkpoint_path = args.exp_dir / "ape" / "checkpoints" / "best_model.pt"
    train_args_path = args.exp_dir / "ape" / "train-args.json"
    encoding_path = pathlib.Path("data/sod/processed/notes") / "encoding.json"
    sae_dir = args.exp_dir / "sparse_steering" / "sae_checkpoints"
    sas_vectors_path = (
        args.exp_dir
        / "sparse_steering"
        / "sas_vectors"
        / f"{args.concept}_sas_vectors.pt"
    )

    # Validate paths
    for path, desc in [
        (checkpoint_path, "Model checkpoint"),
        (train_args_path, "Training args"),
        (encoding_path, "Encoding"),
        (sae_dir, "SAE checkpoints dir"),
        (sas_vectors_path, "SAS vectors"),
    ]:
        if not path.exists():
            logger.error(f"{desc} not found: {path}")
            sys.exit(1)

    logger.info("\nLoading components...")
    model, encoding, _ = load_model(
        checkpoint_path, train_args_path, encoding_path, device
    )
    sae_models = load_sae_models(sae_dir, device)
    sas_vectors = load_sas_vectors(sas_vectors_path)

    # ---- Grid Search ----
    results = defaultdict(dict)
    start_time = time.time()
    config_count = 0

    for layer_name, layer_group in selected_layer_groups.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Layer group: {layer_name} = {layer_group}")
        logger.info(f"{'='*60}")

        for lam in args.lambda_values:
            config_count += 1
            elapsed = time.time() - start_time
            eta = (
                (elapsed / config_count) * (total_configs - config_count)
                if config_count > 0
                else 0
            )

            logger.info(
                f"  [{config_count}/{total_configs}] "
                f"layers={layer_name}, λ={lam:+.1f} "
                f"(elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min)"
            )

            # Optional save directory
            save_dir = None
            if args.save_samples:
                save_dir = (
                    args.output_dir
                    / "samples"
                    / layer_name
                    / f"lambda_{'+' if lam >= 0 else ''}{lam}"
                )

            # Evaluate this configuration
            config_results = evaluate_configuration(
                model=model,
                encoding=encoding,
                sae_models=sae_models,
                sas_vectors=sas_vectors,
                concept=args.concept,
                layer_group=layer_group,
                steering_strength=lam,
                n_samples=args.n_samples,
                max_seq_len=args.max_seq_len,
                device=device,
                save_dir=save_dir,
            )

            results[layer_name][lam] = config_results

            metric = "pitch" if "pitch" in args.concept else "duration"
            logger.info(
                f"    → mean_{metric}={config_results[f'mean_{metric}']:.2f}, "
                f"n_notes={config_results['n_notes']}, "
                f"entropy={config_results['pitch_class_entropy']:.3f}, "
                f"scale={config_results['scale_consistency']:.1f}%, "
                f"groove={config_results['groove_consistency']:.1f}%, "
                f"deg={config_results['degradation']['total_degradation']:.2f}"
            )

    total_time = time.time() - start_time
    logger.info(f"\nGrid search complete in {total_time/60:.1f} minutes")

    # ---- Compute shifts from baseline ----
    for layer_name in results:
        baseline = results[layer_name].get(0.0, None)
        if baseline:
            for lam in results[layer_name]:
                results[layer_name][lam]["pitch_shift"] = float(
                    results[layer_name][lam]["mean_pitch"] - baseline["mean_pitch"]
                )
                results[layer_name][lam]["duration_shift"] = float(
                    results[layer_name][lam]["mean_duration"]
                    - baseline["mean_duration"]
                )

    # ---- Analyze each layer group ----
    analyses = {}
    for layer_name in selected_layer_groups:
        if layer_name in results:
            analyses[layer_name] = analyze_layer_group(
                results[layer_name], layer_name, args.lambda_values, args.concept
            )

    # ---- Save results ----
    output_file = args.output_dir / f"layer_lambda_grid_{args.concept}.json"

    # Convert defaultdict to regular dict for JSON
    results_dict = {
        layer: {str(lam): metrics for lam, metrics in lam_dict.items()}
        for layer, lam_dict in results.items()
    }
    analyses_dict = dict(analyses)

    with open(output_file, "w") as f:
        json.dump(
            {
                "concept": args.concept,
                "n_samples": args.n_samples,
                "max_seq_len": args.max_seq_len,
                "lambda_values": args.lambda_values,
                "layer_groups": {k: list(v) for k, v in selected_layer_groups.items()},
                "ground_truth_metrics": GROUND_TRUTH_METRICS,
                "total_time_minutes": total_time / 60,
                "results": results_dict,
                "analyses": analyses_dict,
            },
            f,
            indent=2,
        )
    logger.info(f"Results saved to {output_file}")

    # ---- Print comprehensive results ----
    print_results_table(results, analyses, args.lambda_values, args.concept)

    # ---- Save best config recommendation ----
    ranked = rank_configurations(analyses)
    if ranked:
        best_name, best_analysis = ranked[0]
        recommendation = {
            "concept": args.concept,
            "best_layer_group": best_name,
            "best_layers": selected_layer_groups[best_name],
            "composite_score": best_analysis["composite_score"],
            "pearson_r": best_analysis["pearson_r"],
            "r_squared": best_analysis["r_squared"],
            "slope": best_analysis["slope"],
            "effect_range": best_analysis["effect_range"],
            "is_monotonic": best_analysis["is_monotonic"],
            "avg_degradation": best_analysis["avg_extreme_degradation"],
        }

        # Also find best single layer
        single_ranked = [(n, a) for n, a in ranked if n.startswith("single_")]
        if single_ranked:
            best_single_name, best_single_analysis = single_ranked[0]
            recommendation["best_single_layer"] = best_single_name
            recommendation["best_single_layer_idx"] = selected_layer_groups[best_single_name][0]
            recommendation["best_single_score"] = best_single_analysis[
                "composite_score"
            ]

        rec_file = args.output_dir / f"best_config_{args.concept}.json"
        with open(rec_file, "w") as f:
            json.dump(recommendation, f, indent=2)
        logger.info(f"Best config saved to {rec_file}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Experiment complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
