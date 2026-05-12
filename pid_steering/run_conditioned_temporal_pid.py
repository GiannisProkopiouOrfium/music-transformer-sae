"""Conditioned Temporal PID SAS Evaluation.

Generates conditioned continuations using temporal PID SAS steering
with songs from extreme quantiles. Supports both positive (steer UP)
and negative (steer DOWN) directions.

For each concept × direction:
  - temporal_pid: PID-controlled SAS with cosine ramp target
  - static_smooth: Cosine-ramped static SAS (failure baseline)
  - baseline: Unsteered conditioned generation

Cross-concept isolation: reports changes to ALL attributes when
steering only ONE concept.

Usage:
    python pid_steering/run_conditioned_temporal_pid.py \
        --concept all --n_songs 10 --n_per_song 3 --gpu 0
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pid_steering.temporal_pid_sas_generator import (
    TemporalPIDSASGenerator,
    load_sae_model,
    load_sas_vector,
    get_adaptive_k,
)
from pid_steering.pid_steered_generator import load_model
from pid_steering.conditioned_pid_evaluator import (
    load_song_tokens,
    extract_conditioning_prefix,
    find_extreme_songs,
    measure_concept_from_tokens,
    evaluate_quality_metrics,
    calculate_degradation,
)
from pid_steering.run_single_attribute import compute_generation_metrics

import config_pid
import representation

logger = logging.getLogger(__name__)


def generate_static_smooth_conditioned(
    model,
    encoding,
    sae_model,
    sas_vector,
    target_layer,
    conditioning_prefix: torch.Tensor,
    continuation_len: int,
    lambda_max: float,
    n_ramp_steps: int,
    device: torch.device,
) -> np.ndarray:
    """Generate a single conditioned sample with static smooth SAS."""
    eos = encoding["type_code_map"]["end-of-song"]
    conditioning_prefix = conditioning_prefix.to(device)

    handles = []
    _smooth_hook = None

    if lambda_max != 0.0 and n_ramp_steps > 0:
        sys.path.insert(
            0, str(pathlib.Path(__file__).parent.parent / "sparse_steering")
        )
        from smooth_steering_sas import SmoothSASSteeringHook

        sae_models = {target_layer: sae_model}
        sas_vectors_dict = {
            target_layer: (
                sas_vector if isinstance(sas_vector, np.ndarray) else sas_vector.numpy()
            )
        }
        _smooth_hook = SmoothSASSteeringHook(
            sae_models=sae_models,
            sas_vectors=sas_vectors_dict,
            concept="static_smooth",
            steering_strength=lambda_max,
            layers_to_steer=[target_layer],
            mode="ramp_up",
            schedule="cosine",
            n_ramp=n_ramp_steps,
        )

        layers = model.decoder.net.attn_layers.layers
        layer_module = layers[target_layer]
        if isinstance(layer_module, torch.nn.ModuleList) and len(layer_module) > 1:
            target_module = layer_module[1]
        else:
            target_module = layer_module

        h = target_module.register_forward_hook(
            lambda mod, inp, out, idx=target_layer: _smooth_hook(mod, inp, out, idx)
        )
        handles.append(h)

    try:
        if _smooth_hook is not None:
            _smooth_hook.reset()

        with torch.no_grad():
            generated = model.generate(
                conditioning_prefix,
                continuation_len,
                eos_token=eos,
                temperature=config_pid.TEMPERATURE,
                filter_logits_fn="top_k",
                filter_thres=config_pid.FILTER_THRESHOLD,
                monotonicity_dim=("type", "beat"),
            )
        full_seq = torch.cat((conditioning_prefix, generated), 1).cpu().numpy()[0]
    finally:
        for h in handles:
            h.remove()

    return full_seq


def main():
    parser = argparse.ArgumentParser(
        description="Conditioned temporal PID SAS evaluation"
    )
    parser.add_argument(
        "--concept",
        type=str,
        nargs="+",
        default=["average_pitch", "average_duration"],
        help="Concept(s). Use 'all' for both.",
    )
    parser.add_argument(
        "--n_songs", type=int, default=10, help="Songs per extreme group"
    )
    parser.add_argument(
        "--n_per_song", type=int, default=2, help="Continuations per song"
    )
    parser.add_argument(
        "--continuation_len", type=int, default=config_pid.CONTINUATION_LEN
    )
    parser.add_argument(
        "--conditioning_beats", type=int, default=config_pid.CONDITIONING_BEATS
    )
    parser.add_argument("--target_layer", type=int, default=config_pid.SAS_LAYER)
    parser.add_argument("--target_magnitude", type=float, default=1.0)
    parser.add_argument("--lambda_max", type=float, default=3.0)
    parser.add_argument("--n_ramp_steps", type=int, default=64)
    parser.add_argument(
        "--Kp", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Kp"]
    )
    parser.add_argument(
        "--Ki", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Ki"]
    )
    parser.add_argument(
        "--Kd", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Kd"]
    )
    parser.add_argument(
        "--max_I", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["max_I"]
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "conditioned_temporal_sas",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip_wav", action="store_true")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generation if MIDI already exists on disk",
    )
    parser.add_argument(
        "--song_offset",
        type=int,
        default=0,
        help="Skip first N songs in each extreme group (to generate new ones)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    concepts = args.concept
    if concepts == ["all"]:
        concepts = ["average_pitch", "average_duration"]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model, encoding, _ = load_model(
        config_pid.MODEL_CHECKPOINT,
        config_pid.TRAIN_ARGS_PATH,
        config_pid.ENCODING_PATH,
        device,
    )

    k = get_adaptive_k(args.target_layer)
    sae_path = config_pid.SAE_CHECKPOINT_DIR / f"sae_layer_{args.target_layer}_best.pt"
    sae_model = load_sae_model(sae_path, k, device)

    notes_dir = config_pid.PROJECT_ROOT / "data" / "sod" / "processed" / "notes"
    all_results = {}

    for concept in concepts:
        logger.info(f"\n{'='*70}")
        logger.info(f"Conditioned Temporal PID SAS — {concept}")
        logger.info(f"{'='*70}")

        vector_path = config_pid.SAS_VECTORS_DIR / f"{concept}_sas_vectors.pt"
        sas_vector = load_sas_vector(vector_path, layer_idx=args.target_layer)

        low_songs, high_songs = find_extreme_songs(
            notes_dir,
            encoding,
            concept,
            n_songs=args.n_songs + args.song_offset,
            conditioning_beats=args.conditioning_beats,
        )
        # Apply song offset to skip already-generated songs
        if args.song_offset > 0:
            low_songs = low_songs[args.song_offset :]
            high_songs = high_songs[args.song_offset :]
            logger.info(
                f"Song offset={args.song_offset}: using songs {args.song_offset}-{args.song_offset + len(low_songs) - 1}"
            )

        # Two directions:
        # 1. low songs → steer UP (positive vector)
        # 2. high songs → steer DOWN (negative vector)
        directions = [
            ("low_steer_up", low_songs, sas_vector),
            ("high_steer_down", high_songs, -sas_vector),
        ]

        concept_results = {}

        for dir_name, songs, directed_vector in directions:
            logger.info(
                f"\n--- {dir_name} ({len(songs)} songs × {args.n_per_song} each) ---"
            )

            pid_gen = TemporalPIDSASGenerator(
                model,
                encoding,
                sae_model,
                directed_vector,
                args.target_layer,
            )

            method_results = {"temporal_pid": [], "static_smooth": [], "baseline": []}
            method_diagnostics = []

            for song_idx, (filepath, init_val) in enumerate(songs):
                song_stem = filepath.stem  # SOD piece name
                abs_idx = song_idx + args.song_offset
                tokens = load_song_tokens(filepath, encoding)
                cond = extract_conditioning_prefix(tokens, args.conditioning_beats)

                for rep in range(args.n_per_song):
                    sample_id = f"song{abs_idx}_{song_stem}_rep{rep}"

                    # Check skip_existing
                    if args.skip_existing:
                        pid_path = (
                            args.output_dir
                            / concept
                            / dir_name
                            / "temporal_pid"
                            / f"{sample_id}.mid"
                        )
                        if pid_path.exists():
                            logger.info(f"  Skipping {sample_id} (already exists)")
                            continue

                    # 1. Temporal PID
                    seq, diag = pid_gen.generate_conditioned(
                        conditioning_prefix=cond,
                        continuation_len=args.continuation_len,
                        target_magnitude=args.target_magnitude,
                        Kp=args.Kp,
                        Ki=args.Ki,
                        Kd=args.Kd,
                        max_I=args.max_I,
                        lambda_max=args.lambda_max,
                        n_ramp_beats=args.n_ramp_steps,
                        device=device,
                    )
                    metrics = compute_generation_metrics(seq, encoding)
                    metrics["init_value"] = float(init_val)
                    metrics["sample_id"] = sample_id
                    method_results["temporal_pid"].append(metrics)
                    method_diagnostics.append(diag)

                    # Save MIDI
                    _save_midi(
                        seq,
                        encoding,
                        args.output_dir / concept / dir_name / "temporal_pid",
                        sample_id,
                    )

                    # 2. Static smooth
                    seq_s = generate_static_smooth_conditioned(
                        model,
                        encoding,
                        sae_model,
                        directed_vector,
                        args.target_layer,
                        conditioning_prefix=cond,
                        continuation_len=args.continuation_len,
                        lambda_max=args.lambda_max,
                        n_ramp_steps=args.n_ramp_steps,
                        device=device,
                    )
                    metrics_s = compute_generation_metrics(seq_s, encoding)
                    metrics_s["init_value"] = float(init_val)
                    metrics_s["sample_id"] = sample_id
                    method_results["static_smooth"].append(metrics_s)
                    _save_midi(
                        seq_s,
                        encoding,
                        args.output_dir / concept / dir_name / "static_smooth",
                        sample_id,
                    )

                    # 3. Baseline (unsteered)
                    seq_b = generate_static_smooth_conditioned(
                        model,
                        encoding,
                        sae_model,
                        directed_vector,
                        args.target_layer,
                        conditioning_prefix=cond,
                        continuation_len=args.continuation_len,
                        lambda_max=0.0,
                        n_ramp_steps=0,
                        device=device,
                    )
                    metrics_b = compute_generation_metrics(seq_b, encoding)
                    metrics_b["init_value"] = float(init_val)
                    metrics_b["sample_id"] = sample_id
                    method_results["baseline"].append(metrics_b)
                    _save_midi(
                        seq_b,
                        encoding,
                        args.output_dir / concept / dir_name / "baseline",
                        sample_id,
                    )

                if (song_idx + 1) % 5 == 0:
                    logger.info(f"  Processed {song_idx + 1}/{len(songs)} songs")

            # Aggregate
            dir_summary = {}
            for method, metric_list in method_results.items():
                avg = {}
                for key in [
                    "average_pitch",
                    "average_duration",
                    "pitch_class_entropy",
                    "scale_consistency",
                    "groove_consistency",
                ]:
                    vals = [m[key] for m in metric_list if key in m]
                    avg[f"{key}_mean"] = float(np.nanmean(vals)) if vals else 0
                    avg[f"{key}_std"] = float(np.nanstd(vals)) if vals else 0
                dir_summary[method] = avg

            if method_diagnostics:
                dir_summary["pid_avg_lambda"] = float(
                    np.mean(
                        [
                            np.mean(d["lambda_trajectory"])
                            for d in method_diagnostics
                            if d["lambda_trajectory"]
                        ]
                    )
                )

            concept_results[dir_name] = dir_summary

            # Save diagnostics
            diag_path = args.output_dir / concept / f"diagnostics_{dir_name}.json"
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            with open(diag_path, "w") as f:
                json.dump(method_diagnostics, f, indent=2, default=float)

        all_results[concept] = concept_results
        _print_concept_summary(concept, concept_results)

    # Cross-concept isolation analysis
    print(f"\n{'='*80}")
    print("CROSS-CONCEPT ISOLATION ANALYSIS")
    print(f"{'='*80}")
    for concept in concepts:
        cr = all_results.get(concept, {})
        target_attr = concept  # e.g. "average_pitch"
        other_attr = (
            "average_duration" if concept == "average_pitch" else "average_pitch"
        )

        for dir_name, data in cr.items():
            baseline = data.get("baseline", {})
            pid = data.get("temporal_pid", {})
            static = data.get("static_smooth", {})

            target_shift_pid = pid.get(f"{target_attr}_mean", 0) - baseline.get(
                f"{target_attr}_mean", 0
            )
            target_shift_static = static.get(f"{target_attr}_mean", 0) - baseline.get(
                f"{target_attr}_mean", 0
            )
            collateral_pid = abs(
                pid.get(f"{other_attr}_mean", 0) - baseline.get(f"{other_attr}_mean", 0)
            )
            collateral_static = abs(
                static.get(f"{other_attr}_mean", 0)
                - baseline.get(f"{other_attr}_mean", 0)
            )

            print(f"\n{concept} / {dir_name}:")
            print(
                f"  Target shift:     PID {target_shift_pid:+.2f}  |  Static {target_shift_static:+.2f}"
            )
            print(
                f"  Collateral ({other_attr.replace('average_', '')}): "
                f"PID {collateral_pid:.2f}  |  Static {collateral_static:.2f}"
            )
            if collateral_static > 0:
                ratio = collateral_pid / collateral_static
                print(f"  Isolation ratio:  {ratio:.2f}x (< 1.0 = PID better)")

    # Save all
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "conditioned_temporal_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info(f"\nSaved results to {results_path}")


def _save_midi(seq, encoding, output_dir, sample_id):
    """Save a token sequence as MIDI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        music = representation.decode(seq, encoding)
        if music is not None:
            music.write(str(output_dir / f"{sample_id}.mid"))
    except Exception as e:
        logger.warning(f"MIDI save failed for {sample_id}: {e}")


def _print_concept_summary(concept, concept_results):
    """Print summary table for a concept."""
    for dir_name, data in concept_results.items():
        print(f"\n{'='*70}")
        print(f"Conditioned Temporal PID — {concept} / {dir_name}")
        print(f"{'='*70}")
        print(
            f"{'Method':<18} {'Avg Pitch':<12} {'Avg Dur':<10} "
            f"{'Entropy':<10} {'Scale%':<8} {'Groove%':<9}"
        )
        print("-" * 70)
        for method in ["temporal_pid", "static_smooth", "baseline"]:
            m = data.get(method, {})
            print(
                f"{method:<18} "
                f"{m.get('average_pitch_mean', 0):<12.2f} "
                f"{m.get('average_duration_mean', 0):<10.2f} "
                f"{m.get('pitch_class_entropy_mean', 0):<10.3f} "
                f"{m.get('scale_consistency_mean', 0):<8.1f} "
                f"{m.get('groove_consistency_mean', 0):<9.1f}"
            )
        if "pid_avg_lambda" in data:
            print(f"PID avg λ: {data['pid_avg_lambda']:.3f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
