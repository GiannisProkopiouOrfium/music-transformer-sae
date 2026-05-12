"""Piecewise-Constant λ Experiment for Duration-Up Mitigation.

Tests whether holding λ(t) fixed between PID updates reduces
the within-sequence variation that degrades scale consistency
for duration-up steering.

Variants:
  - update_interval=1  (per-token PID, current default)
  - update_interval=4  (bar-level updates)
  - update_interval=8  (phrase-level updates)
  - update_interval=16 (section-level updates)
  - static SAS (fixed λ=3.0 baseline)

All variants use duration-up steering (positive direction)
with n=40 unconditioned samples.

Usage:
    python pid_steering/run_piecewise_lambda.py --gpu 0
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List

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
    get_target_feature_indices,
)
from pid_steering.temporal_pid_sas_hook import TemporalPIDSASHook
from pid_steering.pid_steered_generator import load_model
from pid_steering.run_single_attribute import compute_generation_metrics
from pid_steering.run_temporal_pid_sas import generate_static_smooth

import config_pid

logger = logging.getLogger(__name__)


def generate_piecewise_pid(
    model,
    encoding,
    sae_model,
    sas_vector,
    target_layer,
    n_samples,
    seq_len,
    target_magnitude,
    Kp,
    Ki,
    Kd,
    max_I,
    lambda_max,
    n_ramp_steps,
    update_interval,
    device,
):
    """Generate with piecewise-constant PID (specified update interval)."""
    import torch.nn as nn

    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    target_feature_indices = get_target_feature_indices(sas_vector, top_n=32)

    # Navigate model
    if hasattr(model, "decoder"):
        transformer = model.decoder.net
    else:
        transformer = model.net
    attn_layers = transformer.attn_layers
    layer_module = attn_layers.layers[target_layer]
    if isinstance(layer_module, nn.ModuleList) and len(layer_module) > 1:
        target_module = layer_module[1]
    else:
        target_module = layer_module

    sequences = []
    diagnostics_list = []

    for i in range(n_samples):
        hook = TemporalPIDSASHook(
            sae_model=sae_model,
            sas_vector=sas_vector,
            target_feature_indices=target_feature_indices,
            target_magnitude=target_magnitude,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            max_I=max_I,
            lambda_min=0.0,
            lambda_max=lambda_max,
            n_ramp_beats=n_ramp_steps,
            update_interval=update_interval,
        )
        hook.reset()

        handle = target_module.register_forward_hook(
            lambda mod, inp, out, h=hook: h(mod, inp, out, target_layer)
        )

        try:
            start = torch.zeros((1, 1, 6), dtype=torch.long, device=device)
            start[:, 0, 0] = sos

            with torch.no_grad():
                generated = model.generate(
                    start,
                    seq_len,
                    eos_token=eos,
                    temperature=config_pid.TEMPERATURE,
                    filter_logits_fn="top_k",
                    filter_thres=config_pid.FILTER_THRESHOLD,
                    monotonicity_dim=("type", "beat"),
                )
            full_seq = torch.cat((start, generated), 1).cpu().numpy()[0]
            sequences.append(full_seq)
            diagnostics_list.append(hook.get_diagnostics())
        finally:
            handle.remove()

        if (i + 1) % 10 == 0:
            diag = diagnostics_list[-1]
            avg_lam = (
                np.mean(diag["lambda_trajectory"]) if diag["lambda_trajectory"] else 0
            )
            logger.info(f"  [{i + 1}/{n_samples}] avg λ={avg_lam:.3f}")

    return sequences, diagnostics_list


def main():
    parser = argparse.ArgumentParser(description="Piecewise-constant λ experiment")
    parser.add_argument(
        "--concept",
        type=str,
        default="average_duration",
        help="Concept to test (default: average_duration)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="up",
        choices=["up", "down"],
        help="Steering direction",
    )
    parser.add_argument("--n_samples", type=int, default=40)
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
    parser.add_argument("--target_layer", type=int, default=config_pid.SAS_LAYER)
    parser.add_argument("--target_magnitude", type=float, default=2.0)
    parser.add_argument("--lambda_max", type=float, default=3.0)
    parser.add_argument("--n_ramp_steps", type=int, default=64)
    parser.add_argument(
        "--Kp", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Kp"]
    )
    parser.add_argument(
        "--Ki", type=float, default=None, help="Ki (default: concept-specific)"
    )
    parser.add_argument(
        "--Kd", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["Kd"]
    )
    parser.add_argument(
        "--max_I", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["max_I"]
    )
    parser.add_argument(
        "--update_intervals",
        type=int,
        nargs="+",
        default=[1, 4, 8, 16],
        help="Update intervals to test",
    )
    parser.add_argument(
        "--include_static",
        action="store_true",
        default=True,
        help="Also run static SAS baseline",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "piecewise_lambda",
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Concept-specific Ki
    KI_MAP = {"average_pitch": 0.05, "average_duration": 0.025}
    Ki = args.Ki if args.Ki is not None else KI_MAP.get(args.concept, 0.05)

    model, encoding, _ = load_model(
        config_pid.MODEL_CHECKPOINT,
        config_pid.TRAIN_ARGS_PATH,
        config_pid.ENCODING_PATH,
        device,
    )

    k = get_adaptive_k(args.target_layer)
    sae_path = config_pid.SAE_CHECKPOINT_DIR / f"sae_layer_{args.target_layer}_best.pt"
    sae_model = load_sae_model(sae_path, k, device)

    vector_path = config_pid.SAS_VECTORS_DIR / f"{args.concept}_sas_vectors.pt"
    sas_vector = load_sas_vector(vector_path, layer_idx=args.target_layer)

    # Direction
    if args.direction == "down":
        sas_vector = -sas_vector

    all_results = {}

    # ─── PID variants with different update intervals ──────────────────
    for interval in args.update_intervals:
        label = f"pid_interval_{interval}"
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Running PID with update_interval={interval} ({args.n_samples} samples)"
        )
        logger.info(f"{'='*60}")

        sequences, diagnostics = generate_piecewise_pid(
            model,
            encoding,
            sae_model,
            sas_vector,
            args.target_layer,
            args.n_samples,
            args.seq_len,
            args.target_magnitude,
            args.Kp,
            Ki,
            args.Kd,
            args.max_I,
            args.lambda_max,
            args.n_ramp_steps,
            interval,
            device,
        )

        metrics_list = [compute_generation_metrics(s, encoding) for s in sequences]

        # Aggregate
        agg = _aggregate_metrics(metrics_list)
        agg["update_interval"] = interval
        agg["avg_lambda"] = float(
            np.mean(
                [
                    np.mean(d["lambda_trajectory"])
                    for d in diagnostics
                    if d["lambda_trajectory"]
                ]
            )
        )
        agg["lambda_std_within"] = float(
            np.mean(
                [
                    np.std(d["lambda_trajectory"])
                    for d in diagnostics
                    if d["lambda_trajectory"]
                ]
            )
        )

        all_results[label] = agg
        logger.info(
            f"  {args.concept}: {agg.get(f'{args.concept}_mean', 0):.2f} | "
            f"Scale: {agg.get('scale_consistency_mean', 0):.1f}% | "
            f"δ: {agg.get('degradation_mean', 0):.2f} | "
            f"avg λ: {agg['avg_lambda']:.3f} | "
            f"λ std: {agg['lambda_std_within']:.4f}"
        )

    # ─── Static SAS baseline ──────────────────────────────────────────
    if args.include_static:
        logger.info(f"\n{'='*60}")
        logger.info(
            f"Running Static SAS λ={args.lambda_max} ({args.n_samples} samples)"
        )
        logger.info(f"{'='*60}")

        static_seqs, _ = generate_static_smooth(
            model,
            encoding,
            sae_model,
            sas_vector,
            args.target_layer,
            lambda_max=args.lambda_max,
            n_ramp_steps=args.n_ramp_steps,
            n_samples=args.n_samples,
            seq_len=args.seq_len,
            device=device,
        )

        static_metrics = [compute_generation_metrics(s, encoding) for s in static_seqs]
        agg_static = _aggregate_metrics(static_metrics)
        agg_static["update_interval"] = "static"
        agg_static["avg_lambda"] = args.lambda_max
        agg_static["lambda_std_within"] = 0.0
        all_results["static_sas"] = agg_static

        logger.info(
            f"  {args.concept}: {agg_static.get(f'{args.concept}_mean', 0):.2f} | "
            f"Scale: {agg_static.get('scale_consistency_mean', 0):.1f}% | "
            f"δ: {agg_static.get('degradation_mean', 0):.2f}"
        )

    # ─── Save & Print ──────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"piecewise_{args.concept}_{args.direction}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info(f"\nSaved to {results_path}")

    _print_summary(args.concept, all_results)


def _aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    """Aggregate a list of per-sample metric dicts."""
    agg = {}
    H0 = config_pid.GROUND_TRUTH["pitch_class_entropy"]
    S0 = config_pid.GROUND_TRUTH["scale_consistency"]
    G0 = config_pid.GROUND_TRUTH["groove_consistency"]

    keys = [
        "average_pitch",
        "average_duration",
        "pitch_class_entropy",
        "scale_consistency",
        "groove_consistency",
    ]
    for key in keys:
        vals = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
        agg[f"{key}_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{key}_std"] = float(np.std(vals)) if vals else float("nan")

    # Degradation
    degs = []
    for m in metrics_list:
        h = m.get("pitch_class_entropy", H0)
        s = m.get("scale_consistency", S0)
        g = m.get("groove_consistency", G0)
        if any(np.isnan(x) for x in [h, s, g]):
            continue
        d = abs(h - H0) + max(0, S0 - s) + max(0, G0 - g)
        degs.append(d)
    agg["degradation_mean"] = float(np.mean(degs)) if degs else float("nan")
    agg["degradation_std"] = float(np.std(degs)) if degs else float("nan")
    agg["n_samples"] = len(metrics_list)

    return agg


def _print_summary(concept, all_results):
    """Print formatted comparison table."""
    print(f"\n{'='*90}")
    print(f"PIECEWISE-CONSTANT λ RESULTS — {concept}")
    print(f"{'='*90}")
    print(
        f"{'Method':<22} {'Interval':<10} {'Attr':<10} {'Scale%':<10} "
        f"{'δ':<8} {'avg λ':<8} {'λ std':<8}"
    )
    print("-" * 90)

    for label, data in sorted(all_results.items()):
        print(
            f"{label:<22} "
            f"{str(data.get('update_interval', '?')):<10} "
            f"{data.get(f'{concept}_mean', 0):<10.2f} "
            f"{data.get('scale_consistency_mean', 0):<10.1f} "
            f"{data.get('degradation_mean', 0):<8.2f} "
            f"{data.get('avg_lambda', 0):<8.3f} "
            f"{data.get('lambda_std_within', 0):<8.4f}"
        )
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
