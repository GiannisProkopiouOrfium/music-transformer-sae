"""Fingerprint sensitivity sweep: N ∈ {8, 16, 32, 64}.

Tests how sensitive temporal PID is to the number of monitored features
(the "concept fingerprint" T). Addresses reviewer concern about coupling
between the measurement signal and the intervention.

Usage:
    python pid_steering/run_fingerprint_sweep.py \
        --concept average_pitch \
        --n_values 8 16 32 64 \
        --n_samples 20 --gpu 0
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
from pid_steering.pid_steered_generator import load_model
from pid_steering.run_single_attribute import compute_generation_metrics

import config_pid

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep concept fingerprint size N for temporal PID"
    )
    parser.add_argument("--concept", type=str, default="average_pitch")
    parser.add_argument("--n_values", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
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
        default=config_pid.PID_EXPERIMENTS_DIR / "fingerprint_sweep",
    )
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

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

    vector_path = config_pid.SAS_VECTORS_DIR / f"{args.concept}_sas_vectors.pt"
    sas_vector = load_sas_vector(vector_path, layer_idx=args.target_layer)

    results = {}
    gt_h, gt_s, gt_g = 2.974, 92.26, 93.05

    for n_val in args.n_values:
        logger.info(f"\n{'='*60}")
        logger.info(f"Fingerprint N = {n_val}")
        logger.info(f"{'='*60}")

        # Override the target feature indices with custom N
        target_indices = get_target_feature_indices(sas_vector, top_n=n_val)
        logger.info(f"  Using {len(target_indices)} target features")

        # Create generator and override its target indices
        pid_gen = TemporalPIDSASGenerator(
            model,
            encoding,
            sae_model,
            sas_vector,
            args.target_layer,
        )
        pid_gen.target_feature_indices = target_indices

        pid_seqs, pid_diags = pid_gen.generate_with_temporal_pid(
            n_samples=args.n_samples,
            seq_len=args.seq_len,
            target_magnitude=args.target_magnitude,
            Kp=args.Kp,
            Ki=args.Ki,
            Kd=args.Kd,
            max_I=args.max_I,
            lambda_max=args.lambda_max,
            n_ramp_beats=args.n_ramp_steps,
            device=device,
        )

        pid_metrics = [compute_generation_metrics(s, encoding) for s in pid_seqs]
        avg_pitch = np.mean([m["average_pitch"] for m in pid_metrics])
        avg_dur = np.mean([m["average_duration"] for m in pid_metrics])
        avg_h = np.mean([m["pitch_class_entropy"] for m in pid_metrics])
        avg_s = np.mean([m["scale_consistency"] for m in pid_metrics])
        avg_g = np.mean([m["groove_consistency"] for m in pid_metrics])
        delta = abs(avg_h - gt_h) + max(0, gt_s - avg_s) + max(0, gt_g - avg_g)
        avg_lam = np.mean(
            [
                np.mean(d["lambda_trajectory"])
                for d in pid_diags
                if d["lambda_trajectory"]
            ]
        )

        results[str(n_val)] = {
            "n_features": int(len(target_indices)),
            "avg_pitch": float(avg_pitch),
            "avg_duration": float(avg_dur),
            "pc_entropy": float(avg_h),
            "scale_consistency": float(avg_s),
            "groove_consistency": float(avg_g),
            "degradation": float(delta),
            "avg_lambda": float(avg_lam),
            "per_sample_metrics": pid_metrics,
        }

        print(
            f"  N={n_val:>3d}: pitch={avg_pitch:.2f}, dur={avg_dur:.2f}, "
            f"δ={delta:.2f}, avg_λ={avg_lam:.3f}"
        )

    # Print summary
    print(f"\n{'='*70}")
    print(f"Fingerprint Sensitivity Sweep — {args.concept}")
    print(f"{'='*70}")
    print(
        f"{'N':>5} {'Pitch':>8} {'Dur':>8} {'PCE':>7} {'Scale%':>7} {'δ':>6} {'Avg λ':>7}"
    )
    print("-" * 50)
    for n_val in args.n_values:
        r = results[str(n_val)]
        print(
            f"{n_val:>5d} {r['avg_pitch']:>8.2f} {r['avg_duration']:>8.2f} "
            f"{r['pc_entropy']:>7.3f} {r['scale_consistency']:>7.1f} "
            f"{r['degradation']:>6.2f} {r['avg_lambda']:>7.3f}"
        )

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"fingerprint_sweep_{args.concept}.json"
    # Remove per_sample_metrics for compact JSON
    save_results = {
        k: {kk: vv for kk, vv in v.items() if kk != "per_sample_metrics"}
        for k, v in results.items()
    }
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
