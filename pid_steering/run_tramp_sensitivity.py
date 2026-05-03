"""T_ramp sensitivity sweep: test PID across different ramp horizons.

Tests how the PID controller adapts to shorter/longer ramp durations
without re-tuning gains. This addresses the reviewer question about
robustness to ramping horizons.
"""

import argparse
import json
import logging
import pathlib
import sys

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
from pid_steering.run_single_attribute import compute_generation_metrics

import config_pid

logger = logging.getLogger(__name__)

GT_H, GT_S, GT_G = 2.974, 92.26, 93.05


def compute_delta(metrics_list):
    deltas = []
    for m in metrics_list:
        d = (
            abs(m["pitch_class_entropy"] - GT_H)
            + max(0, GT_S - m["scale_consistency"])
            + max(0, GT_G - m["groove_consistency"])
        )
        deltas.append(d)
    return {
        "mean": float(np.mean(deltas)),
        "std": float(np.std(deltas)),
    }


def main():
    parser = argparse.ArgumentParser(description="T_ramp sensitivity sweep")
    parser.add_argument(
        "--concept",
        type=str,
        default="average_pitch",
        choices=["average_pitch", "average_duration"],
    )
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
    parser.add_argument("--target_layer", type=int, default=config_pid.SAS_LAYER)
    parser.add_argument("--target_magnitude", type=float, default=1.0)
    parser.add_argument("--lambda_max", type=float, default=3.0)
    parser.add_argument("--max_I", type=float, default=10.0)
    parser.add_argument(
        "--direction", type=str, default="positive", choices=["positive", "negative"]
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "tramp_sensitivity",
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Nominal gains per concept
    if args.concept == "average_pitch":
        Kp, Ki, Kd = 1.0, 0.05, 0.01
    else:
        Kp, Ki, Kd = 1.0, 0.025, 0.01

    # T_ramp values to test
    tramp_values = [16, 32, 64, 128, 256]

    # Load model and SAE
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
    if args.direction == "negative":
        sas_vector = -sas_vector

    all_results = []

    for tramp in tramp_values:
        logger.info(f"=== T_ramp = {tramp} ===")

        pid_gen = TemporalPIDSASGenerator(
            model,
            encoding,
            sae_model,
            sas_vector,
            args.target_layer,
        )
        seqs, diags = pid_gen.generate_with_temporal_pid(
            n_samples=args.n_samples,
            seq_len=args.seq_len,
            target_magnitude=args.target_magnitude,
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            max_I=args.max_I,
            lambda_max=args.lambda_max,
            n_ramp_beats=tramp,
            device=device,
        )

        metrics = [compute_generation_metrics(s, encoding) for s in seqs]
        avg_lambda = float(
            np.mean(
                [
                    np.mean(d["lambda_trajectory"])
                    for d in diags
                    if d["lambda_trajectory"]
                ]
            )
        )
        pitch_vals = [m["average_pitch"] for m in metrics]
        dur_vals = [m["average_duration"] for m in metrics]
        delta_info = compute_delta(metrics)

        all_results.append(
            {
                "T_ramp": tramp,
                "avg_pitch": float(np.mean(pitch_vals)),
                "avg_duration": float(np.mean(dur_vals)),
                "delta": delta_info,
                "avg_lambda": avg_lambda,
            }
        )

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.output_dir / f"tramp_sensitivity_{args.concept}_{args.direction}.json"
    )
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"T_ramp Sensitivity: {args.concept} ({args.direction})")
    print(f"Gains: Kp={Kp}, Ki={Ki}, Kd={Kd}")
    print(f"{'='*70}")
    print(f"{'T_ramp':>8} {'Pitch':>8} {'Dur':>8} {'δ':>8} {'avg λ':>8}")
    print("-" * 48)
    for r in all_results:
        print(
            f"{r['T_ramp']:>8} {r['avg_pitch']:>8.2f} {r['avg_duration']:>8.2f} "
            f"{r['delta']['mean']:>8.2f} {r['avg_lambda']:>8.3f}"
        )

    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
