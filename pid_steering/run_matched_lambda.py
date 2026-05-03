"""Static SAS at matched lambda: test if duration-up degradation is PID-specific.

Runs static SAS at lambda=1.0 (matching PID's avg lambda) for duration-up
to determine whether the scale consistency degradation is caused by PID's
dynamic overshoot or is inherent to the SAS duration-up vector at any lambda.

This directly answers: "If the SAS vector itself is the culprit, why does
static SAS at lambda=3.0 degrade less?"

Hypothesis: Static at lambda=1.0 will show LESS degradation than PID
because PID overshoots lambda during the ramp (briefly hitting lambda_max)
before settling. If static at lambda=1.0 also degrades, the SAS vector
is confirmed as the root cause.
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
    load_sae_model,
    load_sas_vector,
    get_adaptive_k,
)
from pid_steering.pid_steered_generator import load_model
from pid_steering.run_single_attribute import compute_generation_metrics
from pid_steering.run_temporal_pid_sas import generate_static_smooth

import config_pid

logger = logging.getLogger(__name__)

GT_H, GT_S, GT_G = 2.974, 92.26, 93.05


def compute_delta_per_sample(metrics_list):
    deltas = []
    for m in metrics_list:
        d = (
            abs(m["pitch_class_entropy"] - GT_H)
            + max(0, GT_S - m["scale_consistency"])
            + max(0, GT_G - m["groove_consistency"])
        )
        deltas.append(d)
    return deltas


def main():
    parser = argparse.ArgumentParser(
        description="Static SAS at matched lambda for duration-up analysis"
    )
    parser.add_argument("--concept", type=str, default="average_duration")
    parser.add_argument("--n_samples", type=int, default=40)
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
    parser.add_argument("--target_layer", type=int, default=config_pid.SAS_LAYER)
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "matched_lambda",
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Lambda values to test (PID averages ~1.0, static uses 3.0)
    lambda_values = [0.5, 1.0, 1.5, 2.0, 3.0]

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

    all_results = []

    for lam in lambda_values:
        logger.info(f"=== Static SAS at lambda={lam} ===")

        # n_ramp_steps=1 means essentially instant ramp (constant lambda)
        seqs, _ = generate_static_smooth(
            model,
            encoding,
            sae_model,
            sas_vector,
            args.target_layer,
            lambda_max=lam,
            n_ramp_steps=1,
            n_samples=args.n_samples,
            seq_len=args.seq_len,
            device=device,
        )

        metrics = [compute_generation_metrics(s, encoding) for s in seqs]
        deltas = compute_delta_per_sample(metrics)

        pitch_vals = [m["average_pitch"] for m in metrics]
        dur_vals = [m["average_duration"] for m in metrics]
        scale_vals = [m["scale_consistency"] for m in metrics]
        groove_vals = [m["groove_consistency"] for m in metrics]

        all_results.append(
            {
                "lambda": lam,
                "avg_pitch": float(np.mean(pitch_vals)),
                "avg_duration": float(np.mean(dur_vals)),
                "avg_scale": float(np.mean(scale_vals)),
                "avg_groove": float(np.mean(groove_vals)),
                "delta_mean": float(np.mean(deltas)),
                "delta_std": float(np.std(deltas)),
            }
        )

    # Save
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"matched_lambda_{args.concept}_positive.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Static SAS at Matched Lambda: {args.concept} (positive)")
    print(f"{'='*80}")
    print(
        f"{'λ':>6} {'Pitch':>8} {'Dur':>8} {'Scale%':>8} {'Groove%':>8} "
        f"{'δ mean':>8} {'δ std':>8}"
    )
    print("-" * 64)
    for r in all_results:
        print(
            f"{r['lambda']:>6.1f} {r['avg_pitch']:>8.2f} {r['avg_duration']:>8.2f} "
            f"{r['avg_scale']:>8.1f} {r['avg_groove']:>8.1f} "
            f"{r['delta_mean']:>8.2f} {r['delta_std']:>8.2f}"
        )
    print()
    print("Compare with PID: avg_lambda≈1.0, δ=8.45, scale=84.7%")
    print("If static at λ=1.0 shows similar scale drop → SAS vector is the cause")
    print("If static at λ=1.0 maintains scale → PID overshoot is the cause")

    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
