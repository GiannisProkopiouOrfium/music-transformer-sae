"""Gain robustness sweep: perturb Kp and Ki around nominal values.

Tests whether PID steering degrades gracefully under gain perturbation.
Runs pitch-up and duration-up at multiple (Kp, Ki) perturbation levels.

Results go to appendix as a robustness table.
"""

import argparse
import json
import logging
import pathlib
import sys
from itertools import product

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

# Ground-truth for delta computation
GT_H, GT_S, GT_G = 2.974, 92.26, 93.05


def compute_delta(metrics_list):
    """Compute per-sample delta and return mean, std, CI."""
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
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
    }


def run_single_config(
    model, encoding, sae_model, sas_vector, concept, args, device, Kp, Ki, Kd, label
):
    """Run PID generation with a specific gain config."""
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
        n_ramp_beats=args.n_ramp_steps,
        device=device,
    )
    metrics = [compute_generation_metrics(s, encoding) for s in seqs]
    avg_lambda = float(
        np.mean(
            [np.mean(d["lambda_trajectory"]) for d in diags if d["lambda_trajectory"]]
        )
    )

    # Extract key attributes
    pitch_vals = [m["average_pitch"] for m in metrics]
    dur_vals = [m["average_duration"] for m in metrics]
    delta_info = compute_delta(metrics)

    return {
        "label": label,
        "Kp": Kp,
        "Ki": Ki,
        "Kd": Kd,
        "avg_pitch": float(np.mean(pitch_vals)),
        "avg_duration": float(np.mean(dur_vals)),
        "delta": delta_info,
        "avg_lambda": avg_lambda,
        "per_sample_metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Gain robustness sweep")
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
    parser.add_argument("--n_ramp_steps", type=int, default=64)
    parser.add_argument("--max_I", type=float, default=10.0)
    parser.add_argument(
        "--direction", type=str, default="positive", choices=["positive", "negative"]
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "gain_robustness",
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Nominal gains per concept
    if args.concept == "average_pitch":
        nom_Kp, nom_Ki, nom_Kd = 1.0, 0.05, 0.01
    else:
        nom_Kp, nom_Ki, nom_Kd = 1.0, 0.025, 0.01

    # Perturbation factors
    perturbations = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

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

    # 1. Sweep Kp (fix Ki, Kd at nominal)
    logger.info("=== Sweeping Kp ===")
    for factor in perturbations:
        Kp = nom_Kp * factor
        label = f"Kp={Kp:.2f} ({factor:.0%})"
        logger.info(f"Running {label}")
        result = run_single_config(
            model,
            encoding,
            sae_model,
            sas_vector,
            args.concept,
            args,
            device,
            Kp,
            nom_Ki,
            nom_Kd,
            label,
        )
        result["sweep"] = "Kp"
        result["factor"] = factor
        all_results.append(result)

    # 2. Sweep Ki (fix Kp, Kd at nominal)
    logger.info("=== Sweeping Ki ===")
    for factor in perturbations:
        Ki = nom_Ki * factor
        label = f"Ki={Ki:.4f} ({factor:.0%})"
        logger.info(f"Running {label}")
        result = run_single_config(
            model,
            encoding,
            sae_model,
            sas_vector,
            args.concept,
            args,
            device,
            nom_Kp,
            Ki,
            nom_Kd,
            label,
        )
        result["sweep"] = "Ki"
        result["factor"] = factor
        all_results.append(result)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"gain_robustness_{args.concept}_{args.direction}.json"
    # Strip per_sample_metrics for compact output
    compact = []
    for r in all_results:
        c = {k: v for k, v in r.items() if k != "per_sample_metrics"}
        compact.append(c)
    with open(out_path, "w") as f:
        json.dump(compact, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Gain Robustness Results: {args.concept} ({args.direction})")
    print(f"Nominal: Kp={nom_Kp}, Ki={nom_Ki}, Kd={nom_Kd}")
    print(f"{'='*80}")
    print(
        f"{'Sweep':<6} {'Factor':<8} {'Kp':<6} {'Ki':<8} {'Pitch':>8} {'Dur':>8} "
        f"{'δ':>8} {'avg λ':>8}"
    )
    print("-" * 72)
    for r in all_results:
        print(
            f"{r['sweep']:<6} {r['factor']:<8.2f} {r['Kp']:<6.2f} {r['Ki']:<8.4f} "
            f"{r['avg_pitch']:>8.2f} {r['avg_duration']:>8.2f} "
            f"{r['delta']['mean']:>8.2f} {r['avg_lambda']:>8.3f}"
        )

    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
