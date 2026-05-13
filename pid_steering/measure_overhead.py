"""Measure per-token wall-clock overhead of temporal PID vs baseline generation.

Reports the mean and std of per-token generation time for:
  1. Unsteered baseline
  2. Static SAS (fixed λ)
  3. Temporal PID SAS

Usage:
    python pid_steering/measure_overhead.py --n_samples 10 --gpu 0
"""

import argparse
import logging
import pathlib
import sys
import time

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
from pid_steering.run_temporal_pid_sas import generate_static_smooth
import config_pid

logger = logging.getLogger(__name__)


def time_generation(gen_fn, n_samples):
    """Time per-sample generation, returning (times, token_counts)."""
    times = []
    token_counts = []
    for i in range(n_samples):
        t0 = time.perf_counter()
        seqs, _ = gen_fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        # seqs is a list with 1 element; count actual tokens generated
        token_counts.append(len(seqs[0]) - 1)  # subtract SOS token
    return times, token_counts


def main():
    parser = argparse.ArgumentParser(description="Measure PID overhead")
    parser.add_argument("--concept", type=str, default="average_pitch")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
    parser.add_argument("--target_layer", type=int, default=config_pid.SAS_LAYER)
    parser.add_argument("--target_magnitude", type=float, default=2.0)
    parser.add_argument("--lambda_max", type=float, default=3.0)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
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

    pid_gen = TemporalPIDSASGenerator(
        model,
        encoding,
        sae_model,
        sas_vector,
        args.target_layer,
    )

    # Warmup (2 samples each to stabilize CUDA)
    logger.info("Warming up...")
    generate_static_smooth(
        model,
        encoding,
        sae_model,
        sas_vector,
        args.target_layer,
        lambda_max=0.0,
        n_ramp_steps=64,
        n_samples=2,
        seq_len=args.seq_len,
        device=device,
    )
    pid_gen.generate_with_temporal_pid(
        n_samples=2,
        seq_len=args.seq_len,
        target_magnitude=args.target_magnitude,
        lambda_max=args.lambda_max,
        device=device,
    )

    # --- Baseline (lambda=0) ---
    logger.info(f"Timing baseline ({args.n_samples} samples)...")
    baseline_times, baseline_lens = time_generation(
        lambda: generate_static_smooth(
            model,
            encoding,
            sae_model,
            sas_vector,
            args.target_layer,
            lambda_max=0.0,
            n_ramp_steps=64,
            n_samples=1,
            seq_len=args.seq_len,
            device=device,
        ),
        args.n_samples,
    )

    # --- Static SAS (fixed lambda) ---
    logger.info(f"Timing static SAS ({args.n_samples} samples)...")
    static_times, static_lens = time_generation(
        lambda: generate_static_smooth(
            model,
            encoding,
            sae_model,
            sas_vector,
            args.target_layer,
            lambda_max=args.lambda_max,
            n_ramp_steps=64,
            n_samples=1,
            seq_len=args.seq_len,
            device=device,
        ),
        args.n_samples,
    )

    # --- Temporal PID ---
    logger.info(f"Timing temporal PID ({args.n_samples} samples)...")
    pid_times, pid_lens = time_generation(
        lambda: pid_gen.generate_with_temporal_pid(
            n_samples=1,
            seq_len=args.seq_len,
            target_magnitude=args.target_magnitude,
            lambda_max=args.lambda_max,
            device=device,
        ),
        args.n_samples,
    )

    # Results
    baseline_arr = np.array(baseline_times)
    static_arr = np.array(static_times)
    pid_arr = np.array(pid_times)
    baseline_len_arr = np.array(baseline_lens)
    static_len_arr = np.array(static_lens)
    pid_len_arr = np.array(pid_lens)

    # Per-token times using ACTUAL token counts
    baseline_per_tok = baseline_arr / baseline_len_arr * 1000  # ms
    static_per_tok = static_arr / static_len_arr * 1000
    pid_per_tok = pid_arr / pid_len_arr * 1000

    # Overhead: compare per-token times (apples-to-apples)
    overhead_static = (
        (static_per_tok.mean() - baseline_per_tok.mean())
        / baseline_per_tok.mean()
        * 100
    )
    overhead_pid = (
        (pid_per_tok.mean() - baseline_per_tok.mean()) / baseline_per_tok.mean() * 100
    )
    overhead_pid_vs_static = (
        (pid_per_tok.mean() - static_per_tok.mean()) / static_per_tok.mean() * 100
    )

    print(f"\n{'='*70}")
    print(f"Overhead Measurement (n={args.n_samples}, max_seq_len={args.seq_len})")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Total (s)':<16} {'Tokens':<14} {'Per-token (ms)':<16}")
    print(f"{'-'*66}")
    print(
        f"{'Baseline':<20} "
        f"{baseline_arr.mean():.3f}±{baseline_arr.std():.3f}  "
        f"{baseline_len_arr.mean():.0f}±{baseline_len_arr.std():.0f}    "
        f"{baseline_per_tok.mean():.3f}±{baseline_per_tok.std():.3f}"
    )
    print(
        f"{'Static SAS':<20} "
        f"{static_arr.mean():.3f}±{static_arr.std():.3f}  "
        f"{static_len_arr.mean():.0f}±{static_len_arr.std():.0f}    "
        f"{static_per_tok.mean():.3f}±{static_per_tok.std():.3f}"
    )
    print(
        f"{'Temporal PID':<20} "
        f"{pid_arr.mean():.3f}±{pid_arr.std():.3f}  "
        f"{pid_len_arr.mean():.0f}±{pid_len_arr.std():.0f}    "
        f"{pid_per_tok.mean():.3f}±{pid_per_tok.std():.3f}"
    )
    print(f"\nOverhead (per-token, apples-to-apples):")
    print(f"  Static SAS vs Baseline: {overhead_static:+.1f}%")
    print(f"  Temporal PID vs Baseline: {overhead_pid:+.1f}%")
    print(f"  PID vs Static (controller cost only): {overhead_pid_vs_static:+.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
