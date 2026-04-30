"""Temporal PID for SAS: Static Smooth vs PID Smooth Comparison.

Generates the key plot showing the Top-K threshold failure problem:
- Static smooth (cosine ramp): lambda ramps 0→λ, but features vanish
  below Top-K threshold during the ramp → output identical to unsteered
- PID smooth: PID controller pushes lambda past threshold via integral
  accumulation → features survive Top-K → smooth steering actually works

Produces:
- Figure: lambda_t trajectory (PID vs static) over generation steps
- Figure: target feature activation over time (PID vs static)
- Table: metrics comparison
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
from pid_steering.run_single_attribute import compute_generation_metrics

import config_pid

logger = logging.getLogger(__name__)


def generate_static_smooth(
    model,
    encoding,
    sae_model,
    sas_vector,
    target_layer,
    lambda_max: float,
    n_ramp_steps: int,
    n_samples: int,
    seq_len: int,
    device: torch.device,
    target_feature_indices: np.ndarray = None,
) -> Tuple[List[np.ndarray], List[List[float]]]:
    """Generate with static cosine smooth steering (existing method).

    When lambda_max=0, generates unsteered baseline.
    When lambda_max>0, demonstrates the failure mode: during the ramp,
    features fall below Top-K and are zeroed out.

    Returns:
        (sequences, feature_trajectories): generated sequences and
        per-sample feature activation trajectories for comparison plots.
    """
    sequences = []
    feature_trajectories = []  # per-sample list of activation values per step

    sos = encoding["type_code_map"]["start-of-song"]
    eos = encoding["type_code_map"]["end-of-song"]

    handles = []
    _smooth_hook = None
    _activation_log = []  # shared mutable list for the observer

    if lambda_max > 0 and n_ramp_steps > 0:
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

        # Observer: measure target feature activation AFTER steering
        def _observe_hook(module, input, output, layer_idx):
            """Measure feature activation after the steering hook fires."""
            if layer_idx != target_layer or target_feature_indices is None:
                return
            actual = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                a_flat = actual.reshape(-1, actual.shape[-1])
                f_a = sae_model.encode(a_flat)
                act = f_a[:, target_feature_indices].mean().item()
                _activation_log.append(act)

        layers = model.decoder.net.attn_layers.layers
        layer_module = layers[target_layer]
        if isinstance(layer_module, torch.nn.ModuleList) and len(layer_module) > 1:
            target_module = layer_module[1]
        else:
            target_module = layer_module

        # Register steering hook first, then observer
        h1 = target_module.register_forward_hook(
            lambda mod, inp, out, idx=target_layer: _smooth_hook(mod, inp, out, idx)
        )
        h2 = target_module.register_forward_hook(
            lambda mod, inp, out, idx=target_layer: _observe_hook(mod, inp, out, idx)
        )
        handles.extend([h1, h2])

    try:
        for i in range(n_samples):
            if _smooth_hook is not None:
                _smooth_hook.reset()
            _activation_log.clear()

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
            feature_trajectories.append(list(_activation_log))
    finally:
        for h in handles:
            h.remove()

    return sequences, feature_trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Compare static smooth vs temporal PID for SAS"
    )
    parser.add_argument(
        "--concept",
        type=str,
        nargs="+",
        default=["average_pitch"],
        help="Concept(s) to run. Use 'all' for both pitch and duration.",
    )
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
        default=config_pid.PID_EXPERIMENTS_DIR / "temporal_sas_comparison",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--direction",
        type=str,
        choices=["positive", "negative"],
        default="positive",
        help="Steering direction. 'negative' negates the SAS vector.",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Resolve concept list
    concepts = args.concept
    if concepts == ["all"]:
        concepts = ["average_pitch", "average_duration"]

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model (shared across concepts)
    model, encoding, _ = load_model(
        config_pid.MODEL_CHECKPOINT,
        config_pid.TRAIN_ARGS_PATH,
        config_pid.ENCODING_PATH,
        device,
    )

    # Load SAE (shared across concepts — same layer)
    k = get_adaptive_k(args.target_layer)
    sae_path = config_pid.SAE_CHECKPOINT_DIR / f"sae_layer_{args.target_layer}_best.pt"
    sae_model = load_sae_model(sae_path, k, device)

    for concept in concepts:
        logger.info(f"\n{'='*70}")
        logger.info(f"Running temporal PID SAS comparison for: {concept}")
        logger.info(f"{'='*70}")

        vector_path = config_pid.SAS_VECTORS_DIR / f"{concept}_sas_vectors.pt"
        sas_vector = load_sas_vector(vector_path, layer_idx=args.target_layer)

        if args.direction == "negative":
            sas_vector = -sas_vector
            logger.info("Using NEGATED SAS vector (steer DOWN)")

        _run_concept(
            model=model,
            encoding=encoding,
            sae_model=sae_model,
            sas_vector=sas_vector,
            concept=concept,
            args=args,
            device=device,
        )


def _run_concept(model, encoding, sae_model, sas_vector, concept, args, device):
    """Run the temporal PID SAS comparison for a single concept."""

    results = {}

    # 1. Temporal PID
    logger.info("Generating with Temporal PID SAS...")
    pid_generator = TemporalPIDSASGenerator(
        model,
        encoding,
        sae_model,
        sas_vector,
        args.target_layer,
    )
    pid_sequences, pid_diagnostics = pid_generator.generate_with_temporal_pid(
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

    pid_metrics = [compute_generation_metrics(s, encoding) for s in pid_sequences]
    pid_avg = {}
    for key in pid_metrics[0]:
        values = [m[key] for m in pid_metrics]
        pid_avg[f"{key}_mean"] = float(np.nanmean(values))
        pid_avg[f"{key}_std"] = float(np.nanstd(values))

    results["temporal_pid"] = {
        "metrics": pid_avg,
        "per_sample_metrics": pid_metrics,  # per-sample for CI computation
        "avg_lambda": float(
            np.mean(
                [
                    np.mean(d["lambda_trajectory"])
                    for d in pid_diagnostics
                    if d["lambda_trajectory"]
                ]
            )
        ),
        "avg_final_error": float(
            np.mean(
                [
                    d["error_trajectory"][-1]
                    for d in pid_diagnostics
                    if d["error_trajectory"]
                ]
            )
        ),
    }

    # 2. Static smooth steering (demonstrates failure mode)
    logger.info("Generating with static smooth SAS (failure demo)...")
    static_seqs, static_feat_trajs = generate_static_smooth(
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
        target_feature_indices=pid_generator.target_feature_indices,
    )
    static_metrics = [compute_generation_metrics(s, encoding) for s in static_seqs]
    static_avg = {}
    for key in static_metrics[0]:
        values = [m[key] for m in static_metrics]
        static_avg[f"{key}_mean"] = float(np.nanmean(values))
        static_avg[f"{key}_std"] = float(np.nanstd(values))
    results["static_smooth"] = {
        "metrics": static_avg,
        "per_sample_metrics": static_metrics,
    }

    # 3. Unsteered baseline
    logger.info("Generating unsteered baseline...")
    baseline_seqs, _ = generate_static_smooth(
        model,
        encoding,
        sae_model,
        sas_vector,
        args.target_layer,
        lambda_max=0.0,
        n_ramp_steps=0,
        n_samples=args.n_samples,
        seq_len=args.seq_len,
        device=device,
    )
    baseline_metrics = [compute_generation_metrics(s, encoding) for s in baseline_seqs]
    baseline_avg = {}
    for key in baseline_metrics[0]:
        values = [m[key] for m in baseline_metrics]
        baseline_avg[f"{key}_mean"] = float(np.nanmean(values))
        baseline_avg[f"{key}_std"] = float(np.nanstd(values))
    results["baseline"] = {
        "metrics": baseline_avg,
        "per_sample_metrics": baseline_metrics,
    }

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / f"temporal_comparison_{concept}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save diagnostics for plotting
    diag_path = args.output_dir / f"pid_diagnostics_{concept}.json"
    with open(diag_path, "w") as f:
        json.dump([d for d in pid_diagnostics], f, indent=2, default=float)

    # Plot lambda trajectories
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

        # Plot lambda trajectories for first 5 samples
        ax1 = axes[0]
        for i, diag in enumerate(pid_diagnostics[:5]):
            ax1.plot(diag["lambda_trajectory"], alpha=0.7, label=f"PID Sample {i}")
        ax1.set_ylabel(r"$\lambda_t$ (PID-computed)", fontsize=12)
        ax1.set_title(f"Temporal PID λ Trajectories — {concept}", fontsize=14)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Plot feature activation: PID vs Static Smooth comparison
        ax2 = axes[1]
        # PID feature activations (solid lines)
        for i, diag in enumerate(pid_diagnostics[:5]):
            ax2.plot(
                diag["feature_activation_trajectory"],
                alpha=0.7,
                label=f"PID Sample {i}",
                linestyle="-",
            )
        # Static smooth feature activations (dashed lines)
        for i, traj in enumerate(static_feat_trajs[:5]):
            if traj:
                ax2.plot(
                    traj,
                    alpha=0.5,
                    label=f"Static Sample {i}",
                    linestyle="--",
                    color=f"C{i}",
                )
        ax2.set_xlabel("Generation Step", fontsize=12)
        ax2.set_ylabel("Target Feature Activation", fontsize=12)
        ax2.set_title(
            "Target Feature Activation: PID (solid) vs Static Smooth (dashed)",
            fontsize=14,
        )
        ax2.legend(fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = args.output_dir / f"temporal_pid_trajectories_{concept}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved trajectory plot to {plot_path}")
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")

    # Print summary
    print("\n" + "=" * 70)
    print(f"Temporal PID SAS Comparison — {concept}")
    print("=" * 70)
    print(
        f"{'Method':<18} {'Avg Pitch':<12} {'PC Entropy':<12} "
        f"{'Scale Cons':<12} {'Groove Cons':<12}"
    )
    print("-" * 65)
    for method, data in results.items():
        m = data["metrics"]
        print(
            f"{method:<18} "
            f"{m.get('average_pitch_mean', 0):<12.2f} "
            f"{m.get('pitch_class_entropy_mean', 0):<12.3f} "
            f"{m.get('scale_consistency_mean', 0):<12.1f} "
            f"{m.get('groove_consistency_mean', 0):<12.1f}"
        )
    print("=" * 70)
    if "temporal_pid" in results:
        print(f"\nPID avg lambda: {results['temporal_pid']['avg_lambda']:.3f}")
        print(f"PID avg final error: {results['temporal_pid']['avg_final_error']:.4f}")


if __name__ == "__main__":
    main()
