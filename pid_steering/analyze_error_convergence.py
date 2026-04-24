"""Steady-State Error Convergence Analysis.

Replicates PID paper Figure 3 for the symbolic music domain.
Plots the scalar error signal ⟨e(0), e(k)⟩ across 12 sublayers for
P, PI, and PID controllers to demonstrate:
- P controller: nonzero plateau (steady-state error persists)
- PI controller: crosses zero but with overshoot
- PID controller: settles near zero with reduced overshoot

This is the KEY theoretical validation figure for the paper.
"""

import argparse
import json
import logging
import pathlib
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from pid_steering.pid_controller import SpatialPIDController
from pid_steering.sequential_pid_calculator import (
    get_sublayer_modules,
    ActivationCapture,
    SteeringInjector,
)
from pid_steering.conditioned_pid_evaluator import (
    load_song_tokens,
    find_extreme_songs,
)
import config_pid

import representation
import utils

logger = logging.getLogger(__name__)


def compute_error_signals(
    model: nn.Module,
    source_sequences: List,
    target_sequences: List,
    device: torch.device,
    position: str = "last",
) -> Dict[int, torch.Tensor]:
    """Compute the raw difference-in-means error at each sublayer.

    This is the unsteered error: e(k) = mu_target(k) - mu_source(k),
    computed without any interventions.

    Returns:
        Dict mapping sublayer_idx -> error vector (dim,).
    """
    sublayer_modules = get_sublayer_modules(model)
    errors = {}

    for k in range(len(sublayer_modules)):
        _, target_module = sublayer_modules[k]

        # Capture source activations
        cap_src = ActivationCapture(position=position)
        cap_src.register(target_module)
        model.eval()
        with torch.no_grad():
            for seq in source_sequences:
                if isinstance(seq, np.ndarray):
                    seq = torch.from_numpy(seq).long()
                seq = seq.unsqueeze(0).to(device)
                model.decoder.net(seq)
        mu_source = cap_src.get_mean()
        cap_src.remove()
        cap_src.clear()

        # Capture target activations
        cap_tgt = ActivationCapture(position=position)
        cap_tgt.register(target_module)
        with torch.no_grad():
            for seq in target_sequences:
                if isinstance(seq, np.ndarray):
                    seq = torch.from_numpy(seq).long()
                seq = seq.unsqueeze(0).to(device)
                model.decoder.net(seq)
        mu_target = cap_tgt.get_mean()
        cap_tgt.remove()
        cap_tgt.clear()

        errors[k] = (mu_target - mu_source).cpu()

    return errors


def compute_steered_error_trajectory(
    model: nn.Module,
    source_sequences: List,
    target_sequences: List,
    Kp: float,
    Ki: float,
    Kd: float,
    max_I: float,
    alpha: float,
    device: torch.device,
    position: str = "last",
) -> Tuple[List[float], Dict[int, torch.Tensor]]:
    """Compute the error trajectory under a specific PID configuration.

    Applies sequential steering and measures how the error evolves
    across sublayers.

    Returns:
        (scalar_signals, errors): scalar_signals[k] = ⟨e(0), e(k)⟩,
        errors[k] = e(k) vector.
    """
    sublayer_modules = get_sublayer_modules(model)
    n_layers = len(sublayer_modules)

    controller = SpatialPIDController(
        Kp=Kp, Ki=Ki, Kd=Kd, max_I=max_I, dim=config_pid.MODEL_DIM
    )
    controller.to(device)
    controller.reset()

    errors = {}
    steering_hooks = []
    e_0 = None  # Will store e(0) for computing ⟨e(0), e(k)⟩

    for k in range(n_layers):
        _, target_module = sublayer_modules[k]

        # Capture source activations (with prior PID corrections)
        cap_src = ActivationCapture(position=position)
        cap_src.register(target_module)

        # Register all prior steering hooks
        active_handles = []
        for hook in steering_hooks:
            _, mod = sublayer_modules[hook._layer_idx]
            hook.register(mod)
            active_handles.append(hook)

        model.eval()
        with torch.no_grad():
            for seq in source_sequences:
                if isinstance(seq, np.ndarray):
                    seq = torch.from_numpy(seq).long()
                seq = seq.unsqueeze(0).to(device)
                model.decoder.net(seq)

        mu_source = cap_src.get_mean()
        cap_src.remove()
        cap_src.clear()

        # Remove steering hooks for cleanup
        for hook in active_handles:
            hook.remove()

        # Capture target activations (no steering)
        cap_tgt = ActivationCapture(position=position)
        cap_tgt.register(target_module)
        with torch.no_grad():
            for seq in target_sequences:
                if isinstance(seq, np.ndarray):
                    seq = torch.from_numpy(seq).long()
                seq = seq.unsqueeze(0).to(device)
                model.decoder.net(seq)
        mu_target = cap_tgt.get_mean()
        cap_tgt.remove()
        cap_tgt.clear()

        # Error at sublayer k
        e_k = mu_target - mu_source
        errors[k] = e_k.cpu()

        if k == 0:
            e_0 = e_k.clone()

        # PID control
        u_k = controller.compute(e_k)

        # Create steering hook for next iterations
        injector = SteeringInjector(u_k, alpha=alpha, position=position)
        injector._layer_idx = k
        steering_hooks.append(injector)

    # Compute scalar signals: ⟨e(0), e(k)⟩
    e_0_cpu = e_0.cpu()
    e_0_norm = e_0_cpu.norm()
    scalar_signals = []
    for k in range(n_layers):
        dot = torch.dot(e_0_cpu, errors[k])
        # Normalize by ||e(0)||^2 for interpretability
        if e_0_norm > 1e-8:
            scalar_signals.append((dot / (e_0_norm**2)).item())
        else:
            scalar_signals.append(0.0)

    return scalar_signals, errors


def run_error_analysis(
    model: nn.Module,
    source_sequences: List,
    target_sequences: List,
    Kp: float = 1.0,
    Ki: float = 0.3,
    Kd: float = 0.1,
    max_I: float = 5.0,
    alpha: float = 1.0,
    device: torch.device = None,
) -> Dict[str, List[float]]:
    """Run complete P vs PI vs PID error analysis.

    Returns:
        Dict with keys "p", "pi", "pid" mapping to scalar signal lists.
    """
    if device is None:
        device = next(model.parameters()).device

    results = {}

    # P-only (Ki=0, Kd=0)
    logger.info("Computing P-only error trajectory...")
    results["p"], _ = compute_steered_error_trajectory(
        model,
        source_sequences,
        target_sequences,
        Kp=Kp,
        Ki=0.0,
        Kd=0.0,
        max_I=max_I,
        alpha=alpha,
        device=device,
    )

    # PI (Kd=0)
    logger.info("Computing PI error trajectory...")
    results["pi"], _ = compute_steered_error_trajectory(
        model,
        source_sequences,
        target_sequences,
        Kp=Kp,
        Ki=Ki,
        Kd=0.0,
        max_I=max_I,
        alpha=alpha,
        device=device,
    )

    # PID (full)
    logger.info("Computing PID error trajectory...")
    results["pid"], _ = compute_steered_error_trajectory(
        model,
        source_sequences,
        target_sequences,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        max_I=max_I,
        alpha=alpha,
        device=device,
    )

    return results


def plot_error_convergence(
    results: Dict[str, List[float]],
    concept: str,
    output_path: pathlib.Path,
):
    """Plot the scalar error signal for P, PI, PID.

    Mirrors PID paper Figure 3.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    x = list(range(len(results["p"])))

    ax.plot(x, results["p"], "b-o", label="P (DiffMean)", linewidth=2, markersize=6)
    ax.plot(x, results["pi"], "r-s", label="PI", linewidth=2, markersize=6)
    ax.plot(x, results["pid"], "g-^", label="PID", linewidth=2, markersize=6)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Sublayer Index (k)", fontsize=12)
    ax.set_ylabel(r"$\langle e(0), e(k) \rangle / \|e(0)\|^2$", fontsize=12)
    ax.set_title(
        f"Error Convergence — {concept.replace('_', ' ').title()}", fontsize=14
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Steady-state error convergence analysis (P vs PI vs PID)"
    )
    parser.add_argument("--concept", type=str, default="average_pitch")
    parser.add_argument(
        "--Kp", type=float, default=config_pid.SPATIAL_PID_DEFAULTS["Kp"]
    )
    parser.add_argument(
        "--Ki", type=float, default=0.3,
        help="Integral gain for convergence plot (higher than experiment default "
             "to show PI dynamics clearly over 12 sublayers)",
    )
    parser.add_argument(
        "--Kd", type=float, default=0.1,
        help="Derivative gain for convergence plot",
    )
    parser.add_argument(
        "--max_I", type=float, default=config_pid.SPATIAL_PID_DEFAULTS["max_I"]
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--n_songs", type=int, default=20,
        help="Songs per contrastive set (more = smoother means)",
    )
    parser.add_argument(
        "--activations_dir",
        type=pathlib.Path,
        default=config_pid.ACTIVATIONS_DIR,
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_PLOTS_DIR,
    )
    parser.add_argument("--gpu", type=int, default=0)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    from pid_steering.pid_steered_generator import load_model

    model, encoding, train_args = load_model(
        config_pid.MODEL_CHECKPOINT,
        config_pid.TRAIN_ARGS_PATH,
        config_pid.ENCODING_PATH,
        device,
    )

    # Load contrastive sequences — try HDF5/JSON first, fall back to extreme songs
    source_seqs, target_seqs = None, None
    try:
        from pid_steering.sequential_pid_calculator import load_contrastive_sequences

        source_seqs, target_seqs = load_contrastive_sequences(
            args.activations_dir, args.concept
        )
        logger.info("Loaded contrastive sequences from activation pipeline")
    except Exception as e:
        logger.info(f"Activation pipeline data not available ({e}), "
                     f"falling back to find_extreme_songs")

    if source_seqs is None or target_seqs is None:
        notes_dir = config_pid.PROJECT_ROOT / "data" / "sod" / "processed" / "notes"
        low_songs, high_songs = find_extreme_songs(
            notes_dir, encoding, args.concept, n_songs=args.n_songs,
        )
        source_seqs = [load_song_tokens(fp, encoding) for fp, _ in low_songs]
        target_seqs = [load_song_tokens(fp, encoding) for fp, _ in high_songs]
        logger.info(f"Using {len(source_seqs)} source + {len(target_seqs)} target "
                     f"sequences from extreme songs")

    # Run analysis
    results = run_error_analysis(
        model,
        source_seqs,
        target_seqs,
        Kp=args.Kp,
        Ki=args.Ki,
        Kd=args.Kd,
        max_I=args.max_I,
        alpha=args.alpha,
        device=device,
    )

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gains_str = f"Kp{args.Kp}_Ki{args.Ki}_Kd{args.Kd}"
    results_path = args.output_dir / f"error_convergence_{args.concept}_{gains_str}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Plot
    plot_path = args.output_dir / f"error_convergence_{args.concept}_{gains_str}.png"
    plot_error_convergence(results, args.concept, plot_path)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Error Convergence Summary — {args.concept}")
    print(f"Gains: Kp={args.Kp}, Ki={args.Ki}, Kd={args.Kd}, α={args.alpha}")
    print("=" * 60)
    print(f"{'Layer':>6} {'P':>10} {'PI':>10} {'PID':>10}")
    print("-" * 40)
    for k in range(len(results["p"])):
        print(
            f"{k:6d} {results['p'][k]:10.4f} {results['pi'][k]:10.4f} "
            f"{results['pid'][k]:10.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
