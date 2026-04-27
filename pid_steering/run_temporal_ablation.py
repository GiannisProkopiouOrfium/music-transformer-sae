"""PID Component Ablation for Temporal SAS Steering.

Compares P-only, PI, and full PID controllers to quantify the
contribution of each component:
- P-only: proportional only (no memory, no damping)
- PI: proportional + integral (overcomes threshold but no damping)
- PID: full controller (overcomes threshold + smooth response)

Produces:
- Metrics table per ablation
- Lambda trajectory comparison plot
- Feature activation comparison plot
- JSON results
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
)
from pid_steering.pid_steered_generator import load_model
from pid_steering.run_temporal_pid_sas import generate_static_smooth
from pid_steering.run_single_attribute import compute_generation_metrics

import config_pid

logger = logging.getLogger(__name__)

# Ablation configurations: (label, Kp, Ki, Kd)
ABLATION_CONFIGS = {
    "P_only": {"Kp": 1.0, "Ki": 0.0, "Kd": 0.0},
    "PI": {"Kp": 1.0, "Ki": 0.05, "Kd": 0.0},
    "PID": {"Kp": 1.0, "Ki": 0.05, "Kd": 0.01},
}


def main():
    parser = argparse.ArgumentParser(
        description="PID component ablation for temporal SAS"
    )
    parser.add_argument(
        "--concept",
        type=str,
        nargs="+",
        default=["average_pitch", "average_duration"],
    )
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--seq_len", type=int, default=config_pid.MAX_SEQ_LEN)
    parser.add_argument("--target_layer", type=int, default=config_pid.SAS_LAYER)
    parser.add_argument("--target_magnitude", type=float, default=1.0)
    parser.add_argument("--lambda_max", type=float, default=3.0)
    parser.add_argument("--n_ramp_steps", type=int, default=64)
    parser.add_argument(
        "--max_I", type=float, default=config_pid.TEMPORAL_PID_DEFAULTS["max_I"]
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=config_pid.PID_EXPERIMENTS_DIR / "temporal_ablation",
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

    concepts = args.concept
    if concepts == ["all"]:
        concepts = ["average_pitch", "average_duration"]

    for concept in concepts:
        logger.info(f"\n{'='*70}")
        logger.info(f"PID Component Ablation — {concept}")
        logger.info(f"{'='*70}")

        vector_path = config_pid.SAS_VECTORS_DIR / f"{concept}_sas_vectors.pt"
        sas_vector = load_sas_vector(vector_path, layer_idx=args.target_layer)

        pid_gen = TemporalPIDSASGenerator(
            model,
            encoding,
            sae_model,
            sas_vector,
            args.target_layer,
        )

        ablation_results = {}
        all_diagnostics = {}

        # Run each ablation config
        for config_name, gains in ABLATION_CONFIGS.items():
            logger.info(
                f"\n--- {config_name}: Kp={gains['Kp']}, Ki={gains['Ki']}, Kd={gains['Kd']} ---"
            )

            sequences, diagnostics = pid_gen.generate_with_temporal_pid(
                n_samples=args.n_samples,
                seq_len=args.seq_len,
                target_magnitude=args.target_magnitude,
                Kp=gains["Kp"],
                Ki=gains["Ki"],
                Kd=gains["Kd"],
                max_I=args.max_I,
                lambda_max=args.lambda_max,
                n_ramp_beats=args.n_ramp_steps,
                device=device,
            )

            metrics = [compute_generation_metrics(s, encoding) for s in sequences]
            avg_metrics = {}
            for key in metrics[0]:
                values = [m[key] for m in metrics]
                avg_metrics[f"{key}_mean"] = float(np.nanmean(values))
                avg_metrics[f"{key}_std"] = float(np.nanstd(values))

            avg_lambda = float(
                np.mean(
                    [
                        np.mean(d["lambda_trajectory"])
                        for d in diagnostics
                        if d["lambda_trajectory"]
                    ]
                )
            )
            avg_final_error = float(
                np.mean(
                    [
                        d["error_trajectory"][-1]
                        for d in diagnostics
                        if d["error_trajectory"]
                    ]
                )
            )
            lambda_std = float(
                np.mean(
                    [
                        np.std(d["lambda_trajectory"])
                        for d in diagnostics
                        if d["lambda_trajectory"]
                    ]
                )
            )
            avg_feat_act = float(
                np.mean(
                    [
                        np.mean(d["feature_activation_trajectory"])
                        for d in diagnostics
                        if d["feature_activation_trajectory"]
                    ]
                )
            )

            ablation_results[config_name] = {
                "gains": gains,
                "metrics": avg_metrics,
                "avg_lambda": avg_lambda,
                "lambda_std": lambda_std,
                "avg_final_error": avg_final_error,
                "avg_feature_activation": avg_feat_act,
            }
            all_diagnostics[config_name] = diagnostics

            logger.info(
                f"  avg_lambda={avg_lambda:.3f} (±{lambda_std:.3f}), "
                f"final_error={avg_final_error:.4f}, "
                f"feat_act={avg_feat_act:.4f}"
            )

        # Also run baseline (unsteered) for reference
        logger.info("\n--- Baseline (unsteered) ---")
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
        baseline_metrics = [
            compute_generation_metrics(s, encoding) for s in baseline_seqs
        ]
        baseline_avg = {}
        for key in baseline_metrics[0]:
            values = [m[key] for m in baseline_metrics]
            baseline_avg[f"{key}_mean"] = float(np.nanmean(values))
            baseline_avg[f"{key}_std"] = float(np.nanstd(values))
        ablation_results["baseline"] = {"metrics": baseline_avg}

        # Save results
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results_path = args.output_dir / f"ablation_{concept}.json"
        with open(results_path, "w") as f:
            json.dump(ablation_results, f, indent=2, default=float)
        logger.info(f"Saved results to {results_path}")

        # Save diagnostics
        diag_path = args.output_dir / f"ablation_diagnostics_{concept}.json"
        with open(diag_path, "w") as f:
            json.dump(
                all_diagnostics,
                f,
                indent=2,
                default=float,
            )

        # Plot comparison
        _plot_ablation(all_diagnostics, ablation_results, concept, args.output_dir)

        # Print summary table
        print(f"\n{'='*80}")
        print(f"PID Component Ablation — {concept}")
        print(f"{'='*80}")
        print(
            f"{'Config':<10} {'Avg λ':>8} {'λ std':>8} {'Final Err':>10} "
            f"{'Feat Act':>10} {'Avg Pitch':>10} {'Entropy':>10} "
            f"{'Scale%':>8} {'Groove%':>9}"
        )
        print("-" * 95)
        for config_name, data in ablation_results.items():
            m = data["metrics"]
            if config_name == "baseline":
                print(
                    f"{config_name:<10} {'—':>8} {'—':>8} {'—':>10} {'—':>10} "
                    f"{m.get('average_pitch_mean', 0):>10.2f} "
                    f"{m.get('pitch_class_entropy_mean', 0):>10.3f} "
                    f"{m.get('scale_consistency_mean', 0):>8.1f} "
                    f"{m.get('groove_consistency_mean', 0):>9.1f}"
                )
            else:
                print(
                    f"{config_name:<10} "
                    f"{data['avg_lambda']:>8.3f} "
                    f"{data['lambda_std']:>8.3f} "
                    f"{data['avg_final_error']:>10.4f} "
                    f"{data['avg_feature_activation']:>10.4f} "
                    f"{m.get('average_pitch_mean', 0):>10.2f} "
                    f"{m.get('pitch_class_entropy_mean', 0):>10.3f} "
                    f"{m.get('scale_consistency_mean', 0):>8.1f} "
                    f"{m.get('groove_consistency_mean', 0):>9.1f}"
                )
        print(f"{'='*80}")


def _plot_ablation(all_diagnostics, _ablation_results, concept, output_dir):
    """Plot lambda trajectories and feature activations for each ablation."""
    try:
        import matplotlib.pyplot as plt

        configs = [c for c in ABLATION_CONFIGS if c in all_diagnostics]
        colors = {"P_only": "#e74c3c", "PI": "#f39c12", "PID": "#2ecc71"}
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Lambda trajectories (mean ± std across samples)
        ax1 = axes[0]
        for config_name in configs:
            diags = all_diagnostics[config_name]
            # Compute mean trajectory (pad to max length)
            max_len = max(len(d["lambda_trajectory"]) for d in diags)
            padded = np.full((len(diags), max_len), np.nan)
            for i, d in enumerate(diags):
                traj = d["lambda_trajectory"]
                padded[i, : len(traj)] = traj
            mean_traj = np.nanmean(padded, axis=0)
            std_traj = np.nanstd(padded, axis=0)

            ax1.plot(
                mean_traj, label=config_name, color=colors[config_name], linewidth=2
            )
            ax1.fill_between(
                range(len(mean_traj)),
                mean_traj - std_traj,
                mean_traj + std_traj,
                alpha=0.15,
                color=colors[config_name],
            )

        ax1.set_ylabel(r"$\lambda_t$", fontsize=13)
        ax1.set_title(
            f"PID Component Ablation: λ Trajectories — {concept}", fontsize=14
        )
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Feature activation trajectories
        ax2 = axes[1]
        for config_name in configs:
            diags = all_diagnostics[config_name]
            max_len = max(len(d["feature_activation_trajectory"]) for d in diags)
            padded = np.full((len(diags), max_len), np.nan)
            for i, d in enumerate(diags):
                traj = d["feature_activation_trajectory"]
                padded[i, : len(traj)] = traj
            mean_traj = np.nanmean(padded, axis=0)
            std_traj = np.nanstd(padded, axis=0)

            ax2.plot(
                mean_traj, label=config_name, color=colors[config_name], linewidth=2
            )
            ax2.fill_between(
                range(len(mean_traj)),
                mean_traj - std_traj,
                mean_traj + std_traj,
                alpha=0.15,
                color=colors[config_name],
            )

        ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="target")
        ax2.set_xlabel("Generation Step", fontsize=13)
        ax2.set_ylabel("Target Feature Activation", fontsize=13)
        ax2.set_title("Feature Activation by Controller Type", fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = output_dir / f"ablation_trajectories_{concept}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved ablation plot to {plot_path}")

    except ImportError:
        logger.warning("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
