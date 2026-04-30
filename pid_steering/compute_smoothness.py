"""Compute λ-trajectory and feature-activation smoothness from PID diagnostics.

Addresses reviewer Q7: quantitative smoothness metric for PID vs static.

Usage (on EC2):
    python pid_steering/compute_smoothness.py \
        --diag_dir exp/sod/pid_steering/experiments/temporal_sas_comparison

Reads pid_diagnostics_*.json and computes:
  - λ-smoothness: std(Δλ) and max|Δλ| across generation steps
  - Feature activation smoothness: std(Δf_a) for concept fingerprint
  - Transition sharpness: max single-step jump in feature activations
"""

import argparse
import json
import pathlib
import numpy as np


def compute_lambda_smoothness(trajectories):
    """Compute smoothness metrics from λ(t) trajectories."""
    results = []
    for traj in trajectories:
        lam = np.array(traj)
        if len(lam) < 3:
            continue
        dlam = np.diff(lam)
        d2lam = np.diff(dlam)
        results.append(
            {
                "std_delta_lambda": float(np.std(dlam)),
                "max_abs_delta_lambda": float(np.max(np.abs(dlam))),
                "mean_abs_d2_lambda": float(np.mean(np.abs(d2lam))),
                "n_steps": len(lam),
            }
        )
    return results


def compute_feature_smoothness(trajectories):
    """Compute smoothness of feature activation trajectories."""
    results = []
    for traj in trajectories:
        fa = np.array(traj)
        if len(fa) < 3:
            continue
        dfa = np.diff(fa)
        results.append(
            {
                "std_delta_fa": float(np.std(dfa)),
                "max_abs_delta_fa": float(np.max(np.abs(dfa))),
                "mean_abs_delta_fa": float(np.mean(np.abs(dfa))),
                "n_steps": len(fa),
            }
        )
    return results


def static_cosine_smoothness(n_steps, ramp_frac=0.5, lambda_target=3.0):
    """Compute smoothness for a static cosine ramp (baseline comparison).

    Under static cosine ramp with Top-K threshold, feature activations are
    binary: 0 during ramp phase (λ too small), then jump to full value.
    The λ trajectory itself is smooth (cosine), but the *effective* steering
    (feature activations) is a step function.
    """
    t = np.arange(n_steps)
    ramp_end = int(n_steps * ramp_frac)
    # Cosine ramp for λ
    lam = np.where(
        t < ramp_end,
        lambda_target / 2 * (1 - np.cos(np.pi * t / ramp_end)),
        lambda_target,
    )
    dlam = np.diff(lam)
    return {
        "std_delta_lambda": float(np.std(dlam)),
        "max_abs_delta_lambda": float(np.max(np.abs(dlam))),
        "note": "λ is smooth but feature activations are binary (0 then jump)",
    }


def main():
    parser = argparse.ArgumentParser(description="Compute smoothness metrics")
    parser.add_argument(
        "--diag_dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing pid_diagnostics_*.json files",
    )
    args = parser.parse_args()

    for concept in ["average_pitch", "average_duration"]:
        diag_path = args.diag_dir / f"pid_diagnostics_{concept}.json"
        if not diag_path.exists():
            print(f"  [skip] {diag_path} not found")
            continue

        with open(diag_path) as f:
            diagnostics = json.load(f)

        n_samples = len(diagnostics)
        print(f"\n{'='*60}")
        print(f"  {concept} (n={n_samples})")
        print(f"{'='*60}")

        # λ smoothness
        lam_trajs = [
            d["lambda_trajectory"] for d in diagnostics if d.get("lambda_trajectory")
        ]
        lam_results = compute_lambda_smoothness(lam_trajs)
        if lam_results:
            mean_std_dl = np.mean([r["std_delta_lambda"] for r in lam_results])
            mean_max_dl = np.mean([r["max_abs_delta_lambda"] for r in lam_results])
            mean_d2 = np.mean([r["mean_abs_d2_lambda"] for r in lam_results])
            avg_steps = np.mean([r["n_steps"] for r in lam_results])
            print(
                f"\n  PID λ-smoothness ({len(lam_results)} samples, avg {avg_steps:.0f} steps):"
            )
            print(f"    std(Δλ)          = {mean_std_dl:.4f}")
            print(f"    max|Δλ|          = {mean_max_dl:.4f}")
            print(f"    mean|Δ²λ|        = {mean_d2:.4f}")

        # Feature activation smoothness
        fa_trajs = [
            d["feature_activation_trajectory"]
            for d in diagnostics
            if d.get("feature_activation_trajectory")
        ]
        fa_results = compute_feature_smoothness(fa_trajs)
        if fa_results:
            mean_std_dfa = np.mean([r["std_delta_fa"] for r in fa_results])
            mean_max_dfa = np.mean([r["max_abs_delta_fa"] for r in fa_results])
            mean_abs_dfa = np.mean([r["mean_abs_delta_fa"] for r in fa_results])
            print(f"\n  PID feature activation smoothness ({len(fa_results)} samples):")
            print(f"    std(Δf_a)        = {mean_std_dfa:.4f}")
            print(f"    max|Δf_a|        = {mean_max_dfa:.4f}")
            print(f"    mean|Δf_a|       = {mean_abs_dfa:.4f}")

        # Static cosine baseline (for comparison)
        if lam_results:
            static = static_cosine_smoothness(int(avg_steps), lambda_target=3.0)
            print(f"\n  Static cosine ramp baseline ({int(avg_steps)} steps, λ=3.0):")
            print(f"    std(Δλ)          = {static['std_delta_lambda']:.4f}")
            print(f"    max|Δλ|          = {static['max_abs_delta_lambda']:.4f}")
            print(f"    Note: {static['note']}")


if __name__ == "__main__":
    main()
