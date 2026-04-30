"""Plot λ(t) trajectory figures for the PID workshop paper.

Generates:
  1. Single-concept λ(t) trajectory: PID's staircase λ vs static flat line
  2. Dual-concept λ(t) trajectories: both pitch and duration controllers
  3. Error convergence: e(t) → 0 over generation steps
  4. Feature activation trajectory vs setpoint

Usage:
    python pid_steering/plot_lambda_trajectory.py \
        --diagnostics_dir exp/sod/pid_steering/experiments/temporal_sas_comparison \
        --output_dir exp/sod/pid_steering/plots/lambda_trajectories

    python pid_steering/plot_lambda_trajectory.py \
        --mode dual \
        --diagnostics_dir exp/sod/pid_steering/experiments/dual_temporal_pid \
        --output_dir exp/sod/pid_steering/plots/lambda_trajectories
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------- Style ----------
COLORS = {
    "pid_lambda": "#2196F3",
    "static_lambda": "#FF5722",
    "pitch_lambda": "#2196F3",
    "duration_lambda": "#4CAF50",
    "error": "#9C27B0",
    "activation": "#FF9800",
    "setpoint": "#E91E63",
}
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    }
)


def load_diagnostics(path: pathlib.Path):
    with open(path) as f:
        return json.load(f)


def plot_single_concept_lambda(
    diag_list,
    concept: str,
    static_lambda: float,
    output_dir: pathlib.Path,
    n_show: int = 5,
):
    """Hero figure: PID staircase λ(t) vs static flat line."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

    concept_label = "Pitch" if "pitch" in concept else "Duration"

    # Left: individual trajectories
    ax = axes[0]
    for i, diag in enumerate(diag_list[:n_show]):
        traj = diag["lambda_trajectory"]
        steps = np.arange(len(traj))
        alpha = 0.4 if i > 0 else 0.9
        lw = 1.8 if i == 0 else 1.0
        ax.plot(
            steps,
            traj,
            color=COLORS["pid_lambda"],
            alpha=alpha,
            lw=lw,
            label="PID $\\lambda(t)$" if i == 0 else None,
        )

    ax.set_xlabel("Generation Step $t$")
    ax.set_ylabel("Steering Strength $\\lambda$")
    ax.set_title(f"{concept_label}: Individual Trajectories")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.1, None)
    ax.grid(alpha=0.2)

    # Right: mean ± std across all samples
    ax = axes[1]
    max_len = max(len(d["lambda_trajectory"]) for d in diag_list)
    padded = np.full((len(diag_list), max_len), np.nan)
    for i, d in enumerate(diag_list):
        t = d["lambda_trajectory"]
        padded[i, : len(t)] = t

    mean_traj = np.nanmean(padded, axis=0)
    std_traj = np.nanstd(padded, axis=0)
    steps = np.arange(max_len)

    ax.plot(
        steps,
        mean_traj,
        color=COLORS["pid_lambda"],
        lw=2,
        label="PID mean $\\lambda(t)$",
    )
    ax.fill_between(
        steps,
        mean_traj - std_traj,
        mean_traj + std_traj,
        color=COLORS["pid_lambda"],
        alpha=0.15,
        label="$\\pm 1\\sigma$",
    )
    ax.set_xlabel("Generation Step $t$")
    ax.set_title(f"{concept_label}: Mean $\\pm$ Std (n={len(diag_list)})")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.1, None)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out_png = output_dir / f"lambda_trajectory_{concept}.png"
    out_pdf = output_dir / f"lambda_trajectory_{concept}.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close()
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")


def plot_error_convergence(diag_list, concept: str, output_dir: pathlib.Path):
    """Error signal e(t) converging to zero — proves PID stabilizes."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    concept_label = "Pitch" if "pitch" in concept else "Duration"

    max_len = max(len(d["error_trajectory"]) for d in diag_list)
    padded = np.full((len(diag_list), max_len), np.nan)
    for i, d in enumerate(diag_list):
        t = d["error_trajectory"]
        padded[i, : len(t)] = t

    mean_err = np.nanmean(padded, axis=0)
    std_err = np.nanstd(padded, axis=0)
    steps = np.arange(max_len)

    ax.plot(steps, mean_err, color=COLORS["error"], lw=2, label="Mean error $e(t)$")
    ax.fill_between(
        steps, mean_err - std_err, mean_err + std_err, color=COLORS["error"], alpha=0.15
    )
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Generation Step $t$")
    ax.set_ylabel("Error $e(t) = r_{\\mathrm{set}} - \\bar{a}(t)$")
    ax.set_title(f"{concept_label}: PID Error Convergence")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = output_dir / f"error_convergence_{concept}.png"
    fig.savefig(out)
    fig.savefig(output_dir / f"error_convergence_{concept}.pdf")
    plt.close()
    print(f"  Saved: {out}")


def plot_activation_vs_setpoint(
    diag_list, concept: str, setpoint: float, output_dir: pathlib.Path
):
    """Feature activation trajectory vs setpoint — shows PID tracking."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    concept_label = "Pitch" if "pitch" in concept else "Duration"

    max_len = max(len(d["feature_activation_trajectory"]) for d in diag_list)
    padded = np.full((len(diag_list), max_len), np.nan)
    for i, d in enumerate(diag_list):
        t = d["feature_activation_trajectory"]
        padded[i, : len(t)] = t

    mean_act = np.nanmean(padded, axis=0)
    std_act = np.nanstd(padded, axis=0)
    steps = np.arange(max_len)

    ax.plot(
        steps,
        mean_act,
        color=COLORS["activation"],
        lw=2,
        label="Mean activation $\\bar{a}(t)$",
    )
    ax.fill_between(
        steps,
        mean_act - std_act,
        mean_act + std_act,
        color=COLORS["activation"],
        alpha=0.15,
    )
    ax.axhline(
        setpoint,
        color=COLORS["setpoint"],
        ls="--",
        lw=2,
        label=f"Setpoint $r_{{set}}={setpoint}$",
    )
    ax.set_xlabel("Generation Step $t$")
    ax.set_ylabel("Mean Target Feature Activation")
    ax.set_title(f"{concept_label}: Activation Tracking")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = output_dir / f"activation_tracking_{concept}.png"
    fig.savefig(out)
    fig.savefig(output_dir / f"activation_tracking_{concept}.pdf")
    plt.close()
    print(f"  Saved: {out}")


def plot_dual_lambda_trajectories(
    diag_list,
    output_dir: pathlib.Path,
    static_pitch: float = 0.75,
    static_dur: float = 0.75,
    n_show: int = 3,
):
    """Dual PID: both λ_pitch(t) and λ_dur(t) on the same axes."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    # Left: individual dual trajectories
    ax = axes[0]
    for i, diag in enumerate(diag_list[:n_show]):
        p_traj = diag["pitch_lambda_trajectory"]
        d_traj = diag["duration_lambda_trajectory"]
        steps = np.arange(len(p_traj))
        alpha = 0.8 if i == 0 else 0.35
        lw = 1.8 if i == 0 else 1.0
        ax.plot(
            steps,
            p_traj,
            color=COLORS["pitch_lambda"],
            alpha=alpha,
            lw=lw,
            label="$\\lambda_{\\mathrm{pitch}}(t)$" if i == 0 else None,
        )
        ax.plot(
            steps,
            d_traj,
            color=COLORS["duration_lambda"],
            alpha=alpha,
            lw=lw,
            ls="-",
            label="$\\lambda_{\\mathrm{dur}}(t)$" if i == 0 else None,
        )

    ax.axhline(static_pitch, color=COLORS["pitch_lambda"], ls=":", lw=1.5, alpha=0.5)
    ax.axhline(static_dur, color=COLORS["duration_lambda"], ls=":", lw=1.5, alpha=0.5)
    ax.set_xlabel("Generation Step $t$")
    ax.set_ylabel("Steering Strength $\\lambda$")
    ax.set_title("Dual PID: Individual Trajectories")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.1, None)
    ax.grid(alpha=0.2)

    # Right: mean ± std
    ax = axes[1]
    max_len = max(len(d["pitch_lambda_trajectory"]) for d in diag_list)

    for concept_key, color, label in [
        (
            "pitch_lambda_trajectory",
            COLORS["pitch_lambda"],
            "$\\lambda_{\\mathrm{pitch}}$",
        ),
        (
            "duration_lambda_trajectory",
            COLORS["duration_lambda"],
            "$\\lambda_{\\mathrm{dur}}$",
        ),
    ]:
        padded = np.full((len(diag_list), max_len), np.nan)
        for i, d in enumerate(diag_list):
            t = d[concept_key]
            padded[i, : len(t)] = t
        mean_t = np.nanmean(padded, axis=0)
        std_t = np.nanstd(padded, axis=0)
        steps = np.arange(max_len)
        ax.plot(steps, mean_t, color=color, lw=2, label=f"PID {label}")
        ax.fill_between(steps, mean_t - std_t, mean_t + std_t, color=color, alpha=0.12)

    ax.axhline(
        static_pitch,
        color=COLORS["pitch_lambda"],
        ls=":",
        lw=1.5,
        alpha=0.5,
        label=f"Static $\\lambda_p={static_pitch}$",
    )
    ax.axhline(
        static_dur,
        color=COLORS["duration_lambda"],
        ls=":",
        lw=1.5,
        alpha=0.5,
        label=f"Static $\\lambda_d={static_dur}$",
    )
    ax.set_xlabel("Generation Step $t$")
    ax.set_title(f"Dual PID: Mean $\\pm$ Std (n={len(diag_list)})")
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    ax.set_ylim(-0.1, None)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = output_dir / "dual_lambda_trajectories.png"
    fig.savefig(out)
    fig.savefig(output_dir / "dual_lambda_trajectories.pdf")
    plt.close()
    print(f"  Saved: {out}")


def plot_combined_hero_figure(
    diag_pitch, diag_dur, static_lambda: float, output_dir: pathlib.Path
):
    """Combined 2-panel hero figure for the paper: pitch + duration λ(t)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)

    for ax, diag_list, concept_label in [
        (axes[0], diag_pitch, "Pitch"),
        (axes[1], diag_dur, "Duration"),
    ]:
        max_len = max(len(d["lambda_trajectory"]) for d in diag_list)
        padded = np.full((len(diag_list), max_len), np.nan)
        for i, d in enumerate(diag_list):
            t = d["lambda_trajectory"]
            padded[i, : len(t)] = t

        mean_traj = np.nanmean(padded, axis=0)
        std_traj = np.nanstd(padded, axis=0)
        steps = np.arange(max_len)

        ax.plot(
            steps,
            mean_traj,
            color=COLORS["pid_lambda"],
            lw=2,
            label="PID $\\lambda(t)$",
        )
        ax.fill_between(
            steps,
            mean_traj - std_traj,
            mean_traj + std_traj,
            color=COLORS["pid_lambda"],
            alpha=0.15,
            label="$\\pm 1\\sigma$",
        )
        ax.axhline(
            static_lambda,
            color=COLORS["static_lambda"],
            ls="--",
            lw=2,
            label=f"Static $\\lambda={static_lambda}$",
        )
        ax.set_xlabel("Generation Step $t$")
        ax.set_title(concept_label)
        ax.legend(loc="lower right")
        ax.set_ylim(-0.1, None)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Steering Strength $\\lambda$")
    plt.tight_layout()
    out = output_dir / "hero_lambda_trajectories.png"
    fig.savefig(out)
    fig.savefig(output_dir / "hero_lambda_trajectories.pdf")
    plt.close()
    print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot PID λ(t) trajectories")
    parser.add_argument(
        "--mode",
        choices=["single", "dual", "both"],
        default="both",
        help="Which plots to generate",
    )
    parser.add_argument(
        "--diagnostics_dir",
        type=pathlib.Path,
        default=pathlib.Path(
            "exp/sod/pid_steering/experiments/temporal_sas_comparison"
        ),
    )
    parser.add_argument(
        "--dual_diagnostics_dir",
        type=pathlib.Path,
        default=pathlib.Path("exp/sod/pid_steering/experiments/dual_temporal_pid"),
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("exp/sod/pid_steering/plots/lambda_trajectories"),
    )
    parser.add_argument(
        "--static_lambda",
        type=float,
        default=3.0,
        help="Static λ for comparison line (single-concept)",
    )
    parser.add_argument("--static_lambda_pitch", type=float, default=0.75)
    parser.add_argument("--static_lambda_dur", type=float, default=0.75)
    parser.add_argument("--setpoint", type=float, default=1.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("single", "both"):
        print("=== Single-Concept λ(t) Trajectories ===")
        for concept in ["average_pitch", "average_duration"]:
            diag_path = args.diagnostics_dir / f"pid_diagnostics_{concept}.json"
            if not diag_path.exists():
                print(f"  SKIP: {diag_path} not found")
                continue

            diag_list = load_diagnostics(diag_path)
            print(
                f"  {concept}: {len(diag_list)} samples, "
                f"avg steps={np.mean([d['n_steps'] for d in diag_list]):.0f}"
            )

            plot_single_concept_lambda(
                diag_list, concept, args.static_lambda, args.output_dir
            )
            plot_error_convergence(diag_list, concept, args.output_dir)
            plot_activation_vs_setpoint(
                diag_list, concept, args.setpoint, args.output_dir
            )

        # Hero figure: both concepts side by side
        pitch_path = args.diagnostics_dir / "pid_diagnostics_average_pitch.json"
        dur_path = args.diagnostics_dir / "pid_diagnostics_average_duration.json"
        if pitch_path.exists() and dur_path.exists():
            print("  Generating hero figure...")
            plot_combined_hero_figure(
                load_diagnostics(pitch_path),
                load_diagnostics(dur_path),
                args.static_lambda,
                args.output_dir,
            )

    if args.mode in ("dual", "both"):
        print("\n=== Dual-Concept λ(t) Trajectories ===")
        # Try unconditioned first
        for fname in [
            "dual_pid_diagnostics.json",
            "unconditioned/dual_pid_diagnostics.json",
        ]:
            dual_path = args.dual_diagnostics_dir / fname
            if dual_path.exists():
                diag_list = load_diagnostics(dual_path)
                print(f"  Dual unconditioned: {len(diag_list)} samples")

                # Correlation between pitch and duration λ trajectories
                correlations = []
                for d in diag_list:
                    p = np.array(d["pitch_lambda_trajectory"])
                    dur = np.array(d["duration_lambda_trajectory"])
                    min_len = min(len(p), len(dur))
                    if min_len > 2:
                        r = np.corrcoef(p[:min_len], dur[:min_len])[0, 1]
                        correlations.append(r)
                if correlations:
                    print(
                        f"  λ_pitch ↔ λ_dur correlation: r={np.mean(correlations):.3f} ± {np.std(correlations):.3f}"
                    )

                plot_dual_lambda_trajectories(
                    diag_list,
                    args.output_dir,
                    args.static_lambda_pitch,
                    args.static_lambda_dur,
                )
                break
        else:
            print(f"  SKIP: no dual diagnostics found in {args.dual_diagnostics_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
