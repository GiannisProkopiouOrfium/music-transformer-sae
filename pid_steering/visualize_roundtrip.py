"""Visualize and showcase round-trip reversibility.

Creates:
1. Per-sample trajectory plots (baseline vs roundtrip, highlighting Phase 3 overlap)
2. Recovery ratio summary figure
3. Phase-3-only trimmed WAVs for A/B comparison

Usage:
    python pid_steering/visualize_roundtrip.py \
        --results_dir exp/sod/pid_steering/experiments/roundtrip_v7 \
        --wavs_dir roundtrip_wavs \
        --output_dir paper/figures/roundtrip
"""

import argparse
import json
import logging
import math
import pathlib
import sys
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Phase schedule from v7
PHASE_TOKENS = [96, 64, 192]  # steer away, hold, steer back
TOTAL_TOKENS = sum(PHASE_TOKENS)


def load_per_sample_data(results_dir: pathlib.Path, concept: str, scenario: str):
    """Load per-sample results JSON."""
    path = results_dir / concept / f"per_sample_{scenario}.json"
    if not path.exists():
        logger.warning(f"Not found: {path}")
        return []
    with open(path) as f:
        return json.load(f)


def compute_recovery_metrics(samples: List[Dict], concept: str):
    """Compute recovery ratio for each sample.

    Recovery % = (1 - remaining_error / peak_deviation) × 100
    where:
      peak_deviation = |W2_RT - W2_BL|
      remaining_error = |W3_RT - W3_BL|
    """
    results = []
    for s in samples:
        rt = s.get("roundtrip", {})
        bl = s.get("baseline", {})
        st = s.get("static_oneway", {})
        rel = s.get("release", {})

        rt_w = rt.get("windows", [])
        bl_w = bl.get("windows", [])
        st_w = st.get("windows", [])
        rel_w = rel.get("windows", []) if rel else []

        if len(rt_w) < 3 or len(bl_w) < 3:
            continue

        # Extract per-window values
        rt_vals = [w.get(concept, float("nan")) for w in rt_w[:3]]
        bl_vals = [w.get(concept, float("nan")) for w in bl_w[:3]]
        st_vals = (
            [w.get(concept, float("nan")) for w in st_w[:3]]
            if len(st_w) >= 3
            else [float("nan")] * 3
        )
        rel_vals = (
            [w.get(concept, float("nan")) for w in rel_w[:3]]
            if len(rel_w) >= 3
            else [float("nan")] * 3
        )

        if any(math.isnan(v) for v in rt_vals + bl_vals):
            continue

        peak_deviation = abs(rt_vals[1] - bl_vals[1])
        remaining_error = abs(rt_vals[2] - bl_vals[2])

        if peak_deviation < 0.01:
            continue

        recovery_pct = (1.0 - remaining_error / peak_deviation) * 100.0

        results.append(
            {
                "sample_id": s.get("sample_id", ""),
                "rt_windows": rt_vals,
                "bl_windows": bl_vals,
                "st_windows": st_vals,
                "rel_windows": rel_vals,
                "peak_deviation": peak_deviation,
                "remaining_error": remaining_error,
                "recovery_pct": recovery_pct,
            }
        )

    results.sort(key=lambda x: x["recovery_pct"], reverse=True)
    return results


def plot_trajectory(
    sample: Dict, concept: str, scenario: str, output_path: pathlib.Path
):
    """Plot a single sample's trajectory: baseline vs roundtrip."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.error("matplotlib required. pip install matplotlib")
        return

    rt = sample["rt_windows"]
    bl = sample["bl_windows"]
    st = sample["st_windows"]

    # X positions: center of each phase window
    x_centers = [0.5, 1.5, 2.5]
    phase_labels = ["Phase 1\n(Steer Away)", "Phase 2\n(Hold)", "Phase 3\n(Steer Back)"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Phase background shading
    ax.axvspan(-0.05, 1.0, alpha=0.08, color="blue", label="_")
    ax.axvspan(1.0, 2.0, alpha=0.08, color="orange", label="_")
    ax.axvspan(2.0, 3.05, alpha=0.08, color="green", label="_")

    # Plot lines
    ax.plot(
        x_centers,
        bl,
        "o--",
        color="#7f7f7f",
        linewidth=2,
        markersize=8,
        label="Baseline (no steering)",
        zorder=3,
    )
    ax.plot(
        x_centers,
        rt,
        "o-",
        color="#1f77b4",
        linewidth=2.5,
        markersize=10,
        label="Round-trip PID",
        zorder=4,
    )
    ax.plot(
        x_centers,
        st,
        "o:",
        color="#d62728",
        linewidth=1.5,
        markersize=6,
        label="Static one-way",
        zorder=2,
        alpha=0.7,
    )

    # Highlight Phase 3 gap (remaining error)
    ax.annotate(
        "",
        xy=(2.55, bl[2]),
        xytext=(2.55, rt[2]),
        arrowprops=dict(arrowstyle="<->", color="#1f77b4", lw=1.5),
    )
    ax.text(
        2.65,
        (bl[2] + rt[2]) / 2,
        f"Error: {sample['remaining_error']:.1f}",
        fontsize=9,
        color="#1f77b4",
        va="center",
    )

    # Recovery annotation
    ax.text(
        0.98,
        0.02,
        f"Recovery: {sample['recovery_pct']:.0f}%",
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1f77b4", alpha=0.15),
    )

    attr_label = "Pitch (semitones)" if "pitch" in concept else "Duration (ticks)"
    ax.set_ylabel(attr_label, fontsize=11)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(phase_labels, fontsize=10)
    ax.set_xlim(-0.1, 3.1)
    ax.legend(loc="upper left" if "low" in scenario else "lower left", fontsize=9)
    ax.set_title(
        f"{scenario.replace('_', ' ').title()} — {sample['sample_id']}", fontsize=11
    )
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved trajectory: {output_path}")


def plot_recovery_summary(all_metrics: Dict, output_path: pathlib.Path):
    """Bar chart of recovery ratios across all scenarios."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    scenarios = []
    means = []
    stds = []
    colors = []

    color_map = {
        "average_pitch": "#1f77b4",
        "average_duration": "#ff7f0e",
    }

    for concept, concept_data in all_metrics.items():
        for scenario, samples in concept_data.items():
            if not samples:
                continue
            recoveries = [s["recovery_pct"] for s in samples]
            label = f"{'Pitch' if 'pitch' in concept else 'Duration'}\n{scenario.replace('_', ' ')}"
            scenarios.append(label)
            means.append(np.mean(recoveries))
            stds.append(np.std(recoveries))
            colors.append(color_map.get(concept, "gray"))

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(scenarios))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        alpha=0.8,
        edgecolor="white",
        linewidth=1.2,
    )

    # Value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{mean:.0f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel("Recovery %", fontsize=12)
    ax.set_title(
        "Round-Trip Reversibility: How Much Deviation Is Recovered?", fontsize=12
    )
    ax.set_ylim(0, 100)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3, label="Perfect recovery")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, label="No recovery")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved recovery summary: {output_path}")


def trim_phase3_audio(wavs_dir: pathlib.Path, output_dir: pathlib.Path):
    """Trim WAVs to Phase 3 only for A/B comparison.

    Assumes constant tempo and splits proportionally by phase token counts.
    """
    try:
        import wave
    except ImportError:
        logger.error("wave module not available")
        return

    phase3_start_ratio = (PHASE_TOKENS[0] + PHASE_TOKENS[1]) / TOTAL_TOKENS
    output_dir.mkdir(parents=True, exist_ok=True)
    trimmed = 0

    for concept_dir in sorted(wavs_dir.iterdir()):
        if not concept_dir.is_dir() or concept_dir.name.startswith("."):
            continue
        for scn_dir in sorted(concept_dir.iterdir()):
            if not scn_dir.is_dir() or scn_dir.name.startswith("."):
                continue

            out_scn = output_dir / concept_dir.name / scn_dir.name
            out_scn.mkdir(parents=True, exist_ok=True)

            for wav_file in sorted(scn_dir.glob("*.wav")):
                try:
                    with wave.open(str(wav_file), "rb") as wf:
                        n_channels = wf.getnchannels()
                        sampwidth = wf.getsampwidth()
                        framerate = wf.getframerate()
                        n_frames = wf.getnframes()

                        # Skip to Phase 3 start
                        phase3_frame = int(n_frames * phase3_start_ratio)
                        wf.setpos(phase3_frame)
                        phase3_data = wf.readframes(n_frames - phase3_frame)

                    out_path = out_scn / f"phase3__{wav_file.name}"
                    with wave.open(str(out_path), "wb") as wf_out:
                        wf_out.setnchannels(n_channels)
                        wf_out.setsampwidth(sampwidth)
                        wf_out.setframerate(framerate)
                        wf_out.writeframes(phase3_data)

                    trimmed += 1
                except Exception as e:
                    logger.warning(f"Failed to trim {wav_file}: {e}")

    logger.info(f"Trimmed {trimmed} WAVs to Phase 3 → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize round-trip reversibility")
    parser.add_argument(
        "--results_dir",
        type=pathlib.Path,
        default=pathlib.Path("exp/sod/pid_steering/experiments/roundtrip_v7"),
    )
    parser.add_argument(
        "--wavs_dir",
        type=pathlib.Path,
        default=pathlib.Path("roundtrip_wavs"),
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=pathlib.Path("paper/figures/roundtrip"),
    )
    parser.add_argument(
        "--top_n", type=int, default=3, help="Plot top N samples per scenario"
    )
    parser.add_argument(
        "--trim_phase3", action="store_true", help="Also trim WAVs to Phase 3"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    concepts = ["average_pitch", "average_duration"]
    scenarios = ["low_up_then_down", "high_down_then_up"]

    all_metrics = {}

    for concept in concepts:
        all_metrics[concept] = {}
        for scenario in scenarios:
            samples = load_per_sample_data(args.results_dir, concept, scenario)
            if not samples:
                continue

            metrics = compute_recovery_metrics(samples, concept)
            all_metrics[concept][scenario] = metrics

            if not metrics:
                continue

            # Print summary
            recoveries = [m["recovery_pct"] for m in metrics]
            logger.info(
                f"{concept} / {scenario}: "
                f"Recovery = {np.mean(recoveries):.1f}% ± {np.std(recoveries):.1f}% "
                f"(best: {max(recoveries):.0f}%, worst: {min(recoveries):.0f}%)"
            )

            # Plot top N individual trajectories
            for i, sample in enumerate(metrics[: args.top_n]):
                plot_path = (
                    args.output_dir
                    / concept
                    / scenario
                    / f"trajectory_{i+1}_{sample['sample_id']}.png"
                )
                plot_trajectory(sample, concept, scenario, plot_path)

    # Summary recovery bar chart
    plot_recovery_summary(all_metrics, args.output_dir / "recovery_summary.png")

    # Print the key numbers for the presentation
    print("\n" + "=" * 60)
    print("REVERSIBILITY SUMMARY")
    print("=" * 60)
    for concept in concepts:
        attr = "Pitch" if "pitch" in concept else "Duration"
        for scenario in scenarios:
            metrics = all_metrics.get(concept, {}).get(scenario, [])
            if not metrics:
                continue
            recoveries = [m["recovery_pct"] for m in metrics]
            errors = [m["remaining_error"] for m in metrics]
            peaks = [m["peak_deviation"] for m in metrics]
            print(f"\n{attr} — {scenario}:")
            print(
                f"  Peak deviation (W2):   {np.mean(peaks):.1f} ± {np.std(peaks):.1f}"
            )
            print(
                f"  Remaining error (W3):  {np.mean(errors):.1f} ± {np.std(errors):.1f}"
            )
            print(
                f"  Recovery:              {np.mean(recoveries):.1f}% ± {np.std(recoveries):.1f}%"
            )
            print(
                f"  Best sample:           {metrics[0]['sample_id']} ({metrics[0]['recovery_pct']:.0f}%)"
            )

    # Trim Phase 3 audio for A/B comparison
    if args.trim_phase3 and args.wavs_dir.exists():
        trim_phase3_audio(args.wavs_dir, args.output_dir / "phase3_clips")


if __name__ == "__main__":
    main()
