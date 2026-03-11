"""
Smooth Steering Comparison Experiment.

Compares smooth vs abrupt steering across both SAS and DiffMean methods,
varying ``n_ramp`` to find the optimal ramp-up length for eliminating
audible discontinuity while preserving steering effectiveness.

Features
--------
- **Dual concept**: runs both ``average_pitch`` and ``average_duration``
  (or whichever concepts you specify via ``--concepts``).
- **Per-lambda breakdown**: every (method, n_ramp, lambda) combination is
  tracked individually so you can identify the best lambda.
- **FMD integration**: if ``--compute_fmd`` is passed *and*
  ``frechet_music_distance`` is installed, FMD is computed per experiment
  against the SOD reference set.
- **Unified listening list**: ranks all generated WAVs across every
  experiment by (steering effectiveness × quality) and prints the top-20
  paths so you can listen immediately.

Outputs
-------
- ``smooth_comparison_summary.json``: per-(method, concept, n_ramp, lambda) rows
- ``smooth_comparison_best_lambdas.json``: best lambda per config
- ``smooth_comparison_plot.png``: effectiveness vs n_ramp (per concept)
- ``smooth_comparison_quality.png``: quality breakdown vs n_ramp
- ``smooth_comparison_fmd.json``: FMD results  (only with ``--compute_fmd``)
- ``listening_priority_all.txt``: unified WAV ranking

Usage
-----
    # Full run: both concepts, FMD, 4 ramp values
    python metrics_evaluation/smooth_steering_comparison.py \\
        --concepts average_pitch,average_duration --n_songs 5 --gpu 0

    # Quick test
    python metrics_evaluation/smooth_steering_comparison.py \\
        --concepts average_pitch --n_ramps 0,64 --n_songs 3 --gpu 0

    # With FMD (requires frechet_music_distance + CLaMP2)
    python metrics_evaluation/smooth_steering_comparison.py \\
        --concepts average_pitch --n_songs 5 --gpu 0 --compute_fmd
"""

import argparse
import json
import logging
import pathlib
import subprocess
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

REPO_ROOT = pathlib.Path(__file__).parent.parent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ── Subprocess launchers ─────────────────────────────────────────────────────


def run_sas_experiment(
    concept: str,
    n_songs: int,
    lambdas: str,
    layers: str,
    conditioning_beats: int,
    continuation_len: int,
    output_dir: pathlib.Path,
    experiment_name: str,
    gpu: int = None,
    smooth: bool = False,
    schedule: str = "cosine",
    n_ramp: int = 64,
    n_decay: int = 0,
    lambda_maintain: float = 1.0,
) -> pathlib.Path:
    """Run the SAS conditioned evaluator as a subprocess."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "sparse_steering" / "conditioned_evaluator_sas.py"),
        "--concept",
        concept,
        "--n_songs",
        str(n_songs),
        "--lambdas",
        lambdas,
        "--layers",
        layers,
        "--conditioning_beats",
        str(conditioning_beats),
        "--continuation_len",
        str(continuation_len),
        "--output_dir",
        str(output_dir),
        "--experiment_name",
        experiment_name,
    ]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    if smooth:
        cmd += [
            "--smooth",
            "--schedule",
            schedule,
            "--n_ramp",
            str(n_ramp),
            "--n_decay",
            str(n_decay),
            "--lambda_maintain",
            str(lambda_maintain),
        ]

    logger.info(f"Running SAS experiment: {experiment_name}")
    logger.info(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"SAS experiment failed (exit code {result.returncode})")
    else:
        logger.info(f"SAS experiment completed: {experiment_name}")

    results_file = output_dir / experiment_name / f"conditioned_results_{concept}.json"
    return results_file


def run_dm_experiment(
    concept: str,
    n_songs: int,
    alphas: str,
    conditioning_beats: int,
    continuation_len: int,
    output_dir: pathlib.Path,
    experiment_name: str,
    gpu: int = None,
    smooth: bool = False,
    schedule: str = "cosine",
    n_ramp: int = 64,
    n_decay: int = 0,
    lambda_maintain: float = 1.0,
) -> pathlib.Path:
    """Run the DiffMean conditioned evaluator as a subprocess."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "steering_interventions" / "conditioned_evaluator.py"),
        "--concept",
        concept,
        "--n_songs",
        str(n_songs),
        "--alphas",
        alphas,
        "--conditioning_beats",
        str(conditioning_beats),
        "--continuation_len",
        str(continuation_len),
        "--output_dir",
        str(output_dir),
        "--experiment_name",
        experiment_name,
    ]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    if smooth:
        cmd += [
            "--smooth",
            "--schedule",
            schedule,
            "--n_ramp",
            str(n_ramp),
            "--n_decay",
            str(n_decay),
            "--lambda_maintain",
            str(lambda_maintain),
        ]

    logger.info(f"Running DM experiment: {experiment_name}")
    logger.info(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"DM experiment failed (exit code {result.returncode})")
    else:
        logger.info(f"DM experiment completed: {experiment_name}")

    results_file = output_dir / experiment_name / "conditioned_results.json"
    return results_file


def load_results(filepath: pathlib.Path) -> dict:
    """Load a results JSON, returning empty dict on failure."""
    if not filepath.exists():
        logger.warning(f"Results file not found: {filepath}")
        return {}
    with open(filepath) as f:
        return json.load(f)


# ── Per-lambda extraction ────────────────────────────────────────────────────


def extract_per_lambda_summary(
    results: dict,
    method: str,
    concept: str,
    n_ramp: int,
) -> List[dict]:
    """Extract per-lambda rows from a conditioned evaluation result.

    Returns one summary row per lambda/alpha value tested.
    """
    all_results = results.get("results", [])
    if not all_results:
        return []

    is_duration = "duration" in concept
    key_field = "lambda" if method == "SAS" else "alpha"
    change_field = "duration_change" if is_duration else "pitch_change"

    # Group by lambda/alpha
    grouped: Dict[float, list] = defaultdict(list)
    for r in all_results:
        strength = r.get(key_field, 0.0)
        grouped[strength].append(r)

    rows = []
    for strength, samples in sorted(grouped.items()):
        n = len(samples)

        # Steering success: change in the intended direction
        successes = 0
        for r in samples:
            change = r.get(change_field, 0)
            s = r.get(key_field, 0)
            if (s > 0 and change > 0) or (s < 0 and change < 0):
                successes += 1
        success_rate = successes / n if n > 0 else 0.0

        # Mean absolute change
        abs_changes = [abs(r.get(change_field, 0)) for r in samples]
        mean_abs_change = float(np.mean(abs_changes)) if abs_changes else 0.0

        # Quality degradation
        degradations = [
            r["degradation"]["total_degradation"]
            for r in samples
            if "degradation" in r
            and not np.isnan(r["degradation"].get("total_degradation", np.nan))
        ]
        mean_degradation = float(np.mean(degradations)) if degradations else np.nan

        # Per-metric quality
        entropies = [
            r["quality_metrics"]["pitch_class_entropy"]
            for r in samples
            if "quality_metrics" in r
            and not np.isnan(r["quality_metrics"].get("pitch_class_entropy", np.nan))
        ]
        scales = [
            r["quality_metrics"]["scale_consistency"]
            for r in samples
            if "quality_metrics" in r
            and not np.isnan(r["quality_metrics"].get("scale_consistency", np.nan))
        ]
        grooves = [
            r["quality_metrics"]["groove_consistency"]
            for r in samples
            if "quality_metrics" in r
            and not np.isnan(r["quality_metrics"].get("groove_consistency", np.nan))
        ]

        rows.append(
            {
                "method": method,
                "concept": concept,
                "n_ramp": n_ramp,
                "smooth": n_ramp > 0,
                "strength": strength,
                "n_samples": n,
                "steering_success_rate": success_rate,
                "mean_absolute_change": mean_abs_change,
                "mean_degradation": mean_degradation,
                "mean_pitch_class_entropy": (
                    float(np.mean(entropies)) if entropies else np.nan
                ),
                "mean_scale_consistency": float(np.mean(scales)) if scales else np.nan,
                "mean_groove_consistency": (
                    float(np.mean(grooves)) if grooves else np.nan
                ),
            }
        )

    return rows


def extract_aggregate_summary(per_lambda_rows: List[dict]) -> dict:
    """Aggregate per-lambda rows into one summary (excluding baseline strength=0)."""
    steered = [r for r in per_lambda_rows if r["strength"] != 0.0]
    if not steered:
        return {
            "method": per_lambda_rows[0]["method"] if per_lambda_rows else "",
            "concept": per_lambda_rows[0]["concept"] if per_lambda_rows else "",
            "n_ramp": per_lambda_rows[0]["n_ramp"] if per_lambda_rows else 0,
            "smooth": (
                per_lambda_rows[0].get("smooth", False) if per_lambda_rows else False
            ),
            "n_samples": 0,
        }

    return {
        "method": steered[0]["method"],
        "concept": steered[0]["concept"],
        "n_ramp": steered[0]["n_ramp"],
        "smooth": steered[0]["smooth"],
        "n_samples": sum(r["n_samples"] for r in steered),
        "steering_success_rate": float(
            np.mean([r["steering_success_rate"] for r in steered])
        ),
        "mean_absolute_change": float(
            np.mean([r["mean_absolute_change"] for r in steered])
        ),
        "mean_degradation": float(np.nanmean([r["mean_degradation"] for r in steered])),
        "mean_pitch_class_entropy": float(
            np.nanmean([r["mean_pitch_class_entropy"] for r in steered])
        ),
        "mean_scale_consistency": float(
            np.nanmean([r["mean_scale_consistency"] for r in steered])
        ),
        "mean_groove_consistency": float(
            np.nanmean([r["mean_groove_consistency"] for r in steered])
        ),
    }


def find_best_lambdas(all_per_lambda: List[dict]) -> List[dict]:
    """For each (method, concept, n_ramp), find the lambda with highest success rate.

    Ties broken by mean_absolute_change (higher is better).
    """
    grouped = defaultdict(list)
    for row in all_per_lambda:
        if row["strength"] == 0.0:
            continue
        key = (row["method"], row["concept"], row["n_ramp"])
        grouped[key].append(row)

    best = []
    for key, rows in sorted(grouped.items()):
        winner = max(
            rows, key=lambda r: (r["steering_success_rate"], r["mean_absolute_change"])
        )
        best.append(
            {
                **winner,
                "best_for": f"{key[0]}_{key[1]}_nramp{key[2]}",
            }
        )
    return best


# ── FMD integration ──────────────────────────────────────────────────────────


def compute_fmd_for_experiment(
    experiment_dir: pathlib.Path,
    reference_dir: pathlib.Path,
    gpu: int = 0,
) -> Optional[float]:
    """Compute FMD between experiment MIDIs and SOD reference MIDIs.

    Returns FMD score or None on failure.
    """
    try:
        from frechet_music_distance import FrechetMusicDistance
    except ImportError:
        logger.warning("frechet_music_distance not installed — skipping FMD")
        return None

    # Collect all .mid files in experiment dir
    mid_files = list(experiment_dir.rglob("*.mid"))
    if len(mid_files) < 2:
        logger.warning(
            f"Only {len(mid_files)} MIDIs in {experiment_dir} — need ≥2 for FMD"
        )
        return None

    # Count reference MIDIs
    ref_mids = list(reference_dir.rglob("*.mid"))
    if len(ref_mids) < 2:
        logger.warning(f"Only {len(ref_mids)} reference MIDIs — need ≥2 for FMD")
        return None

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    metric = FrechetMusicDistance(
        feature_extractor="clamp2",
        gaussian_estimator="mle",
        verbose=False,
    )

    try:
        score = metric.score(
            reference_path=str(reference_dir),
            test_path=str(experiment_dir),
        )
        return float(score)
    except Exception as e:
        logger.error(f"FMD computation failed: {e}")
        return None


# ── Unified WAV ranking ──────────────────────────────────────────────────────


def collect_wav_ranking(
    output_dir: pathlib.Path,
    all_per_lambda: List[dict],
) -> List[dict]:
    """Walk all experiment subdirectories and rank WAV files.

    Each WAV is scored by: steering_effectiveness × (1 - degradation).
    """
    wavs = []
    for wav_path in output_dir.rglob("*.wav"):
        rel = wav_path.relative_to(output_dir)
        parts = str(rel).split("/")

        # Try to match this WAV to an experiment
        experiment_name = parts[0] if parts else ""

        # Extract metadata from path structure
        wavs.append(
            {
                "path": str(rel),
                "experiment": experiment_name,
                "filename": wav_path.name,
            }
        )

    # Sort by experiment name for grouping
    wavs.sort(key=lambda x: x["path"])
    return wavs


def build_listening_priority(
    output_dir: pathlib.Path,
    all_results_by_exp: Dict[str, dict],
) -> str:
    """Build a unified listening priority list across all experiments.

    Ranks by: priority = 0.6 × normalized_steering + 0.4 × (1 - normalized_degradation)
    """
    scored = []

    for exp_name, results_data in all_results_by_exp.items():
        for r in results_data.get("results", []):
            if r.get("generated_n_notes", 0) == 0:
                continue
            if "degradation" not in r:
                continue

            deg = r["degradation"].get("total_degradation", np.nan)
            if np.isnan(deg):
                continue

            # Determine steering direction and score
            is_dur = "duration" in r.get("category", "")
            key_field = "lambda" if "lambda" in r else "alpha"
            strength = r.get(key_field, r.get("alpha", r.get("lambda", 0)))

            if is_dur:
                change = abs(r.get("duration_change", 0))
            else:
                change = abs(r.get("pitch_change", 0))

            norm_steer = min(change / 50.0, 1.0)
            norm_qual = 1.0 - min(deg / 100.0, 1.0)
            priority = (norm_steer * 0.6 + norm_qual * 0.4) * 100

            # Build WAV path from experiment structure
            scored.append(
                {
                    "priority": priority,
                    "experiment": exp_name,
                    "song": r.get("song_name", "unknown"),
                    "category": r.get("category", ""),
                    "strength": strength,
                    "change": change,
                    "degradation": deg,
                    "n_notes": r.get("generated_n_notes", 0),
                    "entropy": r.get("quality_metrics", {}).get(
                        "pitch_class_entropy", np.nan
                    ),
                    "scale": r.get("quality_metrics", {}).get(
                        "scale_consistency", np.nan
                    ),
                    "groove": r.get("quality_metrics", {}).get(
                        "groove_consistency", np.nan
                    ),
                }
            )

    scored.sort(key=lambda x: x["priority"], reverse=True)

    lines = [
        "=" * 100,
        "UNIFIED LISTENING PRIORITY LIST  (Smooth Steering Comparison)",
        "Ranked by: Steering Effectiveness (60%) + Low Degradation (40%)",
        "=" * 100,
        "",
    ]

    for i, s in enumerate(scored[:30], 1):
        lines.append(
            f"{i:>3}. [{s['priority']:.1f}] {s['song']}  "
            f"({s['experiment']}, {s['category']}, "
            f"strength={s['strength']:+.2f})"
        )
        lines.append(
            f"     |change|={s['change']:.1f}, degradation={s['degradation']:.1f}, "
            f"notes={s['n_notes']}"
        )
        lines.append(
            f"     quality: entropy={s['entropy']:.2f}, "
            f"scale={s['scale']:.1f}%, groove={s['groove']:.1f}%"
        )
        lines.append("")

    return "\n".join(lines)


# ── Plotting ─────────────────────────────────────────────────────────────────


def generate_comparison_plots(
    aggregate_summaries: List[dict],
    per_lambda_rows: List[dict],
    output_dir: pathlib.Path,
    fmd_results: Optional[Dict] = None,
):
    """Generate comparison plots across concepts, methods, and ramp values."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    concepts = sorted(set(s["concept"] for s in aggregate_summaries))
    n_concepts = len(concepts)

    # ── Figure 1: Effectiveness vs n_ramp (per concept) ──────────────────
    fig, axes = plt.subplots(n_concepts, 3, figsize=(15, 5 * n_concepts), squeeze=False)

    for row, concept in enumerate(concepts):
        concept_label = concept.replace("average_", "").title()

        sas = [
            s
            for s in aggregate_summaries
            if s["method"] == "SAS"
            and s["concept"] == concept
            and s.get("n_samples", 0) > 0
        ]
        dm = [
            s
            for s in aggregate_summaries
            if s["method"] == "DiffMean"
            and s["concept"] == concept
            and s.get("n_samples", 0) > 0
        ]

        # Success rate
        ax = axes[row][0]
        for data, label, color in [
            (sas, "SAS", "#2196F3"),
            (dm, "DiffMean", "#FF9800"),
        ]:
            if data:
                x = [s["n_ramp"] for s in data]
                y = [s["steering_success_rate"] * 100 for s in data]
                ax.plot(x, y, "o-", label=label, color=color, linewidth=2)
        ax.set_xlabel("n_ramp (steps)")
        ax.set_ylabel("Steering Success Rate (%)")
        ax.set_title(f"{concept_label}: Success vs Ramp Length")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Magnitude
        ax = axes[row][1]
        metric_label = (
            "|Duration Change|" if "duration" in concept else "|Pitch Change|"
        )
        for data, label, color in [
            (sas, "SAS", "#2196F3"),
            (dm, "DiffMean", "#FF9800"),
        ]:
            if data:
                x = [s["n_ramp"] for s in data]
                y = [s["mean_absolute_change"] for s in data]
                ax.plot(x, y, "o-", label=label, color=color, linewidth=2)
        ax.set_xlabel("n_ramp (steps)")
        ax.set_ylabel(metric_label)
        ax.set_title(f"{concept_label}: Magnitude vs Ramp Length")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Degradation
        ax = axes[row][2]
        for data, label, color in [
            (sas, "SAS", "#2196F3"),
            (dm, "DiffMean", "#FF9800"),
        ]:
            if data:
                x = [s["n_ramp"] for s in data]
                y = [s["mean_degradation"] for s in data]
                ax.plot(x, y, "o-", label=label, color=color, linewidth=2)
        ax.set_xlabel("n_ramp (steps)")
        ax.set_ylabel("Mean Quality Degradation")
        ax.set_title(f"{concept_label}: Degradation vs Ramp Length")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Smooth Steering: Effectiveness vs Ramp Length",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "smooth_comparison_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved smooth_comparison_plot.png")

    # ── Figure 2: Quality breakdown ──────────────────────────────────────
    quality_metrics = [
        ("mean_pitch_class_entropy", "Pitch Class Entropy"),
        ("mean_scale_consistency", "Scale Consistency (%)"),
        ("mean_groove_consistency", "Groove Consistency (%)"),
    ]

    fig, axes = plt.subplots(n_concepts, 3, figsize=(15, 5 * n_concepts), squeeze=False)

    for row, concept in enumerate(concepts):
        concept_label = concept.replace("average_", "").title()
        sas = [
            s
            for s in aggregate_summaries
            if s["method"] == "SAS"
            and s["concept"] == concept
            and s.get("n_samples", 0) > 0
        ]
        dm = [
            s
            for s in aggregate_summaries
            if s["method"] == "DiffMean"
            and s["concept"] == concept
            and s.get("n_samples", 0) > 0
        ]

        for col, (metric, title) in enumerate(quality_metrics):
            ax = axes[row][col]
            for data, label, color in [
                (sas, "SAS", "#2196F3"),
                (dm, "DiffMean", "#FF9800"),
            ]:
                if data:
                    x = [s["n_ramp"] for s in data]
                    y = [s.get(metric, np.nan) for s in data]
                    ax.plot(x, y, "o-", label=label, color=color, linewidth=2)
            ax.set_xlabel("n_ramp (steps)")
            ax.set_ylabel(title)
            ax.set_title(f"{concept_label}: {title}")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Smooth Steering: Quality Metrics vs Ramp Length",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "smooth_comparison_quality.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    logger.info("Saved smooth_comparison_quality.png")

    # ── Figure 3: Per-lambda effectiveness (best lambda search) ──────────
    steered_rows = [r for r in per_lambda_rows if r["strength"] != 0.0]
    if steered_rows:
        fig, axes = plt.subplots(
            n_concepts, 2, figsize=(14, 5 * n_concepts), squeeze=False
        )

        for row, concept in enumerate(concepts):
            concept_label = concept.replace("average_", "").title()

            for col, method in enumerate(["SAS", "DiffMean"]):
                ax = axes[row][col]
                method_rows = [
                    r
                    for r in steered_rows
                    if r["method"] == method and r["concept"] == concept
                ]

                # Group by n_ramp
                by_ramp = defaultdict(list)
                for r in method_rows:
                    by_ramp[r["n_ramp"]].append(r)

                for n_ramp, ramp_rows in sorted(by_ramp.items()):
                    strengths = [r["strength"] for r in ramp_rows]
                    successes = [r["steering_success_rate"] * 100 for r in ramp_rows]
                    label = f"n_ramp={n_ramp}" if n_ramp > 0 else "abrupt"
                    ax.plot(strengths, successes, "o-", label=label, linewidth=1.5)

                ax.set_xlabel("Lambda / Alpha")
                ax.set_ylabel("Steering Success Rate (%)")
                ax.set_title(f"{concept_label} — {method}: Success per Lambda/Alpha")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.axvline(0, color="grey", linestyle=":", alpha=0.3)

        plt.suptitle(
            "Per-Lambda Steering Success", fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        plt.savefig(
            output_dir / "smooth_comparison_per_lambda.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        logger.info("Saved smooth_comparison_per_lambda.png")

    # ── Figure 4: FMD comparison (if available) ──────────────────────────
    if fmd_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        sas_fmd = {
            k: v
            for k, v in fmd_results.items()
            if k.startswith("sas_") and v is not None
        }
        dm_fmd = {
            k: v
            for k, v in fmd_results.items()
            if k.startswith("dm_") and v is not None
        }

        if sas_fmd or dm_fmd:
            for fmd_data, label, color in [
                (sas_fmd, "SAS", "#2196F3"),
                (dm_fmd, "DiffMean", "#FF9800"),
            ]:
                if fmd_data:
                    # Parse n_ramp from key pattern "method_concept_nramp_N"
                    points = []
                    for k, v in fmd_data.items():
                        parts = k.split("_nramp_")
                        if len(parts) == 2:
                            try:
                                nr = int(parts[1])
                                points.append((nr, v))
                            except ValueError:
                                pass
                    if points:
                        points.sort()
                        ax.plot(
                            [p[0] for p in points],
                            [p[1] for p in points],
                            "o-",
                            label=label,
                            color=color,
                            linewidth=2,
                        )

            ax.set_xlabel("n_ramp (steps)")
            ax.set_ylabel("FMD (lower = better)")
            ax.set_title("FMD vs Ramp Length")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                output_dir / "smooth_comparison_fmd.png", dpi=150, bbox_inches="tight"
            )
            plt.close()
            logger.info("Saved smooth_comparison_fmd.png")


# ── Display ──────────────────────────────────────────────────────────────────


def print_comparison_table(aggregate_summaries: List[dict]):
    """Print aggregate summary table to stdout."""
    print("\n" + "=" * 110)
    print("SMOOTH STEERING COMPARISON — AGGREGATE SUMMARY")
    print("=" * 110)
    print(
        f"{'Method':<10} {'Concept':<18} {'n_ramp':>6} {'Smooth':>6} {'Samples':>8} "
        f"{'Success%':>9} {'|Change|':>9} {'Degrad':>8} "
        f"{'Entropy':>8} {'Scale%':>7} {'Groove%':>8}"
    )
    print("-" * 110)

    for s in sorted(
        aggregate_summaries, key=lambda x: (x["concept"], x["method"], x["n_ramp"])
    ):
        if s.get("n_samples", 0) == 0:
            print(f"{s['method']:<10} {s['concept']:<18} {s['n_ramp']:>6}   (no data)")
            continue
        print(
            f"{s['method']:<10} {s['concept']:<18} {s['n_ramp']:>6} "
            f"{'yes' if s['smooth'] else 'no':>6} "
            f"{s['n_samples']:>8} "
            f"{s['steering_success_rate']*100:>8.1f}% "
            f"{s['mean_absolute_change']:>9.2f} "
            f"{s['mean_degradation']:>8.2f} "
            f"{s.get('mean_pitch_class_entropy', float('nan')):>8.3f} "
            f"{s.get('mean_scale_consistency', float('nan')):>6.1f}% "
            f"{s.get('mean_groove_consistency', float('nan')):>7.1f}%"
        )
    print("=" * 110)


def print_best_lambdas(best: List[dict]):
    """Print best lambda per configuration."""
    print("\n" + "=" * 100)
    print("BEST LAMBDA / ALPHA PER CONFIGURATION")
    print("=" * 100)
    print(
        f"{'Method':<10} {'Concept':<18} {'n_ramp':>6} "
        f"{'Best λ/α':>9} {'Success%':>9} {'|Change|':>9} {'Degrad':>8}"
    )
    print("-" * 100)

    for b in sorted(best, key=lambda x: (x["concept"], x["method"], x["n_ramp"])):
        print(
            f"{b['method']:<10} {b['concept']:<18} {b['n_ramp']:>6} "
            f"{b['strength']:>+9.2f} "
            f"{b['steering_success_rate']*100:>8.1f}% "
            f"{b['mean_absolute_change']:>9.2f} "
            f"{b['mean_degradation']:>8.2f}"
        )
    print("=" * 100)


def print_per_lambda_table(per_lambda_rows: List[dict]):
    """Print per-lambda breakdown."""
    print("\n" + "=" * 120)
    print("PER-LAMBDA BREAKDOWN")
    print("=" * 120)

    # Group by (concept, method, n_ramp)
    grouped = defaultdict(list)
    for r in per_lambda_rows:
        grouped[(r["concept"], r["method"], r["n_ramp"])].append(r)

    for key, rows in sorted(grouped.items()):
        concept, method, n_ramp = key
        smooth_label = f"smooth n_ramp={n_ramp}" if n_ramp > 0 else "abrupt"
        print(f"\n  {method} — {concept} — {smooth_label}:")
        print(
            f"  {'Lambda':>8} {'N':>5} {'Success%':>9} {'|Change|':>9} "
            f"{'Degrad':>8} {'Entropy':>8} {'Scale%':>7} {'Groove%':>8}"
        )
        print("  " + "-" * 90)

        for r in sorted(rows, key=lambda x: x["strength"]):
            print(
                f"  {r['strength']:>+8.2f} {r['n_samples']:>5} "
                f"{r['steering_success_rate']*100:>8.1f}% "
                f"{r['mean_absolute_change']:>9.2f} "
                f"{r['mean_degradation']:>8.2f} "
                f"{r.get('mean_pitch_class_entropy', float('nan')):>8.3f} "
                f"{r.get('mean_scale_consistency', float('nan')):>6.1f}% "
                f"{r.get('mean_groove_consistency', float('nan')):>7.1f}%"
            )
    print("=" * 120)


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare smooth vs abrupt steering across SAS and DiffMean",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--concepts",
        type=str,
        default="average_pitch,average_duration",
        help="Comma-separated concepts to evaluate (default: both pitch and duration)",
    )
    parser.add_argument(
        "--n_songs",
        type=int,
        default=5,
        help="Songs per category",
    )
    parser.add_argument(
        "--n_ramps",
        type=str,
        default="0,32,64,128",
        help="Comma-separated n_ramp values to test",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "sigmoid"],
        help="Schedule function for smooth steering",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0.0,0.5,1.0,1.5,2.0,-0.5,-1.0,-1.5,-2.0",
        help="Comma-separated lambda values for SAS",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.5,1.0,1.5,2.0,-0.5,-1.0,-1.5,-2.0",
        help="Comma-separated alpha values for DiffMean",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="10",
        help="Layers to steer (SAS)",
    )
    parser.add_argument(
        "--conditioning_beats",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--continuation_len",
        type=int,
        default=256,
    )
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=REPO_ROOT / "exp" / "sod" / "smooth_steering_comparison",
    )
    parser.add_argument(
        "--skip_sas",
        action="store_true",
        help="Skip SAS experiments (run DiffMean only)",
    )
    parser.add_argument(
        "--skip_dm",
        action="store_true",
        help="Skip DiffMean experiments (run SAS only)",
    )
    parser.add_argument(
        "--compute_fmd",
        action="store_true",
        help="Compute FMD against SOD reference (requires frechet_music_distance)",
    )
    parser.add_argument(
        "--sod_reference_dir",
        type=pathlib.Path,
        default=REPO_ROOT
        / "exp"
        / "sod"
        / "sparse_steering"
        / "fmd_workspace"
        / "reference_sod",
        help="SOD reference MIDI directory for FMD computation",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    concepts = [c.strip() for c in args.concepts.split(",")]
    n_ramps = [int(x.strip()) for x in args.n_ramps.split(",")]

    logger.info(f"Concepts: {concepts}")
    logger.info(f"n_ramp values: {n_ramps}")
    logger.info(f"Schedule: {args.schedule}")
    logger.info(f"Output: {args.output_dir}")

    all_per_lambda: List[dict] = []
    all_aggregate: List[dict] = []
    all_results_by_exp: Dict[str, dict] = {}
    fmd_results: Dict[str, Optional[float]] = {}

    # ── Run experiments ──────────────────────────────────────────────────
    for concept in concepts:
        for n_ramp in n_ramps:
            smooth = n_ramp > 0

            # SAS
            if not args.skip_sas:
                exp_name = f"sas_{concept}_nramp_{n_ramp}"
                results_file = run_sas_experiment(
                    concept=concept,
                    n_songs=args.n_songs,
                    lambdas=args.lambdas,
                    layers=args.layers,
                    conditioning_beats=args.conditioning_beats,
                    continuation_len=args.continuation_len,
                    output_dir=args.output_dir,
                    experiment_name=exp_name,
                    gpu=args.gpu,
                    smooth=smooth,
                    schedule=args.schedule,
                    n_ramp=n_ramp,
                )
                results = load_results(results_file)
                all_results_by_exp[exp_name] = results

                per_lam = extract_per_lambda_summary(results, "SAS", concept, n_ramp)
                all_per_lambda.extend(per_lam)
                all_aggregate.append(extract_aggregate_summary(per_lam))

                if args.compute_fmd:
                    exp_dir = args.output_dir / exp_name
                    fmd_val = compute_fmd_for_experiment(
                        exp_dir, args.sod_reference_dir, args.gpu or 0
                    )
                    fmd_results[f"sas_{concept}_nramp_{n_ramp}"] = fmd_val

            # DiffMean
            if not args.skip_dm:
                exp_name = f"dm_{concept}_nramp_{n_ramp}"
                results_file = run_dm_experiment(
                    concept=concept,
                    n_songs=args.n_songs,
                    alphas=args.alphas,
                    conditioning_beats=args.conditioning_beats,
                    continuation_len=args.continuation_len,
                    output_dir=args.output_dir,
                    experiment_name=exp_name,
                    gpu=args.gpu,
                    smooth=smooth,
                    schedule=args.schedule,
                    n_ramp=n_ramp,
                )
                results = load_results(results_file)
                all_results_by_exp[exp_name] = results

                per_lam = extract_per_lambda_summary(
                    results, "DiffMean", concept, n_ramp
                )
                all_per_lambda.extend(per_lam)
                all_aggregate.append(extract_aggregate_summary(per_lam))

                if args.compute_fmd:
                    exp_dir = args.output_dir / exp_name
                    fmd_val = compute_fmd_for_experiment(
                        exp_dir, args.sod_reference_dir, args.gpu or 0
                    )
                    fmd_results[f"dm_{concept}_nramp_{n_ramp}"] = fmd_val

    # ── Find best lambdas ────────────────────────────────────────────────
    best_lambdas = find_best_lambdas(all_per_lambda)

    # ── Save results ─────────────────────────────────────────────────────
    summary_file = args.output_dir / "smooth_comparison_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "aggregate": all_aggregate,
                "per_lambda": all_per_lambda,
                "best_lambdas": best_lambdas,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved summary: {summary_file}")

    best_file = args.output_dir / "smooth_comparison_best_lambdas.json"
    with open(best_file, "w") as f:
        json.dump(best_lambdas, f, indent=2)
    logger.info(f"Saved best lambdas: {best_file}")

    if fmd_results:
        fmd_file = args.output_dir / "smooth_comparison_fmd.json"
        with open(fmd_file, "w") as f:
            json.dump(fmd_results, f, indent=2)
        logger.info(f"Saved FMD results: {fmd_file}")

    # ── Build unified listening list ─────────────────────────────────────
    listening_text = build_listening_priority(args.output_dir, all_results_by_exp)
    listening_file = args.output_dir / "listening_priority_all.txt"
    with open(listening_file, "w") as f:
        f.write(listening_text)
    logger.info(f"Saved listening list: {listening_file}")

    # ── Display ──────────────────────────────────────────────────────────
    print_comparison_table(all_aggregate)
    print_best_lambdas(best_lambdas)
    print_per_lambda_table(all_per_lambda)

    if fmd_results:
        print("\n" + "=" * 80)
        print("FMD RESULTS (vs SOD reference)")
        print("=" * 80)
        for k, v in sorted(fmd_results.items()):
            fmd_str = f"{v:.1f}" if v is not None else "N/A"
            print(f"  {k:<45} FMD = {fmd_str}")
        print("=" * 80)

    print("\n" + listening_text[:2000])
    if len(listening_text) > 2000:
        print(f"\n... (full list at {listening_file})")

    # ── Plots ────────────────────────────────────────────────────────────
    generate_comparison_plots(
        all_aggregate,
        all_per_lambda,
        args.output_dir,
        fmd_results if fmd_results else None,
    )

    logger.info("Comparison complete!")


if __name__ == "__main__":
    main()
