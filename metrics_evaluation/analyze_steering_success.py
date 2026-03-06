#!/usr/bin/env python3
"""Post-hoc steering success analysis from saved .npy token files.

Loads all generated .npy files (both SAS and DiffMean, both unconditioned and
conditioned), measures pitch and duration per sample, computes baselines from
the α=0 samples, and reports steering success rates.

Steering success definition:
  - pitch_success:    α_p > 0 → mean_pitch > baseline_pitch  (or < if α_p < 0)
  - duration_success: α_d > 0 → mean_duration > baseline_dur  (or < if α_d < 0)
  - both_success:     pitch_success AND duration_success

For conditioned generation, success compares generated output against the
conditioning context (initial song excerpt), same as test_dual_conditioned.py.

For unconditioned generation, success compares against the α=0 baseline mean
across all 15 samples.

Usage:
    python metrics_evaluation/analyze_steering_success.py \
        --workspace_dir exp/sod/sparse_steering/fmd_workspace

    # Or point directly at experiment dirs:
    python metrics_evaluation/analyze_steering_success.py \
        --sas_uncond_dir exp/sod/sparse_steering/dual_steering/unconditioned_triplet_fixed \
        --dm_uncond_dir steering_interventions/dual_steering/outputs/diffmean_unconditioned \
        --sas_cond_dir exp/sod/sparse_steering/dual_steering/conditioned_ek2 \
        --dm_cond_dir steering_interventions/dual_steering/outputs/diffmean_conditioned
"""

import argparse
import csv
import json
import logging
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np

# Add mmt to path for representation module
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
import representation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

# Ground truth metrics from paper
GROUND_TRUTH = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}


# ─────────────────────────────────────────────────────────────────────────────
# Measurement helpers
# ─────────────────────────────────────────────────────────────────────────────
NAN_QUALITY = {
    "pitch_class_entropy": np.nan,
    "scale_consistency": np.nan,
    "groove_consistency": np.nan,
}


def evaluate_quality(music) -> dict:
    """Compute muspy quality metrics on a decoded Music object."""
    try:
        import muspy

        music.trim(music.resolution * 64)
        if not music.tracks or not music.tracks[0].notes:
            return dict(NAN_QUALITY)

        return {
            "pitch_class_entropy": float(muspy.pitch_class_entropy(music)),
            "scale_consistency": float(muspy.scale_consistency(music)) * 100.0,
            "groove_consistency": float(
                muspy.groove_consistency(music, 4 * music.resolution)
            )
            * 100.0,
        }
    except Exception as e:
        logger.debug(f"evaluate_quality failed: {e}")
        return dict(NAN_QUALITY)


def calculate_degradation(metrics: dict) -> dict:
    """Quality degradation vs ground truth (lower = better)."""
    gt = GROUND_TRUTH
    ed = abs(metrics["pitch_class_entropy"] - gt["pitch_class_entropy"])
    sd = max(0.0, gt["scale_consistency"] - metrics["scale_consistency"])
    gd = max(0.0, gt["groove_consistency"] - metrics["groove_consistency"])
    total = ed + sd + gd
    return {
        "entropy_diff": float(ed),
        "scale_diff": float(sd),
        "groove_diff": float(gd),
        "total_degradation": float(total),
    }


def measure_sample(tokens: np.ndarray, encoding: dict) -> dict:
    """Extract pitch, duration, quality metrics, and degradation from tokens.

    Returns dict with:
        mean_pitch, mean_duration, n_notes,
        pitch_class_entropy, scale_consistency, groove_consistency,
        entropy_diff, scale_diff, groove_diff, total_degradation
    """
    base = {
        "mean_pitch": None,
        "mean_duration": None,
        "n_notes": 0,
        **NAN_QUALITY,
        "entropy_diff": np.nan,
        "scale_diff": np.nan,
        "groove_diff": np.nan,
        "total_degradation": np.nan,
    }
    try:
        music = representation.decode(tokens, encoding)
        pitches = []
        durations = []
        for track in music.tracks:
            for note in track.notes:
                pitches.append(note.pitch)
                durations.append(note.duration)

        if len(pitches) < 5:
            base["n_notes"] = len(pitches)
            return base

        base["mean_pitch"] = float(np.mean(pitches))
        base["mean_duration"] = float(np.mean(durations))
        base["n_notes"] = len(pitches)

        qm = evaluate_quality(music)
        base.update(qm)

        if not np.isnan(qm["pitch_class_entropy"]):
            deg = calculate_degradation(qm)
            base.update(deg)

        return base
    except Exception as e:
        logger.debug(f"measure_sample failed: {e}")
        return base


# ─────────────────────────────────────────────────────────────────────────────
# Unconditioned analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyze_unconditioned(
    exp_dir: pathlib.Path,
    encoding: dict,
    method_label: str,
) -> list:
    """Analyze unconditioned samples from a DiffMean or SAS experiment dir.

    Expected layout:
      DiffMean: {exp_dir}/midi/{strategy}/p+X.XX_d+Y.YY_sN.npy
      SAS:      {exp_dir}/{strategy}/p+X.XX_d+Y.YY_sN.npy   (or midi/{strategy}/...)

    Returns list of per-sample dicts.
    """
    results = []

    # Try both layouts: midi/{strategy}/ and {strategy}/
    strategy_dirs = []
    midi_base = exp_dir / "midi"
    if midi_base.exists():
        strategy_dirs.extend(d for d in midi_base.iterdir() if d.is_dir())
    # Also check direct strategy subdirs (SAS layout)
    for d in exp_dir.iterdir():
        if d.is_dir() and d.name != "midi" and not d.name.startswith("."):
            # Check if it contains .npy files directly (not another level)
            if list(d.glob("*.npy")):
                strategy_dirs.append(d)

    if not strategy_dirs:
        logger.warning(f"No strategy dirs found in {exp_dir}")
        return results

    for strat_dir in sorted(strategy_dirs):
        strategy = strat_dir.name
        npy_files = sorted(strat_dir.glob("*.npy"))
        if not npy_files:
            continue

        logger.info(f"  {method_label}/{strategy}: {len(npy_files)} .npy files")

        # Parse and measure each sample
        samples = []
        for npy in npy_files:
            # Parse filename: p+X.XX_d+Y.YY_sN.npy
            m = re.match(r"p([+-]?\d+\.\d+)_d([+-]?\d+\.\d+)_s(\d+)", npy.stem)
            if not m:
                continue

            lp = float(m.group(1))
            ld = float(m.group(2))
            sample_idx = int(m.group(3))

            tokens = np.load(npy)
            meas = measure_sample(tokens, encoding)

            samples.append(
                {
                    "method": method_label,
                    "mode": "uncond",
                    "strategy": strategy,
                    "lambda_pitch": lp,
                    "lambda_duration": ld,
                    "sample_idx": sample_idx,
                    **meas,
                }
            )

        if not samples:
            continue

        # Compute baseline from α=(0,0) samples
        baseline_samples = [
            s
            for s in samples
            if abs(s["lambda_pitch"]) < 1e-6
            and abs(s["lambda_duration"]) < 1e-6
            and s["mean_pitch"] is not None
        ]
        if not baseline_samples:
            logger.warning(f"  No baseline (0,0) samples for {method_label}/{strategy}")
            baseline_pitch = None
            baseline_dur = None
        else:
            baseline_pitch = float(np.mean([s["mean_pitch"] for s in baseline_samples]))
            baseline_dur = float(
                np.mean([s["mean_duration"] for s in baseline_samples])
            )
            logger.info(
                f"  Baseline ({len(baseline_samples)} samples): "
                f"pitch={baseline_pitch:.1f}, duration={baseline_dur:.1f}"
            )

        # Compute success for each sample
        for s in samples:
            s["baseline_pitch"] = baseline_pitch
            s["baseline_duration"] = baseline_dur

            lp, ld = s["lambda_pitch"], s["lambda_duration"]
            mp, md = s["mean_pitch"], s["mean_duration"]

            # Skip baseline samples (neither concept steered)
            if abs(lp) < 1e-6 and abs(ld) < 1e-6:
                s["pitch_success"] = None
                s["duration_success"] = None
                s["both_success"] = None
                continue

            # Pitch success
            if mp is not None and baseline_pitch is not None and abs(lp) > 1e-6:
                if lp > 0:
                    s["pitch_success"] = bool(mp > baseline_pitch)
                else:
                    s["pitch_success"] = bool(mp < baseline_pitch)
            else:
                s["pitch_success"] = None

            # Duration success
            if md is not None and baseline_dur is not None and abs(ld) > 1e-6:
                if ld > 0:
                    s["duration_success"] = bool(md > baseline_dur)
                else:
                    s["duration_success"] = bool(md < baseline_dur)
            else:
                s["duration_success"] = None

            # Both success (only when both concepts are being steered)
            if abs(lp) > 1e-6 and abs(ld) > 1e-6:
                s["both_success"] = (
                    s["pitch_success"] is True and s["duration_success"] is True
                )
            elif abs(lp) > 1e-6:
                s["both_success"] = s["pitch_success"]
            elif abs(ld) > 1e-6:
                s["both_success"] = s["duration_success"]
            else:
                s["both_success"] = None

        results.extend(samples)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Conditioned analysis
# ─────────────────────────────────────────────────────────────────────────────
def analyze_conditioned(
    exp_dir: pathlib.Path,
    encoding: dict,
    method_label: str,
) -> list:
    """Analyze conditioned samples.

    Both layouts share the same hierarchy:
      {exp_dir}/{scenario}/{strategy}/{lambda_dir}/{song}.npy

    DiffMean lambda_dir: ap+X.X_ad+Y.Y
    SAS lambda_dir:      lp+X.XX_ld+Y.YY

    For conditioned, success = generated pitch/duration moved relative to the
    α/λ=(0,0) baseline for the same song.
    """
    results = []

    if not exp_dir.exists():
        logger.warning(f"Conditioned dir not found: {exp_dir}")
        return results

    # Regex to match either DiffMean or SAS lambda dir naming
    LAMBDA_RE = re.compile(r"(?:ap|lp)([+-]?\d+\.?\d*)_(?:ad|ld)([+-]?\d+\.?\d*)")

    for scenario_dir in sorted(exp_dir.iterdir()):
        if not scenario_dir.is_dir() or scenario_dir.name.startswith("."):
            continue
        scenario = scenario_dir.name

        # Determine expected direction from scenario name
        pitch_sign, dur_sign = _scenario_signs(scenario)

        for strat_dir in sorted(scenario_dir.iterdir()):
            if not strat_dir.is_dir():
                continue
            strategy = strat_dir.name

            # Collect all samples grouped by song
            song_data = defaultdict(list)  # song_name -> [(lp, ld, npy_path)]

            for alpha_dir in sorted(strat_dir.iterdir()):
                if not alpha_dir.is_dir():
                    continue

                m = LAMBDA_RE.match(alpha_dir.name)
                if not m:
                    continue
                lp = float(m.group(1))
                ld = float(m.group(2))

                for npy in sorted(alpha_dir.glob("*.npy")):
                    song_data[npy.stem].append((lp, ld, npy))

            if not song_data:
                continue

            logger.info(
                f"  {method_label}/{scenario}/{strategy}: " f"{len(song_data)} songs"
            )

            for song_name, entries in song_data.items():
                # Find baseline (α/λ = 0) for this song
                baseline_tokens = None
                for lp, ld, npy in entries:
                    if abs(lp) < 1e-6 and abs(ld) < 1e-6:
                        baseline_tokens = np.load(npy)
                        break

                if baseline_tokens is not None:
                    bl_meas = measure_sample(baseline_tokens, encoding)
                    bl_pitch = bl_meas["mean_pitch"]
                    bl_dur = bl_meas["mean_duration"]
                else:
                    bl_pitch = None
                    bl_dur = None

                for lp, ld, npy in entries:
                    if abs(lp) < 1e-6 and abs(ld) < 1e-6:
                        continue  # skip baselines

                    tokens = np.load(npy)
                    meas = measure_sample(tokens, encoding)
                    mp, md = meas["mean_pitch"], meas["mean_duration"]

                    # Pitch success (use scenario direction)
                    if mp is not None and bl_pitch is not None and pitch_sign != 0:
                        pitch_ok = bool(
                            (mp > bl_pitch) if pitch_sign > 0 else (mp < bl_pitch)
                        )
                    else:
                        pitch_ok = None

                    # Duration success
                    if md is not None and bl_dur is not None and dur_sign != 0:
                        dur_ok = bool((md > bl_dur) if dur_sign > 0 else (md < bl_dur))
                    else:
                        dur_ok = None

                    both_ok = (
                        (pitch_ok is True and dur_ok is True)
                        if (pitch_ok is not None and dur_ok is not None)
                        else None
                    )

                    results.append(
                        {
                            "method": method_label,
                            "mode": "cond",
                            "strategy": strategy,
                            "scenario": scenario,
                            "song": song_name,
                            "lambda_pitch": lp,
                            "lambda_duration": ld,
                            "baseline_pitch": bl_pitch,
                            "baseline_duration": bl_dur,
                            "mean_pitch": mp,
                            "mean_duration": md,
                            "n_notes": meas["n_notes"],
                            "pitch_success": pitch_ok,
                            "duration_success": dur_ok,
                            "both_success": both_ok,
                        }
                    )

    return results


def _scenario_signs(scenario: str) -> tuple:
    """Infer intended pitch/duration direction from scenario name.

    Returns (pitch_sign, dur_sign) where +1 = increase, -1 = decrease.
    """
    s = scenario.lower()
    # Pattern: "{from}_to_{to}" e.g. "low_pitch_short_duration_to_high_long"
    if "_to_" in s:
        target = s.split("_to_")[-1]
        pitch_sign = 1 if "high" in target else (-1 if "low" in target else 0)
        dur_sign = 1 if "long" in target else (-1 if "short" in target else 0)
    else:
        pitch_sign = 0
        dur_sign = 0
    return pitch_sign, dur_sign


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation and printing
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(all_results: list):
    """Print comprehensive steering success summary."""

    # ── Overall by method × mode ──
    print("\n" + "=" * 100)
    print(" STEERING SUCCESS ANALYSIS")
    print("=" * 100)

    # Group by (method, mode, strategy)
    groups = defaultdict(list)
    for r in all_results:
        if r.get("both_success") is None:
            continue
        key = (r["method"], r["mode"], r["strategy"])
        groups[key].append(r)

    # Print overall table
    print(
        f"\n{'Method':<12} {'Mode':<8} {'Strategy':<30} "
        f"{'Pitch%':>8} {'Dur%':>8} {'Both%':>8} {'N':>6}"
    )
    print("-" * 90)

    for (method, mode, strategy), samples in sorted(groups.items()):
        n = len(samples)
        pitch_ok = sum(1 for s in samples if s.get("pitch_success") is True)
        pitch_total = sum(1 for s in samples if s.get("pitch_success") is not None)
        dur_ok = sum(1 for s in samples if s.get("duration_success") is True)
        dur_total = sum(1 for s in samples if s.get("duration_success") is not None)
        both_ok = sum(1 for s in samples if s.get("both_success") is True)

        pitch_rate = (pitch_ok / pitch_total * 100) if pitch_total else 0
        dur_rate = (dur_ok / dur_total * 100) if dur_total else 0
        both_rate = (both_ok / n * 100) if n else 0

        print(
            f"{method:<12} {mode:<8} {strategy:<30} "
            f"{pitch_rate:>7.1f}% {dur_rate:>7.1f}% {both_rate:>7.1f}% {n:>6}"
        )

    # ── Per-lambda success rates (unconditioned only) ──
    uncond = [
        r
        for r in all_results
        if r["mode"] == "uncond" and r.get("both_success") is not None
    ]
    if uncond:
        print("\n" + "=" * 100)
        print(" UNCONDITIONED: SUCCESS RATE BY λ VALUE")
        print("=" * 100)

        # Group by (method, strategy)
        uncond_groups = defaultdict(list)
        for r in uncond:
            uncond_groups[(r["method"], r["strategy"])].append(r)

        for (method, strategy), samples in sorted(uncond_groups.items()):
            print(f"\n  {method} / {strategy}:")

            # By λ_pitch (marginal)
            pitch_bins = defaultdict(lambda: {"ok": 0, "total": 0})
            for s in samples:
                lp = s["lambda_pitch"]
                if abs(lp) > 1e-6 and s.get("pitch_success") is not None:
                    pitch_bins[lp]["total"] += 1
                    if s["pitch_success"]:
                        pitch_bins[lp]["ok"] += 1

            if pitch_bins:
                print(f"    {'λ_pitch':>10} {'Success':>10} {'Rate':>8}")
                print("    " + "-" * 30)
                for lp in sorted(pitch_bins.keys()):
                    b = pitch_bins[lp]
                    rate = b["ok"] / b["total"] * 100 if b["total"] else 0
                    print(
                        f"    {lp:>+10.2f} {b['ok']:>5}/{b['total']:<4} {rate:>7.1f}%"
                    )

            # By λ_duration (marginal)
            dur_bins = defaultdict(lambda: {"ok": 0, "total": 0})
            for s in samples:
                ld = s["lambda_duration"]
                if abs(ld) > 1e-6 and s.get("duration_success") is not None:
                    dur_bins[ld]["total"] += 1
                    if s["duration_success"]:
                        dur_bins[ld]["ok"] += 1

            if dur_bins:
                print(f"\n    {'λ_dur':>10} {'Success':>10} {'Rate':>8}")
                print("    " + "-" * 30)
                for ld in sorted(dur_bins.keys()):
                    b = dur_bins[ld]
                    rate = b["ok"] / b["total"] * 100 if b["total"] else 0
                    print(
                        f"    {ld:>+10.2f} {b['ok']:>5}/{b['total']:<4} {rate:>7.1f}%"
                    )

            # Dual success heatmap
            dual = [
                s
                for s in samples
                if abs(s["lambda_pitch"]) > 1e-6 and abs(s["lambda_duration"]) > 1e-6
            ]
            if dual:
                print(
                    f"\n    Dual-concept success rate heatmap (both pitch & duration correct):"
                )
                lp_vals = sorted(set(s["lambda_pitch"] for s in dual))
                ld_vals = sorted(set(s["lambda_duration"] for s in dual))

                col_label = "λ_p \\ λ_d"
                header = f"    {col_label:>10}"
                for ld in ld_vals:
                    header += f" {ld:>+7.2f}"
                print(header)
                print("    " + "-" * (11 + 8 * len(ld_vals)))

                for lp in lp_vals:
                    row = f"    {lp:>+10.2f}"
                    for ld in ld_vals:
                        cell = [
                            s
                            for s in dual
                            if abs(s["lambda_pitch"] - lp) < 1e-6
                            and abs(s["lambda_duration"] - ld) < 1e-6
                        ]
                        if cell:
                            ok = sum(1 for s in cell if s["both_success"])
                            rate = ok / len(cell) * 100
                            row += f" {rate:>6.0f}%"
                        else:
                            row += f" {'—':>7}"
                    print(row)

    # ── Conditioned per-scenario ──
    cond = [
        r
        for r in all_results
        if r["mode"] == "cond" and r.get("both_success") is not None
    ]
    if cond:
        print("\n" + "=" * 100)
        print(" CONDITIONED: SUCCESS RATE BY SCENARIO")
        print("=" * 100)

        cond_groups = defaultdict(list)
        for r in cond:
            cond_groups[(r["method"], r["strategy"], r.get("scenario", "?"))].append(r)

        print(
            f"\n{'Method':<12} {'Strategy':<30} {'Scenario':<45} "
            f"{'Pitch%':>8} {'Dur%':>8} {'Both%':>8} {'N':>5}"
        )
        print("-" * 125)

        for (method, strategy, scenario), samples in sorted(cond_groups.items()):
            n = len(samples)
            pitch_ok = sum(1 for s in samples if s.get("pitch_success") is True)
            pitch_total = sum(1 for s in samples if s.get("pitch_success") is not None)
            dur_ok = sum(1 for s in samples if s.get("duration_success") is True)
            dur_total = sum(1 for s in samples if s.get("duration_success") is not None)
            both_ok = sum(1 for s in samples if s.get("both_success") is True)

            pitch_rate = (pitch_ok / pitch_total * 100) if pitch_total else 0
            dur_rate = (dur_ok / dur_total * 100) if dur_total else 0
            both_rate = (both_ok / n * 100) if n else 0

            print(
                f"{method:<12} {strategy:<30} {scenario:<45} "
                f"{pitch_rate:>7.1f}% {dur_rate:>7.1f}% {both_rate:>7.1f}% {n:>5}"
            )

    # ── Effect sizes (mean pitch/duration deltas) ──
    print("\n" + "=" * 100)
    print(" MEAN STEERING EFFECT (Δ from baseline)")
    print("=" * 100)

    for (method, mode, strategy), samples in sorted(groups.items()):
        valid = [
            s
            for s in samples
            if s["mean_pitch"] is not None and s["baseline_pitch"] is not None
        ]
        if not valid:
            continue

        pitch_deltas = [
            s["mean_pitch"] - s["baseline_pitch"]
            for s in valid
            if s["baseline_pitch"] is not None
        ]
        dur_deltas = [
            s["mean_duration"] - s["baseline_duration"]
            for s in valid
            if s["baseline_duration"] is not None
        ]

        if pitch_deltas:
            print(
                f"  {method:<10} {mode:<7} {strategy:<28}  "
                f"Δpitch: {np.mean(pitch_deltas):>+6.1f} ± {np.std(pitch_deltas):.1f}  "
                f"Δduration: {np.mean(dur_deltas):>+6.1f} ± {np.std(dur_deltas):.1f}  "
                f"(N={len(valid)})"
            )

    # ── Quality degradation ──
    print("\n" + "=" * 100)
    print(" QUALITY DEGRADATION (vs ground truth)")
    print("=" * 100)
    print(
        f"  Ground truth: PCE={GROUND_TRUTH['pitch_class_entropy']:.3f}  "
        f"SC={GROUND_TRUTH['scale_consistency']:.2f}%  "
        f"GC={GROUND_TRUTH['groove_consistency']:.2f}%"
    )

    print(
        f"\n  {'Method':<10} {'Mode':<7} {'Strategy':<28}  "
        f"{'PCE':>6} {'SC%':>7} {'GC%':>7}  "
        f"{'ΔE':>5} {'ΔS':>5} {'ΔG':>5} {'TotDeg':>7}  {'N':>5}"
    )
    print("  " + "-" * 105)

    for (method, mode, strategy), samples in sorted(groups.items()):
        valid_q = [
            s for s in samples if not np.isnan(s.get("total_degradation", np.nan))
        ]
        if not valid_q:
            continue

        pce_m = np.mean([s["pitch_class_entropy"] for s in valid_q])
        sc_m = np.mean([s["scale_consistency"] for s in valid_q])
        gc_m = np.mean([s["groove_consistency"] for s in valid_q])
        ed_m = np.mean([s["entropy_diff"] for s in valid_q])
        sd_m = np.mean([s["scale_diff"] for s in valid_q])
        gd_m = np.mean([s["groove_diff"] for s in valid_q])
        td_m = np.mean([s["total_degradation"] for s in valid_q])

        print(
            f"  {method:<10} {mode:<7} {strategy:<28}  "
            f"{pce_m:>6.3f} {sc_m:>6.1f}% {gc_m:>6.1f}%  "
            f"{ed_m:>5.2f} {sd_m:>5.1f} {gd_m:>5.1f} {td_m:>7.2f}  "
            f"{len(valid_q):>5}"
        )

    # ── Optimal configs (unconditioned) ──
    _print_optimal_configs(all_results)

    print("=" * 100)


def _print_optimal_configs(all_results: list):
    """Find and print optimal λ for each method/strategy (unconditioned).

    Optimal = highest both_success among configs with total_degradation
    below the median degradation, breaking ties by lowest degradation.
    """
    uncond = [
        r
        for r in all_results
        if r["mode"] == "uncond" and r.get("both_success") is not None
    ]
    if not uncond:
        return

    print("\n" + "=" * 100)
    print(" OPTIMAL λ CONFIGURATIONS (unconditioned)")
    print("  Criterion: max both_success%, then min total_degradation")
    print("=" * 100)

    # Group by (method, strategy)
    ms_groups = defaultdict(list)
    for r in uncond:
        ms_groups[(r["method"], r["strategy"])].append(r)

    for (method, strategy), samples in sorted(ms_groups.items()):
        # Aggregate by (λ_p, λ_d)
        lambda_agg = defaultdict(
            lambda: {
                "pitch_ok": 0,
                "dur_ok": 0,
                "both_ok": 0,
                "n": 0,
                "degs": [],
                "pce": [],
                "sc": [],
                "gc": [],
            }
        )
        for s in samples:
            key = (s["lambda_pitch"], s["lambda_duration"])
            a = lambda_agg[key]
            a["n"] += 1
            if s.get("pitch_success"):
                a["pitch_ok"] += 1
            if s.get("duration_success"):
                a["dur_ok"] += 1
            if s.get("both_success"):
                a["both_ok"] += 1
            td = s.get("total_degradation", np.nan)
            if not np.isnan(td):
                a["degs"].append(td)
                a["pce"].append(s["pitch_class_entropy"])
                a["sc"].append(s["scale_consistency"])
                a["gc"].append(s["groove_consistency"])

        # Build sortable list
        configs = []
        for (lp, ld), a in lambda_agg.items():
            if a["n"] == 0:
                continue
            both_rate = a["both_ok"] / a["n"]
            avg_deg = np.mean(a["degs"]) if a["degs"] else 999.0
            configs.append(
                {
                    "lp": lp,
                    "ld": ld,
                    "both_rate": both_rate,
                    "pitch_rate": a["pitch_ok"] / a["n"],
                    "dur_rate": a["dur_ok"] / a["n"],
                    "avg_deg": avg_deg,
                    "avg_pce": np.mean(a["pce"]) if a["pce"] else np.nan,
                    "avg_sc": np.mean(a["sc"]) if a["sc"] else np.nan,
                    "avg_gc": np.mean(a["gc"]) if a["gc"] else np.nan,
                    "n": a["n"],
                }
            )

        # Sort: highest both_rate first, then lowest degradation
        configs.sort(key=lambda c: (-c["both_rate"], c["avg_deg"]))

        print(f"\n  {method} / {strategy} — Top 10 configs:")
        print(
            f"    {'λ_p':>6} {'λ_d':>6}  "
            f"{'Both%':>6} {'Pitch%':>7} {'Dur%':>6}  "
            f"{'PCE':>6} {'SC%':>6} {'GC%':>6}  {'Degrad':>7}  {'N':>4}"
        )
        print("    " + "-" * 80)
        for c in configs[:10]:
            print(
                f"    {c['lp']:>+6.2f} {c['ld']:>+6.2f}  "
                f"{c['both_rate']*100:>5.0f}% {c['pitch_rate']*100:>6.0f}% "
                f"{c['dur_rate']*100:>5.0f}%  "
                f"{c['avg_pce']:>6.3f} {c['avg_sc']:>5.1f}% {c['avg_gc']:>5.1f}%  "
                f"{c['avg_deg']:>7.2f}  {c['n']:>4}"
            )


def save_csv(all_results: list, output_path: pathlib.Path):
    """Save per-sample results to CSV."""
    fieldnames = [
        "method",
        "mode",
        "strategy",
        "scenario",
        "song",
        "lambda_pitch",
        "lambda_duration",
        "sample_idx",
        "baseline_pitch",
        "baseline_duration",
        "mean_pitch",
        "mean_duration",
        "n_notes",
        "pitch_class_entropy",
        "scale_consistency",
        "groove_consistency",
        "entropy_diff",
        "scale_diff",
        "groove_diff",
        "total_degradation",
        "pitch_success",
        "duration_success",
        "both_success",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            clean = {k: (v if v is not None else "") for k, v in row.items()}
            writer.writerow(clean)
    logger.info(f"Saved {len(all_results)} rows to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc steering success analysis from saved .npy files"
    )
    parser.add_argument(
        "--workspace_dir",
        type=pathlib.Path,
        default=None,
        help="FMD workspace dir (optional, used for output only)",
    )
    # SAS dirs
    parser.add_argument(
        "--sas_uncond_dir",
        type=pathlib.Path,
        default=PROJECT_ROOT
        / "exp"
        / "sod"
        / "sparse_steering"
        / "dual_steering"
        / "unconditioned_triplet_fixed",
    )
    parser.add_argument(
        "--sas_cond_dir",
        type=pathlib.Path,
        default=None,
        help="SAS conditioned dir (e.g. conditioned_ek2). "
        "If not set, tries conditioned_ek2 and conditioned_gs_both.",
    )
    # DiffMean dirs
    parser.add_argument(
        "--dm_uncond_dir",
        type=pathlib.Path,
        default=PROJECT_ROOT
        / "steering_interventions"
        / "dual_steering"
        / "outputs"
        / "diffmean_unconditioned",
    )
    parser.add_argument(
        "--dm_cond_dir",
        type=pathlib.Path,
        default=PROJECT_ROOT
        / "steering_interventions"
        / "dual_steering"
        / "outputs"
        / "diffmean_conditioned",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Where to save CSV/JSON output. Defaults to workspace_dir or current dir.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        if args.workspace_dir:
            args.output_dir = args.workspace_dir
        else:
            args.output_dir = (
                PROJECT_ROOT / "exp" / "sod" / "sparse_steering" / "fmd_workspace"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load encoding
    encoding_path = (
        PROJECT_ROOT / "data" / "sod" / "processed" / "notes" / "encoding.json"
    )
    encoding = representation.load_encoding(str(encoding_path))
    logger.info(f"Loaded encoding from {encoding_path}")

    all_results = []

    # ── DiffMean unconditioned ──
    if args.dm_uncond_dir and args.dm_uncond_dir.exists():
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing DiffMean unconditioned: {args.dm_uncond_dir}")
        dm_uncond = analyze_unconditioned(args.dm_uncond_dir, encoding, "DM")
        all_results.extend(dm_uncond)
        logger.info(f"  → {len(dm_uncond)} samples analyzed")

    # ── SAS unconditioned ──
    if args.sas_uncond_dir and args.sas_uncond_dir.exists():
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing SAS unconditioned: {args.sas_uncond_dir}")
        sas_uncond = analyze_unconditioned(args.sas_uncond_dir, encoding, "SAS")
        all_results.extend(sas_uncond)
        logger.info(f"  → {len(sas_uncond)} samples analyzed")

    # ── DiffMean conditioned ──
    if args.dm_cond_dir and args.dm_cond_dir.exists():
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing DiffMean conditioned: {args.dm_cond_dir}")
        dm_cond = analyze_conditioned(args.dm_cond_dir, encoding, "DM")
        all_results.extend(dm_cond)
        logger.info(f"  → {len(dm_cond)} samples analyzed")

    # ── SAS conditioned ──
    sas_cond_dirs = []
    if args.sas_cond_dir:
        sas_cond_dirs = [args.sas_cond_dir]
    else:
        # Try default dirs
        base = PROJECT_ROOT / "exp" / "sod" / "sparse_steering" / "dual_steering"
        for name in ["conditioned_ek2", "conditioned_gs_both"]:
            d = base / name
            if d.exists():
                sas_cond_dirs.append(d)

    for cond_dir in sas_cond_dirs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing SAS conditioned: {cond_dir}")
        sas_cond = analyze_conditioned(cond_dir, encoding, "SAS")
        all_results.extend(sas_cond)
        logger.info(f"  → {len(sas_cond)} samples analyzed")

    if not all_results:
        logger.error("No results found! Check that experiment directories exist.")
        sys.exit(1)

    logger.info(f"\nTotal: {len(all_results)} samples across all experiments")

    # Print summary
    print_summary(all_results)

    # Save CSV
    csv_path = args.output_dir / "steering_success.csv"
    save_csv(all_results, csv_path)

    # Save JSON summary
    summary = {}
    groups = defaultdict(list)
    for r in all_results:
        if r.get("both_success") is None:
            continue
        key = f"{r['method']}_{r['mode']}_{r['strategy']}"
        groups[key].append(r)

    for key, samples in sorted(groups.items()):
        n = len(samples)
        pitch_ok = sum(1 for s in samples if s.get("pitch_success") is True)
        pitch_total = sum(1 for s in samples if s.get("pitch_success") is not None)
        dur_ok = sum(1 for s in samples if s.get("duration_success") is True)
        dur_total = sum(1 for s in samples if s.get("duration_success") is not None)
        both_ok = sum(1 for s in samples if s.get("both_success") is True)

        summary[key] = {
            "n_samples": n,
            "pitch_success_rate": pitch_ok / pitch_total if pitch_total else None,
            "duration_success_rate": dur_ok / dur_total if dur_total else None,
            "both_success_rate": both_ok / n if n else None,
        }

    json_path = args.output_dir / "steering_success_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()
