#!/usr/bin/env python3
"""Rank steered results by control strength + quality, export top N as WAV.

For each exported sample, also exports its baseline (lp=0, ld=0) for A/B comparison.

Usage:
    python sparse_steering/dual_steering/export_best_wavs.py \
        --results_json exp/sod/sparse_steering/dual_steering/conditioned_ek2/conditioned_results.json \
        --npy_root exp/sod/sparse_steering/dual_steering/conditioned_ek2 \
        --out_dir exp/sod/sparse_steering/dual_steering/ranked_wavs \
        --top 5
"""

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "mmt"))
import representation


def compute_score(r: dict, w_control: float = 1.0, w_degrad: float = 10.0) -> float:
    """Score = control_strength - w_degrad * degradation.

    Higher is better.  control_strength = |pitch_delta| + |duration_delta|.
    The degradation weight is higher so that bad-sounding results sink.
    """
    control = abs(r["pitch_delta"]) + abs(r["duration_delta"])
    deg = r["degradation"]["total_degradation"]
    return w_control * control - w_degrad * deg


def npy_path_for(npy_root: pathlib.Path, r: dict) -> pathlib.Path:
    """Build the .npy path from a result record."""
    lp = f"lp{r['lambda_pitch']:+.2f}"
    ld = f"ld{r['lambda_duration']:+.2f}"
    song = r["song_name"].split("/")[-1]
    return npy_root / r["scenario"] / r["strategy"] / f"{lp}_{ld}" / f"{song}.npy"


def baseline_npy_path_for(npy_root: pathlib.Path, r: dict) -> pathlib.Path:
    """Build the baseline (lp=0, ld=0) .npy path for the same song/scenario."""
    song = r["song_name"].split("/")[-1]
    return npy_root / r["scenario"] / r["strategy"] / "lp+0.00_ld+0.00" / f"{song}.npy"


# Short scenario labels
SCENARIO_SHORT = {
    "high_pitch_long_duration_to_low_short": "hi2lo_short",
    "high_pitch_short_duration_to_low_long": "hi2lo_long",
    "low_pitch_long_duration_to_high_short": "lo2hi_short",
    "low_pitch_short_duration_to_high_long": "lo2hi_long",
}


def main():
    parser = argparse.ArgumentParser(description="Export top-ranked steered WAVs")
    parser.add_argument("--results_json", type=pathlib.Path, required=True)
    parser.add_argument("--npy_root", type=pathlib.Path, required=True)
    parser.add_argument("--out_dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--top", type=int, default=5, help="How many to export per scenario"
    )
    parser.add_argument("--skip_wav", action="store_true", help="Only MIDI (faster)")
    parser.add_argument(
        "--w_control",
        type=float,
        default=1.0,
        help="Weight for control strength in score",
    )
    parser.add_argument(
        "--w_degrad",
        type=float,
        default=10.0,
        help="Weight for degradation penalty in score",
    )
    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)

    # Filter: both_success, both lambdas non-zero, degradation exists
    results = [
        r
        for r in data["results"]
        if r.get("both_success") is True
        and r["degradation"] is not None
        and r["lambda_pitch"] != 0
        and r["lambda_duration"] != 0
    ]

    # Score all results
    for r in results:
        r["_score"] = compute_score(r, args.w_control, args.w_degrad)

    # Group by scenario, rank within each, take top N per scenario
    scenarios = sorted(set(r["scenario"] for r in results))
    top = []
    for sc_name in scenarios:
        sc_label = SCENARIO_SHORT.get(sc_name, sc_name[:16])
        sc_results = [r for r in results if r["scenario"] == sc_name]
        sc_results.sort(key=lambda r: r["_score"], reverse=True)
        sc_top = sc_results[: args.top]

        print(f"\n{'─'*95}")
        print(
            f"  {sc_label}  ({len(sc_results)} both-success with both λ≠0)  — top {len(sc_top)}"
        )
        print(f"{'─'*95}")
        print(
            f"  {'#':>2}  {'Score':>6}  {'Deg':>5}  {'dPitch':>7}  {'dDur':>7}  "
            f"{'lp':>5}  {'ld':>5}  Song"
        )
        for j, r in enumerate(sc_top):
            song = r["song_name"].split("/")[-1]
            print(
                f"  {j+1:2d}  {r['_score']:6.1f}  {r['degradation']['total_degradation']:5.2f}  "
                f"{r['pitch_delta']:+7.1f}  {r['duration_delta']:+7.1f}  "
                f"{r['lambda_pitch']:+5.2f}  {r['lambda_duration']:+5.2f}  "
                f"{song}"
            )

        # Tag each with its within-scenario rank for file naming
        for j, r in enumerate(sc_top):
            r["_sc_rank"] = j + 1
        top.extend(sc_top)

    print(f"\nTotal to export: {len(top)} steered + baselines")

    # Load encoding
    encoding = representation.load_encoding(
        str(
            pathlib.Path(__file__).parent.parent.parent
            / "data"
            / "sod"
            / "processed"
            / "notes"
            / "encoding.json"
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Track which baselines we already exported (same song+scenario = same baseline)
    exported_baselines = set()

    print(f"\nConverting {len(top)} steered + baselines...")
    for r in top:
        sc = SCENARIO_SHORT.get(r["scenario"], r["scenario"][:16])
        song = r["song_name"].split("/")[-1]
        deg = r["degradation"]["total_degradation"]
        rank = r["_sc_rank"]

        # Put each scenario in its own subfolder
        sc_dir = args.out_dir / sc
        sc_dir.mkdir(parents=True, exist_ok=True)

        # Steered file
        steered_name = (
            f"{rank:02d}_steered_{song}"
            f"_lp{r['lambda_pitch']:+.2f}_ld{r['lambda_duration']:+.2f}"
            f"_deg{deg:.2f}"
        )
        steered_npy = npy_path_for(args.npy_root, r)
        if steered_npy.exists():
            tokens = np.load(steered_npy)
            music = representation.decode(tokens, encoding)
            music.write(str(sc_dir / f"{steered_name}.mid"))
            if not args.skip_wav:
                music.write_audio(str(sc_dir / f"{steered_name}.wav"))
            print(f"  {sc}/{steered_name}")
        else:
            print(f"  MISSING: {steered_npy}")

        # Baseline (one per song+scenario)
        baseline_key = (r["scenario"], r["song_name"])
        if baseline_key not in exported_baselines:
            exported_baselines.add(baseline_key)
            baseline_name = f"00_baseline_{song}"
            baseline_npy = baseline_npy_path_for(args.npy_root, r)
            if baseline_npy.exists():
                tokens = np.load(baseline_npy)
                music = representation.decode(tokens, encoding)
                music.write(str(sc_dir / f"{baseline_name}.mid"))
                if not args.skip_wav:
                    music.write_audio(str(sc_dir / f"{baseline_name}.wav"))
                print(f"  {sc}/{baseline_name} (baseline)")
            else:
                print(f"  BASELINE MISSING: {baseline_npy}")

    print(f"\nDone! Files in: {args.out_dir}")
    print("Structure: <out_dir>/<scenario>/00_baseline_*.wav + NN_steered_*.wav")
    print(
        "Within each scenario folder, sort alphabetically: baselines first, then steered by rank."
    )


if __name__ == "__main__":
    main()
