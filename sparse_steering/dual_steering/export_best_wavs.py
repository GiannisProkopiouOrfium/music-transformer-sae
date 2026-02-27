#!/usr/bin/env python3
"""Rank steered results by control strength + quality, export top N as WAV.

For each exported sample, also exports its baseline (lp=0, ld=0) for A/B comparison.

Usage:
    python sparse_steering/dual_steering/export_best_wavs.py \
        --results_json exp/sod/sparse_steering/dual_steering/conditioned_ek2/conditioned_results.json \
        --npy_root exp/sod/sparse_steering/dual_steering/conditioned_ek2 \
        --out_dir exp/sod/sparse_steering/dual_steering/ranked_wavs \
        --top 20
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
    parser.add_argument("--top", type=int, default=20, help="How many to export")
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

    # Score and rank
    for r in results:
        r["_score"] = compute_score(r, args.w_control, args.w_degrad)
    results.sort(key=lambda r: r["_score"], reverse=True)

    top = results[: args.top]

    # Print ranking
    print(
        f"\n{'Rank':>4}  {'Score':>6}  {'Deg':>5}  {'dPitch':>7}  {'dDur':>7}  "
        f"{'lp':>5}  {'ld':>5}  {'Scenario':<16}  Song"
    )
    print("-" * 95)
    for i, r in enumerate(top):
        sc = SCENARIO_SHORT.get(r["scenario"], r["scenario"][:16])
        song = r["song_name"].split("/")[-1]
        print(
            f"{i+1:4d}  {r['_score']:6.1f}  {r['degradation']['total_degradation']:5.2f}  "
            f"{r['pitch_delta']:+7.1f}  {r['duration_delta']:+7.1f}  "
            f"{r['lambda_pitch']:+5.2f}  {r['lambda_duration']:+5.2f}  "
            f"{sc:<16}  {song}"
        )

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

    print(f"\nConverting top {len(top)} steered + baselines...")
    for i, r in enumerate(top):
        sc = SCENARIO_SHORT.get(r["scenario"], r["scenario"][:16])
        song = r["song_name"].split("/")[-1]
        deg = r["degradation"]["total_degradation"]

        # Steered file
        steered_name = (
            f"{i+1:02d}_steered_{sc}_{song}"
            f"_lp{r['lambda_pitch']:+.2f}_ld{r['lambda_duration']:+.2f}"
            f"_deg{deg:.2f}"
        )
        steered_npy = npy_path_for(args.npy_root, r)
        if steered_npy.exists():
            tokens = np.load(steered_npy)
            music = representation.decode(tokens, encoding)
            music.write(str(args.out_dir / f"{steered_name}.mid"))
            if not args.skip_wav:
                music.write_audio(str(args.out_dir / f"{steered_name}.wav"))
            print(f"  {steered_name}")
        else:
            print(f"  MISSING: {steered_npy}")

        # Baseline (one per song+scenario)
        baseline_key = (r["scenario"], r["song_name"])
        if baseline_key not in exported_baselines:
            exported_baselines.add(baseline_key)
            baseline_name = f"00_baseline_{sc}_{song}"
            baseline_npy = baseline_npy_path_for(args.npy_root, r)
            if baseline_npy.exists():
                tokens = np.load(baseline_npy)
                music = representation.decode(tokens, encoding)
                music.write(str(args.out_dir / f"{baseline_name}.mid"))
                if not args.skip_wav:
                    music.write_audio(str(args.out_dir / f"{baseline_name}.wav"))
                print(f"  {baseline_name} (baseline)")
            else:
                print(f"  BASELINE MISSING: {baseline_npy}")

    print(f"\nDone! Files in: {args.out_dir}")
    print("Naming: 00_baseline_* = unsteered, NN_steered_* = ranked by score")
    print("Sort alphabetically to get baselines first, then steered in rank order.")


if __name__ == "__main__":
    main()
