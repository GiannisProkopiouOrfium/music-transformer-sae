#!/usr/bin/env python3
"""Convert top-ranked steered MIDIs to WAV for A/B listening comparison.

Default mode (``--paired``, on by default):
    Groups experiments into (method, concept) pairs — abrupt vs smooth —
    finds **the same songs at the same λ/α** across both modes, ranks by
    combined score, and converts both versions side-by-side so you can
    directly compare them.

    Output layout::

        top_wavs/
        └── sas_average_pitch/            ← one dir per (method, concept)
            ├── 01_Song_X_lpos2.00/
            │   ├── abrupt.wav
            │   ├── pulse_128.wav
            │   └── baseline.wav          ← (with --include_baseline)
            ├── 02_Song_Y_lneg1.00/
            │   ├── abrupt.wav
            │   ├── pulse_128.wav
            │   └── baseline.wav
            ...

Independent mode (``--no-paired``):
    Ranks each experiment independently and converts the top-N per experiment
    (the old behaviour).

Requires fluidsynth (``conda install -c conda-forge fluidsynth``).

Usage:
    # Paired A/B comparison (recommended) — top 5 song pairs
    python metrics_evaluation/convert_top_wavs.py \\
        --experiment_dir exp/sod/smooth_final_verification \\
        --top_n 5 --include_baseline

    # Independent mode
    python metrics_evaluation/convert_top_wavs.py \\
        --experiment_dir exp/sod/smooth_final_verification \\
        --top_n 5 --no-paired

    # Then upload to S3
    aws s3 sync exp/sod/smooth_final_verification/top_wavs/ \\
        s3://YOUR_BUCKET/smooth_steering_wavs/ --exclude "*.npy"
"""

import argparse
import json
import logging
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "mmt"))
import representation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


def load_encoding() -> dict:
    """Load MMT encoding dictionary."""
    enc_path = PROJECT_ROOT / "data" / "sod" / "processed" / "notes" / "encoding.json"
    return representation.load_encoding(str(enc_path))


def detect_concept(exp_name: str) -> str:
    """Detect whether experiment targets pitch or duration."""
    if "pitch" in exp_name:
        return "pitch"
    elif "duration" in exp_name:
        return "duration"
    return "unknown"


def detect_method(exp_name: str) -> str:
    """Detect SAS vs DM."""
    if exp_name.startswith("sas_"):
        return "sas"
    elif exp_name.startswith("dm_"):
        return "dm"
    return "unknown"


def collect_sample_metrics(exp_dir: pathlib.Path) -> list:
    """Collect all sample_metrics.json entries from an experiment dir.

    Falls back to conditioned_results.json for DM experiments where
    sample_metrics.json was overwritten (only last alpha survives).
    """
    all_samples = []
    for mf in sorted(exp_dir.rglob("sample_metrics.json")):
        with open(mf) as f:
            entries = json.load(f)
        for entry in entries:
            entry["_metrics_file"] = str(mf)
        all_samples.extend(entries)

    # Fallback: reconstruct from conditioned_results.json for DM experiments
    # where sample_metrics.json only has the last alpha per category
    cr_file = exp_dir / "conditioned_results.json"
    if cr_file.exists():
        with open(cr_file) as f:
            cr = json.load(f)
        results = cr.get("results", [])

        # Check which (category, alpha) combos we already have
        existing = set()
        for s in all_samples:
            lam = s.get("lambda", s.get("alpha", 0))
            existing.add((s.get("category", ""), round(lam, 4)))

        # Group results by (category, alpha) to track per-group index
        from collections import defaultdict

        group_counters = defaultdict(int)
        for r in results:
            alpha = r.get("alpha", r.get("lambda", 0))
            cat = r.get("category", "")
            group_key = (cat, round(alpha, 4))
            idx = group_counters[group_key]
            group_counters[group_key] += 1

            if group_key in existing:
                continue

            # Reconstruct a sample-like entry from the results dict
            deg = r.get("degradation", {})
            entry = {
                "song_name": r.get("song_name", ""),
                "category": cat,
                "alpha": alpha,
                "pitch_change": r.get("pitch_change", 0),
                "duration_change": r.get("duration_change", 0),
                "total_degradation": (
                    deg.get("total_degradation", np.nan)
                    if isinstance(deg, dict)
                    else np.nan
                ),
                "pitch_class_entropy": r.get("quality_metrics", {}).get(
                    "pitch_class_entropy", np.nan
                ),
                "scale_consistency": r.get("quality_metrics", {}).get(
                    "scale_consistency", np.nan
                ),
                "groove_consistency": r.get("quality_metrics", {}).get(
                    "groove_consistency", np.nan
                ),
                "_from_conditioned_results": True,
            }
            # DM .npy pattern: category/sample_category_{cat}_alpha_{alpha}_{idx}.npy
            npy_path = exp_dir / cat / f"sample_category_{cat}_alpha_{alpha}_{idx}.npy"
            if npy_path.exists():
                entry["filepath"] = str(npy_path)
            else:
                # Try .mid fallback
                mid_path = npy_path.with_suffix(".mid")
                if mid_path.exists():
                    entry["filepath"] = str(mid_path)

            all_samples.append(entry)

    return all_samples


def rank_samples(samples: list, concept: str, filter_lams: set = None) -> list:
    """Rank samples by effectiveness: |concept_change| / degradation.

    Higher = better (more steering effect per unit of quality loss).
    Only considers steered samples (lambda != 0 / alpha != 0).
    """
    ranked = []
    for s in samples:
        # Skip baseline
        lam = s.get("lambda", s.get("alpha", 0))
        if abs(lam) < 0.01:
            continue

        # Filter to specific lambdas if requested
        if filter_lams is not None and not any(
            abs(lam - fl) < 0.01 for fl in filter_lams
        ):
            continue

        change_key = f"{concept}_change"
        change = abs(s.get(change_key, 0))
        degrad = s.get("total_degradation", 999)

        if degrad <= 0 or np.isnan(degrad):
            degrad = 0.01  # avoid div by zero

        score = change / degrad
        s["_rank_score"] = score
        s["_abs_change"] = change
        ranked.append(s)

    ranked.sort(key=lambda x: x["_rank_score"], reverse=True)
    return ranked


def convert_to_wav(
    npy_path: pathlib.Path,
    wav_path: pathlib.Path,
    encoding: dict,
) -> bool:
    """Convert .npy tokens → WAV via muspy."""
    try:
        tokens = np.load(npy_path)
        music = representation.decode(tokens, encoding)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        music.write_audio(str(wav_path))
        return True
    except Exception as e:
        logger.error(f"Failed: {npy_path.name}: {e}")
        return False


def midi_to_wav(
    mid_path: pathlib.Path,
    wav_path: pathlib.Path,
) -> bool:
    """Convert .mid → WAV using muspy."""
    try:
        import muspy

        music = muspy.read(str(mid_path))
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        music.write_audio(str(wav_path))
        return True
    except Exception as e:
        logger.error(f"Failed MIDI→WAV: {mid_path.name}: {e}")
        return False


def parse_experiment_name(exp_name: str) -> tuple:
    """Parse experiment name into (method, concept, mode).

    E.g. 'sas_average_pitch_pulse_128' -> ('sas', 'average_pitch', 'pulse_128')
         'dm_average_duration_abrupt'   -> ('dm', 'average_duration', 'abrupt')
         'dm_average_pitch_ramp_up_32'  -> ('dm', 'average_pitch', 'ramp_up_32')
    """
    parts = exp_name.split("_", 1)
    method = parts[0]
    rest = parts[1] if len(parts) > 1 else ""

    for concept in ("average_pitch", "average_duration"):
        if concept in rest:
            mode = rest.replace(concept + "_", "")
            return method, concept, mode

    return method, rest, "unknown"


def run_paired(args, encoding):
    """Paired A/B comparison: same songs across abrupt vs smooth."""
    total_converted = 0

    # 1. Discover all experiments and group by (method, concept)
    groups = {}  # (method, concept) -> {mode: (exp_dir, samples)}
    for exp_subdir in sorted(args.experiment_dir.iterdir()):
        if not exp_subdir.is_dir() or exp_subdir.name in (
            "fmd_workspace",
            "top_wavs",
        ):
            continue

        exp_name = exp_subdir.name
        method, concept, mode = parse_experiment_name(exp_name)

        samples = collect_sample_metrics(exp_subdir)
        if not samples:
            logger.warning(f"  No sample_metrics.json in {exp_name} — skipping")
            continue

        # Filter by method if requested
        if args._filter_methods and method not in args._filter_methods:
            continue

        key = (method, concept)
        if key not in groups:
            groups[key] = {}
        groups[key][mode] = (exp_subdir, samples)

    # 2. For each group, find paired songs and rank
    for (method, concept), modes in sorted(groups.items()):
        concept_key = "pitch" if "pitch" in concept else "duration"
        group_label = f"{method}_{concept}"

        # Need at least abrupt + one smooth mode
        if "abrupt" not in modes:
            logger.warning(f"  No abrupt baseline for {group_label} — skipping pairing")
            continue
        smooth_modes = {m: v for m, v in modes.items() if m != "abrupt"}

        # Filter smooth modes if requested
        if args._filter_modes:
            smooth_modes = {
                m: v for m, v in smooth_modes.items() if m in args._filter_modes
            }
        if not smooth_modes:
            logger.warning(f"  No smooth variant for {group_label} — skipping pairing")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(
            f"Paired comparison: {group_label}  "
            f"(abrupt vs {', '.join(smooth_modes.keys())})"
        )

        _, abrupt_samples = modes["abrupt"]

        # For each smooth mode, find matching (song_name, lambda/alpha) pairs
        for smooth_mode, (smooth_dir, smooth_samples) in sorted(smooth_modes.items()):
            logger.info(f"\n  --- abrupt vs {smooth_mode} ---")

            # Index samples by (song_name, strength)
            strength_key = "lambda" if method == "sas" else "alpha"

            abrupt_index = {}
            for s in abrupt_samples:
                lam = s.get("lambda", s.get("alpha", 0))
                if abs(lam) < 0.01:
                    continue  # skip baseline
                if args._filter_lams is not None and not any(
                    abs(lam - fl) < 0.01 for fl in args._filter_lams
                ):
                    continue
                k = (s["song_name"], round(lam, 2))
                abrupt_index[k] = s

            smooth_index = {}
            for s in smooth_samples:
                lam = s.get("lambda", s.get("alpha", 0))
                if abs(lam) < 0.01:
                    continue
                if args._filter_lams is not None and not any(
                    abs(lam - fl) < 0.01 for fl in args._filter_lams
                ):
                    continue
                k = (s["song_name"], round(lam, 2))
                smooth_index[k] = s

            # Find common (song, lambda) pairs
            common_keys = set(abrupt_index.keys()) & set(smooth_index.keys())
            logger.info(
                f"  Found {len(common_keys)} matching (song, {strength_key}) pairs"
            )

            if not common_keys:
                logger.warning("  No matching pairs found!")
                continue

            # Rank by average score across both modes
            change_key = f"{concept_key}_change"
            paired_ranked = []
            for k in common_keys:
                a = abrupt_index[k]
                s = smooth_index[k]

                a_change = abs(a.get(change_key, 0))
                s_change = abs(s.get(change_key, 0))
                a_deg = max(a.get("total_degradation", 999), 0.01)
                s_deg = max(s.get("total_degradation", 999), 0.01)
                if np.isnan(a_deg):
                    a_deg = 0.01
                if np.isnan(s_deg):
                    s_deg = 0.01

                avg_score = ((a_change / a_deg) + (s_change / s_deg)) / 2
                paired_ranked.append((k, avg_score, a, s))

            paired_ranked.sort(key=lambda x: x[1], reverse=True)
            top_pairs = paired_ranked[: args.top_n]

            # Convert each pair
            pair_dir = args.output_dir / group_label
            for i, ((song, lam), avg_score, a_sample, s_sample) in enumerate(top_pairs):
                lam_str = f"{'pos' if lam >= 0 else 'neg'}{abs(lam):.2f}"
                song_dir = pair_dir / f"{i+1:02d}_{song}_l{lam_str}"

                # Abrupt WAV
                a_wav = song_dir / "abrupt.wav"
                a_npy = (
                    pathlib.Path(a_sample["filepath"])
                    if "filepath" in a_sample
                    else None
                )

                success_a = False
                if a_npy is None:
                    logger.warning(f"  No filepath for abrupt sample {song} α/λ={lam}")
                elif a_wav.exists():
                    success_a = True
                    logger.info(f"  Skipping (exists): {a_wav}")
                else:
                    a_mid = a_npy.with_suffix(".mid")
                    if a_npy.exists():
                        success_a = convert_to_wav(a_npy, a_wav, encoding)
                    elif a_mid.exists():
                        success_a = midi_to_wav(a_mid, a_wav)

                # Smooth WAV
                s_wav = song_dir / f"{smooth_mode}.wav"
                s_npy = (
                    pathlib.Path(s_sample["filepath"])
                    if "filepath" in s_sample
                    else None
                )

                success_s = False
                if s_npy is None:
                    logger.warning(f"  No filepath for smooth sample {song} α/λ={lam}")
                elif s_wav.exists():
                    success_s = True
                    logger.info(f"  Skipping (exists): {s_wav}")
                else:
                    s_mid = s_npy.with_suffix(".mid")
                    if s_npy.exists():
                        success_s = convert_to_wav(s_npy, s_wav, encoding)
                    elif s_mid.exists():
                        success_s = midi_to_wav(s_mid, s_wav)

                if success_a:
                    total_converted += 1
                if success_s:
                    total_converted += 1

                a_deg = a_sample.get("total_degradation", 0)
                s_deg = s_sample.get("total_degradation", 0)
                a_chg = abs(a_sample.get(change_key, 0))
                s_chg = abs(s_sample.get(change_key, 0))

                logger.info(
                    f"  #{i+1}: {song} {strength_key}={lam:+.2f}  "
                    f"abrupt(|Δ|={a_chg:.1f}, deg={a_deg:.1f}) vs "
                    f"{smooth_mode}(|Δ|={s_chg:.1f}, deg={s_deg:.1f})"
                )

                # Baseline for same song
                if args.include_baseline:
                    baseline_wav = song_dir / "baseline.wav"
                    if baseline_wav.exists():
                        logger.info(f"  Skipping (exists): {baseline_wav}")
                        total_converted += 1
                    else:
                        # Find baseline (lambda=0 / alpha=0) for this song in abrupt exp
                        for bs in abrupt_samples:
                            bl = bs.get("lambda", bs.get("alpha", 0))
                            if abs(bl) < 0.01 and bs["song_name"] == song:
                                b_npy = pathlib.Path(bs["filepath"])
                                b_mid = b_npy.with_suffix(".mid")
                                if b_npy.exists():
                                    if convert_to_wav(b_npy, baseline_wav, encoding):
                                        total_converted += 1
                                elif b_mid.exists():
                                    if midi_to_wav(b_mid, baseline_wav):
                                        total_converted += 1
                                break

                # Save pair metadata
                meta = {
                    "song": song,
                    "strength": lam,
                    "concept": concept_key,
                    "method": method,
                    "avg_rank_score": avg_score,
                    "abrupt": {
                        "change": a_chg,
                        "degradation": a_deg,
                        "source": str(a_npy) if a_npy else "",
                    },
                    smooth_mode: {
                        "change": s_chg,
                        "degradation": s_deg,
                        "source": str(s_npy) if s_npy else "",
                    },
                }
                meta_path = song_dir / "pair_info.json"
                meta_path.parent.mkdir(parents=True, exist_ok=True)
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

    return total_converted


def run_independent(args, encoding):
    """Independent mode: rank each experiment separately."""
    total_converted = 0

    for exp_subdir in sorted(args.experiment_dir.iterdir()):
        if not exp_subdir.is_dir() or exp_subdir.name in (
            "fmd_workspace",
            "top_wavs",
        ):
            continue

        exp_name = exp_subdir.name
        concept = detect_concept(exp_name)
        method = detect_method(exp_name)

        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {exp_name} (concept={concept}, method={method})")

        samples = collect_sample_metrics(exp_subdir)
        if not samples:
            logger.warning("  No sample_metrics.json found — trying MIDI fallback")
            midis = sorted(exp_subdir.rglob("*.mid"))
            for mid in midis[: args.top_n]:
                wav_out = args.output_dir / exp_name / mid.name.replace(".mid", ".wav")
                if midi_to_wav(mid, wav_out):
                    total_converted += 1
            continue

        ranked = rank_samples(samples, concept, filter_lams=args._filter_lams)
        logger.info(f"  {len(ranked)} steered samples, selecting top {args.top_n}")

        top = ranked[: args.top_n]
        exp_out = args.output_dir / exp_name
        converted_songs = set()

        for i, s in enumerate(top):
            lam = s.get("lambda", s.get("alpha", 0))
            song = s["song_name"]
            change = s["_abs_change"]
            degrad = s.get("total_degradation", 0)
            score = s["_rank_score"]

            npy_path = pathlib.Path(s["filepath"])
            mid_path = npy_path.with_suffix(".mid")

            lam_str = f"{'pos' if lam >= 0 else 'neg'}{abs(lam):.2f}"
            wav_name = f"{i+1:02d}_steered_{song}_l{lam_str}_d{degrad:.1f}.wav"
            wav_path = exp_out / wav_name

            success = False
            if wav_path.exists():
                success = True
                logger.info(f"  Skipping (exists): {wav_path}")
            elif npy_path.exists():
                success = convert_to_wav(npy_path, wav_path, encoding)
            elif mid_path.exists():
                success = midi_to_wav(mid_path, wav_path)
            else:
                logger.warning(f"  Missing: {npy_path}")

            if success:
                total_converted += 1
                converted_songs.add(song)
                logger.info(
                    f"  #{i+1}: {song} λ/α={lam:+.2f} "
                    f"|Δ{concept}|={change:.1f} deg={degrad:.1f} "
                    f"score={score:.2f}"
                )

            if args.include_baseline and song not in converted_songs:
                for bs in samples:
                    bl = bs.get("lambda", bs.get("alpha", 0))
                    if abs(bl) < 0.01 and bs["song_name"] == song:
                        bnpy = pathlib.Path(bs["filepath"])
                        bmid = bnpy.with_suffix(".mid")
                        bwav = exp_out / f"00_baseline_{song}.wav"
                        if bnpy.exists():
                            convert_to_wav(bnpy, bwav, encoding)
                        elif bmid.exists():
                            midi_to_wav(bmid, bwav)
                        break

    return total_converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert top-ranked steered samples to WAV for A/B comparison"
    )
    parser.add_argument(
        "--experiment_dir",
        type=pathlib.Path,
        default=PROJECT_ROOT / "exp" / "sod" / "smooth_final_verification",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Output dir (default: <experiment_dir>/top_wavs)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Top N samples per (method, concept) pair to convert",
    )
    parser.add_argument(
        "--include_baseline",
        action="store_true",
        help="Also convert the baseline (λ=0/α=0) for the same songs",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        default=True,
        help="Paired A/B mode: same songs across abrupt vs smooth (default)",
    )
    parser.add_argument(
        "--no-paired",
        dest="paired",
        action="store_false",
        help="Independent mode: rank each experiment separately",
    )
    parser.add_argument(
        "--filter_lambdas",
        type=str,
        default=None,
        help="Comma-separated lambda/alpha values to include (e.g. '1.0,-1.0,2.0'). Others are excluded.",
    )
    parser.add_argument(
        "--filter_modes",
        type=str,
        default=None,
        help="Comma-separated smooth modes to include (e.g. 'warmup_hold_32'). Others are excluded.",
    )
    parser.add_argument(
        "--filter_methods",
        type=str,
        default=None,
        help="Comma-separated methods to include (e.g. 'dm' or 'dm,sas'). Others are excluded.",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.experiment_dir / "top_wavs"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Parse filter_lambdas into a set of floats
    filter_lams = None
    if args.filter_lambdas:
        filter_lams = {float(x.strip()) for x in args.filter_lambdas.split(",")}
        logger.info(f"Filtering to lambda/alpha values: {sorted(filter_lams)}")
    args._filter_lams = filter_lams

    args._filter_modes = None
    if args.filter_modes:
        args._filter_modes = {m.strip() for m in args.filter_modes.split(",")}
        logger.info(f"Filtering to smooth modes: {sorted(args._filter_modes)}")

    args._filter_methods = None
    if args.filter_methods:
        args._filter_methods = {
            m.strip().lower() for m in args.filter_methods.split(",")
        }
        logger.info(f"Filtering to methods: {sorted(args._filter_methods)}")

    encoding = load_encoding()

    if args.paired:
        total_converted = run_paired(args, encoding)
    else:
        total_converted = run_independent(args, encoding)

    logger.info(f"\n{'='*60}")
    logger.info(f"DONE: {total_converted} WAV files total")
    logger.info(f"Output: {args.output_dir}")

    summary_path = args.output_dir / "conversion_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Total WAVs: {total_converted}\n")
        f.write(f"Output: {args.output_dir}\n")
        f.write(f"Mode: {'paired' if args.paired else 'independent'}\n")


if __name__ == "__main__":
    main()
