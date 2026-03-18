#!/usr/bin/env python3
"""
Convert Best Beatwise vs Abrupt Pairs to WAV for A/B Listening.

Reads comparison_results.json from the beatwise comparison experiment,
groups by category (pitch_up, pitch_down, dur_up, dur_down), ranks pairs
by |change| / degradation, and converts the top-N to WAV side-by-side.

Output layout::

    best_wavs/
    ├── pitch_up/
    │   ├── 01_Musicalion-2568_lam+1.50/
    │   │   ├── abrupt.wav
    │   │   └── beatwise.wav
    │   ├── 02_Musicalion-3512_lam+1.00/
    │   │   ├── abrupt.wav
    │   │   └── beatwise.wav
    │   ...
    ├── pitch_down/
    │   ├── 01_Kunstderfuge-389_lam-1.00/
    │   │   ├── abrupt.wav
    │   │   └── beatwise.wav
    │   ...
    ├── dur_up/
    │   ...
    └── dur_down/
        ...

Usage::

    # Convert top 5 per category
    python sparse_steering/convert_beatwise_wavs.py --top_n 5

    # Then upload to S3
    aws s3 sync exp/sod/sparse_steering/beatwise_comparison/best_wavs/ \\
        s3://YOUR-BUCKET/beatwise_comparison/
"""

import argparse
import json
import logging
import pathlib
import sys
from collections import defaultdict

import numpy as np

REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "mmt"))
import representation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = (
    REPO_ROOT / "exp" / "sod" / "sparse_steering" / "beatwise_comparison"
)


def load_encoding() -> dict:
    enc_path = REPO_ROOT / "data" / "sod" / "processed" / "notes" / "encoding.json"
    return representation.load_encoding(str(enc_path))


def mid_to_wav(mid_path: pathlib.Path, wav_path: pathlib.Path) -> bool:
    try:
        import muspy

        music = muspy.read(str(mid_path))
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        music.write_audio(str(wav_path))
        return True
    except Exception as e:
        logger.error(f"Failed MIDI→WAV: {mid_path}: {e}")
        return False


def npy_to_wav(npy_path: pathlib.Path, wav_path: pathlib.Path, encoding: dict) -> bool:
    try:
        tokens = np.load(npy_path)
        music = representation.decode(tokens, encoding)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        music.write_audio(str(wav_path))
        return True
    except Exception as e:
        logger.error(f"Failed NPY→WAV: {npy_path}: {e}")
        return False


def classify_category(result: dict) -> str:
    """Map a result entry to one of: pitch_up, pitch_down, dur_up, dur_down."""
    concept = result.get("concept", "")
    lam = result.get("lambda", 0)
    if "pitch" in concept:
        return "pitch_up" if lam > 0 else "pitch_down"
    else:
        return "dur_up" if lam > 0 else "dur_down"


def main():
    parser = argparse.ArgumentParser(
        description="Convert best beatwise vs abrupt pairs to WAV"
    )
    parser.add_argument(
        "--results_dir",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing comparison_results.json and the MIDI/NPY files",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Top N pairs per category to convert",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=None,
        help="Output dir (default: <results_dir>/best_wavs)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.results_dir / "best_wavs"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_file = args.results_dir / "comparison_results.json"
    if not results_file.exists():
        logger.error(f"Not found: {results_file}")
        sys.exit(1)

    with open(results_file) as f:
        all_results = json.load(f)
    logger.info(f"Loaded {len(all_results)} results from {results_file}")

    encoding = load_encoding()

    # Group by (category, song_name, lambda) → {mode: result}
    pairs = defaultdict(dict)
    for r in all_results:
        if r.get("lambda", 0) == 0:
            continue
        cat = classify_category(r)
        key = (cat, r["song_name"], r["lambda"])
        pairs[key][r["mode"]] = r

    # Keep only complete pairs (both abrupt and beatwise present)
    complete_pairs = {
        k: v for k, v in pairs.items() if "abrupt" in v and "beatwise" in v
    }
    logger.info(f"Found {len(complete_pairs)} complete A/B pairs")

    # Group by category, rank by average score
    by_category = defaultdict(list)
    for (cat, song, lam), modes in complete_pairs.items():
        a = modes["abrupt"]
        b = modes["beatwise"]

        is_dur = "dur" in cat
        change_key = "duration_change" if is_dur else "pitch_change"

        a_change = abs(a.get(change_key, 0))
        b_change = abs(b.get(change_key, 0))
        a_deg = max(a.get("total_degradation", 999), 0.01)
        b_deg = max(b.get("total_degradation", 999), 0.01)
        if np.isnan(a_deg):
            a_deg = 0.01
        if np.isnan(b_deg):
            b_deg = 0.01

        # Score: average of |change|/degradation across both modes
        avg_score = ((a_change / a_deg) + (b_change / b_deg)) / 2

        # Also check both are "successful" (moved in expected direction)
        if lam > 0:
            a_ok = a.get(change_key, 0) > 0
            b_ok = b.get(change_key, 0) > 0
        else:
            a_ok = a.get(change_key, 0) < 0
            b_ok = b.get(change_key, 0) < 0

        by_category[cat].append(
            {
                "score": avg_score,
                "song": song,
                "lambda": lam,
                "abrupt": a,
                "beatwise": b,
                "a_change": a.get(change_key, 0),
                "b_change": b.get(change_key, 0),
                "a_deg": a.get("total_degradation", 0),
                "b_deg": b.get("total_degradation", 0),
                "both_successful": a_ok and b_ok,
            }
        )

    # Sort: prioritize pairs where both modes succeed, then by score
    for cat in by_category:
        by_category[cat].sort(
            key=lambda x: (x["both_successful"], x["score"]), reverse=True
        )

    total_converted = 0
    category_labels = {
        "pitch_up": "Pitch: Low → High",
        "pitch_down": "Pitch: High → Low",
        "dur_up": "Duration: Short → Long",
        "dur_down": "Duration: Long → Short",
    }

    print(f"\n{'=' * 80}")
    print("BEAT-WISE vs ABRUPT — BEST PAIRS FOR LISTENING")
    print(f"{'=' * 80}")

    for cat in ("pitch_up", "pitch_down", "dur_up", "dur_down"):
        ranked = by_category.get(cat, [])
        if not ranked:
            logger.warning(f"No pairs for {cat}")
            continue

        top = ranked[: args.top_n]
        cat_dir = args.output_dir / cat

        print(f"\n{'─' * 70}")
        print(f"  {category_labels[cat]}  (top {len(top)})")
        print(f"{'─' * 70}")

        for i, p in enumerate(top):
            lam = p["lambda"]
            song = p["song"]
            lam_str = f"{'pos' if lam >= 0 else 'neg'}{abs(lam):.2f}"
            folder_name = f"{i+1:02d}_{song}_lam{lam:+.2f}"
            pair_dir = cat_dir / folder_name

            ok_marker = "✓" if p["both_successful"] else "✗"
            print(
                f"  {i+1}. [{ok_marker}] {song}  λ={lam:+.2f}  "
                f"score={p['score']:.1f}"
            )
            print(f"     ABRUPT:   Δ={p['a_change']:+.1f}  deg={p['a_deg']:.2f}")
            print(f"     BEATWISE: Δ={p['b_change']:+.1f}  deg={p['b_deg']:.2f}")

            # Convert abrupt
            a_filepath = pathlib.Path(p["abrupt"].get("filepath", ""))
            a_mid = a_filepath.with_suffix(".mid")
            a_wav = pair_dir / "abrupt.wav"

            if a_wav.exists():
                logger.info(f"  Exists: {a_wav}")
                total_converted += 1
            elif a_mid.exists():
                if mid_to_wav(a_mid, a_wav):
                    total_converted += 1
            elif a_filepath.exists() and a_filepath.suffix == ".npy":
                if npy_to_wav(a_filepath, a_wav, encoding):
                    total_converted += 1
            else:
                logger.warning(f"  Not found: {a_mid} or {a_filepath}")

            # Convert beatwise
            b_filepath = pathlib.Path(p["beatwise"].get("filepath", ""))
            b_mid = b_filepath.with_suffix(".mid")
            b_wav = pair_dir / "beatwise.wav"

            if b_wav.exists():
                logger.info(f"  Exists: {b_wav}")
                total_converted += 1
            elif b_mid.exists():
                if mid_to_wav(b_mid, b_wav):
                    total_converted += 1
            elif b_filepath.exists() and b_filepath.suffix == ".npy":
                if npy_to_wav(b_filepath, b_wav, encoding):
                    total_converted += 1
            else:
                logger.warning(f"  Not found: {b_mid} or {b_filepath}")

            # Save pair metadata
            meta = {
                "rank": i + 1,
                "category": cat,
                "song": song,
                "lambda": lam,
                "score": p["score"],
                "both_successful": p["both_successful"],
                "abrupt_change": p["a_change"],
                "abrupt_degradation": p["a_deg"],
                "beatwise_change": p["b_change"],
                "beatwise_degradation": p["b_deg"],
            }
            pair_dir.mkdir(parents=True, exist_ok=True)
            with open(pair_dir / "pair_info.json", "w") as f:
                json.dump(meta, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"  Converted {total_converted} WAV files → {args.output_dir}")
    print(f"{'=' * 80}")
    print(f"\nUpload to S3:")
    print(f"  aws s3 sync {args.output_dir}/ s3://YOUR-BUCKET/beatwise_comparison/")


if __name__ == "__main__":
    main()
