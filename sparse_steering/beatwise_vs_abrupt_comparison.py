#!/usr/bin/env python3
"""
Beat-wise Smooth Steering vs Abrupt Steering Comparison.

Generates paired A/B samples: for each song × λ, produces both an
abrupt-steered version (constant λ from token 1) and a beat-wise
smooth-steered version (gradual ramp over musical beats — constant
time intervals regardless of token density).

Categories
----------
- Pitch:    low → high  (positive λ on low-pitch songs)
            high → low  (negative λ on high-pitch songs)
- Duration: short → long (positive λ on short-duration songs)
            long → short (negative λ on long-duration songs)

Outputs
-------
For each category × λ × mode (abrupt / beat-wise):
  - .npy  (full token sequence: conditioning + generated)
  - .mid  (MIDI for listening / conversion)

A paired comparison table and JSON results are saved at the end.
Best samples are ranked for easy conversion to WAV.

Usage
-----
    python sparse_steering/beatwise_vs_abrupt_comparison.py \
        --n_songs 5 --lambdas 0.75,1.0,1.5,-0.75,-1.0,-1.5

    # Quick test
    python sparse_steering/beatwise_vs_abrupt_comparison.py \
        --n_songs 3 --lambdas 1.0,-1.0
"""

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add project paths
REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import music_x_transformers
import representation
import utils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = REPO_ROOT / "exp" / "sod" / "ape" / "checkpoints" / "best_model.pt"
DEFAULT_TRAIN_ARGS = REPO_ROOT / "exp" / "sod" / "ape" / "train-args.json"
DEFAULT_ENCODING = REPO_ROOT / "data" / "sod" / "processed" / "notes" / "encoding.json"
DEFAULT_SAE_DIR = REPO_ROOT / "exp" / "sod" / "sparse_steering" / "sae_checkpoints"
DEFAULT_SAS_DIR = REPO_ROOT / "exp" / "sod" / "sparse_steering" / "sas_vectors"
DEFAULT_NOTES_DIR = REPO_ROOT / "data" / "sod" / "processed" / "notes"

GROUND_TRUTH_METRICS = {
    "pitch_class_entropy": 2.974,
    "scale_consistency": 92.26,
    "groove_consistency": 93.05,
}

CONDITIONING_BEATS = 16
CONTINUATION_LEN = 256
LAYERS_TO_STEER = [10]


# ── Import SAS infrastructure ───────────────────────────────────────────────

from steered_generator_sas import (
    load_model,
    load_sae_models,
    load_sas_vectors,
    register_steering_hooks,
    remove_hooks,
    extract_pitches_from_tokens,
    extract_durations_from_tokens,
)
from smooth_steering_sas import register_smooth_steering_hooks
from conditioned_evaluator_sas import (
    find_extreme_pitch_songs,
    find_extreme_duration_songs,
    load_song_tokens,
    extract_conditioning_prefix,
    evaluate_quality_metrics,
    calculate_degradation,
    measure_pitch_from_tokens,
)


# ── Core generation ──────────────────────────────────────────────────────────


def generate_one(
    model,
    encoding: dict,
    sae_models: dict,
    sas_vectors: dict,
    concept: str,
    device: torch.device,
    filepath: pathlib.Path,
    lam: float,
    mode: str,  # "abrupt" or "beatwise"
    n_ramp_beats: int,
    conditioning_beats: int,
    continuation_len: int,
) -> Tuple[np.ndarray, dict]:
    """Generate a single steered continuation.

    Returns (full_token_sequence, metrics_dict).
    """
    tokens = load_song_tokens(filepath, encoding)
    conditioning = extract_conditioning_prefix(tokens, conditioning_beats).to(device)
    eos = encoding["type_code_map"]["end-of-song"]

    if mode == "abrupt":
        handles = register_steering_hooks(
            model, sae_models, sas_vectors, concept, lam, LAYERS_TO_STEER
        )
    else:
        # Beat-wise smooth ramp: gradual mode with beat_mode=True
        # Uses "warmup_hold" — cosine S-curve ramp over n_ramp_beats beats,
        # then holds at full λ for the rest of the piece.
        handles = register_smooth_steering_hooks(
            model,
            sae_models,
            sas_vectors,
            concept,
            steering_strength=lam,
            layers_to_steer=LAYERS_TO_STEER,
            mode="warmup_hold",
            schedule="cosine",
            n_ramp=n_ramp_beats,
            beat_mode=True,
        )

    try:
        with torch.no_grad():
            generated = model.generate(
                conditioning.clone(),
                continuation_len,
                eos_token=eos,
                temperature=1.0,
                filter_logits_fn="top_k",
                filter_thres=0.9,
                monotonicity_dim=("type", "beat"),
            )
    finally:
        remove_hooks(handles)

    full_seq = torch.cat((conditioning, generated), 1).cpu().numpy()[0]
    gen_only = generated.cpu().numpy()[0]

    # Measure
    m = measure_pitch_from_tokens(gen_only, encoding)
    qm = evaluate_quality_metrics(full_seq, encoding)
    deg = calculate_degradation(qm, GROUND_TRUTH_METRICS)

    cond_pitch = m.get("mean", 0)
    cond_dur = m.get("duration_mean", 0)

    # Also compute conditioning values from original tokens
    from conditioned_evaluator_sas import (
        calculate_initial_pitch,
        calculate_initial_duration,
    )

    init_pitch = calculate_initial_pitch(tokens, encoding, conditioning_beats) or 0.0
    init_dur = calculate_initial_duration(tokens, encoding, conditioning_beats) or 0.0

    metrics = {
        "song_name": filepath.stem,
        "lambda": float(lam),
        "mode": mode,
        "init_pitch": float(init_pitch),
        "init_duration": float(init_dur),
        "gen_mean_pitch": float(m["mean"]),
        "gen_mean_duration": float(m["duration_mean"]),
        "gen_n_notes": int(m["n_notes"]),
        "pitch_change": float(m["mean"] - init_pitch),
        "duration_change": float(m["duration_mean"] - init_dur),
        "quality": qm,
        "degradation": deg,
        "total_degradation": float(deg.get("total_degradation", np.nan)),
    }

    return full_seq, metrics


# ── Main comparison loop ─────────────────────────────────────────────────────


def run_comparison(
    model,
    encoding: dict,
    sae_models: dict,
    device: torch.device,
    sas_dir: pathlib.Path,
    notes_dir: pathlib.Path,
    output_dir: pathlib.Path,
    n_songs: int,
    lambdas: List[float],
    n_ramp_beats: int,
    conditioning_beats: int,
    continuation_len: int,
):
    """Run full paired comparison: abrupt vs beat-wise for all categories."""
    all_results = []

    concepts = [
        ("average_pitch", "pitch", find_extreme_pitch_songs),
        ("average_duration", "duration", find_extreme_duration_songs),
    ]

    for concept, short_name, find_fn in concepts:
        logger.info(f"\n{'=' * 72}")
        logger.info(f"  Concept: {short_name.upper()}")
        logger.info(f"{'=' * 72}")

        # Load SAS vectors
        sas_path = sas_dir / f"{concept}_sas_vectors.pt"
        if not sas_path.exists():
            logger.error(f"SAS vectors not found: {sas_path}")
            continue
        sas_vectors = load_sas_vectors(sas_path)

        # Find extreme songs
        low_songs, high_songs = find_fn(
            notes_dir, encoding, n_songs, conditioning_beats
        )

        low_cat = f"low_{short_name}"
        high_cat = f"high_{short_name}"

        logger.info(f"  Low  songs ({low_cat}): {len(low_songs)}")
        logger.info(f"  High songs ({high_cat}): {len(high_songs)}")

        for lam in lambdas:
            # Direction-aware: positive λ → low songs, negative λ → high songs
            if lam > 0:
                songs = low_songs
                category = low_cat
            elif lam < 0:
                songs = high_songs
                category = high_cat
            else:
                # baseline at λ=0 — do both categories but just a few songs
                songs = low_songs[:2]
                category = low_cat

            for filepath, init_val in songs:
                # Deterministic seed per (song, λ) so abrupt & beatwise
                # share the same random state → identical conditioning output,
                # diverging only when the intervention strength differs.
                song_seed = hash((filepath.stem, lam)) % (2**31)

                for mode in ("abrupt", "beatwise"):
                    torch.manual_seed(song_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(song_seed)

                    logger.info(
                        f"  [{mode:>8}] {short_name} λ={lam:+.2f} "
                        f"| {filepath.stem} (init={init_val:.1f})"
                    )

                    full_seq, metrics = generate_one(
                        model,
                        encoding,
                        sae_models,
                        sas_vectors,
                        concept,
                        device,
                        filepath,
                        lam,
                        mode,
                        n_ramp_beats,
                        conditioning_beats,
                        continuation_len,
                    )
                    metrics["category"] = category
                    metrics["concept"] = concept

                    # Save .npy and .mid
                    lam_str = f"lambda_{'pos' if lam >= 0 else 'neg'}{abs(lam):.2f}"
                    save_dir = output_dir / short_name / category / lam_str / mode
                    save_dir.mkdir(parents=True, exist_ok=True)

                    npy_path = save_dir / f"{filepath.stem}.npy"
                    np.save(npy_path, full_seq)
                    metrics["filepath"] = str(npy_path)

                    try:
                        music = representation.decode(full_seq, encoding)
                        music.write(str(save_dir / f"{filepath.stem}.mid"))
                    except Exception as e:
                        logger.error(f"MIDI save error: {e}")

                    all_results.append(metrics)

    return all_results


# ── Analysis & reporting ─────────────────────────────────────────────────────


def analyze_results(results: List[dict]) -> dict:
    """Group results by (concept, category, lambda, mode) and compute stats."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        key = (r["concept"], r["category"], r["lambda"], r["mode"])
        groups[key].append(r)

    summary = {}
    for key, entries in sorted(groups.items()):
        concept, category, lam, mode = key
        is_dur = "duration" in concept

        change_key = "duration_change" if is_dur else "pitch_change"
        changes = [e[change_key] for e in entries if e["gen_n_notes"] > 0]
        degs = [
            e["total_degradation"]
            for e in entries
            if not np.isnan(e["total_degradation"])
        ]

        # Success: did steering move in expected direction?
        if lam > 0:
            successes = sum(1 for c in changes if c > 0)
        elif lam < 0:
            successes = sum(1 for c in changes if c < 0)
        else:
            successes = len(changes)

        summary[str(key)] = {
            "concept": concept,
            "category": category,
            "lambda": lam,
            "mode": mode,
            "n": len(entries),
            "success_rate": successes / len(changes) if changes else 0,
            "mean_change": float(np.mean(changes)) if changes else 0,
            "std_change": float(np.std(changes)) if len(changes) > 1 else 0,
            "mean_degradation": float(np.mean(degs)) if degs else float("nan"),
        }

    return summary


def print_comparison_table(results: List[dict]):
    """Print a formatted paired comparison table."""
    from collections import defaultdict

    # Group by (concept, category, lambda) → {mode: [results]}
    paired = defaultdict(lambda: defaultdict(list))
    for r in results:
        key = (r["concept"], r["category"], r["lambda"])
        paired[key][r["mode"]].append(r)

    # Print per-concept
    for concept in ("average_pitch", "average_duration"):
        short = "PITCH" if "pitch" in concept else "DURATION"
        is_dur = "duration" in concept
        change_key = "duration_change" if is_dur else "pitch_change"
        unit = "ticks" if is_dur else "ST"

        print(f"\n{'═' * 100}")
        print(f"  {short} STEERING: Abrupt vs Beat-wise Smooth")
        print(f"{'═' * 100}")
        print(
            f"  {'Category':<18} {'λ':>5}  {'Mode':<10}  {'N':>3}  "
            f"{'Success%':>8}  {'ΔMean':>8}  {'Deg':>6}  {'Best Song':<20}"
        )
        print(f"  {'-' * 92}")

        concept_results = [r for r in results if r["concept"] == concept]
        categories = sorted(set(r["category"] for r in concept_results))

        for cat in categories:
            cat_results = [r for r in concept_results if r["category"] == cat]
            lam_vals = sorted(set(r["lambda"] for r in cat_results))

            for lam in lam_vals:
                if lam == 0:
                    continue
                for mode in ("abrupt", "beatwise"):
                    entries = [
                        r
                        for r in cat_results
                        if r["lambda"] == lam and r["mode"] == mode
                    ]
                    if not entries:
                        continue

                    changes = [e[change_key] for e in entries if e["gen_n_notes"] > 0]
                    degs = [
                        e["total_degradation"]
                        for e in entries
                        if not np.isnan(e["total_degradation"])
                    ]

                    if lam > 0:
                        n_success = sum(1 for c in changes if c > 0)
                    else:
                        n_success = sum(1 for c in changes if c < 0)

                    success_pct = (n_success / len(changes) * 100) if changes else 0
                    mean_chg = np.mean(changes) if changes else 0
                    mean_deg = np.mean(degs) if degs else float("nan")

                    # Find best sample (highest |change| / degradation)
                    best = ""
                    if entries:
                        best_entry = max(
                            entries,
                            key=lambda e: abs(e[change_key])
                            / max(e["total_degradation"], 0.01),
                        )
                        best = best_entry["song_name"][:20]

                    cat_label = cat if mode == "abrupt" else ""
                    print(
                        f"  {cat_label:<18} {lam:>+5.2f}  {mode:<10}  "
                        f"{len(entries):>3}  {success_pct:>7.1f}%  "
                        f"{mean_chg:>+7.1f}  {mean_deg:>6.2f}  {best:<20}"
                    )
                # Separator between abrupt/beatwise pairs
                print(f"  {'':>18} {'':>5}  {'─' * 70}")

        print()


def print_aggregate_summary(results: List[dict]):
    """Print high-level averages across all categories."""
    from collections import defaultdict

    by_mode = defaultdict(lambda: {"changes": [], "degs": [], "successes": [], "n": []})

    for r in results:
        if r["lambda"] == 0:
            continue
        mode = r["mode"]
        is_dur = "duration" in r["concept"]
        change_key = "duration_change" if is_dur else "pitch_change"
        change = r[change_key]
        deg = r["total_degradation"]

        success = (r["lambda"] > 0 and change > 0) or (r["lambda"] < 0 and change < 0)

        by_mode[mode]["changes"].append(abs(change))
        if not np.isnan(deg):
            by_mode[mode]["degs"].append(deg)
        by_mode[mode]["successes"].append(1 if success else 0)

    print(f"\n{'═' * 70}")
    print(f"  OVERALL SUMMARY: Abrupt vs Beat-wise")
    print(f"{'═' * 70}")
    print(f"  {'Mode':<12}  {'N':>5}  {'Success%':>9}  " f"{'|ΔMean|':>9}  {'Deg':>7}")
    print(f"  {'-' * 55}")

    for mode in ("abrupt", "beatwise"):
        d = by_mode[mode]
        n = len(d["successes"])
        sr = np.mean(d["successes"]) * 100 if d["successes"] else 0
        mc = np.mean(d["changes"]) if d["changes"] else 0
        md = np.mean(d["degs"]) if d["degs"] else float("nan")
        print(f"  {mode:<12}  {n:>5}  {sr:>8.1f}%  " f"{mc:>+8.1f}  {md:>7.2f}")
    print(f"{'═' * 70}")


def generate_listening_list(results: List[dict], output_dir: pathlib.Path):
    """Rank all samples by steering effectiveness / degradation, print top-20."""
    scored = []
    for r in results:
        if r["lambda"] == 0 or r["gen_n_notes"] == 0:
            continue
        is_dur = "duration" in r["concept"]
        change_key = "duration_change" if is_dur else "pitch_change"
        change = abs(r[change_key])
        deg = max(r["total_degradation"], 0.01)
        if np.isnan(deg):
            deg = 99.0
        score = change / deg

        short = "pitch" if "pitch" in r["concept"] else "duration"
        direction = "increase" if r["lambda"] > 0 else "decrease"

        scored.append(
            {
                "score": score,
                "mode": r["mode"],
                "concept": short,
                "direction": direction,
                "category": r["category"],
                "lambda": r["lambda"],
                "change": r[change_key],
                "degradation": r["total_degradation"],
                "song": r["song_name"],
                "filepath": r.get("filepath", ""),
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)

    lines = [
        "=" * 100,
        "LISTENING PRIORITY LIST  (Beat-wise vs Abrupt Comparison)",
        "Ranked by: |steering change| / degradation",
        "=" * 100,
        "",
    ]

    for i, s in enumerate(scored[:30], 1):
        lines.append(
            f"{i:>2}. [{s['score']:.1f}]  {s['mode']:<10}  "
            f"{s['concept']:<8} {s['direction']:<10}  "
            f"λ={s['lambda']:+.2f}  Δ={s['change']:+.1f}  "
            f"deg={s['degradation']:.2f}  {s['song']}"
        )
        lines.append(f"    {s['filepath']}")
        lines.append("")

    txt = "\n".join(lines)
    print(txt)

    list_path = output_dir / "listening_priority.txt"
    with open(list_path, "w") as f:
        f.write(txt)
    logger.info(f"Saved listening list to {list_path}")


def generate_paired_listening_list(results: List[dict], output_dir: pathlib.Path):
    """Find the best paired samples (same song, same λ, abrupt vs beatwise)
    and output them for easy A/B comparison."""
    from collections import defaultdict

    # Group by (concept, category, lambda, song_name) → {mode: result}
    pairs = defaultdict(dict)
    for r in results:
        if r["lambda"] == 0:
            continue
        key = (r["concept"], r["category"], r["lambda"], r["song_name"])
        pairs[key][r["mode"]] = r

    # Find complete pairs and score them
    scored_pairs = []
    for key, modes in pairs.items():
        if "abrupt" not in modes or "beatwise" not in modes:
            continue
        a = modes["abrupt"]
        b = modes["beatwise"]
        is_dur = "duration" in a["concept"]
        change_key = "duration_change" if is_dur else "pitch_change"

        # Score: average of |change|/deg for both modes
        a_score = abs(a[change_key]) / max(a["total_degradation"], 0.01)
        b_score = abs(b[change_key]) / max(b["total_degradation"], 0.01)
        avg_score = (a_score + b_score) / 2

        scored_pairs.append(
            {
                "score": avg_score,
                "concept": "pitch" if "pitch" in a["concept"] else "duration",
                "category": a["category"],
                "lambda": a["lambda"],
                "song": a["song_name"],
                "abrupt": {
                    "change": a[change_key],
                    "degradation": a["total_degradation"],
                    "filepath": a.get("filepath", ""),
                },
                "beatwise": {
                    "change": b[change_key],
                    "degradation": b["total_degradation"],
                    "filepath": b.get("filepath", ""),
                },
            }
        )

    scored_pairs.sort(key=lambda x: x["score"], reverse=True)

    lines = [
        "=" * 110,
        "PAIRED A/B LISTENING LIST  (Abrupt vs Beat-wise — Same Song, Same λ)",
        "Ranked by: average |change| / degradation across both modes",
        "=" * 110,
        "",
    ]

    for i, p in enumerate(scored_pairs[:20], 1):
        direction = "↑" if p["lambda"] > 0 else "↓"
        lines.append(
            f"{i:>2}. [{p['score']:.1f}]  {p['concept']:<8} {direction}  "
            f"λ={p['lambda']:+.2f}  {p['song']}"
        )
        lines.append(
            f"    ABRUPT:   Δ={p['abrupt']['change']:+.1f}  "
            f"deg={p['abrupt']['degradation']:.2f}"
        )
        lines.append(f"      → {p['abrupt']['filepath']}")
        lines.append(
            f"    BEATWISE: Δ={p['beatwise']['change']:+.1f}  "
            f"deg={p['beatwise']['degradation']:.2f}"
        )
        lines.append(f"      → {p['beatwise']['filepath']}")
        lines.append("")

    txt = "\n".join(lines)
    print(txt)

    path = output_dir / "paired_listening_priority.txt"
    with open(path, "w") as f:
        f.write(txt)
    logger.info(f"Saved paired list to {path}")


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare abrupt vs beat-wise smooth SAS steering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n_songs", type=int, default=5, help="Songs per category")
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0.75,1.0,1.5,-0.75,-1.0,-1.5",
        help="Comma-separated λ values (positive for low→high, negative for high→low)",
    )
    parser.add_argument(
        "--n_ramp_beats",
        type=int,
        default=32,
        help="Number of beats for the smooth ramp (default: 32 beats ≈ 8 bars)",
    )
    parser.add_argument(
        "--conditioning_beats",
        type=int,
        default=CONDITIONING_BEATS,
    )
    parser.add_argument(
        "--continuation_len",
        type=int,
        default=CONTINUATION_LEN,
    )
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=REPO_ROOT / "exp" / "sod" / "sparse_steering" / "beatwise_comparison",
    )

    # Path overrides
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train_args", type=pathlib.Path, default=DEFAULT_TRAIN_ARGS)
    parser.add_argument("--encoding_path", type=pathlib.Path, default=DEFAULT_ENCODING)
    parser.add_argument("--sae_dir", type=pathlib.Path, default=DEFAULT_SAE_DIR)
    parser.add_argument("--sas_dir", type=pathlib.Path, default=DEFAULT_SAS_DIR)
    parser.add_argument("--notes_dir", type=pathlib.Path, default=DEFAULT_NOTES_DIR)

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    lambdas = sorted([float(x.strip()) for x in args.lambdas.split(",")])

    logger.info(f"Device: {device}")
    logger.info(f"λ values: {lambdas}")
    logger.info(f"n_ramp_beats: {args.n_ramp_beats}")
    logger.info(f"n_songs: {args.n_songs}")
    logger.info(f"Output: {args.output_dir}")

    # Load model
    logger.info("Loading model...")
    model, encoding, train_args = load_model(
        args.checkpoint, args.train_args, args.encoding_path, device
    )
    sae_models = load_sae_models(args.sae_dir, device)

    # Run comparison
    t0 = time.time()
    all_results = run_comparison(
        model,
        encoding,
        sae_models,
        device,
        args.sas_dir,
        args.notes_dir,
        args.output_dir,
        args.n_songs,
        lambdas,
        args.n_ramp_beats,
        args.conditioning_beats,
        args.continuation_len,
    )
    elapsed = time.time() - t0

    # Save raw results
    results_path = args.output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Saved {len(all_results)} results to {results_path}")

    # Analysis
    summary = analyze_results(all_results)
    summary_path = args.output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print tables
    print_comparison_table(all_results)
    print_aggregate_summary(all_results)

    # Listening lists
    generate_listening_list(all_results, args.output_dir)
    generate_paired_listening_list(all_results, args.output_dir)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  Done! ({elapsed:.0f}s)")
    logger.info(f"  Results: {results_path}")
    logger.info(f"  Paired list: {args.output_dir / 'paired_listening_priority.txt'}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
