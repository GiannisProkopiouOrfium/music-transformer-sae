#!/usr/bin/env python3
"""
Beat-wise Smooth Steering vs Abrupt Steering — DiffMean (DM) Version.

Same methodology as the SAS beatwise comparison, but using DiffMean
steering vectors (additive intervention in the residual stream).

For each song × α, generates:
  1. Abrupt: constant α from token 1
  2. Beat-wise: cosine S-curve ramp over musical beats (constant time intervals)

Categories
----------
- Pitch:    low → high  (α > 0 on low-pitch songs)
            high → low  (α < 0 on high-pitch songs)
- Duration: short → long (α > 0 on short-duration songs)
            long → short (α < 0 on long-duration songs)

Usage
-----
    python steering_interventions/beatwise_vs_abrupt_dm.py \
        --n_songs 5 --alphas 0.5,1.0,1.5,-0.5,-1.0,-1.5

    # Quick test
    python steering_interventions/beatwise_vs_abrupt_dm.py \
        --n_songs 3 --alphas 1.0,-1.0
"""

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Project paths
REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "mmt"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import config
import music_x_transformers
import representation
import utils

from steered_generator import SteeredGenerator, load_steering_vectors
from smooth_steering_dm import SmoothSteeringHook
from conditioned_evaluator import (
    find_extreme_pitch_songs,
    find_extreme_duration_songs,
    load_song_tokens,
    extract_conditioning_prefix,
    evaluate_quality_metrics,
    calculate_degradation,
    measure_pitch_from_tokens,
    calculate_initial_pitch,
    calculate_initial_duration,
    GROUND_TRUTH_METRICS,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ── Core generation ──────────────────────────────────────────────────────────


def generate_one_dm(
    model: nn.Module,
    encoding: dict,
    steering_vectors: Dict[int, torch.Tensor],
    device: torch.device,
    filepath: pathlib.Path,
    alpha: float,
    mode: str,  # "abrupt" or "beatwise"
    n_ramp_beats: int,
    conditioning_beats: int,
    continuation_len: int,
) -> Tuple[np.ndarray, dict]:
    """Generate a single DM-steered continuation.

    Returns (full_token_sequence, metrics_dict).
    """
    tokens = load_song_tokens(filepath, encoding)
    conditioning = extract_conditioning_prefix(tokens, conditioning_beats).to(device)
    eos = encoding["type_code_map"]["end-of-song"]

    init_pitch = calculate_initial_pitch(tokens, encoding, conditioning_beats) or 0.0
    init_dur = calculate_initial_duration(tokens, encoding, conditioning_beats) or 0.0

    if mode == "abrupt":
        # Use SteeredGenerator for constant-alpha steering
        generator = SteeredGenerator(model, steering_vectors, encoding)
        generated = generator.generate(
            conditioning.clone(),
            continuation_len,
            alpha=alpha,
            target_layers=None,  # all layers
            eos_token=eos,
            temperature=config.GENERATION_TEMPERATURE,
            filter_logits_fn=config.GENERATION_FILTER,
            filter_thres=config.GENERATION_FILTER_THRESHOLD,
            monotonicity_dim=("type", "beat"),
        )
    else:
        # Beat-wise smooth: register SmoothSteeringHook with beat_mode
        attn_layers = model.decoder.net.attn_layers
        smooth_hooks = []

        for layer_idx in range(len(attn_layers.layers)):
            if layer_idx not in steering_vectors:
                continue
            hook = SmoothSteeringHook(
                steering_vector=steering_vectors[layer_idx],
                alpha=alpha,
                mode="warmup_hold",
                schedule="cosine",
                n_ramp=n_ramp_beats,
                beat_mode=True,
            )
            layer_module_list = attn_layers.layers[layer_idx]
            if (
                isinstance(layer_module_list, nn.ModuleList)
                and len(layer_module_list) > 1
            ):
                target_module = layer_module_list[1]
            else:
                target_module = layer_module_list
            hook.register(target_module)
            smooth_hooks.append(hook)

        # Beat observer pre-hook on the transformer wrapper
        wrapper = model.decoder.net

        def _beat_observer(module, args):
            x = args[0]  # (batch, seq_len, 6)
            beat_val = x[0, -1, 1].item()
            for h in smooth_hooks:
                h.observe_beat(int(beat_val))

        obs_handle = wrapper.register_forward_pre_hook(_beat_observer)

        try:
            with torch.no_grad():
                generated = model.generate(
                    conditioning.clone(),
                    continuation_len,
                    eos_token=eos,
                    temperature=config.GENERATION_TEMPERATURE,
                    filter_logits_fn=config.GENERATION_FILTER,
                    filter_thres=config.GENERATION_FILTER_THRESHOLD,
                    monotonicity_dim=("type", "beat"),
                )
        finally:
            for h in smooth_hooks:
                h.remove()
            obs_handle.remove()

    full_seq = torch.cat((conditioning, generated), 1).cpu().numpy()[0]
    gen_only = generated.cpu().numpy()[0]

    m = measure_pitch_from_tokens(gen_only, encoding)
    qm = evaluate_quality_metrics(full_seq, encoding)
    deg = calculate_degradation(qm, GROUND_TRUTH_METRICS)

    metrics = {
        "song_name": filepath.stem,
        "alpha": float(alpha),
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
    model: nn.Module,
    encoding: dict,
    device: torch.device,
    notes_dir: pathlib.Path,
    output_dir: pathlib.Path,
    n_songs: int,
    alphas: List[float],
    n_ramp_beats: int,
    conditioning_beats: int,
    continuation_len: int,
):
    """Run full paired comparison: abrupt vs beat-wise DM steering."""
    all_results = []

    concepts = [
        ("average_pitch", "pitch", find_extreme_pitch_songs),
        ("average_duration", "duration", find_extreme_duration_songs),
    ]

    for concept, short_name, find_fn in concepts:
        logger.info(f"\n{'=' * 72}")
        logger.info(f"  Concept: {short_name.upper()}")
        logger.info(f"{'=' * 72}")

        # Load DM steering vectors
        sv_path = (
            config.OUTPUT_DIR / "steering_vectors" / f"{concept}_steering_vectors.pt"
        )
        if not sv_path.exists():
            logger.error(f"Steering vectors not found: {sv_path}")
            continue
        steering_vectors, _ = load_steering_vectors(sv_path)
        steering_vectors = {k: v.to(device) for k, v in steering_vectors.items()}
        logger.info(f"  Loaded DM vectors from {sv_path}")

        # Find extreme songs
        low_songs, high_songs = find_fn(
            notes_dir, encoding, n_songs, conditioning_beats
        )

        low_cat = f"low_{short_name}"
        high_cat = f"high_{short_name}"
        logger.info(f"  Low  songs ({low_cat}): {len(low_songs)}")
        logger.info(f"  High songs ({high_cat}): {len(high_songs)}")

        for alpha in alphas:
            if alpha > 0:
                songs = low_songs
                category = low_cat
            elif alpha < 0:
                songs = high_songs
                category = high_cat
            else:
                songs = low_songs[:2]
                category = low_cat

            for filepath, init_val in songs:
                for mode in ("abrupt", "beatwise"):
                    logger.info(
                        f"  [{mode:>8}] {short_name} α={alpha:+.2f} "
                        f"| {filepath.stem} (init={init_val:.1f})"
                    )

                    full_seq, metrics = generate_one_dm(
                        model,
                        encoding,
                        steering_vectors,
                        device,
                        filepath,
                        alpha,
                        mode,
                        n_ramp_beats,
                        conditioning_beats,
                        continuation_len,
                    )
                    metrics["category"] = category
                    metrics["concept"] = concept

                    # Save .npy and .mid
                    alpha_str = (
                        f"alpha_{'pos' if alpha >= 0 else 'neg'}{abs(alpha):.2f}"
                    )
                    save_dir = output_dir / short_name / category / alpha_str / mode
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


# ── Analysis & reporting (same structure as SAS version) ─────────────────────


def print_comparison_table(results: List[dict]):
    """Print formatted paired comparison table."""
    from collections import defaultdict

    for concept in ("average_pitch", "average_duration"):
        short = "PITCH" if "pitch" in concept else "DURATION"
        is_dur = "duration" in concept
        change_key = "duration_change" if is_dur else "pitch_change"

        print(f"\n{'═' * 100}")
        print(f"  {short} STEERING (DiffMean): Abrupt vs Beat-wise Smooth")
        print(f"{'═' * 100}")
        print(
            f"  {'Category':<18} {'α':>5}  {'Mode':<10}  {'N':>3}  "
            f"{'Success%':>8}  {'ΔMean':>8}  {'Deg':>6}  {'Best Song':<20}"
        )
        print(f"  {'-' * 92}")

        concept_results = [r for r in results if r["concept"] == concept]
        categories = sorted(set(r["category"] for r in concept_results))

        for cat in categories:
            cat_results = [r for r in concept_results if r["category"] == cat]
            alpha_vals = sorted(set(r["alpha"] for r in cat_results))

            for alpha in alpha_vals:
                if alpha == 0:
                    continue
                for mode in ("abrupt", "beatwise"):
                    entries = [
                        r
                        for r in cat_results
                        if r["alpha"] == alpha and r["mode"] == mode
                    ]
                    if not entries:
                        continue

                    changes = [e[change_key] for e in entries if e["gen_n_notes"] > 0]
                    degs = [
                        e["total_degradation"]
                        for e in entries
                        if not np.isnan(e["total_degradation"])
                    ]

                    if alpha > 0:
                        n_success = sum(1 for c in changes if c > 0)
                    else:
                        n_success = sum(1 for c in changes if c < 0)

                    success_pct = (n_success / len(changes) * 100) if changes else 0
                    mean_chg = np.mean(changes) if changes else 0
                    mean_deg = np.mean(degs) if degs else float("nan")

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
                        f"  {cat_label:<18} {alpha:>+5.2f}  {mode:<10}  "
                        f"{len(entries):>3}  {success_pct:>7.1f}%  "
                        f"{mean_chg:>+7.1f}  {mean_deg:>6.2f}  {best:<20}"
                    )
                print(f"  {'':>18} {'':>5}  {'─' * 70}")

        print()


def print_aggregate_summary(results: List[dict]):
    """Print high-level averages."""
    from collections import defaultdict

    by_mode = defaultdict(lambda: {"changes": [], "degs": [], "successes": []})

    for r in results:
        if r["alpha"] == 0:
            continue
        mode = r["mode"]
        is_dur = "duration" in r["concept"]
        change_key = "duration_change" if is_dur else "pitch_change"
        change = r[change_key]
        deg = r["total_degradation"]

        success = (r["alpha"] > 0 and change > 0) or (r["alpha"] < 0 and change < 0)
        by_mode[mode]["changes"].append(abs(change))
        if not np.isnan(deg):
            by_mode[mode]["degs"].append(deg)
        by_mode[mode]["successes"].append(1 if success else 0)

    print(f"\n{'═' * 70}")
    print(f"  OVERALL SUMMARY (DiffMean): Abrupt vs Beat-wise")
    print(f"{'═' * 70}")
    print(f"  {'Mode':<12}  {'N':>5}  {'Success%':>9}  {'|ΔMean|':>9}  {'Deg':>7}")
    print(f"  {'-' * 55}")

    for mode in ("abrupt", "beatwise"):
        d = by_mode[mode]
        n = len(d["successes"])
        sr = np.mean(d["successes"]) * 100 if d["successes"] else 0
        mc = np.mean(d["changes"]) if d["changes"] else 0
        md = np.mean(d["degs"]) if d["degs"] else float("nan")
        print(f"  {mode:<12}  {n:>5}  {sr:>8.1f}%  {mc:>+8.1f}  {md:>7.2f}")
    print(f"{'═' * 70}")


def generate_paired_listening_list(results: List[dict], output_dir: pathlib.Path):
    """Find best paired samples for A/B comparison."""
    from collections import defaultdict

    pairs = defaultdict(dict)
    for r in results:
        if r["alpha"] == 0:
            continue
        key = (r["concept"], r["category"], r["alpha"], r["song_name"])
        pairs[key][r["mode"]] = r

    scored_pairs = []
    for key, modes in pairs.items():
        if "abrupt" not in modes or "beatwise" not in modes:
            continue
        a = modes["abrupt"]
        b = modes["beatwise"]
        is_dur = "duration" in a["concept"]
        change_key = "duration_change" if is_dur else "pitch_change"

        a_score = abs(a[change_key]) / max(a["total_degradation"], 0.01)
        b_score = abs(b[change_key]) / max(b["total_degradation"], 0.01)
        avg_score = (a_score + b_score) / 2

        scored_pairs.append(
            {
                "score": avg_score,
                "concept": "pitch" if "pitch" in a["concept"] else "duration",
                "category": a["category"],
                "alpha": a["alpha"],
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
        "PAIRED A/B LISTENING LIST — DiffMean  (Abrupt vs Beat-wise)",
        "Ranked by: average |change| / degradation across both modes",
        "=" * 110,
        "",
    ]

    for i, p in enumerate(scored_pairs[:20], 1):
        direction = "↑" if p["alpha"] > 0 else "↓"
        lines.append(
            f"{i:>2}. [{p['score']:.1f}]  {p['concept']:<8} {direction}  "
            f"α={p['alpha']:+.2f}  {p['song']}"
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
        description="Compare abrupt vs beat-wise smooth DiffMean steering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n_songs", type=int, default=5)
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.5,1.0,1.5,-0.5,-1.0,-1.5",
        help="Comma-separated α values",
    )
    parser.add_argument(
        "--n_ramp_beats",
        type=int,
        default=32,
        help="Beats for the smooth ramp (default: 32 ≈ 8 bars)",
    )
    parser.add_argument("--conditioning_beats", type=int, default=4)
    parser.add_argument("--continuation_len", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        default=REPO_ROOT
        / "exp"
        / "sod"
        / "steering_interventions"
        / "beatwise_comparison_dm",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    alphas = sorted([float(x.strip()) for x in args.alphas.split(",")])

    logger.info(f"Device: {device}")
    logger.info(f"α values: {alphas}")
    logger.info(f"n_ramp_beats: {args.n_ramp_beats}")
    logger.info(f"n_songs: {args.n_songs}")
    logger.info(f"Output: {args.output_dir}")

    # Load model
    logger.info("Loading model...")
    train_args = utils.load_json(config.MODEL_DIR / "train-args.json")
    encoding = representation.load_encoding(config.NOTES_DIR / "encoding.json")

    model = music_x_transformers.MusicXTransformer(
        dim=train_args["dim"],
        encoding=encoding,
        depth=train_args["layers"],
        heads=train_args["heads"],
        max_seq_len=train_args["max_seq_len"],
        max_beat=train_args["max_beat"],
        rotary_pos_emb=train_args["rel_pos_emb"],
        use_abs_pos_emb=train_args["abs_pos_emb"],
        emb_dropout=train_args["dropout"],
        attn_dropout=train_args["dropout"],
        ff_dropout=train_args["dropout"],
    ).to(device)

    checkpoint_path = config.CHECKPOINT_DIR / "best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    logger.info("Model loaded")

    # Run comparison
    t0 = time.time()
    all_results = run_comparison(
        model,
        encoding,
        device,
        config.NOTES_DIR,
        args.output_dir,
        args.n_songs,
        alphas,
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

    # Print tables
    print_comparison_table(all_results)
    print_aggregate_summary(all_results)

    # Listening lists
    generate_paired_listening_list(all_results, args.output_dir)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  Done! ({elapsed:.0f}s)")
    logger.info(f"  Results: {results_path}")
    logger.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
