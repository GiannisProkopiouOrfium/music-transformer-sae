#!/usr/bin/env python3
"""Phase 2 — Conditioned Dual-SAS Steering (4 Scenarios).

Finds songs with extreme pitch + duration combinations, conditions the model
on the first N beats, then generates continuations with dual-concept SAS
steering to fight both properties simultaneously.

Scenarios (all fight BOTH concepts):
  1. Low pitch + short duration  → steer high + long
  2. High pitch + long duration  → steer low + short
  3. Low pitch + long duration   → steer high + short
  4. High pitch + short duration → steer low + long

For each scenario × strategy × alpha combo:
  • Generate continuations conditioned on each extreme song
  • Measure generated pitch mean & duration mean
  • Evaluate quality degradation
  • Determine SUCCESS: did steering move both features in the desired direction?

Output:
  - conditioned_results.json
  - per-scenario listening lists
  - Generated MIDI (optional)

Usage:
    python sparse_steering/dual_steering/test_dual_conditioned.py
    python sparse_steering/dual_steering/test_dual_conditioned.py \
        --strategies direct cross_concept_masking \
        --n_songs 5 \
        --conditioning_beats 16 \
        --layers 10
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
from tqdm import tqdm

# ── Path setup ───────────────────────────────────────────────────────────────
_THIS_DIR = pathlib.Path(__file__).parent
_SPARSE_DIR = _THIS_DIR.parent
_PROJECT_ROOT = _SPARSE_DIR.parent

sys.path.insert(0, str(_SPARSE_DIR))
sys.path.insert(0, str(_PROJECT_ROOT / "mmt"))

import music_x_transformers
import representation
import utils

from config_dual import (
    DEFAULT_CHECKPOINT,
    DEFAULT_ENCODING,
    DEFAULT_NOTES_DIR,
    DEFAULT_TRAIN_ARGS,
    CONDITIONED_ALPHA_DURATION,
    CONDITIONED_ALPHA_PITCH,
    CONDITIONED_OUTPUT_DIR,
    CONDITIONED_SCENARIOS,
    CONDITIONING_BEATS,
    DEFAULT_LAYERS_TO_STEER,
    DURATION_HIGH_THRESHOLD,
    DURATION_LOW_THRESHOLD,
    GROUND_TRUTH_METRICS,
    N_SONGS_PER_SCENARIO,
    PITCH_HIGH_THRESHOLD,
    PITCH_LOW_THRESHOLD,
    SEQ_LEN,
    STRATEGIES,
)
from dual_steered_generator import (
    register_dual_hooks,
    register_expanded_k_hooks,
    register_sequential_hooks,
    register_budget_allocation_hooks,
    register_dense_sas_hooks,
    remove_hooks,
)
from sparse_vector_composer import SparseVectorComposer, load_and_create_composer

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Song discovery — find extreme pitch+duration combos
# ════════════════════════════════════════════════════════════════════════════


def find_extreme_dual_songs(
    notes_dir: pathlib.Path,
    encoding: dict,
    n_songs: int = 5,
    conditioning_beats: int = 16,
    cache_file: Optional[pathlib.Path] = None,
) -> Dict[str, List[Tuple[pathlib.Path, float, float]]]:
    """Scan dataset for songs with extreme pitch AND duration characteristics.

    Categories:
      low_pitch_short_duration:  mean_pitch < 60.0  AND mean_dur < 6.5
      low_pitch_long_duration:   mean_pitch < 60.0  AND mean_dur > 14.5
      high_pitch_short_duration: mean_pitch > 67.6  AND mean_dur < 6.5
      high_pitch_long_duration:  mean_pitch > 67.6  AND mean_dur > 14.5

    Returns:
        {category: [(filepath, mean_pitch, mean_duration), ...]}
    """
    if cache_file is None:
        cache_file = notes_dir / "extreme_dual_songs_cache.json"

    # Try cache
    if cache_file.exists():
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            result = {}
            for cat, songs in cached.items():
                result[cat] = [
                    (pathlib.Path(s["path"]), s["pitch"], s["duration"])
                    for s in songs[:n_songs]
                ]
            logger.info(f"Loaded extreme songs from cache: {cache_file}")
            for cat, songs in result.items():
                logger.info(f"  {cat}: {len(songs)} songs")
            return result
        except Exception as e:
            logger.warning(f"Cache load failed: {e}")

    logger.info(f"Scanning {notes_dir} for extreme pitch+duration songs...")

    candidates = {
        "low_pitch_short_duration": [],
        "low_pitch_long_duration": [],
        "high_pitch_short_duration": [],
        "high_pitch_long_duration": [],
    }

    scanned = 0
    for subfolder in sorted(notes_dir.iterdir()):
        if not subfolder.is_dir():
            continue
        for npy_file in sorted(subfolder.glob("*.npy")):
            try:
                notes = np.load(npy_file)
                if notes.ndim != 2 or notes.shape[1] != 5:
                    continue

                # Use first conditioning_beats only
                beat_vals = notes[:, 0]
                mask = beat_vals < conditioning_beats
                if mask.sum() < 10:
                    continue

                segment = notes[mask]
                mean_pitch = float(segment[:, 2].mean())  # col 2 = pitch
                mean_dur = float(segment[:, 3].mean())  # col 3 = duration

                scanned += 1

                # Classify
                low_p = mean_pitch < PITCH_LOW_THRESHOLD
                high_p = mean_pitch > PITCH_HIGH_THRESHOLD
                short_d = mean_dur < DURATION_LOW_THRESHOLD
                long_d = mean_dur > DURATION_HIGH_THRESHOLD

                if low_p and short_d:
                    candidates["low_pitch_short_duration"].append(
                        (npy_file, mean_pitch, mean_dur)
                    )
                if low_p and long_d:
                    candidates["low_pitch_long_duration"].append(
                        (npy_file, mean_pitch, mean_dur)
                    )
                if high_p and short_d:
                    candidates["high_pitch_short_duration"].append(
                        (npy_file, mean_pitch, mean_dur)
                    )
                if high_p and long_d:
                    candidates["high_pitch_long_duration"].append(
                        (npy_file, mean_pitch, mean_dur)
                    )

            except Exception:
                continue

    logger.info(f"Scanned {scanned} songs")

    # Sort by "extremeness" and take top N
    # Low pitch + short dur → lowest pitch, shortest dur
    candidates["low_pitch_short_duration"].sort(key=lambda x: (x[1], x[2]))
    # Low pitch + long dur → lowest pitch, longest dur
    candidates["low_pitch_long_duration"].sort(key=lambda x: (x[1], -x[2]))
    # High pitch + short dur → highest pitch, shortest dur
    candidates["high_pitch_short_duration"].sort(key=lambda x: (-x[1], x[2]))
    # High pitch + long dur → highest pitch, longest dur
    candidates["high_pitch_long_duration"].sort(key=lambda x: (-x[1], -x[2]))

    result = {}
    for cat in candidates:
        result[cat] = candidates[cat][:n_songs]
        logger.info(f"  {cat}: {len(result[cat])} songs found")
        for path, p, d in result[cat]:
            logger.info(
                f"    {path.parent.name}/{path.stem}: pitch={p:.1f}, dur={d:.1f}"
            )

    # Save cache
    try:
        cache_data = {}
        for cat, songs in result.items():
            cache_data[cat] = [
                {"path": str(s[0]), "pitch": s[1], "duration": s[2]} for s in songs
            ]
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"Saved cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to save cache: {e}")

    return result


def extract_conditioning_prefix(
    filepath: pathlib.Path, encoding: dict, n_beats: int
) -> Tuple[torch.Tensor, np.ndarray]:
    """Load song, encode to tokens, extract first N beats as primer.

    Returns:
        (conditioning_tensor, full_tokens)
    """
    notes = np.load(filepath)
    tokens = representation.encode_notes(notes, encoding)

    # Find cutoff at n_beats
    cond_len = 0
    for i, tok in enumerate(tokens):
        if tok[1] >= n_beats:  # col 1 = beat position
            break
        cond_len = i + 1

    if cond_len < 2:
        cond_len = min(16, len(tokens))

    cond_tokens = tokens[:cond_len]
    cond_tensor = torch.from_numpy(cond_tokens).long().unsqueeze(0)  # (1, T, 6)
    return cond_tensor, tokens


# ════════════════════════════════════════════════════════════════════════════
# Measurement helpers (same as unconditioned)
# ════════════════════════════════════════════════════════════════════════════


def extract_pitches(tokens, encoding):
    try:
        music = representation.decode(tokens, encoding)
        pitches = []
        for track in music.tracks:
            for note in track.notes:
                pitches.append(note.pitch)
        return pitches
    except Exception as e:
        logger.warning(f"Error extracting pitches: {e}")
        return []


def extract_durations(tokens, encoding):
    try:
        music = representation.decode(tokens, encoding)
        durations = []
        for track in music.tracks:
            for note in track.notes:
                durations.append(note.duration)
        return durations
    except Exception as e:
        logger.warning(f"Error extracting durations: {e}")
        return []


def evaluate_quality(tokens, encoding):
    try:
        import muspy

        music = representation.decode(tokens, encoding)
        music.trim(music.resolution * 64)

        if not music.tracks or not music.tracks[0].notes:
            return {
                "pitch_class_entropy": np.nan,
                "scale_consistency": np.nan,
                "groove_consistency": np.nan,
            }
        return {
            "pitch_class_entropy": float(muspy.pitch_class_entropy(music)),
            "scale_consistency": float(muspy.scale_consistency(music)) * 100,
            "groove_consistency": float(
                muspy.groove_consistency(music, 4 * music.resolution)
            )
            * 100,
        }
    except Exception as e:
        logger.warning(f"Error evaluating quality: {e}")
        return {
            "pitch_class_entropy": np.nan,
            "scale_consistency": np.nan,
            "groove_consistency": np.nan,
        }


def calculate_degradation(metrics):
    """Calculate quality degradation from ground truth.

    Returns NaN for total_degradation if any metric is NaN
    (matching DiffMean behaviour: failed samples are excluded from averages).
    """
    gt = GROUND_TRUTH_METRICS
    ed = abs(metrics["pitch_class_entropy"] - gt["pitch_class_entropy"])
    sd = max(0, gt["scale_consistency"] - metrics["scale_consistency"])
    gd = max(0, gt["groove_consistency"] - metrics["groove_consistency"])
    return {
        "entropy_diff": float(ed),
        "scale_diff": float(sd),
        "groove_diff": float(gd),
        "total_degradation": float(ed + sd + gd),
    }


# ════════════════════════════════════════════════════════════════════════════
# Core conditioned evaluation
# ════════════════════════════════════════════════════════════════════════════


def evaluate_conditioned_scenario(
    model,
    encoding: dict,
    sae_models: dict,
    composer: SparseVectorComposer,
    strategy: str,
    scenario_name: str,
    scenario_cfg: dict,
    songs: List[Tuple[pathlib.Path, float, float]],
    alpha_pitch_list: List[float],
    alpha_duration_list: List[float],
    layers_to_steer: List[int],
    conditioning_beats: int,
    seq_len: int,
    device: torch.device,
    output_dir: Optional[pathlib.Path] = None,
) -> List[dict]:
    """Evaluate one scenario across all songs and alpha combos."""
    eos = encoding["type_code_map"]["end-of-song"]
    pitch_sign = scenario_cfg["alpha_pitch_sign"]
    dur_sign = scenario_cfg["alpha_duration_sign"]

    # Filter alphas: only relevant signs + baseline
    pitch_alphas = sorted(
        set([0.0] + [a * pitch_sign for a in alpha_pitch_list if a > 0])
    )
    dur_alphas = sorted(
        set([0.0] + [a * dur_sign for a in alpha_duration_list if a > 0])
    )

    logger.info(f"Scenario: {scenario_name}")
    logger.info(f"  Pitch alphas: {pitch_alphas}")
    logger.info(f"  Duration alphas: {dur_alphas}")
    logger.info(f"  Songs: {len(songs)}")

    results = []

    for song_path, init_pitch, init_dur in songs:
        song_name = f"{song_path.parent.name}/{song_path.stem}"

        # Load conditioning prefix
        try:
            cond_tensor, _ = extract_conditioning_prefix(
                song_path, encoding, conditioning_beats
            )
            cond_tensor = cond_tensor.to(device)
        except Exception as e:
            logger.warning(f"  Failed to load {song_name}: {e}")
            continue

        for lp in pitch_alphas:
            for ld in dur_alphas:
                # Compose & register the appropriate hook type
                if strategy in ("expanded_k", "expanded_k_2x"):
                    km = 2.0 if strategy == "expanded_k_2x" else 1.5
                    combined = composer.compose(lp, ld, strategy)
                    handles = register_expanded_k_hooks(
                        model, sae_models, combined, layers_to_steer, k_multiplier=km
                    )
                elif strategy in (
                    "opposite_sign_masking_ek2",
                    "cross_concept_masking_ek2",
                    "gram_schmidt_ek2",
                ):
                    combined = composer.compose(lp, ld, strategy)
                    handles = register_expanded_k_hooks(
                        model, sae_models, combined, layers_to_steer, k_multiplier=2.0
                    )
                elif strategy == "norm_balanced_ek175":
                    combined = composer.compose(lp, ld, strategy)
                    handles = register_expanded_k_hooks(
                        model, sae_models, combined, layers_to_steer, k_multiplier=1.75
                    )
                elif strategy == "sequential":
                    pitch_vecs, dur_vecs = composer.compose_separate(lp, ld)
                    handles = register_sequential_hooks(
                        model, sae_models, pitch_vecs, dur_vecs, layers_to_steer
                    )
                elif strategy == "topk_budget":
                    combined = composer.compose(lp, ld, strategy)
                    handles = register_budget_allocation_hooks(
                        model, sae_models, combined, composer.pitch_vectors,
                        composer.duration_vectors, layers_to_steer,
                    )
                elif strategy == "sas_dense":
                    combined = composer.compose(lp, ld, strategy)
                    handles = register_dense_sas_hooks(
                        model, sae_models, combined, layers_to_steer
                    )
                else:
                    combined = composer.compose(lp, ld, strategy)
                    handles = register_dual_hooks(
                        model, sae_models, combined, layers_to_steer
                    )

                try:
                    with torch.no_grad():
                        generated = model.generate(
                            cond_tensor.clone(), seq_len, eos_token=eos
                        )
                    # model.generate() already strips the input tokens,
                    # so `generated` contains ONLY the new continuation.
                    continuation = generated[0].cpu().numpy()
                finally:
                    remove_hooks(handles)

                # Save full sequence (conditioning + continuation) as .npy
                if output_dir is not None:
                    save_dir = (
                        output_dir
                        / scenario_name
                        / strategy
                        / f"lp{lp:+.2f}_ld{ld:+.2f}"
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)
                    full_seq = np.concatenate(
                        [cond_tensor[0].cpu().numpy(), continuation], axis=0
                    )
                    npy_name = song_path.stem + ".npy"
                    np.save(save_dir / npy_name, full_seq)

                pitches = extract_pitches(continuation, encoding)
                durations = extract_durations(continuation, encoding)

                gen_pitch = float(np.mean(pitches)) if pitches else None
                gen_dur = float(np.mean(durations)) if durations else None

                qm = evaluate_quality(continuation, encoding)
                deg = calculate_degradation(qm)

                # Determine success
                pitch_success = None
                dur_success = None
                if gen_pitch is not None:
                    if pitch_sign > 0:
                        pitch_success = gen_pitch > init_pitch
                    else:
                        pitch_success = gen_pitch < init_pitch
                if gen_dur is not None:
                    if dur_sign > 0:
                        dur_success = gen_dur > init_dur
                    else:
                        dur_success = gen_dur < init_dur

                both_success = (
                    (pitch_success is True and dur_success is True)
                    if (lp != 0 or ld != 0)
                    else None
                )

                results.append(
                    {
                        "scenario": scenario_name,
                        "strategy": strategy,
                        "song_name": song_name,
                        "song_path": str(song_path),
                        "initial_pitch": init_pitch,
                        "initial_duration": init_dur,
                        "lambda_pitch": lp,
                        "lambda_duration": ld,
                        "generated_mean_pitch": gen_pitch,
                        "generated_mean_duration": gen_dur,
                        "pitch_delta": (gen_pitch - init_pitch) if gen_pitch else None,
                        "duration_delta": (gen_dur - init_dur) if gen_dur else None,
                        "pitch_success": pitch_success,
                        "duration_success": dur_success,
                        "both_success": both_success,
                        "generated_n_notes": len(pitches),
                        "quality": qm,
                        "degradation": deg,
                    }
                )

                status = "✓" if both_success else ("○" if both_success is None else "✗")
                logger.info(
                    f"  {status} {song_name} λp={lp:+.2f} λd={ld:+.2f} → "
                    f"pitch={gen_pitch:.1f} dur={gen_dur:.1f}"
                    if gen_pitch
                    else f"  ✗ {song_name} λp={lp:+.2f} λd={ld:+.2f} → no notes"
                )

    return results


# ════════════════════════════════════════════════════════════════════════════
# Analysis helpers
# ════════════════════════════════════════════════════════════════════════════


def analyze_results(results: List[dict]) -> dict:
    """Compute aggregate statistics from conditioned results."""
    from scipy import stats as scipy_stats

    analysis = {"by_scenario": {}, "by_strategy": {}, "overall": {}}

    # Group by scenario × strategy
    groups = {}
    for r in results:
        key = (r["scenario"], r["strategy"])
        groups.setdefault(key, []).append(r)

    for (scenario, strategy), group in groups.items():
        steered = [
            r for r in group if r["lambda_pitch"] != 0 or r["lambda_duration"] != 0
        ]
        n_total = len(steered)
        n_both_success = sum(1 for r in steered if r["both_success"] is True)
        n_pitch_success = sum(1 for r in steered if r["pitch_success"] is True)
        n_dur_success = sum(1 for r in steered if r["duration_success"] is True)

        pitch_deltas = [
            r["pitch_delta"] for r in steered if r["pitch_delta"] is not None
        ]
        dur_deltas = [
            r["duration_delta"] for r in steered if r["duration_delta"] is not None
        ]

        # Use nanmean to skip NaN degradation from failed quality evaluations
        deg_vals = [r["degradation"]["total_degradation"] for r in steered]
        entry = {
            "n_configs": n_total,
            "both_success_rate": n_both_success / n_total if n_total else 0,
            "pitch_success_rate": n_pitch_success / n_total if n_total else 0,
            "duration_success_rate": n_dur_success / n_total if n_total else 0,
            "mean_pitch_delta": float(np.mean(pitch_deltas)) if pitch_deltas else 0,
            "mean_duration_delta": float(np.mean(dur_deltas)) if dur_deltas else 0,
            "mean_degradation": (
                float(np.nanmean(deg_vals))
                if steered
                else 0
            ),
        }

        analysis["by_scenario"].setdefault(scenario, {})[strategy] = entry
        analysis["by_strategy"].setdefault(strategy, {})[scenario] = entry

    # Overall per strategy
    for strategy in set(r["strategy"] for r in results):
        steered = [
            r
            for r in results
            if r["strategy"] == strategy
            and (r["lambda_pitch"] != 0 or r["lambda_duration"] != 0)
        ]
        n = len(steered)
        analysis["overall"][strategy] = {
            "n_configs": n,
            "both_success_rate": (
                sum(1 for r in steered if r["both_success"]) / n if n else 0
            ),
            "pitch_success_rate": (
                sum(1 for r in steered if r["pitch_success"]) / n if n else 0
            ),
            "duration_success_rate": (
                sum(1 for r in steered if r["duration_success"]) / n if n else 0
            ),
            "mean_degradation": (
                float(np.nanmean([r["degradation"]["total_degradation"] for r in steered]))
                if steered
                else 0
            ),
        }

    return analysis


# ════════════════════════════════════════════════════════════════════════════
# Model / SAE loading (same as unconditioned)
# ════════════════════════════════════════════════════════════════════════════


def load_model(checkpoint, train_args_path, encoding_path, device):
    train_args = utils.load_json(str(train_args_path))
    encoding = representation.load_encoding(str(encoding_path))
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
    ckpt = torch.load(str(checkpoint), map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, encoding, train_args


def load_sae_models(sae_dir, device):
    from sae_model import SparseAutoencoder
    from steered_generator_sas import get_adaptive_k

    sae_models = {}
    for layer_idx in range(12):
        ckpt_path = sae_dir / f"sae_layer_{layer_idx}_best.pt"
        if not ckpt_path.exists():
            continue
        k = get_adaptive_k(layer_idx)
        sae = SparseAutoencoder(
            input_dim=512,
            sparse_dim=4096,
            k=k,
            tied_weights=True,
            normalize_input=True,
        )
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = checkpoint["model_state_dict"]
        if "_normalization_fitted" not in sd:
            sd["_normalization_fitted"] = torch.tensor(
                1 if sd.get("input_mean", torch.zeros(1)).abs().sum() > 0 else 0
            )
        sae.load_state_dict(sd)
        sae.eval()
        sae.to(device)
        sae_models[layer_idx] = sae
    return sae_models


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Conditioned dual-SAS steering (4 scenarios)"
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train_args", type=pathlib.Path, default=DEFAULT_TRAIN_ARGS)
    parser.add_argument("--encoding", type=pathlib.Path, default=DEFAULT_ENCODING)
    parser.add_argument("--notes_dir", type=pathlib.Path, default=DEFAULT_NOTES_DIR)
    parser.add_argument("--sae_dir", type=pathlib.Path, default=None)
    parser.add_argument("--pitch_vectors", type=pathlib.Path, default=None)
    parser.add_argument("--duration_vectors", type=pathlib.Path, default=None)
    parser.add_argument(
        "--output_dir", type=pathlib.Path, default=CONDITIONED_OUTPUT_DIR
    )
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Specific scenarios to run (default: all 4)",
    )
    parser.add_argument(
        "--alphas_pitch", type=float, nargs="+", default=CONDITIONED_ALPHA_PITCH
    )
    parser.add_argument(
        "--alphas_duration", type=float, nargs="+", default=CONDITIONED_ALPHA_DURATION
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=DEFAULT_LAYERS_TO_STEER
    )
    parser.add_argument("--n_songs", type=int, default=N_SONGS_PER_SCENARIO)
    parser.add_argument("--conditioning_beats", type=int, default=CONDITIONING_BEATS)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    from config_dual import SAE_CHECKPOINT_DIR, PITCH_SAS_VECTORS, DURATION_SAS_VECTORS

    if args.sae_dir is None:
        args.sae_dir = SAE_CHECKPOINT_DIR
    if args.pitch_vectors is None:
        args.pitch_vectors = PITCH_SAS_VECTORS
    if args.duration_vectors is None:
        args.duration_vectors = DURATION_SAS_VECTORS

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / "conditioned_dual.log"),
            logging.StreamHandler(),
        ],
    )

    # Device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Load model
    logger.info("Loading model...")
    model, encoding, _ = load_model(
        args.checkpoint, args.train_args, args.encoding, device
    )

    # Load SAEs
    logger.info("Loading SAE models...")
    sae_models = load_sae_models(args.sae_dir, device)

    # Load composer
    logger.info("Creating SparseVectorComposer...")
    composer = load_and_create_composer(
        str(args.pitch_vectors), str(args.duration_vectors)
    )

    # Find extreme songs
    logger.info("Finding extreme songs...")
    extreme_songs = find_extreme_dual_songs(
        args.notes_dir, encoding, args.n_songs, args.conditioning_beats
    )

    # Select scenarios
    scenarios = args.scenarios or list(CONDITIONED_SCENARIOS.keys())

    all_results = []
    t0 = time.time()

    for scenario_name in scenarios:
        if scenario_name not in CONDITIONED_SCENARIOS:
            logger.warning(f"Unknown scenario: {scenario_name}, skipping")
            continue

        cfg = CONDITIONED_SCENARIOS[scenario_name]
        song_category = cfg["song_category"]
        songs = extreme_songs.get(song_category, [])

        if not songs:
            logger.warning(
                f"No songs found for {song_category}, skipping {scenario_name}"
            )
            continue

        for strategy in args.strategies:
            logger.info(f"\n{'='*60}")
            logger.info(f"Scenario: {scenario_name} | Strategy: {strategy}")
            logger.info(f"{'='*60}")

            scenario_results = evaluate_conditioned_scenario(
                model=model,
                encoding=encoding,
                sae_models=sae_models,
                composer=composer,
                strategy=strategy,
                scenario_name=scenario_name,
                scenario_cfg=cfg,
                songs=songs,
                alpha_pitch_list=args.alphas_pitch,
                alpha_duration_list=args.alphas_duration,
                layers_to_steer=args.layers,
                conditioning_beats=args.conditioning_beats,
                seq_len=args.seq_len,
                device=device,
                output_dir=args.output_dir,
            )
            all_results.extend(scenario_results)

    elapsed = time.time() - t0

    # Analyze
    logger.info("\nAnalyzing results...")
    analysis = analyze_results(all_results)

    # Save
    out_path = args.output_dir / "conditioned_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "scenarios": scenarios,
                    "strategies": args.strategies,
                    "layers_to_steer": args.layers,
                    "n_songs": args.n_songs,
                    "conditioning_beats": args.conditioning_beats,
                    "elapsed_seconds": elapsed,
                },
                "results": all_results,
                "analysis": analysis,
            },
            f,
            indent=2,
        )

    logger.info(f"Saved results to {out_path}")

    # Print summary
    print("\n" + "=" * 72)
    print("DUAL-SAS CONDITIONED EVALUATION COMPLETE")
    print("=" * 72)
    print(f"Scenarios: {len(scenarios)}")
    print(f"Strategies: {args.strategies}")
    print(f"Total configs evaluated: {len(all_results)}")
    print(f"Time: {elapsed / 60:.1f} minutes")

    print("\n── Success Rates by Strategy ──")
    for strategy, stats in sorted(analysis["overall"].items()):
        print(
            f"  {strategy:<28s}  both={stats['both_success_rate']:.1%}  "
            f"pitch={stats['pitch_success_rate']:.1%}  "
            f"dur={stats['duration_success_rate']:.1%}  "
            f"deg={stats['mean_degradation']:.2f}"
        )

    print("\n── Success Rates by Scenario ──")
    for scenario, strats in sorted(analysis["by_scenario"].items()):
        print(f"\n  {scenario}:")
        for strategy, stats in sorted(strats.items()):
            print(
                f"    {strategy:<28s}  both={stats['both_success_rate']:.1%}  "
                f"Δpitch={stats['mean_pitch_delta']:+.1f}  "
                f"Δdur={stats['mean_duration_delta']:+.1f}  "
                f"deg={stats['mean_degradation']:.2f}"
            )

    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
