#!/usr/bin/env python3
"""Phase 1 — Unconditioned Dual-SAS Grid Search.

Tests all composition strategies across a grid of (λ_pitch, λ_duration) pairs.
For each configuration, generates N samples and evaluates:
  • Mean generated pitch & duration
  • Quality metrics (pitch_class_entropy, scale_consistency, groove_consistency)
  • Total degradation from ground-truth baselines

Test matrix:
  Strategies: direct, cross_concept_masking, gram_schmidt_pitch, gram_schmidt_duration
  Alpha pitch:    from config_dual.ALPHA_PITCH_GRID
  Alpha duration: from config_dual.ALPHA_DURATION_GRID
  Samples/config: 5

Output:
  - JSON with per-config metrics → unconditioned_results.json
  - Generated MIDI files (optional)
  - Log file

Usage:
    python sparse_steering/dual_steering/test_dual_unconditioned.py
    python sparse_steering/dual_steering/test_dual_unconditioned.py \
        --strategies direct cross_concept_masking \
        --n_samples 5 \
        --layers 10
"""

import argparse
import json
import logging
import pathlib
import sys
import time
from typing import Dict, List, Optional

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
    DEFAULT_TRAIN_ARGS,
    ALPHA_DURATION_GRID,
    ALPHA_PITCH_GRID,
    DEFAULT_LAYERS_TO_STEER,
    GROUND_TRUTH_METRICS,
    N_SAMPLES_PER_CONFIG,
    SEQ_LEN,
    STRATEGIES,
    UNCONDITIONED_OUTPUT_DIR,
)
from dual_steered_generator import (
    register_dual_hooks,
    register_expanded_k_hooks,
    register_sequential_hooks,
    register_budget_allocation_hooks,
    register_dense_sas_hooks,
    remove_hooks,
)
from sparse_vector_composer import (
    SparseVectorComposer,
    load_and_create_composer,
)

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Measurement helpers
# ════════════════════════════════════════════════════════════════════════════


def extract_pitches(tokens: np.ndarray, encoding: dict) -> List[float]:
    """Extract MIDI pitch values from generated tokens."""
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


def extract_durations(tokens: np.ndarray, encoding: dict) -> List[float]:
    """Extract duration values (ticks) from generated tokens."""
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


def evaluate_quality(tokens: np.ndarray, encoding: dict) -> dict:
    """Evaluate quality metrics via muspy."""
    try:
        import muspy

        music = representation.decode(tokens, encoding)
        music.trim(music.resolution * 64)

        if not music.tracks or not music.tracks[0].notes:
            return {
                "pitch_class_entropy": 0.0,
                "scale_consistency": 0.0,
                "groove_consistency": 0.0,
            }

        pce = float(muspy.pitch_class_entropy(music))
        sc = float(muspy.scale_consistency(music)) * 100.0
        gc = float(muspy.groove_consistency(music, 4 * music.resolution)) * 100.0

        return {
            "pitch_class_entropy": pce,
            "scale_consistency": sc,
            "groove_consistency": gc,
        }
    except Exception as e:
        logger.warning(f"Quality evaluation failed: {e}")
        return {
            "pitch_class_entropy": 0.0,
            "scale_consistency": 0.0,
            "groove_consistency": 0.0,
        }


def calculate_degradation(metrics: dict) -> dict:
    """Calculate quality degradation from ground-truth baselines."""
    gt = GROUND_TRUTH_METRICS
    entropy_diff = abs(metrics["pitch_class_entropy"] - gt["pitch_class_entropy"])
    scale_diff = max(0, gt["scale_consistency"] - metrics["scale_consistency"])
    groove_diff = max(0, gt["groove_consistency"] - metrics["groove_consistency"])
    return {
        "entropy_diff": float(entropy_diff),
        "scale_diff": float(scale_diff),
        "groove_diff": float(groove_diff),
        "total_degradation": float(entropy_diff + scale_diff + groove_diff),
    }


# ════════════════════════════════════════════════════════════════════════════
# Core evaluation for one configuration
# ════════════════════════════════════════════════════════════════════════════


def evaluate_config(
    model,
    encoding: dict,
    sae_models: dict,
    composer: SparseVectorComposer,
    strategy: str,
    lambda_pitch: float,
    lambda_duration: float,
    layers_to_steer: List[int],
    n_samples: int,
    seq_len: int,
    device: torch.device,
    save_midi: bool = False,
    output_dir: Optional[pathlib.Path] = None,
    k_multiplier: float = 1.5,
) -> dict:
    """Generate samples for one (strategy, λ_p, λ_d) config and evaluate."""
    eos = encoding["type_code_map"]["end-of-song"]

    # Register the appropriate hook type for this strategy
    if strategy in ("expanded_k", "expanded_k_2x"):
        km = 2.0 if strategy == "expanded_k_2x" else k_multiplier
        combined = composer.compose(lambda_pitch, lambda_duration, strategy)
        handles = register_expanded_k_hooks(
            model, sae_models, combined, layers_to_steer, k_multiplier=km
        )
    elif strategy in ("opposite_sign_masking_ek2", "cross_concept_masking_ek2"):
        # Masking composition + expanded K (2×) hook
        combined = composer.compose(lambda_pitch, lambda_duration, strategy)
        handles = register_expanded_k_hooks(
            model, sae_models, combined, layers_to_steer, k_multiplier=2.0
        )
    elif strategy == "sequential":
        pitch_vecs, dur_vecs = composer.compose_separate(lambda_pitch, lambda_duration)
        handles = register_sequential_hooks(
            model, sae_models, pitch_vecs, dur_vecs, layers_to_steer
        )
    elif strategy == "topk_budget":
        combined = composer.compose(lambda_pitch, lambda_duration, strategy)
        # Pass raw (unscaled) vectors so the hook can compute feature masks
        handles = register_budget_allocation_hooks(
            model,
            sae_models,
            combined,
            composer.pitch_vectors,
            composer.duration_vectors,
            layers_to_steer,
        )
    elif strategy == "sas_dense":
        # SAS-informed dense steering: project sparse→dense via SAE decoder
        combined = composer.compose(lambda_pitch, lambda_duration, strategy)
        handles = register_dense_sas_hooks(model, sae_models, combined, layers_to_steer)
    else:
        combined = composer.compose(lambda_pitch, lambda_duration, strategy)
        handles = register_dual_hooks(model, sae_models, combined, layers_to_steer)

    all_pitches = []
    all_durations = []
    all_quality = []
    all_degradation = []
    valid = 0

    try:
        for i in range(n_samples):
            with torch.no_grad():
                start = torch.zeros(1, 1, 6, dtype=torch.long, device=device)
                generated = model.generate(start, seq_len, eos_token=eos)

            tokens = generated[0].cpu().numpy()

            pitches = extract_pitches(tokens, encoding)
            durations = extract_durations(tokens, encoding)

            if not pitches or not durations:
                continue

            valid += 1
            all_pitches.append(float(np.mean(pitches)))
            all_durations.append(float(np.mean(durations)))

            qm = evaluate_quality(tokens, encoding)
            all_quality.append(qm)
            all_degradation.append(calculate_degradation(qm))

            # Save MIDI if requested
            if save_midi and output_dir is not None:
                midi_dir = output_dir / "midi" / strategy
                midi_dir.mkdir(parents=True, exist_ok=True)
                fname = f"p{lambda_pitch:+.2f}_d{lambda_duration:+.2f}_s{i}.npy"
                np.save(midi_dir / fname, tokens)

    finally:
        remove_hooks(handles)

    if valid == 0:
        return {
            "strategy": strategy,
            "lambda_pitch": lambda_pitch,
            "lambda_duration": lambda_duration,
            "valid_samples": 0,
            "pitch_mean": None,
            "duration_mean": None,
            "quality": None,
            "degradation": None,
        }

    # Aggregate quality
    avg_quality = {}
    for key in ["pitch_class_entropy", "scale_consistency", "groove_consistency"]:
        vals = [q[key] for q in all_quality if q[key] > 0]
        avg_quality[key] = float(np.mean(vals)) if vals else 0.0

    avg_deg = {}
    for key in ["entropy_diff", "scale_diff", "groove_diff", "total_degradation"]:
        vals = [d[key] for d in all_degradation]
        avg_deg[key] = float(np.mean(vals))

    return {
        "strategy": strategy,
        "lambda_pitch": lambda_pitch,
        "lambda_duration": lambda_duration,
        "valid_samples": valid,
        "pitch_mean": float(np.mean(all_pitches)),
        "pitch_std": float(np.std(all_pitches)),
        "duration_mean": float(np.mean(all_durations)),
        "duration_std": float(np.std(all_durations)),
        "quality": avg_quality,
        "degradation": avg_deg,
    }


# ════════════════════════════════════════════════════════════════════════════
# Model / SAE loading (reuse from single-concept code)
# ════════════════════════════════════════════════════════════════════════════


def load_model(checkpoint, train_args_path, encoding_path, device):
    """Load MMT model."""
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
    """Load trained SAE models for all layers."""
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

    logger.info(f"Loaded {len(sae_models)} SAE models")
    return sae_models


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Unconditioned dual-SAS grid search"
    )
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train_args", type=pathlib.Path, default=DEFAULT_TRAIN_ARGS)
    parser.add_argument("--encoding", type=pathlib.Path, default=DEFAULT_ENCODING)
    parser.add_argument("--sae_dir", type=pathlib.Path, default=None)
    parser.add_argument("--pitch_vectors", type=pathlib.Path, default=None)
    parser.add_argument("--duration_vectors", type=pathlib.Path, default=None)
    parser.add_argument(
        "--output_dir", type=pathlib.Path, default=UNCONDITIONED_OUTPUT_DIR
    )
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES)
    parser.add_argument(
        "--alphas_pitch", type=float, nargs="+", default=ALPHA_PITCH_GRID
    )
    parser.add_argument(
        "--alphas_duration", type=float, nargs="+", default=ALPHA_DURATION_GRID
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=DEFAULT_LAYERS_TO_STEER
    )
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES_PER_CONFIG)
    parser.add_argument("--seq_len", type=int, default=SEQ_LEN)
    parser.add_argument("--save_midi", action="store_true")
    parser.add_argument(
        "--k_multiplier",
        type=float,
        default=1.5,
        help="K multiplier for expanded_k strategy (default 1.5)",
    )
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    # Resolve defaults from config
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
            logging.FileHandler(args.output_dir / "unconditioned_grid.log"),
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
    model, encoding, train_args = load_model(
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

    # Build experiment matrix
    configs = []
    for strategy in args.strategies:
        for lp in args.alphas_pitch:
            for ld in args.alphas_duration:
                configs.append((strategy, lp, ld))

    n_configs = len(configs)
    total_gens = n_configs * args.n_samples
    logger.info(
        f"Experiment matrix: {len(args.strategies)} strategies × "
        f"{len(args.alphas_pitch)} pitch × {len(args.alphas_duration)} duration "
        f"= {n_configs} configs × {args.n_samples} samples = {total_gens} generations"
    )

    # Run
    results = []
    t0 = time.time()

    for strategy, lp, ld in tqdm(configs, desc="Grid search"):
        try:
            result = evaluate_config(
                model=model,
                encoding=encoding,
                sae_models=sae_models,
                composer=composer,
                strategy=strategy,
                lambda_pitch=lp,
                lambda_duration=ld,
                layers_to_steer=args.layers,
                n_samples=args.n_samples,
                seq_len=args.seq_len,
                device=device,
                save_midi=args.save_midi,
                output_dir=args.output_dir,
                k_multiplier=args.k_multiplier,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed: {strategy} p={lp} d={ld}: {e}")
            results.append(
                {
                    "strategy": strategy,
                    "lambda_pitch": lp,
                    "lambda_duration": ld,
                    "error": str(e),
                }
            )

    elapsed = time.time() - t0

    # Save
    out_path = args.output_dir / "unconditioned_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "metadata": {
                    "strategies": args.strategies,
                    "alphas_pitch": args.alphas_pitch,
                    "alphas_duration": args.alphas_duration,
                    "layers_to_steer": args.layers,
                    "n_samples": args.n_samples,
                    "seq_len": args.seq_len,
                    "elapsed_seconds": elapsed,
                    "n_configs": n_configs,
                    "total_generations": total_gens,
                },
                "results": results,
            },
            f,
            indent=2,
        )

    logger.info(f"\nSaved results to {out_path}")
    logger.info(
        f"Total time: {elapsed / 60:.1f} min ({elapsed / max(n_configs, 1):.1f}s/config)"
    )

    # Print summary
    print("\n" + "=" * 72)
    print("DUAL-SAS UNCONDITIONED GRID SEARCH COMPLETE")
    print("=" * 72)
    print(f"Configs tested: {n_configs}")
    print(f"Total generations: {total_gens}")
    print(f"Time: {elapsed / 60:.1f} minutes")
    print(f"Results: {out_path}")

    # Per-strategy summary
    valid_results = [r for r in results if r.get("pitch_mean") is not None]
    if valid_results:
        print("\n" + "-" * 72)
        print("PER-STRATEGY SUMMARY")
        print("-" * 72)
        for strat in args.strategies:
            sr = [r for r in valid_results if r["strategy"] == strat]
            if not sr:
                print(f"\n  {strat}: NO valid samples")
                continue
            pitches = [r["pitch_mean"] for r in sr]
            durations = [r["duration_mean"] for r in sr]
            degs = [
                r["degradation"]["total_degradation"]
                for r in sr
                if r.get("degradation")
            ]
            pitch_range = max(pitches) - min(pitches)
            dur_range = max(durations) - min(durations)
            # Find baseline (0,0)
            baseline = [
                r for r in sr if r["lambda_pitch"] == 0 and r["lambda_duration"] == 0
            ]
            bl_pitch = baseline[0]["pitch_mean"] if baseline else np.mean(pitches)
            bl_dur = baseline[0]["duration_mean"] if baseline else np.mean(durations)
            print(f"\n  {strat}:")
            print(
                f"    Valid configs: {len(sr)}/{len([r for r in results if r['strategy'] == strat])}"
            )
            print(f"    Baseline (0,0): pitch={bl_pitch:.1f}, duration={bl_dur:.1f}")
            print(
                f"    Pitch range:    {min(pitches):.1f} → {max(pitches):.1f} (Δ={pitch_range:.1f})"
            )
            print(
                f"    Duration range: {min(durations):.1f} → {max(durations):.1f} (Δ={dur_range:.1f})"
            )
            print(
                f"    Mean degradation: {np.mean(degs):.2f}"
                if degs
                else "    Degradation: N/A"
            )
    else:
        print("\n  WARNING: All results have null values — check extraction functions!")


if __name__ == "__main__":
    main()
